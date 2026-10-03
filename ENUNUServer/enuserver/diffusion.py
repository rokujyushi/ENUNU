"""拡散モデル (DiffSinger 系の mgc / mel / bap) のサンプラ設定と高速化。

- 設定: 優先順は config コマンド > 環境変数 > 既定値 (diffusion_settings)。
- サンプラ: nnsvs の PLMS に加えて DDIM (η=0) と η=1 strided ancestral を使えるようにする。
- CUDA Graphs: デノイザーを記録して再生する (GraphedDenoiser)。
"""
import copy
import logging
import os
import types
from collections import OrderedDict

import torch

logger = logging.getLogger(__name__)

DIFFUSION_STREAMS = ('mgc', 'mel', 'bap')
DIFFUSION_METHODS = ('ddpm', 'ddim', 'plms', 'eta1')
# 既定値。mgc / mel は DDIM 25 ステップで 100 ステップとほぼ同等の品質、
# bap は PLMS 10 ステップだとなめらかになりすぎるため 20 ステップ (いずれも 2026-09-25 に数値比較・試聴で決定)。
DEFAULT_DIFFUSION = {
    'mgc': {'method': 'ddim', 'steps': 25},
    'mel': {'method': 'ddim', 'steps': 25},
    'bap': {'method': 'plms', 'steps': 20},
    'other': {'method': 'ddpm', 'steps': 100},
}
# config コマンドで変更された設定 (None なら環境変数・既定値)
_diffusion_override = None


def cuda_graphs_enabled() -> bool:
    """ENUNU_CUDA_GRAPHS=0 で CUDA Graphs を使わない。"""
    return os.environ.get('ENUNU_CUDA_GRAPHS', '1') != '0'


def parse_diffusion_spec(spec: str) -> dict:
    """ 'ddim:25' / 'ddpm' / '25' (手法は既定のまま) を {'method', 'steps'} にする。"""
    method, _, steps = spec.strip().lower().partition(':')
    if method.isdigit() and not steps:
        return {'steps': int(method)}
    if method not in DIFFUSION_METHODS:
        raise ValueError(f'unknown diffusion method: {method}')
    result = {'method': method}
    if steps:
        result['steps'] = int(steps)
    return result


def _legacy_env_diffusion() -> dict | None:
    """旧環境変数 (ENUNU_DIFFUSION_SPEEDUP / TARGETS / METHOD) が指定されていれば、その意味どおりの設定を返す。"""
    keys = ("ENUNU_DIFFUSION_SPEEDUP", "ENUNU_DIFFUSION_TARGETS", "ENUNU_DIFFUSION_METHOD")
    if not any(k in os.environ for k in keys):
        return None
    try:
        speedup = max(int(os.environ.get("ENUNU_DIFFUSION_SPEEDUP", "10")), 1)
    except ValueError:
        speedup = 10
    targets = os.environ.get("ENUNU_DIFFUSION_TARGETS", "bap").lower()
    method = os.environ.get("ENUNU_DIFFUSION_METHOD", "plms").lower()
    if method not in DIFFUSION_METHODS:
        method = 'plms'
    settings = {k: {'method': 'ddpm', 'steps': 100} for k in (*DIFFUSION_STREAMS, 'other')}
    if speedup > 1:
        for k in DIFFUSION_STREAMS:
            if targets == 'all' or targets in k:
                settings[k] = {'method': method, 'steps': max(1, 100 // speedup)}
    return settings


def diffusion_settings() -> dict:
    """拡散モデルのサンプラ設定を返す。優先順: config コマンド > 環境変数 > 既定値。

    環境変数:
      ENUNU_DIFFUSION_MGC / _MEL / _BAP : 'ddim:25' のように手法:ステップ数
      (旧) ENUNU_DIFFUSION_SPEEDUP / TARGETS / METHOD : 指定されている場合は従来の意味で解釈
    """
    if _diffusion_override is not None:
        return copy.deepcopy(_diffusion_override)
    return _base_diffusion_settings()


def _base_diffusion_settings() -> dict:
    """config コマンドを除いた設定 (環境変数 > 既定値)。"""
    settings = _legacy_env_diffusion() or copy.deepcopy(DEFAULT_DIFFUSION)
    for k in DIFFUSION_STREAMS:
        spec = os.environ.get(f"ENUNU_DIFFUSION_{k.upper()}")
        if spec:
            try:
                settings[k].update(parse_diffusion_spec(spec))
            except ValueError as e:
                logger.warning("ENUNU_DIFFUSION_%s を無視します: %s", k.upper(), e)
    return settings


def set_diffusion_settings(request: dict) -> dict:
    """config コマンドの diffusion 設定を検証して反映し、反映後の設定を返す。

    request 例: {'steps': 25}  (mgc と mel のステップ数だけ変える)
               {'mgc': {'method': 'ddim', 'steps': 25}, 'bap': 'plms:20'}
               {'reset': True}  (環境変数・既定値に戻す)
               {'reset': True, 'bap': {'steps': 30}}  (既定値から始めて bap だけ変える)

    reset が無い指定は、今の設定に重ねる。OpenUtau は毎回 reset 付きで全体を送るので、
    前に送った指定が残らない。
    """
    global _diffusion_override
    reset = bool(request.get('reset'))
    if reset and not any(k in request for k in ('steps', *DIFFUSION_STREAMS)):
        _diffusion_override = None
        return diffusion_settings()
    settings = _base_diffusion_settings() if reset else diffusion_settings()
    if 'steps' in request:
        for k in ('mgc', 'mel'):
            settings[k]['steps'] = request['steps']
    for k in DIFFUSION_STREAMS:
        if k in request:
            value = request[k]
            settings[k].update(parse_diffusion_spec(value) if isinstance(value, str) else value)
    for k, v in settings.items():
        if v.get('method') not in DIFFUSION_METHODS:
            raise ValueError(f'unknown diffusion method for {k}: {v.get("method")}')
        if not isinstance(v.get('steps'), int) or isinstance(v['steps'], bool) or v['steps'] < 1:
            raise ValueError(f'diffusion steps for {k} must be a positive integer')
    _diffusion_override = settings
    return copy.deepcopy(settings)


def apply_diffusion_settings(acoustic_model, device, settings: dict) -> None:
    """acoustic_model 配下の GaussianDiffusion ごとにサンプラとステップ数を設定する。

    settings: diffusion_settings() の戻り値 ({'mgc': {'method': 'ddim', 'steps': 25}, ...})。
    モジュール名に mgc / mel / bap を含むものにそれぞれの設定を、それ以外には 'other' を使う。
    何度呼んでもよい (config コマンドで実行中に変更する)。
    """
    try:
        from nnsvs.diffsinger.diffusion import GaussianDiffusion, extract
    except Exception as e:
        logger.warning("GaussianDiffusion を import できず拡散設定を適用できません: %s", e)
        return
    applied = []
    for name, m in acoustic_model.named_modules():
        if not isinstance(m, GaussianDiffusion):
            continue
        short = name.split(".")[-1] if name else "(root)"
        stream = next((k for k in DIFFUSION_STREAMS if k in short), 'other')
        method, steps = settings[stream]['method'], settings[stream]['steps']
        # デノイザーを CUDA Graphs で実行する (1回だけ包む)
        if (torch.device(device).type == 'cuda' and cuda_graphs_enabled()
                and not isinstance(m.denoise_fn, GraphedDenoiser)):
            m.denoise_fn = GraphedDenoiser(m.denoise_fn)
        # 以前に差し替えたサンプラを外してから設定し直す
        m.__dict__.pop('p_sample_plms', None)
        interval = max(1, round(m.K_step / steps)) if method != 'ddpm' else 1
        if interval <= 1:
            m.pndm_speedup = None
            method = 'ddpm'
        else:
            m.pndm_speedup = interval
            if method == 'ddim':
                _bind_ddim_sampler(m, extract)
            elif method == 'eta1':
                _bind_eta1_sampler(m, extract)
        applied.append(f'{short}={method}:{-(-m.K_step // interval)}')
    if applied:
        logger.info("拡散サンプラ: %s", ", ".join(applied))


def _bind_ddim_sampler(gauss_diffusion, extract_fn) -> None:
    """GaussianDiffusion の p_sample_plms を DDIM (η=0) に置き換える。"""

    @torch.no_grad()
    def ddim_step(self, x, t, interval, cond):
        a_t = extract_fn(self.alphas_cumprod, t, x.shape)
        a_prev = extract_fn(
            self.alphas_cumprod,
            torch.max(t - interval, torch.zeros_like(t)),
            x.shape,
        )
        sqrt_a_t = a_t.sqrt()
        sqrt_one_minus_a_t = (1 - a_t).sqrt()
        sqrt_a_prev = a_prev.sqrt()
        sqrt_one_minus_a_prev = (1 - a_prev).sqrt()
        noise_pred = self.denoise_fn(x, t, cond=cond)
        x0_hat = (x - sqrt_one_minus_a_t * noise_pred) / sqrt_a_t
        x0_hat = x0_hat.clamp(-1.0, 1.0)
        if bool((t == 0).all()):
            return x0_hat
        eps_eff = (x - sqrt_a_t * x0_hat) / sqrt_one_minus_a_t.clamp(min=1e-8)
        return sqrt_a_prev * x0_hat + sqrt_one_minus_a_prev * eps_eff

    gauss_diffusion.p_sample_plms = types.MethodType(ddim_step, gauss_diffusion)


def _bind_eta1_sampler(gauss_diffusion, extract_fn) -> None:
    """GaussianDiffusion の p_sample_plms を η=1 strided ancestral に置き換える。"""

    @torch.no_grad()
    def eta1_step(self, x, t, interval, cond):
        t_prev = torch.max(t - interval, torch.zeros_like(t))
        a_t = extract_fn(self.alphas_cumprod, t, x.shape)
        a_prev = extract_fn(self.alphas_cumprod, t_prev, x.shape)
        sqrt_a_t = a_t.sqrt()
        sqrt_one_minus_a_t = (1 - a_t).sqrt().clamp(min=1e-8)
        noise_pred = self.denoise_fn(x, t, cond=cond)
        x0_hat = ((x - sqrt_one_minus_a_t * noise_pred) / sqrt_a_t).clamp(-1.0, 1.0)
        if bool((t == 0).all()):
            return x0_hat
        eps_eff = (x - sqrt_a_t * x0_hat) / sqrt_one_minus_a_t
        sigma2 = ((1 - a_prev) / (1 - a_t)).clamp(min=0.0) \
            * (1 - a_t / a_prev).clamp(min=0.0)
        dir_coef = (1 - a_prev - sigma2).clamp(min=0.0).sqrt()
        return (
            a_prev.sqrt() * x0_hat
            + dir_coef * eps_eff
            + sigma2.sqrt() * torch.randn_like(x)
        )

    gauss_diffusion.p_sample_plms = types.MethodType(eta1_step, gauss_diffusion)


class GraphedDenoiser(torch.nn.Module):
    """拡散モデルのデノイザー (DiffNet) を CUDA Graphs で実行するラッパー。

    デノイザーは 1 回の推論で 20〜100 回呼ばれ、1 回ごとの計算は小さいので、GPU の計算より
    Python とカーネル起動のオーバーヘッドが律速になっている (フレーズを長くしても時間がほぼ変わらない)。
    入力の形 (フレーズの長さ) ごとに 1 回グラフを記録し、以降は入力をコピーして再生するだけにする。
    記録に失敗した形は通常の実行に戻す。ENUNU_CUDA_GRAPHS=0 で無効 (ラップしない)。
    """

    def __init__(self, inner, max_graphs=4):
        super().__init__()
        self.inner = inner
        self.in_dim = inner.in_dim
        self.max_graphs = max_graphs
        self.graphs = OrderedDict()
        self.failed = set()
        # 同じモデルのグラフ間でメモリプールを共有する (順番に 1 つずつしか再生しないので安全)
        self.pool = None

    def forward(self, x, t, cond):
        if not x.is_cuda or torch.is_grad_enabled():
            return self.inner(x, t, cond=cond)
        key = (tuple(x.shape), tuple(t.shape), tuple(cond.shape), x.dtype, t.dtype, cond.dtype, x.device)
        if key in self.failed:
            return self.inner(x, t, cond=cond)
        entry = self.graphs.get(key)
        if entry is None:
            try:
                entry = self._capture(x, t, cond)
            except Exception as e:  # noqa: BLE001
                logger.warning('CUDA Graphs capture failed for %s, running eagerly: %s', key[0], e)
                self.failed.add(key)
                return self.inner(x, t, cond=cond)
            self.graphs[key] = entry
            while len(self.graphs) > self.max_graphs:
                self.graphs.popitem(last=False)
        else:
            self.graphs.move_to_end(key)
        graph, static_x, static_t, static_cond, static_out = entry
        static_x.copy_(x)
        static_t.copy_(t)
        static_cond.copy_(cond)
        graph.replay()
        # 次の再生で上書きされるので複製して返す (PLMS は過去の出力を保持する)
        return static_out.clone()

    def _capture(self, x, t, cond):
        static_x, static_t, static_cond = x.clone(), t.clone(), cond.clone()
        # 記録の前に別ストリームで数回実行しておく (cuDNN のアルゴリズム選択などを済ませる)
        stream = torch.cuda.Stream(device=x.device)
        stream.wait_stream(torch.cuda.current_stream(x.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.inner(static_x, static_t, cond=static_cond)
        torch.cuda.current_stream(x.device).wait_stream(stream)
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            static_out = self.inner(static_x, static_t, cond=static_cond)
        return graph, static_x, static_t, static_cond, static_out
