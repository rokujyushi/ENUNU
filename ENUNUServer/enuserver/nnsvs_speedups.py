#!/usr/bin/env python3
"""NNSVS / nnmnkwii / pysptk の遅い部分を、結果を変えずに速くするオーバーライド。

pip で入れたライブラリは書き換えず、import 時に関数を差し替える (apply() を1回呼ぶ)。
LSTM 以外は元の実装と完全に同じ結果を返す (tests/test_unit.py で確認)。

- nnmnkwii の質問照合 (pattern_matching_binary / pattern_matching_continous_position):
  音素ラベルごとに数千個の正規表現を Python のループで検索していた。
  結果を (質問セット, ラベル) ごとに覚えておく。1音符を直すと変わるのは前後の数音素だけなので、
  2回目以降のリクエストではほとんど検索しない。
- nnmnkwii の linguistic_features: 1回の acoustic で同じラベルに対して最大3回呼ばれる
  (predict_acoustic / postprocess_acoustic / gen_spsvs_static_features)。同じ入力なら結果を使い回す。
- pysptk.util.mcepalpha: サンプリングレートだけで決まる定数を毎回数値探索していた。
- torch.nn.LSTM: 系列が1本だけの PackedSequence は通常のテンソルとして実行する (cuDNN のパック経路が非常に遅い)。
  カーネルが変わるので出力は 1e-6 程度ずれる。
"""
import functools
from collections import OrderedDict

import numpy as np

_applied = False

# (id(質問セット), ラベル) -> 照合結果。質問セットは破棄されると id が再利用されるので参照を持っておく
_binary_cache = {}
_numeric_cache = {}
_dict_refs = {}
_PATTERN_CACHE_LIMIT = 200_000

_feature_cache = OrderedDict()
_FEATURE_CACHE_LIMIT = 8


def _remember_dict(d):
    _dict_refs.setdefault(id(d), d)


def _cached(cache, original, question_dict, label):
    key = (id(question_dict), label)
    result = cache.get(key)
    if result is None:
        if len(cache) > _PATTERN_CACHE_LIMIT:
            cache.clear()
        _remember_dict(question_dict)
        result = original(question_dict, label)
        cache[key] = result
    return result.copy()


def _labels_key(hts_labels):
    return (tuple(hts_labels.start_times), tuple(hts_labels.end_times), tuple(hts_labels.contexts),
            getattr(hts_labels, 'frame_shift', None))


def apply():
    """オーバーライドを適用する (何度呼んでもよい)。"""
    global _applied
    if _applied:
        return
    _applied = True

    from nnmnkwii.frontend import merlin
    import pysptk.util

    original_binary = merlin.pattern_matching_binary
    original_numeric = merlin.pattern_matching_continous_position
    original_features = merlin.linguistic_features

    @functools.wraps(original_binary)
    def pattern_matching_binary(binary_dict, label):
        return _cached(_binary_cache, original_binary, binary_dict, label)

    @functools.wraps(original_numeric)
    def pattern_matching_continous_position(numeric_dict, label):
        return _cached(_numeric_cache, original_numeric, numeric_dict, label)

    @functools.wraps(original_features)
    def linguistic_features(hts_labels, *args, **kwargs):
        try:
            key = (_labels_key(hts_labels), tuple(id(a) for a in args),
                   tuple(sorted((k, v if isinstance(v, (int, float, str, bool, type(None))) else id(v))
                                for k, v in kwargs.items())))
        except (AttributeError, TypeError):
            return original_features(hts_labels, *args, **kwargs)
        cached = _feature_cache.get(key)
        if cached is None:
            for a in args:
                _remember_dict(a)
            cached = original_features(hts_labels, *args, **kwargs)
            _feature_cache[key] = cached
            while len(_feature_cache) > _FEATURE_CACHE_LIMIT:
                _feature_cache.popitem(last=False)
        else:
            _feature_cache.move_to_end(key)
        # 呼び出し側 (enunu._acoustic_input など) が書き換えるので複製して返す
        return cached.copy()

    merlin.pattern_matching_binary = pattern_matching_binary
    merlin.pattern_matching_continous_position = pattern_matching_continous_position
    merlin.linguistic_features = linguistic_features

    pysptk.util.mcepalpha = functools.lru_cache(maxsize=16)(pysptk.util.mcepalpha)
    # nnsvs.gen は `pysptk.util.mcepalpha(...)` の形で呼ぶので、モジュール属性の差し替えで効く

    _patch_single_sequence_lstm()
    _patch_mc2sp()


@functools.lru_cache(maxsize=16)
def _freqt_matrix(in_order, out_order, alpha):
    """pysptk.freqt (周波数ワーピング) は線形変換なので、単位ベクトルを通して変換行列を作る。"""
    from pysptk.sptk import freqt
    basis = np.eye(in_order + 1, dtype=np.float64)
    matrix = np.stack([freqt(basis[k], out_order, alpha) for k in range(in_order + 1)])
    matrix.setflags(write=False)
    return matrix   # (in_order + 1, out_order + 1)


def _patch_mc2sp():
    """pysptk.mc2sp を全フレームまとめて計算する版に差し替える。

    元の実装は 1 フレームずつ freqt を呼び、さらに Python の for ループで対称な配列を組み立てていた
    (use_world_codec=False の旧来のモデルで、6.5 秒のフレーズの WORLD パラメータ生成に 1.2 秒以上)。
    freqt を行列積に、対称化と FFT を配列演算にする。結果は行列積の丸め (1e-12 程度) を除いて同じ。
    """
    import pysptk
    import pysptk.conversion

    original = pysptk.conversion.mc2sp
    if getattr(original, '_enunu_vectorized', False):
        return

    @functools.wraps(original)
    def mc2sp(mc, alpha, fftlen):
        mc = np.asarray(mc, dtype=np.float64)
        if mc.ndim > 2:
            return original(mc, alpha, fftlen)
        frames = np.atleast_2d(mc)
        half = int(fftlen // 2)
        c = frames @ _freqt_matrix(frames.shape[-1] - 1, half, float(-alpha))
        c[:, 0] *= 2.0
        symc = np.zeros((c.shape[0], int(fftlen)))
        symc[:, :half + 1] = c
        symc[:, half + 1:] = c[:, half - 1:0:-1]
        sp = np.exp(np.fft.rfft(symc, axis=-1).real)
        return sp[0] if mc.ndim == 1 else sp

    mc2sp._enunu_vectorized = True
    pysptk.conversion.mc2sp = mc2sp
    pysptk.mc2sp = mc2sp   # nnsvs.gen は pysptk.mc2sp(...) の形で呼ぶ


def _patch_single_sequence_lstm():
    """系列が1本だけの PackedSequence を LSTM に渡されたら、通常のテンソルとして実行する。

    nnsvs の FFConvLSTM や lf0 モデルのエンコーダーは可変長のために pack_padded_sequence を使うが、
    サーバーは常にバッチサイズ 1 なのでパックの意味がない。cuDNN のパック入力の経路は非常に遅く
    (RTX 5060 Ti / cuDNN 9.10 で 600 フレームの 2 層 BiLSTM が 98 ms、通常のテンソルなら 1.3 ms)、
    これが vuv モデルと lf0 モデルの時間の大半を占めていた。出力は元どおり PackedSequence に戻して返す。
    """
    import torch
    from torch.nn.utils.rnn import PackedSequence

    lstm_cls = torch.nn.LSTM
    if getattr(lstm_cls, '_enunu_single_sequence', False):
        return
    original_forward = lstm_cls.forward

    def forward(self, input, hx=None):
        if not (isinstance(input, PackedSequence) and bool((input.batch_sizes == 1).all())):
            return original_forward(self, input, hx)
        data = input.data.unsqueeze(0) if self.batch_first else input.data.unsqueeze(1)
        output, hidden = original_forward(self, data, hx)
        output = output.squeeze(0) if self.batch_first else output.squeeze(1)
        return PackedSequence(output, input.batch_sizes, input.sorted_indices, input.unsorted_indices), hidden

    lstm_cls.forward = forward
    lstm_cls._enunu_single_sequence = True


class _DecoderStepGraph:
    """自己回帰デコーダー (Tacotron 系の NonAttentiveDecoder) の1ステップを CUDA グラフにしたもの。

    1ステップの形はフレーズの長さに関係なく同じなので、モデルごとに1回だけ記録して全ステップで再生する。
    LSTM の状態と前のステップの出力はグラフ内で静的バッファを更新して引き継ぐ。
    residual=True は ResF0NonAttentiveDecoder (lf0: 楽譜の音高 + 残差)、False は NonAttentiveDecoder (mgc/bap など)。
    """

    def __init__(self, decoder, enc_dim, device, dtype, residual=True):
        import torch
        self.decoder = decoder
        self.residual_mode = residual
        rf = decoder.reduction_factor
        hidden = decoder.lstm[0].hidden_size
        self.enc = torch.zeros(1, enc_dim, device=device, dtype=dtype)
        self.lf0 = torch.zeros(1, 1, rf, device=device, dtype=dtype)
        # 前のステップの出力は、元の forward と同じく (1, out_dim, rf) の最後のフレームのビューとして持つ。
        # dropout はテンソルのメモリ配置 (stride) によって乱数の割り当てが変わるので、連続したコピーにすると
        # 同じシードでもマスクが変わる (Yeonu の mgc: rf=2, 60 次元で不一致になった)
        self.prev_full = torch.zeros(1, decoder.out_dim, rf, device=device, dtype=dtype)
        self.prev = self.prev_full[:, :, -1]
        # prenet が無いデコーダーは前の出力に直接 dropout (またはノイズ) をかける。この部分はグラフの外で
        # 元と同じように実行し、結果だけグラフに渡す (1 ステップあたりカーネルが 1 つ増えるだけ)
        self.external_noise = decoder.prenet is None
        self.prenet_in = torch.zeros(1, decoder.out_dim, device=device, dtype=dtype)
        self.h = [torch.zeros(1, hidden, device=device, dtype=dtype) for _ in decoder.lstm]
        self.c = [torch.zeros(1, hidden, device=device, dtype=dtype) for _ in decoder.lstm]
        # 準備の実行と記録でも dropout が乱数を消費するので、乱数の状態を保存して戻す
        # (戻さないと、グラフを記録したリクエストだけ結果が変わり、シード固定の再現性が崩れる)
        rng_state = torch.cuda.get_rng_state(device)
        try:
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._step()
            torch.cuda.current_stream(device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out, self.residual = self._step()
        finally:
            torch.cuda.set_rng_state(rng_state, device)

    def _step(self):
        """元の forward のループ本体 (推論時) と同じ計算。状態は静的バッファに書き戻す。"""
        import torch
        d = self.decoder
        if self.external_noise:
            prenet_out = self.prenet_in   # run() でグラフの外で計算して入れる
        else:
            prenet_out = d.prenet(self.prev)
        xs = torch.cat([self.enc, prenet_out], dim=1)
        h, c = d.lstm[0](xs, (self.h[0], self.c[0]))
        new_h, new_c = [h], [c]
        for i in range(1, len(d.lstm)):
            h, c = d.lstm[i](new_h[i - 1], (self.h[i], self.c[i]))
            new_h.append(h)
            new_c.append(c)
        hcs = torch.cat([new_h[-1], self.enc], dim=1)
        out = d.feat_out(hcs).view(1, d.out_dim, -1)
        lf0_residual = out[:, :1, :]   # residual=False では使わない
        if self.residual_mode:
            if d.scaled_tanh:
                max_lf0_ratio = 600 * np.log(2) / 1200
                lf0_residual = max_lf0_ratio * torch.tanh(out[:, d.out_lf0_idx, :]).unsqueeze(1)
            else:
                lf0_residual = out[:, d.out_lf0_idx, :].unsqueeze(1)
            lf0_pred = (self.lf0 + lf0_residual - d.out_lf0_mean) / d.out_lf0_scale
            out[:, d.out_lf0_idx, :] = lf0_pred.squeeze(1)
        for i in range(len(d.lstm)):
            self.h[i].copy_(new_h[i])
            self.c[i].copy_(new_c[i])
        self.prev_full.copy_(out)
        return out, lf0_residual

    def _noise(self, prev):
        """prenet が無いデコーダーの入力 (元の forward と同じ)。"""
        import torch
        import torch.nn.functional as F
        d = self.decoder
        if getattr(d, 'prenet_noise_std', 0) > 0:
            return prev + torch.randn_like(prev) * d.prenet_noise_std
        return F.dropout(prev, d.prenet_dropout, training=True)

    def run(self, encoder_outs, lf0_score_denorm=None):
        import torch
        d = self.decoder
        rf = d.reduction_factor
        steps = encoder_outs.shape[1]
        for x in (*self.h, *self.c):
            x.zero_()
        # 最初の入力 (go frame): ResF0 は 0、NonAttentiveDecoder は initial_value。
        # 元の forward と同じく連続した (1, out_dim) のテンソルとして作る (dropout の乱数の割り当てを合わせる)
        go_frame = torch.full((1, d.out_dim), 0.0 if self.residual_mode else float(getattr(d, 'initial_value', 0.0)),
                              device=encoder_outs.device, dtype=encoder_outs.dtype)
        self.prev.copy_(go_frame)
        outs = torch.empty(1, d.out_dim, steps * rf, device=encoder_outs.device, dtype=encoder_outs.dtype)
        residuals = None
        if self.residual_mode:
            residuals = torch.empty(1, 1, steps * rf, device=encoder_outs.device, dtype=encoder_outs.dtype)
        for t in range(steps):
            self.enc.copy_(encoder_outs[:, t])
            if self.residual_mode:
                self.lf0.copy_(lf0_score_denorm[:, :, t * rf:(t + 1) * rf])
            if self.external_noise:
                self.prenet_in.copy_(self._noise(go_frame if t == 0 else self.prev))
            self.graph.replay()
            outs[:, :, t * rf:(t + 1) * rf] = self.out
            if self.residual_mode:
                residuals[:, :, t * rf:(t + 1) * rf] = self.residual
        if self.residual_mode:
            return outs.transpose(1, 2), residuals.transpose(1, 2)
        return outs.transpose(1, 2)


def _graph_runner(decoder, encoder_outs, residual):
    """デコーダーに紐づくグラフを返す (入力の次元などが変わったら記録し直す)。失敗したら None。"""
    key = (encoder_outs.shape[2], encoder_outs.device, encoder_outs.dtype)
    runner = getattr(decoder, '_enunu_step_graph', None)
    if runner is None or runner[0] != key:
        try:
            runner = (key, _DecoderStepGraph(decoder, key[0], key[1], key[2], residual))
        except Exception as e:  # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning('decoder CUDA Graphs capture failed (%s): %s',
                                                type(decoder).__name__, e)
            decoder._enunu_graph_failed = True
            return None
        decoder._enunu_step_graph = runner
    return runner[1]


def _downsample(decoder, encoder_outs):
    """元の forward と同じ reduction_factor による間引き。"""
    if decoder.reduction_factor > 1:
        if decoder.conv_downsample is not None:
            return decoder.conv_downsample(encoder_outs.transpose(1, 2)).transpose(1, 2)
        return encoder_outs[:, decoder.reduction_factor - 1::decoder.reduction_factor]
    return encoder_outs


def apply_cuda_graphs():
    """自己回帰デコーダーを CUDA Graphs で実行する (推論時・CUDA のみ)。

    対象: ResF0NonAttentiveDecoder (lf0) と NonAttentiveDecoder (NPSS 系の mgc / bap など)。
    元の実装は1ステップごとに Python で数十個の演算を呼び、フレーム数/reduction_factor 回ループしていた
    (3秒のフレーズで約150ステップ)。prenet の dropout もグラフ内で同じ乱数の並びになり、結果は一致する。
    """
    import torch
    from nnsvs.acoustic_models import tacotron_f0
    from nnsvs.tacotron import decoder as tacotron_decoder

    def usable(self, encoder_outs, decoder_targets):
        return (decoder_targets is None and encoder_outs.is_cuda and not torch.is_grad_enabled()
                and encoder_outs.shape[0] == 1 and not getattr(self, '_enunu_graph_failed', False))

    res_cls = tacotron_f0.ResF0NonAttentiveDecoder
    if not getattr(res_cls, '_enunu_graphed', False):
        original_res = res_cls.forward

        def res_forward(self, encoder_outs, in_lens, decoder_targets=None):
            if not usable(self, encoder_outs, decoder_targets):
                return original_res(self, encoder_outs, in_lens, decoder_targets)
            # ループ前の処理は元の forward と同じ
            lf0_score = encoder_outs[:, :, self.in_lf0_idx].unsqueeze(-1)
            lf0_score_denorm = (lf0_score * (self.in_lf0_max - self.in_lf0_min) + self.in_lf0_min).transpose(1, 2)
            downsampled = _downsample(self, encoder_outs)
            runner = _graph_runner(self, downsampled, residual=True)
            if runner is None:
                return original_res(self, encoder_outs, in_lens, decoder_targets)
            # lf0_score_denorm は reduction_factor 倍の長さ (パディング済み)
            return runner.run(downsampled, lf0_score_denorm)

        res_cls.forward = res_forward
        res_cls._enunu_graphed = True

    plain_cls = tacotron_decoder.NonAttentiveDecoder
    if not getattr(plain_cls, '_enunu_graphed', False):
        original_plain = plain_cls.forward

        def plain_forward(self, encoder_outs, in_lens, decoder_targets=None):
            if not usable(self, encoder_outs, decoder_targets):
                return original_plain(self, encoder_outs, in_lens, decoder_targets)
            downsampled = _downsample(self, encoder_outs)
            runner = _graph_runner(self, downsampled, residual=False)
            if runner is None:
                return original_plain(self, encoder_outs, in_lens, decoder_targets)
            return runner.run(downsampled)

        plain_cls.forward = plain_forward
        plain_cls._enunu_graphed = True


def clear_caches():
    _binary_cache.clear()
    _numeric_cache.clear()
    _feature_cache.clear()
