#!/usr/bin/env python3
"""NNSVS / nnmnkwii / pysptk の遅い部分を、結果を変えずに速くするオーバーライド。

pip で入れたライブラリは書き換えず、import 時に関数を差し替える (apply() を1回呼ぶ)。
どれも元の実装と完全に同じ結果を返す (tests/test_unit.py で確認)。

- nnmnkwii の質問照合 (pattern_matching_binary / pattern_matching_continous_position):
  音素ラベルごとに数千個の正規表現を Python のループで検索していた。
  結果を (質問セット, ラベル) ごとに覚えておく。1音符を直すと変わるのは前後の数音素だけなので、
  2回目以降のリクエストではほとんど検索しない。
- nnmnkwii の linguistic_features: 1回の acoustic で同じラベルに対して最大3回呼ばれる
  (predict_acoustic / postprocess_acoustic / gen_spsvs_static_features)。同じ入力なら結果を使い回す。
- pysptk.util.mcepalpha: サンプリングレートだけで決まる定数を毎回数値探索していた。
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


class _DecoderStepGraph:
    """ResF0NonAttentiveDecoder の自己回帰1ステップを CUDA グラフにしたもの。

    1ステップの形はフレーズの長さに関係なく同じなので、モデルごとに1回だけ記録して全ステップで再生する。
    LSTM の状態と前のステップの出力はグラフ内で静的バッファを更新して引き継ぐ。
    """

    def __init__(self, decoder, enc_dim, device, dtype):
        import torch
        self.decoder = decoder
        rf = decoder.reduction_factor
        hidden = decoder.lstm[0].hidden_size
        self.enc = torch.zeros(1, enc_dim, device=device, dtype=dtype)
        self.lf0 = torch.zeros(1, 1, rf, device=device, dtype=dtype)
        self.prev = torch.zeros(1, decoder.out_dim, device=device, dtype=dtype)
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
        import torch.nn.functional as F
        d = self.decoder
        if d.prenet is not None:
            prenet_out = d.prenet(self.prev)
        else:
            prenet_out = F.dropout(self.prev, d.prenet_dropout, training=True)
        xs = torch.cat([self.enc, prenet_out], dim=1)
        h, c = d.lstm[0](xs, (self.h[0], self.c[0]))
        new_h, new_c = [h], [c]
        for i in range(1, len(d.lstm)):
            h, c = d.lstm[i](new_h[i - 1], (self.h[i], self.c[i]))
            new_h.append(h)
            new_c.append(c)
        hcs = torch.cat([new_h[-1], self.enc], dim=1)
        out = d.feat_out(hcs).view(1, d.out_dim, -1)
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
        self.prev.copy_(out[:, :, -1])
        return out, lf0_residual

    def run(self, encoder_outs, lf0_score_denorm):
        import torch
        rf = self.decoder.reduction_factor
        steps = encoder_outs.shape[1]
        for x in (*self.h, *self.c, self.prev):
            x.zero_()
        outs = torch.empty(1, self.decoder.out_dim, steps * rf, device=encoder_outs.device, dtype=encoder_outs.dtype)
        residuals = torch.empty(1, 1, steps * rf, device=encoder_outs.device, dtype=encoder_outs.dtype)
        for t in range(steps):
            self.enc.copy_(encoder_outs[:, t])
            self.lf0.copy_(lf0_score_denorm[:, :, t * rf:(t + 1) * rf])
            self.graph.replay()
            outs[:, :, t * rf:(t + 1) * rf] = self.out
            residuals[:, :, t * rf:(t + 1) * rf] = self.residual
        return outs.transpose(1, 2), residuals.transpose(1, 2)


def apply_cuda_graphs():
    """lf0 の自己回帰デコーダーを CUDA Graphs で実行する (推論時・CUDA のみ)。

    元の実装は1ステップごとに Python で数十個の演算を呼び、フレーム数/reduction_factor 回ループしていた
    (3秒のフレーズで約150ステップ)。prenet の dropout はグラフ内でも乱数を使うので、
    元の実装とは乱数の並びが変わる (同じシードなら毎回同じ結果になる)。
    """
    import torch
    from nnsvs.acoustic_models import tacotron_f0

    cls = tacotron_f0.ResF0NonAttentiveDecoder
    if getattr(cls, '_enunu_graphed', False):
        return
    original_forward = cls.forward

    def forward(self, encoder_outs, in_lens, decoder_targets=None):
        if (decoder_targets is not None or not encoder_outs.is_cuda or torch.is_grad_enabled()
                or encoder_outs.shape[0] != 1 or getattr(self, '_enunu_graph_failed', False)):
            return original_forward(self, encoder_outs, in_lens, decoder_targets)
        # ループ前の処理は元の forward と同じ
        lf0_score = encoder_outs[:, :, self.in_lf0_idx].unsqueeze(-1)
        lf0_score_denorm = (lf0_score * (self.in_lf0_max - self.in_lf0_min) + self.in_lf0_min).transpose(1, 2)
        if self.reduction_factor > 1:
            if self.conv_downsample is not None:
                encoder_outs = self.conv_downsample(encoder_outs.transpose(1, 2)).transpose(1, 2)
            else:
                encoder_outs = encoder_outs[:, self.reduction_factor - 1::self.reduction_factor]
        key = (encoder_outs.shape[2], encoder_outs.device, encoder_outs.dtype)
        runner = getattr(self, '_enunu_step_graph', None)
        if runner is None or runner[0] != key:
            try:
                runner = (key, _DecoderStepGraph(self, key[0], key[1], key[2]))
            except Exception as e:  # noqa: BLE001
                import logging
                logging.getLogger(__name__).warning('lf0 decoder CUDA Graphs capture failed: %s', e)
                self._enunu_graph_failed = True
                return original_forward(self, encoder_outs, in_lens, decoder_targets)
            self._enunu_step_graph = runner
        # lf0_score_denorm は reduction_factor 倍の長さ (パディング済み)
        return runner[1].run(encoder_outs, lf0_score_denorm)

    cls.forward = forward
    cls._enunu_graphed = True


def clear_caches():
    _binary_cache.clear()
    _numeric_cache.clear()
    _feature_cache.clear()
