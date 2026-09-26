"""nnsvs の GV ポストフィルタを、フレームごとのパワーを保つものにする。

GV はフレーズ内の mgc の分散を学習データの分散に合わせて広げるが、
OpenUtau は休符ごとにフレーズを分けて送るので、短いフレーズでは分散が小さく倍率が大きくなる
(「ま」だけのフレーズで中央値 4 倍、最大 10 倍。曲全体では中央値 2.2 倍)。
すると m などの鼻音で低域が持ち上がりすぎ、40 dB 以上のノイズになる。
Merlin のポストフィルタと同じく c0 でパワーを揃えると、スペクトルの強調は残したまま防げる。
"""
from contextlib import contextmanager

import nnsvs.gen
import numpy as np
import pysptk
import pyworld


def world_spectrum_decoder(config):
    """nnsvs の config から、mgc (フレーム, 次数) -> パワースペクトル (フレーム, fftlen // 2 + 1) の関数を作る。"""
    sample_rate = config.sample_rate
    fftlen = pyworld.get_cheaptrick_fft_size(sample_rate)
    if config.get('use_world_codec', False):
        def decode(mgc):
            return pyworld.decode_spectral_envelope(
                np.ascontiguousarray(mgc, dtype=np.float64), sample_rate, fftlen)
    else:
        alpha = pysptk.util.mcepalpha(sample_rate)

        def decode(mgc):
            return pysptk.mc2sp(np.ascontiguousarray(mgc, dtype=np.float64), fftlen=fftlen, alpha=alpha)
    return decode


@contextmanager
def energy_preserving_gv(decode):
    """nnsvs の GV ポストフィルタ (variance_scaling) を、フレームごとのパワーを保つものに差し替える。

    decode: world_spectrum_decoder の戻り値。
    c0 を Δ 増やすと対数パワーが k*Δ 増える (mc2sp なら k=2) ので、k は decode から求める。
    """
    original = nnsvs.gen.variance_scaling

    def variance_scaling(gv, feats, offset=2, note_frame_indices=None):
        out = original(gv, feats, offset=offset, note_frame_indices=note_frame_indices)
        idx = slice(None) if note_frame_indices is None else note_frame_indices
        unit = np.zeros((2, feats.shape[1]))
        unit[1, 0] = 1
        k = np.mean(np.diff(np.log(decode(unit)), axis=0))
        tiny = np.finfo(np.float64).tiny
        before = np.maximum(decode(feats[idx]).sum(1), tiny)
        after = np.maximum(decode(out[idx]).sum(1), tiny)
        out[idx, 0] += np.log(before / after) / k
        return out

    nnsvs.gen.variance_scaling = variance_scaling
    try:
        yield
    finally:
        nnsvs.gen.variance_scaling = original
