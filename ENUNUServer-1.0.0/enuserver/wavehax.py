"""Wavehax ボコーダーに対応する (taroushirani/nnsvs の wavehax-support ブランチ相当)。

nnsvs の load_vocoder を包み、vocoder_model.yaml の generator が wavehax.* なら Wavehax として読む。
wavehax パッケージ (taroushirani/wavehax@nnsvs) は Wavehax の音源を読むときだけ使う。

合成には nnsvs.gen.predict_waveform の uSFGAN 分岐をそのまま使う。特徴量の前処理 (bap の補正と正規化) が
同じで、違うのは F0 の渡し方だけなので、学習時の設定 (data.use_continuous_f0) から F0 の種類を決めて渡す。
wavehax-support ブランチは常に連続 F0 を渡すが、nnsvs のレシピの既定は use_continuous_f0: false
(無声区間の F0 を 0 にして学習) なので、それに合わせる。
"""
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

_applied = False


class WavehaxWrapper(nn.Module):
    """nnsvs の USFGANWrapper と同じ inference(f0, aux_feats) で Wavehax の generator を呼ぶ。"""

    def __init__(self, config, generator):
        super().__init__()
        self.generator = generator
        self.config = config
        # nnsvs.gen.predict_waveform の uSFGAN 分岐が読むのは data.sine_f0_type だけ。
        # contf0: 連続 F0 をそのまま渡す / f0: 無声区間 (vuv < vuv_threshold) を 0 にして渡す
        continuous = bool(OmegaConf.select(config, 'data.use_continuous_f0', default=False))
        self.f0_config = OmegaConf.create({'data': {'sine_f0_type': 'contf0' if continuous else 'f0'}})

    def inference(self, f0, aux_feats):
        """
        Args:
            f0 (numpy.ndarray): F0 (T, 1)
            aux_feats (Tensor): 正規化済みの補助特徴量 (T, C)
        """
        device = aux_feats.device
        cond = aux_feats.unsqueeze(0).transpose(2, 1).float()
        f0 = torch.from_numpy(np.asarray(f0, dtype=np.float32)).view(1, 1, -1).to(device)
        # iSTFT の n_fft (48kHz で 960) が2のべき乗でなく GPU の半精度 FFT が使えないので、
        # svs_synthe の autocast を切って fp32 で動かす (Wavehax は小さいので fp32 でも速い)
        with torch.autocast(device.type, enabled=False):
            return self.generator.inference(cond, f0)


def _overlap_add(frames, hop):
    """(B, n_fft, T) のフレームを hop ずつずらして足す。単位行列カーネルの conv_transpose1d と同じ結果。

    n_fft を hop ごとの k 個に分けると、フレーム t の j 番目は出力の t + j 番目の区間に入る。
    """
    B, n_fft, T = frames.shape
    k = n_fft // hop
    chunks = frames.reshape(B, k, hop, T)
    out = frames.new_zeros(B, hop, T + k - 1)
    for j in range(k):
        out[:, :, j:j + T] += chunks[:, j]
    return out.transpose(1, 2).reshape(B, 1, -1)


def _stft_inverse(self, real, imag, norm=None):
    """wavehax.modules.STFT.inverse と同じ iSTFT。重ね合わせを加算で行う。

    元の実装は重ね合わせに単位行列カーネルの conv_transpose1d を使うが、synthe は合成中だけ
    torch.use_deterministic_algorithms(True) にするので、GPU で非常に遅い実装が選ばれる
    (2.5 秒の音声で約 2.8 秒)。加算なら決定的なまま速い。
    """
    if self.n_fft % self.hop_length:
        return _original_stft_inverse(self, real, imag, norm)
    assert real.shape == imag.shape and real.ndim == 3
    assert real.size(1) == self.n_bins

    frames = real.shape[2]
    samples = frames * self.hop_length

    x = torch.fft.irfft(torch.complex(real, imag), dim=1, norm=norm)
    x = _overlap_add(x * self.window, self.hop_length)
    window_envelope = _overlap_add(self.window_envelope.expand(1, -1, frames), self.hop_length)

    # パディングを取り除いて、窓の重なりで割る
    pad = (self.n_fft - self.hop_length) // 2
    x = x[..., pad:samples + pad]
    window_envelope = window_envelope[..., pad:samples + pad]
    assert (window_envelope > 1e-11).all()
    return x / window_envelope


_original_stft_inverse = None


def _patch_stft():
    global _original_stft_inverse
    from wavehax.modules import STFT

    if _original_stft_inverse is None:
        _original_stft_inverse = STFT.inverse
        STFT.inverse = _stft_inverse


def _load_config(model_dir):
    """nnsvs.util.load_vocoder と同じ順でボコーダーの設定を探す。無ければ None。"""
    for name in ('vocoder_model.yaml', 'config.yml', 'config.yaml'):
        if (model_dir / name).exists():
            return OmegaConf.load(model_dir / name)
    return None


def is_wavehax_config(config):
    return str(OmegaConf.select(config, 'generator._target_', default='')).startswith('wavehax.')


def _in_scaler(model_dir, vocoder_config, acoustic_config):
    """音響特徴量全体の in_vocoder_scaler から、ボコーダーに渡すストリームの分だけ取り出す。"""
    from nnsvs.multistream import get_static_stream_sizes
    from nnsvs.util import StandardScaler

    stream_sizes = get_static_stream_sizes(
        acoustic_config.stream_sizes,
        acoustic_config.has_dynamic_features,
        acoustic_config.num_windows,
    )
    mean, var, scale = (np.load(model_dir / f'in_vocoder_scaler_{k}.npy') for k in ('mean', 'var', 'scale'))
    # Wavehax のデータ設定は uSFGAN の aux_feats ではなく feat_names を使う
    if 'mel' in list(vocoder_config.data.feat_names):
        # streams: (mel, lf0, vuv)
        dims = np.arange(stream_sizes[0])
    else:
        # streams: (mgc, lf0, vuv, bap)
        dims = np.r_[0:stream_sizes[0], sum(stream_sizes[:3]):sum(stream_sizes[:4])]
    return StandardScaler(mean[dims], var[dims], scale[dims])


def load_wavehax(path, device, acoustic_config, vocoder_config):
    try:
        import wavehax  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            'この音源は Wavehax ボコーダーを使います。wavehax パッケージを入れてください: '
            'python -m pip install --no-deps git+https://github.com/taroushirani/wavehax@nnsvs einops') from e
    from hydra.utils import instantiate

    _patch_stft()
    path = Path(path)
    # パック済みモデルは state_dict だけなので weights_only=True で読める
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    generator = instantiate(vocoder_config.generator)
    generator.load_state_dict(checkpoint['model']['generator'])
    generator.to(device).eval()
    vocoder = WavehaxWrapper(vocoder_config, generator)
    return vocoder, _in_scaler(path.parent, vocoder_config, acoustic_config), vocoder_config


def predict_waveform(engine, multistream_features, vuv_threshold=0.5):
    """Wavehax で波形を生成する。engine は vocoder が WavehaxWrapper の SPSVS。"""
    from nnsvs.gen import predict_waveform as nnsvs_predict_waveform

    return nnsvs_predict_waveform(
        device=engine.device,
        multistream_features=multistream_features,
        vocoder=engine.vocoder,
        vocoder_config=engine.vocoder.f0_config,
        vocoder_in_scaler=engine.vocoder_in_scaler,
        sample_rate=engine.sample_rate,
        frame_period=engine.config.frame_period,
        use_world_codec=engine.config.get('use_world_codec', False),
        feature_type=engine.feature_type,
        vocoder_type='usfgan',
        vuv_threshold=vuv_threshold,
    )


def apply():
    """nnsvs の load_vocoder を Wavehax 対応版に差し替える (何度呼んでもよい)。"""
    global _applied
    if _applied:
        return
    _applied = True

    import nnsvs.svs
    import nnsvs.util

    original = nnsvs.util.load_vocoder

    def load_vocoder(path, device, acoustic_config):
        config = _load_config(Path(path).parent)
        if config is not None and is_wavehax_config(config):
            return load_wavehax(path, device, acoustic_config, config)
        return original(path, device, acoustic_config)

    # nnsvs.svs は `from nnsvs.util import load_vocoder` で取り込んでいるので、両方差し替える
    nnsvs.util.load_vocoder = load_vocoder
    nnsvs.svs.load_vocoder = load_vocoder
