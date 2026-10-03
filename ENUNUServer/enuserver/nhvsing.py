"""NHVSing ボコーダーに対応する (rokujyushi/NHVSing@package で nnsvs の melf0 用に学習したもの)。

nnsvs の load_vocoder を包み、vocoder_model.yaml の generator が nhvsing.* なら NHVSing として読む。
nhvsing パッケージは NHVSing の音源を読むときだけ使う。

NHVSing は mel・連続 F0・無声フラグを受け取る DSP ボコーダーで、nnsvs の MelF0 と同じ mel で学習してあるので、
音響モデルが出した mel (正規化を外した log10) をそのまま渡す (in_vocoder_scaler は使わない)。
mel の仕様が音響モデルと違うと音がずれるので、読み込み時に vocoder_model.yaml の data と照らし合わせる。
"""
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

_applied = False


class NHVSingWrapper(nn.Module):
    def __init__(self, config, generator):
        super().__init__()
        self.generator = generator
        self.config = config


def _load_config(model_dir):
    """nnsvs.util.load_vocoder と同じ順でボコーダーの設定を探す。無ければ None。"""
    for name in ('vocoder_model.yaml', 'config.yml', 'config.yaml'):
        if (model_dir / name).exists():
            return OmegaConf.load(model_dir / name)
    return None


def is_nhvsing_config(config):
    return str(OmegaConf.select(config, 'generator._target_', default='')).startswith('nhvsing.')


def check_compatible(config, acoustic_config, vocoder_config):
    """音源の設定 (config.yaml) と音響モデルの出力が、NHVSing の学習時の mel と合うか確かめる。"""
    from nnsvs.multistream import get_static_stream_sizes

    feature_type = config.get('feature_type', 'world')
    if feature_type != 'melf0':
        raise RuntimeError(f'NHVSing ボコーダーは melf0 の音源でしか使えません (この音源の feature_type: {feature_type})')
    stream_sizes = get_static_stream_sizes(
        acoustic_config.stream_sizes,
        acoustic_config.has_dynamic_features,
        acoustic_config.num_windows,
    )
    data = vocoder_config.data
    if stream_sizes[0] != data.mel.num_mels:
        raise RuntimeError(f'NHVSing ボコーダーの mel の次元 ({data.mel.num_mels}) が、'
                           f'音響モデルの mel の次元 ({stream_sizes[0]}) と違います')
    sample_rate = config.get('sample_rate', 48000)
    if sample_rate != data.sample_rate:
        raise RuntimeError(f'NHVSing ボコーダーのサンプリング周波数 ({data.sample_rate}) が、'
                           f'音源のサンプリング周波数 ({sample_rate}) と違います')
    hop_size = config.frame_period * sample_rate / 1000
    if hop_size != data.hop_size:
        raise RuntimeError(f'NHVSing ボコーダーの hop ({data.hop_size} サンプル) が、'
                           f'音源のフレーム周期 ({config.frame_period} ms = {hop_size:g} サンプル) と違います')


def load_nhvsing(path, device, acoustic_config, vocoder_config):
    try:
        import nhvsing  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            'この音源は NHVSing ボコーダーを使います。nhvsing パッケージを入れてください: '
            'python -m pip install --no-deps git+https://github.com/rokujyushi/NHVSing@package') from e
    from hydra.utils import instantiate

    path = Path(path)
    # 音源の設定 (sample_rate / frame_period / feature_type) は SPSVS と同じくモデルのフォルダの config.yaml にある
    check_compatible(OmegaConf.load(path.parent / 'config.yaml'), acoustic_config, vocoder_config)
    # パック済みモデルは state_dict だけなので weights_only=True で読める。weight norm は書き出し時に外してある
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    generator = instantiate(vocoder_config.generator)
    generator.load_state_dict(checkpoint['model']['generator'])
    generator.to(device).eval()
    return NHVSingWrapper(vocoder_config, generator), None, vocoder_config


@torch.no_grad()
def predict_waveform(engine, multistream_features, vuv_threshold=0.5):
    """NHVSing で波形を生成する。engine は vocoder が NHVSingWrapper の SPSVS。

    雑音は torch.normal で CPU 上に作るので、合成の前に torch.manual_seed を固定すれば
    デバイスによらず同じ波形になる (synthe は UST のハッシュからシードを決めている)。
    """
    mel, lf0, vuv = multistream_features
    device = engine.device
    x = torch.from_numpy(np.asarray(mel, dtype=np.float32)).unsqueeze(0).to(device)
    # nnsvs の lf0 ストリームは無声区間も補間した連続値
    cf0 = torch.from_numpy(np.exp(np.asarray(lf0, dtype=np.float32))).view(1, 1, -1).to(device)
    uv = torch.from_numpy((np.asarray(vuv) < vuv_threshold).astype(np.float32)).view(1, 1, -1).to(device)
    # 学習も書き出し時の出力の確認も fp32 なので、svs_synthe の autocast を切って fp32 で動かす
    # (NHVSing は小さいので fp32 でも速い)
    with torch.autocast(torch.device(device).type, enabled=False):
        wav = engine.vocoder.generator(x, cf0, uv)
    return wav.view(-1).cpu().numpy()


def apply():
    """nnsvs の load_vocoder を NHVSing 対応版に差し替える (何度呼んでもよい)。"""
    global _applied
    if _applied:
        return
    _applied = True

    import nnsvs.svs
    import nnsvs.util

    original = nnsvs.util.load_vocoder

    def load_vocoder(path, device, acoustic_config):
        config = _load_config(Path(path).parent)
        if config is not None and is_nhvsing_config(config):
            return load_nhvsing(path, device, acoustic_config, config)
        return original(path, device, acoustic_config)

    # nnsvs.svs は `from nnsvs.util import load_vocoder` で取り込んでいるので、両方差し替える
    nnsvs.util.load_vocoder = load_vocoder
    nnsvs.svs.load_vocoder = load_vocoder
