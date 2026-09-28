"""パック済みの NNSVS/ENUNU モデルフォルダから timelag / duration / acoustic を ONNX に書き出す。

    python -m onnx_export.export MODEL_DIR OUT_DIR

MODEL_DIR には config.yaml、{stage}_model.yaml、{stage}_model.pth、各 scaler の .npy がある
(旧形式の enuconfig.yaml のフォルダは、先に ENUNU が行う変換でこの形にしておく)。
OUT_DIR に {stage}.onnx と manifest.json (入出力名・次元・scaler・書き出せなかった理由) を作る。
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from .wrappers import InferenceWrapper, export_patches

logger = logging.getLogger(__name__)

STAGES = ('timelag', 'duration', 'acoustic')
OPSET = 17


def load_model(model_dir, stage):
    """{stage}_model.yaml と {stage}_model.pth からモデルを作る。"""
    model_dir = Path(model_dir)
    config = OmegaConf.load(model_dir / f'{stage}_model.yaml')
    model = instantiate(config.netG)
    checkpoint = torch.load(model_dir / f'{stage}_model.pth', map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval(), config


def load_scalers(model_dir, stage):
    """in/out の scaler を {名前: 配列のリスト} で返す。"""
    model_dir = Path(model_dir)
    scalers = {}
    for io, keys in (('in', ('min', 'scale')), ('out', ('mean', 'var', 'scale'))):
        arrays = {k: model_dir / f'{io}_{stage}_scaler_{k}.npy' for k in keys}
        if all(p.exists() for p in arrays.values()):
            scalers[io] = {k: np.load(p).astype(float).tolist() for k, p in arrays.items()}
    return scalers


def model_config(config):
    """C# 側の後処理 (MLPG など) に必要な、モデルの設定を取り出す。"""
    keys = ('stream_sizes', 'has_dynamic_features', 'num_windows', 'stream_weights')
    return {k: OmegaConf.to_container(config[k]) for k in keys if k in config}


def export_model(model, in_dim, path, opset=OPSET):
    """モデルを ONNX にして path に書く。書き出した入出力の情報を返す。

    in_dim は入力の次元。モデルによっては属性に持たないので、config の値を渡す。
    """
    wrapper = InferenceWrapper(model).eval()
    dummy = torch.randn(1, 37, in_dim)
    kwargs = {
        'input_names': ['x'],
        'output_names': wrapper.output_names,
        'dynamic_axes': {name: {0: 'batch', 1: 'frames'} for name in ['x', *wrapper.output_names]},
        'opset_version': opset,
    }
    with export_patches(), torch.no_grad():
        try:
            # torch 2.9 以降は既定が dynamo 版の exporter。動的な形状を確実に扱える TorchScript 版を使う
            torch.onnx.export(wrapper, (dummy,), str(path), dynamo=False, **kwargs)
        except TypeError:  # dynamo 引数のない古い torch
            torch.onnx.export(wrapper, (dummy,), str(path), **kwargs)
    return {'inputs': ['x'], 'outputs': wrapper.output_names, 'in_dim': int(in_dim)}


def export_dir(model_dir, out_dir, stages=STAGES):
    """モデルフォルダの各段を書き出す。段ごとの結果 (成功/失敗) を manifest に残す。"""
    model_dir, out_dir = Path(model_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'opset': OPSET, 'stages': {}}
    for stage in stages:
        entry = {}
        try:
            model, config = load_model(model_dir, stage)
            entry['class'] = config.netG['_target_']
            entry.update(export_model(model, config.netG.in_dim, out_dir / f'{stage}.onnx'))
            entry['file'] = f'{stage}.onnx'
            entry['scalers'] = load_scalers(model_dir, stage)
            entry['model_config'] = model_config(config)
        except Exception as e:  # 未対応のモデルは理由を残して次の段に進む
            logger.warning('%s: 書き出せませんでした: %s', stage, e)
            entry['error'] = f'{type(e).__name__}: {e}'
        manifest['stages'][stage] = entry
    with open(out_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('model_dir')
    parser.add_argument('out_dir')
    parser.add_argument('--stages', nargs='+', choices=STAGES, default=list(STAGES))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    manifest = export_dir(args.model_dir, args.out_dir, args.stages)
    for stage, entry in manifest['stages'].items():
        print(stage, entry.get('file') or f"失敗: {entry['error']}")


if __name__ == '__main__':
    main()
