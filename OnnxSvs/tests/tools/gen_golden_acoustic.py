"""acoustic 推論と MLPG の C# 実装のテスト用に、nnsvs / nnmnkwii の出力 (正解) を作る。

    python tests/tools/gen_golden_acoustic.py

- golden_mlpg.json: nnmnkwii の mlpg と nnsvs の multi_stream_mlpg (ランダムな入力)
- acoustic_models/ と golden_acoustic.json: ランダムな重みの acoustic モデル (MDN と決定的なモデル) を
  onnx_export で書き出し、同じモデルを nnsvs.gen.predict_acoustic で動かした結果
"""
import json
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_golden_timing import DATA, HED, REPO, write_model  # noqa: E402
from nnmnkwii import paramgen  # noqa: E402
from nnmnkwii.io import hts  # noqa: E402
from nnsvs import gen  # noqa: E402
from nnsvs.multistream import get_windows, multi_stream_mlpg  # noqa: E402
from nnsvs.util import MinMaxScaler, StandardScaler  # noqa: E402

sys.path.insert(0, str(REPO))
from onnx_export.export import export_dir, load_model  # noqa: E402

LABEL = DATA / 'sample_full.lab'
M = 'nnsvs.model.'

# 静的特徴の構成: mgc 8 次元 + lf0 1 + vuv 1 + bap 2 (lf0 と bap は動的特徴あり、vuv はなし)
STREAM_SIZES = [8, 1, 1, 2]
HAS_DYNAMIC = [True, True, False, True]
NUM_WINDOWS = 3
# nnsvs の stream_sizes は動的特徴を含んだ次元 (例: [180, 3, 1, 15])
STREAM_FULL = [s * (NUM_WINDOWS if d else 1) for s, d in zip(STREAM_SIZES, HAS_DYNAMIC)]
OUT_DIM = sum(STREAM_FULL)

VARIANTS = {
    'mdn': ((M + 'Conv1dResnetMDN', dict(hidden_dim=16, num_layers=2, num_gaussians=2)), 1.5),
    'det': ((M + 'Conv1dResnet', dict(hidden_dim=16, num_layers=2)), 3.0),
}


def gen_mlpg(rng):
    windows = get_windows(3)
    cases = {}
    for T, static_dim in ((30, 4), (5, 2), (2, 1)):
        mean = rng.normal(size=(T, static_dim * 3))
        var = rng.uniform(0.2, 2.0, size=(T, static_dim * 3))
        y = paramgen.mlpg(mean, var, windows)
        cases[f'{T}x{static_dim}'] = {'mean': mean.tolist(), 'var': var.tolist(), 'y': y.tolist()}
    # ストリームごと (動的特徴ありとなし)、分散は全フレーム共通
    T = 25
    mean = rng.normal(size=(T, OUT_DIM))
    gv = rng.uniform(0.2, 2.0, size=OUT_DIM)
    y = multi_stream_mlpg(mean, gv, windows, STREAM_FULL, HAS_DYNAMIC)
    cases['multistream'] = {'mean': mean.tolist(), 'var': gv.tolist(), 'y': y.tolist(),
                            'stream_sizes': STREAM_FULL, 'has_dynamic': HAS_DYNAMIC}
    return cases


def run_python(model_dir, binary_dict, numeric_dict):
    labels = hts.load(LABEL).round_()
    model, _ = load_model(model_dir, 'acoustic')
    cfg = OmegaConf.load(model_dir / 'acoustic_model.yaml')
    in_scaler = MinMaxScaler(np.load(model_dir / 'in_acoustic_scaler_min.npy'),
                             np.load(model_dir / 'in_acoustic_scaler_scale.npy'))
    out_scaler = StandardScaler(np.load(model_dir / 'out_acoustic_scaler_mean.npy'),
                                np.load(model_dir / 'out_acoustic_scaler_var.npy'),
                                np.load(model_dir / 'out_acoustic_scaler_scale.npy'))
    results = {}
    for shift in (0, 100):
        pred = gen.predict_acoustic('cpu', labels, model, cfg, in_scaler, out_scaler,
                                    binary_dict, numeric_dict, f0_shift_in_cent=shift)
        rows = sorted(set(range(0, len(pred), 97)) | {len(pred) - 1})
        results[f'shift{shift}'] = {
            'frames': len(pred), 'dim': pred.shape[1], 'rows': rows,
            'y': np.asarray(pred, dtype=float)[rows].tolist(),
        }
    return results


def main():
    rng = np.random.default_rng(0)
    (DATA / 'golden_mlpg.json').write_text(json.dumps(gen_mlpg(rng)), encoding='utf-8')

    binary_dict, numeric_dict = hts.load_question_set(HED)
    in_dim = len(binary_dict) + len(numeric_dict) + 4  # コーステコーディングの 4 列
    models_dir = DATA / 'acoustic_models'
    if models_dir.exists():
        shutil.rmtree(models_dir)
    golden = {}
    for i, (name, ((target, params), weight_scale)) in enumerate(VARIANTS.items()):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            write_model(
                model_dir, 'acoustic', target, params, in_dim, seed=30 + i, weight_scale=weight_scale,
                out_dim=OUT_DIM,
                extra_config={'stream_sizes': STREAM_FULL,
                              'has_dynamic_features': HAS_DYNAMIC, 'num_windows': NUM_WINDOWS},
                out_mean=rng.normal(size=OUT_DIM), out_scale=rng.uniform(0.5, 2.0, size=OUT_DIM))
            export_dir(model_dir, models_dir / name, stages=['acoustic'])
            golden[name] = run_python(model_dir, binary_dict, numeric_dict)
    out = DATA / 'golden_acoustic.json'
    out.write_text(json.dumps(golden), encoding='utf-8')
    print(out, out.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
