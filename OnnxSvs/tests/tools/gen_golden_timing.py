"""timelag / duration の C# 実装のテスト用に、ONNX モデルと nnsvs の出力 (正解) を作る。

    python tests/tools/gen_golden_timing.py

ランダムな重みのモデル (確率モデルと決定的なモデル) を作って onnx_export で書き出し、
同じモデルを nnsvs.gen の predict_timelag / predict_duration / postprocess_duration で動かした結果を
golden_timing.json に保存する。
"""
import json
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

warnings.filterwarnings('ignore')
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from nnmnkwii.io import hts  # noqa: E402
from nnsvs import gen  # noqa: E402
from nnsvs.util import MinMaxScaler, StandardScaler  # noqa: E402
from sklearn.preprocessing import MinMaxScaler as SkMinMaxScaler  # noqa: E402

from onnx_export.export import export_dir, load_model  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / 'OnnxSvs.Tests' / 'data'
HED = DATA / 'jp_qst001_nnsvs.hed'
LABEL = DATA / 'sample_score.lab'
M = 'nnsvs.model.'

# 名前: (timelag のクラスと引数, duration のクラスと引数)
VARIANTS = {
    'det': (
        (M + 'VariancePredictor', dict(hidden_dim=16, num_layers=2)),
        (M + 'VariancePredictor', dict(hidden_dim=16, num_layers=2)),
    ),
    'mdn': (
        (M + 'MDN', dict(hidden_dim=16, num_layers=2, num_gaussians=2)),
        (M + 'RMDN', dict(hidden_dim=8, num_layers=1, num_gaussians=2)),
    ),
}
WEIGHT_SCALE = {'det': 2.0, 'mdn': 6.0}
# 出力の scaler (平均, 標準偏差)。time-lag は ±数フレーム、音素長は数〜十数フレーム程度になる値にする
OUT_SCALE = {'timelag': (0.0, 3.0), 'duration': (12.0, 5.0)}


def write_model(model_dir, stage, target, params, in_dim, seed, weight_scale):
    cls = getattr(__import__(target.rsplit('.', 1)[0], fromlist=['x']), target.rsplit('.', 1)[1])
    kwargs = dict(in_dim=in_dim, out_dim=1, **params)
    torch.manual_seed(seed)
    model = cls(**kwargs).eval()
    with torch.no_grad():  # 初期値のままだと出力がほぼ一定になり、テストの意味が薄いので重みを大きくする
        for param in model.parameters():
            param.mul_(weight_scale)
    torch.save({'state_dict': model.state_dict()}, model_dir / f'{stage}_model.pth')
    with open(model_dir / f'{stage}_model.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump({
            'stream_sizes': [1],
            'has_dynamic_features': [False],
            'netG': {'_target_': target, **kwargs},
        }, f)
    # nnsvs のパック済みモデルの scaler は float32 (float64 だと nnsvs 自身がモデルに double を渡して失敗する)
    f32 = np.float32
    np.save(model_dir / f'in_{stage}_scaler_min.npy', np.full(in_dim, -0.1, dtype=f32))
    np.save(model_dir / f'in_{stage}_scaler_scale.npy', np.full(in_dim, 0.02, dtype=f32))
    mean, scale = OUT_SCALE[stage]
    np.save(model_dir / f'out_{stage}_scaler_mean.npy', np.array([mean], dtype=f32))
    np.save(model_dir / f'out_{stage}_scaler_scale.npy', np.array([scale], dtype=f32))
    np.save(model_dir / f'out_{stage}_scaler_var.npy', np.array([scale ** 2], dtype=f32))


def scalers(model_dir, stage):
    in_min = np.load(model_dir / f'in_{stage}_scaler_min.npy')
    in_scale = np.load(model_dir / f'in_{stage}_scaler_scale.npy')
    out = StandardScaler(
        np.load(model_dir / f'out_{stage}_scaler_mean.npy'),
        np.load(model_dir / f'out_{stage}_scaler_var.npy'),
        np.load(model_dir / f'out_{stage}_scaler_scale.npy'),
    )
    return in_min, in_scale, out


def sklearn_minmax(in_min, in_scale):
    """クリップの判定 (isinstance(sklearn の MinMaxScaler)) に当たる scaler。"""
    sk = SkMinMaxScaler().fit(np.array([[0.0] * len(in_min), [1.0] * len(in_min)]))
    sk.min_, sk.scale_ = in_min, in_scale
    return sk


def run_python(model_dir, labels_path, binary_dict, numeric_dict, clip):
    labels = hts.load(labels_path)
    labels.frame_shift = 50000
    labels.round_()
    result = {}
    parts = {}
    for stage in ('timelag', 'duration'):
        model, config = load_model(model_dir, stage)
        in_min, in_scale, out = scalers(model_dir, stage)
        in_scaler = sklearn_minmax(in_min, in_scale) if clip else MinMaxScaler(in_min, in_scale)
        cfg = OmegaConf.load(model_dir / f'{stage}_model.yaml')
        parts[stage] = (model, cfg, in_scaler, out)
    lag = gen.predict_timelag(
        'cpu', labels, parts['timelag'][0], parts['timelag'][1], parts['timelag'][2], parts['timelag'][3],
        binary_dict, numeric_dict, force_clip_input_features=clip)
    dur = gen.predict_duration(
        'cpu', labels, parts['duration'][0], parts['duration'][1], parts['duration'][2], parts['duration'][3],
        binary_dict, numeric_dict, force_clip_input_features=clip)
    out_labels = gen.postprocess_duration(labels, dur, lag)
    result['lag'] = np.asarray(lag, dtype=float).reshape(-1).tolist()
    if isinstance(dur, tuple):
        result['duration_mean'] = np.asarray(dur[0], dtype=float).reshape(-1).tolist()
        result['duration_sigma2'] = np.asarray(dur[1], dtype=float).reshape(-1).tolist()
    else:
        result['duration_mean'] = np.asarray(dur, dtype=float).reshape(-1).tolist()
    result['labels'] = [
        [int(s), int(e), c]
        for s, e, c in zip(out_labels.start_times, out_labels.end_times, out_labels.contexts)
    ]
    return result


def main():
    binary_dict, numeric_dict = hts.load_question_set(HED)
    in_dim = len(binary_dict) + len(numeric_dict)
    golden = {}
    models_dir = DATA / 'timing_models'
    if models_dir.exists():
        shutil.rmtree(models_dir)
    for i, (name, (tl, du)) in enumerate(VARIANTS.items()):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            write_model(model_dir, 'timelag', *tl, in_dim, seed=10 + i, weight_scale=WEIGHT_SCALE[name])
            write_model(model_dir, 'duration', *du, in_dim, seed=20 + i, weight_scale=WEIGHT_SCALE[name])
            export_dir(model_dir, models_dir / name, stages=['timelag', 'duration'])
            for clip in (False, True):
                golden[f'{name}_clip' if clip else name] = run_python(
                    model_dir, LABEL, binary_dict, numeric_dict, clip)
    out = DATA / 'golden_timing.json'
    out.write_text(json.dumps(golden), encoding='utf-8')
    print(out, out.stat().st_size, 'bytes')
    for k, v in golden.items():
        print(k, 'lag', v['lag'][:6], 'dur', [round(x, 1) for x in v['duration_mean'][:6]])


if __name__ == '__main__':
    main()
