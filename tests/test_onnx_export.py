"""onnx_export の出力が、PyTorch の model.inference と一致することを確かめる。

    python -m pytest tests/test_onnx_export.py
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from onnx_export.export import export_dir, load_model  # noqa: E402

M = 'nnsvs.model.'
A = 'nnsvs.acoustic_models.'
IN_DIM = 40
# (クラス, 引数, 出力次元)。実際の ENUNU モデルで使われているクラスを含む
CASES = {
    'MDN': (M + 'MDN', dict(hidden_dim=32, num_layers=2, num_gaussians=3), 1),
    'MDNv2': (M + 'MDNv2', dict(hidden_dim=32, num_layers=2, num_gaussians=3), 1),
    'MDN_dimwise': (M + 'MDN', dict(hidden_dim=32, num_layers=2, num_gaussians=3, dim_wise=True), 4),
    'VariancePredictor': (M + 'VariancePredictor', dict(hidden_dim=32, num_layers=3), 1),
    'FFN': (M + 'FFN', dict(hidden_dim=32, num_layers=2), 1),
    'LSTMRNN': (M + 'LSTMRNN', dict(hidden_dim=16, num_layers=2), 1),
    'RMDN': (M + 'RMDN', dict(hidden_dim=16, num_layers=2, num_gaussians=3), 1),
    'RMDN_dimwise': (M + 'RMDN', dict(hidden_dim=16, num_layers=2, num_gaussians=3, dim_wise=True), 5),
    'Conv1dResnet': (M + 'Conv1dResnet', dict(hidden_dim=32, num_layers=3), 20),
    'Conv1dResnetMDN': (M + 'Conv1dResnetMDN', dict(hidden_dim=32, num_layers=3, num_gaussians=3), 20),
    'FFConvLSTM': (M + 'FFConvLSTM', dict(ff_hidden_dim=32, conv_hidden_dim=32, lstm_hidden_dim=16), 12),
    'ResSkipF0FFConvLSTM': (
        A + 'ResSkipF0FFConvLSTM',
        dict(ff_hidden_dim=32, conv_hidden_dim=32, lstm_hidden_dim=16, in_lf0_idx=IN_DIM - 1,
             out_lf0_idx=0),
        12,
    ),
    'ResF0Conv1dResnet': (
        A + 'ResF0Conv1dResnet',
        dict(hidden_dim=32, num_layers=3, in_lf0_idx=IN_DIM - 1, out_lf0_idx=0),
        20,
    ),
}


def make_model_dir(tmp_path, target, params, out_dim, stage='timelag'):
    """ランダムな重みのモデルを、パック済みモデルフォルダの形で保存する。"""
    cls = load_class(target)
    kwargs = dict(in_dim=IN_DIM, out_dim=out_dim, **params)
    torch.manual_seed(0)
    model = cls(**kwargs).eval()
    torch.save({'state_dict': model.state_dict()}, tmp_path / f'{stage}_model.pth')
    with open(tmp_path / f'{stage}_model.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump({'netG': {'_target_': target, **kwargs}}, f)
    rng = np.random.default_rng(0)
    np.save(tmp_path / f'in_{stage}_scaler_min.npy', rng.normal(size=IN_DIM))
    np.save(tmp_path / f'in_{stage}_scaler_scale.npy', rng.uniform(0.5, 2, size=IN_DIM))
    for k in ('mean', 'var', 'scale'):
        np.save(tmp_path / f'out_{stage}_scaler_{k}.npy', rng.uniform(0.5, 2, size=out_dim))
    return model


def load_class(target):
    module, name = target.rsplit('.', 1)
    return getattr(__import__(module, fromlist=[name]), name)


@pytest.mark.parametrize('name', CASES)
def test_onnx_matches_torch(tmp_path, name):
    target, params, out_dim = CASES[name]
    model_dir, out_dir = tmp_path / 'model', tmp_path / 'onnx'
    model_dir.mkdir()
    make_model_dir(model_dir, target, params, out_dim)

    manifest = export_dir(model_dir, out_dir, stages=['timelag'])
    entry = manifest['stages']['timelag']
    assert 'error' not in entry, entry.get('error')
    assert set(entry['scalers']) == {'in', 'out'}

    model, _ = load_model(model_dir, 'timelag')
    session = ort.InferenceSession(str(out_dir / entry['file']))
    # 書き出しに使った長さ (37) と違うフレーム数でも動くこと
    for frames in (53, 7):
        x = torch.randn(1, frames, IN_DIM)
        with torch.no_grad():
            expected = model.inference(x, torch.tensor([frames]))
        expected = expected if isinstance(expected, tuple) else (expected,)
        actual = session.run(entry['outputs'], {'x': x.numpy()})
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            np.testing.assert_allclose(a, e.numpy(), rtol=1e-4, atol=1e-5)


def test_unsupported_model_is_reported(tmp_path):
    """未対応のモデルは、例外で止まらず manifest に理由が残る。"""
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    (model_dir / 'timelag_model.yaml').write_text('netG:\n  _target_: nnsvs.model.DoesNotExist\n')
    manifest = export_dir(model_dir, tmp_path / 'onnx', stages=['timelag'])
    assert 'error' in manifest['stages']['timelag']
