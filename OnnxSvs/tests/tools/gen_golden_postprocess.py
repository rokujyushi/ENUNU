"""acoustic の後処理 (GV、vuv 補正、休符埋め、軌跡のなめらか化) の C# 実装のテスト用に、
nnsvs.gen.postprocess_acoustic の出力 (正解) を作る。

    python tests/tools/gen_golden_postprocess.py

入力の音響特徴は乱数で作った (モデルは使わない)。golden_postprocess.json に、入力と、設定ごとの出力
(行は一定間隔で抜き出したもの) を書く。
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_golden_acoustic import HAS_DYNAMIC, LABEL, NUM_WINDOWS, STREAM_FULL, STREAM_SIZES  # noqa: E402
from gen_golden_timing import DATA, HED  # noqa: E402
from nnmnkwii.io import hts  # noqa: E402
from nnsvs import gen  # noqa: E402
from nnsvs.util import StandardScaler  # noqa: E402

CASES = {
    # 既定 (GV あり、なめらか化あり)
    'default': dict(),
    # vuv の補正、休符埋め、f0 のシフト
    'fix_fill_shift': dict(force_fix_vuv=True, fill_silence_to_rest=True, f0_shift_in_cent=50,
                           vuv_threshold=0.3),
    # 相対 f0、GV なし、なめらか化なし
    'relative_plain': dict(relative_f0=True, post_filter_type='none', trajectory_smoothing=False),
}


def smooth_noise(rng, n, width):
    k = np.ones(width) / width
    return np.convolve(rng.normal(size=n + width - 1), k, mode='valid')


def make_features(rng, T):
    mgc = np.stack([smooth_noise(rng, T, 9) * 2 + (3.0 if d == 0 else 0.0) for d in range(STREAM_SIZES[0])], 1)
    mgc += rng.normal(scale=0.05, size=mgc.shape)
    lf0 = (np.log(300) + 0.1 * np.sin(np.arange(T) / 40) + rng.normal(scale=0.02, size=T))[:, None]
    vuv = np.clip(0.5 + smooth_noise(rng, T, 15) * 4, 0, 1)[:, None]
    bap = rng.normal(loc=-20, scale=25, size=(T, STREAM_SIZES[3]))
    feats = np.concatenate([mgc, lf0, vuv, bap], 1)
    return np.round(feats, 5).astype(np.float32)


def main():
    rng = np.random.default_rng(1)
    binary_dict, numeric_dict = hts.load_question_set(HED)
    labels = hts.load(LABEL).round_()
    T = int(labels.num_frames())
    feats = make_features(rng, T)
    static_var = np.round(rng.uniform(0.5, 2.0, size=feats.shape[1]), 5)
    scaler = StandardScaler(np.zeros(feats.shape[1]), static_var, np.sqrt(static_var))
    config = OmegaConf.create({'stream_sizes': STREAM_FULL, 'has_dynamic_features': HAS_DYNAMIC,
                               'num_windows': NUM_WINDOWS})
    rows = [int(r) for r in sorted(set(range(0, T, 11)) | {T - 1})]
    golden = {'frames': T, 'features': feats.tolist(), 'static_var': static_var.tolist(),
              'rows': rows, 'cases': {}}
    for name, kwargs in CASES.items():
        mgc, lf0, vuv, bap = gen.postprocess_acoustic(
            'cpu', feats.copy(), labels, binary_dict, numeric_dict, config, scaler,
            sample_rate=48000, frame_period=5, feature_type='world', **{'post_filter_type': 'gv', **kwargs})
        golden['cases'][name] = {
            'options': kwargs,
            'mgc': np.asarray(mgc, dtype=float)[rows].tolist(),
            'lf0': np.asarray(lf0, dtype=float)[rows, 0].tolist(),
            'vuv': np.asarray(vuv, dtype=float)[rows, 0].tolist(),
            'bap': np.asarray(bap, dtype=float)[rows].tolist(),
        }
    out = DATA / 'golden_postprocess.json'
    out.write_text(json.dumps(golden), encoding='utf-8')
    print(out, out.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
