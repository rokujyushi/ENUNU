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
    # ビブラートのストリーム (正弦波方式: 6 ストリーム、差分方式: 5 ストリーム)
    'vib_sine': dict(vibrato_scale=1.5, _input='sine'),
    'vib_diff': dict(vibrato_scale=0.5, trajectory_smoothing=False, _input='diff'),
}
# 静的な次元 (mgc, lf0, vuv, bap, [vib, vib_flags]) と、動的特徴があるか
LAYOUTS = {
    'base': ([8, 1, 1, 2], [True, True, False, True]),
    'diff': ([8, 1, 1, 2, 1], [True, True, False, True, True]),
    'sine': ([8, 1, 1, 2, 2, 1], [True, True, False, True, True, False]),
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


def add_vibrato_streams(rng, base, T):
    """base (T x 12) の後ろに、差分方式 (1 列)、正弦波方式 (振幅 cent、周波数 Hz、フラグ) のビブラートを足す。"""
    on = np.zeros(T, dtype=bool)
    for start, length in ((300, 160), (700, 220), (T - 120, 120)):  # 最後は末尾まで続く
        on[start:start + length] = True
        # ビブラートの区間とその先は有声にしておく (無声を挟むと、nnsvs のなめらか化で f0 が負になり log が NaN になる)
        base[max(0, start - 10):start + length + 60, 9] = 1.0
    m_a = np.where(on, rng.uniform(20, 170, size=T), 0.0)
    m_f = np.where(on, rng.uniform(2, 9, size=T), 0.0)
    flags = on.astype(float)
    # 差分方式は f0 (Hz) にそのまま足すので、無声のフレームは 0 にしておく (nnsvs は無声でも足して、負の f0 に log をかける)
    diff = np.sin(np.arange(T) / 6.0) * 6.0 * (base[:, 9] >= 0.5)
    sine = np.concatenate([base, np.round(np.stack([m_a, m_f, flags], 1), 5)], 1).astype(np.float32)
    diff = np.concatenate([base, np.round(diff[:, None], 5)], 1).astype(np.float32)
    return {'base': base, 'diff': diff, 'sine': sine}


def full_sizes(static, dynamic):
    return [s * (NUM_WINDOWS if d else 1) for s, d in zip(static, dynamic)]


def main():
    rng = np.random.default_rng(1)
    binary_dict, numeric_dict = hts.load_question_set(HED)
    labels = hts.load(LABEL).round_()
    T = int(labels.num_frames())
    inputs = add_vibrato_streams(rng, make_features(rng, T), T)
    feats = inputs['base']
    static_var = np.round(rng.uniform(0.5, 2.0, size=inputs['sine'].shape[1]), 5)
    rows = [int(r) for r in sorted(set(range(0, T, 11)) | {T - 1})]
    golden = {'frames': T, 'static_var': static_var.tolist(), 'rows': rows,
              'inputs': {'base': inputs['base'].tolist(),  # vib_* は base の後ろに足す列だけ
                         'diff': inputs['diff'][:, 12:].tolist(), 'sine': inputs['sine'][:, 12:].tolist()},
              'layouts': {k: v[0] for k, v in LAYOUTS.items()}, 'cases': {}}
    for name, kwargs in CASES.items():
        kwargs = dict(kwargs)
        layout = kwargs.pop('_input', 'base')
        static, dynamic = LAYOUTS[layout]
        feats = inputs[layout]
        var = static_var[:feats.shape[1]]
        scaler = StandardScaler(np.zeros(len(var)), var, np.sqrt(var))
        config = OmegaConf.create({'stream_sizes': full_sizes(static, dynamic),
                                   'has_dynamic_features': dynamic, 'num_windows': NUM_WINDOWS})
        mgc, lf0, vuv, bap = gen.postprocess_acoustic(
            'cpu', feats.copy(), labels, binary_dict, numeric_dict, config, scaler,
            sample_rate=48000, frame_period=5, feature_type='world', **{'post_filter_type': 'gv', **kwargs})
        golden['cases'][name] = {
            'options': kwargs, 'input': layout,
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
