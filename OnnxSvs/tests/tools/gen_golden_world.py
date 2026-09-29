"""WORLD 合成の前処理 (メルケプストラムからスペクトルへ、WORLD のパラメータ作り) の C# 実装のテスト用に、
pysptk / pyworld / nnsvs の出力 (正解) を作る。

    python tests/tools/gen_golden_world.py

- golden_world.json: mcepalpha、cheaptrick の FFT 長、mc2sp、gen_world_params (メルケプストラム版)
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pysptk
import pyworld

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_golden_timing import DATA  # noqa: E402
from nnsvs.gen import gen_world_params  # noqa: E402

FS_LIST = [16000, 22050, 24000, 44100, 44600, 48000]


def main():
    rng = np.random.default_rng(2)
    golden = {
        'alpha': {str(fs): float(pysptk.util.mcepalpha(fs)) for fs in FS_LIST},
        'fft_size': {str(fs): int(pyworld.get_cheaptrick_fft_size(fs)) for fs in FS_LIST},
    }

    # mc2sp: 次数、alpha、fftlen を変えて (スペクトルは対数で比べる)
    cases = []
    for order, alpha, fftlen in ((59, 0.55, 2048), (24, 0.42, 512), (10, 0.35, 96)):
        mc = np.round(rng.normal(scale=0.4, size=(3, order + 1)) * np.exp(-np.arange(order + 1) / 12), 6)
        mc[:, 0] += 2.0
        sp = pysptk.mc2sp(np.ascontiguousarray(mc), alpha=alpha, fftlen=fftlen)
        cases.append({'alpha': alpha, 'fftlen': fftlen, 'mc': mc.tolist(), 'log_sp': np.log(sp).tolist()})
    golden['mc2sp'] = cases

    # gen_world_params (use_world_codec=False、メルケプストラムの bap)。
    # mgc の 0 次は大きめ、bap は小さい値にして、aperiodicity が 0 から 1 の外に出る場合も含める
    T, fs = 6, 48000
    mgc = np.round(rng.normal(scale=0.3, size=(T, 60)) * np.exp(-np.arange(60) / 15), 6)
    mgc[:, 0] += 3.0
    bap = np.round(rng.normal(scale=0.6, size=(T, 8)) * np.exp(-np.arange(8) / 4), 6)
    bap[:, 0] -= 1.5
    lf0 = np.round(np.log(np.array([220, 230, 240, 250, 260, 270.0])), 6)[:, None]
    vuv = np.array([[0.9], [0.2], [0.7], [0.5], [0.49], [1.0]])
    f0, sp, ap = gen_world_params(mgc, lf0, vuv, bap, fs, vuv_threshold=0.5, use_world_codec=False)
    golden['gen_world_params'] = {
        'fs': fs, 'vuv_threshold': 0.5, 'mgc': mgc.tolist(), 'lf0': lf0[:, 0].tolist(),
        'vuv': vuv[:, 0].tolist(), 'bap': bap.tolist(),
        'f0': f0.tolist(), 'log_sp': np.log(sp).tolist(), 'ap': ap.tolist(),
    }
    out = DATA / 'golden_world.json'
    out.write_text(json.dumps(golden), encoding='utf-8')
    print(out, out.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
