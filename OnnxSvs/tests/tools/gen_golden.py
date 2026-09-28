"""C# 側のテスト用に、nnmnkwii の出力 (正解) を JSON に書き出す。

    python tests/tools/gen_golden.py

data/ の .hed と sample_full.lab から、次を golden_frontend.json に保存する。
- phone: 音素単位の特徴 (subphone_features=None, add_frame_features=False)
- frames_per_phone / cc: フレーム単位 (coarse_coding) のフレーム数と、末尾 4 列 (位置 3 つ + 音素長)
- rows: フレーム単位の特徴の一部の行 (全次元)
"""
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore')
from nnmnkwii.frontend import merlin  # noqa: E402
from nnmnkwii.io import hts  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / 'OnnxSvs.Tests' / 'data'


def main():
    labels = hts.load(DATA / 'sample_full.lab').round_()  # nnsvs も特徴量の前に丸める
    golden = {}
    for hed in ('jp_qst001_nnsvs.hed', 'jp_dev_latest.hed'):
        binary_dict, numeric_dict = hts.load_question_set(DATA / hed)
        phone = merlin.linguistic_features(
            labels, binary_dict, numeric_dict, add_frame_features=False, subphone_features=None)
        frame = merlin.linguistic_features(
            labels, binary_dict, numeric_dict, add_frame_features=True,
            subphone_features='coarse_coding', frame_shift=50000)
        dim = len(binary_dict) + len(numeric_dict)
        frames_per_phone = [int(e // 50000 - s // 50000) for s, e in zip(labels.start_times, labels.end_times)]
        assert frame.shape == (sum(frames_per_phone), dim + 4)
        pick = sorted({0, 1, 7, frame.shape[0] // 2, frame.shape[0] - 1})
        golden[hed] = {
            'dim': dim,
            'phone': phone.astype(float).tolist(),
            'frames_per_phone': frames_per_phone,
            'cc': frame[:, dim:].astype(float).tolist(),
            'rows': {str(i): frame[i].astype(float).tolist() for i in pick},
        }
    out = DATA / 'golden_frontend.json'
    out.write_text(json.dumps(golden), encoding='utf-8')
    print(out, out.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
