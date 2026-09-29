# 移植元

`src/OnnxSvs` の一部は、次の Python のコードを C# に移したものです。どちらも MIT License です。

| C# | 移植元 |
|---|---|
| `QuestionSet` (質問ファイルの読み込み、`Wildcards2Regex`) | nnmnkwii `nnmnkwii/io/hts.py` の `load_question_set` / `wildcards2regex` |
| `HtsLabel` (ラベルの読み込み、`Rounded`) | nnmnkwii `nnmnkwii/io/hts.py` の `HTSLabelFile.load` / `round_` |
| `LinguisticFeatures` (言語特徴量、コーステコーディング) | nnmnkwii `nnmnkwii/frontend/merlin.py` |
| `Scaler` (MinMax / Standard) | NNSVS `nnsvs/util.py` の `MinMaxScaler` / `StandardScaler` |
| `TimingPredictor`、`Conditioning` (time-lag / 音素長の予測、音高の加工、音素長の決定) | NNSVS `nnsvs/gen.py` の `predict_timelag` / `predict_duration` / `postprocess_duration`、`nnsvs/io/hts.py` の `get_note_indices` / `get_pitch_indices`、nnmnkwii `preprocessing/f0.py` の `interp1d` |
| `Mlpg` (MLPG、ストリームごとの MLPG) | nnmnkwii `nnmnkwii/paramgen/_mlpg.py` の `mlpg`、NNSVS `nnsvs/multistream.py` の `multi_stream_mlpg` |
| `AcousticPredictor` (acoustic の推論) | NNSVS `nnsvs/gen.py` の `predict_acoustic` |
| `AcousticPostprocess` (GV、vuv 補正、休符埋め、なめらか化) | NNSVS `nnsvs/gen.py` の `postprocess_acoustic` / `gen_spsvs_static_features` / `correct_vuv_by_phone` / `_get_nonrest_frame_soft_mask` / `_fill_silence_to_world_params`、`nnsvs/postfilters.py` の `variance_scaling`、`nnsvs/util.py` の `extract_static_scaler` |
| `Sptk` (mcepalpha、freqt、mc2sp) | pysptk `pysptk/util.py` の `mcepalpha`、`pysptk/conversion.py` の `mc2sp` (MIT License, Copyright (c) 2015 Ryuichi Yamamoto)。`freqt` は SPTK (Modified BSD License。著作権表示は SPTK の LICENSE を参照) の同名の関数と同じ式 (pysptk の出力との一致で確かめた) |
| `WorldVocoder` (WORLD のパラメータ作り) | NNSVS `nnsvs/gen.py` の `gen_world_params`、pyworld の `get_cheaptrick_fft_size` (WORLD: BSD 3-Clause License、mmorise/World。著作権表示は WORLD の LICENSE を参照) |
| `WorldlineNative` (P/Invoke の宣言) | OpenUtau `OpenUtau.Core/Render/Worldline.cs` の `DecodeMgc` / `DecodeBap` / `WorldSynthesis` の宣言 (MIT License, Copyright (c) 2014 StAkira)。ライブラリ本体は含めない |
| `Dsp` (ローパスフィルタ) | NNSVS `nnsvs/dsp.py` の `lowpass_filter`。SciPy (BSD-3-Clause) の `signal.butter` / `filtfilt` / `lfilter_zi` と同じ手順を、ドキュメントとソースを読んで C# で書き直したもの |

- nnmnkwii: https://github.com/r9y9/nnmnkwii (Copyright (c) 2017 Ryuichi Yamamoto)
- NNSVS: https://github.com/nnsvs/nnsvs (Copyright (c) 2020 Ryuichi Yamamoto)

## テストデータ

`tests/OnnxSvs.Tests/data` には次のデータが入っています。

- `jp_qst001_nnsvs.hed`、`jp_dev_latest.hed`: NNSVS のレシピ (MIT License) のもの
- `sample_score.lab`: 下の `sample_full.lab` から、音符ごとに時刻をそろえて作ったもの (同じく CC BY 3.0)
- `timing_models/`、`golden_timing.json`: ランダムな重みのモデルと、それを nnsvs で動かした結果 (元データを含まない。ラベルは `sample_score.lab` から)
- `sample_full.lab`: NNSVS のテスト用ラベル `tests/data/nitech_jp_song070_f001_004.lab` の先頭 40 行。
  元は Nagoya Institute of Technology の日本語歌声データベース "NIT SONG070 F001" で、
  **Creative Commons Attribution 3.0** のライセンスです。
  Copyright (c) 2004-2015 Nagoya Institute of Technology, Department of Computer Science
  (released by HTS Working Group, http://hts.sp.nitech.ac.jp/)。
  ラベルの一部を切り出して使っています。
- `acoustic_models/`、`golden_mlpg.json`、`golden_acoustic.json`、`golden_postprocess.json`: ランダムな重み・乱数の入力と、それを nnsvs / nnmnkwii で動かした結果。`golden_acoustic.json` と `golden_postprocess.json` は `sample_full.lab` のフレーム数・ラベルから作っており、同じく CC BY 3.0 の表示が必要です。
- `golden_frontend.json`: 上の 2 つから nnmnkwii で計算した特徴量。`sample_full.lab` の派生物なので、同じく CC BY 3.0 の表示が必要です。

## 実行時に参照するパッケージ (NuGet)

- Microsoft.ML.OnnxRuntime: MIT License (リポジトリには含めず、ビルド時に取得)
- テストのみ: xunit、xunit.runner.visualstudio (Apache-2.0)、Microsoft.NET.Test.Sdk (MIT)

移植は Python のソースを読んで行い、OpenUtau の C# 実装 (`EnunuOnnx`) のコードは使っていません。
