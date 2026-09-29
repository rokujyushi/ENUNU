# OnnxSvs

`onnx_export` で書き出した ONNX モデルを、ONNX Runtime (C#) で動かすための推論ライブラリ。
Python を使わずに、HTS ラベルから音声を作るところまでを目指しています (作業中)。

| 段 | 状態 |
|---|---|
| ラベル (`HtsLabel`)、質問ファイル (`QuestionSet`)、言語特徴量 (`LinguisticFeatures`)、scaler (`Scaler`) | 実装済み。nnmnkwii の出力と一致することをテスト済み |
| ONNX 推論 (`ModelStage`)、timelag / duration の予測と音素長の決定 (`TimingPredictor`) | 実装済み。nnsvs の出力と一致することをテスト済み。動的特徴量 (MLPG) を使う timelag / duration は未対応 |
| acoustic の ONNX 推論 (`AcousticPredictor`)、MLPG (`Mlpg`) | 実装済み。nnsvs / nnmnkwii の出力と一致することをテスト済み (MDN と決定的モデル、f0 シフトあり・なし) |
| acoustic の後処理 (`AcousticPostprocess`、`Dsp`): GV、ストリーム分割、vuv 補正、相対 f0、休符埋め、軌跡のなめらか化 | 実装済み (WORLD のみ)。nnsvs の出力と一致することをテスト済み。ビブラートのストリーム (差分方式・正弦波方式、`Vibrato`) にも対応。merlin / 学習済み post-filter は未対応 |
| WORLD による合成 | 未実装 |

後処理のあとに、外で作った f0 の変化 (UST のビブラートなど) を重ねるときは `WorldParams.WithF0DeltaCents` を使います。

## テスト

```
dotnet test
```

`tests/OnnxSvs.Tests/data/golden_frontend.json` は、Python の nnmnkwii の出力 (正解) です。
`.hed` やラベルを変えたときは `python tests/tools/gen_golden.py` で作り直します (nnsvs と nnmnkwii が必要)。
`golden_timing.json` と `timing_models/` (ランダムな重みの小さなモデルを onnx_export で書き出したもの) は
`python tests/tools/gen_golden_timing.py` で作り直します。
`golden_mlpg.json`、`golden_acoustic.json`、`acoustic_models/` は `python tests/tools/gen_golden_acoustic.py` で作り直します。
`golden_postprocess.json` は `python tests/tools/gen_golden_postprocess.py` で作り直します (入力は乱数の音響特徴)。
テスト用の `.hed` とラベルは NNSVS のもの (MIT) です。
