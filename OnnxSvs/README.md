# OnnxSvs

`onnx_export` で書き出した ONNX モデルを、ONNX Runtime (C#) で動かすための推論ライブラリ。
Python を使わずに、HTS ラベルから音声を作るところまでを目指しています (作業中)。

| 段 | 状態 |
|---|---|
| ラベル (`HtsLabel`)、質問ファイル (`QuestionSet`)、言語特徴量 (`LinguisticFeatures`)、scaler (`Scaler`) | 実装済み。nnmnkwii の出力と一致することをテスト済み |
| ONNX 推論 (`ModelStage`)、timelag / duration の予測と音素長の決定 (`TimingPredictor`) | 実装済み。nnsvs の出力と一致することをテスト済み。動的特徴量 (MLPG) を使うモデルは未対応 |
| acoustic の ONNX 推論、MLPG、GV | 未実装 |
| WORLD による合成 | 未実装 |

## テスト

```
dotnet test
```

`tests/OnnxSvs.Tests/data/golden_frontend.json` は、Python の nnmnkwii の出力 (正解) です。
`.hed` やラベルを変えたときは `python tests/tools/gen_golden.py` で作り直します (nnsvs と nnmnkwii が必要)。
`golden_timing.json` と `timing_models/` (ランダムな重みの小さなモデルを onnx_export で書き出したもの) は
`python tests/tools/gen_golden_timing.py` で作り直します。
テスト用の `.hed` とラベルは NNSVS のもの (MIT) です。
