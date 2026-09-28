# OnnxSvs

`onnx_export` で書き出した ONNX モデルを、ONNX Runtime (C#) で動かすための推論ライブラリ。
Python を使わずに、HTS ラベルから音声を作るところまでを目指しています (作業中)。

| 段 | 状態 |
|---|---|
| ラベル (`HtsLabel`)、質問ファイル (`QuestionSet`)、言語特徴量 (`LinguisticFeatures`)、scaler (`Scaler`) | 実装済み。nnmnkwii の出力と一致することをテスト済み |
| timelag / duration / acoustic の ONNX 推論、音素長の決定、MLPG、GV | 未実装 |
| WORLD による合成 | 未実装 |

## テスト

```
dotnet test
```

`tests/OnnxSvs.Tests/data/golden_frontend.json` は、Python の nnmnkwii の出力 (正解) です。
`.hed` やラベルを変えたときは `python tests/tools/gen_golden.py` で作り直します (nnsvs と nnmnkwii が必要)。
テスト用の `.hed` とラベルは NNSVS のもの (MIT) です。
