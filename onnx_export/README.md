# onnx_export

NNSVS / ENUNU のモデルを ONNX に書き出します。推論側 (C# + ONNX Runtime) は別途。

```
pip install onnx onnxruntime
python -m onnx_export.export MODEL_DIR OUT_DIR
```

`MODEL_DIR` はパック済みのモデルフォルダ (`config.yaml`、`timelag_model.yaml`/`.pth`、scaler の `.npy` など)。
旧形式 (`enuconfig.yaml`) のフォルダは、ENUNU が読み込むときに行う変換でこの形にしてから渡します。

`OUT_DIR` に `timelag.onnx` / `duration.onnx` / `acoustic.onnx` と `manifest.json` ができます。
`manifest.json` には、段ごとのモデルのクラス・入出力名・入力次元・scaler の値、書き出せなかった段の理由が入ります。

## 書き出しの約束

- 入力 `x` は `(バッチ, フレーム, 次元)`。フレーム数は可変。バッチ 1・全フレーム有効の推論として書き出します。
- 出力は、MDN のモデルなら `mu`, `sigma` (最も重みの大きい成分)、そうでなければ `y`。
  どちらも `model.inference(x, lengths)` の値と一致します (`tests/test_onnx_export.py` で確認)。

## 対応状況

| 種類 | 状態 |
|---|---|
| MDN, MDNv2, Conv1dResnet(MDN), VariancePredictor, FFN | 対応 |
| RMDN, LSTMRNN, FFConvLSTM, ResSkipF0FFConvLSTM, ResF0Conv1dResnet | 対応 |
| NPSS 系 (自己回帰デコーダ)、GaussianDiffusion、ボコーダ、ポストフィルタ | 未対応 (次以降) |

未対応のモデルは `manifest.json` に理由が残り、ほかの段の書き出しは続きます。
