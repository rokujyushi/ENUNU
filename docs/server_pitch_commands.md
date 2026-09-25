# ENUNUServer 追加コマンド: `pitch` / `acoustic_f0`

リクエストの形式は既存コマンドと同じ JSON 配列です。

```
[step, ust_path, wav_path, singer_name(hash), duration]
```

どちらのコマンドも `<ust_path の stem>_enutemp/` をワークフォルダとして使います（`acoustic` と同じ）。

## `pitch` — ピッチ（F0）だけを推定する

```json
["pitch", "<cache>/enu-xxxx.tmp", "", "<voicebankNameHash>", "600"]
```

レスポンス:

```json
{"result": {"path_f0": "<enutemp>/pitch_f0.npy", "lf0_conditioning": true}}
```

- `pitch_f0.npy`: float64、形状 `(T,)`、単位は Hz、フレーム周期は `frame_period`（通常 5 ms）。休符フレームは 0 です。
  フレーム数は同じ UST に対する `acoustic` の `f0.npy` と一致します。
- `lf0_conditioning: true` のモデル（音響モデルがピッチ専用のサブモデル `lf0_model` を持つ場合）は
  `lf0_model` だけを実行します。拡散モデルなどは回しません。
  `false` のモデル（旧来の一括予測モデル）は音響モデル全体を実行して F0 だけを返します。
- `f0.npy` とは別のファイルに書き出すので、`acoustic` のキャッシュ判定には影響しません。

## `acoustic_f0` — エディタのピッチを条件にして音響特徴量を作り直す

先にクライアントが `<enutemp>/editorf0.npy` を書き込んでおきます。
（float64、形状 `(T,)`、単位は Hz。0 のフレームはモデル自身のピッチを使います。）

```json
["acoustic_f0", "<cache>/enu-xxxx.tmp", "", "<voicebankNameHash>", "600"]
```

レスポンス: `acoustic` と同じ項目に `lf0_conditioning` が加わります。

```json
{"result": {"path_f0": "...", "path_spectrogram": "...", "path_aperiodicity": "...",
            "path_mel": "...", "path_vuv": "...", "lf0_conditioning": true}}
```

- `lf0_conditioning: true` のモデルは、`lf0_model` の出力を editorf0 に置き換えます。
  そのうえで mgc / bap / mel / vuv を生成し直すので、声色や有声/無声の判定がエディタのピッチに合います。
- `false` のモデルでは editorf0 は無視され、`acoustic` と同じ結果になります（警告ログを出します）。
- `f0.npy` の中身は editorf0 に沿ったピッチになります。モデルが予測したピッチを表示したい場合は、`pitch` の `pitch_f0.npy` を使ってください。

## 想定しているクライアントの流れ（EnunuRenderer）

1. `pitch` → `pitch_f0.npy`。フレーム数を得るのと、描画用ピッチ（LoadRenderedPitch）に使います。
2. フレーム数 = `len(pitch_f0)` として editorF0 を作り、`editorf0.npy` に保存します。
3. `acoustic_f0` → f0 / sp / ap（WORLD）または mel / vuv（melf0）を受け取ります。
4. これまでどおり WORLD 合成、または `synthe` を呼びます。

## 動作確認（2026-09-25、RTX 5060 Ti、8 秒のフレーズ）

| 音源 | 構造 | pitch | acoustic | pitch と acoustic の F0 差（平均） |
|---|---|---|---|---|
| ENUNU_KanadeShia_20251101 | world 拡散 + HN-uSFGAN | 1.42 s | 5.26 s | 0.63 cent |
| ENUNU_欲音ルコ♀_melf0-diffusion_V100 | melf0 拡散 + SiFi-GAN | 0.37 s | 2.52 s | 0.10 cent |
| ENUNU_夏目悠李_RMDN_v0.0.3 | 旧形式 RMDN（フォールバック） | 0.54 s | 1.33 s | 0.00 cent |

- 時間はリクエスト全体の時間で、UST 変換と拡張機能の実行を含みます。
- KanadeShia の pitch は、プロセス内で最初の推論のため CUDA の初期化分が含まれています。

`acoustic_f0` の効果（KanadeShia、乱数シード固定、有声フレームの平均 |log sp 差|）:

| 条件 | acoustic との差 |
|---|---|
| acoustic をもう一度実行 | 0.000 |
| editorf0 = モデルのピッチ（±0 cent） | 0.026 |
| editorf0 = +200 cent | 0.868 |
| editorf0 = −300 cent | 0.943 |
