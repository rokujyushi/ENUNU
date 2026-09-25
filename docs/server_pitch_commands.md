# ENUNUServer 追加コマンド: `pitch` / `acoustic_f0`

リクエストの形式は既存コマンドと同じ JSON 配列です。

```
[step, ust_path, wav_path, singer_name(hash), duration]
```

どちらのコマンドも `<ust_path の stem>_enutemp/` をワークフォルダとして使います（`acoustic` と同じ）。

## 後方互換について

旧クライアント（SimpleENUNUServer / ENUNUServer / 韓国語版 ENUNUServer 向けの OpenUtau）を壊さないよう、
以下の追加はすべて「任意のリクエスト要素」と「レスポンスへの項目追加」だけで行っています。
既存の項目名・意味・`ver_check` の `name` / `version` は変えていません。

### `ver_check` の `features`（追加項目）

```json
{"result": {"name": "SimpleENUNUServer", "version": "1.0.0", "author": "roku10shi",
            "features": {"commands": ["timing", "acoustic", "pitch", "acoustic_f0", "synthe", "config"],
                         "style_shift": true, "pitch_n_frames": true,
                         "diffusion": {"mgc": {"method": "ddim", "steps": 25},
                                       "mel": {"method": "ddim", "steps": 25},
                                       "bap": {"method": "plms", "steps": 20},
                                       "other": {"method": "ddpm", "steps": 100}}}}}
```

- `features` が無いサーバーは旧版とみなし、`pitch` / `acoustic_f0` を使わないでください。
- `lf0_conditioning` はモデルを読むまで分からないので、ここではなく `pitch` / `acoustic_f0` のレスポンスで返します。

### 拡散モデルのステップ数と `config` コマンド

拡散モデル（DiffSinger 系の mgc / mel / bap）はサンプリングを間引いて高速化しています。
既定値は mgc / mel が DDIM 25 ステップ、bap が PLMS 20 ステップです。
（mgc / mel は 25 ステップの DDIM で 100 ステップとほぼ同じ品質。bap は PLMS 10 ステップだとなめらかになりすぎるため 20 ステップ。2026-09-25 に数値比較と試聴で決定）

`method` は `ddpm`（間引きなし）/ `ddim` / `plms` / `eta1` のどれかです。
設定の優先順位は、`config` コマンド → 環境変数 → 既定値 です。

- 環境変数 `ENUNU_DIFFUSION_MGC` / `_MEL` / `_BAP`: `ddim:25` のように「手法:ステップ数」で指定します。数字だけならステップ数だけを変えます。
- 旧環境変数 `ENUNU_DIFFUSION_SPEEDUP` / `TARGETS` / `METHOD` を指定している場合は、従来の意味で解釈します
  （対象外のストリームは間引きなし、`SPEEDUP=1` はすべて間引きなし）。

`config` コマンドでは、実行中に全エンジンの設定を変えられます（OpenUtau の環境設定から送る想定）。
`ver_check` の後に送ってください。歌手を指定する必要はありません。

```json
["config", {"diffusion": {"steps": 25}}]
["config", {"diffusion": {"mgc": {"method": "ddim", "steps": 25}, "bap": "plms:20"}}]
["config", {"diffusion": {"reset": true}}]
```

- `steps` だけを送ると、mgc と mel のステップ数だけが変わります。
- レスポンスは `{"result": {"diffusion": <反映後の設定>}}`、不正な値のときは `{"error": "..."}` です。
- 設定を変えると、`acoustic` は以前の `features.npz` を使わずに計算し直します。
  クライアント側で npy をキャッシュしている場合は、ステップ数をキャッシュのキーに含めてください。

### 拡張機能の実行

Python の拡張機能（`.py`）は、プロセス起動のコストを省くため、サーバーと同じプロセス内で実行します
（1回あたり 0.2〜0.3 秒の短縮）。引数・カレントディレクトリ・`sys.path[0]` は subprocess と同じにしています。
`ENUNU_EXTENSION_INPROCESS=0` にすると、従来どおり subprocess で実行します。

### `style_shift`（任意）

- `timing` 以外のコマンド（`acoustic` / `pitch` / `synthe`）は `request[5]`、`acoustic_f0` は `request[6]` に
  整数（半音）を置くとスタイルシフトします。省略・数値以外は 0（従来どおり）。
- ピッチは変えずに声色だけを変えます（USTフラグ `S5` などで使う拡張機能 style_shifter と同じ考え方）。
  拡張機能と併用すると二重にかかるので、どちらか一方にしてください。

### 音響特徴量のファイルキャッシュ（`features.npz`）と `editorf0.npy`

ニューラルボコーダ向けに、音響特徴量はワークフォルダの `features.npz` にそのまま保存します
（WORLD: mgc / lf0 / vuv / bap、melf0: mel / lf0 / vuv。作った条件と UST のハッシュも一緒に保存）。
メモリには最後の1フレーズ分しか持たないので、フレーズが増えてもメモリは増えません。
サーバーを再起動しても、ワークフォルダが残っていれば推論なしで合成できます。

- `acoustic`: `features.npz` が同じ UST・`acoustic`・同じ style_shift のものなら推論を省略します。
- `acoustic_f0`: `features.npz` が同じ UST・同じ style_shift・同じエディタのピッチ（配列のハッシュ）の `acoustic_f0` のものなら推論を省略し、それ以外は計算して保存します。
- `pitch`: `pitch_f0.npy` の横に条件を `pitch_f0.json` として保存し、同じ UST・同じ style_shift なら推論を省略します。
- どのキャッシュも、拡散の設定（`config` / 環境変数）が変わったら使いません。
- `synthe`: `features.npz` があれば推論せずに合成します。無ければ `acoustic` を実行してから合成します。
  - **ピッチだけ変えた場合（旧クライアント互換）**: ワークフォルダに `editorf0.npy`（float64、Hz、`(T,)`）を置いて
    `synthe` を呼ぶと、キャッシュ済みの特徴量の lf0 をそのピッチに差し替えて合成します。0 のフレームはモデルのピッチのままです。
  - `features.npz` が無い旧版のワークフォルダでも、melf0 モデルなら `mel.npy` / `vuv.npy` / `f0.npy` から復元します。
  - `synthe` の style_shift は、キャッシュが無く作り直すときだけ使います。

## `pitch` — ピッチ（F0）だけを推定する

```json
["pitch", "<cache>/enu-xxxx.tmp", "", "<voicebankNameHash>", "600"]
```

レスポンス:

```json
{"result": {"path_f0": "<enutemp>/pitch_f0.npy", "lf0_conditioning": true, "n_frames": 1234}}
```

- `n_frames`: `pitch_f0.npy` のフレーム数。`acoustic_f0` に送る配列の長さはこれに合わせます（ファイルを読む必要がありません）。

- `pitch_f0.npy`: float64、形状 `(T,)`、単位は Hz、フレーム周期は `frame_period`（通常 5 ms）。休符フレームは 0 です。
  フレーム数は同じ UST に対する `acoustic` の `f0.npy` と一致します。
- `lf0_conditioning: true` のモデル（音響モデルがピッチ専用のサブモデル `lf0_model` を持つ場合）は
  `lf0_model` だけを実行します。拡散モデルなどは回しません。
  `false` のモデル（旧来の一括予測モデル）は音響モデル全体を実行して F0 だけを返します。
- `f0.npy` とは別のファイルに書き出すので、`acoustic` のキャッシュ判定には影響しません。

## `acoustic_f0` — エディタのピッチを条件にして音響特徴量を作り直す

エディタの F0 配列をリクエストに直接埋め込みます（ファイル経由不要）。

```json
["acoustic_f0", "<cache>/enu-xxxx.tmp", "", "<voicebankNameHash>", "600", [0.0, 220.5, 220.5, ...]]
```

- `request[5]`: float64 配列 (Hz)、形状 `(T,)`。0 のフレームはモデル自身のピッチを使います。
  フレーム数は `pitch` コマンドの `pitch_f0.npy` の長さと一致させてください。

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

1. `pitch` → `pitch_f0.npy`。描画用ピッチ（LoadRenderedPitch）に使います。
2. フレーム数 = レスポンスの `n_frames` として editorF0 配列を作ります。
3. `["acoustic_f0", ..., editorF0.ToList()]` — f0 配列をリクエスト `[5]` に直接埋め込んで送信します。
   → f0 / sp / ap（WORLD）または mel / vuv（melf0）を受け取ります。
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
