# ENUNUServer 追加コマンド: `pitch` / `acoustic_f0`

リクエストは JSON 配列です。ENUNUServer 2 では、`ver_check` と `config` 以外のすべてのコマンドで `request[5]`（style_shift）が必須です。

```
[step, ust_path, wav_path, singer_name(hash), duration, style_shift]
["acoustic_f0", ust_path, wav_path, singer_name(hash), duration, style_shift, editor_f0]
```

- `duration`: 最後に使ってから、歌手のモデルを保持する秒数（文字列、OpenUtau は `"600"`）。
- `style_shift`: 整数（半音）。使わない場合は `0` を送ります（[`style_shift`](#style_shift) を参照）。
- `pitch` / `acoustic_f0` は `<ust_path の stem>_enutemp/` をワークフォルダとして使います（`acoustic` と同じ）。

## クライアントとの組み合わせ

ENUNUServer 2 は、**最新の OpenUtau（`EnunuConnection` を持つクライアント）とセットで使います**。
旧クライアント（SimpleENUNUServer / ENUNUServer 1 向けの OpenUtau）は `request[5]` を送らないので、このサーバーではエラーになります。
旧クライアントを使う場合は、ENUNUServer 1 を使ってください。

| | ENUNUServer 1 | ENUNUServer 2（このサーバー） |
|---|---|---|
| `ver_check` の `version` | `1.0.0` | `2.0.0` |
| `request[5]` | 任意（style_shift） | 必須（style_shift） |
| `acoustic_f0` の f0 | `request[5]`（style_shift は `[6]`） | `request[6]`（style_shift は `[5]`） |
| `ver_check` より前のコマンド | `run ver_check.` を返す | 処理する |

最新の OpenUtau は、旧サーバー（SimpleENUNUServer 0.5 / ENUNUServer 0.6 / ENUNUServer 1）にもつながります。
旧サーバーは `request[5]` から後を読まないので、6 要素で送っても問題ありません。
韓国語 Phonemizer が 15555 番に送る `["timing", ust_path]` は、別のサーバー向けで、このサーバーには来ません。

### `ver_check` の `features`

```json
{"result": {"name": "SimpleENUNUServer", "version": "2.0.0", "author": "roku10shi",
            "features": {"commands": ["timing", "acoustic", "pitch", "acoustic_f0", "synthe", "config"],
                         "style_shift": true, "pitch_n_frames": true,
                         "diffusion": {"mgc": {"method": "ddim", "steps": 25},
                                       "mel": {"method": "ddim", "steps": 25},
                                       "bap": {"method": "plms", "steps": 20},
                                       "other": {"method": "ddpm", "steps": 100}}}}}
```

- クライアントは、使えるコマンドを `version` ではなく `features.commands` で判断してください。
  `features` が無いサーバーは旧版とみなし、`pitch` / `acoustic_f0` / `config` を使わないでください。
- `lf0_conditioning` はモデルを読むまで分からないので、ここではなく `pitch` / `acoustic_f0` のレスポンスで返します。

### `ver_check` より前のコマンド

旧サーバーは `ver_check` を受けるまで、ほかのコマンドに `{"error": "run ver_check."}` を返していました。
このため、サーバーを再起動すると、クライアントが `ver_check` を送り直すまで合成できませんでした。
このサーバーは `ver_check` より前でもすべてのコマンドを処理します。
クライアントは、旧サーバー（ENUNUServer 1 など）に対応するために、`run ver_check.` を受けたら `ver_check` を送って再送する処理を残してください。

### 拡散モデルのステップ数と `config` コマンド

拡散モデル（DiffSinger 系の mgc / mel / bap）はサンプリングを間引いて高速化しています。
既定値は mgc / mel が DDIM 25 ステップ、bap が PLMS 20 ステップです。
（mgc / mel は 25 ステップの DDIM で 100 ステップとほぼ同じ品質。bap は PLMS 10 ステップだとなめらかになりすぎるため 20 ステップ。2026-09-25 に数値比較と試聴で決定）

`method` は `ddpm`（間引きなし）/ `ddim` / `plms` / `eta1` のどれかです。
設定の優先順位は、`config` コマンド → 環境変数 → 既定値 です。

- 環境変数 `ENUNU_DIFFUSION_MGC` / `_MEL` / `_BAP`: `ddim:25` のように「手法:ステップ数」で指定します。数字だけならステップ数だけを変えます。
- 旧環境変数 `ENUNU_DIFFUSION_SPEEDUP` / `TARGETS` / `METHOD` を指定している場合は、従来の意味で解釈します
  （対象外のストリームは間引きなし、`SPEEDUP=1` はすべて間引きなし）。

`config` コマンドでは、実行中に全エンジンの設定を変えられます。
OpenUtau は環境設定を変えたときには送らず、合成のリクエストの前に、環境設定の値を読んで送ります。
`features.commands` に `config` があるサーバーにだけ送ってください。歌手を指定する必要はありません。

```json
["config", {"diffusion": {"steps": 25}}]
["config", {"diffusion": {"mgc": {"method": "ddim", "steps": 25}, "bap": "plms:20"}}]
["config", {"diffusion": {"reset": true}}]
["config", {"diffusion": {"reset": true, "bap": {"steps": 30}}}]
["config", {"diffusion": {"reset": true, "mgc": "ddpm", "mel": "ddpm", "bap": "ddpm"}}]
```

- `steps` だけを送ると、mgc と mel のステップ数だけが変わります。
- `reset` が無い指定は、今の設定に重ねます。`reset` だけを送ると、環境変数・既定値に戻します。
- `reset` とストリームの指定を一緒に送ると、環境変数・既定値から始めて、そのストリームだけを変えます。
  前に送った指定は残りません。不正な値が含まれていれば、何も変えずにエラーを返します。
- OpenUtau（環境設定の「ENUNU」）は、acoustic 系のリクエストの前に毎回 `reset` 付きで全体を送ります。
  - 「おすすめ（高速化）」: `reset` と、ステップ数を指定したストリームだけ（`{"steps": N}`、手法は既定のまま）。
  - 「モデルの設定に従う」: `reset` と、mgc / mel / bap をすべて `ddpm`（間引きなし）。nnsvs のモデルは
    `pndm_speedup` を持てないので、モデル本来の設定は常に間引きなし（`K_step` 回、通常 100 回）です。
  - 同じ値を何度送っても、キャッシュは無効になりません（キャッシュの判定には設定の値が入っているため）。
- レスポンスは `{"result": {"diffusion": <反映後の設定>}}`、不正な値のときは `{"error": "..."}` です。
- 設定を変えると、`acoustic` は以前の `features.npz` を使わずに計算し直します。
  クライアント側で npy をキャッシュしている場合は、ステップ数をキャッシュのキーに含めてください。

### CUDA Graphs

GPU で動かす場合、拡散モデルのデノイザーを CUDA Graphs で実行します。デノイザーは1回の推論で数十回呼ばれ、
1回ごとの計算が小さいため、Python とカーネル起動のオーバーヘッドが律速になっていたためです。
- フレーズの長さごとに初回だけグラフを記録し、以降は再生します。結果はグラフなしと完全に一致します。
- 拡散の時間は、3秒のフレーズで約 1/3、8秒のフレーズで約 1/2 になります（長さが変わってグラフを記録し直す初回でも速くなります）。
- グラフはモデルごとに最大4つまで保持します。VRAM は、長さの違うフレーズが続くと最大で約 300 MiB 増えます。
- `ENUNU_CUDA_GRAPHS=0` で無効にできます。記録に失敗した長さは、自動で通常の実行に戻します。

### nnsvs / nnmnkwii / pysptk のオーバーライド（`enuserver/nnsvs_speedups.py`）

pip で入れたライブラリは書き換えず、import 時に関数を差し替えています。LSTM の差し替え以外は、結果が元の実装と完全に一致します。
- **質問照合のキャッシュ**（nnmnkwii）: 音素ラベルごとに数千個の正規表現を Python で検索していたので、結果を
  （質問セット, ラベル）ごとに覚えます。1音符を直すと変わるのは前後の数音素だけなので、2回目以降はほとんど検索しません。
- **`linguistic_features` の使い回し**: 1回の acoustic で同じラベルに対して3回呼ばれていたので、同じ入力なら結果を使い回します。
- **`mcepalpha` のキャッシュ**（pysptk）: サンプリングレートだけで決まる定数を、毎回数値探索していました。
- **`mc2sp` のベクトル化**（pysptk）: `use_world_codec=False` の旧来の WORLD モデル（Conv1dResnet / RMDN / NPSS など）は、
  WORLD パラメータを作るときにメルケプストラムをスペクトルに戻します。元の実装は1フレームずつ `freqt` を呼び、Python のループで
  対称な配列を組み立てていました（6.5秒のフレーズで 1.1〜1.3 秒）。`freqt` を行列積に、残りを配列演算にして約15倍速です
  （結果の相対差は 1e-14 程度）。OpenUtau は WORLD モデルでは必ずこの npy を読むので、旧来の音源すべてに効きます。
- **自己回帰デコーダーの CUDA Graphs 化**（lf0 の `ResF0NonAttentiveDecoder`、NPSS 系の mgc / bap の `NonAttentiveDecoder`）:
  1ステップの形はフレーズの長さに関係なく同じなので、モデルごとに1回だけ記録して全ステップで再生します。結果は一致します。
  prenet が無いデコーダーは前の出力に直接 dropout をかけますが、dropout はテンソルのメモリ配置（stride）で乱数の割り当てが変わるので、
  この部分だけグラフの外で元と同じ形のビューに対して実行しています。`pitch` が約2倍速、Yeonu（NPSS）の音響モデルが約2.4倍速になります。
- **系列が1本だけの LSTM をパックせずに実行**（`torch.nn.LSTM`）: nnsvs の vuv モデル（FFConvLSTM）と lf0 モデルのエンコーダーは
  可変長のために `pack_padded_sequence` を使いますが、サーバーは常にバッチサイズ 1 です。cuDNN のパック入力の経路は非常に遅く
  （RTX 5060 Ti / cuDNN 9.10 で、600 フレームの 2 層 BiLSTM が 98 ms、通常のテンソルなら 1.3 ms）、残っていた時間の大半を占めていました。
  カーネルが変わるので出力は 1e-6 程度ずれ、拡散を通った後で mgc/bap/mel に最大 0.002〜0.02 程度の差が出ます
  （シードを変えたときのばらつきより桁違いに小さい）。何度実行しても同じ結果になる点は変わりません。

`ENUNU_NNSVS_SPEEDUPS=0` で CUDA Graphs 以外を、`ENUNU_CUDA_GRAPHS=0` で CUDA Graphs（拡散と自己回帰デコーダー）を無効にできます。

6.5秒のフレーズでの比較（RTX 5060 Ti。「元」は両方を無効にした状態）:

| | 元 | 最適化あり |
|---|---|---|
| KanadeShia acoustic / pitch | 2.50 s / 0.67 s | 0.61 s / 0.18 s |
| 欲音ルコ melf0 acoustic / pitch | 1.69 s / 0.75 s | 0.33 s / 0.11 s |

モデルの種類ごとの比較（6.5秒のフレーズ、2026-09-25。「元」は最適化をすべて無効にした状態）:

| 音響モデル | 代表の音源 | acoustic 元 → 現在 | synthe 元 → 現在 | 結果の差 |
|---|---|---|---|---|
| Conv1dResnetMDN（旧形式で最多） | KanadeShia_30kx4 | 1.69 → 0.30 s | 1.76 → 0.47 s | 特徴量は一致 |
| Conv1dResnet | Haruqa | 1.30 → 0.18 s | 1.43 → 0.42 s | 特徴量は一致 |
| FFConvLSTM | A.I.CHI | 1.51 → 0.37 s | 1.51 → 0.42 s | 特徴量は一致 |
| ResSkipF0FFConvLSTM | 夏目悠李 | 1.51 → 0.33 s | 1.50 → 0.42 s | 特徴量は一致 |
| RMDN | 夏目悠李 v0.0.3 | 1.63 → 0.46 s | 1.54 → 0.41 s | 特徴量は一致 |
| NPSS（拡散なし、全ストリーム自己回帰） | Yeonu | 3.32 → 0.95 s | 1.50 → 0.43 s | 対数スペクトル差 0.07 dB |
| NPSS + 拡散 + HN-uSFGAN | KanadeShia_20251101 | 2.67 → 0.49 s | 0.98 → 0.64 s | 対数スペクトル差 0.13 dB |
| melf0 + 拡散 + SiFi-GAN | 欲音ルコ melf0 | 1.12 → 0.28 s | 0.14 → 0.11 s | 対数スペクトル差 0.34 dB |

- 旧来の WORLD モデルは、`mc2sp` のベクトル化が効いています（WORLD パラメータ生成が 1.1〜1.4 秒 → 0.09 秒）。
  サーバーの WORLD ボコーダも同じ処理を使うので、`synthe` も速くなります。
- 差が出るのは LSTM の差し替えとボコーダの fp16 によるもので、聴いて分かる差の目安（約 1 dB）より十分小さい値です。

### ボコーダの fp16 実行

`synthe` のニューラルボコーダ（HN-uSFGAN など）は計算量が律速なので、GPU では fp16（autocast）で動かします。
- KanadeShia（HN-uSFGAN、6.5秒）で 1.05 s → 0.70 s。fp32 との SNR は 54〜60 dB で、fp32 自体の GPU の非決定性による差（約 57 dB）と同程度です。
- 決定的アルゴリズムと組み合わせても毎回同じ波形になり、後に動かす音響モデルへの影響もありません。
- 万一非有限値が出た場合は fp32 で合成し直します。`ENUNU_VOCODER_FP16=0` で無効にできます。
- TF32 と cudnn.benchmark も試しましたが、効果はありませんでした。

### Wavehax ボコーダー（`enuserver/wavehax.py`）

`vocoder_model.yaml` の `generator._target_` が `wavehax.` で始まる音源は Wavehax で合成します
（taroushirani/nnsvs の wavehax-support ブランチ相当。パッケージは taroushirani/wavehax@nnsvs）。
- 特徴量の前処理（bap の補正と正規化）は uSFGAN と同じなので、nnsvs の uSFGAN の経路をそのまま使います。
- F0 は学習設定 `data.use_continuous_f0` に合わせます。nnsvs のレシピの既定は `false`（無声区間の F0 を 0 にして学習）なので、
  無声区間（vuv < 閾値）を 0 にして渡します。wavehax-support ブランチは常に連続 F0 を渡すので、ここだけ違います。
- iSTFT の重ね合わせを加算に置き換えています。元の実装は単位行列カーネルの `conv_transpose1d` で、
  `synthe` の決定的アルゴリズムの下では GPU で非常に遅くなります（2.5 秒の音声で 2.83 s → 0.055 s、結果は同じ）。
- n_fft（48kHz で 960）が2のべき乗でなく GPU の半精度 FFT が使えないので、fp16 にはせず fp32 で動かします。
- MS-Wavehax（`wavehax.generators.MultiScaleWavehaxGenerator`、Interspeech 2025）も同じ経路で読めます。
  melf0 の音源では mel だけを渡します。48kHz・hop 240・mel 80 次元の乱数 generator で 10 秒を合成すると、
  GPU で 0.079 s（Wavehax 0.139 s）、CPU で 1.67 s（同 2.81 s）でした（決定的アルゴリズムの下でも遅くなりません）。

### NumPy 2 対応（`enuserver/nnsvs_compat.py`）

Python 3.13 では NumPy 2 が必須です。nnsvs の `lowpass_filter` は `scipy.signal.butter` に要素1個のリストを渡していて、
NumPy 2 + SciPy 1.18 では TypeError になる（trajectory_smoothing で必ず通る）ので、スカラーで渡す版に差し替えます。
NumPy 1.26 でも結果は同じです。
nnsvs を taroushirani/nnsvs（fm-support、`642fdbc`）に切り替えた後は nnsvs 側で直っているので、この差し替えは結果を変えません
（旧 oatsu-gh/nnsvs@enunu-python312 に戻したときのために残しています）。

### 拡張機能の実行

Python の拡張機能（`.py`）は、プロセス起動のコストを省くため、サーバーと同じプロセス内で実行します
（1回あたり 0.2〜0.3 秒の短縮）。引数とカレントディレクトリは subprocess と同じにしています。
さらに拡張機能のフォルダを `sys.path` に入れるので、同じフォルダのモジュールも import できます
（同梱の埋め込み Python は `._pth` があるため、subprocess 実行ではできません）。
`ENUNU_EXTENSION_INPROCESS=0` にすると、従来どおり subprocess で実行します。

音響特徴量の拡張機能（acoustic_editor）には、これまでどおり mgc / f0 / vuv / bap を CSV で渡します。
拡張機能が書き換えなかった CSV は読み直さず、元の配列をそのまま使います（変更の有無はファイルの中身のハッシュで判定します）。
そのため、変更していないストリームには CSV を経由した丸めが入りません。

### `style_shift`

- すべてのコマンドで `request[5]` に整数（半音）を置きます（`acoustic_f0` も `[5]` です）。数値以外は 0 として扱います。
  `timing` は値を使いませんが、形をそろえるために送ります。
- 0 以外にすると、フレーズ全体をスタイルシフトします。ピッチは変えずに声色だけを変えます
  （USTフラグ `S5` などで使う拡張機能 style_shifter と同じ考え方）。
  拡張機能と併用すると二重にかかるので、どちらか一方にしてください。
- OpenUtau はノートごとのスタイルシフトを UST の `S` フラグで送るので、`request[5]` はいつも 0 です。

### 音響特徴量のファイルキャッシュ（`features.npz`）と `editorf0.npy`

ニューラルボコーダ向けに、音響特徴量はワークフォルダの `features.npz` にそのまま保存します
（WORLD: mgc / lf0 / vuv / bap、melf0: mel / lf0 / vuv。作った条件と UST のハッシュも一緒に保存）。
メモリには最後の1フレーズ分しか持たないので、フレーズが増えてもメモリは増えません。
サーバーを再起動しても、ワークフォルダが残っていれば推論なしで合成できます。

- `acoustic`: `features.npz` が同じ UST・`acoustic`・同じ style_shift のものなら推論を省略します。
- `acoustic_f0`: `features.npz` が同じ UST・同じ style_shift・同じエディタのピッチ（配列のハッシュ）の `acoustic_f0` のものなら推論を省略し、それ以外は計算して保存します。
- `pitch`: `pitch_f0.npy` の横に条件を `pitch_f0.json` として保存し、同じ UST・同じ style_shift なら推論を省略します。
  lf0_model を持つモデルでは、lf0_model の生の出力も `pitch_lf0.npy` に保存し、同じ UST・style_shift の
  `acoustic` / `acoustic_f0` では lf0_model を実行せずにそれを使います（1回 0.2〜0.6 秒の短縮）。
- どのキャッシュも、拡散の設定（`config` / 環境変数）が変わったら使いません。

### 乱数のシード

lf0_model の dropout、拡散のノイズ、ボコーダのノイズは乱数です。シードを UST のハッシュと style_shift から
決めているので、同じフレーズは何度合成し直しても同じ特徴量・同じ波形になります。
- lf0_model の後でもシードを設定し直すので、「`pitch` → `acoustic`（lf0 を再利用）」と「`acoustic` 単体」の結果は一致します。
- `synthe` の合成中だけ `torch.use_deterministic_algorithms` を有効にしています。GPU の一部の演算が非決定的なためで、
  ボコーダが 0.01〜0.03 秒ほど遅くなります（サーバーは起動時に `CUBLAS_WORKSPACE_CONFIG=:4096:8` を設定します）。
- `synthe`: `features.npz` があれば推論せずに合成します。無ければ `acoustic` を実行してから合成します。
  - **ピッチだけ変えた場合**: ワークフォルダに `editorf0.npy`（float64、Hz、`(T,)`）を置いて
    `synthe` を呼ぶと、キャッシュ済みの特徴量の lf0 をそのピッチに差し替えて合成します。0 のフレームはモデルのピッチのままです。
    OpenUtau は `synthe` の前に毎回これを書きます（`lf0_conditioning: false` のモデルは、この方法でエディタのピッチを使います）。
  - `features.npz` が無い旧版のワークフォルダでも、melf0 モデルなら `mel.npy` / `vuv.npy` / `f0.npy` から復元します。
  - `synthe` の style_shift は、キャッシュが無く作り直すときだけ使います。
  - **tmp が無い場合**: OpenUtau の「選択ノートのキャッシュ削除」は `enu-*.tmp` だけを消して `_enutemp` を残すので、
    その後の `synthe` では tmp がありません。この場合は `<tmp名>_enutemp/temp.ust` を代わりに使い、
    `features.npz`（melf0 は旧形式の npy も可）から合成します。キャッシュも無い場合はエラーを返します。

## `pitch` — ピッチ（F0）だけを推定する

```json
["pitch", "<cache>/enu-xxxx.tmp", "", "<voicebankNameHash>", "600", 0]
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
["acoustic_f0", "<cache>/enu-xxxx.tmp", "", "<voicebankNameHash>", "600", 0, [0.0, 220.5, 220.5, ...]]
```

- `request[5]`: style_shift（ほかのコマンドと同じ）。
- `request[6]`: float64 配列 (Hz)、形状 `(T,)`。0 のフレームはモデル自身のピッチを使います。
  フレーム数は `pitch` コマンドの `pitch_f0.npy` の長さと一致させてください。

レスポンス: `acoustic` と同じ項目に `lf0_conditioning` が加わります。

```json
{"result": {"path_f0": "...", "path_spectrogram": "...", "path_aperiodicity": "...",
            "path_mel": "...", "path_vuv": "...", "lf0_conditioning": true}}
```

- `path_spectrogram` / `path_aperiodicity` のファイル（`acoustic` も同じ）は、OpenUtau が自分で WORLD 合成する音源だけに作ります。
  つまり `feature_type: world` で、`extensions.wav_synthesizer` に `synthe` が無い音源です。
  それ以外（melf0、`wav_synthesizer: synthe`）では OpenUtau は読まないので作りません。1 フレーズで約 13MB あり、キャッシュの大半を占めていたためです。
  判定は `EnunuRenderer` の `useSynthe` の逆（`ENUNU.client_reads_world_params`）なので、どちらかを変えるときは両方直してください。
- `lf0_conditioning: true` のモデルは、`lf0_model` の出力を editorf0 に置き換えます。
  そのうえで mgc / bap / mel / vuv を生成し直すので、声色や有声/無声の判定がエディタのピッチに合います。
- `false` のモデルでは editorf0 は無視され、`acoustic` と同じ結果になります（警告ログを出します）。
- `f0.npy` の中身は editorf0 に沿ったピッチになります。モデルが予測したピッチを表示したい場合は、`pitch` の `pitch_f0.npy` を使ってください。

## 想定しているクライアントの流れ（EnunuRenderer）

1. `pitch` → `pitch_f0.npy`。描画用ピッチ（LoadRenderedPitch）に使います。
2. フレーム数 = レスポンスの `n_frames` として editorF0 配列を作ります。
3. `["acoustic_f0", ust, "", hash, "600", 0, editorF0]` — f0 配列をリクエスト `[6]` に直接埋め込んで送信します。
   → f0 / sp / ap（WORLD）または mel / vuv（melf0）を受け取ります。
   `lf0_conditioning: false` のモデルは、代わりに `acoustic` を送ります（`acoustic_f0` ではピッチを変えるたびにキャッシュが外れるため）。
4. これまでどおり WORLD 合成、または `synthe` を呼びます（`synthe` の前に `editorf0.npy` を書きます）。

クライアント側の設計は [openutau_client_design.md](openutau_client_design.md) を見てください。

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
