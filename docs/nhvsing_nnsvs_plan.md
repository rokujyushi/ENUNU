# NHVSing を nnsvs の melf0 音源で使う計画

作成: 2026-09-28（ENUNUServer のセッションで調査・合意した内容）
フォーク: https://github.com/rokujyushi/NHVSing （元: https://github.com/wavtechyukky/NHVSing 、MIT）

## 目的

ENUNU の melf0 音源（音響モデルが mel + lf0 + vuv を出す）で、ボコーダーを SiFiGAN から NHVSing に差し替えられるようにする。
NHVSing は CPU で速い（元の README では ONNX で RTF 約 0.07、NSF-HiFiGAN の約 8 倍）ので、GPU の無い環境での合成時間を縮めるのが狙い。

作業は 2 つのリポジトリに分かれる。

| リポジトリ | やること |
|---|---|
| rokujyushi/NHVSing（このフォーク） | pip で入れられるパッケージにする / nnsvs の mel で学習する / nnsvs の形式で書き出す |
| ENUNUServer | 書き出したモデルを読んで合成する（`enuserver/` に読み込み処理を追加） |

## 決まっていること

- **ONNX ではなく PyTorch のチェックポイントで受け渡す。** ONNX で動くなら OpenUtau 側で直接動かせてしまい、サーバーで対応する意味が薄いため。
- **mel を出す音源だけが対象。** WORLD（mgc/bap）の音源は対象外。
- **Wavehax と同じ形にする。** Wavehax は taroushirani/wavehax のフォークを `pip install --no-deps git+...` で入れ、
  `vocoder_model.yaml` の `generator._target_` でクラスを決めて `state_dict` を読んでいる（`ENUNUServer-1.0.0/enuserver/wavehax.py`）。
  NHVSing もこれに揃える。
- **mel の形式はフォーク側で nnsvs に合わせる。** サーバー側では mel を変換しない。
  配布済みの NHVSing の重み（128 次元 mel・44.1kHz 用、しかも非商用）は使えないので、一から学習する。

## 合わせる mel の形式

| | nnsvs melf0（48kHz、今の音源） | NHVSing V3 の元の設定 |
|---|---|---|
| サンプリング周波数 | 48000 | 44100 |
| hop | 240（5ms） | 256 |
| fft / 窓長 | 1024 / 960 | 2048 / 2048 |
| mel 次元 | 80 | 128 |
| fmin / fmax | 30 / 24000（nnsvs の既定値。レシピで上書きしていないか要確認） | 40 / 16000 |
| 対数 | log10、下限 1e-10 | 自然対数、下限 1e-5 |
| STFT の端 | librosa.stft の center=True、reflect | center=False、(fft-hop)/2 の reflect pad |

nnsvs 側の計算は `parallel_wavegan.bin.preprocess.logmelfilterbank`（`nnsvs/data/data_source.py` の MelF0 から呼ばれる）。
設定は `nnsvs/bin/conf/prepare_features/acoustic/melf0_48k.yaml`。手元の melf0 音源 4 つ（欲音ルコ melf0、波音リツ クリスクロス、
KanadeShia MEL、雨星サイファ）はすべて `stream_sizes: [80, 1, 1]`、48kHz、5ms。

## サーバーとの受け渡し形式

音源フォルダに nnsvs の packed model と同じ名前で置く。

- `vocoder_model.pth`: `{'model': {'generator': state_dict}}`。weight norm は外した状態で保存する
  （サーバーで `remove_weight_norm` を呼ばずに済むように。yaml 側は `use_weight_norm: false`）。
- `vocoder_model.yaml`:

```yaml
generator:
  _target_: nhvsing.model.NHVSingV3   # パッケージ化後の実際のパスに合わせる
  vocoder_cfg: {...}                   # NHVSingV3(vocoder_cfg, ltv_filter_cfg) の引数そのまま
  ltv_filter_cfg: {...}
data:
  feat_names: [mel]
  sample_rate: 48000
  hop_size: 240
  mel:                                 # サーバーが音響モデルと照らし合わせるための仕様
    num_mels: 80
    fft_size: 1024
    win_length: 960
    fmin: 30
    fmax: 24000
    log_base: 10
```

サーバーでの推論時の入力（案。読み込み処理を作るときに確定させる）:
- `x`: 音響モデルが出した mel（正規化を外した log10 の値）`(1, T, 80)`。`in_vocoder_scaler` は使わない。
- `cf0`: `exp(lf0)` の連続 F0 `(1, 1, T)`。
- `uv`: `vuv < 閾値` を 1 とした `(1, 1, T)`。

学習でこれと同じ前処理をしていれば、サーバーでの変換は要らない。

## 作業手順（フォーク側）

### 1. pip で入れられるパッケージにする（最優先・小さく）

今はフォルダ直下前提の import（`from model import ...`、`from onnx_model import ...` など）なので、サーバーから import できない。

- `nhvsing/` パッケージにまとめて相対 import にする。学習や前処理のスクリプトはこれまで通り動くようにする。
- `pyproject.toml` を置く。推論に要らない依存（tensorboard、onnx、onnxruntime、RMVPE など）は必須にしない。
- 推論で import する範囲（model / onnx_model / layers / dsp）が torch・numpy・librosa だけで動くこと。

**完了の目安:** ENUNUServer 同梱の Python 3.13 で
`python-3.13.15-embed-amd64\python.exe -m pip install --no-deps git+https://github.com/rokujyushi/NHVSing@<ブランチ>`
が通り、`from nhvsing.model import NHVSingV3` ができる。
（組み込み Python は setup.py から自パッケージを import するとビルドに失敗する。Wavehax で踏んだので、バージョンは直接書く）

ここまでできたら、ENUNUServer 側の読み込み処理を並行して作り始められる（乱数のモデルでテストできるため）。

### 2. 48kHz・hop 240 で動くか確かめる

元の設定は hop 256 のみ。乱数初期化の `NHVSingV3` を `sample_rate: 48000, hop_size: 240, in_channels: 80` で作り、
出力長が `T * 240` になること、LTV FIR（窓 2×hop）や `fft_size` との組み合わせで破綻しないことを確認する。

### 3. 学習データを nnsvs の mel で作る

mel の計算は次の 3 か所にあり、全部を nnsvs の形式に揃える必要がある。

| 場所 | 役割 |
|---|---|
| `preprocess.py` の `make_mel_fn` | 学習データの入力 mel |
| `dsp.py` の `wav_to_mel_torch` | pitch augmentation で mel を作り直す（`train_v3.py` の `pitch_augment_batch`） |
| `dataset.py` の振幅 augmentation | 振幅を alpha 倍したときの mel の補正（ln なら `+ln(alpha)`、log10 なら `+log10(alpha)`） |

注意点:
- 損失用の mel（`mel_loss`）は生成音声と正解音声の両方から計算するので、入力 mel と同じ形式でなくてもよい。
  ただし入力 mel を損失の正解に流用している箇所が無いか確認する。
- 前処理が学習用の波形を RMS 正規化（`target_rms: 0.111`）している。mel をその正規化後の波形から計算しないと、
  mel と波形の音量が合わなくなる。nnsvs の音響モデルは正規化していない音量で mel を出すので、
  正規化するかどうか自体を見直す。
- F0 は元は RMVPE で取っている。推論時の F0 は nnsvs の音響モデルが出す lf0/vuv なので、
  nnsvs のレシピで作った特徴量（mel・lf0・vuv）と波形をそのまま使うのが一番ずれが少ない。
  nnsvs のレシピの dump に正規化前の特徴量があるはずなので、そこから npz を作る変換スクリプトを書く案を先に検討する。

### 4. 学習

配布済みの重みは入力次元もサンプリング周波数も違うので、一から学習する。
`config_v3_2.yaml` を元に 48kHz・hop 240・mel 80 の設定を作る。
単一話者の音源ごとに学習するか、複数話者でまとめて学習するかは、データ量を見て決める。

### 5. nnsvs の形式で書き出す

学習のチェックポイントから `vocoder_model.pth` と `vocoder_model.yaml`（上の形式）を作るスクリプトを用意する。
weight norm を外すこと、`data.mel` に学習時の mel 仕様をそのまま書くこと。

## 作業手順（ENUNUServer 側。手順 1 の後）

- `enuserver/nhvsing.py` を作る。`enuserver/wavehax.py` と同じく `nnsvs.util.load_vocoder` を包み、
  `generator._target_` が `nhvsing.` で始まるときだけ読む。
- 読み込み時に、音響モデルの `feature_type` が `melf0` であることと、`data.mel` が音響モデルの mel 仕様と一致することを確かめ、
  合わなければ分かりやすいエラーにする。
- 合成は `enunu.py` の `ENUNU.predict_waveform` に分岐を足す（Wavehax と同じ場所）。
- NHVSing は雑音を `torch.normal` で作るので、同じ入力で同じ波形になるか（サーバーのキャッシュと決定性の扱い）を確認する。
- 乱数のモデルでテストを `tests/test_unit.py` に足し、`docs/server_pitch_commands.md` に説明を書く。
- 依存は `requirements.txt` にコメントで書く（Wavehax と同じく `--no-deps` で入れる）。

## 未確認のこと

- hop 240・48kHz で NHVSing の DSP 部分がそのまま動くか（手順 2）。
- nnsvs の melf0 レシピで fmin/fmax を上書きしていないか。
- nnsvs のレシピの dump にある特徴量ファイルの名前と形（手順 3 の変換スクリプトの前提）。
- 学習に必要なデータ量と時間。

## 参考

- NHVSing: https://github.com/wavtechyukky/NHVSing
- NHV の元論文: Liu et al., "Neural Homomorphic Vocoder", Interspeech 2020
- Wavehax の組み込み例: `ENUNUServer-1.0.0/enuserver/wavehax.py`、`docs/server_pitch_commands.md` の「Wavehax ボコーダー」
- nnsvs の mel 計算: `parallel_wavegan/bin/preprocess.py` の `logmelfilterbank`、`nnsvs/bin/conf/prepare_features/acoustic/melf0_48k.yaml`
