# NHVSing 引き継ぎ：フォーク側の作業結果と ENUNUServer 側の残り

作成: 2026-09-29（NHVSing フォークのセッションから引き継ぎ）
前提の計画: [nhvsing_nnsvs_plan.md](nhvsing_nnsvs_plan.md)

計画の「作業手順（フォーク側）」の 1〜5 は終わった。残りは「作業手順（ENUNUServer 側）」だけ。
この資料は、そのために ENUNUServer 側で知っておくことをまとめたもの。

## 1. 今どうなっているか

| 項目 | 状態 |
|---|---|
| 手順 1: pip で入るパッケージ | 済。`nhvsing` パッケージ。同梱の Python 3.13 で `pip install --no-deps git+...@package` と `from nhvsing.model import NHVSingV3` を確認済み |
| 手順 2: 48kHz・hop 240 | 済。`test_hop240.py`（23 件）が通る |
| 手順 3: nnsvs の mel で学習データ | 済。`preprocess_nnsvs.py`、`config_nnsvs_48k.yaml` |
| 手順 4: 学習 | 済。本学習 350 エポック + 仕上げ（学習率 1/5）50 エポック = 400 エポック |
| 手順 5: nnsvs 形式での書き出し | 済。**`I:\NHVSing-1\exported_models\nnsvs48k\`**（400 エポック版。git には入れていない。下の 3 章） |
| ENUNUServer 側 | 6 章の 1〜6 は済（8 章）。7 の耳での聞き比べと、実際の音源フォルダへの差し替えが残り |

## 2. フォーク側の成果物

リポジトリ: https://github.com/rokujyushi/NHVSing 、**ブランチ `package`**（main にはまだ入れていない）。
手元の作業コピー: `I:\NHVSing-1`

| ファイル | 役割 |
|---|---|
| `nhvsing/`（model, onnx_model, layers, dsp, nhv_vocoder） | 推論に要るものだけのパッケージ。依存は torch / numpy / librosa / PyYAML |
| `pyproject.toml` | バージョン 0.1.0 を直接書いている（setup.py なし） |
| `preprocess_nnsvs.py` | wav → 学習用 npz。nnsvs の MelF0 と同じ mel と F0 |
| `config_nnsvs_48k.yaml` / `config_nnsvs_48k_polish.yaml` | 本学習の設定 / 仕上げの設定 |
| `export_nnsvs.py` | スナップショット → `vocoder_model.pth` / `vocoder_model.yaml` |
| `test_hop240.py` | 48kHz・hop 240・mel 80 での動作確認 |

`train_v3.py` と `dataset.py` にも変更がある（mel の形式の切り替え、Windows で DataLoader の worker が動くようにした修正、エポックごとの CUDA キャッシュの解放）。
元の ln-mel の経路は変更前とビット一致。

インストール（Wavehax と同じく `--no-deps`）:

```
python-3.13.15-embed-amd64\python.exe -m pip install --no-deps git+https://github.com/rokujyushi/NHVSing@package
```

## 3. 学習したモデル

- 学習データ: 約 7.1 時間（歌声 6.2 時間 = 波音リツ歌声DB 4.6h・yuu_20 1.25h・shigure 0.4h、話し声 0.9h = ITA コーパス nhoshio / typeB）。
  すべて 44.1kHz を 48kHz に上げたもの。検証用に 4 曲（リツ 1st_color、yuu 17sai、shigure 02、ITA EMOTION100_099）を外してある。
  変換済みデータは `I:\NHVSing-1\dataset\nnsvs48k\`。
- スナップショット: `I:\NHVSing-1\snapshots_nnsvs_48k\`。残してあるのは 150（識別器を入れる前）、200、300、350（本学習の最終）。
  仕上げの 360〜400 もある。
- 損失（mel / stft、1 エポックあたり 93 バッチ換算）: 0 エポック 60.2 / 272 → 150 エポック 12.4 / 100 → 350 エポック 11.0 / 97.5
  → 仕上げ後の 400 エポック **10.5 / 96.3**（仕上げの最初の 10 エポックで下がり、あとは横ばい）。
- TensorBoard: `logs_nnsvs_48k/`。検証用の曲の正解音と生成音を 10 エポックごとに聞ける。

**書き出したもの**

`I:\NHVSing-1\exported_models\nnsvs48k\vocoder_model.pth`（1.7MB）と `vocoder_model.yaml`。400 エポック版。
次のコマンドで作り、サーバーと同じ読み方で読み直して、学習時のモデルと出力が一致すること（差 0）を確かめてある。

```
cd I:\NHVSing-1
<同梱 Python> export_nnsvs.py --config config_nnsvs_48k_polish.yaml --ckpt snapshots_nnsvs_48k/000400epoch.pth --out exported_models/nnsvs48k
```

- **git には入れていない。** 元のリポジトリは `exported_models/` を追跡する設定だが、学習データ（波音リツ歌声DB ほか）の利用規約があるので、
  公開リポジトリに push するかどうかは持ち主が決める。
- 仕上げで音が良くなったかは、まだ耳で確かめていない。TensorBoard で 350 と 400 を聞き比べられる。
  悪くなっていたら `--config config_nnsvs_48k.yaml --ckpt snapshots_nnsvs_48k/000350epoch.pth` で書き出し直す。
- 同梱の組み込み Python はスクリプトのあるフォルダを `sys.path` に入れない。`export_nnsvs.py` は自分で入れるのでそのまま動くが、
  `train_v3.py` は `-c "import sys, runpy; sys.path.insert(0, 'I:/NHVSing-1'); ..."` のように runpy で起動する必要がある。
- スナップショットは 1 個 700MB。400 と比較用に少し残して、ほかは消してよい。

## 4. サーバーとの受け渡し形式（確定）

計画書の案どおり。実際に書き出して確かめた。

- `vocoder_model.pth`: `{'model': {'generator': state_dict}}`。weight norm は外してある。**1.7MB**。
  `torch.load(path, weights_only=True)` で読める。
- `vocoder_model.yaml`:

```yaml
generator:
  _target_: nhvsing.model.NHVSingV3
  vocoder_cfg: {hop_size: 240, in_channels: 80, sample_rate: 48000, use_weight_norm: false, n_harmonic: 200, noise_std: 0.03, f0_upsample: linear, ...}
  ltv_filter_cfg: {hop_size: 240, in_channels: 80, ccep_size: 256, fft_size: 1024, ola_mode: hann, use_v3: true, ...}
data:
  feat_names: [mel]
  sample_rate: 48000
  hop_size: 240
  mel: {num_mels: 80, fft_size: 1024, win_length: 960, hop_size: 240, fmin: 30.0, fmax: 24000.0, eps: 1.0e-10, log_base: 10}
nhvsing: {source_ckpt: 000400epoch.pth, epoch: 400, config: config_nnsvs_48k_polish.yaml}
```

- 読み方は Wavehax（`enuserver/wavehax.py`）と同じでよい:
  `generator = hydra.utils.instantiate(config.generator)` → `generator.load_state_dict(ckpt['model']['generator'])`。
  `export_nnsvs.py` はこの読み方で読み直し、学習時のモデルと出力が一致すること（差 0）を確かめてから終わる。
  hydra は vocoder_cfg / ltv_filter_cfg を DictConfig のまま渡すが、NHVSingV3 はそれで動く（確認済み）。

## 5. 推論の入出力（確定）

```python
wav = generator(x, cf0, uv)   # 返り値: (1, T*240) float32、-1〜1 に clamp 済み
```

| 引数 | 形 | 中身 |
|---|---|---|
| `x` | `(1, T, 80)` float32 | 音響モデルが出した mel。**正規化を外した log10 の値そのまま**（nnsvs の multistream_features の mel ストリーム）。`in_vocoder_scaler` は使わない |
| `cf0` | `(1, 1, T)` float32 | `exp(lf0)` の連続 F0 [Hz]。nnsvs の lf0 ストリームは既に連続値 |
| `uv` | `(1, 1, T)` float32 | **1 = 無声**、0 = 有声。`vuv < 閾値（0.5）` を 1 にする |

- 学習データの F0 は nnsvs の MelF0 と同じ作り方（harvest、D4C の非周期性で vuv、log-F0 を補間して 20Hz で平滑化）。
  なので音響モデルの lf0 / vuv をそのまま渡せばよい。
- **決定性**: 雑音を `torch.normal` で作るので、呼ぶたびに雑音成分が変わる。同じ seed（`torch.manual_seed`）なら出力は完全に一致する（確認済み）。
  サーバーのキャッシュと合わせるなら、合成の前に seed を固定する。
- **最後の 1 hop（5ms）は 0 までフェードアウトする。** Hann 窓の重ね合わせで、最後のフレームだけ後ろに重なる相手がいないため。
  hop 256 の元の実装でも同じ形なので仕様。フレーズの末尾は普通は無音なので問題になりにくいはず。
- **速度（CPU、10 秒の入力、350 エポック版）**: 1 スレッドで RTF 0.146、4 スレッドで 0.087、8 スレッドで 0.080。
  GPU で仕上げの学習が動いている間に測ったので、空いていればもう少し速いかもしれない。

## 6. ENUNUServer 側でやること

計画書の「作業手順（ENUNUServer 側）」のとおり。

1. `enuserver/nhvsing.py` を作る。`wavehax.py` と同じく `nnsvs.util.load_vocoder` を包み、`generator._target_` が `nhvsing.` で始まるときだけ読む。
2. 読み込み時に次を確かめ、合わなければ分かりやすいエラーにする。
   - 音響モデルの `feature_type` が `melf0` であること
   - `stream_sizes[0]`（mel の次元）が `data.mel.num_mels` と同じこと
   - `sample_rate` / `frame_period` が `data.sample_rate` / `data.hop_size` と合うこと
3. `enunu.py` の `ENUNU.predict_waveform` に分岐を足す（Wavehax と同じ場所）。入力は 5 章のとおり。
4. 決定性: 合成の前に seed を固定するかどうかを決める（サーバーのキャッシュの扱いに合わせる）。
5. `tests/test_unit.py` に乱数初期化のモデルでのテストを足す（`use_weight_norm: false` の NHVSingV3 を作って書き出せばよい）。
6. `docs/server_pitch_commands.md` に説明を書き、`requirements.txt` にコメントで依存を書く（`--no-deps` で入れる）。
7. **実際の音源で鳴らす。** 手元の melf0 音源（欲音ルコ melf0、波音リツ クリスクロス、KanadeShia MEL、雨星サイファ）のフォルダで、
   `vocoder_model.pth` / `.yaml` を書き出したものに差し替えて合成し、SiFiGAN と聞き比べる。
   **音源フォルダの元のファイルは必ず退避してから差し替えること。**

## 7. 気をつけること・分かっていないこと

- **別人の声への汎化は未確認。** 学習データの歌声は波音リツ・yuu・shigure の 3 人分だけ。リツ以外の音源（ルコ、サイファ、KanadeShia）で
  どこまで鳴るかは、6 章の 7 で聞いてみないと分からない。足りなければ、その音源の歌声 DB で追加学習する
  （`train_v3.py --finetune_from <スナップショット>`）。
- **22〜24kHz の帯域が学習データでは空。** 学習データはすべて 44.1kHz を 48kHz に上げたものなので、この帯域の mel はほぼ無音の値（log10 で約 -7.45）。
  リツの音響モデルの統計では最上位の帯域に中身がある（平均 -5.4）ので、推論では学習で見ていない値が来る。
  mel 損失は log10 1e-5（-5）で下を切って学習しているので影響は小さいはずだが、音で確かめる。
- **fmin / fmax は nnsvs のコードの既定値（30 / 24000）を前提にしている。** 手元の 4 音源のレシピが残っておらず、上書きしていないかは確かめられていない
  （統計ファイルからは判定できなかった）。音響モデルの mel と合わないと音がずれるので、変に聞こえたら最初に疑う。
- 同梱 Python で日本語コメント入りの yaml を `open()` で読むと cp932 で落ちる。`encoding='utf-8'` を付けるか `PYTHONUTF8=1`。
- 学習は 16GB の GPU でメモリぎりぎり。バッチ 16 × 蓄積 2（実効 32）にしてある。学習中に ENUNU サーバーなど GPU を使うものを動かすと遅くなる。

## 8. ENUNUServer 側の結果（2026-09-29）

- `enuserver/nhvsing.py` を追加し、`enunu.py` で `nhvsing.apply()` と `predict_waveform` の分岐（`vocoder_type` は `auto` か `nhvsing`）を足した。
  読み込み時の照合（feature_type / mel 次元 / sample_rate / hop）は音源の `config.yaml` と音響モデルの `stream_sizes` で行う。
- 決定性: `synthe` は既に UST のハッシュから `torch.manual_seed` しているので、追加の処理は要らなかった。雑音は CPU で作るので GPU でも同じ。
- テスト: `tests/test_unit.py` の `TestNHVSing`（乱数初期化の小さい NHVSingV3 で合成・同じシードで一致・照合のエラー 4 種）。単体テスト 36 件すべて通る。
- 同梱 Python 3.13 に `pip install --no-deps git+https://github.com/rokujyushi/NHVSing@package` で入れた。
- 実音源: 音源フォルダは触らず、スクラッチパッドにコピーして `vocoder_model.*` を 400 エポック版に差し替えて合成した（母音だけの 7.5 秒のフレーズ）。
  | 音源 | synthe（GPU、全体） SiFiGAN → NHVSing | ボコーダーの CPU RTF 1 / 4 スレッド | F0 差の中央値 / 95% | 有声無声の一致 |
  |---|---|---|---|---|
  | 欲音ルコ melf0 | 1.14 s → 0.68 s | 0.224 / 0.084 | +0.4 / 17 cent | 0.957 |
  | KanadeShia MEL | 0.80 s → 0.61 s | 0.162 / 0.086 | −0.8 / 65 cent | 0.978 |
  | 雨星サイファ MS | 0.79 s → 0.64 s | 0.150 / 0.084 | −0.6 / 41 cent | 0.997 |

  mel の差（SiFiGAN の出力との log10 L1）は 0.28〜0.32、高い帯域（40〜79）の方が大きい（0.36〜0.37）。音の良し悪しは耳で確かめる必要がある。
- 波音リツ クリスクロスは、SiFiGAN のままでも合成できなかった（音源の拡張機能 `style_shifter.py` が、2 回目の呼び出しで f0_editor と判定され、
  `--f0` が無いまま f0 を読もうとして落ちる）。NHVSing とは別の既存の問題。
