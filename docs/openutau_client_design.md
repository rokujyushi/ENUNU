# OpenUtau クライアント改修の設計（ENUNUServer 2 対応）

OpenUtau の ENUNU クライアント（`OpenUtau.Core/Enunu/`）を、このサーバーの新しいコマンドに対応させ、
あわせてサーバーとの接続まわりを作り直すための設計です。サーバー側の仕様は [server_pitch_commands.md](server_pitch_commands.md) を見てください。

- この文書は設計を話し合いながら更新します。実装はユーザーが行い、第1段階は Claude が叩き台を書きました（[叩き台の状態](#叩き台の状態)）。
- ENUNUServer 2 は最新のクライアントとセットで使います。旧クライアントを使う人には ENUNUServer 1 を使ってもらいます。
- **ブランチ**（OpenUtau リポジトリ）: PR を分けるため、2つの大きな機能を別々のブランチで作り、テスト用のブランチで一緒にマージして確かめます。
  - `add/DiffusionENUNU`（master から分岐）: ENUNU の変更だけ。この文書の内容です。
  - `add/generic-live-pitch`（未マージ）: リアルタイムピッチを DiffSinger 以外にも広げる改修。
    `IRenderer` に `LivePitchCost` / `SupportsFastLivePitch` / `LoadRenderedPitch(phrase, positions, PitchGenerationOptions)` が追加され、
    部分書き戻し（選んだノートの範囲だけを書き戻し、境目を 50 ms でクロスフェード）は `PitchRetake.BuildWriteBackMask` が行います。
  - 両方にまたがる変更（ENUNU の `LivePitchCost` を新しいサーバーで Light にする、[1-6](#1-6-loadrenderedpitch)）は、後からマージする方の PR に入れます。
- 見出しの番号（1-1、2-3 など）は下の「進め方」の表と対応しています。話し合いのときはこの番号で呼びます。

## 決定事項

| 日付 | 決めたこと |
|---|---|
| 2026-09-25 | 旧サーバー（`features` が無い）では今と同じ動きを保つ。新しいコマンドは `features` があるサーバーにだけ送る |
| 2026-09-25 | `EnunuClient` は通信だけにする（送る、受け取る、タイムアウト）。プロトコルの知識や再送は入れない |
| 2026-09-25 | サーバーとのやりとりは `EnunuConnection` にまとめる。メソッドはサーバーのコマンドと1対1（`Timing` / `Acoustic` / `Pitch` / `AcousticF0` / `Synthe` / `Config`） |
| 2026-09-25 | サーバーは `ver_check` より前のコマンドも受け付ける（サーバー側は変更済み）。再起動しても `run ver_check.` にならない |
| 2026-09-25 | ポートは環境設定で指定できるようにする（空なら自動）。サーバーを `ENUNU_SERVER_PORT` で別の番号で起動する場合に対応するため |
| 2026-09-25 | ver_check のタイムアウトは環境設定で変えられるようにする（既定 3 秒。今は 1 秒固定） |
| 2026-09-25 | サーバーから応答が無いときは、ポート番号は出さずに「サーバーを起動（再起動）してください」と伝える |
| 2026-09-25 | 先に合成の流れ（第1段階）を動かして確認し、接続の作り直し（第2段階）はその後に行う |
| 2026-09-26 | config は「送ったかどうか」を覚えずに、acoustic 系のリクエストの前に毎回送る（[1-7](#1-7-拡散ステップ数)） |
| 2026-09-26 | サーバーは ENUNUServer 2（`version` は `2.0.0`）として新しいクライアント専用にする。旧クライアントは ENUNUServer 1 を使う |
| 2026-09-26 | リクエストは全コマンドで `[5]` に style_shift（必須、クライアントはいつも 0）。`acoustic_f0` の f0 は `[6]`（[1-3](#1-3-enunuconnection-の骨組み)） |
| 2026-09-26 | 新しいクライアントは旧サーバーにもつながるようにする（旧サーバーは `[5]` 以降を読まないので 6 要素で送ってよい） |
| 2026-09-26 | `voiced` は「ノートの範囲の外（前後の余白）なら false」。子音など f0 = 0 のフレームは補間して書き戻す（[1-6](#1-6-loadrenderedpitch)） |
| 2026-09-26 | ENUNU の変更とリアルタイムピッチは別ブランチ・別 PR にし、テスト用ブランチで一緒に確かめる |
| 2026-09-26 | 環境設定を変えたときにはサーバーにアクセスしない。設定画面は `Preferences` に保存するだけにし、サーバーに関わる設定（config、ポートなど）は、次に合成などのリクエストを送るときに読んで使う |
| 2026-09-26 | 拡散ステップはストリームごと（mgc / bap / mel）に設定する。手法は選ばせず、「おすすめ（高速化）」と「モデルの設定に従う（間引きなし）」の切り替えを1つ置く（[1-7](#1-7-拡散ステップ数)） |
| 2026-09-27 | 変換表に無い歌詞は EnunuPhonemizer が休符（`R`）にし、ログに警告を出す。空白で区切った各トークンが変換表のキーか音素名（表の右側）なら、そのまま通す（例: `i B`、`u R`）。「さ子音」「息_あ」のような歌詞の特別な規則は作らない。サーバーは変えない（表に無い日本語の歌詞は score.full を cp932 で読めずに落ちていた） |

## 進め方

### 第1段階: 合成の流れを新しいサーバーに対応させる

| # | 変更 | 主なファイル |
|---|---|---|
| [1-1](#1-1-enunuclientsend) | `EnunuClient.Send`（`object[]` を送り、応答の文字列を返す） | EnunuClient.cs |
| [1-2](#1-2-ver_check-の-features) | `VersionResult.features` | EnunuUtils.cs |
| [1-3](#1-3-enunuconnection-の骨組み) | `EnunuConnection` の骨組み | EnunuConnection.cs（新規） |
| [1-4](#1-4-ust-の準備を共通化する) | UST の準備を共通化する、`LoadRenderedPitch` の lock | EnunuRenderer.cs |
| [1-5](#1-5-render) | Render: `pitch` → `acoustic_f0` | EnunuRenderer.cs |
| [1-6](#1-6-loadrenderedpitch) | LoadRenderedPitch: `pitch` を使う、リアルタイムピッチ | EnunuRenderer.cs |
| [1-7](#1-7-拡散ステップ数) | 拡散ステップ数（config と wav のキャッシュキー） | Preferences.cs、PreferencesViewModel.cs、PreferencesDialog.axaml、Strings*.axaml、EnunuConnection.cs、EnunuRenderer.cs |
| [1-8](#1-8-キャッシュ削除で-_enutemp-も消す) | キャッシュ削除で `_enutemp` も消す | RenderPhrase.cs |

おすすめの順番: 1-1 → 1-2 → 1-3 → 1-4 → 1-5 で音が出ることを確かめ、次に 1-6、1-7、1-8。
最後に [動作確認](#動作確認) の第1段階を行います。

#### 叩き台の状態

2026-09-26 に `add/DiffusionENUNU` の作業ツリーへ叩き台を書きました（未コミット）。それ以前の作業中の変更は `stash@{0}`「WIP before Claude draft」にあります。

| # | 状態 |
|---|---|
| 1-1 〜 1-5 | 済み |
| 1-6 | `LoadRenderedPitch(phrase)` は済み。`LivePitchCost` と options 付きのオーバーロードは `add/generic-live-pitch` とそろってから |
| 1-7 | 済み（ストリームごとのステップ数と「モデルの設定に従う」の切り替え、環境設定の「ENUNU」のページ） |
| 1-8 | 済み（`RenderPhrase.AddCacheDirectory`） |
| 2-4 | Phonemizer の `timing` を `EnunuConnection` に移すところまで済み（`SetPortNum` を削除したため先に行った） |

テスト: `OpenUtau.Test/Core/Enunu/EnunuConnectionTest.cs`（[動作確認](#動作確認)）。

### 第2段階: 接続の作り直し

| # | 変更 | 主なファイル |
|---|---|---|
| [2-1](#2-1-環境設定) | 環境設定: ポートと ver_check のタイムアウト | Preferences.cs、環境設定の画面 |
| [2-2](#2-2-接続先の判定と判定し直し) | 接続先の判定と判定し直し。`SetPortNum` を削除 | EnunuConnection.cs、EnunuUtils.cs |
| [2-3](#2-3-応答が無いときと旧サーバーの再起動) | 応答が無いときの通知、旧サーバーの `run ver_check.` への対応 | EnunuConnection.cs、Strings*.axaml |
| [2-4](#2-4-phonemizer-の移行) | 日本語・英語の Phonemizer を `EnunuConnection` に移す | EnunuPhonemizer.cs、EnunuEnglishPhonemizer.cs |

---

## 第1段階

### 1-1. EnunuClient.Send

`acoustic_f0` は `request[6]` に `double[]` を、`[5]` に数値の style_shift を入れるので、今の `string[]` では送れません。
また、今の `SendRequest<T>` は応答が無いと `default(T)` を返すので、「タイムアウト」と「成功」の区別がつきません。

```csharp
// 応答の文字列を返す。タイムアウトなら null。プロトコルのことは何も知らない
internal string? Send(object[] args, string port, int second) { ...今の SendRequest の本体... }

// 既存の呼び出し (Phonemizer、韓国語 Phonemizer) のために残す
internal T SendRequest<T>(string[] args, string port, int second = 300) {
    var message = Send(args, port, second);
    return string.IsNullOrEmpty(message) ? (T)Activator.CreateInstance(typeof(T))! : Json.Deserialize<T>(message)!;
}
```

- `Json.Serialize(object[])` は `double[]` を数値の配列として出力し、`string` は今と同じ JSON になります。
- double は往復できる最短の表記で出力されるので、同じピッチなら毎回同じ値になり、サーバーのキャッシュ（配列のハッシュ）に当たります。
- 8 秒のフレーズで約 1600 フレーム、JSON にして約 30 KB です。

### 1-2. ver_check の features

```csharp
public struct VersionResult {
    public string name;
    public string version;
    public string author;
    public EnunuServerFeatures features;   // 追加。旧サーバーは null
}

public class EnunuServerFeatures {
    public string[] commands;
    public bool style_shift;
    public bool pitch_n_frames;
    public Dictionary<string, EnunuDiffusionSetting> diffusion;   // {"mgc": {method, steps}, ...}

    public bool Has(string command) => commands != null && commands.Contains(command);
    public bool SupportsPitch => Has("pitch") && Has("acoustic_f0");
}

public class EnunuDiffusionSetting { public string method; public int steps; }
```

`Json` は `IncludeFields = true` の System.Text.Json なので、フィールド名はサーバーの JSON のキーと同じにします。

### 1-3. EnunuConnection の骨組み

Renderer は、新しいコマンドを最初から `EnunuConnection.Inst.Pitch(...)` の形で呼びます。
こうしておけば、第2段階ではこのクラスの中身を変えるだけで済み、Renderer を触り直す必要がありません。

```
EnunuRenderer / EnunuPhonemizer / EnunuEnglishPhonemizer
        │  EnunuConnection.Inst.Acoustic(...) / Pitch(...) / Features
        ▼
EnunuConnection   接続先・サーバーの機能を持つ。サーバーのコマンドと1対1のメソッド
        │  EnunuClient.Inst.Send(object[] request, port, timeout)
        ▼
EnunuClient       ZMQ で送って受け取るだけ
```

```csharp
class EnunuConnection : Util.SingletonBase<EnunuConnection> {
    readonly object stateLock = new();
    string? port;                       // null = まだ判定していない
    EnunuServerFeatures? features;      // 旧サーバーは null

    /// UI からも読む。通信しない。まだ判定していなければ null
    public EnunuServerFeatures? Features => features;

    public TimingResponse     Timing(string ust, string vbHash)
    public AcousticResponse   Acoustic(string ust, string vbHash)
    public PitchResponse      Pitch(string ust, string vbHash)
    public AcousticResponse   AcousticF0(string ust, string vbHash, double[] editorF0)
    public SyntheResponse     Synthe(string ust, string wav, string vbHash)
    public ConfigResponse     Config(object body)

    // [command, ust, wav, voicebank, "600", style shift (いつも 0), ...extra]。acoustic_f0 は extra に f0 を渡すので [6] に入る
    internal static object[] CommandRequest(string command, string ust, string wav, string vbHash, params object[] extra)

    T Request<T>(object[] request) where T : IEnunuResponse {
        string port = EnsureConnected();
        string? message = EnunuClient.Inst.Send(request, port, RequestTimeoutSec);
        if (string.IsNullOrEmpty(message)) throw new Exception(...);   // 第2段階でわかりやすいメッセージにする
        var response = Json.Deserialize<T>(message)!;
        if (response.Error != null) throw new Exception(response.Error);
        return response;
    }

    string EnsureConnected() {
        lock (stateLock) {
            if (port == null) {
                // 第1段階では今の SetPortNum と同じ決め方 (15556 に ver_check、応答が無ければ 15555)
                (port, features) = Detect();
            }
            return port;
        }
    }
}
```

- リクエストの送信は lock の外で行います。Render は `lockObj` で1つずつ送っており、Phonemizer やリアルタイムピッチとは並んで送れるようにしておきます。
- エラーの応答と応答が無いときは、`Request` の中で例外にします。呼び出し側で `error` を確かめる必要はありません。
  第1段階では、判定し直しとユーザー向けのメッセージはまだ入れません（[2-2](#2-2-接続先の判定と判定し直し)、[2-3](#2-3-応答が無いときと旧サーバーの再起動)）。
- パスや音源のハッシュはフィールドに持たず、引数で受け取ります。こうすると1つのインスタンスを全トラック・全スレッドで使えます。
- `lf0_conditioning` は `Pitch` / `AcousticF0` の応答から音源ごとに記録し、`Lf0Conditioning(vbHash)` で読めます。
- レスポンスの型は `EnunuConnection.cs` にまとめ、`IEnunuResponse`（`Error`）を付けます:
  - `PitchResponse`: `path_f0`、`lf0_conditioning`（bool）、`n_frames`
  - `AcousticResponse`: 今の項目に `lf0_conditioning`（`bool?`、`acoustic_f0` のときだけ入る）を追加

### 1-4. UST の準備を共通化する

今は `Render` の中で、npy が無いときだけ UST（`enu-{hash}.tmp`）を書いています。
新しい流れでは、`LoadRenderedPitch`（リアルタイムピッチ）が Render より先に `pitch` を送ります。そのとき UST が無いとサーバーはエラーを返します。

1. **パスを作る処理を1か所にまとめる**。今は `Render` と `LoadRenderedPitch` の両方に同じコードがあります。
   ```csharp
   record EnunuPaths(string ustPath, string enutmpPath, string wavPath, string voicebankNameHash);
   EnunuPaths PreparePaths(RenderPhrase phrase)
   ```
2. **UST が無ければ書く `EnsureUst`**。
   ```csharp
   void EnsureUst(RenderPhrase phrase, EnunuConfig config, string ustPath) {
       if (File.Exists(ustPath)) return;
       var enunuNotes = PhraseToEnunuNotes(phrase, config);
       EnunuUtils.WriteUst(enunuNotes, phrase.phones.First().tempo, phrase.singer, ustPath);
   }
   ```
   tmp のパスは音素とノートの内容のハッシュなので、ファイルがあれば中身も同じです。
   `Render`（新旧どちらのサーバーでも）と `LoadRenderedPitch` の両方で、コマンドを送る前に呼びます。
3. **`LoadRenderedPitch` も `lockObj` を取る**。リアルタイムピッチは Render と別のスレッドで動くので、
   同じフレーズの tmp を書く処理と `_enutemp` の npy を読む処理が重なることがあります。
   待つのは Render が処理中の1フレーズ分だけです（サーバーも要求を1つずつ処理するので、並べても速くはなりません）。

### 1-5. Render

`features` が無いサーバーでは今の処理のままです。新しいサーバーでは次の流れにします。

```
if wav がある → 読むだけ（今と同じ）
EnsureUst
lf0cond = lf0Conditioning[voicebankNameHash]      // static Dictionary<string,bool>。分からなければ null
if lf0cond != false:
    pr = EnunuConnection.Inst.Pitch(ust, vb)       // 結果はサーバーにキャッシュされる (pitch_f0.json)
    lf0Conditioning[vb] = pr.lf0_conditioning
if lf0Conditioning[vb] == true:
    editorF0 = SampleCurve(phrase, phrase.pitches, 0, fp, pr.n_frames, head, tail, ToneToFreq)
    ac = EnunuConnection.Inst.AcousticF0(ust, vb, editorF0)
else:
    ac = EnunuConnection.Inst.Acoustic(ust, vb)     // 旧来のモデル
ac.error != null → throw
以降は今と同じ:
  WORLD  → f0 / sp / ap を読んで Worldline.WorldSynthesis（editorF0 と f0<50 のマスクも今と同じ）
  synthe → editorf0.npy を毎回書く → EnunuConnection.Inst.Synthe(...)
```

- **`lf0_conditioning = false` のモデルでは `acoustic_f0` を使わない**。このモデルでは `acoustic_f0` は editor_f0 を無視しますが、
  キャッシュのキーには editor_f0 のハッシュが入るので、ピッチを変えるたびに無駄な再計算になります。
  これまでどおり `acoustic`（キャッシュに当たる）と、`synthe` の前の `editorf0.npy` によるピッチの差し替えを使います。
- **新しいサーバーでは、npy があるかどうかで acoustic を省く処理をやめる**。`acoustic_f0` の後の npy は、ピッチによって中身が変わるからです。
  キャッシュの判定はサーバーの `features.npz` に任せます。wav のキャッシュが先にあるので、リクエストが増えるのは wav が無いときだけです。
- **`editorf0.npy` は `acoustic_f0` の後でも必ず書く**。値は同じなので害はありません。書かないと、前に書いた古いものが `synthe` で使われます。
- editorF0 の先頭と末尾（head / tail）のフレームは 0 のままにします。0 のフレームはモデルのピッチになるので、そのままで正しく動きます。
- **エラー処理**: 今の synthe 系の分岐は、acoustic のエラーをログに出すだけで、その後 `np.Load` で落ちます。例外を投げるようにします。
- style_shift はリクエストの `[5]` ではいつも 0 を送ります。ノートごとの値は UST の `S` フラグで送っているので、`[5]` にも入れると二重にかかります。

### 1-6. LoadRenderedPitch

`add/DiffusionENUNU`（master）には `LoadRenderedPitch(phrase)` しかないので、まずこれを実装します。
`add/generic-live-pitch` とそろった後は、`LoadRenderedPitch(phrase, positions, PitchGenerationOptions options)` からこれを呼ぶだけにします。options は使いません。

- **新しいサーバー**: `lockObj` → `EnsureUst` → `Pitch` → `path_f0`（`pitch_f0.npy`）を読みます。lf0_conditioning もここで記録します（1-5 と同じ辞書）。
- **旧サーバー**: 今と同じく、`f0.npy` があれば読みます。

**無声フレーム**: 今は `FreqToTone(0)` が -∞ になり、`y > minPitD` の判定で飛ばされています。
その結果、ノートを動かした後も子音の部分に古い PITD が残ります。
- f0 > 0 のフレームを tone に変換し、0 のフレームは前後の有声フレームから線形補間します（`F0ToTones`）。両端は一番近い値で埋めます。
- `result.voiced` は「ノートの範囲の外（前後の余白）なら false」にします。
  master の `NoteBatchEdits` は `voiced == false` のフレームを無音として書き戻さないので、`f0 > 0` にすると子音の部分が書き戻されず、古い PITD が残ります。
  部分書き戻しのクロスフェード（`BuildCrossfadeWeights`、`add/generic-live-pitch`）も、この `voiced` をフェードの基準に使います。テスト用ブランチで確かめます。
- 有声フレームが1つも無ければ null を返します。

**フレームの時刻**: UST の先頭は長さ `headTicks` の R なので、フレーム j の時刻は `TickPosToMsPos(position − headTicks) + j·fp` です（`FrameMs`）。
- 以前の `LoadRenderedPitch` は `t += framePeriod` を代入より先に実行していたので、1フレーム（5 ms）遅れていました。
- 以前の `SampleCurve` は `phrase.leadingMs`（先頭の音素の先行発声）を基準にしていたので、先行発声が 0 でないとずれていました。
- 叩き台では両方とも `FrameMs` を使います。リアルタイムピッチでは「pitch → PITD → editorF0 → acoustic_f0」と値が行き来するので、
  ずれがあると、声色の計算に使うピッチが実際より少しずれます。
  **旧サーバーの WORLD の合成でも、ピッチの位置が最大で半フレームと先行発声の分だけ変わる**ので、聞いて確かめます。
- あわせて `SampleCurve` は、カーブの範囲外のフレームを 0 ではなく既定値（gender なら 0.5）で埋めます。

**リアルタイムピッチ**（`add/generic-live-pitch` とそろってから）:
- `LivePitchCost => EnunuConnection.Inst.Features?.SupportsPitch == true ? Light : Heavy`
  （pitch は 0.1〜0.2 秒。旧来のモデルは音響モデル全体を実行しますが、それでも 0.3〜0.5 秒です）。
  この getter は UI スレッドから呼ばれるので、通信せずに記録した値を読むだけにします。判定がまだなら Heavy です。
- `SupportsFastLivePitch` は false のままにします（pitch にはステップ数の設定がありません）。
- `retakeMask` は null で返します。`BuildWriteBackMask` が、選んだノートの部分だけを書き戻します。

### 1-7. 拡散ステップ数

音源の種類によって使う拡散ストリームが違います（WORLD 系は mgc と bap、melf0 系は mel）。そのため、ストリームごとに設定できるようにします。
サーバーの設定はサーバー全体で1つなので、全音源に共通です。

**環境設定（「ENUNU」のページ）**:

| 項目 | `Preferences` | 内容 |
|---|---|---|
| 拡散モデルのサンプリング | `EnunuDiffusionMode`（0 / 1） | 0: おすすめ（高速化）。1: モデルの設定に従う（間引きなし、遅い） |
| ステップ数：スペクトル（mgc） | `EnunuDiffusionStepsMgc` | おすすめのときだけ有効。0（「おすすめの値」）はサーバーのおすすめ（DDIM 25） |
| ステップ数：非周期性（bap） | `EnunuDiffusionStepsBap` | 同上（PLMS 20） |
| ステップ数：メルスペクトログラム（mel） | `EnunuDiffusionStepsMel` | 同上（DDIM 25） |

- 選べるステップ数は 0（おすすめの値）/ 5 / 10 / 15 / 20 / 25 / 30 / 50 / 100 です。手法（DDIM / PLMS など）は選ばせず、サーバーの既定のままにします。
- 「モデルの設定に従う」: nnsvs のモデルは `pndm_speedup` を持てない（指定すると `NotImplementedError`）ので、
  モデル本来の設定は常に間引きなし（`acoustic_model.yaml` の `K_step` 回、通常 100 回）です。

**config**:
- **acoustic 系のリクエスト（`Acoustic` / `AcousticF0` / `Synthe`）の前に毎回送ります**。送るのは
  `features` の commands に `config` があるサーバーだけです（`EnunuConnection.DiffusionConfigRequest`）。
  - 毎回 `reset` を付けて全体を送ります。サーバーは `reset` とストリームの指定を一緒に受けると、既定値から始めてそのストリームだけを変えます（2026-09-26 にサーバー側を変更）。
    これで、前に送ったストリームの指定が残りません。
  - おすすめ: `{"reset": true, "mgc": {"steps": 10}, "bap": {"steps": 30}}`（0 のストリームは送らない）
  - モデルの設定: `{"reset": true, "mgc": "ddpm", "mel": "ddpm", "bap": "ddpm"}`
  - 毎回送る理由: サーバーは ver_check なしでもコマンドを受け付けるようになったので、クライアントはサーバーの再起動に気づけません。
    「送ったかどうか」を覚えておくと、再起動後に設定が既定値に戻ったことがわかりません。
    サーバーのキャッシュ判定には設定の**値**が入っているので（`features_meta` の `diffusion`）、同じ値を何度送ってもキャッシュは無効になりません。
    1回の往復は数ミリ秒です。
  - 送るのに失敗してもログを出すだけにして、合成は続けます。
- **wav のキャッシュキーに設定を含める**。含めないと、設定を変えても前の wav が使われます。
  ```csharp
  // EnunuDiffusionPreferences.CacheKey: 設定の XXH64 (セッションをまたいで同じ値)。すべて既定なら 0 → 今のパスのまま
  if (features?.diffusion != null) wavHash += EnunuDiffusionPreferences.Current.CacheKey;
  ```
  「モデルの設定に従う」ではステップ数を使わないので、ステップ数が違っても同じキーです。
  tmp や `_enutemp` のパスには設定を含めません。npy はサーバー側で、設定が変われば作り直されます。

### 1-8. キャッシュ削除で _enutemp も消す

`RenderPhrase.DeleteCacheFiles` はファイルしか消さないので、「選択ノートのキャッシュ削除」をしても `_enutemp` が残り、再推論されません。

- `RenderPhrase.AddCacheDirectory(string)` を追加し、登録したフォルダを `DeleteCacheFiles` で `Directory.Delete(dir, true)` します。例外は catch してログに出すだけにします。
- 名前の前方一致でフォルダを消す案もありましたが、ほかのレンダラーに影響しないよう、消すフォルダを明示的に登録する形にしました。
- `EnunuRenderer.Render` で `phrase.AddCacheDirectory(paths.EnutmpPath)` を呼びます。
- `_enutemp` が消えると `features.npz` と `pitch_f0.json` も消えるので、サーバーは作り直します。

---

## 第2段階

### 今の接続の問題

| 状況 | 今の動き | 原因 |
|---|---|---|
| OpenUtau を起動した後で ENUNUServer を起動した | ずっと 15555 に送るので、つながらない | `SetPortNum` の結果をインスタンスの `port` に一度だけ入れる |
| 旧サーバーを再起動した | すべてのリクエストが `run ver_check.` になる | クライアントが ver_check を送り直さない（新しいサーバーでは起きない） |
| サーバーを別のポートで起動した | つながらない | 候補が 15556 と 15555 だけ |
| サーバーが落ちている | 1回のリクエストで 300 秒待ち、その後 `np.Load` で落ちる | タイムアウトとエラーを区別していない |
| Renderer と Phonemizer | それぞれ `SetPortNum` を呼んで `port` を持つ | 判定が共通になっていない |

`SetPortNum` は、「どのポートか」「どのサーバーか」「ver_check 済みか」を、失敗すると 15555 になる1つの戻り値にまとめてしまっています。

### 2-1. 環境設定

| 項目 | 既定 | 内容 |
|---|---|---|
| `EnunuServerPort`（string） | 空（自動） | 空: 15556 に ver_check を送り、応答が無ければ 15555（今と同じ）。番号: その番号だけを使い、15555 には切り替えない。ver_check の応答が無くても、そのポートに送る（旧 SimpleEnunuServer 系も考えて） |
| `EnunuVerCheckTimeoutSec`（int） | 3 | ver_check を待つ秒数。サーバーが別のフレーズを処理中だと ver_check も待たされるので、短すぎると「サーバーが無い」と誤って判断する |
| `EnunuDiffusionMode` / `EnunuDiffusionStepsMgc` / `Bap` / `Mel`（int） | 0 | [1-7](#1-7-拡散ステップ数)（済み） |

### 2-2. 接続先の判定と判定し直し

```csharp
string? portSetting;   // 判定したときの Preferences.EnunuServerPort

string EnsureConnected() {
    lock (stateLock) {
        var setting = Preferences.Default.EnunuServerPort ?? "";
        if (port != null && portSetting == setting) return port;
        (port, features) = Detect(setting, Preferences.Default.EnunuVerCheckTimeoutSec);
        portSetting = setting;
        return port;
    }
}

void Disconnect() { lock (stateLock) { port = null; features = null; } }
```

- 判定し直すきっかけ: まだ判定していないとき、ポートの設定が変わったとき、応答が無かったとき（`Disconnect`）、旧サーバーから `run ver_check.` を受けたとき。
- 「ポートの設定が変わったとき」も、設定画面で変えた時点では何も送りません。次のリクエストで `EnsureConnected` が
  `Preferences.EnunuServerPort` と判定したときの値（`portSetting`）を比べ、違っていればそこで ver_check を送ります。
  定期的に試し直す処理は入れません。
- 2つのスレッドが同時に失敗すると、ver_check が1回余分に送られることがあります。害はないので、わかりやすさを優先します。
- `EnunuUtils.SetPortNum` は削除します（中身は `Detect` に移ります）。

### 2-3. 応答が無いときと旧サーバーの再起動

```csharp
T Request<T>(object[] request, int timeoutSec = 300) {
    string port = EnsureConnected();
    string? message = EnunuClient.Inst.Send(request, port, timeoutSec);
    if (message != null && IsRunVerCheck(message)) {   // 旧サーバーを再起動した場合
        SendVerCheck(port);
        message = EnunuClient.Inst.Send(request, port, timeoutSec);
    }
    if (string.IsNullOrEmpty(message)) {
        Disconnect();                                    // 次のリクエストで判定し直す
        throw new MessageCustomizableException(
            "ENUNU server is not responding.", "<translate:errors.enunu.noresponse>", null);
    }
    return Json.Deserialize<T>(message)!;
}
```

- **メッセージ**: ポート番号は出さず、何をすればよいかを伝えます。
  - `Strings.axaml`: `errors.enunu.noresponse` = "No response from the ENUNU server. Please start (or restart) the server."
  - `Strings.ja-JP.axaml`: 「ENUNU サーバーから応答がありません。サーバーを起動（または再起動）してください。」
- Render で `MessageCustomizableException` を投げると、`RenderEngine` がそのメッセージをエラーダイアログに出します（新しい通知の処理は不要です）。
- `run ver_check.` の判定は、応答の JSON の `error` を見るだけです。コマンドは何度実行しても結果が同じなので、1回再送しても問題ありません。

### 2-4. Phonemizer の移行

- `EnunuPhonemizer`（日本語）と `EnunuEnglishPhonemizer` の `timing` を `EnunuConnection.Inst.Timing(...)` にし、`port` フィールドを削除します。
- `EnunuKoreanPhonemizer` は 15555 固定の別サーバー（`["timing", ustPath]` の2要素）なので、今のまま `EnunuClient` を使います。

---

## 互換性

新しいクライアントから見たサーバーごとの動き:

| サーバー | 判定 | 動き |
|---|---|---|
| ENUNUServer 0.6（15555） | 15556 から応答なし（自動のとき） | 今と同じ |
| SimpleEnunuServer 0.5 / ENUNUServer 1（main の 1.0.0）（15556、`features` なし） | name あり、features が null | 今と同じ（npy の有無で判定し、editorf0.npy を使う。リアルタイムピッチは Heavy） |
| `features` 付きの 1.0.0（`feature/server-cache-and-speedup` の途中の版だけ。f0 は `[5]`） | features あり | **つながらない**: クライアントは f0 を `[6]` に送るので `acoustic_f0` が失敗する。公開していない版なので対応しない |
| ENUNUServer 2（このサーバー） | features あり | 1-5 〜 1-7 の流れ |
| 韓国語 Phonemizer | 15555 の `timing` | 影響なし |

旧クライアント（今出ている OpenUtau）は ENUNUServer 2 では動きません（`[5]` を送らないため）。旧クライアントを使う人には ENUNUServer 1 を使ってもらいます。

## 動作確認

**第1段階**:
- 使う音源は3種類です。
  - KanadeShia_20251101（world 拡散 + HN-uSFGAN、synthe）
  - 欲音ルコ melf0（melf0 + SiFi-GAN）
  - 夏目悠李 RMDN v0.0.3（lf0_model なし）
- 確かめる操作と結果:
  - ノートを動かす → リアルタイムピッチ → 再生
  - ピッチを描く → 再生（`acoustic_f0` が送られ、`pitch` はキャッシュに当たる）
  - ステップ数を変える → 再生（作り直される）
  - キャッシュを削除 → 再生（作り直される）
- `Plugins/ENUNUServer-0.6.0` で、今と同じ動きになること。

**第2段階**:
- OpenUtau の後にサーバーを起動する → 合成できる
- サーバーを再起動する → 合成できる
- ポートの設定を変えて、`ENUNU_SERVER_PORT` で起動したサーバーにつながる
- サーバーを止める → 「サーバーを起動してください」が出る

**ユニットテスト（`OpenUtau.Test/Core/Enunu/EnunuConnectionTest.cs`）**:

.NET 10 の SDK では `dotnet test` が使えないので、xunit の実行ファイルとして動かします。

```
dotnet run --project OpenUtau.Test/OpenUtau.Test.csproj -- -class "OpenUtau.Core.EnunuConnectionTest"
```

- 済み: 無声フレームの補間（`F0ToTones`）、リクエストの JSON の形（`[5]` = 0、`acoustic_f0` の f0 は `[6]`、config）、
  `features` がある応答と無い応答の読み込み、レスポンスの読み込み（`lf0_conditioning` の bool / bool?、エラー）
- 済み: 実際のサーバーとの通し（pitch → acoustic_f0 → synthe）。環境変数 `ENUNU_TEST_VOICE` に音源のフォルダを指定し、
  15556 番でサーバーを起動しておいたときだけ動きます（2026-09-26、KanadeShia で通過）
- 未: `FrameMs` と `SampleCurve` のフレーム位置、旧サーバーでは wav のパスが今と変わらないこと

## 未決事項

- リアルタイムピッチで応答が無かったとき: 今は `RealTimePitchGenerationService` がログを出すだけです。トーストを出すかどうか（第2段階で決める）。
- ver_check のタイムアウトに加えて、「一度 15556 で応答があったら、判定し直すときに 15555 へ切り替えない」ルールも入れるかどうか。

## 範囲外の既存の不具合

- ~~`EnunuUtils.WriteUst` は `timbre` が空だと `Flags=` の行を書きません。そのため、スタイル指定の無いノートでは `S{style_shift}` もサーバーに届きません。~~
  2026-09-26 にユーザーが修正（`Flags=` の行を毎回書き、style_shift が 0 以外のときだけ `S` を付ける）。
