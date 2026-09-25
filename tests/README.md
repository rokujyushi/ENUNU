# ENUNUServer テスト

`ENUNUServer-1.0.0` フォルダで、同梱の Python から実行します（pytest は不要）。

```
cd ENUNUServer-1.0.0
python-3.12.10-embed-amd64\python.exe -m unittest discover -s ..\tests -v
```

一部だけ実行する場合は `-p test_unit.py` のようにファイルを指定するか、`-k editorf0` のように名前で絞り込みます。
（同梱の埋め込み Python ではテストフォルダが sys.path に入らないので、`python -m unittest test_e2e` の形では実行できません）

| ファイル | 内容 | 所要時間 |
|---|---|---|
| `test_unit.py` | モデルを使わない単体テスト（style_shift の解釈、拡散設定の優先順位と検証、features.npz の入出力、editorf0 の差し替え、拡張機能のプロセス内実行） | 1 秒未満 |
| `test_e2e.py` | 実モデルでコマンドをプロセス内から呼ぶ（フレーズ切り替え、再起動後の synthe、editorf0 で +200 cent、UST / style_shift / 拡散設定の変更でキャッシュが無効になること、pitch / acoustic_f0 のキャッシュ、旧 melf0 ワークフォルダ） | 音源 1 つあたり 15〜20 秒 |
| `test_zmq.py` | サーバープロセスを 15599 番ポートで起動し、OpenUtau と同じ形式のリクエストを送る（ver_check の必須、config、エンジンの有効期限） | 約 30 秒 |

- 実モデルを使うテストは、既定では `F:\UTAU\voice` の KanadeShia（WORLD）と欲音ルコ melf0 を使います。
  音源が見つからない場合はスキップします。環境変数 `ENUNU_TEST_VOICES` に `;` 区切りで音源フォルダを指定すると変えられます。
- テスト用の tmp は OpenUtau の `EnunuUtils.WriteUst` と同じ形式で、1 ノート = 1 音素です（歌詞は母音のみ）。
