# 移植元

`src/OnnxSvs` の一部は、次の Python のコードを C# に移したものです。どちらも MIT License です。

| C# | 移植元 |
|---|---|
| `QuestionSet` (質問ファイルの読み込み、`Wildcards2Regex`) | nnmnkwii `nnmnkwii/io/hts.py` の `load_question_set` / `wildcards2regex` |
| `HtsLabel` (ラベルの読み込み、`Rounded`) | nnmnkwii `nnmnkwii/io/hts.py` の `HTSLabelFile.load` / `round_` |
| `LinguisticFeatures` (言語特徴量、コーステコーディング) | nnmnkwii `nnmnkwii/frontend/merlin.py` |
| `Scaler` (MinMax / Standard) | NNSVS `nnsvs/util.py` の `MinMaxScaler` / `StandardScaler` |

- nnmnkwii: https://github.com/r9y9/nnmnkwii (Copyright (c) 2017 Ryuichi Yamamoto)
- NNSVS: https://github.com/nnsvs/nnsvs (Copyright (c) 2020 Ryuichi Yamamoto)

`tests/OnnxSvs.Tests/data` の `.hed` とラベルは NNSVS のテスト用データです。
移植は Python のソースを読んで行い、OpenUtau の C# 実装 (`EnunuOnnx`) のコードは使っていません。
