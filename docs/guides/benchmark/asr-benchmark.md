# ASR Benchmark Guide

ASRエンジンの精度・性能を評価するためのベンチマークツールです。

## クイックスタート

```bash
# 依存関係のインストール
uv sync --extra benchmark --extra engines-torch

# クイックベンチマーク実行（約2分）
uv run python -m benchmarks.asr --mode quick

# 結果確認
ls benchmark_results/
```

## データセット準備

### 実行モードとデータソース

| モード | データソース | ファイル数 | 用途 |
|--------|-------------|-----------|------|
| `debug` | `tests/assets/audio/` | 1/言語 | CIスモークテスト |
| `quick` | `tests/assets/prepared/` | 30/言語 | 高速ベンチマーク |
| `standard` | `tests/assets/prepared/` | 100/言語 | 標準評価 |
| `full` | `tests/assets/prepared/` | 全ファイル | 本格ベンチマーク |

### データセット構造

```
tests/assets/
├── audio/          # git追跡（debug用）
│   ├── ja/         # 日本語サンプル
│   └── en/         # 英語サンプル
├── prepared/       # git無視（standard/full用）
│   ├── ja/         # JSUT変換データ
│   └── en/         # LibriSpeech変換データ
└── source/         # git無視（元コーパス）
    ├── jsut/       # JSUT corpus
    └── librispeech/ # LibriSpeech test-clean
```

### データセット生成

`quick`/`standard`/`full`モードでは、事前にベンチマークデータを準備する必要があります。

#### 1. ソースコーパスのダウンロード

```bash
# 自動ダウンロード（推奨）
uv run python scripts/download_benchmark_assets.py

# 手動ダウンロードの場合
# - JSUT: https://sites.google.com/site/shinnosuketakamichi/publication/jsut
# - LibriSpeech: https://www.openslr.org/12 (test-clean)
```

#### 2. データ変換

```bash
# standardモード用（100ファイル/言語）
uv run python scripts/prepare_benchmark_data.py --mode standard

# fullモード用（全ファイル）
uv run python scripts/prepare_benchmark_data.py --mode full

# カスタム制限
uv run python scripts/prepare_benchmark_data.py --ja-limit 500 --en-limit 200
```

**出力形式**: WAV, 16kHz, mono, 16bit, -1dBFS peak normalized

## CLI オプション

```bash
python -m benchmarks.asr [OPTIONS]
```

### 基本オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--mode` | 実行モード（debug/quick/standard/full） | `quick` |
| `--language`, `-l` | 評価言語（複数指定可） | `ja en` |
| `--engine`, `-e` | 評価エンジン（複数指定可） | モード依存 |
| `--runs`, `-r` | RTF測定の繰り返し回数 | `1` |
| `--device` | 使用デバイス（cuda/cpu） | `cuda` |
| `--output-dir`, `-o` | 結果出力ディレクトリ | `benchmark_results/` |
| `--verbose`, `-v` | 詳細ログ出力 | `false` |

### 使用例

```bash
# 日本語のみ、特定エンジン
uv run python -m benchmarks.asr --language ja --engine parakeet_ja whispers2t_large_v3

# 標準モード、3回測定（RTF精度向上）
uv run python -m benchmarks.asr --mode standard --runs 3

# CPUで実行
uv run python -m benchmarks.asr --mode quick --device cpu

# カスタム出力先
uv run python -m benchmarks.asr --mode standard --output-dir ./my_results
```

## 評価指標

### 精度指標

| 指標 | 説明 | 対象言語 |
|------|------|---------|
| **WER** | Word Error Rate（単語誤り率） | 英語など |
| **CER** | Character Error Rate（文字誤り率） | 日本語など |

- 値が小さいほど高精度
- 0% = 完全一致、100% = 全エラー
- 日本語はCER、英語はWERを主指標として使用

#### 採点の前の正規化

正解と ASR の出力の両方に同じ規則をかけてから比べる ([`benchmarks/common/text_normalization.py`](../../../benchmarks/common/text_normalization.py)、[#488](https://github.com/Mega-Gorilla/livecap-cli/issues/488))。

| 対象 | 規則 |
|------|------|
| 共通 | HTML 実体参照を戻す、NFKC で文字幅をそろえる、音の表記 (`[...]` `♪`、括弧は中身が既知の語だけの `(笑)` `(laughs)` `(male)`)・CSJ 形式のフィラー `(F えー)`・話者交代の印 (`>>`、行頭の `-`) を除く。話者名 (`トミー：`) は、`speaker_labels` で名前を渡したときだけ除く (印の直後でも推定では消さない。`午前10:30` や `>> Answer: ...` のような発話まで消えるため) |
| 日本語 (CER) | 句読点・記号・空白を消し、英字を小文字にする。**数字は変換しない** (漢数字と算用数字の揺れは誤りに数える) |
| 英語 (WER) | Whisper 標準の英語の正規化 (transformers 5.22 の `EnglishTextNormalizer`)。小文字化、括弧の中身とフィラー (um / uh) の除去、縮約・数字・英米の綴りの統一 |
| その他の言語 | 小文字にし、句読点と記号を消す |

#### 誤りの内訳と、並べて出す値

長尺の字幕 (YODAS2) の採点では、割合 1 つでは見えないものを、[`benchmarks/common/error_breakdown.py`](../../../benchmarks/common/error_breakdown.py) で並べて出す ([#488](https://github.com/Mega-Gorilla/livecap-cli/issues/488))。`count_errors()` が置換・脱落・挿入の件数を返し、`variant` で比べ方を選ぶ。単位は、日本語が文字、それ以外が語。

| `variant` | そろえること | 何が分かるか |
|------|------|------|
| `plain` | 上の正規化だけ (`calculate_cer()` / `calculate_wer()` と同じ割合) | 主の値 (短文の読み上げ) |
| `fillers` | 片方にしか無い口ぐせ (えー、えっと、あのー、まあ、なんか) を数えない | 口ぐせを書くか省くかの流儀を除いた精度。**長尺の字幕では主の値** |
| `digits` | 日本語の中の算用数字を漢数字に直す (`3つ` と `三つ`) | 数字の書き方を除いた精度 |
| `kana` | `digits` に加えて読みに直す (`出来る` と `できる`) | 漢字とかなの書き分けを除いた精度 |

- `fillers` は、口ぐせの語を文から消さない。字幕と ASR を突き合わせたあとで、**片方にしか無い部分が口ぐせだけ**のときに外す。「あの人」の「あの」のように両方にある語は、一致として残る。字幕にしか無い口ぐせは、正解の長さからも外す
- 限界: 片方にしか無い「あの」「まあ」「なんか」は、口ぐせでなくても外れる (ASR が「あの人」の「あの」を落とした場合など)。1 文字の「え」「あ」と、「その」「こう」「ちょっと」は口ぐせに入れていない
- `fillers` `digits` `kana` は日本語だけの区別。英語の口ぐせと数字は、上の正規化が先にそろえている
- 内訳の読み方: 字幕が言葉を省いたり要約したりしていると**挿入**が増え、ASR が言葉を落とすと**脱落**が増える
- 複数の動画は `aggregate()` で件数を足し、全体を合わせた値 (誤りの総数 ÷ 正解の総数) にする。量の多い話者に引きずられていないかは、`macro_average()` (話者ごとの割合の平均) を並べて見る
- `digits` と `kana` は dev の依存 (`kanjize`、`pykakasi`) を使う

### 性能指標

| 指標 | 説明 | 目安 |
|------|------|------|
| **RTF** | Real-Time Factor（処理時間/音声長） | < 1.0 = リアルタイム超 |
| **VRAM** | ピークGPUメモリ使用量 | 低いほど軽量 |

### RTF の解釈

- RTF = 0.1: 10秒の音声を1秒で処理（10倍速）
- RTF = 1.0: リアルタイム処理
- RTF = 2.0: 10秒の音声を20秒で処理（半分速）

## デフォルトエンジン

モードごとのデフォルトエンジン:

| 言語 | debug/quick/standard | full |
|------|---------------------|------|
| 日本語 | parakeet_ja, whispers2t_large_v3 | 全対応エンジン |
| 英語 | parakeet, whispers2t_large_v3 | 全対応エンジン |

利用可能なエンジン一覧:
```bash
uv run python -c "from livecap_cli import EngineMetadata; print('\n'.join(EngineMetadata.get_all().keys()))"
```

## 結果の解釈

### 出力ファイル

```
benchmark_results/YYYYMMDD_HHMMSS_MODE/
├── summary.md          # Markdown形式のサマリー
└── raw/
    ├── engine1_ja.csv  # 個別結果（エンジン×言語）
    └── engine1_en.csv
```

### summary.md の読み方

```markdown
## Results by Language

### JA
| Engine              | CER   | WER    | RTF   | VRAM    | Files |
|---------------------|-------|--------|-------|---------|-------|
| parakeet_ja         | 5.2%  | 12.3%  | 0.089 | 2048MB  | 100   |
| whispers2t_large_v3 | 3.8%  | 9.1%   | 0.156 | 4096MB  | 100   |

**Best CER:** whispers2t_large_v3 (3.8%)
```

- **CER/WER**: 低いほど高精度
- **RTF**: 低いほど高速
- **VRAM**: 低いほど省メモリ
- **Files**: 評価ファイル数

### CSVファイルの構造

```csv
file_id,reference,transcript,cer,wer,rtf,duration_sec
jsut_basic5000_0001,水をマレーシアから買わなくてはならない,水をマレーシアから買わなくてはならない,0.0000,0.0000,0.0890,3.52
```

## CI/GitHub Actions での実行

### ワークフローディスパッチ

GitHub Actions から手動実行できます:

1. Actions タブ → "ASR Benchmark" 選択
2. "Run workflow" クリック
3. パラメータを設定:
   - **mode**: debug/quick/standard/full
   - **languages**: ja,en（カンマ区切り）
   - **engine_preset**: default/all/custom
   - **engines**: custom選択時のエンジン一覧
   - **runs**: RTF測定回数

### 入力パラメータ

| パラメータ | 説明 | デフォルト |
|-----------|------|-----------|
| `mode` | 実行モード | `quick` |
| `languages` | 評価言語（カンマ区切り） | `ja,en` |
| `engine_preset` | エンジン選択方式 | `default` |
| `engines` | カスタムエンジン（preset=custom時） | - |
| `runs` | RTF測定回数 | `1` |

### 結果の確認

- **Step Summary**: ワークフロー実行結果に summary.md が表示
- **Artifacts**: `benchmark-results-*` としてダウンロード可能（90日間保持）

## 長尺の評価 (YODAS2 日本語)

配信に近い長い音声 (1 本 5〜30 分) で測る評価 ([`benchmarks/longform/`](../../../benchmarks/longform/)、[#488](https://github.com/Mega-Gorilla/livecap-cli/issues/488))。データの作り方と限界は [YODAS2 長尺評価データ](./yodas2-longform-dataset.md) を参照。

```bash
# 1. 音声を用意する (最終評価用 31 本、16 束)
python scripts/prepare_yodas2_audio.py --split test

# 2. 書き起こす (ASR ごとに別の process で回すとよい。同じ設定なら、書き起こし済みの動画は飛ばす)
python -m benchmarks.longform transcribe --split test --engine qwen3asr_large --device gpu
python -m benchmarks.longform transcribe --split test --engine whispers2t:large-v3 --device gpu

# 3. 採点して benchmark_results/longform/test/summary.{json,md} を書く
python -m benchmarks.longform score --split test
```

| 項目 | 内容 |
|------|------|
| 書き起こし | **CLI の `livecap-cli transcribe <file>` と同じ部品・同じ既定値**: VAD で区切る (engine 別の preset、1 秒ずつ流す) → EnergyGate → 認識 → confidence filter。既定値は `livecap_cli.cli.build_parser()` から得るので、CLI の既定が変われば評価も追従する。違いは、モデルを 1 回だけ読み込んで動画を続けて処理することだけ |
| 再開 | 今回と同じ設定・同じ音声で書き起こしに成功した動画は飛ばす。動画ごとの記録に設定の識別子 (`settings_id`) と音声の SHA-256 を残して確かめるので、`--force` を途中で止めた残りや、中身の違う音声の記録は使い回さず、やり直す。`run.json` (その出力先で最後に始めた設定) が今回と違えば、engine を読み込む前に止まる (`--force` で全部やり直すか、`--out-dir` を分ける)。失敗した記録、読めない記録、採点に要る項目・型がそろわない記録もやり直し、終わったあとに失敗が残ればエラーで終える。採点も、`run.json` と違う設定の記録や項目の欠けた記録があれば止まる |
| `whispers2t` のサイズ | CLI の `--model-size` の既定は `base`。`--engine whispers2t:large-v3` のようにサイズを指定できる |
| 主の値 | **口ぐせを除き (`fillers`)、表記だけ違う区間 (コードや URL の読み上げなど) を除いた CER**。全体の値は誤りの総数 ÷ 正解の総文字数。話者ごとの CER の平均も並べる |
| 採点から外す区間 | 字幕の側で外す区間 (`excluded`、主の値では `notation` も) は、ASR の出力を時刻で切らずに字幕の全体と突き合わせ、外す区間の文字に当たった誤り (置換・脱落と、外す区間に接する挿入) を数えない (`count_errors_in_parts()`)。ASR の区間の切り方 (VAD の区間の長さ) に結果が依らない。外す区間に接する挿入を数えない分だけ近似 |
| 並べて出す値 | 置換・脱落・挿入の内訳、そのままの CER、数字をそろえた CER、読みで比べた CER、RTF、読み込み時間、GPU の使用量 (torch の分だけ)、区間の長さ、EnergyGate / filter で落ちた区間の数と長さの合計。区間の長さは、全部の動画に下の `vad_segments` があれば VAD の区間 (落ちたものを含む)、無ければ ASR の出力の区間で、表に基準を書く。落ちた長さの合計は、全部の動画に `vad_segments` があるときだけ出す (一部なら「未集計」と本数) |
| 動画ごとの記録 | `<出力先>/<ASR>/<audio_id>.json`。出力の区間に加えて、VAD が切った区間ごとの時刻と結果 (`vad_segments`: `asr` / `empty` / `energy_gate:*` / `confidence_filter:*`) を残す。書き起こしの本文を含むので commit しない (`.gitignore` で除く) |
| VAD の区間の長さの上限 (実験) | `transcribe --max-speech-ms 20000` のように渡すと、CLI と同じ preset の VAD の `max_speech_ms` だけを置き換える。結果は `<engine>-max<N>ms` (例: `qwen3asr-max20000ms`) に分けて保存し、実際に使った VAD の設定と backend を `run.json` に残す。CLI にこの選択肢は無い |

### 結果 (2026-11-18、最終評価用 31 本・8 人・5.7 時間)

[`benchmark_results/longform/test/summary.md`](../../../benchmark_results/longform/test/summary.md) (動画ごと・話者ごとの件数は `summary.json`)。Ubuntu・RTX 4070 Ti、CLI の既定値 (`--vad auto`、confidence filter `on`)。

| ASR | CER (主) | 置換 | 脱落 | 挿入 | 話者平均 | RTF | ASR の出力の区間の長さ (中央 / 最長, 秒) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `whispers2t` large-v3 | **12.17** | 4.79 | 4.72 | 2.66 | 11.96 | 0.064 | 2.0 / 55 |
| `qwen3asr_large` (1.7B) | **12.50** | 6.63 | 2.52 | 3.35 | 11.84 | 0.100 | 2.1 / 51 |
| `qwen3asr` (0.6B、既定・推奨) | 18.05 | 7.88 | 6.83 | 3.34 | 17.46 | 0.088 | 3.8 / 193 |
| `reazonspeech` | 19.81 | 5.38 | 12.37 | 2.05 | 20.21 | 0.049 (CPU) | 2.3 / 73 |
| `parakeet_ja` | 31.83 | 3.39 | 26.48 | 1.95 | 30.90 | 0.009 | 14.0 / 540 |
| `whispers2t` base (**CLI の既定のサイズ**) | 37.69 | 19.53 | 5.75 | 12.41 | 35.54 | 0.015 | 2.0 / 55 |

読むときの注意:

- **データの候補選びと一部の字幕の直しに、Whisper large-v3・Qwen3-ASR 1.7B・ReazonSpeech の出力を使っている** ([YODAS2 長尺評価データ](./yodas2-longform-dataset.md) の 5 章・6 章)。この 3 つに有利な方向の偏りがありうる
- **VAD の区間が長くなる ASR ほど、脱落が多い。** 日本語の VAD preset には区間の長さの上限が無く (`max_speech_ms` = 0)、1 本 4 秒前後の短い読み上げ (JSUT) で調整されている。長い音声では区間が数分に伸び、`parakeet_ja` で最長 540 秒、`qwen3asr` で 193 秒になった。リアルタイムも同じ preset を使う
- CLI の `whispers2t` の既定のサイズは `base` で、large-v3 との差は 25 ポイント
- RTF は VAD と後段を含む処理時間 ÷ 音声の長さ。`reazonspeech` は sherpa-onnx の CPU 推論

## トラブルシューティング

### よくあるエラー

#### データセットが見つからない

```
DatasetError: Prepared directory not found: tests/assets/prepared/ja
```

**解決策**: データセットを準備してください
```bash
uv run python scripts/download_benchmark_assets.py
uv run python scripts/prepare_benchmark_data.py --mode standard
```

#### CUDAが利用できない

```
RuntimeError: CUDA not available
```

**解決策**: CPUモードで実行するか、CUDA環境を確認
```bash
# CPUで実行
uv run python -m benchmarks.asr --device cpu

# CUDA確認
python -c "import torch; print(torch.cuda.is_available())"
```

#### メモリ不足（OOM）

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**解決策**:
- より小さいエンジンを使用
- バッチサイズを減らす（エンジン依存）
- `--device cpu` で実行

#### エンジンのロードエラー

```
ImportError: No module named 'torch'
```

**解決策**: 必要な依存関係をインストール
```bash
uv sync --extra engines-torch --extra engines-nemo
```

### 環境変数

| 変数 | 説明 | デフォルト |
|------|------|-----------|
| `LIVECAP_JSUT_DIR` | JSUTコーパスのパス | `tests/assets/source/jsut/jsut_ver1.1` |
| `LIVECAP_LIBRISPEECH_DIR` | LibriSpeechのパス | `tests/assets/source/librispeech/test-clean` |
| `LIVECAP_CORE_MODELS_DIR` | モデルキャッシュ | OS依存 |

## 関連ドキュメント

### VAD リファレンス

- [VAD バックエンドリファレンス](../../reference/vad/backends.md) - 各バックエンドの詳細仕様
- [VADConfig リファレンス](../../reference/vad/config.md) - 共通パラメータの詳細
- [VAD バックエンド比較](../../reference/vad/comparison.md) - ベンチマーク結果

### ガイド

- [VAD Benchmark Guide](./vad-benchmark.md) - VADベンチマークガイド
- [VAD Optimization Guide](./vad-optimization.md) - VADパラメータ最適化ガイド
