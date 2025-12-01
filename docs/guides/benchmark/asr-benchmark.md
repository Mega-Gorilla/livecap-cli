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
uv run python -c "from engines.metadata import EngineMetadata; print('\n'.join(EngineMetadata.get_all().keys()))"
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
ImportError: silero_vad is required
```

**解決策**: 必要な依存関係をインストール
```bash
uv sync --extra vad --extra engines-torch --extra engines-nemo
```

### 環境変数

| 変数 | 説明 | デフォルト |
|------|------|-----------|
| `LIVECAP_JSUT_DIR` | JSUTコーパスのパス | `tests/assets/source/jsut/jsut_ver1.1` |
| `LIVECAP_LIBRISPEECH_DIR` | LibriSpeechのパス | `tests/assets/source/librispeech/test-clean` |
| `LIVECAP_CORE_MODELS_DIR` | モデルキャッシュ | OS依存 |

## 関連ドキュメント

- [VAD Benchmark Guide](./vad-benchmark.md) - VADベンチマークガイド
- [VAD Optimization Guide](./vad-optimization.md) - VADパラメータ最適化ガイド
