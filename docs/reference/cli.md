# LiveCap CLI リファレンス

LiveCap CLI は音声文字起こしのためのコマンドラインインターフェースです。

## インストール

```bash
# 推奨セット
pip install livecap-cli[recommended]

# フル機能
pip install livecap-cli[all]

# 開発用
pip install -e ".[engines-torch,dev]"
```

## コマンド一覧

| コマンド | 説明 |
|---------|------|
| `livecap-cli info` | インストール診断情報を表示 |
| `livecap-cli devices` | オーディオ入力デバイス一覧を表示 |
| `livecap-cli engines` | 利用可能な ASR エンジン一覧を表示 |
| `livecap-cli translators` | 利用可能な翻訳器一覧を表示 |
| `livecap-cli transcribe` | 音声を文字起こし |

---

## `livecap-cli info`

インストール状態の診断情報を表示します。

```bash
# テキスト形式
livecap-cli info

# JSON 形式
livecap-cli info --as-json
```

### オプション

| オプション | 説明 |
|-----------|------|
| `--ensure-ffmpeg` | FFmpeg バイナリの検出/ダウンロードを試行 |
| `--as-json` | JSON 形式で出力 |

### 出力例

```
livecap-cli diagnostics:
  FFmpeg: /usr/bin/ffmpeg
  Models root: /home/user/.cache/LiveCap/models
  Cache root: /home/user/.cache/LiveCap/cache
  CUDA available: yes (NVIDIA GeForce RTX 4090)
  VAD backends: silero, tenvad, webrtc
  ASR engines: reazonspeech, whispers2t, parakeet, parakeet_ja, canary, voxtral
  Translator: not registered (fallback only)
```

---

## `livecap-cli devices`

利用可能なオーディオ入力デバイスを一覧表示します。

```bash
livecap-cli devices
```

### 出力例

```
[0] HDA Intel PCH: ALC892 Analog (hw:0,0)
[1] USB Audio Device: USB Audio (hw:1,0) (default)
```

---

## `livecap-cli engines`

利用可能な ASR エンジンを一覧表示します。

```bash
livecap-cli engines
```

### 出力例

```
reazonspeech: ReazonSpeech K2 v2 [cpu, cuda]
whispers2t: WhisperS2T [cpu, cuda]
parakeet: NVIDIA Parakeet TDT 0.6B v2 [cpu, cuda]
parakeet_ja: NVIDIA Parakeet TDT CTC 0.6B JA [cpu, cuda]
canary: NVIDIA Canary 1B Flash [cpu, cuda]
voxtral: MistralAI Voxtral Mini 3B [cpu, cuda]
```

---

## `livecap-cli translators`

利用可能な翻訳器を一覧表示します。

```bash
livecap-cli translators
```

### 出力例

```
google: Google Translate
opus_mt: Helsinki-NLP Opus-MT
riva_instruct: NVIDIA Riva Translate 4B Instruct (GPU)
```

---

## `livecap-cli transcribe`

音声ファイルまたはマイク入力を文字起こしします。

### ファイル文字起こし

```bash
# 基本
livecap-cli transcribe input.mp4 -o output.srt

# エンジン指定（--device gpu は内部で cuda にマップ）
livecap-cli transcribe input.wav -o output.srt --engine whispers2t --device gpu

# 翻訳付き
livecap-cli transcribe input.mp4 -o output.srt --translate google --target-lang en

# 言語指定
livecap-cli transcribe input.mp4 -o output.srt --language ja
```

### リアルタイム文字起こし

```bash
# マイクから（デバイスID 0）
livecap-cli transcribe --realtime --mic 0

# エンジンとデバイス指定
livecap-cli transcribe --realtime --mic 0 --engine whispers2t --device gpu

# VAD バックエンド指定
livecap-cli transcribe --realtime --mic 0 --vad silero
```

### オプション

| オプション | 説明 | デフォルト |
|-----------|------|----------|
| `input_file` | 入力ファイル（ファイルモード時必須） | - |
| `-o`, `--output` | 出力 SRT ファイル（ファイルモード時必須） | - |
| `--realtime` | リアルタイムモードを有効化 | `False` |
| `--mic` | マイクデバイス ID（リアルタイム時必須） | - |
| `--engine` | ASR エンジン ID | `whispers2t` |
| `--device` | デバイス（`auto`/`gpu`/`cpu`） | `auto` |
| `--language` | 言語コード（例: `ja`, `en`） | `ja` |
| `--model-size` | WhisperS2T モデルサイズ | `base` |
| `--vad` | VAD バックエンド（`auto`/`silero`/`tenvad`/`webrtc`） | `auto` |
| `--translate` | 翻訳器 ID（例: `google`） | - |
| `--target-lang` | 翻訳先言語 | `en` |

### モデルサイズ（WhisperS2T）

| サイズ | VRAM | 説明 |
|--------|------|------|
| `tiny` | ~1GB | 高速、低精度 |
| `base` | ~1GB | バランス型（デフォルト） |
| `small` | ~2GB | 中精度 |
| `medium` | ~5GB | 高精度 |
| `large-v3` | ~10GB | 最高精度 |
| `large-v3-turbo` | ~6GB | 高速・高精度 |

---

## 使用例

### 基本的な文字起こし

```bash
# 動画ファイルを日本語で文字起こし
livecap-cli transcribe meeting.mp4 -o meeting.srt --language ja

# 英語音声を GPU で処理
livecap-cli transcribe podcast.wav -o podcast.srt --language en --device gpu
```

### 高精度モデルの使用

```bash
# Whisper Large-v3 を使用
livecap-cli transcribe interview.mp4 -o interview.srt \
  --engine whispers2t --model-size large-v3 --device gpu
```

### 翻訳付き文字起こし

```bash
# 日本語音声を英語字幕に
livecap-cli transcribe japanese_video.mp4 -o english_subtitles.srt \
  --language ja --translate google --target-lang en
```

### リアルタイム会議録

```bash
# マイク 0 から日本語でリアルタイム文字起こし
livecap-cli transcribe --realtime --mic 0 --language ja --engine whispers2t

# Ctrl+C で停止
```

---

## 環境変数

| 変数 | 説明 | デフォルト |
|------|------|----------|
| `LIVECAP_CORE_MODELS_DIR` | モデルキャッシュディレクトリ | `appdirs.user_cache_dir("LiveCap", "PineLab")/models` |
| `LIVECAP_CORE_CACHE_DIR` | 一般キャッシュディレクトリ | `appdirs.user_cache_dir("LiveCap", "PineLab")/cache` |
| `LIVECAP_FFMPEG_BIN` | FFmpeg バイナリディレクトリ | システム PATH |

> **Note**: appdirs がない場合は `~/.livecap/{models,cache}` にフォールバック。
> Linux: `~/.cache/LiveCap/...`、macOS: `~/Library/Caches/LiveCap/...`、Windows: `%LOCALAPPDATA%\LiveCap\Cache\...`

---

## 終了コード

| コード | 説明 |
|--------|------|
| `0` | 成功 |
| `1` | エラー（依存関係不足、ファイル未発見など） |

---

## 関連ドキュメント

- [リアルタイム文字起こしガイド](../guides/realtime-transcription.md)
- [API 仕様書](../architecture/core-api-spec.md)
- [機能一覧](feature-inventory.md)
