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
| `livecap-cli levels` | マイク入力レベルを監視しノイズゲート推奨閾値を算出 |
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
  ASR engines: reazonspeech, whispers2t, parakeet, parakeet_ja, canary, voxtral, qwen3asr
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

## `livecap-cli levels`

マイク入力の dB レベルをリアルタイム表示し、環境ノイズから `--noise-gate-threshold` の推奨値を算出します（Issue #278, #280）。

推奨閾値アルゴリズム: `noise_peak (95パーセンタイル) + 10 dB`。`±5 dB` の「死のゾーン」を避けるため、安全マージン側の保守的な値を推奨します（根拠: livecap-gui PR #294 実測）。

### 使用例

```bash
# 対話モード（Ctrl+C で停止、バーチャート表示）
livecap-cli levels --mic 0

# 指定秒数で自動停止
livecap-cli levels --mic 0 --duration 5

# JSON 出力（GUI / スクリプト連携向け）
livecap-cli levels --mic 0 --duration 5 --json
```

### オプション

| オプション | 説明 | デフォルト |
|-----------|------|----------|
| `--mic` | マイクデバイス ID | `0` |
| `--duration` | N 秒後に自動停止（未指定時は Ctrl+C まで） | `None` |
| `--json` | `NoiseAnalysis` を JSON で stdout に出力（バーチャート抑制） | `False` |

### 出力例（対話モード）

```
Monitoring mic 0... Press Ctrl+C to stop.

  -60dB       -40dB       -20dB        0dB
    |           |           |           |
    ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░   -42.3 dB

Noise floor: ~-74.2 dB (25%ile)
Noise peak:  ~-58.5 dB (95%ile)
Suggested --noise-gate-threshold: -49 dB
  (Danger zone: -79 ~ -69 dB — avoid thresholds here)
```

### 出力例（`--json`）

```json
{
  "noise_floor_db": -74.2,
  "noise_peak_db": -58.5,
  "suggested_threshold_db": -48.5,
  "danger_zone": [-79.2, -69.2],
  "safe_zone_min_db": -53.5,
  "sample_count": 50,
  "duration_s": 5.0
}
```

### 推奨ワークフロー

```bash
# Step 1: 推奨閾値を取得
THRESHOLD=$(livecap-cli levels --mic 0 --duration 5 --json | jq -r .suggested_threshold_db)

# Step 2: その値で実運用
livecap-cli transcribe --realtime --mic 0 \
  --noise-gate --noise-gate-threshold "$THRESHOLD"
```

### ⚠️ `suggested_threshold_db` の位置づけ

`suggested_threshold_db` は **「キャリブレーション済み出発点」** として設計された値で、環境ノイズから算出された **本命アルゴリズム後の推奨値** です。ただし現行 NoiseGate 本体はまだ先行導入段階です:

| | 現行 (PR #279 + PR #281) | Follow-up (Issue #280 の PR B で対応予定) |
|---|---|---|
| 閾値 | 単一閾値 | **ヒステリシス** (open/close 別閾値) |
| ゲート閉鎖時 | soft-mute (`-60 dB` に減衰) | **hard-mute** (完全無音) |
| Flicker 対策 | 無し | ヒステリシス + hard-mute で解決 |

このため、現行実装では環境や発話音量によっては推奨値の追加調整が必要です:

- **クリーンな環境 / 大きめの声量**: 推奨値そのままで OK
- **ノイジーな環境 / 小さめの声量**: 推奨値より **低め** (より保守的) に調整
  - 例: 推奨値 `-17 dB` で発話がゲート閉鎖される場合 → `-25 dB` や `-35 dB` に下げる
- **whisper (whispers2t) エンジン利用時**: 特に影響を受けやすい。
  推奨値が speech peak 付近の場合、gate flicker によるハルシネーション (「ご視聴ありがとうございました」等) が発生することがあります。この場合は閾値を下げるか、`reazonspeech` / `parakeet_ja` / `qwen3asr` 等の別エンジンを検討してください。

詳細な背景は [Issue #280](https://github.com/Mega-Gorilla/livecap-cli/issues/280) を参照。

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
qwen3asr: Qwen3-ASR 0.6B [cpu, cuda]
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
| `--noise-gate` | ノイズゲートを有効化（VAD 前段で環境ノイズを減衰） | `False` |
| `--noise-gate-threshold` | ゲート開放閾値 (dB)。`levels` コマンドで推奨値を算出可能 | `-35` |
| `--noise-gate-attack` | アタック時間 (ms) | `0.5` |
| `--noise-gate-release` | リリース時間 (ms) | `30` |

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

### ノイズゲート併用（環境ノイズ対策）

```bash
# Step 1: 環境ノイズを計測して推奨閾値を取得
livecap-cli levels --mic 0 --duration 5

# Step 2: 推奨値をそのまま適用してリアルタイム文字起こし
livecap-cli transcribe --realtime --mic 0 \
  --noise-gate --noise-gate-threshold -49
```

ノイズゲートは VAD の前段で音量ベースのゲーティングを行い、環境ノイズによる VAD 誤検出（ハルシネーションの原因）を抑制します。numba JIT で高速化されており、実時間比で無視できるオーバーヘッドで動作します。

> **⚠️ 現行 NoiseGate は先行導入段階です**
> 現行実装は単一閾値 + soft-mute で構成されており、閾値が speech peak 付近にある場合は flicker によって **逆にハルシネーションを増やす** ことがあります ([Issue #280](https://github.com/Mega-Gorilla/livecap-cli/issues/280))。
>
> **エンジン別の実地ガイダンス**:
> - **`whispers2t`**: YouTube dataset バイアスにより flicker で「ご視聴ありがとう...」等のハルシネーションが発生しやすい。`levels` の推奨値ではなく **保守的な値** (例: `noise_floor + 5 dB`) を使うか、別エンジンを推奨。
> - **`reazonspeech` / `parakeet_ja` / `qwen3asr`**: CTC / 別アーキテクチャのため flicker への耐性が高く、`levels` の推奨値をそのまま使用可。
>
> [Issue #280](https://github.com/Mega-Gorilla/livecap-cli/issues/280) の follow-up (PR B: ヒステリシス + hard-mute) 完了後は、推奨値 (`noise_peak + 10 dB`) がより安定して使えるようになります。

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
