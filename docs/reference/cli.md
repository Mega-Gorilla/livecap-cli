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

推奨閾値アルゴリズム: `peak_p95 (per-chunk |x|.max() の 95%ile) + 6 dB`（Issue #291）。NoiseGate の envelope follower (per-sample peak 追跡) と単位を揃えています。`danger_zone` (`floor ± 5 dB`) は RMS-unit の diagnostic で、手動閾値設定時の「死のゾーン」回避指針です。

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
| `--engine-min-rms-margin` | `suggested_engine_min_rms_dbfs` の safety margin (dB)。`+6` で `noise_rms_p95 + 6` を suggest。 (#292) | `6.0` |

### 出力例（対話モード）

```
Monitoring mic 0... Press Ctrl+C to stop.

  -60dB       -40dB       -20dB        0dB
    |           |           |           |
    ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░   -42.3 dB

Noise floor:    ~-74.2 dB (RMS 25%ile)
Noise RMS p95:  ~-58.5 dB (RMS 95%ile)
Peak p95:       ~-47.0 dB (|x|.max() 95%ile, threshold の基準)
Suggested --noise-gate-threshold: -41 dB (= peak_p95 + 6; per-sample peak unit)
Suggested --engine-min-rms:       -52 dB (= noise_rms_p95 + 6; RMS-unit, calibrated from chunk RMS p95; #292 EnergyGate)
  (Danger zone: -79 ~ -69 dB — RMS-unit; avoid manually setting thresholds here)
```

### 出力例（`--json`）

```json
{
  "noise_floor_db": -74.2,
  "noise_rms_p95_db": -58.5,
  "peak_p95_db": -47.0,
  "suggested_threshold_db": -41.0,
  "suggested_engine_min_rms_dbfs": -52.5,
  "danger_zone": [-79.2, -69.2],
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

### `suggested_threshold_db` の位置づけ

`suggested_threshold_db` は **「キャリブレーション済み出発点」** です。現行 NoiseGate (自動ヒステリシス + hard-mute) に対してほぼそのまま使える値ですが、以下のケースでは追加チューニングが有効です:

- **稀に残る fragmentation が気になる場合**: 既定 `release_ms=100` で多くの状況をカバー済み ([PR C 検証](../benchmarks/noise-gate-ab.md#6-pr-c-結果-既定-release_ms100-の採用後))。さらに余裕が欲しい場合は `--noise-gate-release 150` や `200` を試す
- **非常に静かな発話 / 極端な低 SNR 環境**: 手動で閾値を下げる
- **whisper 系エンジンで残留ハルシネーションが気になる場合**: `reazonspeech` / `parakeet_ja` / `qwen3asr` に切り替えると whisper 固有のバイアス問題を回避できます

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
| `--noise-gate-close-threshold` | ゲート閉鎖閾値 (dB)、ヒステリシス用。未指定なら open − 6 dB に自動解決 | `None` (auto) |
| `--noise-gate-attack` | アタック時間 (ms) | `0.5` |
| `--noise-gate-release` | リリース時間 (ms)。既定は PR C ([#283](https://github.com/Mega-Gorilla/livecap-cli/issues/283)) で `30 → 100` に変更 | `100` |
| `--noise-gate-floor` | ゲート閉鎖時の出力減衰 (dB)。未指定で hard-mute。`-60` を渡せば旧 soft-mute | `None` (hard-mute) |
| `--engine-min-rms` | EnergyGate threshold (per-segment RMS dBFS, #292)。`off` / `disabled` / `none` / `=-inf` で opt-out。**NoiseGate threshold とは物理量が異なる** | `-45.0` |
| `--engine-energy-metric` | EnergyGate の per-segment energy 指標。`max_frame_rms` (default、padding 希釈耐性) / `whole_rms` (aggressive) / `p95_frame_rms` (中庸) / `top3_frame_rms` (transient resistance) | `max_frame_rms` |
| `--engine-energy-frame-ms` | frame-based metric の窓長 (ms)。`whole_rms` では無視 | `32.0` |

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

ノイズゲートは VAD の前段で音量ベースのゲーティングを行い、環境ノイズによる VAD 誤検出（ハルシネーションの原因）を抑制します。既定で自動ヒステリシス (`threshold_db - 6 dB`) + hard-mute が有効化されており、flicker による擬似パルス生成を抑制します。numba JIT で高速化されており、実時間比で無視できるオーバーヘッドで動作します。

> **💡 アドバンスドチューニング**
>
> - **攻撃的な閾値 (speech peak 付近) を試す場合**: 既定 `--noise-gate-release 100` (PR C で `30 → 100` に変更済み) がほぼすべての状況をカバーします。さらに緩める場合は `150` や `200` を試す。旧挙動 `30` を明示する場合は `--noise-gate-release 30` を指定 ([PR C 検証データ](../benchmarks/noise-gate-ab.md))。
> - **旧挙動 (PR #281 までの単一閾値 + `-60 dB` soft-mute) を再現したい場合**: `--noise-gate-close-threshold` に `--noise-gate-threshold` と同じ値を渡し、`--noise-gate-floor -60` を指定します。併せて `--noise-gate-release 30` で短い release も再現。
> - **死のゾーン回避**: `levels` コマンドの `danger_zone` は **RMS-unit diagnostic**（chunk RMS の `noise_floor ± 5 dB`）で、手動で閾値をこの RMS 範囲に設定すると floor の揺らぎで gate がフリッカーします。`suggested_threshold_db` (peak-unit) はこのゾーンと単位が異なるため直接比較できませんが、出発点として安全に使えます。

### EnergyGate の限界と推奨運用 (#292)

`--engine-min-rms` (EnergyGate) は VAD を通過した segment のうち energy が低いものを engine に渡さず短絡することで、低 RMS / 純ノイズ segment に対する hallucination ("うん"/"ピッ"/"え?"/"どうぞ" 等) を削減します。ただし以下の **限界** があります:

- **Silver bullet ではない**。実音源プローブ (#292) では parakeet_ja が silent audio に対して 100% hallucinate し、`-45 dBFS` threshold (max_frame_rms) で削減できるのは 26% のみ (stress test, VAD bypass)。残りは noisy silence で max frame が threshold を上回る。
- **Engine 別 hallucination 傾向**: ReazonSpeech は同条件で hallucination が顕著に少なく、whispers2t / parakeet 系は出やすい。engine 選択も重要な判断材料。
- **物理量に注意**: `--engine-min-rms` は **per-segment / per-frame RMS** unit、`--noise-gate-threshold` は **per-sample peak envelope** unit。同じ値を共有してはいけません。`levels` コマンドが両方の suggested 値を別個に出力します。
- **`levels` の calibration 窓と EnergyGate の判定窓は異なる**: `levels` は `MicrophoneSource` chunk (典型 100 ms) の RMS p95 を計測しますが、EnergyGate default は per-segment max **32 ms frame** RMS を判定します。両者とも RMS unit (peak unit ではない) であり、観測対象の noise (ファン定常音等) では概ね近い値になりますが、impulsive noise が混入する環境では `suggested_engine_min_rms_dbfs` は若干 conservative 寄りの目安として扱ってください。実 EnergyGate と完全に同じ統計を取りたい場合は `--engine-energy-frame-ms 100` で window を揃える、あるいは production で微調整する運用が現実的です。

**推奨運用**:

1. `livecap-cli levels --mic 0 --duration 5 --json` で calibration し `suggested_threshold_db` (NoiseGate) と `suggested_engine_min_rms_dbfs` (EnergyGate) を取得
2. `--noise-gate --noise-gate-threshold <val1> --engine-min-rms <val2>` を併用 (NoiseGate = pre-VAD per-sample peak / EnergyGate = pre-engine per-segment RMS の相補的 2 層防御)
3. hallucination が依然残る場合は engine を ReazonSpeech 等の hallucination 耐性の高いものに切替

**実測ベースの削減率** (テスト結果.mov / 73 windows × 1s / parakeet_ja での 3 条件評価、#292 PR コメント参照):

| 条件 | Hallucination 残存 | 削減率 |
|---|---:|---:|
| なにもなし (baseline) | 73/73 (100 %) | — |
| EG `--engine-min-rms -45` (default) | 54/73 | **26 %** |
| EG `--engine-min-rms <calibrated>` (= -34.5 in this env) | 16/73 | **78 %** |
| NoiseGate + EG (両方 calibrated) | 0/73 | **100 %** |

→ **default は保守的設定** で、whisper recording / 遠距離マイク / 低 gain 環境を壊さない安全寄りの値です。**Hallucination が顕在化する環境では `levels` calibration が 2-3 倍の効果**を発揮します。最強防御は **NoiseGate + EnergyGate 両方を calibration 値で併用**。

**なぜ default は -45 dBFS のままか**:
- ささやき声 (max_frame_rms ≈ -40 dBFS) を false-drop しない境界
- 環境 (noise floor) は user ごとに大きく異なるため、universal な値は存在しない
- 「**conservative default + `levels` で aggressive 化**」のアーキテクチャ (silent failure を作らない)

**Metric の選択** (`--engine-energy-metric`):

| 用途 | 推奨 metric |
|---|---|
| 通常 production (VAD default threshold) | `max_frame_rms` (default、padding 希釈耐性) |
| Stress test / VAD threshold を下げて使用 | `whole_rms` (aggressive、実測で 75 % 削減 @ -45 dBFS) |
| 単発 transient false-pass を抑えたい | `top3_frame_rms` |

`--engine-min-rms off` で完全 opt-out できます (whisper 録音などで小さい音を確実に拾いたい場合)。

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
