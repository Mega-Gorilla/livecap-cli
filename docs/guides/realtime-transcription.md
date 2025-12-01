# リアルタイム文字起こしガイド

> **対象バージョン:** livecap-core 1.0.0+
> **作成日:** 2025-11-25
> **関連:** [Phase 1 実装計画](../planning/phase1-implementation-plan.md)

このガイドでは、livecap-core のリアルタイム文字起こし機能の使い方を説明します。

---

## 概要

Phase 1 で実装されたリアルタイム文字起こし機能は、以下のコンポーネントで構成されています：

| コンポーネント | 役割 |
|---------------|------|
| `StreamTranscriber` | VAD + ASR を組み合わせたストリーミング処理 |
| `VADProcessor` | Silero VAD を使った音声活動検出 |
| `AudioSource` | 音声入力の抽象化（FileSource, MicrophoneSource） |

```
AudioSource ──▶ StreamTranscriber ──▶ TranscriptionResult
                     │
                     ├── VADProcessor (音声セグメント検出)
                     │
                     └── Engine (ASR 推論)
```

---

## クイックスタート

### インストール

```bash
# 基本インストール（VAD バックエンド含む）
pip install livecap-core

# PyTorch エンジン付き（WhisperS2T, ReazonSpeech）
pip install livecap-core[engines-torch]

# NeMo エンジン付き（Parakeet）
pip install livecap-core[engines-nemo]
```

> **Note**: VAD バックエンド（Silero, WebRTC, TenVAD）はデフォルト依存関係に含まれています。

### 最小構成

```python
from livecap_core import StreamTranscriber, FileSource
from engines import EngineFactory

# エンジン初期化
engine = EngineFactory.create_engine("whispers2t_base", "cuda")
engine.load_model()

# 文字起こし
with StreamTranscriber(engine=engine) as transcriber:
    with FileSource("audio.wav") as source:
        for result in transcriber.transcribe_sync(source):
            print(f"{result.text}")
```

---

## 詳細な使い方

### 1. 同期 API（transcribe_sync）

最もシンプルな使い方です。

```python
from livecap_core import StreamTranscriber, FileSource
from engines import EngineFactory

engine = EngineFactory.create_engine("whispers2t_base", "cuda")
engine.load_model()

with StreamTranscriber(engine=engine, source_id="my-source") as transcriber:
    with FileSource("audio.wav") as source:
        for result in transcriber.transcribe_sync(source):
            print(f"[{result.start_time:.2f}s - {result.end_time:.2f}s] {result.text}")
            print(f"  確信度: {result.confidence:.2f}")
            print(f"  ソース: {result.source_id}")
```

### 2. 非同期 API（transcribe_async）

asyncio を使った非同期処理に対応しています。

```python
import asyncio
from livecap_core import StreamTranscriber, MicrophoneSource
from engines import EngineFactory

async def realtime_transcribe():
    engine = EngineFactory.create_engine("whispers2t_base", "cuda")
    engine.load_model()

    transcriber = StreamTranscriber(engine=engine)

    async with MicrophoneSource(device_id=0) as mic:
        async for result in transcriber.transcribe_async(mic):
            print(f"{result.text}")

    transcriber.close()

asyncio.run(realtime_transcribe())
```

### 3. コールバック API（feed_audio + callbacks）

低レベル API を使って、より細かい制御が可能です。

> **注意**: `feed_audio()` は VAD でセグメントを検出した際に `engine.transcribe()` を呼び出すため、ブロッキングが発生します（数十ms〜数百ms）。完全な非同期処理が必要な場合は `transcribe_async()` を使用してください。

```python
from livecap_core import StreamTranscriber, FileSource
from engines import EngineFactory

engine = EngineFactory.create_engine("whispers2t_base", "cuda")
engine.load_model()

# コールバック関数
def on_result(result):
    print(f"[確定] {result.text}")

def on_interim(interim):
    print(f"[途中経過] {interim.text}")

# トランスクライバー設定
transcriber = StreamTranscriber(engine=engine)
transcriber.set_callbacks(
    on_result=on_result,
    on_interim=on_interim,
)

# 音声を手動で入力
with FileSource("audio.wav") as source:
    for chunk in source:
        transcriber.feed_audio(chunk, source.sample_rate)

# 最終セグメントを処理
final = transcriber.finalize()
if final:
    print(f"[最終] {final.text}")

transcriber.close()
```

---

## VAD 設定

### VADProcessor の概要

`VADProcessor` は音声活動検出（Voice Activity Detection）を行うクラスです。以下の方法で設定できます：

| 方法 | 用途 | コード例 |
|------|------|---------|
| `from_language()` | 言語に最適化（推奨） | `VADProcessor.from_language("ja")` |
| デフォルト | 汎用（Silero VAD） | `VADProcessor()` |
| `config=` | パラメータ調整 | `VADProcessor(config=VADConfig(...))` |
| `backend=` | バックエンド指定 | `VADProcessor(backend=WebRTCVAD(...))` |
| 組み合わせ | 完全カスタマイズ | `VADProcessor(config=..., backend=...)` |

```python
from livecap_core import VADProcessor, VADConfig
from livecap_core.vad.backends import WebRTCVAD, TenVAD

# 言語に最適化（推奨）
vad = VADProcessor.from_language("ja")  # 日本語 → TenVAD

# デフォルト（Silero VAD）
vad = VADProcessor()

# パラメータ調整
vad = VADProcessor(config=VADConfig(threshold=0.7, min_speech_ms=300))

# バックエンド指定
vad = VADProcessor(backend=WebRTCVAD(mode=3))

# 完全カスタマイズ
vad = VADProcessor(
    config=VADConfig(min_speech_ms=300, min_silence_ms=150),
    backend=WebRTCVAD(mode=1),
)
```

---

### 言語別 VAD 最適化（推奨）

`VADProcessor.from_language()` を使うと、ベンチマーク結果に基づいて言語に最適な VAD バックエンドとパラメータが自動選択されます。

```python
from livecap_core import StreamTranscriber, VADProcessor
from engines import EngineFactory

# 1. 言語に最適化された VAD を作成
vad = VADProcessor.from_language("ja")  # 日本語 → TenVAD

# 2. エンジンを作成
engine = EngineFactory.create_engine("parakeet_ja", device="cuda")
engine.load_model()

# 3. StreamTranscriber に VAD を注入
with StreamTranscriber(engine=engine, vad_processor=vad) as transcriber:
    with FileSource("audio.wav") as source:
        for result in transcriber.transcribe_sync(source):
            print(f"{result.text}")
```

#### サポート言語と推奨 VAD

| 言語 | コード | 推奨 VAD | スコア | 備考 |
|------|--------|----------|--------|------|
| 日本語 | `ja` | TenVAD | 7.2% CER | Silero 比 -1.9% 改善 |
| 英語 | `en` | WebRTC | 3.3% WER | Silero 比 -2.6% 改善 |

> **Note**: スコアは [Issue #126](https://github.com/Mega-Gorilla/livecap-cli/issues/126) の Phase D ベンチマーク結果に基づいています。

#### エラーハンドリング

```python
from livecap_core import VADProcessor

try:
    vad = VADProcessor.from_language("zh")  # 未サポート言語
except ValueError as e:
    print(f"Error: {e}")
    # "No optimized preset for language 'zh'. Supported languages: en, ja.
    #  Use VADProcessor() for default Silero VAD."

    # フォールバック: デフォルトの Silero VAD を使用
    vad = VADProcessor()
```

#### TenVAD のライセンス警告

日本語で TenVAD を使用する場合、初回起動時にライセンス警告が表示されます：

```
UserWarning: TenVAD is licensed under LGPL-2.1. See https://github.com/AgoraIO-Extensions/AgoraVAD
```

これは TEN Framework のライセンス要件に基づく警告であり、動作には影響しません。警告を抑制したい場合：

```python
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    vad = VADProcessor.from_language("ja")
```

---

### バックエンドの選択

3 種類の VAD バックエンドから選択できます。

| バックエンド | 特徴 | 推奨用途 |
|-------------|------|---------|
| `SileroVAD` | 高精度、ニューラルネット | 汎用（デフォルト） |
| `WebRTCVAD` | 高速、低メモリ | 英語、リソース制約環境 |
| `TenVAD` | 日本語向け高精度 | 日本語 |

```python
from livecap_core import VADProcessor
from livecap_core.vad.backends import SileroVAD, WebRTCVAD, TenVAD

# Silero VAD（デフォルト）
vad = VADProcessor()  # または VADProcessor(backend=SileroVAD())

# WebRTC VAD
vad = VADProcessor(backend=WebRTCVAD(mode=3, frame_duration_ms=30))

# TenVAD
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    vad = VADProcessor(backend=TenVAD(hop_size=256))
```

#### バックエンド固有パラメータ

| バックエンド | パラメータ | 説明 |
|-------------|-----------|------|
| `SileroVAD` | `threshold` | 音声検出閾値（0.0-1.0） |
| `SileroVAD` | `onnx` | ONNX モデル使用（デフォルト: True） |
| `WebRTCVAD` | `mode` | 攻撃性（0=緩, 3=厳格） |
| `WebRTCVAD` | `frame_duration_ms` | フレーム長（10, 20, 30ms） |
| `TenVAD` | `hop_size` | フレームホップサイズ |

> **Note**: 詳細は [VAD バックエンド比較](../reference/vad-comparison.md) を参照してください。

---

### パラメータのカスタマイズ

`VADConfig` でセグメント検出のパラメータを調整できます。

#### デフォルト値

```python
from livecap_core import VADConfig

config = VADConfig()
print(config.threshold)        # 0.5  - 音声検出閾値
print(config.min_speech_ms)    # 250  - 最小音声継続時間（ms）
print(config.min_silence_ms)   # 100  - 無音判定時間（ms）
print(config.speech_pad_ms)    # 100  - 発話前後のパディング（ms）
```

#### 環境別の設定例

```python
from livecap_core import VADProcessor, VADConfig

# ノイズ環境向け（厳しめ）
noisy_config = VADConfig(
    threshold=0.7,           # 高めの閾値
    min_speech_ms=300,       # 長めの最小音声時間
    min_silence_ms=200,      # 長めの無音判定時間
)
vad = VADProcessor(config=noisy_config)

# 静かな環境向け（緩め）
quiet_config = VADConfig(
    threshold=0.3,           # 低めの閾値
    min_speech_ms=150,       # 短めの最小音声時間
    min_silence_ms=80,       # 短めの無音判定時間
)
vad = VADProcessor(config=quiet_config)
```

#### 中間結果の設定

長い発話中に途中経過を受け取りたい場合：

```python
config = VADConfig(
    interim_min_duration_ms=2000,  # 2秒以上で中間結果送信
    interim_interval_ms=1000,      # 1秒ごとに送信
)
```

---

## 音声ソース

### FileSource（テスト・デバッグ用）

```python
from livecap_core import FileSource

# 基本的な使い方
with FileSource("audio.wav") as source:
    for chunk in source:
        print(f"チャンクサイズ: {len(chunk)}")

# カスタム設定
source = FileSource(
    "audio.wav",
    sample_rate=16000,   # 出力サンプルレート（自動リサンプリング）
    chunk_ms=100,        # チャンクサイズ（ミリ秒）
    realtime=False,      # リアルタイムシミュレーション無効
)
```

### MicrophoneSource（マイク入力）

```python
from livecap_core import MicrophoneSource, DeviceInfo

# デバイス一覧を取得
devices = MicrophoneSource.list_devices()
for dev in devices:
    default = " (default)" if dev.is_default else ""
    print(f"{dev.index}: {dev.name}{default}")

# デフォルトデバイスを使用
with MicrophoneSource() as mic:
    for chunk in mic:
        print(f"音声入力: {len(chunk)} samples")

# 特定のデバイスを指定
with MicrophoneSource(device_id=2, sample_rate=16000, chunk_ms=100) as mic:
    for chunk in mic:
        process(chunk)
```

> **注意**: MicrophoneSource は PortAudio に依存しています。CI 環境など PortAudio がインストールされていない場合、インポート時にエラーが発生します。

---

## エラーハンドリング

```python
from livecap_core import (
    StreamTranscriber,
    FileSource,
    TranscriptionError,
    EngineError,
)

try:
    with StreamTranscriber(engine=engine) as transcriber:
        with FileSource("audio.wav") as source:
            for result in transcriber.transcribe_sync(source):
                print(result.text)
except EngineError as e:
    print(f"エンジンエラー: {e}")
except TranscriptionError as e:
    print(f"文字起こしエラー: {e}")
```

---

## 結果型

### TranscriptionResult（確定結果）

```python
from livecap_core import TranscriptionResult

# 属性
result: TranscriptionResult
print(result.text)        # 文字起こしテキスト
print(result.start_time)  # 開始時刻（秒）
print(result.end_time)    # 終了時刻（秒）
print(result.duration)    # 長さ（秒）= end_time - start_time
print(result.is_final)    # 確定フラグ（常に True）
print(result.confidence)  # 確信度（0.0-1.0）
print(result.source_id)   # ソース識別子

# SRT 形式に変換
srt_entry = result.to_srt_entry(index=1)
print(srt_entry)
# 1
# 00:00:00,000 --> 00:00:03,500
# こんにちは
```

### InterimResult（中間結果）

```python
from livecap_core import InterimResult

# 属性
interim: InterimResult
print(interim.text)             # 途中経過のテキスト
print(interim.accumulated_time) # 累積時間（秒）
print(interim.source_id)        # ソース識別子
```

---

## テスト

### Mock を使ったユニットテスト

```python
import pytest
import numpy as np
from livecap_core import StreamTranscriber, VADSegment, VADState

class MockEngine:
    def transcribe(self, audio, sample_rate):
        return ("テスト結果", 0.95)

    def get_required_sample_rate(self):
        return 16000

class MockVADProcessor:
    def __init__(self):
        self._state = VADState.SILENCE

    def process_chunk(self, audio, sample_rate):
        # 常に確定セグメントを返す
        return [VADSegment(
            audio=audio,
            start_time=0.0,
            end_time=len(audio) / sample_rate,
            is_final=True,
        )]

    def finalize(self):
        return None

    def reset(self):
        pass

    @property
    def state(self):
        return self._state

def test_stream_transcriber():
    transcriber = StreamTranscriber(
        engine=MockEngine(),
        vad_processor=MockVADProcessor(),
    )

    audio = np.zeros(16000, dtype=np.float32)
    transcriber.feed_audio(audio, 16000)

    result = transcriber.get_result(timeout=1.0)
    assert result is not None
    assert result.text == "テスト結果"

    transcriber.close()
```

### E2E テストの実行

```bash
# Mock 統合テスト（CI 環境）
uv run python -m pytest tests/integration/realtime/test_mock_realtime_flow.py -v

# E2E テスト（self-hosted runners、SileroVAD + 実エンジン）
LIVECAP_ENABLE_REALTIME_E2E=1 uv run python -m pytest tests/integration/realtime/test_e2e_realtime_flow.py -v
```

---

## トラブルシューティング

### Q: MicrophoneSource のインポートでエラーが出る

**A:** PortAudio がインストールされていない可能性があります。

```bash
# Ubuntu/Debian
sudo apt-get install libportaudio2

# macOS
brew install portaudio

# Windows
# sounddevice パッケージに同梱されています
```

### Q: VAD が音声を検出しない

**A:** 閾値を下げてみてください。

```python
config = VADConfig(threshold=0.3)  # デフォルト 0.5 → 0.3
```

### Q: 文字起こし結果が途切れる

**A:** `min_silence_ms` を長くしてみてください。

```python
config = VADConfig(min_silence_ms=300)  # デフォルト 100 → 300
```

### Q: GPU メモリが不足する

**A:** CPU モードで実行するか、小さいモデルを使用してください。

```python
engine = EngineFactory.create_engine("whispers2t_tiny", "cpu")
```

---

## 関連ドキュメント

- [VAD Bayesian 最適化ガイド](./vad-optimization.md) - カスタムパラメータチューニング
- [VAD バックエンド比較](../reference/vad-comparison.md) - Silero / TenVAD / WebRTC の比較
- [API 仕様書](../architecture/core-api-spec.md#8-phase-1-リアルタイム文字起こし-api)
- [機能一覧](../reference/feature-inventory.md#22-リアルタイム文字起こし)
- [テストガイド](../testing/README.md)
