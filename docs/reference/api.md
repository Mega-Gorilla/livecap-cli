# API リファレンス

livecap-cli をライブラリとして使用するための API リファレンスです。

## 目次

- [インポート](#インポート)
- [EngineFactory](#enginefactory)
- [MicrophoneSource](#microphonesource)
- [DeviceInfo](#deviceinfo)
- [StreamTranscriber](#streamtranscriber)
- [TranscriptionResult](#transcriptionresult)
- [VADConfig](#vadconfig)
- [FileTranscriptionPipeline](#filetranscriptionpipeline)

---

## インポート

```python
from livecap_cli import (
    # エンジン
    EngineFactory,
    EngineMetadata,
    EngineInfo,

    # 音声ソース
    MicrophoneSource,
    FileSource,
    AudioSource,
    DeviceInfo,

    # 文字起こし
    StreamTranscriber,
    TranscriptionResult,
    InterimResult,
    FileTranscriptionPipeline,

    # VAD
    VADConfig,
    VADProcessor,
    VADSegment,
    VADState,

    # エラー
    TranscriptionError,
    EngineError,
)
```

---

## EngineFactory

ASR エンジンの作成・管理を行うファクトリークラス。

### メソッド

| メソッド | 戻り値 | 説明 |
|---------|--------|------|
| `get_available_engines()` | `Dict[str, Dict[str, str]]` | 利用可能なエンジン一覧を取得 |
| `get_engine_info(engine_type)` | `Optional[Dict[str, Any]]` | 特定エンジンの詳細情報を取得 |
| `get_engines_for_language(lang_code)` | `Dict[str, Dict[str, Any]]` | 指定言語に対応したエンジン一覧を取得 |
| `create_engine(engine_type, device, **options)` | `BaseEngine` | エンジンインスタンスを作成 |

### 使用例

```python
from livecap_cli import EngineFactory

# エンジン一覧を取得
engines = EngineFactory.get_available_engines()
for engine_id, info in engines.items():
    print(f"{engine_id}: {info['name']}")
    # 出力例: whispers2t: WhisperS2T

# 特定エンジンの詳細情報を取得
info = EngineFactory.get_engine_info("whispers2t")
print(info)
# {
#     'name': 'WhisperS2T',
#     'description': 'Multilingual ASR model with selectable model sizes...',
#     'supported_languages': ['en', 'ja', 'zh', ...],
#     'default_params': {'model_size': 'large-v3', ...},
#     'available_model_sizes': ['tiny', 'base', 'small', ...]
# }

# 日本語対応エンジンを取得
ja_engines = EngineFactory.get_engines_for_language("ja")
for engine_id in ja_engines:
    print(engine_id)  # reazonspeech, parakeet_ja, whispers2t

# エンジンを作成
engine = EngineFactory.create_engine(
    "whispers2t",
    device="cuda",        # "cuda", "cpu", または None (自動検出)
    model_size="base",    # whispers2t 固有オプション
)
engine.load_model()       # モデルをロード（必須）
```

### `get_engine_info()` の戻り値

| キー | 型 | 説明 |
|-----|-----|------|
| `name` | `str` | エンジン表示名 |
| `description` | `str` | エンジンの説明 |
| `supported_languages` | `List[str]` | 対応言語コード一覧 |
| `default_params` | `Dict[str, Any]` | デフォルトパラメータ |
| `available_model_sizes` | `Optional[List[str]]` | 選択可能なモデルサイズ（whispers2t のみ） |

### 利用可能なエンジン

| ID | モデル | 言語 | 備考 |
|----|--------|------|------|
| `reazonspeech` | ReazonSpeech K2 v2 | ja | 日本語特化 |
| `parakeet` | Parakeet TDT 0.6B v2 | en | 英語特化 |
| `parakeet_ja` | Parakeet TDT CTC 0.6B JA | ja | 日本語特化 |
| `canary` | Canary 1B Flash | en, de, fr, es | 多言語 |
| `voxtral` | Voxtral Mini 3B | en, es, fr 等 | 多言語 |
| `whispers2t` | WhisperS2T | 99言語 | モデルサイズ選択可 |

---

## MicrophoneSource

マイク入力からの音声キャプチャを行うクラス。

### コンストラクタ

```python
MicrophoneSource(
    device: Optional[int | str] = None,  # デバイスインデックスまたは名前（None=デフォルト）
    sample_rate: int = 16000,            # サンプリングレート
    chunk_ms: int = 100,                 # チャンクサイズ（ミリ秒）
)
```

### クラスメソッド

| メソッド | 戻り値 | 説明 |
|---------|--------|------|
| `list_devices()` | `List[DeviceInfo]` | 利用可能なマイクデバイス一覧を取得 |

### 使用例

```python
from livecap_cli import MicrophoneSource

# デバイス一覧を取得
devices = MicrophoneSource.list_devices()
for dev in devices:
    default_mark = " (default)" if dev.is_default else ""
    print(f"[{dev.index}] {dev.name} (ch:{dev.channels}){default_mark}")

# マイクから音声をキャプチャ
with MicrophoneSource(device=0) as mic:
    for chunk in mic:  # 同期イテレータ
        process(chunk)  # numpy.ndarray (float32)

# 非同期使用
async with MicrophoneSource() as mic:
    async for chunk in mic:
        await process(chunk)
```

> **Note**: MicrophoneSource は PortAudio に依存しています。`sudo apt-get install libportaudio2` (Ubuntu) または `brew install portaudio` (macOS) が必要です。

---

## DeviceInfo

オーディオデバイス情報を格納する dataclass。

### 属性

| 属性 | 型 | 説明 |
|-----|-----|------|
| `index` | `int` | デバイスインデックス |
| `name` | `str` | デバイス名 |
| `channels` | `int` | 入力チャンネル数 |
| `sample_rate` | `int` | デフォルトサンプリングレート |
| `is_default` | `bool` | デフォルトデバイスかどうか |

### 使用例

```python
from livecap_cli import MicrophoneSource

devices = MicrophoneSource.list_devices()
for dev in devices:
    print(f"Index: {dev.index}")
    print(f"Name: {dev.name}")
    print(f"Channels: {dev.channels}")
    print(f"Sample Rate: {dev.sample_rate}")
    print(f"Is Default: {dev.is_default}")
```

---

## StreamTranscriber

VAD + ASR を組み合わせたストリーミング文字起こしクラス。

### コンストラクタ

```python
StreamTranscriber(
    engine: TranscriptionEngine,              # ASR エンジン（必須）
    translator: Optional[BaseTranslator] = None,  # 翻訳エンジン
    source_lang: Optional[str] = None,        # ソース言語（translator 使用時必須）
    target_lang: Optional[str] = None,        # ターゲット言語（translator 使用時必須）
    vad_config: Optional[VADConfig] = None,   # VAD 設定
    vad_processor: Optional[VADProcessor] = None,  # VAD プロセッサ（テスト用）
    source_id: str = "default",               # ソース識別子
    max_workers: int = 1,                     # ワーカースレッド数
)
```

### パラメータ詳細

| パラメータ | 必須 | デフォルト | 説明 |
|-----------|:---:|-----------|------|
| `engine` | ✅ | - | 文字起こしエンジン（`load_model()` 済みであること） |
| `translator` | - | `None` | 翻訳エンジン（設定時は source_lang/target_lang 必須） |
| `source_lang` | - | `None` | 翻訳元言語コード（例: `"ja"`） |
| `target_lang` | - | `None` | 翻訳先言語コード（例: `"en"`） |
| `vad_config` | - | `None` | VAD 設定（vad_processor 未指定時に使用） |
| `vad_processor` | - | `None` | カスタム VAD プロセッサ（テスト用） |
| `source_id` | - | `"default"` | 音声ソースの識別子 |
| `max_workers` | - | `1` | 文字起こし用スレッドプールのワーカー数 |

### メソッド

| メソッド | 説明 |
|---------|------|
| `transcribe_sync(audio_source)` | 同期ストリーム処理（Iterator を返す） |
| `transcribe_async(audio_source)` | 非同期ストリーム処理（AsyncIterator を返す） |
| `feed_audio(audio, sample_rate)` | 音声チャンクを入力（低レベル API） |
| `get_result(timeout)` | 確定結果を取得 |
| `set_callbacks(on_result, on_interim)` | コールバックを設定 |
| `finalize()` | 残りのセグメントを文字起こし |
| `reset()` | 状態をリセット |
| `close()` | リソースを解放 |

### 使用例

```python
from livecap_cli import StreamTranscriber, MicrophoneSource, EngineFactory

# エンジンを準備
engine = EngineFactory.create_engine("whispers2t", device="cuda", model_size="base")
engine.load_model()

# 基本的な使い方（同期）
with StreamTranscriber(engine=engine) as transcriber:
    with MicrophoneSource() as mic:
        for result in transcriber.transcribe_sync(mic):
            print(f"[{result.start_time:.2f}s] {result.text}")

# 非同期使用
async with MicrophoneSource() as mic:
    async for result in transcriber.transcribe_async(mic):
        print(result.text)

# コールバック方式
transcriber = StreamTranscriber(engine=engine)
transcriber.set_callbacks(
    on_result=lambda r: print(f"[確定] {r.text}"),
    on_interim=lambda r: print(f"[途中] {r.text}"),
)

with MicrophoneSource() as mic:
    for chunk in mic:
        transcriber.feed_audio(chunk, mic.sample_rate)
```

---

## TranscriptionResult

文字起こし結果を格納する dataclass。

### 属性

| 属性 | 型 | 説明 |
|-----|-----|------|
| `text` | `str` | 文字起こしテキスト |
| `start_time` | `float` | 開始時間（秒） |
| `end_time` | `float` | 終了時間（秒） |
| `is_final` | `bool` | 確定結果かどうか |
| `confidence` | `float` | 信頼度スコア |
| `language` | `str` | 検出された言語コード（= 翻訳元言語） |
| `source_id` | `str` | 音声ソース識別子 |
| `translated_text` | `Optional[str]` | 翻訳テキスト |
| `target_language` | `Optional[str]` | 翻訳先言語 |

### メソッド

| メソッド | 戻り値 | 説明 |
|---------|--------|------|
| `to_srt_entry(index)` | `str` | SRT 形式の字幕エントリに変換 |
| `duration` | `float` | 発話時間（秒）をプロパティとして取得 |

### 使用例

```python
for result in transcriber.transcribe_sync(mic):
    print(f"Text: {result.text}")
    print(f"Time: {result.start_time:.2f}s - {result.end_time:.2f}s")
    print(f"Duration: {result.duration:.2f}s")
    print(f"Confidence: {result.confidence:.2f}")

    if result.translated_text:
        print(f"Translation: {result.translated_text}")

    # SRT 出力
    print(result.to_srt_entry(index=1))
    # 1
    # 00:00:00,000 --> 00:00:02,500
    # こんにちは
```

---

## VADConfig

VAD（音声活動検出）の設定を格納する dataclass。

### コンストラクタ

```python
VADConfig(
    threshold: float = 0.5,              # 音声検出閾値
    neg_threshold: Optional[float] = None,  # ノイズ閾値（None=自動）
    min_speech_ms: int = 250,            # 最小音声継続時間
    min_silence_ms: int = 100,           # 音声終了判定の無音時間
    speech_pad_ms: int = 100,            # 発話前後のパディング
    max_speech_ms: int = 0,              # 最大発話時間（0=無制限）
    interim_min_duration_ms: int = 2000, # 中間結果の最小時間
    interim_interval_ms: int = 1000,     # 中間結果の送信間隔
)
```

### パラメータ詳細

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `threshold` | `float` | `0.5` | 音声検出閾値（0.0-1.0） |
| `neg_threshold` | `float` | `None` | ノイズ閾値（None = threshold - 0.15） |
| `min_speech_ms` | `int` | `250` | 音声判定に必要な最小継続時間（ミリ秒） |
| `min_silence_ms` | `int` | `100` | 音声終了判定に必要な無音継続時間（ミリ秒） |
| `speech_pad_ms` | `int` | `100` | 発話前後のパディング（ミリ秒） |
| `max_speech_ms` | `int` | `0` | 最大発話時間（0 = 無制限） |
| `interim_min_duration_ms` | `int` | `2000` | 中間結果送信の最小発話時間 |
| `interim_interval_ms` | `int` | `1000` | 中間結果の送信間隔 |

### 使用例

```python
from livecap_cli import VADConfig, StreamTranscriber, EngineFactory

# デフォルト設定
config = VADConfig()

# カスタム設定（より敏感な検出）
config = VADConfig(
    threshold=0.4,        # 低い閾値でより敏感に
    min_speech_ms=200,    # 短い発話も検出
    min_silence_ms=150,   # 少し長い無音で区切る
)

# 辞書から作成
config = VADConfig.from_dict({
    'threshold': 0.6,
    'min_speech_ms': 300,
})

# StreamTranscriber に設定
engine = EngineFactory.create_engine("whispers2t", model_size="base")
engine.load_model()

transcriber = StreamTranscriber(
    engine=engine,
    vad_config=config,
)
```

---

## FileTranscriptionPipeline

ファイルからの一括文字起こしを行うパイプライン。

### 使用例

```python
from livecap_cli import FileTranscriptionPipeline, EngineFactory

engine = EngineFactory.create_engine("whispers2t", device="cuda", model_size="base")
engine.load_model()

pipeline = FileTranscriptionPipeline()
result = pipeline.process_file(
    file_path="audio.wav",
    segment_transcriber=lambda audio, sr: engine.transcribe(audio, sr)[0],
)

print(f"Output: {result.output_path}")
print(f"Subtitles: {len(result.subtitles)}")
```

---

## 関連ドキュメント

- [CLI リファレンス](cli.md) - コマンドライン操作
- [リアルタイム文字起こしガイド](../guides/realtime-transcription.md) - 詳細なガイド
- [サンプル README](../../examples/README.md) - 実行可能なサンプルコード
