# LiveCap Core API 仕様書

> **バージョン:** 1.0.0
> **最終更新:** 2025-11-25

## 1. 概要

LiveCap Core は音声認識（ASR）エンジンとファイル文字起こしパイプラインを提供するPythonライブラリです。

### 主な機能

- 複数のASRエンジン（ReazonSpeech, Whisper, Parakeet, Canary）のサポート
- ファイルベースの文字起こしパイプライン
- FFmpeg/モデルキャッシュの自動管理
- 多言語対応（日本語、英語、中国語など15言語以上）

## 2. パッケージ構成

```
livecap_core/
├── __init__.py              # 公開APIの再エクスポート
├── cli.py                   # CLIエントリーポイント・診断機能
├── i18n.py                  # 国際化ヘルパー
├── languages.py             # 言語定義（Languagesクラス）
├── transcription_types.py   # イベントTypedDict定義
├── config/
│   ├── __init__.py          # 設定エクスポート
│   ├── defaults.py          # デフォルト設定
│   ├── schema.py            # TypedDictスキーマ
│   └── validator.py         # 設定バリデーション
├── resources/
│   ├── __init__.py          # リソースマネージャーエクスポート
│   ├── model_manager.py     # モデルキャッシュ管理
│   ├── ffmpeg_manager.py    # FFmpegバイナリ管理
│   └── resource_locator.py  # リソースパス解決
├── transcription/
│   ├── __init__.py          # 文字起こしエクスポート
│   └── file_pipeline.py     # FileTranscriptionPipeline
└── utils/
    └── __init__.py          # ユーティリティ関数

engines/                     # ASRエンジン実装（別パッケージ）
├── __init__.py              # BaseEngine, EngineFactoryエクスポート
├── base_engine.py           # 抽象基底クラス
├── engine_factory.py        # エンジンファクトリ
├── metadata.py              # エンジンメタデータ定義
├── reazonspeech_engine.py   # ReazonSpeech（日本語）
├── whispers2t_engine.py     # WhisperS2T（多言語）
├── parakeet_engine.py       # NVIDIA Parakeet（英語）
├── canary_engine.py         # NVIDIA Canary（多言語）
└── voxtral_engine.py        # Voxtral（多言語）
```

## 3. 公開API

### 3.1 トップレベルエクスポート (`livecap_core`)

```python
from livecap_core import (
    # 言語ユーティリティ
    Languages,

    # ファイル文字起こしパイプライン
    FileTranscriptionPipeline,
    FileTranscriptionProgress,
    FileProcessingResult,
    FileSubtitleSegment,
    FileTranscriptionCancelled,

    # イベントヘルパー
    create_transcription_event,
    create_status_event,
    create_error_event,
    create_translation_request_event,
    create_translation_result_event,
    create_subtitle_event,
    validate_event_dict,
    get_event_type_name,
    normalize_to_event_dict,
    format_event_summary,

    # バリデーション
    ValidationError,
)
```

### 3.2 エンジン設定 (`livecap_core.engines.metadata`)

> **Note**: Phase 2 で `livecap_core.config` モジュールは廃止されました。エンジン設定は `EngineMetadata.default_params` で管理されます。

```python
from livecap_core import EngineMetadata

# 利用可能なエンジンを取得
engines = EngineMetadata.get_all()

# 特定言語に対応するエンジンを検索
ja_engines = EngineMetadata.get_engines_for_language("ja")
# → ["reazonspeech", "parakeet_ja", "whispers2t_base", ...]

# エンジンのデフォルトパラメータを確認
info = EngineMetadata.get("reazonspeech")
print(info.default_params)
# → {"temperature": 0.0, "beam_size": 10, "use_int8": False, ...}
```

### 3.3 リソース (`livecap_core.resources`)

```python
from livecap_core.resources import (
    # クラス
    ModelManager,         # モデル/キャッシュディレクトリ管理
    FFmpegManager,        # FFmpegバイナリ管理
    FFmpegNotFoundError,  # FFmpeg未検出例外
    ResourceLocator,      # リソースパス解決

    # シングルトンアクセサ
    get_model_manager,    # シングルトンModelManagerを返す
    get_ffmpeg_manager,   # シングルトンFFmpegManagerを返す
    get_resource_locator, # シングルトンResourceLocatorを返す
    reset_resource_managers,  # シングルトンをリセット（テスト用）
)
```

#### ModelManager API

| プロパティ/メソッド | 説明 |
|-------------------|------|
| `models_root` | モデル保存のルートディレクトリ |
| `cache_root` | キャッシュのルートディレクトリ |
| `get_models_dir(engine_name)` | エンジン固有のモデルディレクトリを取得 |
| `get_temp_dir(purpose)` | 目的別の一時ディレクトリを取得 |
| `download_file(url, ...)` | ファイルをキャッシュにダウンロード |
| `download_file_async(url, ...)` | download_fileの非同期版 |
| `temporary_directory(purpose)` | 一時ディレクトリのコンテキストマネージャ |
| `huggingface_cache()` | HF_HOMEを設定するコンテキストマネージャ |

#### FFmpegManager API

| メソッド | 説明 |
|---------|------|
| `resolve_executable()` | FFmpegバイナリを検索 |
| `resolve_probe()` | FFprobeバイナリを検索 |
| `ensure_executable()` | FFmpegの存在を確認（必要なら自動ダウンロード） |
| `ensure_executable_async()` | 非同期版 |
| `configure_environment()` | PATHを設定して実行パスを返す |
| `configure_environment_async()` | 非同期版 |

### 3.4 文字起こし (`livecap_core.transcription`)

```python
from livecap_core.transcription import (
    # メインパイプライン
    FileTranscriptionPipeline,

    # データクラス
    FileTranscriptionProgress,
    FileProcessingResult,
    FileSubtitleSegment,

    # 例外
    FileTranscriptionCancelled,

    # コールバック型
    ProgressCallback,
    StatusCallback,
    FileResultCallback,
    ErrorCallback,
    SegmentTranscriber,
    Segmenter,
)
```

#### FileTranscriptionPipeline

```python
class FileTranscriptionPipeline:
    def __init__(
        self,
        config: Dict[str, Any],
        ffmpeg_manager: Optional[FFmpegManager] = None,
        segmenter: Optional[Segmenter] = None,
    ): ...

    def process_file(
        self,
        file_path: Path,
        segment_transcriber: SegmentTranscriber,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> FileProcessingResult: ...

    def process_files(
        self,
        file_paths: List[Path],
        segment_transcriber: SegmentTranscriber,
        progress_callback: Optional[ProgressCallback] = None,
        status_callback: Optional[StatusCallback] = None,
        result_callback: Optional[FileResultCallback] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> None: ...

    def close(self) -> None: ...
```

### 3.5 言語コード変換 (`livecap_core.engines.metadata`)

```python
from livecap_core.engines import EngineMetadata

# BCP-47 → ISO 639-1 変換（ASRエンジン用）
iso_code = EngineMetadata.to_iso639_1("zh-CN")  # -> "zh"
iso_code = EngineMetadata.to_iso639_1("pt-BR")  # -> "pt"
iso_code = EngineMetadata.to_iso639_1("ja")     # -> "ja"

# 言語に対応したエンジンを取得
engines = EngineMetadata.get_engines_for_language("zh-CN")
# -> ["whispers2t"]
```

> **Note**: `livecap_core.languages` モジュールは Issue #168 で廃止されました。
> 言語コード変換には `EngineMetadata.to_iso639_1()` を使用してください。

### 3.6 CLI (`livecap_core.cli`)

```python
from livecap_core.cli import (
    main,              # CLIエントリーポイント
    diagnose,          # プログラム的な診断実行
    DiagnosticReport,  # 診断結果データクラス
)
```

CLI使用法:
```bash
# 診断を実行（FFmpeg, CUDA, VAD backends, ASR engines を表示）
python -m livecap_core --info

# JSON形式で出力
python -m livecap_core --as-json

# FFmpegを確保
python -m livecap_core --ensure-ffmpeg
```

## 4. Engines パッケージ

### 4.1 エンジンファクトリ

```python
from livecap_core import EngineFactory, EngineMetadata

# エンジンを作成（EngineMetadata.default_params が自動適用）
engine = EngineFactory.create_engine(
    engine_type="whispers2t_base",
    device="cuda",  # または "cpu"
)

# パラメータを上書きする場合
engine = EngineFactory.create_engine(
    engine_type="reazonspeech",
    device="cpu",
    use_int8=True,  # default_params を上書き
)

# モデルをロード
engine.load_model()

# 音声を文字起こし
text, confidence = engine.transcribe(audio_data, sample_rate)
```

### 4.2 利用可能なエンジン

#### ReazonSpeech

| エンジンID | モデル名 | モデルサイズ | 対応言語 |
|-----------|---------|-------------|---------|
| `reazonspeech` | ReazonSpeech K2 v2 | 159MB | ja |

#### NVIDIA Parakeet

| エンジンID | モデル名 | モデルサイズ | 対応言語 |
|-----------|---------|-------------|---------|
| `parakeet` | Parakeet TDT 0.6B v2 | 1.2GB | en |
| `parakeet_ja` | Parakeet TDT CTC 0.6B JA | 600MB | ja |

#### NVIDIA Canary

| エンジンID | モデル名 | モデルサイズ | 対応言語 |
|-----------|---------|-------------|---------|
| `canary` | Canary 1B Flash | 1.5GB | en, de, fr, es |

#### MistralAI Voxtral

| エンジンID | モデル名 | モデルサイズ | 対応言語 |
|-----------|---------|-------------|---------|
| `voxtral` | Voxtral Mini 3B | 3GB | en, es, fr, pt, hi, de, nl, it |

#### WhisperS2T (OpenAI Whisper)

| エンジンID | モデル名 | モデルサイズ | 対応言語 |
|-----------|---------|-------------|---------|
| `whispers2t_tiny` | Whisper Tiny | 39MB | 多言語（13言語） |
| `whispers2t_base` | Whisper Base | 74MB | 多言語（13言語） |
| `whispers2t_small` | Whisper Small | 244MB | 多言語（13言語） |
| `whispers2t_medium` | Whisper Medium | 769MB | 多言語（13言語） |
| `whispers2t_large_v3` | Whisper Large-v3 | 1.55GB | 多言語（13言語） |

> **WhisperS2T対応言語**: ja, en, zh-CN, zh-TW, ko, de, fr, es, ru, ar, pt, it, hi

### 4.3 BaseEngine インターフェース

```python
class BaseEngine(ABC):
    def __init__(self, device: Optional[str], config: Optional[Dict]): ...

    def load_model(self) -> None: ...
    def set_progress_callback(self, callback: ProgressCallback) -> None: ...

    @abstractmethod
    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[str, float]: ...

    @abstractmethod
    def get_engine_name(self) -> str: ...

    @abstractmethod
    def get_supported_languages(self) -> List[str]: ...

    @abstractmethod
    def get_required_sample_rate(self) -> int: ...

    def is_initialized(self) -> bool: ...
    def cleanup(self) -> None: ...
```

## 5. 環境変数

| 変数名 | 説明 | デフォルト |
|--------|------|-----------|
| `LIVECAP_CORE_MODELS_DIR` | モデル保存ディレクトリ | `~/.livecap/models`（またはappdirs） |
| `LIVECAP_CORE_CACHE_DIR` | キャッシュディレクトリ | `~/.livecap/cache`（またはappdirs） |
| `LIVECAP_FFMPEG_BIN` | FFmpegバイナリディレクトリ | 自動検出 |

## 6. 使用例

### 6.1 ファイル文字起こしパイプライン

```python
from pathlib import Path
from livecap_core import FileTranscriptionPipeline, FileTranscriptionProgress

def transcribe_segment(audio_chunk, sample_rate):
    # ここでASR推論を実行
    return "transcribed text"

def on_progress(progress: FileTranscriptionProgress):
    print(f"[{progress.current}/{progress.total}] {progress.status}")

pipeline = FileTranscriptionPipeline()

result = pipeline.process_file(
    file_path=Path("audio.wav"),
    segment_transcriber=transcribe_segment,
    progress_callback=on_progress,
)

for segment in result.subtitles:
    print(f"{segment.start:.2f}-{segment.end:.2f}: {segment.text}")

pipeline.close()
```

### 6.2 エンジンを直接使用

```python
from livecap_core import EngineFactory
import numpy as np

# 英語音声を文字起こし
engine = EngineFactory.create_engine(
    engine_type="whispers2t_base",
    device="cuda",
    language="en",  # 言語を明示指定
)

engine.load_model()

# audio_dataはfloat32サンプルのnumpy配列を想定
audio_data = np.zeros(16000, dtype=np.float32)  # 1秒の無音
sample_rate = 16000

text, confidence = engine.transcribe(audio_data, sample_rate)
print(f"文字起こし結果: {text} (確信度: {confidence:.2f})")
```

### 6.3 リソース管理

```python
from livecap_core.resources import get_model_manager, get_ffmpeg_manager

# モデル管理
model_manager = get_model_manager()
models_dir = model_manager.get_models_dir("whispers2t")
print(f"モデル保存先: {models_dir}")

# FFmpeg管理
ffmpeg_manager = get_ffmpeg_manager()
ffmpeg_path = ffmpeg_manager.ensure_executable()
print(f"FFmpegパス: {ffmpeg_path}")
```

### 6.4 言語コード変換

```python
from livecap_core.engines import EngineMetadata

# BCP-47 → ISO 639-1 変換（ASRエンジン用）
print(EngineMetadata.to_iso639_1("zh-CN"))  # "zh"
print(EngineMetadata.to_iso639_1("pt-BR"))  # "pt"
print(EngineMetadata.to_iso639_1("ZH-TW"))  # "zh" (大文字も自動正規化)

# 言語に対応するエンジンを取得
engines = EngineMetadata.get_engines_for_language("ja")
print(engines)  # ["reazonspeech", "parakeet_ja", "whispers2t"]

engines = EngineMetadata.get_engines_for_language("zh-CN")
print(engines)  # ["whispers2t"]
```

## 7. インストール

```bash
# 基本インストール
pip install livecap-core

# 翻訳サポート付き
pip install livecap-core[translation]

# 開発ツール付き
pip install livecap-core[dev]

# PyTorchエンジン付き（ReazonSpeech, Whisper）
pip install livecap-core[engines-torch]

# NeMoエンジン付き（Parakeet, Canary）
pip install livecap-core[engines-nemo]
```

## 8. Phase 1: リアルタイム文字起こし API

Phase 1 で追加されたリアルタイム文字起こし機能の API です。

### 8.1 トップレベルエクスポート（Phase 1 追加分）

```python
from livecap_core import (
    # 結果型
    TranscriptionResult,
    InterimResult,

    # ストリーミング
    StreamTranscriber,
    TranscriptionEngine,  # Protocol
    TranscriptionError,
    EngineError,

    # 音声ソース
    AudioSource,
    DeviceInfo,
    FileSource,
    MicrophoneSource,  # 遅延インポート（PortAudio依存）

    # VAD
    VADConfig,
    VADProcessor,
    VADSegment,
    VADState,
)
```

### 8.2 結果型

#### TranscriptionResult

```python
@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """文字起こし結果（確定）"""
    text: str
    start_time: float
    end_time: float
    is_final: bool = True
    confidence: float = 1.0
    language: str = ""
    source_id: str = "default"

    @property
    def duration(self) -> float: ...
    def to_srt_entry(self, index: int) -> str: ...
```

#### InterimResult

```python
@dataclass(frozen=True, slots=True)
class InterimResult:
    """中間結果（確定前の途中経過）"""
    text: str
    accumulated_time: float
    source_id: str = "default"
```

### 8.3 StreamTranscriber

VAD プロセッサと ASR エンジンを組み合わせてリアルタイム文字起こしを行うクラス。

```python
class StreamTranscriber:
    def __init__(
        self,
        engine: TranscriptionEngine,
        vad_config: Optional[VADConfig] = None,
        vad_processor: Optional[VADProcessor] = None,
        source_id: str = "default",
        max_workers: int = 1,
    ): ...

    # 低レベル API
    def feed_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> None: ...
    def get_result(self, timeout: Optional[float] = None) -> Optional[TranscriptionResult]: ...
    def get_interim(self) -> Optional[InterimResult]: ...

    # コールバック API
    def set_callbacks(
        self,
        on_result: Optional[Callable[[TranscriptionResult], None]] = None,
        on_interim: Optional[Callable[[InterimResult], None]] = None,
    ) -> None: ...

    # 高レベル API
    def transcribe_sync(self, audio_source: AudioSource) -> Iterator[TranscriptionResult]: ...
    async def transcribe_async(self, audio_source: AudioSource) -> AsyncIterator[TranscriptionResult]: ...

    # 制御
    def finalize(self) -> Optional[TranscriptionResult]: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...
```

| メソッド | 説明 |
|---------|------|
| `feed_audio()` | 音声チャンクを入力。VADでセグメント検出時は文字起こし実行のためブロッキング |
| `get_result()` | 確定結果を取得（ブロッキング） |
| `get_interim()` | 中間結果を取得（ノンブロッキング） |
| `set_callbacks()` | 結果受信時のコールバックを設定 |
| `transcribe_sync()` | AudioSource から同期的に文字起こし |
| `transcribe_async()` | AudioSource から非同期的に文字起こし |
| `finalize()` | 残っているセグメントを処理して結果を返す |
| `reset()` | 内部状態をリセット |
| `close()` | リソースを解放 |

### 8.4 VAD モジュール

#### VADConfig

```python
@dataclass(frozen=True, slots=True)
class VADConfig:
    """VAD設定（すべてミリ秒単位で統一）"""
    threshold: float = 0.5              # 音声検出閾値
    neg_threshold: Optional[float] = None  # ノイズ閾値
    min_speech_ms: int = 250            # 最小音声継続時間
    min_silence_ms: int = 100           # 音声終了判定の無音時間
    speech_pad_ms: int = 100            # 発話前後のパディング
    max_speech_ms: int = 0              # 最大発話時間（0=無制限）
    interim_min_duration_ms: int = 2000 # 中間結果送信の最小時間
    interim_interval_ms: int = 1000     # 中間結果送信間隔

    @classmethod
    def from_dict(cls, config: dict) -> VADConfig: ...
    def to_dict(self) -> dict: ...
```

#### VADProcessor

```python
class VADProcessor:
    """VADプロセッサ（Silero VAD + ステートマシン）"""
    SAMPLE_RATE: int = 16000
    FRAME_SAMPLES: int = 512  # 32ms @ 16kHz

    def __init__(
        self,
        config: Optional[VADConfig] = None,
        backend: Optional[VADBackend] = None,
    ): ...

    def process_chunk(self, audio: np.ndarray, sample_rate: int = 16000) -> list[VADSegment]: ...
    def finalize(self) -> Optional[VADSegment]: ...
    def reset(self) -> None: ...

    @property
    def state(self) -> VADState: ...
    @property
    def config(self) -> VADConfig: ...
```

#### VADSegment / VADState

```python
@dataclass(slots=True)
class VADSegment:
    """検出された音声セグメント"""
    audio: np.ndarray
    start_time: float
    end_time: float
    is_final: bool

class VADState(Enum):
    """VAD状態"""
    SILENCE = 1           # 無音
    POTENTIAL_SPEECH = 2  # 音声の可能性（検証中）
    SPEECH = 3            # 確定した音声
    ENDING = 4            # 音声終了処理中
```

### 8.5 AudioSource モジュール

#### AudioSource (ABC)

```python
class AudioSource(ABC):
    """音声ソースの抽象基底クラス"""

    def __init__(self, sample_rate: int = 16000, chunk_ms: int = 100): ...

    @property
    def sample_rate(self) -> int: ...
    @property
    def is_active(self) -> bool: ...

    @abstractmethod
    def start(self) -> None: ...
    @abstractmethod
    def stop(self) -> None: ...
    @abstractmethod
    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]: ...

    # 同期/非同期イテレータ
    def __iter__(self) -> Iterator[np.ndarray]: ...
    async def __aiter__(self) -> AsyncIterator[np.ndarray]: ...

    # コンテキストマネージャ
    def __enter__(self) -> AudioSource: ...
    def __exit__(self, *args) -> None: ...
    async def __aenter__(self) -> AudioSource: ...
    async def __aexit__(self, *args) -> None: ...
```

#### FileSource

```python
class FileSource(AudioSource):
    """ファイルからの音声ストリーム（テスト・デバッグ用）"""

    def __init__(
        self,
        file_path: Path | str,
        sample_rate: int = 16000,
        chunk_ms: int = 100,
        realtime: bool = False,  # リアルタイムシミュレーション
    ): ...
```

#### MicrophoneSource

```python
class MicrophoneSource(AudioSource):
    """sounddevice ベースのマイク入力"""

    def __init__(
        self,
        device_id: Optional[int] = None,
        sample_rate: int = 16000,
        chunk_ms: int = 100,
    ): ...

    @classmethod
    def list_devices(cls) -> list[DeviceInfo]: ...
```

> **注意**: MicrophoneSource は遅延インポートされます（PortAudio 依存）。CI 環境など PortAudio がインストールされていない環境では、明示的にインポートするまでエラーは発生しません。

#### DeviceInfo

```python
@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """オーディオデバイス情報"""
    index: int
    name: str
    channels: int
    sample_rate: int
    is_default: bool = False
```

### 8.6 例外型

```python
class TranscriptionError(Exception):
    """文字起こしエラーの基底クラス"""
    pass

class EngineError(TranscriptionError):
    """エンジン関連のエラー"""
    pass
```

### 8.7 使用例

#### 同期ストリーム処理

```python
from livecap_core import StreamTranscriber, FileSource, EngineFactory

engine = EngineFactory.create_engine("whispers2t_base", "cuda")
engine.load_model()

with StreamTranscriber(engine=engine) as transcriber:
    with FileSource("audio.wav") as source:
        for result in transcriber.transcribe_sync(source):
            print(f"[{result.start_time:.2f}s] {result.text}")
```

#### 非同期ストリーム処理

```python
import asyncio
from livecap_core import StreamTranscriber, MicrophoneSource

async def main():
    engine = EngineFactory.create_engine("whispers2t_base", "cuda")
    engine.load_model()

    transcriber = StreamTranscriber(engine=engine)

    async with MicrophoneSource() as mic:
        async for result in transcriber.transcribe_async(mic):
            print(f"{result.text}")

asyncio.run(main())
```

#### コールバック方式

```python
transcriber = StreamTranscriber(engine=engine)

transcriber.set_callbacks(
    on_result=lambda r: print(f"[確定] {r.text}"),
    on_interim=lambda r: print(f"[途中] {r.text}"),
)

with FileSource("audio.wav") as source:
    for chunk in source:
        transcriber.feed_audio(chunk, source.sample_rate)

final = transcriber.finalize()
if final:
    print(f"[最終] {final.text}")

transcriber.close()
```

## 9. 互換性ポリシー

- **安定API**: `__all__` に記載された全シンボルは安定版とみなされる
- **破壊的変更**: メジャーバージョン更新時のみ
- **非推奨化**: 削除前に最低1マイナーバージョンの警告期間
- **TypedDictフィールド**: 既存フィールドは維持される。追加フィールドはオプショナル
