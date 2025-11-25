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

### 3.2 設定 (`livecap_core.config`)

```python
from livecap_core.config import (
    DEFAULT_CONFIG,       # デフォルト設定辞書
    get_default_config,   # DEFAULT_CONFIGのディープコピーを返す
    merge_config,         # ユーザー設定をデフォルトにマージ
    ConfigValidator,      # 設定バリデータクラス
    ValidationError,      # バリデーションエラークラス
)
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

### 3.5 言語 (`livecap_core.languages`)

```python
from livecap_core.languages import Languages, LanguageInfo

# 主要メソッド
Languages.normalize(code: str) -> Optional[str]
Languages.get_info(code: str) -> Optional[LanguageInfo]
Languages.get_display_name(code: str, english: bool = False) -> str
Languages.get_supported_codes() -> Set[str]
Languages.get_engines_for_language(code: str) -> List[str]
Languages.is_valid(code: str) -> bool
```

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
# 診断を実行
python -m livecap_core

# デフォルト設定を出力
python -m livecap_core --dump-config

# JSON形式で出力
python -m livecap_core --as-json

# FFmpegを確保
python -m livecap_core --ensure-ffmpeg
```

## 4. Engines パッケージ

### 4.1 エンジンファクトリ

```python
from engines import EngineFactory, BaseEngine

# エンジンを作成
engine = EngineFactory.create_engine(
    engine_type="whispers2t_base",
    device="cuda",  # または "cpu"
    config=config_dict,
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
from livecap_core.config import get_default_config

def transcribe_segment(audio_chunk, sample_rate):
    # ここでASR推論を実行
    return "transcribed text"

def on_progress(progress: FileTranscriptionProgress):
    print(f"[{progress.current}/{progress.total}] {progress.status}")

config = get_default_config()
pipeline = FileTranscriptionPipeline(config=config)

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
from engines import EngineFactory
from livecap_core.config import get_default_config
import numpy as np

config = get_default_config()
config["transcription"]["engine"] = "whispers2t_base"
config["transcription"]["input_language"] = "en"

engine = EngineFactory.create_engine(
    engine_type="whispers2t_base",
    device="cuda",
    config=config,
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

### 6.4 言語ユーティリティ

```python
from livecap_core import Languages

# 言語コードの正規化
print(Languages.normalize("JA"))      # "ja"
print(Languages.normalize("zh-TW"))   # "zh-TW"
print(Languages.normalize("zh"))      # "zh-CN"

# 言語情報の取得
info = Languages.get_info("ja")
print(f"{info.display_name} ({info.english_name})")  # 日本語 (Japanese)

# 言語に対応するエンジンを取得
engines = Languages.get_engines_for_language("ja")
print(engines)  # ["reazonspeech", "whispers2t_base", ...]
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

## 8. 互換性ポリシー

- **安定API**: `__all__` に記載された全シンボルは安定版とみなされる
- **破壊的変更**: メジャーバージョン更新時のみ
- **非推奨化**: 削除前に最低1マイナーバージョンの警告期間
- **TypedDictフィールド**: 既存フィールドは維持される。追加フィールドはオプショナル
