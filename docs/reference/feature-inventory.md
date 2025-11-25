# LiveCap Core 機能一覧

> **作成日:** 2025-11-25
> **目的:** 現在実装されている全機能の棚卸しとサンプルコード

---

## 1. パッケージ構成

```
livecap-core/
├── livecap_core/           # コアライブラリ
│   ├── __init__.py         # 公開APIエクスポート
│   ├── cli.py              # CLI・診断機能
│   ├── i18n.py             # 国際化ヘルパー
│   ├── languages.py        # 言語定義
│   ├── transcription_types.py  # イベント型定義
│   ├── config/
│   │   ├── defaults.py     # デフォルト設定
│   │   ├── schema.py       # TypedDictスキーマ
│   │   └── validator.py    # 設定バリデーション
│   ├── resources/
│   │   ├── model_manager.py    # モデルキャッシュ管理
│   │   ├── ffmpeg_manager.py   # FFmpegバイナリ管理
│   │   └── resource_locator.py # リソースパス解決
│   ├── transcription/
│   │   └── file_pipeline.py    # ファイル文字起こしパイプライン
│   └── utils/
│       └── __init__.py
├── engines/                # ASRエンジン実装
│   ├── base_engine.py      # 抽象基底クラス
│   ├── engine_factory.py   # エンジンファクトリ
│   ├── metadata.py         # エンジンメタデータ
│   ├── reazonspeech_engine.py
│   ├── whispers2t_engine.py
│   ├── parakeet_engine.py
│   ├── canary_engine.py
│   └── voxtral_engine.py
└── config/                 # 設定ビルダー
    └── core_config_builder.py
```

---

## 2. 機能別詳細

### 2.1 言語サポート (`livecap_core.languages`)

**概要:** 16言語のメタデータ管理、言語コード正規化、エンジンマッピング

**対応言語:**
- ja (日本語), en (English), zh-CN (簡体中文), zh-TW (繁體中文)
- ko (한국어), de (Deutsch), fr (Français), es (Español)
- ru (Русский), ar (العربية), pt (Português), it (Italiano)
- hi (हिन्दी), nl (Nederlands)
- 地域バリアント: es-ES, es-US, pt-BR

**サンプルコード:**

```python
from livecap_core.languages import Languages, LanguageInfo

# === 言語コードの正規化 ===
print(Languages.normalize("JA"))       # "ja"
print(Languages.normalize("zh-TW"))    # "zh-TW"（繁体字は保持）
print(Languages.normalize("zh"))       # "zh-CN"（デフォルトは簡体字）
print(Languages.normalize("en-us"))    # "en"
print(Languages.normalize("auto"))     # "auto"（特殊コード）

# === 言語情報の取得 ===
info: LanguageInfo = Languages.get_info("ja")
print(f"表示名: {info.display_name}")        # "日本語"
print(f"英語名: {info.english_name}")        # "Japanese"
print(f"国旗: {info.flag}")                  # "🇯🇵"
print(f"ISO 639-1: {info.iso639_1}")         # "ja"
print(f"Windows LCID: {hex(info.windows_lcid)}")  # "0x411"

# === 表示名の取得 ===
print(Languages.get_display_name("ja"))              # "日本語"
print(Languages.get_display_name("ja", english=True)) # "Japanese"

# === エンジンマッピング ===
engines = Languages.get_engines_for_language("ja")
print(engines)  # ["reazonspeech", "whispers2t_base", "whispers2t_tiny", ...]

# === バリデーション ===
print(Languages.is_valid("ja"))      # True
print(Languages.is_valid("invalid")) # False

# === 自動検出モードの判定 ===
print(Languages.is_auto("auto"))     # True
print(Languages.is_auto("ja"))       # False

# === サポート言語一覧 ===
codes = Languages.get_supported_codes()
print(codes)  # {"ja", "en", "zh-CN", "zh-TW", ...}

# === 翻訳サービス対応言語 ===
google_langs = Languages.get_languages_for_translation_service("google")
print(google_langs)  # [("ja", "日本語"), ("en", "English"), ...]

# === Windows LCID から言語コード ===
lang = Languages.from_windows_lcid(0x0411)
print(lang)  # "ja"
```

---

### 2.2 リアルタイム文字起こし (`livecap_core.transcription`, `livecap_core.vad`, `livecap_core.audio_sources`)

**概要:** VAD ベースの音声セグメント検出とストリーミング文字起こし（Phase 1 で実装）

**コンポーネント:**
- `StreamTranscriber`: VAD + ASR 組み合わせストリーム処理
- `VADProcessor`: Silero VAD v5/v6 ベースの音声活動検出
- `AudioSource`: 音声入力抽象化（FileSource, MicrophoneSource）

**対応バックエンド:**
- Silero VAD v5/v6（ONNX）: デフォルト VAD バックエンド

**サンプルコード:**

```python
from livecap_core import (
    StreamTranscriber,
    FileSource,
    MicrophoneSource,
    VADConfig,
    TranscriptionResult,
)
from engines import EngineFactory

# === 基本的な使用方法 ===
engine = EngineFactory.create_engine("whispers2t_base", device="cuda")
engine.load_model()

# FileSource を使ったテスト
with StreamTranscriber(engine=engine) as transcriber:
    with FileSource("audio.wav") as source:
        for result in transcriber.transcribe_sync(source):
            print(f"[{result.start_time:.2f}s] {result.text}")

# === マイク入力からリアルタイム文字起こし ===
with StreamTranscriber(engine=engine) as transcriber:
    with MicrophoneSource(device_id=0) as mic:
        for result in transcriber.transcribe_sync(mic):
            print(f"{result.text}")

# === 非同期 API ===
import asyncio

async def realtime_transcribe():
    async with MicrophoneSource() as mic:
        transcriber = StreamTranscriber(engine=engine)
        async for result in transcriber.transcribe_async(mic):
            print(f"{result.text}")

asyncio.run(realtime_transcribe())

# === コールバック方式 ===
transcriber = StreamTranscriber(engine=engine)
transcriber.set_callbacks(
    on_result=lambda r: print(f"[確定] {r.text}"),
    on_interim=lambda r: print(f"[途中] {r.text}"),
)

with FileSource("audio.wav") as source:
    for chunk in source:
        transcriber.feed_audio(chunk, source.sample_rate)

final = transcriber.finalize()
transcriber.close()

# === カスタム VAD 設定 ===
custom_config = VADConfig(
    threshold=0.6,           # 音声検出閾値（高めに設定）
    min_speech_ms=300,       # 最小音声継続時間
    min_silence_ms=200,      # 無音判定時間
)

transcriber = StreamTranscriber(
    engine=engine,
    vad_config=custom_config,
)

# === デバイス一覧の取得 ===
devices = MicrophoneSource.list_devices()
for dev in devices:
    default_mark = " (default)" if dev.is_default else ""
    print(f"{dev.index}: {dev.name}{default_mark}")
```

---

### 2.3 ファイル文字起こしパイプライン (`livecap_core.transcription`)

**概要:** 音声/動画ファイルの文字起こしとSRT字幕生成

**対応フォーマット:** .wav, .flac, .mp3, .m4a, .aac, .ogg, .wma, .opus + FFmpegで変換可能な動画

**サンプルコード:**

```python
from pathlib import Path
from livecap_core import (
    FileTranscriptionPipeline,
    FileTranscriptionProgress,
    FileProcessingResult,
    FileSubtitleSegment,
    FileTranscriptionCancelled,
)
from livecap_core.config import get_default_config
import numpy as np

# === 基本的な使用方法 ===
def simple_transcriber(audio_data: np.ndarray, sample_rate: int) -> str:
    """音声データを受け取り、文字起こし結果を返すコールバック"""
    # 実際にはASRエンジンを呼び出す
    return "文字起こし結果"

config = get_default_config()
pipeline = FileTranscriptionPipeline(config=config)

try:
    result = pipeline.process_file(
        file_path=Path("audio.wav"),
        segment_transcriber=simple_transcriber,
    )

    print(f"成功: {result.success}")
    print(f"出力パス: {result.output_path}")  # audio.srt

    for segment in result.subtitles:
        print(f"{segment.start:.2f}-{segment.end:.2f}: {segment.text}")
finally:
    pipeline.close()

# === 進捗コールバック付き ===
def on_progress(progress: FileTranscriptionProgress):
    print(f"[{progress.current}/{progress.total}] {progress.status}")
    if progress.context:
        print(f"  詳細: {progress.context}")

result = pipeline.process_file(
    file_path=Path("audio.wav"),
    segment_transcriber=simple_transcriber,
    progress_callback=on_progress,
)

# === 複数ファイル処理 ===
def on_status(message: str):
    print(f"ステータス: {message}")

def on_result(result: FileProcessingResult):
    print(f"完了: {result.source_path} - {'成功' if result.success else '失敗'}")

def on_error(message: str, exception: Exception):
    print(f"エラー: {message}")

# キャンセル制御
cancel_flag = False
def should_cancel() -> bool:
    return cancel_flag

results = pipeline.process_files(
    file_paths=[Path("audio1.wav"), Path("audio2.mp3"), Path("video.mp4")],
    segment_transcriber=simple_transcriber,
    progress_callback=on_progress,
    status_callback=on_status,
    result_callback=on_result,
    error_callback=on_error,
    should_cancel=should_cancel,
    write_subtitles=True,
)

# === カスタムセグメンター ===
from typing import List, Tuple

def custom_segmenter(audio: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
    """音声をセグメントに分割するカスタム関数"""
    duration = len(audio) / sample_rate
    # 5秒ごとにセグメント化
    segments = []
    for start in range(0, int(duration), 5):
        end = min(start + 5, duration)
        segments.append((float(start), float(end)))
    return segments

pipeline_with_segmenter = FileTranscriptionPipeline(
    config=config,
    segmenter=custom_segmenter,
)

# === キャンセル処理 ===
try:
    result = pipeline.process_file(
        file_path=Path("long_audio.wav"),
        segment_transcriber=simple_transcriber,
        should_cancel=lambda: True,  # 即座にキャンセル
    )
except FileTranscriptionCancelled:
    print("処理がキャンセルされました")
```

---

### 2.3 ASRエンジン (`engines`)

**概要:** 複数のASRエンジンを統一インターフェースで提供

**利用可能なエンジン:**

| エンジンID | モデル名 | サイズ | 対応言語 |
|-----------|---------|--------|---------|
| `reazonspeech` | ReazonSpeech K2 v2 | 159MB | ja |
| `parakeet` | Parakeet TDT 0.6B v2 | 1.2GB | en |
| `parakeet_ja` | Parakeet TDT CTC 0.6B JA | 600MB | ja |
| `canary` | Canary 1B Flash | 1.5GB | en, de, fr, es |
| `voxtral` | Voxtral Mini 3B | 3GB | en, es, fr, pt, hi, de, nl, it |
| `whispers2t_tiny` | Whisper Tiny | 39MB | 13言語 |
| `whispers2t_base` | Whisper Base | 74MB | 13言語 |
| `whispers2t_small` | Whisper Small | 244MB | 13言語 |
| `whispers2t_medium` | Whisper Medium | 769MB | 13言語 |
| `whispers2t_large_v3` | Whisper Large-v3 | 1.55GB | 13言語 |

**サンプルコード:**

```python
from engines import EngineFactory, BaseEngine
from engines.metadata import EngineMetadata, EngineInfo
from livecap_core.config import get_default_config
import numpy as np

# === エンジンの作成と使用 ===
config = get_default_config()

# エンジンを作成
engine = EngineFactory.create_engine(
    engine_type="whispers2t_base",
    device="cuda",  # または "cpu"
    config=config,
)

# 進捗コールバックの設定
def on_model_progress(percent: int, message: str):
    print(f"[{percent}%] {message}")

engine.set_progress_callback(on_model_progress)

# モデルのロード
engine.load_model()

# 音声データの文字起こし
audio_data = np.zeros(16000, dtype=np.float32)  # 1秒の無音
sample_rate = 16000

text, confidence = engine.transcribe(audio_data, sample_rate)
print(f"結果: {text} (確信度: {confidence:.2f})")

# クリーンアップ
engine.cleanup()

# === 自動エンジン選択 ===
config["transcription"]["engine"] = "auto"
config["transcription"]["input_language"] = "ja"

engine = EngineFactory.create_engine(
    engine_type="auto",
    device="cuda",
    config=config,
)
# 日本語の場合、reazonspeechが自動選択される

# === 利用可能なエンジン情報 ===
available = EngineFactory.get_available_engines()
for engine_id, info in available.items():
    print(f"{engine_id}: {info['name']} - {info['description']}")

# 特定エンジンの情報
info = EngineFactory.get_engine_info("whispers2t_base")
print(f"名前: {info['name']}")
print(f"説明: {info['description']}")
print(f"対応言語: {info['supported_languages']}")

# 言語別エンジン一覧
ja_engines = EngineFactory.get_engines_for_language("ja")
print(f"日本語対応: {list(ja_engines.keys())}")

# デフォルトエンジンの取得
default_en = EngineFactory.get_default_engine_for_language("en", config)
print(f"英語のデフォルト: {default_en}")

# === エンジンメタデータの直接アクセス ===
metadata: EngineInfo = EngineMetadata.get("reazonspeech")
print(f"ID: {metadata.id}")
print(f"表示名: {metadata.display_name}")
print(f"サイズ: {metadata.model_size}")
print(f"デバイス: {metadata.device_support}")
print(f"ストリーミング: {metadata.streaming}")

# 全エンジン取得
all_engines = EngineMetadata.get_all()
for eid, einfo in all_engines.items():
    print(f"{eid}: {einfo.display_name}")

# 言語別エンジンリスト
ja_engines = EngineMetadata.get_engines_for_language("ja")
print(f"日本語対応: {ja_engines}")
```

---

### 2.4 設定管理 (`livecap_core.config`)

**概要:** デフォルト設定、マージ、バリデーション

**サンプルコード:**

```python
from livecap_core.config import (
    DEFAULT_CONFIG,
    get_default_config,
    merge_config,
    ConfigValidator,
    ValidationError,
)
from config import build_core_config

# === デフォルト設定の取得 ===
config = get_default_config()  # ディープコピーを返す

# 設定のセクション
print(config["audio"]["sample_rate"])           # 16000
print(config["transcription"]["engine"])         # "auto"
print(config["transcription"]["input_language"]) # "ja"
print(config["translation"]["enabled"])          # False

# === 設定のマージ ===
user_config = {
    "transcription": {
        "engine": "whispers2t_base",
        "input_language": "en",
    },
    "translation": {
        "enabled": True,
        "target_language": "ja",
    },
}

merged = merge_config(get_default_config(), user_config)
print(merged["transcription"]["engine"])  # "whispers2t_base"
print(merged["audio"]["sample_rate"])     # 16000（デフォルト値を継承）

# === GUI設定からCore設定への変換 ===
gui_config = {
    "transcription": {
        "engine": "reazonspeech",
    },
    "translation": {
        "enable_translation": True,  # GUI形式のキー
        "translation_service": "google",
    },
}

core_config = build_core_config(gui_config)
# キーが正規化される: enable_translation → enabled
print(core_config["translation"]["enabled"])  # True
print(core_config["translation"]["service"])  # "google"

# === 設定バリデーション ===
errors = ConfigValidator.validate(config)
if errors:
    for err in errors:
        print(f"{err.path}: {err.message}")
else:
    print("設定は有効です")

# 例外を投げるバリデーション
try:
    ConfigValidator.validate_or_raise(config)
except ValueError as e:
    print(f"バリデーションエラー: {e}")

# === デフォルト設定の構造 ===
"""
DEFAULT_CONFIG = {
    "audio": {
        "sample_rate": 16000,
        "chunk_duration": 0.25,
        "input_device": None,
        "processing": {...}
    },
    "multi_source": {
        "max_sources": 3,
        "defaults": {...},
        "sources": {}
    },
    "silence_detection": {
        "vad_threshold": 0.5,
        "vad_min_speech_duration_ms": 250,
        ...
    },
    "transcription": {
        "device": None,
        "engine": "auto",
        "input_language": "ja",
        "language_engines": {...},
        "reazonspeech_config": {...}
    },
    "translation": {
        "enabled": False,
        "service": "google",
        "target_language": "en",
        ...
    },
    "engines": {...},
    "logging": {...},
    "queue": {...},
    "debug": {...},
    "file_mode": {...}
}
"""
```

---

### 2.5 リソース管理 (`livecap_core.resources`)

**概要:** モデルキャッシュ、FFmpegバイナリ、リソースパスの管理

**サンプルコード:**

```python
from livecap_core.resources import (
    ModelManager,
    FFmpegManager,
    FFmpegNotFoundError,
    ResourceLocator,
    get_model_manager,
    get_ffmpeg_manager,
    get_resource_locator,
    reset_resource_managers,
)

# === ModelManager: モデルキャッシュ管理 ===
model_manager = get_model_manager()

# ディレクトリパス
print(f"モデルルート: {model_manager.models_root}")
print(f"キャッシュルート: {model_manager.cache_root}")

# エンジン固有のモデルディレクトリ
whisper_dir = model_manager.get_models_dir("whispers2t")
print(f"Whisperモデル: {whisper_dir}")

# 一時ディレクトリ
temp_dir = model_manager.get_temp_dir("processing")
print(f"一時ディレクトリ: {temp_dir}")

# ファイルダウンロード
def on_progress(downloaded: int, total: int):
    percent = (downloaded / total * 100) if total > 0 else 0
    print(f"ダウンロード中: {percent:.1f}%")

downloaded_path = model_manager.download_file(
    url="https://example.com/model.bin",
    filename="model.bin",
    expected_sha256="abc123...",  # オプション
    progress_callback=on_progress,
)

# 非同期ダウンロード
import asyncio

async def download_async():
    path = await model_manager.download_file_async(
        url="https://example.com/model.bin",
        filename="model.bin",
    )
    return path

# 一時ディレクトリのコンテキストマネージャ
with model_manager.temporary_directory("extraction") as temp:
    # tempは自動的にクリーンアップされる
    print(f"一時作業ディレクトリ: {temp}")

# HuggingFaceキャッシュ管理
with model_manager.huggingface_cache() as cache_dir:
    # HF_HOMEが自動設定される
    print(f"HFキャッシュ: {cache_dir}")

# === FFmpegManager: FFmpegバイナリ管理 ===
ffmpeg_manager = get_ffmpeg_manager()

# FFmpegの検索
try:
    ffmpeg_path = ffmpeg_manager.resolve_executable()
    print(f"FFmpeg: {ffmpeg_path}")
except FFmpegNotFoundError:
    print("FFmpegが見つかりません")

# FFprobeの検索
ffprobe_path = ffmpeg_manager.resolve_probe()
print(f"FFprobe: {ffprobe_path}")

# FFmpegの確保（必要ならダウンロード）
ffmpeg_path = ffmpeg_manager.ensure_executable()
print(f"FFmpeg確保: {ffmpeg_path}")

# 非同期版
async def ensure_async():
    path = await ffmpeg_manager.ensure_executable_async()
    return path

# 環境設定（PATHに追加）
ffmpeg_path = ffmpeg_manager.configure_environment()
print(f"環境設定完了: {ffmpeg_path}")

# === ResourceLocator: リソースパス解決 ===
locator = get_resource_locator()

try:
    bin_path = locator.resolve("ffmpeg-bin")
    print(f"FFmpegバイナリ: {bin_path}")
except FileNotFoundError:
    print("リソースが見つかりません")

# === シングルトンのリセット（テスト用） ===
reset_resource_managers()
```

---

### 2.6 イベントシステム (`livecap_core.transcription_types`)

**概要:** 文字起こし結果、ステータス、エラー等のイベント型定義

**イベント型:**
- `TranscriptionEventDict`: 文字起こし結果
- `StatusEventDict`: ステータス変更
- `ErrorEventDict`: エラー通知
- `TranslationRequestEventDict`: 翻訳リクエスト
- `TranslationResultEventDict`: 翻訳結果
- `SubtitleEventDict`: 字幕送信

**サンプルコード:**

```python
from livecap_core import (
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
)

# === イベント作成 ===

# 文字起こしイベント
transcription = create_transcription_event(
    text="こんにちは",
    source_id="source1",
    is_final=True,
    confidence=0.95,
    language="ja",
)
print(transcription)
# {'event_type': 'transcription', 'text': 'こんにちは', 'source_id': 'source1',
#  'is_final': True, 'timestamp': 1732..., 'confidence': 0.95, 'language': 'ja', 'phase': 'final'}

# ステータスイベント
status = create_status_event(
    status_code="ready",
    message="エンジン準備完了",
    source_id="source1",
    phase="ready",
)
print(status)

# エラーイベント
error = create_error_event(
    error_code="MODEL_LOAD_FAILED",
    message="モデルの読み込みに失敗しました",
    source_id="source1",
    error_details="詳細なエラー情報...",
)
print(error)

# 翻訳リクエスト
translation_req = create_translation_request_event(
    text="こんにちは",
    source_id="source1",
    source_language="ja",
    target_language="en",
)
print(translation_req)

# 翻訳結果
translation_result = create_translation_result_event(
    original_text="こんにちは",
    translated_text="Hello",
    source_id="source1",
    source_language="ja",
    target_language="en",
    confidence=0.98,
)
print(translation_result)

# 字幕イベント
subtitle = create_subtitle_event(
    text="こんにちは",
    source_id="source1",
    destination="obs",  # "obs" or "vrchat"
    is_translated=False,
)
print(subtitle)

# === バリデーション ===
is_valid = validate_event_dict(transcription)
print(f"有効: {is_valid}")  # True

# === ユーティリティ ===
event_type = get_event_type_name(transcription)
print(f"イベントタイプ: {event_type}")  # "transcription"

summary = format_event_summary(transcription)
print(f"サマリー: {summary}")  # "[source1] Final: こんにちは..."

# 古い形式の正規化
old_format = {"text": "テスト", "source_id": "src1"}
normalized = normalize_to_event_dict(old_format)
print(normalized)  # event_type等が追加される
```

---

### 2.7 CLI・診断 (`livecap_core.cli`)

**概要:** コマンドラインからの診断と設定ダンプ

**コマンドライン使用法:**

```bash
# 基本診断
python -m livecap_core

# JSON形式で出力
python -m livecap_core --as-json

# デフォルト設定を出力
python -m livecap_core --dump-config

# FFmpegを確保（ダウンロード）
python -m livecap_core --ensure-ffmpeg
```

**プログラム的使用:**

```python
from livecap_core.cli import diagnose, DiagnosticReport, main
from livecap_core.config import get_default_config

# 診断の実行
report: DiagnosticReport = diagnose(
    ensure_ffmpeg=False,
    config=get_default_config(),
)

print(f"設定有効: {report.config_valid}")
print(f"設定エラー: {report.config_errors}")
print(f"モデルルート: {report.models_root}")
print(f"キャッシュルート: {report.cache_root}")
print(f"FFmpegパス: {report.ffmpeg_path}")
print(f"i18nフォールバック数: {report.i18n.fallback_count}")

# JSON出力
json_output = report.to_json()
print(json_output)

# プログラムからCLIを実行
exit_code = main(["--dump-config"])
```

---

### 2.8 国際化 (`livecap_core.i18n`)

**概要:** 翻訳関数の登録とフォールバックメッセージ管理

**サンプルコード:**

```python
from livecap_core.i18n import (
    translate,
    register_translator,
    register_fallbacks,
    diagnose,
    I18nManager,
)

# === フォールバックの登録 ===
register_fallbacks({
    "status.loading": "読み込み中...",
    "status.ready": "準備完了",
    "error.model_not_found": "モデルが見つかりません: {model_name}",
})

# フォールバックを使った翻訳
message = translate("status.loading")
print(message)  # "読み込み中..."

# プレースホルダー付き
message = translate("error.model_not_found", model_name="whispers2t")
print(message)  # "モデルが見つかりません: whispers2t"

# 未登録キーはキー自体を返す
message = translate("unknown.key")
print(message)  # "unknown.key"

# デフォルト値の指定
message = translate("unknown.key", default="デフォルトメッセージ")
print(message)  # "デフォルトメッセージ"

# === カスタム翻訳関数の登録 ===
def my_translator(key: str, **kwargs) -> str:
    translations = {
        "status.loading": "Loading...",
        "status.ready": "Ready",
    }
    return translations.get(key, key)

register_translator(
    my_translator,
    name="MyTranslator",
    extras=["en", "ja"],
    metadata={"version": "1.0"},
)

# 登録後はカスタム翻訳関数が優先される
message = translate("status.ready")
print(message)  # "Ready"

# === 診断情報 ===
diagnostics = diagnose()
print(f"翻訳関数登録済み: {diagnostics.translator.registered}")
print(f"翻訳関数名: {diagnostics.translator.name}")
print(f"フォールバック数: {diagnostics.fallback_count}")
```

---

## 3. 環境変数

| 変数名 | 説明 | デフォルト |
|--------|------|-----------|
| `LIVECAP_CORE_MODELS_DIR` | モデル保存ディレクトリ | `~/.cache/LiveCap/PineLab/models` |
| `LIVECAP_CORE_CACHE_DIR` | キャッシュディレクトリ | `~/.cache/LiveCap/PineLab/cache` |
| `LIVECAP_FFMPEG_BIN` | FFmpegバイナリディレクトリ | 自動検出 |

---

## 4. 依存関係とインストール

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

---

## 5. 統合使用例

### 5.1 ファイル文字起こし完全例

```python
from pathlib import Path
from livecap_core import FileTranscriptionPipeline, FileTranscriptionProgress
from livecap_core.config import get_default_config, merge_config
from engines import EngineFactory

# 設定の準備
user_config = {
    "transcription": {
        "engine": "whispers2t_base",
        "input_language": "ja",
    },
}
config = merge_config(get_default_config(), user_config)

# エンジンの初期化
engine = EngineFactory.create_engine(
    engine_type=config["transcription"]["engine"],
    device="cuda",
    config=config,
)

def on_model_progress(percent: int, message: str):
    print(f"モデル読込: [{percent}%] {message}")

engine.set_progress_callback(on_model_progress)
engine.load_model()

# 文字起こし関数の定義
def transcriber(audio_data, sample_rate):
    text, confidence = engine.transcribe(audio_data, sample_rate)
    return text

# パイプラインの実行
def on_progress(progress: FileTranscriptionProgress):
    print(f"処理中: [{progress.current}/{progress.total}] {progress.status}")

pipeline = FileTranscriptionPipeline(config=config)

try:
    result = pipeline.process_file(
        file_path=Path("interview.mp4"),
        segment_transcriber=transcriber,
        progress_callback=on_progress,
    )

    if result.success:
        print(f"字幕ファイル: {result.output_path}")
        print(f"セグメント数: {len(result.subtitles)}")
        print(f"音声長: {result.metadata['duration_seconds']:.2f}秒")
    else:
        print(f"エラー: {result.error}")
finally:
    pipeline.close()
    engine.cleanup()
```

---

## 6. 未実装・将来の機能

以下の機能は仕様書に記載があるが、現時点で実装状況の確認が必要:

1. ~~**リアルタイム文字起こし**~~ → **Phase 1 で実装完了** (Section 2.2 参照)
2. **翻訳パイプライン統合** - `translation`設定は存在するが、パイプラインとの統合詳細未確認
3. **PyPI公開** - `pip install livecap-core`は記載されているが未公開
4. **SystemAudioSource** - Windows WASAPI / Linux PulseAudio によるシステム音声キャプチャ（Phase 2 以降）
5. **投機的実行（SpeculativeTranscriber）** - 低遅延化のための投機的文字起こし（Phase 2 以降）
