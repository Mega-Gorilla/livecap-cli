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

### 2.1 言語コード変換 (`livecap_core.engines.metadata`)

**概要:** BCP-47 形式の言語コードを ISO 639-1 に変換し、ASRエンジンで使用

**対応言語:**
- WhisperS2T: 100言語対応（ISO 639-1 および ISO 639-3 コード）
- 地域バリアント（zh-CN, zh-TW, pt-BR など）は自動的に基本言語コードに変換

> **Note**: `livecap_core.languages` モジュールは Issue #168 で廃止されました。
> 言語コード変換には `EngineMetadata.to_iso639_1()` を使用してください。

**サンプルコード:**

```python
from livecap_core.engines import EngineMetadata

# === BCP-47 → ISO 639-1 変換 ===
print(EngineMetadata.to_iso639_1("zh-CN"))  # "zh"
print(EngineMetadata.to_iso639_1("zh-TW"))  # "zh"
print(EngineMetadata.to_iso639_1("pt-BR"))  # "pt"
print(EngineMetadata.to_iso639_1("ja"))     # "ja"
print(EngineMetadata.to_iso639_1("ZH-CN"))  # "zh" (大文字も自動正規化)
print(EngineMetadata.to_iso639_1("yue"))    # "yue" (ISO 639-3 はパススルー)

# === エンジンマッピング ===
engines = EngineMetadata.get_engines_for_language("ja")
print(engines)  # ["reazonspeech", "parakeet_ja", "whispers2t"]

engines = EngineMetadata.get_engines_for_language("zh-CN")
print(engines)  # ["whispers2t"]

# === エンジン情報の取得 ===
info = EngineMetadata.get("whispers2t")
print(info.supported_languages)  # 100言語のリスト
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
    EngineFactory,
)

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
import numpy as np

# === 基本的な使用方法 ===
def simple_transcriber(audio_data: np.ndarray, sample_rate: int) -> str:
    """音声データを受け取り、文字起こし結果を返すコールバック"""
    # 実際にはASRエンジンを呼び出す
    return "文字起こし結果"

pipeline = FileTranscriptionPipeline()

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
from livecap_core import EngineFactory, BaseEngine, EngineMetadata, EngineInfo
import numpy as np

# === エンジンの作成と使用 ===

# エンジンを作成（EngineMetadata.default_params が自動適用）
engine = EngineFactory.create_engine(
    engine_type="whispers2t_base",
    device="cuda",  # または "cpu"
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

# === 言語に対応するエンジンを検索 ===
# Note: engine_type="auto" は廃止されました
ja_engines = EngineMetadata.get_engines_for_language("ja")
print(f"日本語対応エンジン: {ja_engines}")
# → ["reazonspeech", "parakeet_ja", "whispers2t_base", ...]

# 明示的にエンジンを指定
engine = EngineFactory.create_engine(
    engine_type="reazonspeech",  # 日本語には reazonspeech を明示指定
    device="cuda",
)

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

### 2.4 エンジン設定 (`livecap_core.engines.metadata`)

**概要:** エンジンメタデータとデフォルトパラメータの管理（Phase 2 で `livecap_core.config` は廃止）

**サンプルコード:**

```python
from livecap_core import EngineMetadata, EngineInfo, EngineFactory

# === エンジン設定（Phase 2: Config 廃止後） ===

# 利用可能なエンジン一覧を取得
all_engines = EngineMetadata.get_all()
for engine_id, info in all_engines.items():
    print(f"{engine_id}: {info.display_name}")
    print(f"  対応言語: {info.supported_languages}")
    print(f"  デフォルトパラメータ: {info.default_params}")

# 特定言語に対応するエンジンを検索
ja_engines = EngineMetadata.get_engines_for_language("ja")
print(f"日本語対応エンジン: {ja_engines}")
# → ["reazonspeech", "parakeet_ja", "whispers2t_base", ...]

# エンジン作成時にパラメータを上書き
engine = EngineFactory.create_engine(
    engine_type="reazonspeech",
    device="cuda",
    use_int8=True,  # default_params の use_int8=False を上書き
    num_threads=8,  # default_params の num_threads=4 を上書き
)

# WhisperS2T の場合
engine = EngineFactory.create_engine(
    engine_type="whispers2t_base",
    device="cuda",
    language="en",    # 言語を明示指定
    use_vad=False,    # 内蔵VADを無効化
    batch_size=32,    # バッチサイズを変更
)

# === EngineMetadata.default_params の構造 ===
# 各エンジンのデフォルトパラメータは engines/metadata.py で定義
"""
EngineMetadata.default_params 例:

reazonspeech:
    temperature: 0.0
    beam_size: 10
    use_int8: False
    num_threads: 4
    decoding_method: "greedy_search"

whispers2t_base:
    model_size: "base"
    batch_size: 24
    use_vad: True

parakeet:
    model_name: "nvidia/parakeet-tdt-0.6b-v2"
    decoding_strategy: "greedy"

canary:
    model_name: "nvidia/canary-1b-flash"
    beam_size: 1
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
# 基本診断（FFmpeg, CUDA, VAD, ASR engines を表示）
python -m livecap_core --info

# JSON形式で出力
python -m livecap_core --as-json

# FFmpegを確保（ダウンロード）
python -m livecap_core --ensure-ffmpeg
```

**プログラム的使用:**

```python
from livecap_core.cli import diagnose, DiagnosticReport, main

# 診断の実行
report: DiagnosticReport = diagnose(ensure_ffmpeg=False)

print(f"モデルルート: {report.models_root}")
print(f"キャッシュルート: {report.cache_root}")
print(f"FFmpegパス: {report.ffmpeg_path}")
print(f"CUDA利用可能: {report.cuda_available}")
print(f"VADバックエンド: {report.vad_backends}")
print(f"ASRエンジン: {report.available_engines}")
print(f"i18nフォールバック数: {report.i18n.fallback_count}")

# JSON出力
json_output = report.to_json()
print(json_output)

# プログラムからCLIを実行
exit_code = main(["--info"])
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
from livecap_core import FileTranscriptionPipeline, FileTranscriptionProgress, EngineFactory

# エンジンの初期化（Phase 2: 直接パラメータ指定）
engine = EngineFactory.create_engine(
    engine_type="whispers2t_base",
    device="cuda",
    language="ja",  # 言語を明示指定
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

pipeline = FileTranscriptionPipeline()

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
