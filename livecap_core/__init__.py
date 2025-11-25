"""Canonical re-exports for the LiveCap Core public API surface.

Phase 1 で合意した API 一覧（docs/dev-docs/architecture/core-api-spec.md）と
揃え、GUI や外部ツールが `livecap_core` 直下から主要シンボルを取得できるようにする。

Phase 1 Realtime Transcription API:
- StreamTranscriber: VAD + ASR 組み合わせストリーム処理
- TranscriptionResult, InterimResult: 結果型
- AudioSource, FileSource: 音声入力
- MicrophoneSource: マイク入力（遅延インポート、PortAudio必要）
- VADConfig, VADProcessor, VADSegment, VADState: VAD関連
"""

from typing import TYPE_CHECKING

from .languages import Languages
from .transcription_types import (
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
from .transcription.file_pipeline import (
    FileTranscriptionPipeline,
    FileTranscriptionProgress,
    FileProcessingResult,
    FileSubtitleSegment,
    FileTranscriptionCancelled,
)
from .config import ValidationError

# Phase 1: Realtime Transcription
from .transcription import (
    TranscriptionResult,
    InterimResult,
    StreamTranscriber,
    TranscriptionEngine,
    TranscriptionError,
    EngineError,
)
from .audio_sources import (
    AudioSource,
    DeviceInfo,
    FileSource,
)
from .vad import (
    VADConfig,
    VADProcessor,
    VADSegment,
    VADState,
)

# MicrophoneSource は遅延インポート（PortAudio 依存）
if TYPE_CHECKING:
    from .audio_sources import MicrophoneSource

__all__ = [
    "Languages",
    # 後方互換性のため既存のエクスポートを維持
    "create_transcription_event",
    "create_status_event",
    "create_error_event",
    "create_translation_request_event",
    "create_translation_result_event",
    "create_subtitle_event",
    "validate_event_dict",
    "get_event_type_name",
    "normalize_to_event_dict",
    "format_event_summary",
    "FileTranscriptionPipeline",
    "FileTranscriptionProgress",
    "FileProcessingResult",
    "FileSubtitleSegment",
    "FileTranscriptionCancelled",
    "ValidationError",
    # Phase 1: Realtime Transcription
    "TranscriptionResult",
    "InterimResult",
    "StreamTranscriber",
    "TranscriptionEngine",
    "TranscriptionError",
    "EngineError",
    "AudioSource",
    "DeviceInfo",
    "FileSource",
    "MicrophoneSource",
    "VADConfig",
    "VADProcessor",
    "VADSegment",
    "VADState",
]


def __getattr__(name: str):
    """遅延インポート for MicrophoneSource (PortAudio dependency)."""
    if name == "MicrophoneSource":
        from .audio_sources import MicrophoneSource

        return MicrophoneSource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
