"""Transcription helpers exposed by livecap_cli."""

from .file_pipeline import (
    ErrorCallback,
    FileProcessingResult,
    FileResultCallback,
    FileSubtitleSegment,
    FileTranscriptionCancelled,
    FileTranscriptionPipeline,
    FileTranscriptionProgress,
    ProgressCallback,
    Segmenter,
    SegmentOutcome,
    SegmentTranscriber,
    StatusCallback,
)
from .result import InterimResult, TranscriptionResult
from .result_coalescer import ResultCoalescer
from .srt import build_srt, write_srt
from .stream import (
    EngineError,
    StreamTranscriber,
    TranscriptionEngine,
    TranscriptionError,
)
from .utterance import (
    REASON_EMPTY_AUDIO,
    REASON_ENERGY_GATE,
    REASON_ENGINE_EMPTY,
    REASON_FILTER_REJECT,
    StaticSettledReason,
    UtteranceSettledEvent,
)

__all__ = [
    # File transcription (existing)
    "FileTranscriptionPipeline",
    "FileTranscriptionProgress",
    "FileProcessingResult",
    "FileSubtitleSegment",
    "FileTranscriptionCancelled",
    "ProgressCallback",
    "StatusCallback",
    "FileResultCallback",
    "ErrorCallback",
    "SegmentOutcome",
    "SegmentTranscriber",
    "Segmenter",
    # SRT serializer (Issue #363)
    "build_srt",
    "write_srt",
    # Realtime transcription (Phase 1)
    "TranscriptionResult",
    "InterimResult",
    "ResultCoalescer",
    "StreamTranscriber",
    "TranscriptionEngine",
    "TranscriptionError",
    "EngineError",
    # Utterance lifecycle (Issue #332)
    "UtteranceSettledEvent",
    "StaticSettledReason",
    "REASON_EMPTY_AUDIO",
    "REASON_ENERGY_GATE",
    "REASON_FILTER_REJECT",
    "REASON_ENGINE_EMPTY",
]
