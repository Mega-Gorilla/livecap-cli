"""Transcription helpers exposed by livecap_core."""

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
    SegmentTranscriber,
    StatusCallback,
)
from .result import InterimResult, TranscriptionResult
from .stream import (
    EngineError,
    StreamTranscriber,
    TranscriptionEngine,
    TranscriptionError,
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
    "SegmentTranscriber",
    "Segmenter",
    # Realtime transcription (Phase 1)
    "TranscriptionResult",
    "InterimResult",
    "StreamTranscriber",
    "TranscriptionEngine",
    "TranscriptionError",
    "EngineError",
]
