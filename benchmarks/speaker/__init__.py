"""Speaker-embedding benchmark module.

Measures GPU/memory/latency/separability of speaker-embedding backends
(TitaNet, ECAPA, pyannote) to inform SpeakerGate backend selection (issue #287).

Usage:
    python -m benchmarks.speaker --backend titanet ecapa pyannote --device cuda
"""

from __future__ import annotations

from .factory import (
    SPEAKER_REGISTRY,
    create_embedding_backend,
    get_all_backend_ids,
    get_backend_info,
)
from .reports import SpeakerBenchmarkReporter, SpeakerBenchmarkResult
from .runner import SpeakerBenchmarkConfig, SpeakerBenchmarkRunner

__all__ = [
    "SpeakerBenchmarkConfig",
    "SpeakerBenchmarkRunner",
    "SpeakerBenchmarkResult",
    "SpeakerBenchmarkReporter",
    "SPEAKER_REGISTRY",
    "create_embedding_backend",
    "get_all_backend_ids",
    "get_backend_info",
]
