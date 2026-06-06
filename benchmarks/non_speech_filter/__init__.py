"""Ad-hoc benchmark runner for the non-speech filter pipeline (Issue #295 PR-0).

Exposes ``NonSpeechFilterBenchmarkRunner`` for evaluating the multi-layered
defense (NoiseGate + VAD + EnergyGate, plus future Layer 1/2/3/4 additions)
against synthetic and real audio corpora across all supported VAD backends
and optionally a real ASR engine.

Run via:

    python -m benchmarks.non_speech_filter --mode quick

See ``docs/benchmarks/non-speech-filter.md`` for full usage.
"""

from .report import NonSpeechFilterReport, NonSpeechFilterRunRecord
from .runner import NonSpeechFilterBenchmarkConfig, NonSpeechFilterBenchmarkRunner

__all__ = [
    "NonSpeechFilterReport",
    "NonSpeechFilterRunRecord",
    "NonSpeechFilterBenchmarkConfig",
    "NonSpeechFilterBenchmarkRunner",
]
