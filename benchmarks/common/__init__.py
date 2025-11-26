"""Common modules for benchmark framework.

This package provides shared functionality:
- metrics: WER/CER/RTF calculation, memory measurement
- text_normalization: Text preprocessing for evaluation
- datasets: Dataset management and audio file handling
- engines: ASR engine management with caching
- reports: Output formatting (JSON, Markdown, Console)
"""

from __future__ import annotations

from .metrics import (
    BenchmarkMetrics,
    calculate_wer,
    calculate_cer,
    calculate_rtf,
    measure_ram,
    GPUMemoryTracker,
)
from .text_normalization import (
    normalize_en,
    normalize_ja,
    normalize_text,
)
from .datasets import (
    AudioFile,
    Dataset,
    DatasetManager,
    DatasetError,
)
from .engines import (
    BenchmarkEngineManager,
    TranscriptionEngine,
)
from .reports import (
    BenchmarkReporter,
    BenchmarkResult,
    BenchmarkSummary,
)

__all__ = [
    # metrics
    "BenchmarkMetrics",
    "calculate_wer",
    "calculate_cer",
    "calculate_rtf",
    "measure_ram",
    "GPUMemoryTracker",
    # text_normalization
    "normalize_en",
    "normalize_ja",
    "normalize_text",
    # datasets
    "AudioFile",
    "Dataset",
    "DatasetManager",
    "DatasetError",
    # engines
    "BenchmarkEngineManager",
    "TranscriptionEngine",
    # reports
    "BenchmarkReporter",
    "BenchmarkResult",
    "BenchmarkSummary",
]
