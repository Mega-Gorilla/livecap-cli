"""Benchmark framework for ASR and VAD evaluation.

This package provides tools for benchmarking:
- ASR engines (accuracy, speed, memory usage)
- VAD backends (with ASR integration)

Usage:
    python -m benchmarks.asr --mode quick
    python -m benchmarks.vad --mode standard
"""

from __future__ import annotations

__version__ = "0.1.0"
