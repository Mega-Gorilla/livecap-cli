"""VAD ベンチマークモジュール

VAD + ASR パイプラインのベンチマーク機能を提供します。

Usage:
    python -m benchmarks.vad --mode quick
    python -m benchmarks.vad --engine parakeet_ja --vad silero --language ja
"""

from __future__ import annotations

from .runner import VADBenchmarkRunner, VADBenchmarkConfig
from .factory import create_vad, get_all_vad_ids, get_vad_config, VAD_REGISTRY
from .backends import VADBenchmarkBackend, VADProcessorWrapper

__all__ = [
    # Runner
    "VADBenchmarkRunner",
    "VADBenchmarkConfig",
    # Factory
    "create_vad",
    "get_all_vad_ids",
    "get_vad_config",
    "VAD_REGISTRY",
    # Backends
    "VADBenchmarkBackend",
    "VADProcessorWrapper",
]
