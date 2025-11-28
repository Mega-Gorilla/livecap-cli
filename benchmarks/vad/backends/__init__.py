"""VAD バックエンド（ベンチマーク専用）

このモジュールには、ベンチマーク専用の VAD バックエンドが含まれます。
本番環境での使用には、livecap_core.vad.backends を使用してください。
"""

from __future__ import annotations

from .base import VADBenchmarkBackend
from .processor_wrapper import VADProcessorWrapper


def __getattr__(name: str):
    """遅延インポート for benchmark VAD backends."""
    if name == "JaVADPipeline":
        from .javad import JaVADPipeline

        return JaVADPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "VADBenchmarkBackend",
    "VADProcessorWrapper",
    "JaVADPipeline",
]
