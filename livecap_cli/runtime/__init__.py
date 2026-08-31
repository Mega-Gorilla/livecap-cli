"""フレームワークのランタイム初期化 (Issue #422)。

engine / translator / VAD のどれか 1 つに置いても足りない設定を、**共有の 1 箇所**
で決める層である。torch を触る入口は engine だけではない (CLI の CUDA 照会、
デバイス解決、Silero VAD、riva translator、NeMo の jit パッチ) ので、engine 個別
対応では抜ける。
"""

from __future__ import annotations

from .pytorch import (
    ENV_KERNEL_CACHE_PATH,
    ENV_USE_KERNEL_CACHE,
    PyTorchRuntimeDecision,
    PyTorchRuntimeError,
    configure_pytorch_runtime,
    current_pytorch_runtime,
)

__all__ = [
    "ENV_KERNEL_CACHE_PATH",
    "ENV_USE_KERNEL_CACHE",
    "PyTorchRuntimeDecision",
    "PyTorchRuntimeError",
    "configure_pytorch_runtime",
    "current_pytorch_runtime",
]
