"""フレームワークのランタイム初期化 (Issue #422)。

engine / translator / VAD のどれか 1 つに置いても足りない設定を、**共有の 1 箇所**
で決める層である。torch を触る入口は engine だけではない (CLI の CUDA 照会、
デバイス解決、Silero VAD、riva translator、NeMo の jit パッチ) ので、engine 個別
対応では抜ける。

同じ理由で、**process 全体に効く CUDA メモリの解放** (:func:`release_idle_cuda_memory`、
Issue #462) もここに置く。cuBLAS workspace は process 内の static map なので、
engine / translator 単体の ``cleanup()`` の責務にはできない。
"""

from __future__ import annotations

from .cuda_memory import CudaMemoryRelease, release_idle_cuda_memory
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
    "CudaMemoryRelease",
    "PyTorchRuntimeDecision",
    "PyTorchRuntimeError",
    "configure_pytorch_runtime",
    "current_pytorch_runtime",
    "release_idle_cuda_memory",
]
