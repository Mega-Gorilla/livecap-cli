from __future__ import annotations

"""Shared engine utilities (device / dtype detection, temp dirs, model paths)."""

import logging
from pathlib import Path
from typing import Optional

from livecap_cli.resources import get_model_manager

__all__ = [
    "get_models_dir",
    "get_temp_dir",
    "detect_device",
    "TORCH_DTYPES",
    "resolve_torch_dtype",
    "get_available_vram",
    "can_fit_on_gpu",
]


def _model_manager():
    return get_model_manager()


def get_models_dir() -> Path:
    """Return the shared models directory (``models_root``)."""
    return _model_manager().get_models_dir()


def get_temp_dir(purpose: str = "runtime") -> Path:
    """Return a cache-backed temp directory for the given purpose."""
    return _model_manager().get_temp_dir(purpose)


def detect_device(requested_device: Optional[str], engine_name: str) -> str:
    """
    デバイスを検出して返す。

    Args:
        requested_device: 要求されたデバイス ("cuda", "cpu", None=auto)
        engine_name: エンジン名（ログ用）

    Returns:
        使用するデバイス ("cuda" または "cpu")
    """
    logger = logging.getLogger(__name__)

    if requested_device == "cpu":
        logger.info("Using CPU for %s (explicitly requested).", engine_name)
        return "cpu"

    try:
        import torch

        version = torch.__version__
        if torch.cuda.is_available():
            logger.info("Using CUDA for %s (PyTorch %s).", engine_name, version)
            return "cuda"

        if "+cpu" in version:
            logger.warning("PyTorch CPU build detected (%s); falling back to CPU for %s.", version, engine_name)
        else:
            logger.warning("CUDA unavailable (PyTorch %s); falling back to CPU for %s.", version, engine_name)
    except ImportError:
        logger.warning("PyTorch not installed; using CPU for %s.", engine_name)

    return "cpu"


#: torch で推論する engine (Qwen3-ASR / Parakeet / Canary) の ``dtype`` に渡せる値 (#470)。
TORCH_DTYPES = ("auto", "bfloat16", "float16", "float32")

#: bf16 をネイティブに計算できる最初の世代 (Ampere = compute capability 8.x)。
_BF16_MIN_COMPUTE_CAPABILITY = 8


def resolve_torch_dtype(requested: str, device: str, engine_name: str) -> str:
    """推論に使う dtype の名前を決める (#470)。

    ``"auto"`` は **CUDA かつ bf16 をネイティブに計算できる GPU (Ampere 以降) なら
    ``"bfloat16"``**、それ以外 (CPU / それより前の GPU) は ``"float32"``。bf16 は fp32 と
    指数部の幅が同じなので fp16 のような桁あふれが起きず、重みのメモリーは半分になる。
    Turing 以前の GPU は bf16 を模擬実行するだけで遅いので fp32 に留める (fp16 は桁あふれの
    検証をしていないので自動では選ばない)。明示値はそのまま使う。

    返り値は名前 (``getattr(torch, name)`` で dtype になる)。cache key やログでは torch を
    import せずに使える。

    Raises:
        ValueError: ``requested`` が :data:`TORCH_DTYPES` に無い。
    """
    logger = logging.getLogger(__name__)
    if requested not in TORCH_DTYPES:
        raise ValueError(f"Unsupported dtype: {requested!r}. Choose from {TORCH_DTYPES}")
    if requested != "auto":
        logger.info("Using %s for %s (explicitly requested).", requested, engine_name)
        return requested
    if device != "cuda":
        logger.info("Using float32 for %s (auto: %s).", engine_name, device)
        return "float32"

    import torch

    major, minor = torch.cuda.get_device_capability()
    if major >= _BF16_MIN_COMPUTE_CAPABILITY:
        logger.info("Using bfloat16 for %s (auto: GPU compute capability %d.%d).", engine_name, major, minor)
        return "bfloat16"
    logger.info(
        "Using float32 for %s (auto: GPU compute capability %d.%d has no native bfloat16).",
        engine_name, major, minor,
    )
    return "float32"


def get_available_vram() -> Optional[int]:
    """
    利用可能な VRAM（MB）を返す。

    Returns:
        VRAM（MB）。GPU なしまたは torch 未インストールの場合は None

    Note:
        torch がインストールされていない場合でも CTranslate2 は
        CUDA を使用可能。この関数は便利機能であり、必須ではない。
    """
    try:
        import torch

        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return free // (1024 * 1024)
    except ImportError:
        pass
    return None


def can_fit_on_gpu(required_mb: int, safety_margin: float = 0.9) -> bool:
    """
    指定サイズが GPU に収まるか確認。

    Args:
        required_mb: 必要な VRAM（MB）
        safety_margin: 安全マージン（デフォルト 0.9 = 90%）

    Returns:
        収まる場合 True。GPU なしまたは torch なしの場合は False
    """
    available = get_available_vram()
    if available is None:
        return False
    return available * safety_margin >= required_mb
