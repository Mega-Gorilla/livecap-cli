from __future__ import annotations

"""Shared engine utilities (device detection, temp dirs, model paths)."""

import logging
from pathlib import Path
from typing import Optional

from livecap_cli.resources import get_model_manager

__all__ = [
    "get_models_dir",
    "get_temp_dir",
    "detect_device",
    "get_available_vram",
    "can_fit_on_gpu",
]


def _model_manager():
    return get_model_manager()


def get_models_dir(engine_name: Optional[str] = None) -> Path:
    """Return the shared models directory (optionally scoped per engine)."""
    return _model_manager().get_models_dir(engine_name)


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
