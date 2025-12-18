from __future__ import annotations

"""Shared engine utilities (device detection, temp dirs, model paths)."""

import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from livecap_cli.resources import get_model_manager

__all__ = [
    "get_models_dir",
    "get_temp_dir",
    "detect_device",
    "unicode_safe_temp_directory",
    "unicode_safe_download_directory",
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


def _override_temp_environment(temp_dir: Path):
    saved = {
        "TEMP": os.environ.get("TEMP"),
        "TMP": os.environ.get("TMP"),
        "TMPDIR": os.environ.get("TMPDIR"),
        "tempdir": tempfile.tempdir,
    }

    temp_dir_str = str(temp_dir)
    os.environ["TEMP"] = temp_dir_str
    os.environ["TMP"] = temp_dir_str
    os.environ["TMPDIR"] = temp_dir_str
    tempfile.tempdir = temp_dir_str

    return saved


def _restore_temp_environment(saved):
    for key in ("TEMP", "TMP", "TMPDIR"):
        value = saved.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    tempfile.tempdir = saved.get("tempdir")


@contextmanager
def unicode_safe_temp_directory():
    """
    Temporarily point tempfile + env vars to a Unicode-safe cache directory.
    """
    temp_dir = get_temp_dir("runtime")
    saved = _override_temp_environment(temp_dir)
    try:
        yield temp_dir
    finally:
        _restore_temp_environment(saved)


def _cleanup_directory(path: Path):
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


@contextmanager
def unicode_safe_download_directory():
    """
    Same as unicode_safe_temp_directory but cleans the download cache afterwards.
    """
    temp_dir = get_temp_dir("downloads")
    saved = _override_temp_environment(temp_dir)
    try:
        yield temp_dir
    finally:
        _restore_temp_environment(saved)
        _cleanup_directory(temp_dir)


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
