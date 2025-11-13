from __future__ import annotations

"""Shared engine utilities (device detection, temp dirs, model paths)."""

import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple

from livecap_core.resources import get_model_manager

__all__ = [
    "get_models_dir",
    "get_temp_dir",
    "detect_device",
    "unicode_safe_temp_directory",
    "unicode_safe_download_directory",
]


def _model_manager():
    return get_model_manager()


def get_models_dir(engine_name: Optional[str] = None) -> Path:
    """Return the shared models directory (optionally scoped per engine)."""
    return _model_manager().get_models_dir(engine_name)


def get_temp_dir(purpose: str = "runtime") -> Path:
    """Return a cache-backed temp directory for the given purpose."""
    return _model_manager().get_temp_dir(purpose)


def detect_device(requested_device: Optional[str], engine_name: str) -> Tuple[str, str]:
    """
    Resolve which compute device an engine should use.

    Returns: (device, compute_type)
    """
    logger = logging.getLogger(__name__)

    if requested_device not in ("cpu",):
        try:
            import torch

            version = torch.__version__
            if torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"
                logger.info("Using CUDA for %s (PyTorch %s).", engine_name, version)
                return device, compute_type

            if "+cpu" in version:
                logger.warning("PyTorch CPU build detected (%s); falling back to CPU for %s.", version, engine_name)
            else:
                logger.warning("CUDA unavailable (PyTorch %s); falling back to CPU for %s.", version, engine_name)
        except ImportError:
            logger.warning("PyTorch not installed; using CPU for %s.", engine_name)

    return "cpu", "float32"


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
