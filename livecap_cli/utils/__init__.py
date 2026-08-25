from __future__ import annotations

"""Shared engine utilities (device detection, temp dirs, model paths)."""

import logging
import os
import tempfile
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from livecap_cli.resources import get_model_manager

__all__ = [
    "get_models_dir",
    "get_temp_dir",
    "detect_device",
    "unicode_safe_temp_directory",
    "unicode_safe_download_directory",
    "TempEnvironmentConflictError",
    "get_available_vram",
    "can_fit_on_gpu",
]


from livecap_cli.paths.errors import TempEnvironmentConflictError  # noqa: F401


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


# TEMP 移設の実装は :mod:`livecap_cli.paths.temp_env` へ移した (Issue #375 PR 2)。
# **ロック実装を 2 つ保守しない**ため、ここは委譲だけを残す。本 module の
# ``unicode_safe_*`` は PR 3 で削除される。
def _temp_environment(purpose: str, *, unique: bool):
    """後方互換の委譲。``base`` を渡さないので**従来どおり ``cache_root`` 配下**。

    ASCII 保証は付かない — それが要る呼び出しは
    :func:`livecap_cli.paths.ascii_safe_temp_environment` を使うこと。
    """
    from livecap_cli.paths.temp_env import temp_environment

    return temp_environment(purpose, unique=unique)


@contextmanager
def unicode_safe_temp_directory():
    """
    Temporarily point tempfile + env vars to a Unicode-safe cache directory.

    .. deprecated::
        **名前に反して ASCII 安全ではない** (``cache_root`` は appdirs 既定では
        ユーザー名を含む)。呼び出しはゼロで、#375 PR 3 で削除される。
    """
    with _temp_environment("runtime", unique=False) as temp_dir:
        yield temp_dir


@contextmanager
def unicode_safe_download_directory():
    """
    Point tempfile + env vars at a per-scope directory under the download cache.

    **共有ディレクトリを再帰削除しない** (Issue #386)。以前はスコープ退出時に
    ``cache_root/downloads`` を丸ごと ``rmtree`` しており、スコープ中に別スレッドが
    作った一時ファイル (発話ごとの wav を含む) まで削除していた。

    .. note::
        **ASCII 安全性は保証しない。** ``cache_root`` はユーザー名を含み得る。
        ASCII 保証は #375 PR 2 の ``ascii_safe_temp_environment()`` の契約であり、
        本ヘルパは #375 PR 3 で置き換えて削除する。
    """
    with _temp_environment("downloads", unique=True) as temp_dir:
        yield temp_dir


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
