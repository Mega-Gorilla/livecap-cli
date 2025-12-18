"""Resource management helpers for Live Cap core."""
from __future__ import annotations

from .model_manager import ModelManager
from .ffmpeg_manager import FFmpegManager, FFmpegNotFoundError
from .resource_locator import ResourceLocator

_model_manager: ModelManager | None = None
_ffmpeg_manager: FFmpegManager | None = None
_resource_locator: ResourceLocator | None = None


def get_model_manager(force_reset: bool = False) -> ModelManager:
    """Return a singleton ModelManager instance."""
    global _model_manager
    if force_reset or _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def get_ffmpeg_manager(force_reset: bool = False) -> FFmpegManager:
    """Return a singleton FFmpegManager instance."""
    global _ffmpeg_manager
    if force_reset or _ffmpeg_manager is None:
        _ffmpeg_manager = FFmpegManager()
    return _ffmpeg_manager


def get_resource_locator(force_reset: bool = False) -> ResourceLocator:
    """Return a singleton ResourceLocator instance."""
    global _resource_locator
    if force_reset or _resource_locator is None:
        _resource_locator = ResourceLocator()
    return _resource_locator


def reset_resource_managers() -> None:
    """
    Reset cached singleton instances.

    Intended for use in tests when environment overrides change.
    """
    global _model_manager, _ffmpeg_manager, _resource_locator
    _model_manager = None
    _ffmpeg_manager = None
    _resource_locator = None


# Phase 1 public surface for resource helpers (see docs/dev-docs/architecture/core-api-spec.md)
__all__ = [
    "ModelManager",
    "FFmpegManager",
    "FFmpegNotFoundError",
    "ResourceLocator",
    "get_model_manager",
    "get_ffmpeg_manager",
    "get_resource_locator",
    "reset_resource_managers",
]
