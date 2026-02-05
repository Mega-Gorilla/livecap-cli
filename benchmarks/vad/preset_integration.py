"""Preset integration for VAD benchmark.

Provides helper functions to create VAD backends with optimized presets
from livecap_cli.vad.presets.

Usage:
    from benchmarks.vad.preset_integration import (
        create_vad_with_preset,
        get_preset_vad_ids,
        OPTIMIZABLE_VADS,
    )

    # Create VAD with optimized parameters
    vad = create_vad_with_preset("silero", "ja")

    # Get VAD IDs that have presets
    vad_ids = get_preset_vad_ids()  # ["silero", "tenvad", "webrtc"]
"""

from __future__ import annotations

import logging
from typing import Any

from livecap_cli.vad.config import VADConfig
from livecap_cli.vad.presets import (
    VAD_OPTIMIZED_PRESETS,
    get_available_presets,
    get_optimized_preset,
)

from .backends import VADBenchmarkBackend
from .factory import create_vad

__all__ = [
    "OPTIMIZABLE_VADS",
    "create_vad_with_preset",
    "get_preset_vad_ids",
    "get_preset_config",
    "is_preset_available",
]

logger = logging.getLogger(__name__)

# VAD types that have optimized presets
# Note: javad_* are excluded because they don't support external parameter tuning
OPTIMIZABLE_VADS = ["silero", "tenvad", "webrtc"]


def get_preset_vad_ids() -> list[str]:
    """Get VAD IDs that have optimized presets.

    Returns:
        List of VAD type identifiers (e.g., ["silero", "tenvad", "webrtc"])

    Note:
        These are base VAD types, not mode variants (e.g., "webrtc" not "webrtc_mode0").
        For preset mode, use these IDs directly. The preset specifies the optimal
        backend parameters (including mode for WebRTC).
    """
    return list(VAD_OPTIMIZED_PRESETS.keys())


def is_preset_available(vad_type: str, language: str) -> bool:
    """Check if a preset is available for VAD type and language.

    Args:
        vad_type: VAD backend type (e.g., "silero", "tenvad", "webrtc")
        language: Language code (e.g., "ja", "en")

    Returns:
        True if preset exists, False otherwise
    """
    return get_optimized_preset(vad_type, language) is not None


def get_preset_config(vad_type: str, language: str) -> dict[str, Any] | None:
    """Get preset configuration for VAD type and language.

    Args:
        vad_type: VAD backend type
        language: Language code

    Returns:
        Preset dictionary with "vad_config", optional "backend", and "metadata" keys.
        Returns None if no preset exists.
    """
    return get_optimized_preset(vad_type, language)


def create_vad_with_preset(
    vad_type: str,
    language: str,
) -> VADBenchmarkBackend:
    """Create a VAD backend with optimized preset parameters.

    Loads the Bayesian-optimized parameters from livecap_cli/vad/presets/
    and creates a VAD backend configured with those parameters.

    Args:
        vad_type: VAD backend type ("silero", "tenvad", "webrtc")
        language: Language code ("ja", "en")

    Returns:
        VADBenchmarkBackend instance configured with optimized parameters

    Raises:
        ValueError: If no preset exists for the vad_type/language combination
        ImportError: If required VAD package is not installed

    Example:
        >>> vad = create_vad_with_preset("silero", "ja")
        >>> # Uses optimized threshold=0.294, neg_threshold=0.123, etc.
    """
    preset = get_optimized_preset(vad_type, language)
    if preset is None:
        available = get_available_presets()
        available_str = ", ".join(f"{v}/{l}" for v, l in available)
        raise ValueError(
            f"No preset for {vad_type}/{language}. "
            f"Available: {available_str}"
        )

    # Extract backend-specific parameters (if any)
    backend_params = preset.get("backend", {})

    # Extract VADConfig parameters
    vad_config_dict = preset.get("vad_config", {})
    vad_config = VADConfig.from_dict(vad_config_dict)

    logger.debug(
        f"Creating {vad_type} for {language} with preset: "
        f"backend={backend_params}, vad_config={vad_config_dict}"
    )

    # Create VAD with preset parameters
    return create_vad(vad_type, backend_params=backend_params, vad_config=vad_config)
