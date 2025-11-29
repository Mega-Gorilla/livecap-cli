"""VAD Factory for benchmark.

Provides registry and factory functions for creating VAD backends.
"""

from __future__ import annotations

import logging
from typing import Any

from livecap_core.vad.config import VADConfig

from .backends import VADBenchmarkBackend, VADProcessorWrapper

logger = logging.getLogger(__name__)


# Registry: VAD configuration definitions
VAD_REGISTRY: dict[str, dict[str, Any]] = {
    # Protocol-compliant VADs (use VADProcessorWrapper)
    "silero": {
        "type": "protocol",
        "backend_class": "SileroVAD",
        "module": "livecap_core.vad.backends.silero",
        "params": {"threshold": 0.5, "onnx": True},
    },
    # WebRTC VAD (mode specified via backend_params)
    # Mode 0-3: aggressiveness level (0=quality, 3=aggressive)
    # Use backend_params={"mode": N} to override default
    "webrtc": {
        "type": "protocol",
        "backend_class": "WebRTCVAD",
        "module": "livecap_core.vad.backends.webrtc",
        "params": {"mode": 0, "frame_duration_ms": 20},
    },
    "tenvad": {
        "type": "protocol",
        "backend_class": "TenVAD",
        "module": "livecap_core.vad.backends.tenvad",
        "params": {"hop_size": 256, "threshold": 0.5},
    },
    # JaVAD (directly implements process_audio)
    "javad_tiny": {
        "type": "javad",
        "model": "tiny",
    },
    "javad_balanced": {
        "type": "javad",
        "model": "balanced",
    },
    "javad_precise": {
        "type": "javad",
        "model": "precise",
    },
}


def create_vad(
    vad_id: str,
    backend_params: dict[str, Any] | None = None,
    vad_config: VADConfig | None = None,
) -> VADBenchmarkBackend:
    """Create a VAD backend instance.

    Creates a new instance each time (no caching) to ensure
    clean state for each benchmark run.

    Args:
        vad_id: VAD identifier (key in VAD_REGISTRY)
        backend_params: Custom backend parameters (overrides registry defaults).
            For protocol VADs: threshold, mode, frame_duration_ms, hop_size, etc.
            For JaVAD: model (tiny, balanced, precise)
        vad_config: Custom VADConfig for segment detection parameters.
            Only applies to protocol-compliant VADs (Silero, WebRTC, TenVAD).
            JaVAD does not support VADConfig.

    Returns:
        VADBenchmarkBackend instance

    Raises:
        ValueError: Unknown vad_id
        ImportError: Required package not installed

    Example:
        # Default parameters
        vad = create_vad("silero")

        # Custom backend parameters
        vad = create_vad("silero", backend_params={"threshold": 0.7})

        # Custom VADConfig for segment detection
        config = VADConfig(min_speech_ms=300, min_silence_ms=200)
        vad = create_vad("silero", vad_config=config)

        # Both custom backend params and VADConfig
        vad = create_vad(
            "tenvad",
            backend_params={"hop_size": 160, "threshold": 0.6},
            vad_config=VADConfig(min_speech_ms=200),
        )
    """
    if vad_id not in VAD_REGISTRY:
        available = ", ".join(sorted(VAD_REGISTRY.keys()))
        raise ValueError(f"Unknown VAD: {vad_id}. Available: {available}")

    registry_config = VAD_REGISTRY[vad_id]

    if registry_config["type"] == "javad":
        return _create_javad(registry_config, backend_params)
    else:
        return _create_protocol_vad(registry_config, backend_params, vad_config)


def _create_javad(
    registry_config: dict,
    backend_params: dict[str, Any] | None = None,
) -> VADBenchmarkBackend:
    """Create JaVAD pipeline.

    Args:
        registry_config: Configuration from VAD_REGISTRY
        backend_params: Optional custom parameters (model)
    """
    from .backends.javad import JaVADPipeline

    # Use custom model if provided, otherwise use registry default
    model = registry_config["model"]
    if backend_params and "model" in backend_params:
        model = backend_params["model"]

    return JaVADPipeline(model=model)


def _create_protocol_vad(
    registry_config: dict,
    backend_params: dict[str, Any] | None = None,
    vad_config: VADConfig | None = None,
) -> VADBenchmarkBackend:
    """Create Protocol-compliant VAD wrapped for benchmark.

    Args:
        registry_config: Configuration from VAD_REGISTRY
        backend_params: Optional custom backend parameters (overrides registry defaults)
        vad_config: Optional VADConfig for segment detection
    """
    import importlib

    # Dynamic import
    module = importlib.import_module(registry_config["module"])
    backend_class = getattr(module, registry_config["backend_class"])

    # Merge registry defaults with custom params
    params = registry_config["params"].copy()
    if backend_params:
        params.update(backend_params)

    # Create backend instance
    backend = backend_class(**params)

    # Wrap for benchmark interface
    return VADProcessorWrapper(backend, config=vad_config)


def get_all_vad_ids() -> list[str]:
    """Get all available VAD IDs.

    Returns:
        List of VAD identifiers
    """
    return list(VAD_REGISTRY.keys())


def get_vad_config(vad_id: str) -> dict[str, Any]:
    """Get VAD configuration from registry.

    Args:
        vad_id: VAD identifier

    Returns:
        Configuration dictionary

    Raises:
        ValueError: Unknown vad_id
    """
    if vad_id not in VAD_REGISTRY:
        available = ", ".join(sorted(VAD_REGISTRY.keys()))
        raise ValueError(f"Unknown VAD: {vad_id}. Available: {available}")

    return VAD_REGISTRY[vad_id].copy()


__all__ = [
    "VAD_REGISTRY",
    "VADConfig",
    "create_vad",
    "get_all_vad_ids",
    "get_vad_config",
]
