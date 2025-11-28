"""VAD Factory for benchmark.

Provides registry and factory functions for creating VAD backends.
"""

from __future__ import annotations

import logging
from typing import Any

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
    "webrtc_mode0": {
        "type": "protocol",
        "backend_class": "WebRTCVAD",
        "module": "livecap_core.vad.backends.webrtc",
        "params": {"mode": 0, "frame_duration_ms": 20},
    },
    "webrtc_mode1": {
        "type": "protocol",
        "backend_class": "WebRTCVAD",
        "module": "livecap_core.vad.backends.webrtc",
        "params": {"mode": 1, "frame_duration_ms": 20},
    },
    "webrtc_mode2": {
        "type": "protocol",
        "backend_class": "WebRTCVAD",
        "module": "livecap_core.vad.backends.webrtc",
        "params": {"mode": 2, "frame_duration_ms": 20},
    },
    "webrtc_mode3": {
        "type": "protocol",
        "backend_class": "WebRTCVAD",
        "module": "livecap_core.vad.backends.webrtc",
        "params": {"mode": 3, "frame_duration_ms": 20},
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


def create_vad(vad_id: str) -> VADBenchmarkBackend:
    """Create a VAD backend instance.

    Creates a new instance each time (no caching) to ensure
    clean state for each benchmark run.

    Args:
        vad_id: VAD identifier (key in VAD_REGISTRY)

    Returns:
        VADBenchmarkBackend instance

    Raises:
        ValueError: Unknown vad_id
        ImportError: Required package not installed
    """
    if vad_id not in VAD_REGISTRY:
        available = ", ".join(sorted(VAD_REGISTRY.keys()))
        raise ValueError(f"Unknown VAD: {vad_id}. Available: {available}")

    config = VAD_REGISTRY[vad_id]

    if config["type"] == "javad":
        return _create_javad(config)
    else:
        return _create_protocol_vad(config)


def _create_javad(config: dict) -> VADBenchmarkBackend:
    """Create JaVAD pipeline."""
    from .backends.javad import JaVADPipeline

    return JaVADPipeline(model=config["model"])


def _create_protocol_vad(config: dict) -> VADBenchmarkBackend:
    """Create Protocol-compliant VAD wrapped for benchmark."""
    import importlib

    # Dynamic import
    module = importlib.import_module(config["module"])
    backend_class = getattr(module, config["backend_class"])

    # Create backend instance
    backend = backend_class(**config["params"])

    # Wrap for benchmark interface
    return VADProcessorWrapper(backend)


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
    "create_vad",
    "get_all_vad_ids",
    "get_vad_config",
]
