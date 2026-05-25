"""Factory for speaker-embedding benchmark backends.

Mirrors ``benchmarks/vad/factory.py``: a registry of backend ids plus a
factory that instantiates them lazily.
"""

from __future__ import annotations

import importlib
from typing import Any

from .backends.base import SpeakerEmbeddingBackend

# Registry: backend id -> import location + default params.
SPEAKER_REGISTRY: dict[str, dict[str, Any]] = {
    "titanet": {
        "module": "benchmarks.speaker.backends.titanet",
        "class": "TitaNetBackend",
        "params": {},
        "license": "CC-BY-4.0 (attribution required)",
        "extra": "engines-nemo",
    },
    "ecapa": {
        "module": "benchmarks.speaker.backends.ecapa",
        "class": "ECAPABackend",
        "params": {},
        "license": "Apache-2.0 (toolkit)",
        "extra": "speaker-speechbrain",
    },
    "pyannote": {
        "module": "benchmarks.speaker.backends.pyannote",
        "class": "PyannoteBackend",
        "params": {},
        # Default = wespeaker-voxceleb-resnet34-LM (pyannote 3.1 pipeline embedding).
        # CC-BY-4.0, NOT gated (no token needed); weights derive from VoxCeleb
        # (commercial gray area).
        "license": "CC-BY-4.0 (VoxCeleb-derived; not gated)",
        "extra": "speaker-pyannote",
    },
    "mock": {
        "module": "benchmarks.speaker.backends.mock",
        "class": "MockEmbeddingBackend",
        "params": {},
        "license": "n/a (test backend)",
        "extra": None,
    },
}


def create_embedding_backend(
    backend_id: str, params: dict[str, Any] | None = None
) -> SpeakerEmbeddingBackend:
    """Instantiate a backend by id (does NOT call ``load()``).

    Args:
        backend_id: Key in ``SPEAKER_REGISTRY``.
        params: Optional overrides merged over registry defaults.

    Raises:
        ValueError: Unknown backend id.
    """
    if backend_id not in SPEAKER_REGISTRY:
        available = ", ".join(sorted(SPEAKER_REGISTRY))
        raise ValueError(f"Unknown backend: {backend_id}. Available: {available}")

    spec = SPEAKER_REGISTRY[backend_id]
    module = importlib.import_module(spec["module"])
    cls = getattr(module, spec["class"])

    merged = dict(spec.get("params", {}))
    if params:
        merged.update(params)
    return cls(**merged)


def get_all_backend_ids() -> list[str]:
    """All registered backend ids."""
    return list(SPEAKER_REGISTRY.keys())


def get_backend_info(backend_id: str) -> dict[str, Any]:
    """Registry entry for reporting (license/extra/etc.)."""
    if backend_id not in SPEAKER_REGISTRY:
        available = ", ".join(sorted(SPEAKER_REGISTRY))
        raise ValueError(f"Unknown backend: {backend_id}. Available: {available}")
    return SPEAKER_REGISTRY[backend_id].copy()


__all__ = [
    "SPEAKER_REGISTRY",
    "create_embedding_backend",
    "get_all_backend_ids",
    "get_backend_info",
]
