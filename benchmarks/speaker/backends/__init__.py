"""Speaker-embedding benchmark backends.

Backends are imported lazily to avoid pulling heavy dependencies (NeMo,
SpeechBrain, pyannote.audio) unless actually used.
"""

from __future__ import annotations

from .base import SpeakerEmbeddingBackend


def __getattr__(name: str):
    """Lazy import of concrete backends."""
    if name == "MockEmbeddingBackend":
        from .mock import MockEmbeddingBackend

        return MockEmbeddingBackend
    if name == "TitaNetBackend":
        from .titanet import TitaNetBackend

        return TitaNetBackend
    if name == "ECAPABackend":
        from .ecapa import ECAPABackend

        return ECAPABackend
    if name == "PyannoteBackend":
        from .pyannote import PyannoteBackend

        return PyannoteBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SpeakerEmbeddingBackend",
    "MockEmbeddingBackend",
    "TitaNetBackend",
    "ECAPABackend",
    "PyannoteBackend",
]
