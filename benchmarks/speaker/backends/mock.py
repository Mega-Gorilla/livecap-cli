"""Mock speaker-embedding backend for tests.

Produces deterministic embeddings from cheap audio features (no heavy model,
no GPU). Designed so that audio with distinct dominant frequencies maps to
well-separated embeddings, letting separability metrics be tested meaningfully.
"""

from __future__ import annotations

import numpy as np


class MockEmbeddingBackend:
    """Deterministic embedding backend with zero heavy dependencies.

    The embedding is the magnitude spectrum of the audio resampled into
    ``embedding_dim`` frequency bins. Two pure tones at different frequencies
    therefore yield clearly separable embeddings.
    """

    def __init__(self, embedding_dim: int = 32) -> None:
        self._dim = embedding_dim
        self._loaded = False

    def load(self, device: str) -> None:  # noqa: ARG002 - device unused for mock
        self._loaded = True

    def extract_embedding(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> np.ndarray:
        if not self._loaded:
            raise RuntimeError("MockEmbeddingBackend.load() must be called first")

        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return np.zeros(self._dim, dtype=np.float32)

        # Magnitude spectrum, binned to embedding_dim.
        spectrum = np.abs(np.fft.rfft(audio))
        if spectrum.size < self._dim:
            spectrum = np.pad(spectrum, (0, self._dim - spectrum.size))
        # Average-pool into exactly embedding_dim bins.
        bins = np.array_split(spectrum, self._dim)
        emb = np.array([float(b.mean()) if b.size else 0.0 for b in bins], dtype=np.float32)

        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    @property
    def name(self) -> str:
        return "mock"

    @property
    def embedding_dim(self) -> int:
        return self._dim


__all__ = ["MockEmbeddingBackend"]
