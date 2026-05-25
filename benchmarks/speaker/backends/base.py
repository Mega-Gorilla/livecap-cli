"""Speaker embedding backend protocol for benchmarking.

Provides a unified interface for speaker-embedding extractors (TitaNet, ECAPA,
pyannote) so they can be measured with the same runner.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class SpeakerEmbeddingBackend(Protocol):
    """Unified interface for speaker-embedding backends in benchmarking.

    Contract:
        - ``load(device)`` is called once before any extraction. Heavy model
          loading happens here (so load time / model GPU memory is measured).
        - ``extract_embedding`` receives 16 kHz mono float32 audio. The
          ``sample_rate`` is passed explicitly and is always 16000 in this
          benchmark (segments originate from the 16 kHz VAD pipeline).

    Example:
        backend = create_embedding_backend("titanet")
        backend.load("cuda")
        emb = backend.extract_embedding(segment_audio, sample_rate=16000)
    """

    def load(self, device: str) -> None:
        """Load the model onto ``device`` ("cuda" or "cpu")."""
        ...

    def extract_embedding(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> np.ndarray:
        """Return a 1-D speaker embedding for the given audio.

        Args:
            audio: float32 mono audio in [-1.0, 1.0].
            sample_rate: Sample rate in Hz (always 16000 in this benchmark).

        Returns:
            1-D ``np.ndarray`` of shape ``(embedding_dim,)``.
        """
        ...

    @property
    def name(self) -> str:
        """Backend identifier for reporting."""
        ...

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the produced embedding (after load)."""
        ...


__all__ = ["SpeakerEmbeddingBackend"]
