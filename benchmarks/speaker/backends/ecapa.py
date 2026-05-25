"""SpeechBrain ECAPA-TDNN speaker-embedding backend.

Uses ``speechbrain/spkrec-ecapa-voxceleb``. Requires the ``speaker-speechbrain``
extra (SpeechBrain toolkit is Apache-2.0; no HF gated token needed).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ECAPABackend:
    """Speaker embeddings from SpeechBrain ECAPA-TDNN."""

    def __init__(self, source: str = "speechbrain/spkrec-ecapa-voxceleb") -> None:
        self._source = source
        self._classifier = None
        self._torch = None
        self._device = "cpu"
        self._dim = 192

    def load(self, device: str) -> None:
        try:
            import torch
        except ImportError as e:  # pragma: no cover - import guard
            raise ImportError("ECAPA requires torch.") from e

        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            try:
                # Older SpeechBrain layout
                from speechbrain.pretrained import EncoderClassifier  # type: ignore
            except ImportError as e:  # pragma: no cover - import guard
                raise ImportError(
                    "ECAPA requires SpeechBrain. Install with: "
                    "uv pip install speechbrain  (or extra: speaker-speechbrain)"
                ) from e

        self._torch = torch
        self._device = device
        logger.info("Loading SpeechBrain ECAPA model: %s", self._source)
        self._classifier = EncoderClassifier.from_hparams(
            source=self._source,
            run_opts={"device": device},
        )

    def extract_embedding(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> np.ndarray:
        if self._classifier is None:
            raise RuntimeError("ECAPABackend.load() must be called first")

        torch = self._torch
        audio = np.asarray(audio, dtype=np.float32)
        signal = torch.from_numpy(audio).unsqueeze(0).to(self._device)

        with torch.no_grad():
            emb = self._classifier.encode_batch(signal)
        # encode_batch -> (batch, 1, dim)
        emb_np = emb.squeeze().detach().cpu().numpy().astype(np.float32)
        emb_np = np.atleast_1d(emb_np)
        self._dim = int(emb_np.shape[0])
        return emb_np

    @property
    def name(self) -> str:
        return "ecapa"

    @property
    def embedding_dim(self) -> int:
        return self._dim


__all__ = ["ECAPABackend"]
