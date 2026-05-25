"""NVIDIA NeMo TitaNet speaker-embedding backend.

Uses ``EncDecSpeakerLabelModel`` (titanet_large by default). Available via the
``engines-nemo`` extra (already a dependency of the Parakeet engine).

License note: TitaNet model weights are CC-BY-4.0 (attribution required).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class TitaNetBackend:
    """Speaker embeddings from NeMo TitaNet."""

    def __init__(self, model_name: str = "titanet_large") -> None:
        self._model_name = model_name
        self._model = None
        self._torch = None
        self._device = "cpu"
        self._dim = 192  # titanet_large embedding size

    def load(self, device: str) -> None:
        try:
            import torch
            from nemo.collections.asr.models import EncDecSpeakerLabelModel
        except ImportError as e:  # pragma: no cover - import guard
            raise ImportError(
                "TitaNet requires NeMo. Install with: "
                "uv sync --extra engines-nemo --extra engines-torch"
            ) from e

        self._torch = torch
        self._device = device
        logger.info("Loading TitaNet model: %s", self._model_name)
        self._model = EncDecSpeakerLabelModel.from_pretrained(model_name=self._model_name)
        self._model = self._model.to(device)
        self._model.eval()

    def extract_embedding(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("TitaNetBackend.load() must be called first")

        torch = self._torch
        audio = np.asarray(audio, dtype=np.float32)
        signal = torch.from_numpy(audio).unsqueeze(0).to(self._device)
        length = torch.tensor([audio.shape[0]], device=self._device)

        with torch.no_grad():
            _logits, emb = self._model.forward(
                input_signal=signal, input_signal_length=length
            )
        emb_np = emb.squeeze(0).detach().cpu().numpy().astype(np.float32)
        self._dim = int(emb_np.shape[0])
        return emb_np

    @property
    def name(self) -> str:
        return "titanet"

    @property
    def embedding_dim(self) -> int:
        return self._dim


__all__ = ["TitaNetBackend"]
