"""pyannote.audio speaker-embedding backend.

Uses ``pyannote/wespeaker-voxceleb-resnet34-LM`` by default — this is the
embedding model pyannote's speaker-diarization-3.1 pipeline uses, and is the
representative "pyannote" embedding in 4.x. (The legacy ``pyannote/embedding``
model is a separately-gated repo.)

Requires the ``speaker-pyannote`` extra AND Hugging Face access:

    1. Accept the model terms on its HF page (gated repo).
    2. Set HF_TOKEN / HUGGING_FACE_HUB_TOKEN, or run `hf auth login`.

If the token is missing or access is not granted, ``load()`` raises a clear
error and the runner skips this backend.

License note: pyannote model wrappers are MIT, but the weights derive from
VoxCeleb (research-oriented dataset → commercial use is a gray area), and gated
access applies.
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def _resolve_hf_token() -> str | None:
    """Resolve an HF token from env vars or a prior `hf auth login` / `huggingface-cli login`."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        token = os.environ.get(var)
        if token:
            return token
    # Fall back to the cached login token (huggingface_hub.get_token covers both
    # the env vars above and the credential stored by `hf auth login`).
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:  # pragma: no cover - very old hub without get_token
        return None


class PyannoteBackend:
    """Speaker embeddings from pyannote.audio (window='whole')."""

    def __init__(
        self, model_name: str = "pyannote/wespeaker-voxceleb-resnet34-LM"
    ) -> None:
        self._model_name = model_name
        self._inference = None
        self._torch = None
        self._device = "cpu"
        self._dim = 256

    def load(self, device: str) -> None:
        try:
            import torch
            from pyannote.audio import Inference, Model
        except ImportError as e:  # pragma: no cover - import guard
            raise ImportError(
                "pyannote backend requires pyannote.audio. Install with: "
                "uv pip install pyannote.audio  (or extra: speaker-pyannote)"
            ) from e

        token = _resolve_hf_token()
        if token is None:
            raise RuntimeError(
                f"{self._model_name} is gated. Accept terms at "
                f"https://huggingface.co/{self._model_name} and set HF_TOKEN "
                "(or run `hf auth login`)."
            )

        self._torch = torch
        self._device = device
        logger.info("Loading pyannote model: %s", self._model_name)
        try:
            # pyannote 4.x uses `token=`; 3.x uses `use_auth_token=`.
            try:
                model = Model.from_pretrained(self._model_name, token=token)
            except TypeError:
                model = Model.from_pretrained(self._model_name, use_auth_token=token)
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ("gated", "401", "403", "awaiting", "authorized")):
                raise RuntimeError(
                    f"{self._model_name} access denied. Accept the model terms at "
                    f"https://huggingface.co/{self._model_name} with the logged-in "
                    "account, then retry."
                ) from e
            raise
        if model is None:
            raise RuntimeError(
                "pyannote returned no model (check access to pyannote/embedding)."
            )
        self._inference = Inference(model, window="whole", device=torch.device(device))

    def extract_embedding(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> np.ndarray:
        if self._inference is None:
            raise RuntimeError("PyannoteBackend.load() must be called first")

        torch = self._torch
        audio = np.asarray(audio, dtype=np.float32)
        waveform = torch.from_numpy(audio).unsqueeze(0)  # (channel=1, samples)
        emb = self._inference(
            {"waveform": waveform, "sample_rate": sample_rate}
        )
        emb_np = np.asarray(emb, dtype=np.float32).reshape(-1)
        self._dim = int(emb_np.shape[0])
        return emb_np

    @property
    def name(self) -> str:
        return "pyannote"

    @property
    def embedding_dim(self) -> int:
        return self._dim


__all__ = ["PyannoteBackend"]
