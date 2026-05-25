"""pyannote.audio speaker-embedding backend.

Uses ``pyannote/wespeaker-voxceleb-resnet34-LM`` by default — this is the
embedding model pyannote's speaker-diarization-3.1 pipeline uses, and is the
representative "pyannote" embedding in 4.x. (The legacy ``pyannote/embedding``
model is a separately-gated repo.)

Requires the ``speaker-pyannote`` extra. The default model is **CC-BY-4.0 and
NOT gated**, so no Hugging Face token is needed:

    model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")

A token is used only if available (e.g. to point ``model_name`` at a gated
repo). If a gated model's access is denied, ``load()`` raises a clear error and
the runner skips this backend.

License note: the weights derive from VoxCeleb (CC-BY-4.0; research-oriented
dataset → commercial use is a gray area).
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

        # The default model (wespeaker-voxceleb-resnet34-LM) is CC-BY-4.0 and NOT
        # gated, so a token is optional. We still pass one if available (so a
        # gated model can be used via override), and skip gracefully on a 403.
        token = _resolve_hf_token()

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
                    f"{self._model_name} is gated and access was denied. Accept the "
                    f"model terms at https://huggingface.co/{self._model_name} and set "
                    "HF_TOKEN (or run `hf auth login`), then retry."
                ) from e
            raise
        if model is None:
            raise RuntimeError(
                f"pyannote returned no model (check access to {self._model_name})."
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
