"""EfficientAT model loader + 1-second window inference (Phase D).

PR-D0 scope: research-only off-line evaluation. The model is loaded from a
manual clone of ``fschmid56/EfficientAT`` (path resolved via
``LIVECAP_SED_EFFICIENTAT_PATH``); we do **not** add a runtime dependency on
the repository for production code.

EfficientAT's ``helpers/utils.py`` reads ``metadata/class_labels_indices.csv``
at module-import time using a path relative to the current working directory.
We therefore enter the EfficientAT directory while importing and loading the
checkpoint, then restore the original working directory before returning.

Inference path:

    16 kHz mono waveform → librosa.resample → 32 kHz waveform
    → 1-second slices (= 32000 samples each, zero-padded last slice)
    → AugmentMelSTFT (128 mel bands)
    → MN / DyMN model forward → 527-dim logits
    → sigmoid → per-class probabilities

The function :func:`compute_window_probs` returns a ``(n_windows, 527)``
matrix matching the metric calculation unit pinned in Issue #305 v3
(``window-level primary``).
"""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


EnvironmentName = Literal["mn04_as", "dymn04_as", "dymn10_as"]


# Canonical EfficientAT preprocessing parameters (from inference.py).
SED_TARGET_SAMPLE_RATE = 32_000
SED_MEL_WINDOW_SAMPLES = 800
SED_MEL_HOP_SAMPLES = 320
SED_N_MELS = 128
SED_WINDOW_SECONDS = 1.0
"""1-second window — matches Issue #305 v3 ``metric calculation unit``."""


@dataclass(frozen=True)
class LoadedSED:
    """Loaded EfficientAT model bundle ready for inference."""

    variant: EnvironmentName
    model: object  # torch.nn.Module, but we avoid the import at module top
    mel: object  # torch.nn.Module (AugmentMelSTFT)
    device: str


def resolve_efficientat_path(override: Path | None = None) -> Path:
    """Return the path to the EfficientAT clone, raising if it is absent."""

    if override is not None:
        candidate = Path(override)
    else:
        env_value = os.environ.get("LIVECAP_SED_EFFICIENTAT_PATH")
        if env_value:
            candidate = Path(env_value)
        else:
            candidate = Path(".tmp/EfficientAT")

    candidate = candidate.resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(
            f"EfficientAT clone not found at {candidate}. See "
            "benchmarks/sed/README.md for setup."
        )
    metadata = candidate / "metadata" / "class_labels_indices.csv"
    if not metadata.is_file():
        raise FileNotFoundError(
            f"EfficientAT clone at {candidate} is incomplete (missing {metadata})"
        )
    return candidate


@contextlib.contextmanager
def _enter_efficientat(path: Path) -> Iterator[None]:
    """Run a block with cwd + ``sys.path`` set up for EfficientAT imports.

    Reverts both on exit even if the body raises.
    """

    saved_cwd = os.getcwd()
    inserted_path = str(path)
    os.chdir(path)
    sys.path.insert(0, inserted_path)
    try:
        yield
    finally:
        try:
            sys.path.remove(inserted_path)
        except ValueError:  # pragma: no cover - defensive
            pass
        os.chdir(saved_cwd)


def resolve_device(device: str) -> str:
    """Map ``auto`` to the best available device, otherwise pass through."""

    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_model(
    variant: EnvironmentName,
    device: str = "cpu",
    efficientat_path: Path | None = None,
) -> LoadedSED:
    """Load an EfficientAT model + its mel preprocessing pipeline.

    The first call for a given variant downloads the checkpoint via
    ``torch.hub`` into ``<efficientat_path>/resources/`` (the EfficientAT
    default). Subsequent calls reuse the cached file.
    """

    import torch

    path = resolve_efficientat_path(efficientat_path)
    resolved_device = resolve_device(device)

    with _enter_efficientat(path):
        from helpers.utils import NAME_TO_WIDTH  # type: ignore[import-not-found]
        from models.mn.model import get_model as get_mobilenet  # type: ignore[import-not-found]
        from models.dymn.model import get_model as get_dymn  # type: ignore[import-not-found]
        from models.preprocess import AugmentMelSTFT  # type: ignore[import-not-found]

        if variant.startswith("dymn"):
            model = get_dymn(
                width_mult=NAME_TO_WIDTH(variant), pretrained_name=variant
            )
        else:
            model = get_mobilenet(
                width_mult=NAME_TO_WIDTH(variant), pretrained_name=variant
            )

        mel = AugmentMelSTFT(
            n_mels=SED_N_MELS,
            sr=SED_TARGET_SAMPLE_RATE,
            win_length=SED_MEL_WINDOW_SAMPLES,
            hopsize=SED_MEL_HOP_SAMPLES,
        )

    torch_device = torch.device(resolved_device)
    model.to(torch_device).eval()
    mel.to(torch_device).eval()

    return LoadedSED(variant=variant, model=model, mel=mel, device=resolved_device)


def _iterate_windows(
    waveform: np.ndarray, window_samples: int
) -> Iterator[np.ndarray]:
    """Yield consecutive non-overlapping windows, zero-padding the final one."""

    n = waveform.shape[0]
    if n == 0:
        return
    n_windows = max(1, (n + window_samples - 1) // window_samples)
    for i in range(n_windows):
        start = i * window_samples
        end = start + window_samples
        chunk = waveform[start:end]
        if chunk.shape[0] < window_samples:
            padded = np.zeros(window_samples, dtype=waveform.dtype)
            padded[: chunk.shape[0]] = chunk
            chunk = padded
        yield chunk


def compute_window_probs(
    audio_16k: np.ndarray,
    bundle: LoadedSED,
) -> np.ndarray:
    """Compute per-window AudioSet probabilities for a 16 kHz mono waveform.

    Returns a ``(n_windows, 527)`` array of sigmoid-activated probabilities,
    one row per 1-second slice (the last slice is zero-padded if the input
    duration is not a multiple of 1 s).
    """

    import librosa
    import torch

    if audio_16k.ndim != 1:
        raise ValueError(
            f"audio_16k must be a 1-D mono waveform, got shape {audio_16k.shape}"
        )

    waveform_32k = librosa.resample(
        audio_16k.astype(np.float32, copy=False),
        orig_sr=16_000,
        target_sr=SED_TARGET_SAMPLE_RATE,
        res_type="kaiser_best",
    )
    window_samples = int(SED_WINDOW_SECONDS * SED_TARGET_SAMPLE_RATE)

    device = torch.device(bundle.device)
    rows: list[np.ndarray] = []

    with torch.no_grad():
        for chunk in _iterate_windows(waveform_32k, window_samples):
            tensor = torch.from_numpy(chunk).unsqueeze(0).to(device)
            spec = bundle.mel(tensor)
            preds, _features = bundle.model(spec.unsqueeze(0))
            probs = torch.sigmoid(preds.float()).squeeze().cpu().numpy()
            rows.append(probs.astype(np.float32, copy=False))

    return np.stack(rows, axis=0) if rows else np.zeros((0, 527), dtype=np.float32)


__all__ = [
    "SED_TARGET_SAMPLE_RATE",
    "SED_WINDOW_SECONDS",
    "SED_MEL_WINDOW_SAMPLES",
    "SED_MEL_HOP_SAMPLES",
    "SED_N_MELS",
    "LoadedSED",
    "resolve_efficientat_path",
    "resolve_device",
    "load_model",
    "compute_window_probs",
]
