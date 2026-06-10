"""EfficientAT inference smoke test (Issue #305 PR-D0 Phase D).

This is the only test that requires the manual EfficientAT clone — the
``efficientat_path`` fixture in ``conftest.py`` skips this module when the
clone is absent (CI runs that did not opt in to ``LIVECAP_SED_EFFICIENTAT_PATH``
will skip it silently).

Scope is intentionally minimal: load the smallest variant (``mn04_as``,
~3.88 MB), run a single inference on a short synthetic waveform, and verify
the output shape matches ``(n_windows, 527)``. Production-grade numeric
behaviour is exercised by the full pipeline in :mod:`benchmarks.sed.orchestrator`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmarks.sed.class_mapping import NUM_AUDIOSET_CLASSES
from benchmarks.sed.inference import (
    SED_WINDOW_SECONDS,
    compute_window_probs,
    load_model,
)


pytestmark = pytest.mark.sed_evaluation


def test_load_model_and_one_window_inference(
    efficientat_path: Path | None,
) -> None:
    if efficientat_path is None:
        pytest.skip(
            "EfficientAT clone not available; set LIVECAP_SED_EFFICIENTAT_PATH "
            "or clone to .tmp/EfficientAT (see benchmarks/sed/README.md)."
        )

    bundle = load_model(
        "mn04_as", device="cpu", efficientat_path=efficientat_path
    )
    assert bundle.variant == "mn04_as"
    assert bundle.device in {"cpu", "cuda"}

    # 1.5 s of 16 kHz silence: padding logic should produce 2 windows.
    duration_seconds = 1.5
    audio = np.zeros(int(duration_seconds * 16_000), dtype=np.float32)

    probs = compute_window_probs(audio, bundle)

    expected_windows = int(np.ceil(duration_seconds / SED_WINDOW_SECONDS))
    assert probs.shape == (expected_windows, NUM_AUDIOSET_CLASSES)
    assert probs.dtype == np.float32
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0), (
        "sigmoid output must lie in [0, 1]"
    )
