"""Canonical pipeline + backend helpers for the non-speech filter harness.

Used by both the pytest baseline tests (``tests/integration/non_speech_filter``)
and the ad-hoc benchmark runner. Centralising the backend factory + pipeline
construction keeps the gate / engine stack one ``Edit`` away from updating
both consumers — a property Phase 1 PR-B/C/A rely on.

These helpers raise ``ImportError`` or ``FileNotFoundError`` for missing
optional dependencies; callers in pytest catch those and convert to
``pytest.skip`` while the runner records them as skipped report entries.
"""

from __future__ import annotations

import json
import warnings
from math import gcd
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from livecap_cli.audio.noise_gate import NoiseGate
from livecap_cli.transcription.stream import StreamTranscriber
from livecap_cli.vad.config import VADConfig
from livecap_cli.vad.processor import VADProcessor

from .corpus import CorpusItem

if TYPE_CHECKING:
    from livecap_cli.vad.backends import VADBackend


SUPPORTED_BACKENDS: tuple[str, ...] = ("silero", "tenvad", "webrtc")
"""VAD backends evaluated by the harness. Single source of truth."""


# ---------- Backend construction ---------------------------------------------


def create_backend(backend_type: str) -> "VADBackend":
    """Construct a fresh VAD backend instance for ``backend_type``.

    Raises:
        ImportError: When the optional dependency for the requested backend
            is unavailable. Callers (pytest fixture, benchmark runner) decide
            whether this is a skip or a hard failure.
        ValueError: When ``backend_type`` is not a known identifier.

    The TenVAD license ``UserWarning`` is suppressed locally because the
    docs and CHANGELOG already document the license caveat; emitting it on
    every fresh-pipeline construction would drown the test/benchmark output.
    """
    if backend_type == "silero":
        from livecap_cli.vad.backends.silero import SileroVAD

        return SileroVAD(onnx=True)
    if backend_type == "tenvad":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            from livecap_cli.vad.backends.tenvad import TenVAD

            return TenVAD()
    if backend_type == "webrtc":
        from livecap_cli.vad.backends.webrtc import WebRTCVAD

        return WebRTCVAD()
    raise ValueError(
        f"Unknown VAD backend {backend_type!r}; "
        f"supported: {SUPPORTED_BACKENDS}"
    )


# ---------- Pipeline construction --------------------------------------------


def build_pipeline(
    backend_type: str,
    *,
    engine: Any,
    vad_config: VADConfig | None = None,
    enable_noise_gate: bool = True,
    enable_energy_gate: bool = True,
    noise_gate_threshold_db: float = -50.0,
    engine_min_rms_dbfs: float | None = None,
) -> tuple[StreamTranscriber, Any]:
    """Construct a fresh baseline pipeline (NoiseGate + VAD + EnergyGate).

    Args:
        backend_type: One of ``SUPPORTED_BACKENDS``.
        engine: Already-constructed engine instance (e.g. ``MockEngine`` or
            ``WhisperS2TEngine``). Caller owns its lifecycle.
        vad_config: Optional custom VAD config; defaults to ``VADConfig()``.
        enable_noise_gate: If False, no NoiseGate is attached.
        enable_energy_gate: If False, ``engine_min_rms_dbfs = -inf`` is
            forced so the EnergyGate is fully disabled.
        noise_gate_threshold_db: NoiseGate open threshold (dB).
        engine_min_rms_dbfs: Optional override for the EnergyGate threshold.
            ``None`` uses ``StreamTranscriber``'s built-in default (-45 dBFS).

    Returns:
        ``(StreamTranscriber, engine)``. The second element is the
        ``engine`` argument echoed back so callers can use a single
        ``(transcriber, engine) = build_pipeline(...)`` pattern without
        having to thread the engine separately.
    """
    backend = create_backend(backend_type)
    vad_processor = VADProcessor(
        config=vad_config or VADConfig(),
        backend=backend,
    )
    noise_gate = (
        NoiseGate(threshold_db=noise_gate_threshold_db, sample_rate=16000)
        if enable_noise_gate
        else None
    )
    kwargs: dict[str, Any] = {
        "engine": engine,
        "vad_processor": vad_processor,
        "noise_gate": noise_gate,
    }
    if engine_min_rms_dbfs is not None:
        kwargs["engine_min_rms_dbfs"] = engine_min_rms_dbfs
    elif not enable_energy_gate:
        kwargs["engine_min_rms_dbfs"] = float("-inf")
    return StreamTranscriber(**kwargs), engine


# ---------- Real-audio corpus loader -----------------------------------------


def load_real_corpus_items(directory: Path) -> list[CorpusItem]:
    """Load real audio fixtures from ``directory/manifest.json``.

    Schema documented in ``docs/benchmarks/non-speech-filter.md``: each
    manifest entry has ``file``, ``label``, ``kind`` (``"negative"`` /
    ``"positive"``) and optional ``is_short_utterance``. Files at any
    sample rate are mixed to mono and resampled to 16 kHz.

    Raises:
        FileNotFoundError: When ``manifest.json`` is missing.
        ImportError: When ``soundfile`` is not installed.
    """
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json missing in {directory}; "
            "see docs/benchmarks/non-speech-filter.md for the expected schema."
        )
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - environment dep
        raise ImportError(
            "soundfile is required for real corpus loading"
        ) from exc

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items: list[CorpusItem] = []
    for entry in manifest:
        audio_path = directory / entry["file"]
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            from scipy.signal import resample_poly

            g = gcd(int(sr), 16000)
            audio = resample_poly(audio, 16000 // g, int(sr) // g)
        items.append(
            CorpusItem(
                label=entry["label"],
                kind=entry["kind"],
                is_short_utterance=bool(entry.get("is_short_utterance", False)),
                audio=audio.astype(np.float32, copy=False),
            )
        )
    return items
