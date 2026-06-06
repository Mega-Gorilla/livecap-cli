"""Pytest fixtures for the non-speech filter evaluation harness (PR-0).

Provides:
- MockEngine: instrumented engine that tracks transcribe_count and outputs.
- backend_type / vad_backend_factory: parametrized fixture covering all
  available backends (Silero / TenVAD / WebRTC); missing backends skip.
- synthetic_corpus_items: deterministic synthetic corpus built once per session.
- real_corpus_items: optionally loads WAV+kind metadata from a directory
  pointed to by ``LIVECAP_NON_SPEECH_CORPUS_DIR``; otherwise skipped.
- transcriber_factory: zero-arg factory used by ``evaluate_pipeline``.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pytest

from livecap_cli.audio.noise_gate import NoiseGate
from livecap_cli.transcription.stream import StreamTranscriber
from livecap_cli.vad.config import VADConfig
from livecap_cli.vad.processor import VADProcessor

from .corpus import CorpusItem, build_synthetic_corpus


# ---------- MockEngine ----------------------------------------------------


class MockEngine:
    """Engine stand-in for filter-behavior measurement.

    Tracks ``transcribe_count`` (incremented on every ``transcribe()`` call)
    and stores returned text in ``last_texts``. Default return text is empty
    so the result coalescer / translation path treats this as a non-emitting
    transcription — we only care whether the engine was invoked.
    """

    def __init__(self, return_text: str = "", sample_rate: int = 16000) -> None:
        self._return_text = return_text
        self._sample_rate = sample_rate
        self.transcribe_count = 0
        self.last_texts: list[str] = []
        self.last_audio_durations: list[float] = []

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> tuple[str, float]:
        self.transcribe_count += 1
        duration = len(audio) / max(1, sample_rate)
        self.last_audio_durations.append(duration)
        self.last_texts.append(self._return_text)
        return self._return_text, 1.0

    def get_required_sample_rate(self) -> int:
        return self._sample_rate

    def get_engine_name(self) -> str:
        return "mock"

    def get_supported_languages(self) -> list[str]:
        return ["en", "ja"]

    def cleanup(self) -> None:
        pass


# ---------- VAD backend fixtures -----------------------------------------


_BACKEND_IDS = ("silero", "tenvad", "webrtc")


def _create_backend(backend_type: str):
    """Return a fresh ``VADBackend`` instance for ``backend_type``.

    Skips the test if the optional dependency is unavailable. TenVAD's
    license warning is captured so it does not pollute pytest output.
    """
    if backend_type == "silero":
        try:
            from livecap_cli.vad.backends.silero import SileroVAD
        except ImportError as exc:  # pragma: no cover - environment dep
            pytest.skip(f"SileroVAD backend unavailable: {exc}")
        return SileroVAD(onnx=True)
    if backend_type == "tenvad":
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                from livecap_cli.vad.backends.tenvad import TenVAD

                return TenVAD()
        except ImportError as exc:  # pragma: no cover
            pytest.skip(f"TenVAD backend unavailable: {exc}")
    if backend_type == "webrtc":
        try:
            from livecap_cli.vad.backends.webrtc import WebRTCVAD
        except ImportError as exc:  # pragma: no cover
            pytest.skip(f"WebRTCVAD backend unavailable: {exc}")
        return WebRTCVAD()
    raise ValueError(f"Unknown backend type: {backend_type!r}")


@pytest.fixture(params=_BACKEND_IDS)
def backend_type(request: pytest.FixtureRequest) -> str:
    """Parametrize over each supported VAD backend identifier."""
    return request.param


# ---------- Corpus fixtures ----------------------------------------------


@pytest.fixture(scope="session")
def synthetic_corpus_items() -> list[CorpusItem]:
    """Deterministic synthetic corpus built once per session."""
    return build_synthetic_corpus()


def _load_real_corpus_items(directory: Path) -> list[CorpusItem]:
    """Load real audio fixtures from ``directory``.

    Expected layout:
    ``directory/manifest.json`` — JSON list of
    ``{"file": "...", "label": "...", "kind": "negative"|"positive",
       "is_short_utterance": bool}``.
    Audio files must be 16 kHz mono WAV. Stereo input is mixed to mono.
    """
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json missing in {directory}; "
            "see docs/benchmarks/non-speech-filter.md for the expected schema."
        )
    try:
        import soundfile as sf  # local import — optional dep
    except ImportError as exc:  # pragma: no cover - environment dep
        raise ImportError(
            "soundfile is required for real corpus loading"
        ) from exc
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items: list[CorpusItem] = []
    for entry in manifest:
        rel_path = entry["file"]
        kind = entry["kind"]
        label = entry["label"]
        is_short = bool(entry.get("is_short_utterance", False))
        audio_path = directory / rel_path
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            from scipy.signal import resample_poly

            from math import gcd

            g = gcd(int(sr), 16000)
            audio = resample_poly(audio, 16000 // g, int(sr) // g)
        audio = audio.astype(np.float32, copy=False)
        items.append(
            CorpusItem(
                label=label,
                kind=kind,
                is_short_utterance=is_short,
                audio=audio,
            )
        )
    return items


@pytest.fixture(scope="session")
def real_corpus_items() -> list[CorpusItem]:
    """Real audio corpus loaded from ``LIVECAP_NON_SPEECH_CORPUS_DIR``.

    Skipped when the environment variable is not set; the docs describe the
    manifest format used by ``_load_real_corpus_items``.
    """
    env = os.environ.get("LIVECAP_NON_SPEECH_CORPUS_DIR")
    if not env:
        pytest.skip("LIVECAP_NON_SPEECH_CORPUS_DIR not set; real corpus disabled")
    directory = Path(env).expanduser().resolve()
    if not directory.exists():
        pytest.skip(f"LIVECAP_NON_SPEECH_CORPUS_DIR not found: {directory}")
    return _load_real_corpus_items(directory)


# ---------- Pipeline factory ---------------------------------------------


def _build_baseline_transcriber(
    backend_type: str,
    *,
    mock_engine_factory: Callable[[], MockEngine] = MockEngine,
    enable_noise_gate: bool = True,
    enable_energy_gate: bool = True,
    noise_gate_threshold_db: float = -50.0,
    engine_min_rms_dbfs: Optional[float] = None,
    vad_config: Optional[VADConfig] = None,
) -> tuple[StreamTranscriber, MockEngine]:
    """Construct one fresh baseline pipeline.

    ``engine_min_rms_dbfs=None`` → uses StreamTranscriber's default (-45 dBFS).
    """
    engine = mock_engine_factory()
    backend = _create_backend(backend_type)
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
    transcriber = StreamTranscriber(**kwargs)
    return transcriber, engine


@pytest.fixture
def transcriber_factory(backend_type: str) -> Callable[[], tuple[StreamTranscriber, MockEngine]]:
    """Factory producing a fresh baseline ``(transcriber, engine)`` pair.

    Each call constructs a new backend instance and engine so per-corpus
    state does not leak. Returned by parametrized ``backend_type``.
    """

    def _factory() -> tuple[StreamTranscriber, MockEngine]:
        return _build_baseline_transcriber(backend_type)

    return _factory


@pytest.fixture
def baselines_dir() -> Path:
    """Directory where per-backend baseline JSON files are written."""
    path = Path(__file__).parent / "baselines"
    path.mkdir(parents=True, exist_ok=True)
    return path
