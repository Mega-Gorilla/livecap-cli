"""Pytest fixtures for the non-speech filter evaluation harness (PR-0).

Thin wrappers around :mod:`benchmarks.non_speech_filter`. The benchmark
package owns the canonical corpus / metrics / pipeline definitions; this
conftest only provides pytest-specific glue (parametrization, skip-on-
missing-dep, baseline-output directory).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import pytest

from benchmarks.non_speech_filter import (
    CorpusItem,
    MockEngine,
    SUPPORTED_BACKENDS,
    build_pipeline,
    build_synthetic_corpus,
    load_real_corpus_items,
)
from livecap_cli.transcription.stream import StreamTranscriber

# Re-export so tests can ``from .conftest import MockEngine`` (kept for the
# existing test_baseline.py imports; the canonical home is
# ``benchmarks.non_speech_filter``).
__all__ = ["MockEngine"]


# ---------- VAD backend parametrization ----------------------------------


@pytest.fixture(params=SUPPORTED_BACKENDS)
def backend_type(request: pytest.FixtureRequest) -> str:
    """Parametrize over each supported VAD backend identifier."""
    return request.param


# ---------- Corpus fixtures ----------------------------------------------


@pytest.fixture(scope="session")
def synthetic_corpus_items() -> list[CorpusItem]:
    """Deterministic synthetic corpus built once per session."""
    return build_synthetic_corpus()


@pytest.fixture(scope="session")
def real_corpus_items() -> list[CorpusItem]:
    """Real audio corpus loaded from ``LIVECAP_NON_SPEECH_CORPUS_DIR``.

    Skipped when the environment variable is not set; the docs describe the
    manifest format used by ``load_real_corpus_items``.
    """
    env = os.environ.get("LIVECAP_NON_SPEECH_CORPUS_DIR")
    if not env:
        pytest.skip("LIVECAP_NON_SPEECH_CORPUS_DIR not set; real corpus disabled")
    directory = Path(env).expanduser().resolve()
    if not directory.exists():
        pytest.skip(f"LIVECAP_NON_SPEECH_CORPUS_DIR not found: {directory}")
    try:
        return load_real_corpus_items(directory)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


# ---------- Pipeline factory ---------------------------------------------


def _build_baseline(
    backend_type: str,
    *,
    mock_engine_factory: Callable[[], MockEngine] = MockEngine,
) -> tuple[StreamTranscriber, MockEngine]:
    """Construct one fresh baseline pipeline, skipping if the backend is unavailable."""
    try:
        return build_pipeline(backend_type, engine=mock_engine_factory())
    except ImportError as exc:
        pytest.skip(f"{backend_type} backend unavailable: {exc}")


@pytest.fixture
def transcriber_factory(backend_type: str) -> Callable[[], tuple[StreamTranscriber, MockEngine]]:
    """Factory producing a fresh baseline ``(transcriber, engine)`` pair.

    Each call constructs a fresh backend and engine so per-corpus state does
    not leak between items.
    """

    def _factory() -> tuple[StreamTranscriber, MockEngine]:
        return _build_baseline(backend_type)

    return _factory


@pytest.fixture
def baselines_dir() -> Path:
    """Directory where per-backend baseline JSON files are written."""
    path = Path(__file__).parent / "baselines"
    path.mkdir(parents=True, exist_ok=True)
    return path


# Public alias used by test_baseline.py's hallucination probe to construct
# a baseline pipeline with a custom MockEngine (different return_text).
def build_baseline(
    backend_type: str,
    *,
    mock_engine_factory: Callable[[], MockEngine] = MockEngine,
) -> tuple[StreamTranscriber, MockEngine]:
    """Construct a baseline pipeline, skipping if ``backend_type`` is unavailable."""
    return _build_baseline(backend_type, mock_engine_factory=mock_engine_factory)
