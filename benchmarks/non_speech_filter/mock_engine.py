"""Engine adapters used by the non-speech filter harness.

Two adapters share the same ``transcribe_count`` / ``last_texts`` surface so
the metric layer can stay engine-agnostic:

- ``MockEngine``: standalone stand-in, used when no real engine is needed.
- ``InstrumentedEngine``: thin wrapper that delegates to a real
  ``TranscriptionEngine`` while recording calls and outputs. This is what
  makes ``non_empty_hallucination_rate`` actually measurable for
  ``--engine whispers2t`` style runs.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class MockEngine:
    """Engine stand-in for filter-behavior measurement.

    Tracks ``transcribe_count`` (incremented on every ``transcribe`` call)
    and stores returned text in ``last_texts``. Default return text is empty
    so the result coalescer / translation path treats this as a non-emitting
    transcription — we only care whether the engine was invoked. Set
    ``return_text`` to simulate engine output for hallucination-rate probes.
    """

    def __init__(self, return_text: str = "", sample_rate: int = 16000) -> None:
        self._return_text = return_text
        self._sample_rate = sample_rate
        self.transcribe_count: int = 0
        self.last_texts: list[str] = []

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> tuple[str, float]:
        self.transcribe_count += 1
        self.last_texts.append(self._return_text)
        return self._return_text, 1.0

    def get_required_sample_rate(self) -> int:
        return self._sample_rate

    def get_engine_name(self) -> str:
        return "mock"

    def get_supported_languages(self) -> list[str]:
        return ["en", "ja"]

    def cleanup(self) -> None:  # pragma: no cover - StreamTranscriber may call
        pass


class InstrumentedEngine:
    """Transparent wrapper that adds ``transcribe_count`` / ``last_texts``.

    Real engines (``WhisperS2TEngine``, ``ParakeetEngine``, ...) do not
    expose call counters or output history. Wrapping them via
    :class:`InstrumentedEngine` lets :func:`evaluate_pipeline` measure
    ``non_empty_hallucination_rate`` and engine-call counts using the same
    code path it uses for :class:`MockEngine`.

    All non-``transcribe`` methods delegate to the wrapped instance so the
    wrapper behaves identically to the underlying engine from the
    :class:`StreamTranscriber` perspective.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.transcribe_count: int = 0
        self.last_texts: list[str] = []

    @property
    def inner(self) -> Any:
        """The wrapped engine, exposed for advanced inspection / lifecycle hooks."""
        return self._inner

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> tuple[str, float]:
        self.transcribe_count += 1
        result = self._inner.transcribe(audio, sample_rate)
        # Real engines historically return (text, confidence); guard for
        # anything else so an unexpected shape surfaces in tests rather than
        # corrupting last_texts.
        if isinstance(result, tuple) and result and isinstance(result[0], str):
            text = result[0]
        elif isinstance(result, str):
            text = result
        else:
            text = ""
        self.last_texts.append(text)
        return result if isinstance(result, tuple) else (text, 1.0)

    # ---- Plain delegation -------------------------------------------------

    def get_required_sample_rate(self) -> int:
        return self._inner.get_required_sample_rate()

    def get_engine_name(self) -> str:
        return self._inner.get_engine_name()

    def get_supported_languages(self) -> list[str]:
        return self._inner.get_supported_languages()

    def cleanup(self) -> None:
        return self._inner.cleanup()
