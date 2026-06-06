"""MockEngine used by both the pytest baseline tests and the benchmark runner.

A single canonical implementation so the engine surface stays in sync as the
``TranscriptionEngine`` protocol evolves.
"""

from __future__ import annotations

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
