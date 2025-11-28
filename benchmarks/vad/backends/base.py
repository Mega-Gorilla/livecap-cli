"""VAD Benchmark Backend Protocol.

Provides a unified interface for all VAD backends in benchmarking.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class VADBenchmarkBackend(Protocol):
    """Unified interface for VAD backends in benchmarking.

    This protocol allows both Protocol-compliant VADs (Silero, WebRTC, TenVAD)
    and batch-only VADs (JaVAD) to be used with the same interface.

    Example:
        vad = create_vad("silero")
        segments = vad.process_audio(audio, sample_rate)
        print(f"Detected {len(segments)} segments")
        print(f"Config: {vad.config}")
    """

    def process_audio(
        self, audio: np.ndarray, sample_rate: int
    ) -> list[tuple[float, float]]:
        """Process entire audio and return detected speech segments.

        Args:
            audio: Audio data in float32 format [-1.0, 1.0]
            sample_rate: Sample rate in Hz

        Returns:
            List of segments as (start_time, end_time) tuples in seconds
        """
        ...

    @property
    def name(self) -> str:
        """Backend identifier for reporting."""
        ...

    @property
    def config(self) -> dict:
        """Configuration parameters for reproducibility.

        Returns:
            Dictionary of VAD-specific parameters.
            Example: {"mode": 3, "frame_duration_ms": 20}
        """
        ...


__all__ = ["VADBenchmarkBackend"]
