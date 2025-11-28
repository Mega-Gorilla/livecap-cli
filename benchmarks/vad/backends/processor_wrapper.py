"""VADProcessorWrapper for Protocol-compliant VAD backends.

Wraps VADProcessor for batch processing in benchmarks.
"""

from __future__ import annotations

import logging
from math import gcd

import numpy as np

from livecap_core.vad.backends import VADBackend
from livecap_core.vad.config import VADConfig
from livecap_core.vad.processor import VADProcessor

logger = logging.getLogger(__name__)


class VADProcessorWrapper:
    """Wrap Protocol-compliant VAD backends for benchmark interface.

    This wrapper enables Silero, WebRTC, and TenVAD backends to be used
    with the VADBenchmarkBackend interface for batch processing.

    The wrapper:
    1. Resamples entire audio to 16kHz first (avoids timing drift)
    2. Processes in frame_size multiples
    3. Collects all segments as (start_time, end_time) tuples

    Args:
        backend: A Protocol-compliant VADBackend instance
        config: Optional VADConfig for segment detection parameters

    Example:
        from livecap_core.vad.backends import SileroVAD
        wrapper = VADProcessorWrapper(SileroVAD(threshold=0.5))
        segments = wrapper.process_audio(audio, sample_rate=16000)
    """

    SAMPLE_RATE = 16000

    def __init__(
        self,
        backend: VADBackend,
        config: VADConfig | None = None,
    ):
        self._backend = backend
        self._config = config or VADConfig()
        self._processor = VADProcessor(config=self._config, backend=backend)

    def process_audio(
        self, audio: np.ndarray, sample_rate: int
    ) -> list[tuple[float, float]]:
        """Process entire audio and return detected speech segments.

        Resamples to 16kHz first to ensure timing accuracy,
        then processes in frame_size multiples.

        Args:
            audio: Audio data in float32 format [-1.0, 1.0]
            sample_rate: Sample rate in Hz

        Returns:
            List of segments as (start_time, end_time) tuples in seconds
        """
        # Resample to 16kHz first (ensures timing accuracy)
        if sample_rate != self.SAMPLE_RATE:
            audio_16k = self._resample(audio, sample_rate, self.SAMPLE_RATE)
        else:
            audio_16k = audio

        # Reset processor state
        self._processor.reset()

        # Process in chunks (frame_size multiples for alignment)
        frame_size = self._backend.frame_size
        chunk_size = frame_size * 100  # ~3.2s for 512 frame_size

        segments: list[tuple[float, float]] = []

        for i in range(0, len(audio_16k), chunk_size):
            chunk = audio_16k[i : i + chunk_size]
            vad_segments = self._processor.process_chunk(chunk, self.SAMPLE_RATE)

            for seg in vad_segments:
                if seg.is_final:
                    segments.append((seg.start_time, seg.end_time))

        # Get any remaining segment
        final = self._processor.finalize()
        if final is not None:
            segments.append((final.start_time, final.end_time))

        return segments

    def _resample(
        self, audio: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """Resample audio using scipy's polyphase filter."""
        from scipy import signal

        g = gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g

        resampled = signal.resample_poly(audio, up, down)
        return resampled.astype(np.float32)

    @property
    def name(self) -> str:
        """Backend identifier for reporting."""
        return self._backend.name

    @property
    def config(self) -> dict:
        """Configuration parameters for reproducibility."""
        # Get backend-specific config if available
        if hasattr(self._backend, "config"):
            return self._backend.config

        # Fallback: return basic info
        return {
            "backend": self._backend.name,
            "frame_size": self._backend.frame_size,
        }


__all__ = ["VADProcessorWrapper"]
