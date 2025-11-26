"""Benchmark metrics calculation.

Provides:
- WER (Word Error Rate) calculation using jiwer
- CER (Character Error Rate) calculation using jiwer
- RTF (Real-Time Factor) calculation
- Memory measurement (RAM via tracemalloc, GPU via torch.cuda)
"""

from __future__ import annotations

import tracemalloc
from dataclasses import dataclass, field
from typing import Callable, TypeVar, Any

from .text_normalization import normalize_text

# Optional imports
try:
    from jiwer import wer as jiwer_wer, cer as jiwer_cer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


__all__ = [
    "BenchmarkMetrics",
    "calculate_wer",
    "calculate_cer",
    "calculate_rtf",
    "measure_ram",
    "GPUMemoryTracker",
]


@dataclass
class BenchmarkMetrics:
    """Container for benchmark evaluation metrics."""

    # Accuracy metrics
    wer: float | None = None  # Word Error Rate (0.0 - 1.0+)
    cer: float | None = None  # Character Error Rate (0.0 - 1.0+)

    # Performance metrics
    rtf: float | None = None  # Real-Time Factor (lower is faster)
    latency_ms: float | None = None  # Processing latency in milliseconds
    audio_duration_s: float | None = None  # Audio duration in seconds
    processing_time_s: float | None = None  # Processing time in seconds

    # Memory metrics
    memory_peak_mb: float | None = None  # Peak RAM usage in MB
    gpu_memory_model_mb: float | None = None  # GPU memory after model load
    gpu_memory_peak_mb: float | None = None  # Peak GPU memory during inference

    # Metadata
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "wer": self.wer,
            "cer": self.cer,
            "rtf": self.rtf,
            "latency_ms": self.latency_ms,
            "audio_duration_s": self.audio_duration_s,
            "processing_time_s": self.processing_time_s,
            "memory_peak_mb": self.memory_peak_mb,
            "gpu_memory_model_mb": self.gpu_memory_model_mb,
            "gpu_memory_peak_mb": self.gpu_memory_peak_mb,
        }
        if self.extra:
            result["extra"] = self.extra
        return result


def calculate_wer(
    reference: str,
    hypothesis: str,
    *,
    lang: str | None = None,
    normalize: bool = True,
) -> float:
    """Calculate Word Error Rate (WER).

    WER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=reference words

    Args:
        reference: Ground truth transcript
        hypothesis: ASR output transcript
        lang: Language code for normalization (if normalize=True)
        normalize: If True, apply language-specific normalization

    Returns:
        WER as a float (0.0 = perfect, 1.0 = 100% error)

    Raises:
        ImportError: If jiwer is not installed
    """
    if not JIWER_AVAILABLE:
        raise ImportError("jiwer is required for WER calculation. Install with: pip install jiwer")

    if normalize and lang:
        reference = normalize_text(reference, lang=lang)
        hypothesis = normalize_text(hypothesis, lang=lang)

    # Handle empty reference
    if not reference.strip():
        return 0.0 if not hypothesis.strip() else 1.0

    return jiwer_wer(reference, hypothesis)


def calculate_cer(
    reference: str,
    hypothesis: str,
    *,
    lang: str | None = None,
    normalize: bool = True,
) -> float:
    """Calculate Character Error Rate (CER).

    CER is similar to WER but operates on characters instead of words.
    Particularly useful for languages without clear word boundaries (e.g., Japanese).

    Args:
        reference: Ground truth transcript
        hypothesis: ASR output transcript
        lang: Language code for normalization (if normalize=True)
        normalize: If True, apply language-specific normalization

    Returns:
        CER as a float (0.0 = perfect, 1.0 = 100% error)

    Raises:
        ImportError: If jiwer is not installed
    """
    if not JIWER_AVAILABLE:
        raise ImportError("jiwer is required for CER calculation. Install with: pip install jiwer")

    if normalize and lang:
        reference = normalize_text(reference, lang=lang)
        hypothesis = normalize_text(hypothesis, lang=lang)

    # Handle empty reference
    if not reference:
        return 0.0 if not hypothesis else 1.0

    return jiwer_cer(reference, hypothesis)


def calculate_rtf(audio_duration: float, processing_time: float) -> float:
    """Calculate Real-Time Factor (RTF).

    RTF = processing_time / audio_duration
    - RTF < 1.0: Faster than real-time
    - RTF = 1.0: Real-time
    - RTF > 1.0: Slower than real-time

    Args:
        audio_duration: Audio duration in seconds
        processing_time: Processing time in seconds

    Returns:
        RTF as a float (lower is better)
    """
    if audio_duration <= 0:
        return 0.0
    return processing_time / audio_duration


T = TypeVar("T")


def measure_ram(func: Callable[[], T]) -> tuple[T, float]:
    """Measure peak RAM usage during function execution.

    Uses Python's tracemalloc to measure memory allocation.
    Note: This only measures Python heap memory, not native memory.

    Args:
        func: Function to execute and measure

    Returns:
        Tuple of (function result, peak memory in MB)
    """
    tracemalloc.start()
    try:
        result = func()
        _, peak = tracemalloc.get_traced_memory()
        peak_mb = peak / (1024 * 1024)
        return result, peak_mb
    finally:
        tracemalloc.stop()


class GPUMemoryTracker:
    """Track GPU memory usage during benchmark execution.

    Usage:
        tracker = GPUMemoryTracker()

        # After model load
        model_memory = tracker.get_allocated()

        # Before inference
        tracker.reset_peak()

        # Run inference
        result = engine.transcribe(audio)

        # Get peak memory
        peak_memory = tracker.get_peak()
    """

    def __init__(self) -> None:
        """Initialize GPU memory tracker."""
        self._available = TORCH_AVAILABLE and torch.cuda.is_available()

    @property
    def available(self) -> bool:
        """Check if GPU memory tracking is available."""
        return self._available

    def reset_peak(self) -> None:
        """Reset peak memory statistics."""
        if self._available:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    def synchronize(self) -> None:
        """Synchronize CUDA operations."""
        if self._available:
            torch.cuda.synchronize()

    def get_allocated(self) -> float | None:
        """Get current allocated GPU memory in MB."""
        if not self._available:
            return None
        self.synchronize()
        return torch.cuda.memory_allocated() / (1024 * 1024)

    def get_peak(self) -> float | None:
        """Get peak allocated GPU memory in MB."""
        if not self._available:
            return None
        self.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 * 1024)

    def get_reserved(self) -> float | None:
        """Get current reserved GPU memory in MB."""
        if not self._available:
            return None
        self.synchronize()
        return torch.cuda.memory_reserved() / (1024 * 1024)
