"""Audio processing utilities."""

from .analysis import (
    ENERGY_METRICS,
    ENGINE_MIN_RMS_SAFETY_MARGIN_DB,
    PEAK_SAFETY_MARGIN_DB,
    NoiseAnalysis,
    _segment_energy_dbfs,
    analyze_noise_samples,
)
from .noise_gate import NoiseGate

__all__ = [
    "ENERGY_METRICS",
    "ENGINE_MIN_RMS_SAFETY_MARGIN_DB",
    "NoiseAnalysis",
    "NoiseGate",
    "PEAK_SAFETY_MARGIN_DB",
    "_segment_energy_dbfs",
    "analyze_noise_samples",
]
