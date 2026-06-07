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
from .transient_detector import (
    VALID_MODES as TRANSIENT_DETECTOR_MODES,
    TransientDetector,
    TransientDetectorConfig,
    TransientDetectorTelemetry,
    TransientFeatures,
)

__all__ = [
    "ENERGY_METRICS",
    "ENGINE_MIN_RMS_SAFETY_MARGIN_DB",
    "NoiseAnalysis",
    "NoiseGate",
    "PEAK_SAFETY_MARGIN_DB",
    "TRANSIENT_DETECTOR_MODES",
    "TransientDetector",
    "TransientDetectorConfig",
    "TransientDetectorTelemetry",
    "TransientFeatures",
    "_segment_energy_dbfs",
    "analyze_noise_samples",
]
