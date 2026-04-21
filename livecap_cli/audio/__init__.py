"""Audio processing utilities."""

from .noise_gate import NoiseAnalysis, NoiseGate, analyze_noise_samples

__all__ = ["NoiseAnalysis", "NoiseGate", "analyze_noise_samples"]
