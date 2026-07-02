"""Tests for ``benchmarks.confidence_calibration._mix_snr`` (Issue #338 Layer 3).

Verifies:
  * SNR accuracy within ±0.5 dB (Plan D8) — the main promise of this helper.
  * Length matching (tile / truncate).
  * Clip detection + renormalization preserves SNR ratio.
  * Determinism (same input → same output).
  * Edge cases (empty / zero-power / single sample).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from benchmarks.confidence_calibration._mix_snr import (
    check_and_renorm,
    compute_snr_db,
    match_length,
    mix_at_snr,
)


def _sine(freq: float, duration_sec: float, sr: int = 16000, amplitude: float = 0.5) -> np.ndarray:
    """Deterministic sine wave for SNR testing (known RMS = amplitude / sqrt(2))."""
    t = np.arange(int(duration_sec * sr)) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _white_noise(duration_sec: float, seed: int = 0, sr: int = 16000, amplitude: float = 0.1) -> np.ndarray:
    """Deterministic (seeded) white noise."""
    rng = np.random.RandomState(seed)
    return (amplitude * rng.randn(int(duration_sec * sr))).astype(np.float32)


# --------------------- match_length ---------------------------------------


class TestMatchLength:
    def test_source_longer_truncates(self):
        ref = np.ones(1000, dtype=np.float32)
        src = np.arange(1500, dtype=np.float32)
        out = match_length(ref, src)
        assert len(out) == 1000
        assert out[0] == 0.0
        assert out[-1] == 999.0

    def test_source_shorter_tiles(self):
        ref = np.ones(1000, dtype=np.float32)
        src = np.array([1, 2, 3], dtype=np.float32)
        out = match_length(ref, src)
        assert len(out) == 1000
        # tile 後 truncate: 1 2 3 1 2 3 1 ...
        assert out[0] == 1.0
        assert out[1] == 2.0
        assert out[2] == 3.0
        assert out[3] == 1.0

    def test_source_equal_length_copies(self):
        ref = np.zeros(500, dtype=np.float32)
        src = np.linspace(-1, 1, 500, dtype=np.float32)
        out = match_length(ref, src)
        assert len(out) == 500
        assert np.array_equal(out, src)
        # Verify it's a copy, not a view (mutating out shouldn't touch src)
        out[0] = 999.0
        assert src[0] != 999.0

    def test_source_empty_returns_zeros(self):
        ref = np.ones(100, dtype=np.float32)
        out = match_length(ref, np.array([], dtype=np.float32))
        assert len(out) == 100
        assert (out == 0.0).all()

    def test_returns_float32(self):
        ref = np.zeros(50, dtype=np.float32)
        src = np.arange(75, dtype=np.float64)
        out = match_length(ref, src)
        assert out.dtype == np.float32


# --------------------- mix_at_snr — SNR accuracy --------------------------


class TestMixAtSnrAccuracy:
    @pytest.mark.parametrize("target_snr_db", [-5.0, 0.0, 5.0, 10.0, 20.0])
    def test_snr_matches_target_within_half_db(self, target_snr_db):
        # Deterministic speech (sine) and noise (seeded white)
        speech = _sine(freq=440.0, duration_sec=1.0, amplitude=0.5)
        noise = _white_noise(duration_sec=1.0, seed=42, amplitude=0.3)
        mixed = mix_at_snr(speech, noise, target_snr_db)
        noise_component = mixed - speech
        actual_snr = compute_snr_db(speech, noise_component)
        assert actual_snr is not None
        assert abs(actual_snr - target_snr_db) < 0.5, (
            f"SNR mismatch: target {target_snr_db} vs actual {actual_snr}"
        )

    @pytest.mark.parametrize("target_snr_db", [-5.0, 0.0, 5.0, 10.0, 20.0])
    def test_snr_accuracy_with_different_noise(self, target_snr_db):
        # Different noise seed but same SNR should still be within tolerance
        speech = _sine(freq=880.0, duration_sec=0.5, amplitude=0.4)
        noise = _white_noise(duration_sec=0.5, seed=7, amplitude=0.2)
        mixed = mix_at_snr(speech, noise, target_snr_db)
        noise_component = mixed - speech
        actual = compute_snr_db(speech, noise_component)
        assert actual is not None
        assert abs(actual - target_snr_db) < 0.5

    @pytest.mark.parametrize("target_snr_db", [-5.0, 0.0, 5.0, 10.0, 20.0])
    def test_snr_accuracy_with_low_freq_speech(self, target_snr_db):
        # Low-frequency speech + noise: different RMS profile
        speech = _sine(freq=110.0, duration_sec=0.8, amplitude=0.6)
        noise = _white_noise(duration_sec=0.8, seed=123, amplitude=0.15)
        mixed = mix_at_snr(speech, noise, target_snr_db)
        noise_component = mixed - speech
        actual = compute_snr_db(speech, noise_component)
        assert actual is not None
        assert abs(actual - target_snr_db) < 0.5


# --------------------- mix_at_snr — length handling -----------------------


class TestMixAtSnrLength:
    def test_output_length_equals_speech(self):
        speech = _sine(440.0, 1.0)
        noise_short = _white_noise(0.3, seed=0)  # shorter
        noise_long = _white_noise(2.0, seed=1)   # longer

        mixed_short = mix_at_snr(speech, noise_short, 10.0)
        mixed_long = mix_at_snr(speech, noise_long, 10.0)

        assert len(mixed_short) == len(speech)
        assert len(mixed_long) == len(speech)

    def test_returns_float32(self):
        speech = _sine(440.0, 0.5).astype(np.float64)
        noise = _white_noise(0.5, seed=0).astype(np.float64)
        mixed = mix_at_snr(speech, noise, 5.0)
        assert mixed.dtype == np.float32


# --------------------- mix_at_snr — edge cases ----------------------------


class TestMixAtSnrEdgeCases:
    def test_empty_speech_returns_empty(self):
        speech = np.array([], dtype=np.float32)
        noise = _white_noise(1.0, seed=0)
        out = mix_at_snr(speech, noise, 10.0)
        assert len(out) == 0

    def test_zero_power_noise_returns_speech(self):
        speech = _sine(440.0, 0.5)
        silent_noise = np.zeros(8000, dtype=np.float32)
        out = mix_at_snr(speech, silent_noise, 10.0)
        # noise power == 0 → speech unchanged
        assert np.array_equal(out, speech.astype(np.float32))

    def test_zero_power_speech_returns_speech_unchanged(self):
        silent_speech = np.zeros(8000, dtype=np.float32)
        noise = _white_noise(0.5, seed=0)
        out = mix_at_snr(silent_speech, noise, 10.0)
        # speech power == 0 → still zero (SNR undefined)
        assert (out == 0.0).all()
        assert len(out) == len(silent_speech)

    def test_single_sample_speech(self):
        speech = np.array([0.5], dtype=np.float32)
        noise = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        out = mix_at_snr(speech, noise, 0.0)
        assert len(out) == 1

    def test_extreme_positive_snr_noise_negligible(self):
        # SNR = 60 dB → noise scale ≈ 0.001, mixed ≈ speech
        speech = _sine(440.0, 0.5, amplitude=0.5)
        noise = _white_noise(0.5, seed=0, amplitude=0.5)
        mixed = mix_at_snr(speech, noise, 60.0)
        # 差分は極小
        diff = np.abs(mixed - speech).max()
        assert diff < 0.05

    def test_extreme_negative_snr_noise_dominant(self):
        # SNR = -30 dB → noise dominates
        speech = _sine(440.0, 0.5, amplitude=0.1)
        noise = _white_noise(0.5, seed=0, amplitude=0.05)
        mixed = mix_at_snr(speech, noise, -30.0)
        # Should still return same length
        assert len(mixed) == len(speech)
        # Absolute peak likely > 1.0 (clipped) — that's caller's job to renorm
        assert mixed.dtype == np.float32


# --------------------- mix_at_snr — determinism ---------------------------


class TestMixAtSnrDeterminism:
    def test_same_input_same_output(self):
        speech = _sine(440.0, 0.5)
        noise = _white_noise(0.5, seed=42)
        m1 = mix_at_snr(speech, noise, 10.0)
        m2 = mix_at_snr(speech, noise, 10.0)
        assert np.array_equal(m1, m2)

    def test_different_snr_different_output(self):
        speech = _sine(440.0, 0.5)
        noise = _white_noise(0.5, seed=0)
        m0 = mix_at_snr(speech, noise, 0.0)
        m10 = mix_at_snr(speech, noise, 10.0)
        assert not np.array_equal(m0, m10)


# --------------------- check_and_renorm -----------------------------------


class TestCheckAndRenorm:
    def test_no_clip_returns_unchanged(self):
        audio = np.array([0.3, -0.5, 0.7, -0.2], dtype=np.float32)
        out, clipped = check_and_renorm(audio)
        assert not clipped
        assert np.array_equal(out, audio)

    def test_clip_detected_and_renormalized_to_0_95(self):
        audio = np.array([0.5, -1.5, 2.0, -0.3], dtype=np.float32)
        out, clipped = check_and_renorm(audio)
        assert clipped
        assert np.abs(out).max() == pytest.approx(0.95, abs=1e-6)

    def test_snr_ratio_preserved_after_renorm(self):
        # Manufacture a mix at target 10 dB, ensure it clips, verify SNR is preserved
        speech = _sine(440.0, 0.5, amplitude=0.9)
        noise = _white_noise(0.5, seed=0, amplitude=0.6)
        mixed = mix_at_snr(speech, noise, -3.0)  # low SNR → likely clips
        renormed, clipped = check_and_renorm(mixed)
        if clipped:
            # After renorm, both speech and noise are scaled by same factor
            # so SNR should be preserved.
            # We recompute: renormed = mixed * s → renormed - speech*s == (mixed - speech)*s
            scale = np.abs(renormed).max() / np.abs(mixed).max()
            noise_component_before = mixed - speech
            noise_component_after = renormed - speech * scale
            snr_before = compute_snr_db(speech, noise_component_before)
            snr_after = compute_snr_db(speech * scale, noise_component_after)
            assert snr_before is not None and snr_after is not None
            assert abs(snr_after - snr_before) < 0.5

    def test_empty_returns_unchanged(self):
        audio = np.array([], dtype=np.float32)
        out, clipped = check_and_renorm(audio)
        assert not clipped
        assert len(out) == 0

    def test_exactly_1_0_not_renormed(self):
        # Peak exactly at 1.0 shouldn't trigger renorm
        audio = np.array([-1.0, 0.5, -0.3, 0.9], dtype=np.float32)
        out, clipped = check_and_renorm(audio)
        assert not clipped
        assert np.array_equal(out, audio)


# --------------------- compute_snr_db (helper for tests) ------------------


class TestComputeSnrDb:
    def test_known_snr(self):
        # speech RMS = 0.5 / sqrt(2) ≈ 0.354, power = 0.125
        # noise RMS = 0.0354, power = 0.00125 → SNR = 10*log10(0.125/0.00125) = 20 dB
        speech = _sine(440.0, 1.0, amplitude=0.5)
        noise = _sine(880.0, 1.0, amplitude=0.05)
        snr = compute_snr_db(speech, noise)
        assert snr is not None
        assert abs(snr - 20.0) < 0.1

    def test_zero_noise_returns_none(self):
        speech = _sine(440.0, 1.0)
        noise = np.zeros(len(speech), dtype=np.float32)
        assert compute_snr_db(speech, noise) is None
