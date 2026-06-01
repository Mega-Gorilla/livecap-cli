"""analyze_noise_samples() / NoiseAnalysis のユニットテスト。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from livecap_cli.audio import (
    PEAK_SAFETY_MARGIN_DB,
    NoiseAnalysis,
    analyze_noise_samples,
)


class TestAnalyzeNoiseSamples:
    """analyze_noise_samples() のユニットテスト。"""

    def test_basic_analysis(self):
        # 100 samples: 75 @ -70 dB (下位25%以下), 25 @ -50 dB (上位25%)
        rms = [-70.0] * 75 + [-50.0] * 25
        # peak は意図的に rms と同値 → suggested = peak_p95 + 6 を確認しやすく
        peaks = list(rms)
        result = analyze_noise_samples(rms, peaks, sample_rate_hz=10.0)

        assert isinstance(result, NoiseAnalysis)
        # 25%ile は下位25%の境界 → -70 dB
        assert result.noise_floor_db == pytest.approx(-70.0, abs=1.0)
        # 95%ile は上位5%の境界 → -50 dB
        assert result.noise_rms_p95_db == pytest.approx(-50.0, abs=1.0)
        assert result.peak_p95_db == pytest.approx(-50.0, abs=1.0)
        # suggested = peak_p95 + 6 (PEAK_SAFETY_MARGIN_DB)
        assert result.suggested_threshold_db == pytest.approx(
            result.peak_p95_db + PEAK_SAFETY_MARGIN_DB
        )
        # danger_zone = (floor - 5, floor + 5) (RMS-unit diagnostic)
        assert result.danger_zone[0] == pytest.approx(result.noise_floor_db - 5.0)
        assert result.danger_zone[1] == pytest.approx(result.noise_floor_db + 5.0)
        # metadata
        assert result.sample_count == 100
        assert result.duration_s == pytest.approx(10.0)

    def test_empty_samples_db_raises(self):
        with pytest.raises(ValueError, match="samples_db must not be empty"):
            analyze_noise_samples([], [-50.0])

    def test_empty_peak_samples_db_raises(self):
        with pytest.raises(
            ValueError, match="peak_samples_db must not be empty"
        ):
            analyze_noise_samples([-50.0], [])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            analyze_noise_samples(
                [-60.0, -55.0, -50.0], [-40.0, -35.0]
            )

    def test_numpy_array_input(self):
        rms = np.array([-60.0, -55.0, -50.0, -45.0])
        peaks = np.array([-50.0, -45.0, -40.0, -35.0])
        result = analyze_noise_samples(rms, peaks, sample_rate_hz=2.0)
        assert result.sample_count == 4
        assert result.duration_s == pytest.approx(2.0)
        # peak_p95 of [-50, -45, -40, -35] ≈ -35.x (95th percentile)
        assert result.peak_p95_db > -40.0
        assert result.suggested_threshold_db == pytest.approx(
            result.peak_p95_db + PEAK_SAFETY_MARGIN_DB
        )

    def test_invalid_sample_rate(self):
        with pytest.raises(ValueError, match="positive"):
            analyze_noise_samples([-50.0], [-40.0], sample_rate_hz=0)
        with pytest.raises(ValueError, match="positive"):
            analyze_noise_samples([-50.0], [-40.0], sample_rate_hz=-1.0)

    def test_dataclass_is_frozen(self):
        """NoiseAnalysis は frozen dataclass。"""
        result = analyze_noise_samples([-60.0, -50.0], [-50.0, -40.0])
        with pytest.raises(FrozenInstanceError):
            result.noise_floor_db = 0.0  # type: ignore[misc]


class TestUnitMatch:
    """新 API が NoiseGate envelope follower と単位を揃えていることの検証。"""

    def test_white_noise_crest_factor(self):
        """white noise: peak は RMS の ~11 dB 上 (crest factor)。
        suggested = peak_p95 + 6 が RMS p95 + 10 (旧バグ算法) より上に来る。
        """
        # 100 chunk 分の white noise を模擬: RMS -50 dB / peak -39 dB
        rms = [-50.0] * 100
        peaks = [-39.0] * 100  # crest factor 11 dB
        result = analyze_noise_samples(rms, peaks)

        assert result.peak_p95_db == pytest.approx(-39.0, abs=0.1)
        assert result.suggested_threshold_db == pytest.approx(-33.0, abs=0.1)
        # 新 suggested (-33) > 旧 suggested (RMS p95 + 10 = -40)
        old_suggested = result.noise_rms_p95_db + 10.0
        assert result.suggested_threshold_db > old_suggested + 5.0

    def test_impulsive_noise(self):
        """impulsive noise (低 RMS / 高 peak): 旧 path が under-margin だった
        ケースの回帰。"""
        # 低 RMS (-55 dB) で時々大きな peak (-31 dB) が混じる → crest 24 dB
        rms = [-55.0] * 80 + [-50.0] * 20
        peaks = [-45.0] * 80 + [-31.0] * 20  # 20% に大 peak
        result = analyze_noise_samples(rms, peaks)

        # peak p95 は上位の -31 を拾う
        assert result.peak_p95_db == pytest.approx(-31.0, abs=1.0)
        # 旧 suggested (RMS p95 + 10) = ~-40 dB → peak (-31) より大きく低い
        old_suggested = result.noise_rms_p95_db + 10.0
        # 新 suggested (= peak_p95 + 6) は旧より大幅に高い
        assert result.suggested_threshold_db > old_suggested + 10.0

    def test_sine_wave(self):
        """sine wave: peak = RMS + 3 dB → 新 path が over-margin でない。"""
        # sine wave の場合 peak = RMS * sqrt(2) → +3.01 dB
        rms = [-30.0] * 100
        peaks = [-26.99] * 100  # RMS + 3 dB
        result = analyze_noise_samples(rms, peaks)

        assert result.peak_p95_db == pytest.approx(-26.99, abs=0.1)
        # suggested = -26.99 + 6 = -20.99 dB
        assert result.suggested_threshold_db == pytest.approx(-20.99, abs=0.1)
        # sine wave への過剰 margin は無い (RMS との差は 9 dB のみ)
        assert (
            result.suggested_threshold_db - result.noise_rms_p95_db
            == pytest.approx(9.0, abs=0.5)
        )
