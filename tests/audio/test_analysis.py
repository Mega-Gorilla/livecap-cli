"""analyze_noise_samples() / NoiseAnalysis のユニットテスト。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from livecap_cli.audio import NoiseAnalysis, analyze_noise_samples


class TestAnalyzeNoiseSamples:
    """analyze_noise_samples() のユニットテスト。"""

    def test_basic_analysis(self):
        # 100 samples: 75 @ -70 dB (下位25%以下), 25 @ -50 dB (上位25%)
        samples = [-70.0] * 75 + [-50.0] * 25
        result = analyze_noise_samples(samples, sample_rate_hz=10.0)

        assert isinstance(result, NoiseAnalysis)
        # 25%ile は下位25%の境界 → -70 dB
        assert result.noise_floor_db == pytest.approx(-70.0, abs=1.0)
        # 95%ile は上位5%の境界 → -50 dB
        assert result.noise_peak_db == pytest.approx(-50.0, abs=1.0)
        # suggested = peak + 10
        assert result.suggested_threshold_db == pytest.approx(
            result.noise_peak_db + 10.0
        )
        # danger_zone = (floor - 5, floor + 5)
        assert result.danger_zone[0] == pytest.approx(result.noise_floor_db - 5.0)
        assert result.danger_zone[1] == pytest.approx(result.noise_floor_db + 5.0)
        # safe_zone = peak + 5
        assert result.safe_zone_min_db == pytest.approx(result.noise_peak_db + 5.0)
        # metadata
        assert result.sample_count == 100
        assert result.duration_s == pytest.approx(10.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            analyze_noise_samples([])

    def test_numpy_array_input(self):
        samples = np.array([-60.0, -55.0, -50.0, -45.0])
        result = analyze_noise_samples(samples, sample_rate_hz=2.0)
        assert result.sample_count == 4
        assert result.duration_s == pytest.approx(2.0)

    def test_invalid_sample_rate(self):
        with pytest.raises(ValueError, match="positive"):
            analyze_noise_samples([-50.0], sample_rate_hz=0)
        with pytest.raises(ValueError, match="positive"):
            analyze_noise_samples([-50.0], sample_rate_hz=-1.0)

    def test_dataclass_is_frozen(self):
        """NoiseAnalysis は frozen dataclass。"""
        result = analyze_noise_samples([-60.0, -50.0])
        with pytest.raises(FrozenInstanceError):
            result.noise_floor_db = 0.0  # type: ignore[misc]
