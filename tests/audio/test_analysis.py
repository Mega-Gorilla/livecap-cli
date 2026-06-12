"""analyze_noise_samples() / NoiseAnalysis のユニットテスト。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from livecap_cli.audio import (
    ENERGY_METRICS,
    ENGINE_MIN_RMS_SAFETY_MARGIN_DB,
    PEAK_SAFETY_MARGIN_DB,
    NoiseAnalysis,
    _segment_energy_dbfs,
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


class TestNoiseAnalysisEngineMinRms:
    """#292: NoiseAnalysis.suggested_engine_min_rms_dbfs と margin kwarg。"""

    def test_default_margin(self):
        """default margin で suggested_engine_min_rms_dbfs が
        noise_rms_p95_db + ENGINE_MIN_RMS_SAFETY_MARGIN_DB になる。"""
        rms = [-50.0] * 100
        peaks = [-39.0] * 100
        result = analyze_noise_samples(rms, peaks)

        assert result.suggested_engine_min_rms_dbfs == pytest.approx(
            result.noise_rms_p95_db + ENGINE_MIN_RMS_SAFETY_MARGIN_DB
        )

    def test_custom_margin_kwarg(self):
        """engine_min_rms_margin_db を user 任意に変更可能。"""
        rms = [-50.0] * 100
        peaks = [-39.0] * 100
        # margin=10 で suggested = noise_rms_p95 + 10
        result = analyze_noise_samples(
            rms, peaks, engine_min_rms_margin_db=10.0
        )
        assert result.suggested_engine_min_rms_dbfs == pytest.approx(
            result.noise_rms_p95_db + 10.0
        )

    def test_negative_margin(self):
        """負の margin (suggested を noise_rms_p95 より低くする) も許容。"""
        rms = [-50.0] * 100
        peaks = [-39.0] * 100
        result = analyze_noise_samples(
            rms, peaks, engine_min_rms_margin_db=-3.0
        )
        assert result.suggested_engine_min_rms_dbfs == pytest.approx(
            result.noise_rms_p95_db - 3.0
        )


class TestNoiseAnalysisPeakSafetyMargin:
    """Issue #327: NoiseAnalysis.suggested_threshold_db と peak_safety_margin_db kwarg。

    `analyze_noise_samples(peak_safety_margin_db=...)` で hardcoded `+6 dB`
    を user-tunable に。AT4040 等 self-noise <15 dBA の studio コンデンサー
    マイクで負値を渡せることが motivation (peak_p95 ≈ -60 dB は既に
    conservative、user は更に低い threshold を望む)。
    """

    def test_default_peak_margin_unchanged(self):
        """default 省略時は既存挙動 (peak_p95 + PEAK_SAFETY_MARGIN_DB = 6.0)
        と bit-identical (backward compat ではなく sensible default の確認)。"""
        rms = [-50.0] * 10
        peaks = [-40.0] * 10
        result = analyze_noise_samples(rms, peaks)
        assert result.suggested_threshold_db == pytest.approx(
            result.peak_p95_db + PEAK_SAFETY_MARGIN_DB
        )
        # default = 6.0
        assert result.suggested_threshold_db == pytest.approx(-40.0 + 6.0)

    def test_custom_peak_margin_kwarg(self):
        """peak_safety_margin_db を user 任意に変更可能
        (typical USB mic, margin=3 で suggested = peak_p95 + 3)。"""
        rms = [-50.0] * 10
        peaks = [-40.0] * 10
        result = analyze_noise_samples(
            rms, peaks, peak_safety_margin_db=3.0
        )
        assert result.suggested_threshold_db == pytest.approx(
            result.peak_p95_db + 3.0
        )
        assert result.suggested_threshold_db == pytest.approx(-40.0 + 3.0)

    def test_negative_peak_margin_at4040_case(self):
        """AT4040 case: 負 margin (peak_p95 - 5) で studio mic 対応。

        Issue #327 motivation: AT4040 (self-noise 12 dBA SPL) で peak_p95 が
        既に conservative (~-60 dB)、user は更に -65 dB (= peak_p95 - 5)
        の threshold を望む。
        """
        rms = [-70.0] * 10
        peaks = [-60.0] * 10  # AT4040 相当
        result = analyze_noise_samples(
            rms, peaks, peak_safety_margin_db=-5.0
        )
        assert result.suggested_threshold_db == pytest.approx(-60.0 - 5.0)
        # → -65 dB が得られる、user 希望と一致


class TestSegmentEnergyDbfs:
    """#292: _segment_energy_dbfs の 4 metric ごとの動作。"""

    def test_max_frame_rms_basic(self):
        # constant amplitude 0.1 → RMS = 0.1 → -20 dBFS for any frame
        sr = 16000
        audio = np.full(sr, 0.1, dtype=np.float32)
        result = _segment_energy_dbfs(audio, sr, metric="max_frame_rms")
        assert result == pytest.approx(-20.0, abs=0.1)

    def test_whole_rms_basic(self):
        sr = 16000
        audio = np.full(sr, 0.1, dtype=np.float32)
        result = _segment_energy_dbfs(audio, sr, metric="whole_rms")
        assert result == pytest.approx(-20.0, abs=0.1)

    def test_p95_frame_rms_basic(self):
        sr = 16000
        audio = np.full(sr, 0.1, dtype=np.float32)
        result = _segment_energy_dbfs(audio, sr, metric="p95_frame_rms")
        assert result == pytest.approx(-20.0, abs=0.1)

    def test_top3_frame_rms_basic(self):
        sr = 16000
        audio = np.full(sr, 0.1, dtype=np.float32)
        result = _segment_energy_dbfs(audio, sr, metric="top3_frame_rms")
        assert result == pytest.approx(-20.0, abs=0.1)

    def test_padding_dilution_max_frame_vs_whole(self):
        """50ms speech @ -20 dBFS + 950ms silence の segment で:
        - whole_rms は希釈されて非常に低い値になる
        - max_frame_rms は speech 部分の energy を捕捉する
        """
        sr = 16000
        speech_n = int(0.05 * sr)  # 50ms
        total_n = sr  # 1s
        audio = np.zeros(total_n, dtype=np.float32)
        # 50ms speech at amplitude 0.1 (-20 dBFS RMS)
        audio[:speech_n] = 0.1

        whole = _segment_energy_dbfs(audio, sr, metric="whole_rms")
        max_frame = _segment_energy_dbfs(
            audio, sr, metric="max_frame_rms", frame_ms=32.0
        )
        # whole RMS: sqrt(50/1000) * 0.1 = 0.0224 → -33 dBFS
        # max frame (32ms entirely within speech): -20 dBFS
        # → max_frame is significantly higher than whole_rms
        assert max_frame > whole + 8.0  # at least 8 dB difference
        # max_frame should be close to speech amplitude
        assert max_frame == pytest.approx(-20.0, abs=2.0)

    def test_short_audio_falls_back_to_whole_rms(self):
        """audio が 1 frame に満たない場合 whole_rms に fallback。"""
        sr = 16000
        # 10ms < 32ms frame
        audio = np.full(int(0.010 * sr), 0.1, dtype=np.float32)
        for metric in ("max_frame_rms", "p95_frame_rms", "top3_frame_rms"):
            result = _segment_energy_dbfs(audio, sr, metric=metric, frame_ms=32.0)
            assert result == pytest.approx(-20.0, abs=0.1), (
                f"metric {metric} should fallback to whole RMS"
            )

    def test_zero_frame_ms_falls_back_to_whole_rms(self):
        sr = 16000
        audio = np.full(sr, 0.1, dtype=np.float32)
        for metric in ("max_frame_rms", "p95_frame_rms", "top3_frame_rms"):
            result = _segment_energy_dbfs(
                audio, sr, metric=metric, frame_ms=0.0
            )
            assert result == pytest.approx(-20.0, abs=0.1)

    def test_unknown_metric_raises(self):
        sr = 16000
        audio = np.full(sr, 0.1, dtype=np.float32)
        with pytest.raises(ValueError, match="unknown metric"):
            _segment_energy_dbfs(audio, sr, metric="bogus_metric")

    def test_energy_metrics_constant(self):
        """ENERGY_METRICS が公開定数として参照可能。"""
        assert set(ENERGY_METRICS) == {
            "max_frame_rms",
            "whole_rms",
            "p95_frame_rms",
            "top3_frame_rms",
        }

    def test_max_frame_resists_single_transient_more_than_whole(self):
        """単発 transient: max_frame は transient で大きく上がるが
        whole_rms は希釈されて低いまま。"""
        sr = 16000
        audio = np.zeros(sr, dtype=np.float32)
        # 32ms transient at amplitude 0.3 (~-10 dBFS RMS for that frame)
        audio[:int(0.032 * sr)] = 0.3
        max_frame = _segment_energy_dbfs(audio, sr, metric="max_frame_rms")
        whole = _segment_energy_dbfs(audio, sr, metric="whole_rms")
        top3 = _segment_energy_dbfs(audio, sr, metric="top3_frame_rms")
        # max_frame should pick up the transient
        assert max_frame > -15.0  # close to -10 dBFS
        # whole_rms should be much lower (transient is only 3.2% of segment)
        assert whole < -25.0
        # top3 mean averages transient with quieter frames → between max and whole
        assert whole < top3 < max_frame
