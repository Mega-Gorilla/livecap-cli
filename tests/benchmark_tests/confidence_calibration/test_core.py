"""Tests for ``benchmarks.confidence_calibration._core`` (Issue #338 PR-α)。

Pure logic test、実 model / I/O 不要。``LabeledSample`` を合成して
``sweep_threshold()`` を呼び、confusion matrix / recommended_threshold が
正しいかを pin する。
"""

from __future__ import annotations

import pytest

from benchmarks.confidence_calibration._core import (
    LabeledSample,
    SweepReport,
    ThresholdMetrics,
    report_to_dict,
    sweep_threshold,
)


# ----------------- Confusion matrix の基本 ---------------------------------


class TestConfusionMatrix:
    """``_confusion_matrix()`` の基本動作 (sweep_threshold 経由で test)。"""

    def test_perfect_separation_reject_if_less(self):
        """完全分離: speech は ``>= threshold``、non_speech は ``< threshold``。"""
        samples = [
            LabeledSample(signal_value=-0.05, label="speech"),
            LabeledSample(signal_value=-0.10, label="speech"),
            LabeledSample(signal_value=-0.45, label="non_speech"),
            LabeledSample(signal_value=-0.50, label="non_speech"),
        ]
        report = sweep_threshold(
            samples,
            engine="test",
            signal_field="avg_logprob",
            direction="reject_if_less",
            threshold_min=-0.2,
            threshold_max=-0.2,
            step=0.01,
        )
        m = report.recommended_metrics
        assert m.tp == 2 and m.fp == 0 and m.tn == 2 and m.fn == 0
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.false_reject_rate == 0.0

    def test_perfect_separation_reject_if_greater(self):
        """no_speech_prob 等 (高いほど reject すべき) のケース。"""
        samples = [
            LabeledSample(signal_value=0.05, label="speech"),
            LabeledSample(signal_value=0.10, label="speech"),
            LabeledSample(signal_value=0.80, label="non_speech"),
            LabeledSample(signal_value=0.90, label="non_speech"),
        ]
        report = sweep_threshold(
            samples,
            engine="test",
            signal_field="no_speech_prob",
            direction="reject_if_greater",
            threshold_min=0.5,
            threshold_max=0.5,
            step=0.01,
        )
        m = report.recommended_metrics
        assert m.tp == 2 and m.fp == 0 and m.tn == 2 and m.fn == 0
        assert m.f1 == 1.0

    def test_noisy_speech_treated_as_speech(self):
        """``noisy_speech`` label は speech 扱い (reject されたら false reject)。"""
        samples = [
            LabeledSample(signal_value=-0.05, label="speech"),
            LabeledSample(signal_value=-0.25, label="noisy_speech"),  # reject されたら FP
            LabeledSample(signal_value=-0.50, label="non_speech"),
        ]
        report = sweep_threshold(
            samples,
            engine="test",
            signal_field="avg_logprob",
            direction="reject_if_less",
            threshold_min=-0.20,
            threshold_max=-0.20,
            step=0.01,
        )
        m = report.recommended_metrics
        # threshold -0.20 で: -0.05 pass、-0.25 reject (FP)、-0.50 reject (TP)
        assert m.tp == 1
        assert m.fp == 1  # ← noisy_speech が reject されて FP
        assert m.tn == 1
        assert m.fn == 0

    def test_none_signal_excluded(self):
        """``signal_value=None`` は sweep から除外、``excluded_count`` に反映。"""
        samples = [
            LabeledSample(signal_value=-0.10, label="speech"),
            LabeledSample(signal_value=None, label="speech"),  # 除外
            LabeledSample(signal_value=-0.50, label="non_speech"),
        ]
        report = sweep_threshold(
            samples,
            engine="test",
            signal_field="avg_logprob",
            direction="reject_if_less",
            threshold_min=-0.30,
            threshold_max=-0.30,
            step=0.01,
        )
        assert report.excluded_count == 1
        assert sum(report.sample_count.values()) == 2


# ----------------- Sweep の monotonicity (sanity check) -------------------


class TestSweepMonotonicity:
    """direction = reject_if_less の場合、threshold ↑ で reject 数 ↓ (= FP+TP ↓)。"""

    def test_reject_count_decreases_with_loose_threshold(self):
        samples = [
            LabeledSample(signal_value=-0.05, label="speech"),
            LabeledSample(signal_value=-0.20, label="speech"),
            LabeledSample(signal_value=-0.30, label="non_speech"),
            LabeledSample(signal_value=-0.50, label="non_speech"),
        ]
        report = sweep_threshold(
            samples,
            engine="test",
            signal_field="avg_logprob",
            direction="reject_if_less",
            threshold_min=-0.6,
            threshold_max=0.0,
            step=0.1,
        )
        # threshold ↑ (緩い) → 全 sample が pass に近づく
        reject_counts = [m.tp + m.fp for m in report.sweep]
        # 単調非増加 (loose 方向で reject 減少 or 不変)
        # direction=reject_if_less で threshold ↑ ⇒ value < threshold の sample 増加
        # → 実際は逆: threshold が高い (例: 0.0) ほど多くの sample が value < 0.0 で reject
        # 逆: threshold ↑ (= -0.6 → 0.0) で reject 数 ↑
        # よってここでは reject_counts は monotonic non-decreasing
        for i in range(len(reject_counts) - 1):
            assert reject_counts[i] <= reject_counts[i + 1], (
                f"reject count should be non-decreasing as threshold increases "
                f"(reject_if_less): index {i}={reject_counts[i]}, "
                f"{i+1}={reject_counts[i+1]}, thresholds={[m.threshold for m in report.sweep]}"
            )


# ----------------- Optimal threshold selection ----------------------------


class TestRecommendedThreshold:
    def test_max_f1_default(self):
        samples = [
            LabeledSample(signal_value=-0.05, label="speech"),
            LabeledSample(signal_value=-0.08, label="speech"),
            LabeledSample(signal_value=-0.40, label="non_speech"),
            LabeledSample(signal_value=-0.50, label="non_speech"),
        ]
        report = sweep_threshold(
            samples,
            engine="test",
            signal_field="avg_logprob",
            direction="reject_if_less",
            threshold_min=-0.6,
            threshold_max=0.0,
            step=0.05,
        )
        # speech mean ~ -0.065、non-speech mean ~ -0.45、最適 threshold は ~-0.2 周辺
        # F1 = 1.0 になる範囲のうち threshold **小** (= conservative、tie-break)
        # を選ぶ (PR #339 codex-review fix、false reject 最小化のため)
        assert report.recommended_metrics.f1 == 1.0
        # -0.40 と -0.08 の間の任意 threshold で F1=1.0、tie-break で小さい方
        # = より conservative = false reject 抑制方向
        assert -0.40 < report.recommended_threshold <= -0.10
        # tie-break direction: F1 同点なら threshold 小を選ぶ
        # 最も conservative (threshold 最小) な F1=1.0 を期待
        # threshold step=0.05 で -0.35 が最小 F1=1.0 範囲端 (≤ -0.40 は -0.50 を取り逃す)
        # 期待値: -0.35 (F1=1.0 + 最小 threshold)
        assert report.recommended_threshold == pytest.approx(-0.35, abs=0.01)

    def test_criterion_youden_j(self):
        samples = [
            LabeledSample(signal_value=-0.05, label="speech"),
            LabeledSample(signal_value=-0.50, label="non_speech"),
        ]
        report = sweep_threshold(
            samples,
            engine="test",
            signal_field="avg_logprob",
            direction="reject_if_less",
            threshold_min=-0.6,
            threshold_max=0.0,
            step=0.05,
            criterion="youden_j",
        )
        assert report.criterion == "youden_j"
        assert report.recommended_metrics.youden_j == 1.0  # 完全分離

    def test_tie_break_reject_if_less_prefers_small_threshold(self):
        """``reject_if_less`` 同点時 threshold **小** = conservative (PR #339 fix)。"""
        # speech -0.05、non_speech -0.50 で完全分離、threshold -0.40 / -0.30 /
        # -0.20 / -0.10 すべて F1=1.0 → 同点
        samples = [
            LabeledSample(signal_value=-0.05, label="speech"),
            LabeledSample(signal_value=-0.50, label="non_speech"),
        ]
        report = sweep_threshold(
            samples,
            engine="test",
            signal_field="avg_logprob",
            direction="reject_if_less",
            threshold_min=-0.40,
            threshold_max=-0.10,
            step=0.10,
        )
        # F1=1.0 同点で direction=reject_if_less なら threshold **最小** (-0.40)
        # が選ばれる (より conservative = false reject 抑制方向)
        assert report.recommended_metrics.f1 == 1.0
        assert report.recommended_threshold == pytest.approx(-0.40, abs=0.001)

    def test_tie_break_reject_if_greater_prefers_large_threshold(self):
        """``reject_if_greater`` 同点時 threshold **大** = conservative (PR #339 fix)。"""
        # no_speech_prob: speech 0.05、non_speech 0.95 で完全分離
        samples = [
            LabeledSample(signal_value=0.05, label="speech"),
            LabeledSample(signal_value=0.95, label="non_speech"),
        ]
        report = sweep_threshold(
            samples,
            engine="test",
            signal_field="no_speech_prob",
            direction="reject_if_greater",
            threshold_min=0.30,
            threshold_max=0.80,
            step=0.10,
        )
        # F1=1.0 同点で direction=reject_if_greater なら threshold **最大** (0.80)
        # が選ばれる (= 0.80 より上の sample しか reject されない、conservative)
        assert report.recommended_metrics.f1 == 1.0
        assert report.recommended_threshold == pytest.approx(0.80, abs=0.001)


# ----------------- Edge cases ---------------------------------------------


class TestEdgeCases:
    def test_empty_samples_raises(self):
        with pytest.raises(ValueError, match="No samples"):
            sweep_threshold(
                [],
                engine="test",
                signal_field="avg_logprob",
                direction="reject_if_less",
                threshold_min=-0.5,
                threshold_max=-0.1,
                step=0.1,
            )

    def test_all_none_samples_raises(self):
        samples = [
            LabeledSample(signal_value=None, label="speech"),
            LabeledSample(signal_value=None, label="non_speech"),
        ]
        with pytest.raises(ValueError, match="No samples"):
            sweep_threshold(
                samples,
                engine="test",
                signal_field="avg_logprob",
                direction="reject_if_less",
                threshold_min=-0.5,
                threshold_max=-0.1,
                step=0.1,
            )

    def test_invalid_step_raises(self):
        samples = [LabeledSample(signal_value=-0.1, label="speech")]
        with pytest.raises(ValueError, match="step must be positive"):
            sweep_threshold(
                samples,
                engine="test",
                signal_field="avg_logprob",
                direction="reject_if_less",
                threshold_min=-0.5,
                threshold_max=-0.1,
                step=0.0,
            )

    def test_min_gt_max_raises(self):
        samples = [LabeledSample(signal_value=-0.1, label="speech")]
        with pytest.raises(ValueError, match="threshold_min .* > threshold_max"):
            sweep_threshold(
                samples,
                engine="test",
                signal_field="avg_logprob",
                direction="reject_if_less",
                threshold_min=0.0,
                threshold_max=-0.5,
                step=0.1,
            )

    def test_all_speech_no_non_speech(self):
        """全 speech、non_speech が無い corpus でも crash しない。"""
        samples = [
            LabeledSample(signal_value=-0.05, label="speech"),
            LabeledSample(signal_value=-0.10, label="speech"),
        ]
        report = sweep_threshold(
            samples,
            engine="test",
            signal_field="avg_logprob",
            direction="reject_if_less",
            threshold_min=-0.5,
            threshold_max=0.0,
            step=0.1,
        )
        # 全 speech なので tp = 0、recall は分母 0 で 0.0、precision も同様
        for m in report.sweep:
            assert m.tp == 0
            assert m.fn == 0


# ----------------- report_to_dict serialization ---------------------------


class TestReportSerialization:
    def test_round_trip_json_compatible(self):
        """``report_to_dict`` 結果が JSON serializable。"""
        import json

        samples = [
            LabeledSample(signal_value=-0.10, label="speech"),
            LabeledSample(signal_value=-0.50, label="non_speech"),
        ]
        report = sweep_threshold(
            samples,
            engine="reazonspeech",
            signal_field="avg_logprob",
            direction="reject_if_less",
            threshold_min=-0.6,
            threshold_max=0.0,
            step=0.1,
            metadata={"quantization": "float32", "language": "ja"},
        )
        d = report_to_dict(report)
        s = json.dumps(d, ensure_ascii=False)  # raises if not serializable
        loaded = json.loads(s)
        assert loaded["engine"] == "reazonspeech"
        assert loaded["signal_field"] == "avg_logprob"
        assert loaded["direction"] == "reject_if_less"
        assert loaded["metadata"]["quantization"] == "float32"
        assert "sweep" in loaded
        assert "recommended_threshold" in loaded
