"""Tests for ``benchmarks.confidence_calibration._core`` (Issue #338 PR-α)。

Pure logic test、実 model / I/O 不要。``LabeledSample`` を合成して
``sweep_threshold()`` を呼び、confusion matrix / recommended_threshold が
正しいかを pin する。
"""

from __future__ import annotations

import pytest

from benchmarks.confidence_calibration._core import (
    BreakdownReport,
    LabeledSample,
    SweepReport,
    ThresholdMetrics,
    _breakdown_key,
    compute_breakdowns,
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

    def test_empty_breakdown_serializes_to_empty_dict(self):
        """Phase 6a: ``breakdown_by=None`` 時、 JSON 上 ``"breakdown": {}``
        として現れる (Phase 1 report との backward compat 保証)。"""
        import json

        samples = [
            LabeledSample(signal_value=-0.10, label="speech"),
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
        d = report_to_dict(report)
        # additive で "breakdown" key は存在するが空 dict
        assert "breakdown" in d
        assert d["breakdown"] == {}
        # JSON round-trip verify
        loaded = json.loads(json.dumps(d, ensure_ascii=False))
        assert loaded["breakdown"] == {}


# ----------------- Phase 6a: _breakdown_key --------------------------------


class TestBreakdownKey:
    def test_none_returns_sentinel(self):
        assert _breakdown_key(None) == "__none__"

    def test_float_returns_str(self):
        assert _breakdown_key(10.0) == "10.0"
        assert _breakdown_key(-5.0) == "-5.0"
        assert _breakdown_key(0.0) == "0.0"

    def test_int_returns_str(self):
        assert _breakdown_key(10) == "10"
        assert _breakdown_key(0) == "0"

    def test_string_returns_as_is(self):
        assert _breakdown_key("clapping") == "clapping"
        assert _breakdown_key("") == ""

    def test_bool_returns_str(self):
        assert _breakdown_key(True) == "True"
        assert _breakdown_key(False) == "False"


# ----------------- Phase 6a: compute_breakdowns ----------------------------


class TestComputeBreakdowns:
    def test_single_key_single_value(self):
        samples = [
            LabeledSample(signal_value=-0.10, label="speech", metadata={"snr_db": 10.0}),
            LabeledSample(signal_value=-0.20, label="speech", metadata={"snr_db": 10.0}),
        ]
        br = compute_breakdowns(samples, "snr_db", [0.0], "reject_if_less")
        assert br.key == "snr_db"
        assert br.value_counts == {"10.0": 2}
        assert list(br.sweep_by_value.keys()) == ["10.0"]
        assert len(br.sweep_by_value["10.0"]) == 1  # 1 threshold

    def test_single_key_multiple_values(self):
        samples = [
            LabeledSample(signal_value=-0.10, label="speech", metadata={"snr_db": 10.0}),
            LabeledSample(signal_value=-0.20, label="speech", metadata={"snr_db": 0.0}),
            LabeledSample(signal_value=-0.30, label="speech", metadata={"snr_db": -5.0}),
        ]
        br = compute_breakdowns(samples, "snr_db", [-0.25], "reject_if_less")
        assert set(br.value_counts.keys()) == {"10.0", "0.0", "-5.0"}
        assert br.value_counts["10.0"] == 1
        assert br.value_counts["0.0"] == 1
        assert br.value_counts["-5.0"] == 1

    def test_none_value_bucketed_as_none_sentinel(self):
        """clean speech は snr_db field なし → ``__none__`` bucket に集約。"""
        samples = [
            LabeledSample(signal_value=-0.10, label="speech", metadata={}),  # no snr_db
            LabeledSample(signal_value=-0.20, label="speech", metadata={"snr_db": 10.0}),
        ]
        br = compute_breakdowns(samples, "snr_db", [0.0], "reject_if_less")
        assert br.value_counts == {"__none__": 1, "10.0": 1}

    def test_nonexistent_key_all_none_bucket(self):
        """存在しない key を指定 → 全 sample が ``__none__`` bucket。"""
        samples = [
            LabeledSample(signal_value=-0.10, label="speech", metadata={"snr_db": 10.0}),
            LabeledSample(signal_value=-0.20, label="speech", metadata={"snr_db": 0.0}),
        ]
        br = compute_breakdowns(samples, "typo_key", [0.0], "reject_if_less")
        assert br.value_counts == {"__none__": 2}

    def test_bucket_confusion_matrix_isolated_to_subset(self):
        """各 bucket の混同行列は該当 sample のみで計算されることを verify。"""
        samples = [
            # SNR 10 bucket: speech only (正しく分類されれば TN=1)
            LabeledSample(signal_value=-0.05, label="speech", metadata={"snr_db": 10.0}),
            # SNR 0 bucket: non_speech only (正しく分類されれば TP=1)
            LabeledSample(signal_value=-0.50, label="non_speech", metadata={"snr_db": 0.0}),
        ]
        br = compute_breakdowns(samples, "snr_db", [-0.2], "reject_if_less")
        # SNR 10 (speech, -0.05 >= -0.2) → not rejected → TN
        snr10 = br.sweep_by_value["10.0"][0]
        assert snr10.tn == 1 and snr10.tp == 0 and snr10.fp == 0 and snr10.fn == 0
        # SNR 0 (non_speech, -0.50 < -0.2) → rejected → TP
        snr0 = br.sweep_by_value["0.0"][0]
        assert snr0.tp == 1 and snr0.tn == 0 and snr0.fp == 0 and snr0.fn == 0

    def test_string_value_bucket(self):
        samples = [
            LabeledSample(signal_value=-0.10, label="non_speech", metadata={"subtype": "clapping"}),
            LabeledSample(signal_value=-0.20, label="non_speech", metadata={"subtype": "engine"}),
            LabeledSample(signal_value=-0.30, label="non_speech", metadata={"subtype": "clapping"}),
        ]
        br = compute_breakdowns(samples, "subtype", [0.0], "reject_if_less")
        assert br.value_counts == {"clapping": 2, "engine": 1}

    def test_all_thresholds_swept_per_bucket(self):
        samples = [
            LabeledSample(signal_value=-0.10, label="speech", metadata={"snr_db": 10.0}),
        ]
        br = compute_breakdowns(
            samples, "snr_db", [-0.5, -0.2, 0.0], "reject_if_less"
        )
        assert len(br.sweep_by_value["10.0"]) == 3
        # Threshold 値が list 順に対応
        assert br.sweep_by_value["10.0"][0].threshold == -0.5
        assert br.sweep_by_value["10.0"][2].threshold == 0.0

    def test_empty_samples_returns_empty_buckets(self):
        br = compute_breakdowns([], "snr_db", [0.0], "reject_if_less")
        assert br.value_counts == {}
        assert br.sweep_by_value == {}

    def test_float_precision_consistency(self):
        """同じ float 値は同一 bucket key に落ちる (repr でない、 str() の一貫性)。"""
        samples = [
            LabeledSample(signal_value=-0.10, label="speech", metadata={"snr_db": 10.0}),
            LabeledSample(signal_value=-0.20, label="speech", metadata={"snr_db": 10.0}),
        ]
        br = compute_breakdowns(samples, "snr_db", [0.0], "reject_if_less")
        # 2 sample が 1 bucket に集約 (別 key に分かれない)
        assert len(br.value_counts) == 1
        assert br.value_counts["10.0"] == 2

    def test_negative_float_key(self):
        samples = [
            LabeledSample(signal_value=-0.10, label="speech", metadata={"snr_db": -5.0}),
        ]
        br = compute_breakdowns(samples, "snr_db", [0.0], "reject_if_less")
        # 負の SNR も文字列 key で保持
        assert "-5.0" in br.value_counts

    def test_bool_value_bucket(self):
        samples = [
            LabeledSample(signal_value=-0.10, label="speech", metadata={"is_augmented": True}),
            LabeledSample(signal_value=-0.20, label="speech", metadata={"is_augmented": False}),
        ]
        br = compute_breakdowns(samples, "is_augmented", [0.0], "reject_if_less")
        assert br.value_counts == {"True": 1, "False": 1}


# ----------------- Phase 6a: sweep_threshold with breakdown_by -------------


class TestSweepThresholdWithBreakdown:
    def test_default_none_gives_empty_breakdown(self):
        """``breakdown_by=None`` (default) → ``SweepReport.breakdown == {}``。"""
        samples = [
            LabeledSample(signal_value=-0.10, label="speech"),
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
        assert report.breakdown == {}

    def test_single_key_populates_breakdown(self):
        samples = [
            LabeledSample(signal_value=-0.10, label="speech", metadata={"snr_db": 10.0}),
            LabeledSample(signal_value=-0.20, label="speech", metadata={"snr_db": 0.0}),
            LabeledSample(signal_value=-0.50, label="non_speech", metadata={"snr_db": None}),
        ]
        report = sweep_threshold(
            samples,
            engine="test",
            signal_field="avg_logprob",
            direction="reject_if_less",
            threshold_min=-0.6,
            threshold_max=0.0,
            step=0.2,
            breakdown_by=["snr_db"],
        )
        assert "snr_db" in report.breakdown
        br = report.breakdown["snr_db"]
        assert br.key == "snr_db"
        assert set(br.value_counts.keys()) == {"10.0", "0.0", "__none__"}

    def test_multiple_keys_independent(self):
        samples = [
            LabeledSample(
                signal_value=-0.10,
                label="non_speech",
                metadata={"snr_db": 10.0, "subtype": "clapping"},
            ),
            LabeledSample(
                signal_value=-0.20,
                label="non_speech",
                metadata={"snr_db": 0.0, "subtype": "engine"},
            ),
        ]
        report = sweep_threshold(
            samples,
            engine="test",
            signal_field="avg_logprob",
            direction="reject_if_less",
            threshold_min=-0.6,
            threshold_max=0.0,
            step=0.2,
            breakdown_by=["snr_db", "subtype"],
        )
        assert set(report.breakdown.keys()) == {"snr_db", "subtype"}
        assert report.breakdown["snr_db"].value_counts == {"10.0": 1, "0.0": 1}
        assert report.breakdown["subtype"].value_counts == {"clapping": 1, "engine": 1}

    def test_breakdown_serializes_to_json_dict(self):
        """``sweep_threshold`` の breakdown → ``report_to_dict`` → ``json.dumps`` round-trip。"""
        import json

        samples = [
            LabeledSample(
                signal_value=-0.10,
                label="speech",
                metadata={"snr_db": 10.0},
            ),
            LabeledSample(
                signal_value=-0.50,
                label="non_speech",
                metadata={"snr_db": 10.0},
            ),
        ]
        report = sweep_threshold(
            samples,
            engine="test",
            signal_field="avg_logprob",
            direction="reject_if_less",
            threshold_min=-0.6,
            threshold_max=0.0,
            step=0.2,
            breakdown_by=["snr_db"],
        )
        d = report_to_dict(report)
        loaded = json.loads(json.dumps(d, ensure_ascii=False))
        assert "snr_db" in loaded["breakdown"]
        assert loaded["breakdown"]["snr_db"]["key"] == "snr_db"
        assert loaded["breakdown"]["snr_db"]["value_counts"] == {"10.0": 2}
        assert "sweep_by_value" in loaded["breakdown"]["snr_db"]
        # sweep_by_value["10.0"] は ThresholdMetrics dict の list
        first_metric = loaded["breakdown"]["snr_db"]["sweep_by_value"]["10.0"][0]
        assert "threshold" in first_metric
        assert "false_reject_rate" in first_metric
