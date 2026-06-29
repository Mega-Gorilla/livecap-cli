"""Signal-agnostic threshold sweep core (Issue #338 PR-α)。

Stage 1 (``parse_observe.py``) と Stage 2 (``sweep.py``、PR-β) の両方で
共有する pure logic。Engine.transcribe() を呼ばない、I/O 不要、test しやすい。

設計判断:
- Signal direction は engine の signal によって反転する:
  - ``avg_logprob`` / ``token_confidence_mean``: 値が **低い** ほど低 confidence
    → ``value < threshold`` で reject (``"reject_if_less"``)
  - ``no_speech_prob``: 値が **高い** ほど非音声確信度
    → ``value > threshold`` で reject (``"reject_if_greater"``)
- Confusion matrix の "positive class" = ``"non_speech"`` (= filter が reject すべき)
  - TP: non_speech を reject
  - FP: speech を reject (= false reject、user 痛い)
  - TN: speech を pass
  - FN: non_speech を pass (= false pass、user 軽微痛い)
- Recommended threshold: max F1 を default、``criterion`` arg で
  ``"f1"`` / ``"youden_j"`` / ``"min_fp_rate"`` 等を切替
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

Direction = Literal["reject_if_less", "reject_if_greater"]
Criterion = Literal["f1", "youden_j", "precision", "recall"]


@dataclass(frozen=True)
class LabeledSample:
    """1 audio sample の measurement + label。

    Attributes:
        signal_value: engine が出した signal (avg_logprob / no_speech_prob 等)。
            ``None`` の sample は sweep から除外 (fail-open path)。
        label: ground truth、``"speech"`` (filter は pass すべき) または
            ``"non_speech"`` (filter は reject すべき)。``"noisy_speech"`` は
            ``"speech"`` 扱い (= reject されたら false reject)。
        path: 元 audio file path (report の trace 用、optional)。
        metadata: engine/language/quantization 等の追加 info (optional)。
    """

    signal_value: Optional[float]
    label: Literal["speech", "non_speech", "noisy_speech"]
    path: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ThresholdMetrics:
    """1 threshold での confusion matrix + 評価 metrics。"""

    threshold: float
    tp: int  # non_speech を reject
    fp: int  # speech を reject (false reject、痛い)
    tn: int  # speech を pass
    fn: int  # non_speech を pass (false pass、軽微痛い)
    precision: float  # TP / (TP + FP)、reject の正確性
    recall: float  # TP / (TP + FN)、non_speech 検出率
    f1: float
    youden_j: float  # sensitivity + specificity - 1
    false_reject_rate: float  # FP / (FP + TN)、speech を reject する割合 (user 体感)


@dataclass(frozen=True)
class SweepReport:
    """Sweep の全結果 + recommendation。"""

    engine: str
    signal_field: str
    direction: Direction
    sample_count: dict[str, int]  # {"speech": N, "non_speech": M, ...}
    excluded_count: int  # signal_value=None で除外された数
    sweep: list[ThresholdMetrics]
    recommended_threshold: float
    recommended_metrics: ThresholdMetrics
    criterion: Criterion
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalize_label(label: str) -> Literal["speech", "non_speech"]:
    """noisy_speech → speech (= reject されたら false reject)。"""
    return "non_speech" if label == "non_speech" else "speech"


def _classify(value: float, threshold: float, direction: Direction) -> bool:
    """``True`` if rejected。"""
    if direction == "reject_if_less":
        return value < threshold
    return value > threshold


def _confusion_matrix(
    samples: list[LabeledSample],
    threshold: float,
    direction: Direction,
) -> ThresholdMetrics:
    tp = fp = tn = fn = 0
    for s in samples:
        if s.signal_value is None:
            continue  # 上位で除外済の想定だが defensive
        rejected = _classify(s.signal_value, threshold, direction)
        norm_label = _normalize_label(s.label)
        if norm_label == "non_speech":
            if rejected:
                tp += 1
            else:
                fn += 1
        else:  # speech
            if rejected:
                fp += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    youden_j = recall + specificity - 1.0  # sensitivity (= recall) + specificity - 1
    false_reject_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return ThresholdMetrics(
        threshold=threshold,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        youden_j=youden_j,
        false_reject_rate=false_reject_rate,
    )


def _select_recommended(
    sweep: list[ThresholdMetrics],
    criterion: Criterion,
) -> ThresholdMetrics:
    if not sweep:
        raise ValueError("Cannot recommend from empty sweep")
    if criterion == "f1":
        key = lambda m: m.f1
    elif criterion == "youden_j":
        key = lambda m: m.youden_j
    elif criterion == "precision":
        key = lambda m: m.precision
    elif criterion == "recall":
        key = lambda m: m.recall
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
    # 同点の場合は threshold が **大きい** ほうを採用 (より loose = false reject 少)
    # avg_logprob の場合 reject_if_less なので threshold 大 = 緩い
    # no_speech_prob の場合 reject_if_greater なので threshold 大 = 厳しい
    # 統一すべきだが、いずれ場合も criterion 値で tie-break → threshold は secondary
    return max(sweep, key=lambda m: (key(m), m.threshold))


def sweep_threshold(
    samples: list[LabeledSample],
    *,
    engine: str,
    signal_field: str,
    direction: Direction,
    threshold_min: float,
    threshold_max: float,
    step: float,
    criterion: Criterion = "f1",
    metadata: Optional[dict[str, Any]] = None,
) -> SweepReport:
    """Threshold sweep を実行し、推奨 threshold を返す。

    Args:
        samples: ``LabeledSample`` の list。``signal_value=None`` は除外される。
        engine: 報告用 engine ID (e.g. ``"reazonspeech"``)。
        signal_field: 報告用 signal name (e.g. ``"avg_logprob"``)。
        direction: ``"reject_if_less"`` (avg_logprob 等) / ``"reject_if_greater"``
            (no_speech_prob)。
        threshold_min/max/step: sweep 範囲、両端含む。``step > 0`` 必須。
        criterion: recommended_threshold を決める基準。
        metadata: report に embed する追加 info (e.g. quantization, language)。

    Returns:
        ``SweepReport`` (sweep 全結果 + recommendation)。

    Raises:
        ValueError: ``step <= 0`` / 全 sample が ``signal_value=None`` / sweep が
            empty (threshold_min > threshold_max + step) の場合。
    """
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    if threshold_min > threshold_max:
        raise ValueError(
            f"threshold_min ({threshold_min}) > threshold_max ({threshold_max})"
        )

    valid_samples = [s for s in samples if s.signal_value is not None]
    excluded_count = len(samples) - len(valid_samples)
    if not valid_samples:
        raise ValueError("No samples with valid signal_value (all excluded)")

    sample_count: dict[str, int] = {}
    for s in valid_samples:
        sample_count[s.label] = sample_count.get(s.label, 0) + 1

    # threshold grid (両端含む)
    thresholds: list[float] = []
    cur = threshold_min
    # numpy 不使用 (依存最小化)、step で float 累積誤差を避けるため整数 index 経由
    n_steps = int(round((threshold_max - threshold_min) / step))
    for i in range(n_steps + 1):
        thresholds.append(round(threshold_min + i * step, 10))

    sweep_results = [
        _confusion_matrix(valid_samples, th, direction) for th in thresholds
    ]

    recommended = _select_recommended(sweep_results, criterion)

    return SweepReport(
        engine=engine,
        signal_field=signal_field,
        direction=direction,
        sample_count=sample_count,
        excluded_count=excluded_count,
        sweep=sweep_results,
        recommended_threshold=recommended.threshold,
        recommended_metrics=recommended,
        criterion=criterion,
        metadata=metadata or {},
    )


def report_to_dict(report: SweepReport) -> dict[str, Any]:
    """``SweepReport`` を JSON serialize 可能な dict に変換。

    output schema は ``docs/research/calibration-corpus-sources.md`` で
    documenting する。
    """
    return {
        "engine": report.engine,
        "signal_field": report.signal_field,
        "direction": report.direction,
        "sample_count": report.sample_count,
        "excluded_count": report.excluded_count,
        "recommended_threshold": report.recommended_threshold,
        "recommended_metrics": _metrics_to_dict(report.recommended_metrics),
        "criterion": report.criterion,
        "sweep": [_metrics_to_dict(m) for m in report.sweep],
        "metadata": report.metadata,
    }


def _metrics_to_dict(m: ThresholdMetrics) -> dict[str, Any]:
    return {
        "threshold": m.threshold,
        "tp": m.tp,
        "fp": m.fp,
        "tn": m.tn,
        "fn": m.fn,
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "youden_j": m.youden_j,
        "false_reject_rate": m.false_reject_rate,
    }
