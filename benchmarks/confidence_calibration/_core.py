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
class BreakdownReport:
    """1 breakdown key に対する per-value 混同行列 sweep の集計 (Phase 6a)。

    Attributes:
        key: 元 manifest field 名 (e.g. ``"snr_db"`` / ``"subtype"``)。 debug 用の
            自己記述。
        value_counts: value → sample count のマップ (e.g. ``{"10.0": 50, "__none__": 449}``)。
            ``None`` 値は ``"__none__"`` bucket に集約 (``_breakdown_key`` で正規化)。
        sweep_by_value: value → 全 threshold での ``ThresholdMetrics`` list。
            各 bucket の sample subset に ``_confusion_matrix`` を全 threshold で
            適用した結果。
    """

    key: str
    value_counts: dict[str, int]
    sweep_by_value: dict[str, list[ThresholdMetrics]]


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
    # Phase 6a additive: per-metadata-key 混同行列 breakdown (default 空)。
    # ``--breakdown-by`` CLI flag / ``sweep_threshold(breakdown_by=...)`` param
    # で指定された key ごとに populate。 未指定時は ``{}`` で JSON 上 additive のみ。
    breakdown: dict[str, BreakdownReport] = field(default_factory=dict)


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


def _breakdown_key(value: Any) -> str:
    """Breakdown 用の bucket key を deterministic に返す (Phase 6a)。

    - ``None`` → ``"__none__"`` (bucket 集約 sentinel、 clean speech が
      ``snr_db`` field を持たない等のケースを表す)
    - float / int / str / bool → ``str(value)`` (例: ``10.0`` → ``"10.0"``、
      ``-5.0`` → ``"-5.0"``、 ``"clapping"`` → ``"clapping"``)
    """
    if value is None:
        return "__none__"
    return str(value)


def compute_breakdowns(
    samples: list[LabeledSample],
    breakdown_key: str,
    thresholds: list[float],
    direction: Direction,
) -> BreakdownReport:
    """1 breakdown key について per-value × per-threshold の混同行列 sweep を計算 (Phase 6a)。

    Samples を ``metadata[breakdown_key]`` の値でグルーピングし、 各 bucket に対して
    全 threshold で ``_confusion_matrix`` を適用する。 sample が該当 key を持たない
    場合や値が ``None`` の場合は ``"__none__"`` bucket に集約 (``_breakdown_key`` 参照)。

    Args:
        samples: valid samples (``signal_value is not None``) の list。 caller で
            事前フィルタ済であることを想定 (``sweep_threshold`` 内で呼出)。
        breakdown_key: manifest 由来の metadata field 名 (e.g. ``"snr_db"`` /
            ``"subtype"`` / ``"noise_source_dataset"``)。
        thresholds: 全 threshold list (``sweep_threshold`` の grid と同一)。
        direction: signal direction (sample-level classification に使用)。

    Returns:
        ``BreakdownReport``。 存在しない key の場合、 全 sample が ``"__none__"``
        bucket に入る (soft warn は caller 側の責務)。
    """
    buckets: dict[str, list[LabeledSample]] = {}
    for s in samples:
        bucket_key = _breakdown_key(s.metadata.get(breakdown_key))
        buckets.setdefault(bucket_key, []).append(s)

    value_counts = {k: len(v) for k, v in buckets.items()}

    sweep_by_value = {
        bucket_key: [_confusion_matrix(bucket, th, direction) for th in thresholds]
        for bucket_key, bucket in buckets.items()
    }

    return BreakdownReport(
        key=breakdown_key,
        value_counts=value_counts,
        sweep_by_value=sweep_by_value,
    )


def _select_recommended(
    sweep: list[ThresholdMetrics],
    criterion: Criterion,
    direction: Direction,
) -> ThresholdMetrics:
    if not sweep:
        raise ValueError("Cannot recommend from empty sweep")
    if criterion == "f1":
        score = lambda m: m.f1
    elif criterion == "youden_j":
        score = lambda m: m.youden_j
    elif criterion == "precision":
        score = lambda m: m.precision
    elif criterion == "recall":
        score = lambda m: m.recall
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    # Tie-break (PR #339 codex-review fix): criterion 値同点時、**より
    # conservative な threshold** (= 少数しか reject されない) を採用する。
    # これにより false reject (speech の reject) が増えるのを最小化、本
    # harness の主目的 (Issue #334 noisy_speech false reject 抑制) と整合。
    #
    # - reject_if_less (avg_logprob / token_confidence_mean):
    #     value < threshold で reject
    #     threshold **小** → 少数しか < threshold にならない → reject 少 (conservative)
    # - reject_if_greater (no_speech_prob):
    #     value > threshold で reject
    #     threshold **大** → 少数しか > threshold にならない → reject 少 (conservative)
    if direction == "reject_if_less":
        tie_break = lambda m: -m.threshold  # 小 threshold ほど max key 大
    else:  # reject_if_greater
        tie_break = lambda m: m.threshold  # 大 threshold ほど max key 大

    return max(sweep, key=lambda m: (score(m), tie_break(m)))


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
    breakdown_by: Optional[list[str]] = None,
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
        breakdown_by: Phase 6a additive。 metadata key list を指定すると、 各 key
            について per-value 混同行列 sweep を計算して ``SweepReport.breakdown``
            に格納。 ``None`` (default) → breakdown なし (現行挙動と 100% 等価)。
            e.g. ``["snr_db", "subtype"]`` で SNR 別 FRR + noise category 別
            pass rate を 1 sweep で取得。

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

    recommended = _select_recommended(sweep_results, criterion, direction)

    # Phase 6a: per-metadata-key breakdown (default 空 dict、 backward compat)
    report_breakdowns: dict[str, BreakdownReport] = {}
    if breakdown_by:
        for key in breakdown_by:
            report_breakdowns[key] = compute_breakdowns(
                valid_samples, key, thresholds, direction
            )

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
        breakdown=report_breakdowns,
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
        # Phase 6a additive: 未指定時は空 dict、 Phase 1 report の schema と互換
        "breakdown": {
            key: {
                "key": br.key,
                "value_counts": br.value_counts,
                "sweep_by_value": {
                    value: [_metrics_to_dict(m) for m in metrics_list]
                    for value, metrics_list in br.sweep_by_value.items()
                },
            }
            for key, br in report.breakdown.items()
        },
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
