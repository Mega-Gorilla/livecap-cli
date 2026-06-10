"""Class-level + reject-signal-level precision/recall (Issue #305 v3 metric report).

Two-axis metric report per Issue #305 v3:

1. **Class-level**: per-class precision/recall on individual AudioSet
   indices. Visualises multi-label calibration drift (a model that is sharp
   on ``Knock`` but soft on ``Tap`` would show up here).
2. **Reject-signal-level**: per-policy precision/recall after aggregating
   the target classes with one of the three threshold policies defined in
   :mod:`benchmarks.sed.class_mapping`. This is the metric used for the
   PR-D0 Go/no-go verdict.

Aggregation unit per Issue #305 v3 is **clip-level max**: for each
recorded clip we take the maximum reject score across the clip's 1-second
windows, then threshold against that single scalar. ``window-level``
probabilities remain available in :class:`PerClipResult` for later
analyses (decision document writer).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from benchmarks.sed.class_mapping import (
    NUM_AUDIOSET_CLASSES,
    POLICIES,
    PolicyFn,
)


ClipKind = Literal["negative", "positive"]
"""``negative`` = should be rejected by SED; ``positive`` = must remain."""


@dataclass(frozen=True)
class PerClipResult:
    """Inference output for one corpus clip.

    ``per_window_probs`` shape is ``(n_windows, 527)``. Class-level metrics
    use the per-clip *max* across windows (single-shot multi-label tagging),
    matching the EfficientAT inference style.
    """

    label: str
    kind: ClipKind
    is_short_utterance: bool
    per_window_probs: np.ndarray

    def __post_init__(self) -> None:
        if self.per_window_probs.ndim != 2:
            raise ValueError(
                f"per_window_probs must be 2-D, got shape {self.per_window_probs.shape}"
            )
        if self.per_window_probs.shape[1] != NUM_AUDIOSET_CLASSES:
            raise ValueError(
                f"per_window_probs must have {NUM_AUDIOSET_CLASSES} columns, "
                f"got {self.per_window_probs.shape[1]}"
            )

    def clip_max(self) -> np.ndarray:
        """``(527,)`` vector of per-class max across the clip's windows."""

        if self.per_window_probs.shape[0] == 0:
            return np.zeros(NUM_AUDIOSET_CLASSES, dtype=np.float32)
        return self.per_window_probs.max(axis=0)


@dataclass(frozen=True)
class ConfusionCounts:
    """Per-threshold binary confusion counts (clip granularity)."""

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0


@dataclass(frozen=True)
class ClassMetric:
    """Class-level precision/recall using clip-max scores."""

    class_index: int
    class_name: str
    threshold: float
    precision: float
    recall: float
    counts: ConfusionCounts


@dataclass(frozen=True)
class SweepCurve:
    """Reject-signal-level threshold sweep for one policy.

    ``thresholds`` / ``precision`` / ``recall`` are aligned 1-D arrays.
    ``per_clip_scores`` stores the clip-level reject score for each clip so
    the decision document can show which threshold flags ``desk_tap``.
    """

    policy: str
    thresholds: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    per_clip_scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Verdict:
    """Outcome of the Issue #305 v3 provisional gate check."""

    target_label: str
    precision_floor: float
    recall_floor: float
    is_provisional: bool
    passed: bool
    passing_policies: tuple[str, ...]
    chosen_policy: str | None
    chosen_threshold: float | None
    chosen_precision: float | None
    chosen_recall: float | None
    target_flagged_at_chosen_threshold: bool | None
    notes: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Reject-signal-level metrics (clip granularity)
# ---------------------------------------------------------------------------


def _clip_reject_scores(
    results: Iterable[PerClipResult], policy: PolicyFn
) -> dict[str, tuple[ClipKind, float]]:
    scores: dict[str, tuple[ClipKind, float]] = {}
    for clip in results:
        clip_max_probs = clip.clip_max()
        # Score the clip-max vector once (matches the clip-level max
        # decision unit pinned in Issue #305 v3).
        scores[clip.label] = (clip.kind, float(policy(clip_max_probs)))
    return scores


def _confusion_at_threshold(
    scores: Mapping[str, tuple[ClipKind, float]], threshold: float
) -> ConfusionCounts:
    tp = fp = fn = tn = 0
    for kind, score in scores.values():
        flagged = score >= threshold
        if kind == "negative":  # should reject
            if flagged:
                tp += 1
            else:
                fn += 1
        else:  # positive — must remain
            if flagged:
                fp += 1
            else:
                tn += 1
    return ConfusionCounts(tp=tp, fp=fp, fn=fn, tn=tn)


def compute_reject_signal_curve(
    results: Iterable[PerClipResult],
    policy_name: str,
    thresholds: Iterable[float],
) -> SweepCurve:
    """Sweep thresholds for one policy and produce a P-R curve."""

    if policy_name not in POLICIES:
        raise KeyError(f"Unknown policy {policy_name!r}; expected one of {sorted(POLICIES)}")
    policy = POLICIES[policy_name]

    results_list = list(results)
    scores = _clip_reject_scores(results_list, policy)
    thr_array = np.asarray(list(thresholds), dtype=np.float32)
    precisions = np.zeros_like(thr_array)
    recalls = np.zeros_like(thr_array)

    for i, t in enumerate(thr_array):
        counts = _confusion_at_threshold(scores, float(t))
        precisions[i] = counts.precision
        recalls[i] = counts.recall

    return SweepCurve(
        policy=policy_name,
        thresholds=thr_array,
        precision=precisions,
        recall=recalls,
        per_clip_scores={label: score for label, (_kind, score) in scores.items()},
    )


# ---------------------------------------------------------------------------
# Class-level metrics (clip granularity)
# ---------------------------------------------------------------------------


def compute_class_level_metrics(
    results: Iterable[PerClipResult],
    class_indices: Iterable[int],
    class_names: Iterable[str],
    threshold: float,
) -> list[ClassMetric]:
    """Per-class precision/recall using clip-max scores.

    The same ``threshold`` is applied to every class index — useful for
    presenting a side-by-side table in the decision document.
    """

    results_list = list(results)
    indices_list = list(class_indices)
    names_list = list(class_names)
    if len(indices_list) != len(names_list):
        raise ValueError("class_indices and class_names must have matching length")

    out: list[ClassMetric] = []
    for idx, name in zip(indices_list, names_list):
        tp = fp = fn = tn = 0
        for clip in results_list:
            clip_score = float(clip.clip_max()[idx])
            flagged = clip_score >= threshold
            if clip.kind == "negative":
                if flagged:
                    tp += 1
                else:
                    fn += 1
            else:
                if flagged:
                    fp += 1
                else:
                    tn += 1
        counts = ConfusionCounts(tp=tp, fp=fp, fn=fn, tn=tn)
        out.append(
            ClassMetric(
                class_index=idx,
                class_name=name,
                threshold=threshold,
                precision=counts.precision,
                recall=counts.recall,
                counts=counts,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Provisional gate (Issue #305 v3)
# ---------------------------------------------------------------------------


PROVISIONAL_PRECISION_FLOOR = 0.70
PROVISIONAL_RECALL_FLOOR = 0.50


def provisional_gate_verdict(
    curves: Iterable[SweepCurve],
    target_label: str,
    precision_floor: float = PROVISIONAL_PRECISION_FLOOR,
    recall_floor: float = PROVISIONAL_RECALL_FLOOR,
) -> Verdict:
    """Apply the Issue #305 v3 provisional accuracy gate.

    The gate passes if **at least one** policy / threshold pair satisfies:

    1. corpus-level precision ≥ ``precision_floor``
    2. corpus-level recall ≥ ``recall_floor``
    3. ``target_label`` is flagged as reject at the chosen threshold

    When multiple (policy, threshold) pairs satisfy the gate, the one with
    the highest recall (and, as a tiebreaker, the highest precision) is
    reported as the chosen pair.

    The verdict is always marked ``is_provisional=True`` because the
    underlying corpus is statistically weak; the decision document must
    record a corpus-expansion judgement for PR-D1.
    """

    curves_list = list(curves)
    passing_policies: list[str] = []
    best: tuple[float, float, str, float, float, float, bool] | None = None

    for curve in curves_list:
        target_score = curve.per_clip_scores.get(target_label)
        for i, threshold in enumerate(curve.thresholds):
            p = float(curve.precision[i])
            r = float(curve.recall[i])
            if p < precision_floor or r < recall_floor:
                continue
            target_flagged = (
                target_score is not None and target_score >= float(threshold)
            )
            if not target_flagged:
                continue
            if curve.policy not in passing_policies:
                passing_policies.append(curve.policy)
            candidate = (r, p, curve.policy, float(threshold), p, r, target_flagged)
            if best is None or candidate > best:
                best = candidate

    if best is None:
        return Verdict(
            target_label=target_label,
            precision_floor=precision_floor,
            recall_floor=recall_floor,
            is_provisional=True,
            passed=False,
            passing_policies=(),
            chosen_policy=None,
            chosen_threshold=None,
            chosen_precision=None,
            chosen_recall=None,
            target_flagged_at_chosen_threshold=None,
            notes=(
                "No (policy, threshold) pair satisfied both the precision "
                f"floor ({precision_floor:.2f}) and the recall floor "
                f"({recall_floor:.2f}) while flagging {target_label!r}.",
            ),
        )

    _r, _p, policy, threshold, precision, recall, target_flagged = best
    return Verdict(
        target_label=target_label,
        precision_floor=precision_floor,
        recall_floor=recall_floor,
        is_provisional=True,
        passed=True,
        passing_policies=tuple(passing_policies),
        chosen_policy=policy,
        chosen_threshold=threshold,
        chosen_precision=precision,
        chosen_recall=recall,
        target_flagged_at_chosen_threshold=target_flagged,
        notes=(
            "Provisional gate satisfied. Corpus is 6 clips (statistically "
            "weak) — PR-D1 must record a corpus-expansion judgement "
            "(ESC-50 / FSD50K subset vs status quo).",
        ),
    )


__all__ = [
    "PROVISIONAL_PRECISION_FLOOR",
    "PROVISIONAL_RECALL_FLOOR",
    "ClipKind",
    "PerClipResult",
    "ConfusionCounts",
    "ClassMetric",
    "SweepCurve",
    "Verdict",
    "compute_reject_signal_curve",
    "compute_class_level_metrics",
    "provisional_gate_verdict",
]
