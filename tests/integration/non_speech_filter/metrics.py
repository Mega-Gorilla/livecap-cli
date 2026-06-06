"""Metrics for the non-speech filter evaluation harness (Issue #295 PR-0).

Provides ``CorpusEvaluation`` and ``evaluate_pipeline()``: given a constructed
``StreamTranscriber`` and a list of ``CorpusItem``, drives the audio through
the pipeline and aggregates baseline metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from .corpus import CorpusItem


@dataclass(frozen=True)
class PerItemResult:
    """Per-corpus-item observation captured during a baseline run."""

    label: str
    kind: str
    is_short_utterance: bool
    engine_calls: int
    latency_ms: float
    sample_texts: tuple[str, ...]


@dataclass(frozen=True)
class CorpusEvaluation:
    """Aggregated metrics over one (backend, pipeline) run.

    All rates are in ``[0.0, 1.0]`` (or ``None`` if the denominator is zero).
    Latencies are in milliseconds.
    """

    backend_name: str
    negative_total: int
    positive_total: int
    short_total: int

    false_asr_trigger_rate: float | None
    speech_recall: float | None
    short_utterance_recall: float | None

    non_empty_hallucination_rate: float | None  # engine 実走時のみ
    added_latency_p50_ms: float
    added_latency_p95_ms: float

    per_label: dict[str, dict[str, Any]] = field(default_factory=dict)
    per_item: tuple[PerItemResult, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "totals": {
                "negative": self.negative_total,
                "positive": self.positive_total,
                "short_utterance": self.short_total,
            },
            "metrics": {
                "false_asr_trigger_rate": self.false_asr_trigger_rate,
                "speech_recall": self.speech_recall,
                "short_utterance_recall": self.short_utterance_recall,
                "non_empty_hallucination_rate": self.non_empty_hallucination_rate,
                "added_latency_p50_ms": self.added_latency_p50_ms,
                "added_latency_p95_ms": self.added_latency_p95_ms,
            },
            "per_label": self.per_label,
        }


def _percentile_ms(times_ms: list[float], p: float) -> float:
    if not times_ms:
        return 0.0
    arr = np.asarray(times_ms, dtype=np.float64)
    return float(np.percentile(arr, p))


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def evaluate_pipeline(
    transcriber_factory: Callable[[], Any],
    corpus: Iterable[CorpusItem],
    *,
    sample_rate: int = 16000,
    measure_hallucination: bool = False,
    backend_name: str = "unknown",
) -> CorpusEvaluation:
    """Run each corpus item through a fresh pipeline and aggregate metrics.

    Args:
        transcriber_factory: Zero-arg factory producing a fresh
            ``StreamTranscriber`` (or compatible interface) and the engine it
            wraps, as a tuple ``(transcriber, engine)``.  The engine is
            expected to expose ``transcribe_count: int`` and (optionally)
            ``last_texts: list[str]`` so non-empty hallucination can be
            measured when ``measure_hallucination`` is True.
        corpus: Iterable of ``CorpusItem``.
        sample_rate: Sample rate of the corpus audio (default 16 kHz).
        measure_hallucination: If True, ``non_empty_hallucination_rate`` is
            computed from engine output text. Requires a real engine.
        backend_name: Descriptive name used in the resulting evaluation.
    """
    per_items: list[PerItemResult] = []
    per_label: dict[str, dict[str, Any]] = {}
    latencies_ms: list[float] = []

    neg_total = 0
    neg_trigger = 0
    pos_total = 0
    pos_trigger = 0
    short_total = 0
    short_trigger = 0
    neg_non_empty_count = 0

    for item in corpus:
        transcriber, engine = transcriber_factory()
        start_count = int(getattr(engine, "transcribe_count", 0))

        t0 = time.perf_counter()
        transcriber.feed_audio(item.audio, sample_rate=sample_rate)
        try:
            transcriber.finalize()
        except Exception:  # pragma: no cover - defensive guard for finalize
            # finalize() failures should never mask the metric run.
            pass
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        end_count = int(getattr(engine, "transcribe_count", 0))
        calls = max(0, end_count - start_count)
        triggered = calls > 0

        sample_texts: tuple[str, ...] = ()
        if measure_hallucination:
            recent = list(getattr(engine, "last_texts", ())[start_count:end_count])
            non_empty = [t for t in recent if isinstance(t, str) and t.strip()]
            sample_texts = tuple(recent)
            if item.kind == "negative" and non_empty:
                neg_non_empty_count += 1

        per_item = PerItemResult(
            label=item.label,
            kind=item.kind,
            is_short_utterance=item.is_short_utterance,
            engine_calls=calls,
            latency_ms=elapsed_ms,
            sample_texts=sample_texts,
        )
        per_items.append(per_item)
        latencies_ms.append(elapsed_ms)
        per_label[item.label] = {
            "kind": item.kind,
            "is_short_utterance": item.is_short_utterance,
            "engine_calls": calls,
            "triggered": triggered,
            "latency_ms": elapsed_ms,
            "sample_texts": list(sample_texts),
        }

        if item.kind == "negative":
            neg_total += 1
            if triggered:
                neg_trigger += 1
        else:
            pos_total += 1
            if triggered:
                pos_trigger += 1
            if item.is_short_utterance:
                short_total += 1
                if triggered:
                    short_trigger += 1

    return CorpusEvaluation(
        backend_name=backend_name,
        negative_total=neg_total,
        positive_total=pos_total,
        short_total=short_total,
        false_asr_trigger_rate=_ratio(neg_trigger, neg_total),
        speech_recall=_ratio(pos_trigger, pos_total),
        short_utterance_recall=_ratio(short_trigger, short_total),
        non_empty_hallucination_rate=(
            _ratio(neg_non_empty_count, neg_total)
            if measure_hallucination
            else None
        ),
        added_latency_p50_ms=_percentile_ms(latencies_ms, 50.0),
        added_latency_p95_ms=_percentile_ms(latencies_ms, 95.0),
        per_label=per_label,
        per_item=tuple(per_items),
    )


METRIC_SCHEMA_VERSION: str = "1"
"""Baseline JSON schema version. Bump when keys change."""


REQUIRED_BASELINE_KEYS: tuple[str, ...] = (
    "schema_version",
    "backend_name",
    "totals",
    "metrics",
    "per_label",
)
"""Required top-level keys in the per-backend baseline JSON file.

Used by ``test_baseline_invariants`` to detect schema regressions early.
"""


def serializable_baseline(evaluation: CorpusEvaluation) -> dict[str, Any]:
    """Convert a ``CorpusEvaluation`` to a JSON-serializable baseline payload."""
    payload = evaluation.to_dict()
    payload["schema_version"] = METRIC_SCHEMA_VERSION
    return payload
