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
    error: str | None = None  # repr(exc) captured when fail_fast=False


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

    non_empty_hallucination_rate: float | None  # engine 実走時のみ (pre-filter、engine が emit した非空 text)
    # PR-A.3 (Issue #308): confidence filter 適用 **後** の non-empty text 率。
    # `non_empty_hallucination_rate` (pre-filter、engine 直出力) と比較することで
    # filter が user の subtitle stream にどれだけ影響したかを直接測定可能。
    # measure_hallucination=True かつ engine 実走時のみ計算、それ以外は None。
    post_filter_hallucination_rate: float | None
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
                "post_filter_hallucination_rate": self.post_filter_hallucination_rate,
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
    fail_fast: bool = True,
) -> CorpusEvaluation:
    """Run each corpus item through a fresh pipeline and aggregate metrics.

    Args:
        transcriber_factory: Zero-arg factory producing a fresh
            ``StreamTranscriber`` (or compatible interface) and the engine it
            wraps, as a tuple ``(transcriber, engine)``. The engine must
            expose ``transcribe_count: int`` and (when
            ``measure_hallucination`` is True) ``last_texts: list[str]``.
            Real engines are wrapped via ``InstrumentedEngine`` by the
            benchmark runner; ``MockEngine`` exposes the surface natively.
        corpus: Iterable of ``CorpusItem``.
        sample_rate: Sample rate of the corpus audio (default 16 kHz).
        measure_hallucination: If True, ``non_empty_hallucination_rate`` is
            computed from engine output text. Requires the engine to record
            ``last_texts`` (``MockEngine`` and ``InstrumentedEngine`` both do).
        backend_name: Descriptive name used in the resulting evaluation.
        fail_fast: When True (default, suited for pytest), pipeline
            exceptions surface immediately. When False, the exception is
            captured in ``per_label[item.label]['error']`` and the item is
            counted as not triggering ASR so the benchmark runner can produce
            a partial report on environment-specific failures.
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
    neg_post_filter_non_empty_count = 0

    for item in corpus:
        transcriber, engine = transcriber_factory()
        start_count = int(getattr(engine, "transcribe_count", 0))
        error: str | None = None

        t0 = time.perf_counter()
        try:
            transcriber.feed_audio(item.audio, sample_rate=sample_rate)
            transcriber.finalize()
        except Exception as exc:
            if fail_fast:
                raise
            error = repr(exc)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # PR-A.3: confidence filter 適用 **後** の TranscriptionResult を queue から
        # drain して post_filter_hallucination_rate を計算。filter が drop した
        # segment は stream.py 側で None drop されて queue に流れないため、ここで
        # 取れるのは「user の字幕に出る text」と同等。
        post_filter_texts: list[str] = []
        if measure_hallucination and error is None and hasattr(transcriber, "get_result"):
            while True:
                tr_result = transcriber.get_result(timeout=0)
                if tr_result is None:
                    break
                text_attr = getattr(tr_result, "text", "")
                if isinstance(text_attr, str):
                    post_filter_texts.append(text_attr)

        end_count = int(getattr(engine, "transcribe_count", 0))
        calls = max(0, end_count - start_count)
        triggered = calls > 0 and error is None

        sample_texts: tuple[str, ...] = ()
        if measure_hallucination and error is None:
            recent = list(getattr(engine, "last_texts", ())[start_count:end_count])
            non_empty = [t for t in recent if isinstance(t, str) and t.strip()]
            sample_texts = tuple(recent)
            if item.kind == "negative" and non_empty:
                neg_non_empty_count += 1
            # post-filter side: queue から取れた text (filter で drop された分は欠落)
            post_filter_non_empty = [
                t for t in post_filter_texts if isinstance(t, str) and t.strip()
            ]
            if item.kind == "negative" and post_filter_non_empty:
                neg_post_filter_non_empty_count += 1

        per_item = PerItemResult(
            label=item.label,
            kind=item.kind,
            is_short_utterance=item.is_short_utterance,
            engine_calls=calls,
            latency_ms=elapsed_ms,
            sample_texts=sample_texts,
            error=error,
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
            "error": error,
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
        post_filter_hallucination_rate=(
            _ratio(neg_post_filter_non_empty_count, neg_total)
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
