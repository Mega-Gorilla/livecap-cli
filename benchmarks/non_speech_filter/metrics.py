"""Metrics for the non-speech filter evaluation harness (Issue #295 PR-0).

Provides ``CorpusEvaluation`` and ``evaluate_pipeline()``: given a constructed
``StreamTranscriber`` and a list of ``CorpusItem``, drives the audio through
the pipeline and aggregates baseline metrics.
"""

from __future__ import annotations

import queue as _queue_mod
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from livecap_cli.transcription.result import TranscriptionResult as _StreamTranscriptionResult

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
    # codex-review on #312 3rd round Item 1 (HIGH): 既存 `speech_recall` /
    # `short_utterance_recall` は engine call (= `triggered`) で計測する
    # pre-filter recall。confidence filter が legit speech を drop しても
    # engine call は発生しているため `speech_recall=1.0` のまま user の
    # 字幕 stream には何も出ない、という乖離が起き得る。post-filter recall
    # を独立 metric として追加し、user の subtitle stream に実際に届く speech
    # 比率を直接測定する。
    post_filter_speech_recall: float | None
    post_filter_short_utterance_recall: float | None
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
                "post_filter_speech_recall": self.post_filter_speech_recall,
                "post_filter_short_utterance_recall": self.post_filter_short_utterance_recall,
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


def _collect_post_filter_texts(
    transcriber: Any, finalized_results: list[Any]
) -> list[str]:
    """confidence filter 通過後の final text を全て集める helper。

    StreamTranscriber が user の subtitle stream に渡す text の発生源は 2 つ:

    1. ``finalize()`` 戻り値 — 最終 VAD segment と coalescer flush 由来の
       final result。``_result_queue`` には put されず caller に返却される
       (``stream.py:460-486``)。
    2. ``_result_queue`` — ``feed_audio`` 経路で確定して ``_emit_result``
       経由で put された final result。``InterimResult`` も同 queue に
       混在 (``stream.py:413``) するが、interim は subtitle stream に
       finalize されないので post-filter metric には含めない。

    両者を合算した結果が「user の字幕に届く text」と等価。filter が drop した
    segment は stream.py 側で ``return None`` され、ここでは観測されない。

    codex-review on #312 3rd round Item 2 (MED): private API
    (``_result_queue``) access を本 helper に集約し、将来 StreamTranscriber
    の queue 実装が変わったときに直す箇所を 1 箇所に閉じ込める。
    """
    texts: list[str] = []
    for r in finalized_results:
        text_attr = getattr(r, "text", None)
        if isinstance(text_attr, str):
            texts.append(text_attr)
    result_queue = getattr(transcriber, "_result_queue", None)
    if result_queue is not None:
        while True:
            try:
                queued = result_queue.get_nowait()
            except _queue_mod.Empty:
                break
            if isinstance(queued, _StreamTranscriptionResult):
                text_attr = getattr(queued, "text", None)
                if isinstance(text_attr, str):
                    texts.append(text_attr)
            # InterimResult / その他は意図的に無視
    return texts


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
    # codex-review on #312 3rd round Item 1 (HIGH): post-filter recall
    # counter。filter 後に user の字幕に届く text を出した positive item を
    # 数える。measure_hallucination=True 時のみ意味を持つ。
    pos_post_filter_count = 0
    short_post_filter_count = 0

    for item in corpus:
        transcriber, engine = transcriber_factory()
        start_count = int(getattr(engine, "transcribe_count", 0))
        error: str | None = None

        # ``finalize()`` の戻り値は ``_result_queue`` には put されず caller に
        # 返却される設計 (stream.py:486)。``feed_audio`` 経路は ``_emit_result``
        # 経由で queue にも put されるため両方を合算する必要がある
        # (codex-review on #312 Item 1 で発覚した metric bug の修正)。
        finalized_results: list[Any] = []
        t0 = time.perf_counter()
        try:
            transcriber.feed_audio(item.audio, sample_rate=sample_rate)
            finalize_ret = transcriber.finalize()
            if isinstance(finalize_ret, list):
                finalized_results = finalize_ret
        except Exception as exc:
            if fail_fast:
                raise
            error = repr(exc)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # codex-review on #312 1st/2nd/3rd round で進化した post-filter text
        # 収集。詳細は ``_collect_post_filter_texts`` 内 docstring 参照。
        post_filter_texts: list[str] = []
        if measure_hallucination and error is None:
            post_filter_texts = _collect_post_filter_texts(
                transcriber, finalized_results
            )

        end_count = int(getattr(engine, "transcribe_count", 0))
        calls = max(0, end_count - start_count)
        triggered = calls > 0 and error is None

        sample_texts: tuple[str, ...] = ()
        post_filter_has_text = False  # codex-review #312 3rd round Item 1 用
        if measure_hallucination and error is None:
            recent = list(getattr(engine, "last_texts", ())[start_count:end_count])
            non_empty = [t for t in recent if isinstance(t, str) and t.strip()]
            sample_texts = tuple(recent)
            if item.kind == "negative" and non_empty:
                neg_non_empty_count += 1
            # post-filter side: helper から取れた text (filter で drop された分は欠落)
            post_filter_non_empty = [
                t for t in post_filter_texts if isinstance(t, str) and t.strip()
            ]
            post_filter_has_text = bool(post_filter_non_empty)
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
            # codex-review on #312 3rd round Item 1 (HIGH): post-filter recall。
            # measure_hallucination=False (MockEngine path) では post_filter_texts
            # が空のまま False を返すので post_filter_has_text=False。その場合
            # CorpusEvaluation 側で None を返すよう post_filter_count を
            # measure_hallucination で gate する (下記 _ratio 引数)。
            if post_filter_has_text:
                pos_post_filter_count += 1
            if item.is_short_utterance:
                short_total += 1
                if triggered:
                    short_trigger += 1
                if post_filter_has_text:
                    short_post_filter_count += 1

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
        # codex-review on #312 3rd round Item 1 (HIGH): post-filter recall は
        # measure_hallucination=True (engine 実走、post_filter_texts が意味を
        # 持つ条件) でのみ報告。MockEngine path では None を返す。
        post_filter_speech_recall=(
            _ratio(pos_post_filter_count, pos_total)
            if measure_hallucination
            else None
        ),
        post_filter_short_utterance_recall=(
            _ratio(short_post_filter_count, short_total)
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
