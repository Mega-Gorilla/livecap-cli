"""Baseline tests for the non-speech filter evaluation harness (PR-0).

These tests exercise the current production pipeline (NoiseGate + VAD +
EnergyGate) against the synthetic corpus and persist per-backend metric
snapshots to ``baselines/{backend}.json``. Subsequent Phase 1 PRs (B/C/A)
read those snapshots to assert improvement without regressing
short-utterance recall.

Markers:
- ``evaluation_harness``: opt-in via ``-m evaluation_harness``.
- ``engine_smoke``: required for the optional hallucination measurement
  (only runs when ``LIVECAP_ENABLE_HALLUCINATION_EVAL=1`` is set).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from benchmarks.non_speech_filter import (
    METRIC_SCHEMA_VERSION,
    REQUIRED_BASELINE_KEYS,
    CorpusItem,
    MockEngine,
    evaluate_pipeline,
    serializable_baseline,
)

from .conftest import build_baseline


# ---------- Synthetic corpus baseline ------------------------------------


@pytest.mark.evaluation_harness
def test_baseline_synthetic_corpus(
    backend_type: str,
    transcriber_factory,
    synthetic_corpus_items: list[CorpusItem],
    baselines_dir: Path,
) -> None:
    """Measure baseline metrics for the synthetic corpus and persist them.

    The result is written to ``baselines/{backend}.json`` so downstream PRs
    can compute deltas without re-running this harness.
    """
    evaluation = evaluate_pipeline(
        transcriber_factory,
        synthetic_corpus_items,
        backend_name=backend_type,
        measure_hallucination=False,
    )

    payload = serializable_baseline(evaluation)
    output = baselines_dir / f"{backend_type}.json"
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Sanity invariants (baseline values can vary, but totals must be > 0).
    assert evaluation.negative_total > 0, "Synthetic corpus must contain negative items"
    assert evaluation.positive_total > 0, "Synthetic corpus must contain positive items"
    assert evaluation.short_total > 0, "Synthetic corpus must contain short utterances"
    assert evaluation.added_latency_p50_ms >= 0.0
    assert evaluation.added_latency_p95_ms >= evaluation.added_latency_p50_ms
    assert set(evaluation.per_label.keys()) == {item.label for item in synthetic_corpus_items}


@pytest.mark.evaluation_harness
def test_baseline_invariants(backend_type: str, baselines_dir: Path) -> None:
    """The persisted baseline JSON must satisfy the documented schema.

    This guards against silent schema drift between PR-0 and downstream PRs.
    """
    output = baselines_dir / f"{backend_type}.json"
    if not output.exists():
        pytest.skip(
            f"Baseline JSON not yet generated for backend {backend_type}; "
            "run test_baseline_synthetic_corpus first."
        )
    payload = json.loads(output.read_text(encoding="utf-8"))

    for key in REQUIRED_BASELINE_KEYS:
        assert key in payload, f"Baseline JSON missing required key: {key!r}"

    assert payload["schema_version"] == METRIC_SCHEMA_VERSION
    assert payload["backend_name"] == backend_type

    metrics = payload["metrics"]
    for required_metric in (
        "false_asr_trigger_rate",
        "speech_recall",
        "short_utterance_recall",
        "added_latency_p50_ms",
        "added_latency_p95_ms",
    ):
        assert required_metric in metrics, f"Missing metric: {required_metric!r}"

    totals = payload["totals"]
    assert totals["negative"] > 0
    assert totals["positive"] > 0
    assert totals["short_utterance"] > 0


# ---------- Real corpus baseline (opt-in) --------------------------------


@pytest.mark.evaluation_harness
def test_baseline_real_corpus(
    backend_type: str,
    transcriber_factory,
    real_corpus_items: list[CorpusItem],
    baselines_dir: Path,
) -> None:
    """Real-audio baseline. Skipped unless ``LIVECAP_NON_SPEECH_CORPUS_DIR`` is set."""
    if not real_corpus_items:
        pytest.skip("Real corpus is empty")
    evaluation = evaluate_pipeline(
        transcriber_factory,
        real_corpus_items,
        backend_name=backend_type,
        measure_hallucination=False,
    )
    payload = serializable_baseline(evaluation)
    output = baselines_dir / f"{backend_type}.real.json"
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    assert evaluation.negative_total + evaluation.positive_total == len(real_corpus_items)


# ---------- Hallucination measurement (opt-in, engine_smoke) -------------


def _hallucination_enabled() -> bool:
    return os.environ.get("LIVECAP_ENABLE_HALLUCINATION_EVAL", "").lower() in {
        "1",
        "true",
        "yes",
    }


@pytest.mark.evaluation_harness
@pytest.mark.engine_smoke
@pytest.mark.skipif(
    not _hallucination_enabled(),
    reason="LIVECAP_ENABLE_HALLUCINATION_EVAL not set; engine run skipped",
)
def test_baseline_hallucination_marker_present(
    backend_type: str,
    synthetic_corpus_items: list[CorpusItem],
) -> None:
    """Marker probe: ensures the hallucination evaluation path is wired up.

    The actual engine-driven hallucination measurement happens in
    ``benchmarks/non_speech_filter`` (ad-hoc runner). This CI probe simply
    verifies that the gate fixtures and ``measure_hallucination=True`` code
    path are reachable when opt-in is requested.
    """

    def factory() -> tuple[object, MockEngine]:
        return build_baseline(
            backend_type,
            mock_engine_factory=lambda: MockEngine(return_text="ご視聴ありがとうございました"),
        )

    evaluation = evaluate_pipeline(
        factory,
        synthetic_corpus_items,
        backend_name=backend_type,
        measure_hallucination=True,
    )

    assert evaluation.non_empty_hallucination_rate is not None
    assert 0.0 <= evaluation.non_empty_hallucination_rate <= 1.0
