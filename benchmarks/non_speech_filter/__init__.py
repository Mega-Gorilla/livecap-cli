"""Non-speech filter evaluation harness (Issue #295 PR-0).

Canonical home for the corpus, metrics, mock engine, pipeline builder and
benchmark runner used by both the pytest baseline tests under
``tests/integration/non_speech_filter`` and the ad-hoc CLI runner.

Run via:

    python -m benchmarks.non_speech_filter --mode quick

See ``docs/benchmarks/non-speech-filter.md`` for full usage.
"""

from .corpus import CorpusItem, build_synthetic_corpus
from .metrics import (
    METRIC_SCHEMA_VERSION,
    REQUIRED_BASELINE_KEYS,
    CorpusEvaluation,
    PerItemResult,
    evaluate_pipeline,
    serializable_baseline,
)
from livecap_cli.audio.transient_detector import (
    TransientDetector,
    TransientDetectorConfig,
)

from .mock_engine import InstrumentedEngine, MockEngine
from .pipeline import (
    SUPPORTED_BACKENDS,
    build_pipeline,
    create_backend,
    load_real_corpus_items,
)
from .report import NonSpeechFilterReport, NonSpeechFilterRunRecord, new_report
from .runner import NonSpeechFilterBenchmarkConfig, NonSpeechFilterBenchmarkRunner

__all__ = [
    # corpus
    "CorpusItem",
    "build_synthetic_corpus",
    # metrics
    "METRIC_SCHEMA_VERSION",
    "REQUIRED_BASELINE_KEYS",
    "CorpusEvaluation",
    "PerItemResult",
    "evaluate_pipeline",
    "serializable_baseline",
    # engine adapters
    "MockEngine",
    "InstrumentedEngine",
    # detector (re-export from livecap_cli.audio for sweep convenience)
    "TransientDetector",
    "TransientDetectorConfig",
    # pipeline
    "SUPPORTED_BACKENDS",
    "build_pipeline",
    "create_backend",
    "load_real_corpus_items",
    # report
    "NonSpeechFilterReport",
    "NonSpeechFilterRunRecord",
    "new_report",
    # runner
    "NonSpeechFilterBenchmarkConfig",
    "NonSpeechFilterBenchmarkRunner",
]
