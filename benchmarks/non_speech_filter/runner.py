"""Non-speech filter benchmark runner (Issue #295 PR-0).

Iterates over (backend × engine × corpus × run_index) and records baseline
metrics into a :class:`NonSpeechFilterReport`. The engine slot is either
``MockEngine`` (default) for fast filter-behavior measurement or a real
engine (e.g. ``whispers2t``) for hallucination text measurement.

This runner sits next to the pytest baseline tests but is independently
usable so it can drive real-audio corpora, real engines and multi-run
aggregation without inflating CI cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from livecap_cli.transcription.stream import StreamTranscriber

from .corpus import CorpusItem, build_synthetic_corpus
from .metrics import evaluate_pipeline
from .mock_engine import MockEngine
from .pipeline import (
    SUPPORTED_BACKENDS,
    build_pipeline,
    create_backend,
    load_real_corpus_items,
)
from .report import NonSpeechFilterReport, NonSpeechFilterRunRecord, new_report

logger = logging.getLogger(__name__)

DEFAULT_MOCK_ENGINE_NAME: str = "mock"


# ---------- Config ------------------------------------------------------------


@dataclass
class NonSpeechFilterBenchmarkConfig:
    """Configuration for one benchmark run.

    Attributes:
        mode: ``"quick"`` (single run, synthetic only) or ``"standard"``
            (multiple runs, synthetic + real if available).
        backends: VAD backends to evaluate. Each must be in
            ``SUPPORTED_BACKENDS``; unavailable ones are reported as skipped.
        engines: Engine identifiers; empty list means ``MockEngine`` only.
        corpus_dir: Path to real-audio fixtures (manifest.json + WAVs).
            ``None`` disables the real corpus.
        runs: Number of repetitions per cell (for noise averaging).
        device: Device hint passed to the real-engine factory.
        output_dir: Where the JSON + Markdown reports are written.
    """

    mode: str = "quick"
    backends: list[str] = field(default_factory=lambda: ["silero"])
    engines: list[str] = field(default_factory=list)
    corpus_dir: Optional[Path] = None
    runs: int = 1
    device: str = "auto"
    output_dir: Path = field(
        default_factory=lambda: Path("benchmark_results") / "non_speech_filter"
    )

    def normalised_backends(self) -> list[str]:
        unique: list[str] = []
        for name in self.backends:
            name = name.strip().lower()
            if not name:
                continue
            if name not in SUPPORTED_BACKENDS:
                raise ValueError(
                    f"Unsupported backend {name!r}; "
                    f"choose from {SUPPORTED_BACKENDS}"
                )
            if name not in unique:
                unique.append(name)
        return unique


# ---------- Runner -----------------------------------------------------------


def _make_pipeline_factory(backend_name: str, engine_factory):
    """Bind ``backend_name`` and ``engine_factory`` into a fresh factory.

    Returning a function from a function eliminates loop-variable late
    binding: even if the call is delayed (future async/parallel use), the
    captured values stay stable. The benchmark runner currently calls the
    factory synchronously per item, but this future-proofs the harness.
    """

    def factory() -> tuple[StreamTranscriber, Any]:
        engine = engine_factory()
        return build_pipeline(backend_name, engine=engine)

    return factory


class NonSpeechFilterBenchmarkRunner:
    """Drive the multi-layered defense across backends × engines × corpora."""

    def __init__(self, config: NonSpeechFilterBenchmarkConfig) -> None:
        self.config = config
        self._engine_manager: Any | None = None  # lazy-imported when needed

    def execute(self) -> NonSpeechFilterReport:
        backends = self.config.normalised_backends()
        engines = list(self.config.engines) or [DEFAULT_MOCK_ENGINE_NAME]

        corpora: dict[str, list[CorpusItem]] = {
            "synthetic": build_synthetic_corpus(),
        }
        if self.config.corpus_dir is not None:
            try:
                corpora["real"] = load_real_corpus_items(self.config.corpus_dir)
            except Exception as exc:  # pragma: no cover - reported as skip
                logger.warning("Real corpus load failed: %s", exc)

        report = new_report(
            mode=self.config.mode,
            device=self.config.device,
            runs=self.config.runs,
        )

        for backend_name in backends:
            try:
                _ = create_backend(backend_name)  # availability probe
            except ImportError as exc:
                report.add_skip(f"backend {backend_name} unavailable: {exc}")
                logger.warning(
                    "Skipping backend %s (ImportError: %s)", backend_name, exc
                )
                continue

            for engine_id in engines:
                engine_factory = self._build_engine_factory(engine_id)
                if engine_factory is None:
                    report.add_skip(f"engine {engine_id} unavailable")
                    continue

                for corpus_name, items in corpora.items():
                    for run_index in range(max(1, self.config.runs)):
                        factory = _make_pipeline_factory(
                            backend_name, engine_factory
                        )
                        evaluation = evaluate_pipeline(
                            factory,
                            items,
                            backend_name=backend_name,
                            measure_hallucination=(
                                engine_id != DEFAULT_MOCK_ENGINE_NAME
                            ),
                        )
                        report.add_record(
                            NonSpeechFilterRunRecord(
                                backend=backend_name,
                                engine=engine_id,
                                corpus=corpus_name,
                                run_index=run_index,
                                negative_total=evaluation.negative_total,
                                positive_total=evaluation.positive_total,
                                short_total=evaluation.short_total,
                                false_asr_trigger_rate=evaluation.false_asr_trigger_rate,
                                speech_recall=evaluation.speech_recall,
                                short_utterance_recall=evaluation.short_utterance_recall,
                                non_empty_hallucination_rate=evaluation.non_empty_hallucination_rate,
                                added_latency_p50_ms=evaluation.added_latency_p50_ms,
                                added_latency_p95_ms=evaluation.added_latency_p95_ms,
                                per_label=evaluation.per_label,
                            )
                        )

        return report

    # ---- Engine factory -----------------------------------------------------

    def _build_engine_factory(self, engine_id: str):
        """Return a zero-arg factory producing engine instances, or None."""
        if engine_id == DEFAULT_MOCK_ENGINE_NAME:
            return MockEngine
        return self._real_engine_factory(engine_id)

    def _real_engine_factory(self, engine_id: str):
        """Use ``BenchmarkEngineManager`` for cached real-engine instances."""
        if self._engine_manager is None:
            from benchmarks.common.engines import BenchmarkEngineManager

            self._engine_manager = BenchmarkEngineManager()
        manager = self._engine_manager
        device = self.config.device

        def factory() -> Any:
            return manager.get_engine(engine_id=engine_id, device=device)

        return factory

    # ---- Report I/O ---------------------------------------------------------

    def write_reports(self, report: NonSpeechFilterReport) -> tuple[Path, Path]:
        """Save JSON + Markdown reports under ``config.output_dir``.

        Returns the (json_path, markdown_path) pair so the CLI can echo them.
        """
        out = self.config.output_dir
        out.mkdir(parents=True, exist_ok=True)
        stamp = report.timestamp.replace(":", "-").replace(".", "-")
        json_path = out / f"non_speech_filter_{stamp}.json"
        md_path = out / f"non_speech_filter_{stamp}.md"
        report.save_json(json_path)
        report.save_markdown(md_path)
        return json_path, md_path
