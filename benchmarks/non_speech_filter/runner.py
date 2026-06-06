"""Non-speech filter benchmark runner (Issue #295 PR-0).

Iterates over (backend × engine × corpus × run_index) and records baseline
metrics into a :class:`NonSpeechFilterReport`. The engine slot is either
``MockEngine`` (default) for fast filter-behavior measurement or a real
engine (e.g. ``whispers2t``) for hallucination text measurement.

This runner is intentionally separate from the pytest baseline tests so it
can drive real-audio corpora, real engines, and multi-run aggregation
without inflating CI cost.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from livecap_cli.audio.noise_gate import NoiseGate
from livecap_cli.transcription.stream import StreamTranscriber
from livecap_cli.vad.config import VADConfig
from livecap_cli.vad.processor import VADProcessor

# Reach into the test harness for the canonical corpus + metric definitions.
# This is a deliberate cross-cut: corpus.py / metrics.py are dependency-light
# (numpy + dataclass) and form the single source of truth for both pytest
# baseline tests and this ad-hoc benchmark runner.
from tests.integration.non_speech_filter.corpus import (
    CorpusItem,
    build_synthetic_corpus,
)
from tests.integration.non_speech_filter.metrics import (
    CorpusEvaluation,
    evaluate_pipeline,
)

from .report import NonSpeechFilterReport, NonSpeechFilterRunRecord, new_report

logger = logging.getLogger(__name__)


SUPPORTED_BACKENDS = ("silero", "tenvad", "webrtc")
DEFAULT_MOCK_ENGINE_NAME = "mock"


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
        device: Device hint passed to ``BenchmarkEngineManager``.
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


# ---------- Backend / pipeline construction (independent of pytest) ----------


def _make_backend(backend_type: str):
    """Construct a fresh VAD backend instance for ``backend_type``.

    Raises ``ImportError`` if the optional dependency is missing — callers
    catch this and record the skip in the report.
    """
    if backend_type == "silero":
        from livecap_cli.vad.backends.silero import SileroVAD

        return SileroVAD(onnx=True)
    if backend_type == "tenvad":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            from livecap_cli.vad.backends.tenvad import TenVAD

            return TenVAD()
    if backend_type == "webrtc":
        from livecap_cli.vad.backends.webrtc import WebRTCVAD

        return WebRTCVAD()
    raise ValueError(f"Unknown backend type: {backend_type!r}")


class _MockEngine:
    """Local MockEngine mirroring the conftest fixture (kept in-runner for independence)."""

    def __init__(self, return_text: str = "") -> None:
        self._return_text = return_text
        self.transcribe_count = 0
        self.last_texts: list[str] = []

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> tuple[str, float]:
        self.transcribe_count += 1
        self.last_texts.append(self._return_text)
        return self._return_text, 1.0

    def get_required_sample_rate(self) -> int:
        return 16000

    def get_engine_name(self) -> str:
        return DEFAULT_MOCK_ENGINE_NAME


def _build_pipeline(
    backend_type: str,
    *,
    engine: Any,
    vad_config: VADConfig | None = None,
    enable_noise_gate: bool = True,
    enable_energy_gate: bool = True,
    noise_gate_threshold_db: float = -50.0,
) -> tuple[StreamTranscriber, Any]:
    """Build a fresh baseline pipeline (NoiseGate + VAD + EnergyGate)."""
    backend = _make_backend(backend_type)
    vad_processor = VADProcessor(
        config=vad_config or VADConfig(),
        backend=backend,
    )
    noise_gate = (
        NoiseGate(threshold_db=noise_gate_threshold_db, sample_rate=16000)
        if enable_noise_gate
        else None
    )
    kwargs: dict[str, Any] = {
        "engine": engine,
        "vad_processor": vad_processor,
        "noise_gate": noise_gate,
    }
    if not enable_energy_gate:
        kwargs["engine_min_rms_dbfs"] = float("-inf")
    transcriber = StreamTranscriber(**kwargs)
    return transcriber, engine


# ---------- Real-corpus loader ------------------------------------------------


def _load_real_corpus(corpus_dir: Path) -> list[CorpusItem]:
    """Load real-audio fixtures from ``corpus_dir/manifest.json``.

    Same schema as the pytest harness; documented in
    ``docs/benchmarks/non-speech-filter.md``.
    """
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json missing in {corpus_dir}; see docs/benchmarks/non-speech-filter.md"
        )
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - environment dep
        raise ImportError("soundfile is required for real corpus loading") from exc
    items: list[CorpusItem] = []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest:
        audio_path = corpus_dir / entry["file"]
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            from scipy.signal import resample_poly

            g = gcd(int(sr), 16000)
            audio = resample_poly(audio, 16000 // g, int(sr) // g)
        items.append(
            CorpusItem(
                label=entry["label"],
                kind=entry["kind"],
                is_short_utterance=bool(entry.get("is_short_utterance", False)),
                audio=audio.astype(np.float32, copy=False),
            )
        )
    return items


# ---------- Runner -----------------------------------------------------------


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
                corpora["real"] = _load_real_corpus(self.config.corpus_dir)
            except Exception as exc:  # pragma: no cover - reported as skip
                logger.warning("Real corpus load failed: %s", exc)

        report = new_report(
            mode=self.config.mode,
            device=self.config.device,
            runs=self.config.runs,
        )

        for backend_name in backends:
            try:
                _ = _make_backend(backend_name)  # availability probe
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

                        def factory() -> tuple[StreamTranscriber, Any]:
                            engine = engine_factory()
                            return _build_pipeline(backend_name, engine=engine)

                        evaluation = evaluate_pipeline(
                            factory,
                            items,
                            backend_name=backend_name,
                            measure_hallucination=(engine_id != DEFAULT_MOCK_ENGINE_NAME),
                        )
                        report.add_record(
                            _evaluation_to_record(
                                evaluation,
                                backend=backend_name,
                                engine=engine_id,
                                corpus=corpus_name,
                                run_index=run_index,
                            )
                        )

        return report

    # ---- Engine factory -----------------------------------------------------

    def _build_engine_factory(self, engine_id: str):
        """Return a zero-arg factory producing engine instances, or None."""
        if engine_id == DEFAULT_MOCK_ENGINE_NAME:
            return _MockEngine
        return self._real_engine_factory(engine_id)

    def _real_engine_factory(self, engine_id: str):
        """Use ``BenchmarkEngineManager`` for cached real-engine instances."""
        if self._engine_manager is None:
            from benchmarks.common.engines import BenchmarkEngineManager

            self._engine_manager = BenchmarkEngineManager()
        manager = self._engine_manager

        def factory() -> Any:
            return manager.get_engine(
                engine_id=engine_id,
                device=self.config.device,
            )

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


def _evaluation_to_record(
    evaluation: CorpusEvaluation,
    *,
    backend: str,
    engine: str,
    corpus: str,
    run_index: int,
) -> NonSpeechFilterRunRecord:
    return NonSpeechFilterRunRecord(
        backend=backend,
        engine=engine,
        corpus=corpus,
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


def run_quick(backends: Iterable[str] | None = None) -> NonSpeechFilterReport:
    """Convenience helper used by ``__main__`` for the default smoke run."""
    cfg = NonSpeechFilterBenchmarkConfig(
        mode="quick",
        backends=list(backends or ["silero"]),
        runs=1,
    )
    return NonSpeechFilterBenchmarkRunner(cfg).execute()
