"""Threshold sweep harness for the Layer 1 transient detector (PR-B).

Iterates over a small grid of named presets (and, optionally, a 1-feature
sensitivity sweep around the ``moderate`` defaults) and produces CSV +
Markdown summaries that PR reviewers can use to pick the production
default. Each grid cell is one
:class:`NonSpeechFilterBenchmarkRunner.execute()` call so the sweep
inherits all the engine wiring already in place — including
``InstrumentedEngine`` and ``fail_fast=False`` error capture.

Run it as a module so ``benchmarks`` resolves cleanly:

    python -m benchmarks.non_speech_filter.sweep \
        --backend silero,tenvad,webrtc --engine mock \
        --corpus-dir .tmp/non_speech_corpus

For real-engine sweeps add ``--engine whispers2t,parakeet_ja,reazonspeech
--device cuda``.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from livecap_cli.audio.transient_detector import TransientDetectorConfig

from .cli import _parse_csv
from .runner import (
    SUPPORTED_BACKENDS,
    NonSpeechFilterBenchmarkConfig,
    NonSpeechFilterBenchmarkRunner,
)

logger = logging.getLogger(__name__)


# ---------- Preset grid --------------------------------------------------


@dataclass(frozen=True)
class NamedPreset:
    """One labelled threshold combination for the sweep."""

    name: str
    config: TransientDetectorConfig


def default_named_presets() -> list[NamedPreset]:
    """Five coarse points plus three hypothesis-driven candidates.

    The first five (``baseline_off`` through ``on_aggressive``) bracket
    sensible tuning ranges and shipped with PR-B (#300). The last three
    were added by the PR-B calibration follow-up to probe specific
    failures observed in the private real corpus:

    - ``on_relaxed_rms``: real-corpus clips sit at -41 to -46 dBFS RMS,
      so the default ``rms_min_db=-35`` rejects > 95 % of frames before
      the AND combination ever fires. This preset drops the floor to
      -45 dBFS while keeping the rest at moderate defaults.
    - ``on_low_freq_aware``: ``desk_tap`` has a centroid below 1 kHz on
      every frame, so the default ``centroid_min_hz=2500`` is a hard
      blocker. This preset widens the centroid window and tightens
      ``voiced_max`` to compensate (keep speech off the applause side).
    - ``on_speech_safe``: maximum safety against recall regression —
      only fires on textbook rapid-burst applause. Useful as a "ceiling"
      datapoint to confirm short-utterance recall stays at 100 %.
    """
    return [
        NamedPreset(
            "baseline_off",
            TransientDetectorConfig(mode="off"),
        ),
        NamedPreset(
            "observe_defaults",
            TransientDetectorConfig(mode="observe"),
        ),
        NamedPreset(
            "on_conservative",
            TransientDetectorConfig(
                mode="on",
                flatness_min=0.40,
                centroid_min_hz=3500.0,
                zcr_min=0.18,
                onset_ratio=5.0,
                voiced_max=0.15,
                rms_min_db=-30.0,
            ),
        ),
        NamedPreset(
            "on_moderate",
            TransientDetectorConfig(mode="on"),  # PR-B v3 defaults
        ),
        NamedPreset(
            "on_aggressive",
            TransientDetectorConfig(
                mode="on",
                flatness_min=0.20,
                centroid_min_hz=2000.0,
                zcr_min=0.08,
                onset_ratio=2.0,
                voiced_max=0.35,
                rms_min_db=-40.0,
            ),
        ),
        # ---- PR-B calibration follow-up additions -----------------------
        NamedPreset(
            "on_relaxed_rms",
            TransientDetectorConfig(
                mode="on",
                # Drop the RMS floor so quiet real recordings reach the
                # AND combination. Other thresholds stay at moderate.
                rms_min_db=-45.0,
            ),
        ),
        NamedPreset(
            "on_low_freq_aware",
            TransientDetectorConfig(
                mode="on",
                # Widen the centroid window so low-frequency thumps
                # (e.g. desk_tap) can pass the spectral condition.
                centroid_min_hz=500.0,
                # Tighten voiced_max to keep low-pitched speech on the
                # speech side of the decision.
                voiced_max=0.15,
            ),
        ),
        NamedPreset(
            "on_speech_safe",
            TransientDetectorConfig(
                mode="on",
                # Stricter than on_conservative on every axis — useful as
                # an upper bound when recall regression is observed
                # elsewhere in the sweep.
                flatness_min=0.45,
                centroid_min_hz=3000.0,
                onset_ratio=5.0,
            ),
        ),
    ]


# ---------- Sweep ---------------------------------------------------------


@dataclass
class SweepCellResult:
    """One row of the sweep table."""

    preset: str
    backend: str
    engine: str
    corpus: str
    mode: str
    false_asr_trigger_rate: float | None
    speech_recall: float | None
    short_utterance_recall: float | None
    non_empty_hallucination_rate: float | None
    added_latency_p50_ms: float
    added_latency_p95_ms: float
    config_summary: str


@dataclass
class SweepReport:
    """Aggregated sweep output."""

    timestamp: str
    device: str
    backends: list[str]
    engines: list[str]
    corpus_dir: Optional[Path]
    cells: list[SweepCellResult] = field(default_factory=list)

    # ---- I/O ------------------------------------------------------------

    def to_csv(self) -> str:
        lines: list[str] = []
        fieldnames = [
            "preset",
            "backend",
            "engine",
            "corpus",
            "mode",
            "false_asr_trigger_rate",
            "speech_recall",
            "short_utterance_recall",
            "non_empty_hallucination_rate",
            "added_latency_p50_ms",
            "added_latency_p95_ms",
            "config_summary",
        ]
        import io

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for cell in self.cells:
            row = dataclasses.asdict(cell)
            writer.writerow(row)
        return buf.getvalue()

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append("# Transient Detector Threshold Sweep")
        lines.append("")
        lines.append(f"- **Date:** {self.timestamp}")
        lines.append(f"- **Device:** {self.device}")
        lines.append(f"- **Backends:** {', '.join(self.backends)}")
        lines.append(f"- **Engines:** {', '.join(self.engines)}")
        lines.append(
            f"- **Corpus dir:** {self.corpus_dir if self.corpus_dir else '(synthetic only)'}"
        )
        lines.append("")

        if not self.cells:
            lines.append("No cells recorded.")
            return "\n".join(lines)

        # Group by (backend, engine, corpus) for table view.
        lines.append("## Summary by Backend × Engine × Corpus")
        lines.append("")
        headers = (
            "Preset",
            "Backend",
            "Engine",
            "Corpus",
            "Mode",
            "False Trigger",
            "Speech Recall",
            "Short Recall",
            "Hallucination",
            "P50 ms",
            "P95 ms",
        )
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for cell in sorted(
            self.cells, key=lambda c: (c.backend, c.engine, c.corpus, c.preset)
        ):
            fmt_pct = lambda v: f"{v:.1%}" if v is not None else "-"
            lines.append(
                "| "
                + " | ".join(
                    [
                        cell.preset,
                        cell.backend,
                        cell.engine,
                        cell.corpus,
                        cell.mode,
                        fmt_pct(cell.false_asr_trigger_rate),
                        fmt_pct(cell.speech_recall),
                        fmt_pct(cell.short_utterance_recall),
                        fmt_pct(cell.non_empty_hallucination_rate),
                        f"{cell.added_latency_p50_ms:.1f}",
                        f"{cell.added_latency_p95_ms:.1f}",
                    ]
                )
                + " |"
            )
        lines.append("")
        return "\n".join(lines)

    def save(self, output_dir: Path) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = self.timestamp.replace(":", "-").replace(".", "-")
        csv_path = output_dir / f"transient_sweep_{stamp}.csv"
        md_path = output_dir / f"transient_sweep_{stamp}.md"
        csv_path.write_text(self.to_csv(), encoding="utf-8")
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        return csv_path, md_path


def _summarise_config(cfg: TransientDetectorConfig) -> str:
    return (
        f"mode={cfg.mode} flatness>{cfg.flatness_min} centroid>{cfg.centroid_min_hz:.0f}Hz "
        f"zcr>{cfg.zcr_min} onset>{cfg.onset_ratio}x voiced<{cfg.voiced_max} "
        f"rms>{cfg.rms_min_db}dBFS"
    )


def run_sweep(
    *,
    backends: list[str],
    engines: list[str],
    corpus_dir: Optional[Path],
    device: str = "auto",
    presets: list[NamedPreset] | None = None,
) -> SweepReport:
    """Execute the sweep, one preset at a time."""
    presets = presets or default_named_presets()
    report = SweepReport(
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        device=device,
        backends=list(backends),
        engines=list(engines),
        corpus_dir=corpus_dir,
    )

    for preset in presets:
        cfg = preset.config
        # The 'off' preset is the no-detector baseline reference.
        transient_config = None if cfg.mode == "off" else cfg

        bench_config = NonSpeechFilterBenchmarkConfig(
            mode="quick",
            backends=backends,
            engines=engines,
            corpus_dir=corpus_dir,
            runs=1,
            device=device,
            transient_config=transient_config,
        )
        runner = NonSpeechFilterBenchmarkRunner(bench_config)
        run_report = runner.execute()

        for rec in run_report.records:
            report.cells.append(
                SweepCellResult(
                    preset=preset.name,
                    backend=rec.backend,
                    engine=rec.engine,
                    corpus=rec.corpus,
                    mode=cfg.mode,
                    false_asr_trigger_rate=rec.false_asr_trigger_rate,
                    speech_recall=rec.speech_recall,
                    short_utterance_recall=rec.short_utterance_recall,
                    non_empty_hallucination_rate=rec.non_empty_hallucination_rate,
                    added_latency_p50_ms=rec.added_latency_p50_ms,
                    added_latency_p95_ms=rec.added_latency_p95_ms,
                    config_summary=_summarise_config(cfg),
                )
            )

    return report


# ---------- CLI ----------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.non_speech_filter.sweep",
        description=(
            "Layer 1 transient detector threshold sweep (Issue #295 PR-B). "
            "Runs the benchmark CLI for each named preset and aggregates "
            "the results into a single CSV + Markdown report."
        ),
    )
    parser.add_argument(
        "--backend",
        type=_parse_csv,
        default=list(SUPPORTED_BACKENDS),
        help=f"Comma-separated VAD backends (default: all of {','.join(SUPPORTED_BACKENDS)}).",
    )
    parser.add_argument(
        "--engine",
        type=_parse_csv,
        default=[],
        help=(
            "Comma-separated engines (default: mock only). "
            "Real engines (whispers2t, parakeet_ja, reazonspeech) require GPU."
        ),
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Optional real-audio fixtures (manifest.json + WAVs).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device hint for real engines.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results") / "non_speech_filter" / "sweep",
        help="Where the sweep CSV + Markdown are written.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.verbose)

    report = run_sweep(
        backends=args.backend,
        engines=args.engine,
        corpus_dir=args.corpus_dir,
        device=args.device,
    )

    csv_path, md_path = report.save(args.output_dir)

    print(report.to_markdown())
    print()
    print(f"CSV report: {csv_path}")
    print(f"Markdown:   {md_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
