"""CLI for the non-speech filter benchmark runner (Issue #295 PR-0).

Usage:

    python -m benchmarks.non_speech_filter --mode quick
    python -m benchmarks.non_speech_filter --backend silero,tenvad,webrtc
    python -m benchmarks.non_speech_filter --engine whispers2t --corpus-dir /path/to/real
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .runner import (
    SUPPORTED_BACKENDS,
    NonSpeechFilterBenchmarkConfig,
    NonSpeechFilterBenchmarkRunner,
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.non_speech_filter",
        description=(
            "Non-speech filter (Issue #295 PR-0) evaluation harness. "
            "Drives synthetic and optional real-audio corpora through the "
            "production pipeline across all supported VAD backends and "
            "(optionally) real ASR engines."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("quick", "standard"),
        default="quick",
        help="quick = 1 run synthetic; standard = N runs synthetic + real if available",
    )
    parser.add_argument(
        "--backend",
        type=_parse_csv,
        default=list(SUPPORTED_BACKENDS),
        help=(
            "Comma-separated VAD backends to evaluate "
            f"(choices: {','.join(SUPPORTED_BACKENDS)}; default: all)"
        ),
    )
    parser.add_argument(
        "--engine",
        type=_parse_csv,
        default=[],
        help=(
            "Comma-separated engine ids (default: mock). "
            "Real engine ids (e.g. 'whispers2t,parakeet_ja') trigger "
            "actual transcription and hallucination measurement."
        ),
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Path to real-audio fixtures (manifest.json + WAVs).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Repetitions per backend×engine×corpus cell (default 1).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device hint for real engines (mock ignores).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results") / "non_speech_filter",
        help="Where JSON + Markdown reports are written (default: benchmark_results/non_speech_filter/).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )

    # Layer 1: DSP transient detector (#295 PR-B).
    parser.add_argument(
        "--transient-filter",
        choices=("off", "observe", "on"),
        default="off",
        help=(
            "Layer 1 DSP transient detector mode "
            "(default: off, leaves the baseline pipeline unchanged). "
            "'observe' computes features + telemetry only. "
            "'on' zeroes out applause-flagged frames."
        ),
    )
    parser.add_argument(
        "--transient-flatness-min", type=float, default=0.30,
        help="Spectral flatness lower bound (default 0.30).",
    )
    parser.add_argument(
        "--transient-centroid-min-hz", type=float, default=2500.0,
        help="Spectral centroid lower bound in Hz (default 2500).",
    )
    parser.add_argument(
        "--transient-zcr-min", type=float, default=0.12,
        help="Zero-crossing rate lower bound (default 0.12).",
    )
    parser.add_argument(
        "--transient-onset-ratio", type=float, default=3.0,
        help="Onset / baseline multiplier (default 3.0).",
    )
    parser.add_argument(
        "--transient-voiced-max", type=float, default=0.25,
        help="Voiced confidence upper bound (default 0.25).",
    )
    parser.add_argument(
        "--transient-rms-min-db", type=float, default=-35.0,
        help="RMS dBFS lower bound (default -35).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.verbose)

    # Standard mode bumps runs to 3 by default; users can still override.
    runs = args.runs
    if args.mode == "standard" and args.runs <= 1:
        runs = 3

    transient_config = None
    if args.transient_filter != "off":
        from livecap_cli.audio.transient_detector import TransientDetectorConfig

        transient_config = TransientDetectorConfig(
            mode=args.transient_filter,
            flatness_min=args.transient_flatness_min,
            centroid_min_hz=args.transient_centroid_min_hz,
            zcr_min=args.transient_zcr_min,
            onset_ratio=args.transient_onset_ratio,
            voiced_max=args.transient_voiced_max,
            rms_min_db=args.transient_rms_min_db,
        )

    config = NonSpeechFilterBenchmarkConfig(
        mode=args.mode,
        backends=args.backend,
        engines=args.engine,
        corpus_dir=args.corpus_dir,
        runs=runs,
        device=args.device,
        output_dir=args.output_dir,
        transient_config=transient_config,
    )

    runner = NonSpeechFilterBenchmarkRunner(config)
    report = runner.execute()
    json_path, md_path = runner.write_reports(report)

    print(report.to_markdown())
    print()
    print(f"JSON report: {json_path}")
    print(f"Markdown:    {md_path}")

    if report.skipped:
        print()
        print("Skipped:")
        for s in report.skipped:
            print(f"  - {s}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
