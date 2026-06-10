"""Argparse + orchestration for ``python -m benchmarks.sed``.

Filled in by Phase G after inference / metrics / latency modules land.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmarks.sed",
        description="Phase 2 SED off-line evaluation (Issue #305 PR-D0)",
    )
    parser.add_argument(
        "--model",
        default="mn04_as",
        choices=("mn04_as", "dymn04_as", "dymn10_as"),
        help="EfficientAT variant to evaluate (default: mn04_as = smallest viable, 3.88 MB).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/sed/2026-06-10"),
        help="Where to write probabilities.csv / latency.csv / metadata.json.",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Override LIVECAP_NON_SPEECH_CORPUS_DIR for this run.",
    )
    parser.add_argument(
        "--efficientat-path",
        type=Path,
        default=None,
        help="Override LIVECAP_SED_EFFICIENTAT_PATH for this run.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Inference device (auto = cuda if available else cpu).",
    )
    parser.add_argument(
        "--latency-iters",
        type=int,
        default=100,
        help="Number of iterations for CPU/GPU latency percentile measurement.",
    )
    parser.add_argument(
        "--skip-latency",
        action="store_true",
        help="Skip latency measurement (probabilities only).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Imported lazily so --help works even when EfficientAT isn't cloned yet.
    from benchmarks.sed.orchestrator import run_evaluation

    return run_evaluation(args)
