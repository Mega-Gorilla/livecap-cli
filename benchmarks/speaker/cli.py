"""CLI for the speaker-embedding benchmark.

Usage:
    python -m benchmarks.speaker --backend titanet ecapa pyannote --device cuda
    python -m benchmarks.speaker --list-backends
    python -m benchmarks.speaker --backend titanet --input path/to/local.wav
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .factory import SPEAKER_REGISTRY, get_all_backend_ids
from .runner import SpeakerBenchmarkConfig, SpeakerBenchmarkRunner

# Default source produced by scripts/prepare_speaker_benchmark.py.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_INPUT = _PROJECT_ROOT / "benchmarks" / "speaker" / "data" / "source_10min.wav"
_DEFAULT_TARGET = _PROJECT_ROOT / "benchmarks" / "speaker" / "data" / "target_enroll.wav"


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    available = ", ".join(get_all_backend_ids())
    parser = argparse.ArgumentParser(
        description="Speaker Embedding Benchmark - GPU/memory/latency/separability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available backends: {available}",
    )
    parser.add_argument(
        "--backend",
        "-b",
        nargs="+",
        default=["titanet"],
        help="Backends to benchmark (default: titanet).",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device for embedding extraction. Default: cuda",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help=f"Source audio. Default: {_DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--target-enroll",
        type=Path,
        default=None,
        help="Optional target-speaker enrollment wav (else larger-cluster centroid).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Output directory. Default: benchmark_results/",
    )
    parser.add_argument(
        "--min-segment-s",
        type=float,
        default=0.3,
        help="Drop VAD segments shorter than this (seconds). Default: 0.3",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Cap number of segments (for quick runs).",
    )
    parser.add_argument(
        "--coresidency",
        action="store_true",
        help="Also measure ASR + backend combined GPU footprint.",
    )
    parser.add_argument(
        "--coresidency-engine",
        default="parakeet_ja",
        help="ASR engine to co-load for the co-residency measurement. Default: parakeet_ja",
    )
    parser.add_argument(
        "--asr-engine",
        default="parakeet_ja",
        help="ASR engine for per-segment transcripts (default: parakeet_ja; "
        "e.g. reazonspeech). Use --no-asr to disable.",
    )
    parser.add_argument(
        "--no-asr",
        dest="asr",
        action="store_false",
        help="Disable per-segment ASR transcript export.",
    )
    parser.add_argument(
        "--language",
        default="ja",
        help="Language for ASR transcripts. Default: ja",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Compute FAR/FRR/EER threshold calibration for a target-speaker gate.",
    )
    parser.add_argument(
        "--label-source",
        choices=["self", "gold", "silver"],
        default="self",
        help="Calibration labels: self=KMeans (optimistic), gold/silver=--labels-file.",
    )
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=None,
        help="JSON {\"labels\": {idx: speaker}} for gold/silver calibration labels.",
    )
    parser.add_argument(
        "--no-isolate",
        dest="isolate",
        action="store_false",
        help="Run all backends in one process (default: isolate each in a subprocess).",
    )
    parser.add_argument(
        "--worker-out",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,  # internal: single-process worker writes result JSON here
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="List available backends and exit.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable DEBUG logging."
    )
    parsed = parser.parse_args(args)

    # External-label calibration must not silently degrade to the optimistic
    # self (KMeans) proxy — require an explicit labels file.
    if parsed.calibrate and parsed.label_source in ("gold", "silver") and not parsed.labels_file:
        parser.error(
            f"--labels-file is required when --label-source is {parsed.label_source}"
        )
    return parsed


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)

    if parsed.list_backends:
        print("Available backends:")
        for bid, spec in SPEAKER_REGISTRY.items():
            print(f"  - {bid:10s} license={spec['license']} extra={spec['extra']}")
        return 0

    logging.basicConfig(
        level=logging.DEBUG if parsed.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    available = set(get_all_backend_ids())
    for bid in parsed.backend:
        if bid not in available:
            logger.error("Unknown backend: %s (available: %s)", bid, ", ".join(sorted(available)))
            return 1

    input_path = parsed.input or _DEFAULT_INPUT
    target_path = parsed.target_enroll
    if target_path is None and _DEFAULT_TARGET.exists():
        target_path = _DEFAULT_TARGET

    # Worker mode (spawned per-backend by an isolated parent run): benchmark
    # in-process and emit the result JSON, no further subprocessing.
    is_worker = parsed.worker_out is not None

    config = SpeakerBenchmarkConfig(
        backends=parsed.backend,
        device=parsed.device,
        input_path=input_path,
        target_enroll_path=target_path,
        output_dir=parsed.output_dir,
        min_segment_s=parsed.min_segment_s,
        max_segments=parsed.max_segments,
        coresidency=parsed.coresidency,
        coresidency_engine=parsed.coresidency_engine,
        isolate=parsed.isolate and not is_worker,
        asr_engine=(parsed.asr_engine if parsed.asr else None),
        language=parsed.language,
        calibrate=parsed.calibrate,
        label_source=parsed.label_source,
        labels_file=parsed.labels_file,
    )

    logger.info("=" * 60)
    logger.info("Speaker Embedding Benchmark")
    logger.info("Backends: %s | Device: %s", config.backends, config.device)
    logger.info("Input: %s", config.input_path)
    logger.info("=" * 60)

    try:
        runner = SpeakerBenchmarkRunner(config)
        output_dir = runner.run()
        logger.info("Results saved to: %s", output_dir)
        if is_worker:
            import json

            parsed.worker_out.write_text(
                json.dumps(
                    {
                        "results": [r.to_dict() for r in runner.reporter.results],
                        "detail": runner._detail,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        return 0
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception as e:
        logger.error("Benchmark failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
