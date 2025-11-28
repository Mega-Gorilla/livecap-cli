"""CLI for VAD benchmark.

Usage:
    python -m benchmarks.vad --mode quick
    python -m benchmarks.vad --engine parakeet_ja --vad silero webrtc_mode3 --language ja
    python -m benchmarks.vad --mode standard --runs 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .runner import VADBenchmarkConfig, VADBenchmarkRunner
from .factory import get_all_vad_ids


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI output.

    Args:
        verbose: If True, enable DEBUG level logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        args: Command-line arguments (defaults to sys.argv)

    Returns:
        Parsed arguments
    """
    available_vads = ", ".join(get_all_vad_ids())

    parser = argparse.ArgumentParser(
        description="VAD Benchmark Runner - Evaluates VAD + ASR pipeline performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available VADs:
  {available_vads}

Examples:
  # Quick benchmark (default engines/VADs, small dataset)
  python -m benchmarks.vad --mode quick

  # Benchmark specific VADs with a specific engine
  python -m benchmarks.vad --engine parakeet_ja --vad silero webrtc_mode3 --language ja

  # Full benchmark with multiple runs
  python -m benchmarks.vad --mode full --runs 3

  # Custom output directory
  python -m benchmarks.vad --mode standard --output-dir ./my_results
        """,
    )

    # Mode and scope
    parser.add_argument(
        "--mode",
        choices=["quick", "standard", "full"],
        default="quick",
        help="Execution mode: quick (small dataset, default engines/VADs), "
        "standard (100 files/lang), full (all files, all VADs). Default: quick",
    )
    parser.add_argument(
        "--language", "-l",
        nargs="+",
        default=["ja", "en"],
        help="Languages to benchmark. Default: ja en",
    )
    parser.add_argument(
        "--engine", "-e",
        nargs="+",
        default=None,
        help="Specific ASR engines to benchmark. If not specified, uses mode defaults.",
    )
    parser.add_argument(
        "--vad",
        nargs="+",
        default=None,
        help="Specific VADs to benchmark. If not specified, uses mode defaults.",
    )

    # Measurement options
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=1,
        help="Number of runs per file for RTF measurement. Default: 1",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use for ASR. Default: cuda",
    )

    # Output options
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory for results. Default: benchmark_results/",
    )

    # Listing
    parser.add_argument(
        "--list-vads",
        action="store_true",
        help="List all available VADs and exit",
    )

    # Logging
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    """Main entry point for CLI.

    Args:
        args: Command-line arguments

    Returns:
        Exit code (0 for success)
    """
    parsed = parse_args(args)

    # Handle --list-vads
    if parsed.list_vads:
        print("Available VADs:")
        for vad_id in get_all_vad_ids():
            print(f"  - {vad_id}")
        return 0

    setup_logging(parsed.verbose)

    logger = logging.getLogger(__name__)

    # Validate VAD names
    if parsed.vad:
        available = set(get_all_vad_ids())
        for vad_id in parsed.vad:
            if vad_id not in available:
                logger.error(f"Unknown VAD: {vad_id}")
                logger.error(f"Available VADs: {', '.join(sorted(available))}")
                return 1

    # Build configuration
    config = VADBenchmarkConfig(
        mode=parsed.mode,
        languages=parsed.language,
        engines=parsed.engine,
        vads=parsed.vad,
        runs=parsed.runs,
        device=parsed.device,
        output_dir=parsed.output_dir,
    )

    logger.info("=" * 60)
    logger.info("VAD Benchmark")
    logger.info("=" * 60)
    logger.info(f"Mode: {config.mode}")
    logger.info(f"Languages: {config.languages}")
    logger.info(f"Engines: {config.engines or 'mode defaults'}")
    logger.info(f"VADs: {config.vads or 'mode defaults'}")
    logger.info(f"Runs: {config.runs}")
    logger.info(f"Device: {config.device}")
    logger.info("=" * 60)

    # Run benchmark
    try:
        runner = VADBenchmarkRunner(config)
        output_dir = runner.run()
        logger.info("=" * 60)
        logger.info(f"Results saved to: {output_dir}")
        logger.info("=" * 60)
        return 0
    except KeyboardInterrupt:
        logger.warning("Benchmark interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
