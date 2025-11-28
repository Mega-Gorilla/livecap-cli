"""CLI for ASR benchmark.

Usage:
    python -m benchmarks.asr --mode quick
    python -m benchmarks.asr --engine parakeet_ja whispers2t_large_v3 --language ja
    python -m benchmarks.asr --mode standard --runs 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .runner import ASRBenchmarkConfig, ASRBenchmarkRunner


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
    parser = argparse.ArgumentParser(
        description="ASR Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick benchmark (default engines, small dataset)
  python -m benchmarks.asr --mode quick

  # Standard benchmark with specific engines
  python -m benchmarks.asr --engine parakeet_ja whispers2t_large_v3 --language ja

  # Full benchmark with multiple runs for statistical accuracy
  python -m benchmarks.asr --mode full --runs 3

  # Custom output directory
  python -m benchmarks.asr --mode standard --output-dir ./my_results
        """,
    )

    # Mode and scope
    parser.add_argument(
        "--mode",
        choices=["debug", "quick", "standard", "full"],
        default="quick",
        help="Execution mode: debug (1 file/lang for CI), quick (30 files/lang), "
        "standard (100 files/lang), full (all files). Default: quick",
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
        help="Specific engines to benchmark. If not specified, uses mode defaults.",
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
        help="Device to use. Default: cuda",
    )

    # Output options
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory for results. Default: benchmark_results/",
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
    setup_logging(parsed.verbose)

    logger = logging.getLogger(__name__)

    # Build configuration
    config = ASRBenchmarkConfig(
        mode=parsed.mode,
        languages=parsed.language,
        engines=parsed.engine,
        runs=parsed.runs,
        device=parsed.device,
        output_dir=parsed.output_dir,
    )

    logger.info("=" * 60)
    logger.info("ASR Benchmark")
    logger.info("=" * 60)
    logger.info(f"Mode: {config.mode}")
    logger.info(f"Languages: {config.languages}")
    logger.info(f"Engines: {config.engines or 'mode defaults'}")
    logger.info(f"Runs: {config.runs}")
    logger.info(f"Device: {config.device}")
    logger.info("=" * 60)

    # Run benchmark
    try:
        runner = ASRBenchmarkRunner(config)
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
