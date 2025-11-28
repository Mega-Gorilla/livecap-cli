"""CLI for VAD parameter optimization.

Usage:
    python -m benchmarks.optimization --vad silero --language ja
    python -m benchmarks.optimization --vad tenvad --language en --n-trials 100
    python -m benchmarks.optimization --vad webrtc --language ja --engine whispers2t_large_v3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .param_spaces import get_supported_vad_types
from .vad_optimizer import VADOptimizer, DEFAULT_OUTPUT_DIR


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
    # Reduce Optuna noise
    if not verbose:
        logging.getLogger("optuna").setLevel(logging.WARNING)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        args: Command-line arguments (defaults to sys.argv)

    Returns:
        Parsed arguments
    """
    available_vads = ", ".join(get_supported_vad_types())

    parser = argparse.ArgumentParser(
        description="VAD Parameter Optimization using Bayesian Optimization (Optuna)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available VADs for optimization:
  {available_vads}

Note: JaVAD uses Grid Search and should be benchmarked separately.

Examples:
  # Optimize Silero for Japanese
  python -m benchmarks.optimization --vad silero --language ja

  # Optimize TenVAD for English with custom engine
  python -m benchmarks.optimization --vad tenvad --language en --engine whispers2t_large_v3

  # Increase trials for better convergence
  python -m benchmarks.optimization --vad webrtc --language ja --n-trials 100

  # Save results to JSON
  python -m benchmarks.optimization --vad silero --language ja --output results.json
        """,
    )

    # Required arguments
    parser.add_argument(
        "--vad",
        required=True,
        choices=get_supported_vad_types(),
        help="VAD type to optimize",
    )
    parser.add_argument(
        "--language", "-l",
        required=True,
        choices=["ja", "en"],
        help="Language to optimize for",
    )

    # Optional arguments
    parser.add_argument(
        "--engine", "-e",
        default=None,
        help="ASR engine ID (default: parakeet_ja for ja, parakeet for en)",
    )
    parser.add_argument(
        "--n-trials", "-n",
        type=int,
        default=50,
        help="Number of optimization trials (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device for ASR engine (default: cuda)",
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "standard"],
        default="quick",
        help="Dataset mode: quick (30 files) or standard (100 files). Default: quick",
    )

    # Output options
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output JSON file path for results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory for Optuna database (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL (overrides --output-dir)",
    )

    # Logging
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    return parser.parse_args(args)


def save_result(result, output_path: Path) -> None:
    """Save optimization result to JSON file.

    Args:
        result: OptimizationResult object
        output_path: Path to save JSON
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)


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

    logger.info("=" * 60)
    logger.info("VAD Parameter Optimization")
    logger.info("=" * 60)
    logger.info(f"VAD: {parsed.vad}")
    logger.info(f"Language: {parsed.language}")
    logger.info(f"Engine: {parsed.engine or 'auto'}")
    logger.info(f"Trials: {parsed.n_trials}")
    logger.info(f"Seed: {parsed.seed}")
    logger.info(f"Device: {parsed.device}")
    logger.info(f"Mode: {parsed.mode}")
    logger.info("=" * 60)

    try:
        # Create optimizer
        optimizer = VADOptimizer(
            vad_type=parsed.vad,
            language=parsed.language,
            engine_id=parsed.engine,
            device=parsed.device,
            mode=parsed.mode,
        )

        # Run optimization
        result = optimizer.optimize(
            n_trials=parsed.n_trials,
            seed=parsed.seed,
            storage=parsed.storage,
            output_dir=parsed.output_dir,
        )

        # Print results
        logger.info("=" * 60)
        logger.info("Optimization Complete")
        logger.info("=" * 60)
        logger.info(f"Best Score: {result.best_score:.4f}")
        logger.info(f"Duration: {result.duration_s:.1f}s")
        logger.info(f"Trials: {result.n_trials}")
        logger.info("")
        logger.info("Best Parameters:")
        for key, value in sorted(result.best_params.items()):
            logger.info(f"  {key}: {value}")

        # Save results if requested
        if parsed.output:
            save_result(result, parsed.output)
            logger.info("")
            logger.info(f"Results saved to: {parsed.output}")

        # Cleanup
        optimizer.cleanup()

        return 0

    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
