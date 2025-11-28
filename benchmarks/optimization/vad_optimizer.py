"""VAD Parameter Optimizer using Bayesian Optimization (Optuna).

Provides the main optimization interface for finding optimal VAD parameters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

import optuna
from optuna.samplers import TPESampler

from benchmarks.common.datasets import AudioFile, DatasetManager
from engines.engine_factory import EngineFactory

from .objective import VADObjective

if TYPE_CHECKING:
    from engines.base_engine import TranscriptionEngine

logger = logging.getLogger(__name__)


# Default optimization results directory
DEFAULT_OUTPUT_DIR = Path("benchmark_results/optimization")


@dataclass
class OptimizationResult:
    """Container for optimization results.

    Attributes:
        vad_type: VAD type that was optimized
        language: Language code
        best_params: Best parameter combination found
        best_score: Best score achieved (CER or WER)
        n_trials: Number of trials completed
        duration_s: Total optimization duration in seconds
        study_name: Name of the Optuna study
        created_at: Timestamp of optimization completion
    """

    vad_type: str
    language: str
    best_params: dict[str, Any]
    best_score: float
    n_trials: int
    duration_s: float = 0.0
    study_name: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "vad_type": self.vad_type,
            "language": self.language,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": self.n_trials,
            "duration_s": self.duration_s,
            "study_name": self.study_name,
            "created_at": self.created_at,
        }


class VADOptimizer:
    """VAD Parameter Optimizer using Bayesian Optimization.

    Uses Optuna with TPE (Tree-structured Parzen Estimator) sampler
    to find optimal VAD parameters that minimize CER (Japanese) or WER (English).

    Args:
        vad_type: VAD type to optimize ("silero", "tenvad", "webrtc")
        language: Language code ("ja" or "en")
        engine_id: ASR engine ID (default: auto-select based on language)
        device: Device for ASR ("cuda" or "cpu")
        mode: Dataset mode ("quick", "standard", "full")

    Example:
        optimizer = VADOptimizer(vad_type="silero", language="ja")
        result = optimizer.optimize(n_trials=50)
        print(f"Best score: {result.best_score}")
        print(f"Best params: {result.best_params}")
    """

    # Default ASR engines per language
    DEFAULT_ENGINES = {
        "ja": "parakeet_ja",
        "en": "parakeet",
    }

    def __init__(
        self,
        vad_type: str,
        language: str,
        engine_id: str | None = None,
        device: str = "cuda",
        mode: str = "quick",
    ) -> None:
        self.vad_type = vad_type
        self.language = language
        self.device = device
        self.mode = mode

        # Select engine
        self.engine_id = engine_id or self.DEFAULT_ENGINES.get(language, "parakeet")

        # Engine and dataset are loaded lazily
        self._engine: TranscriptionEngine | None = None
        self._dataset: list[AudioFile] | None = None

    @property
    def engine(self) -> "TranscriptionEngine":
        """Get ASR engine (lazy loaded)."""
        if self._engine is None:
            self._engine = self._load_engine()
        return self._engine

    @property
    def dataset(self) -> list[AudioFile]:
        """Get dataset (lazy loaded)."""
        if self._dataset is None:
            self._dataset = self._load_dataset()
        return self._dataset

    def _load_engine(self) -> "TranscriptionEngine":
        """Load and initialize ASR engine."""
        logger.info(f"Loading engine: {self.engine_id} on {self.device}")

        engine = EngineFactory.create_engine(
            self.engine_id,
            device=self.device,
        )
        engine.load_model()

        logger.info(f"Engine {self.engine_id} loaded successfully")
        return engine

    def _load_dataset(self) -> list[AudioFile]:
        """Load dataset for optimization."""
        logger.info(f"Loading dataset: {self.language} ({self.mode} mode)")

        manager = DatasetManager()
        dataset = manager.get_dataset(self.language, mode=self.mode)

        logger.info(f"Loaded {len(dataset)} files for {self.language}")
        return list(dataset.files)

    def optimize(
        self,
        n_trials: int = 50,
        seed: int = 42,
        storage: str | None = None,
        output_dir: Path | None = None,
    ) -> OptimizationResult:
        """Run Bayesian optimization to find optimal VAD parameters.

        Args:
            n_trials: Number of optimization trials
            seed: Random seed for reproducibility
            storage: Optuna storage URL (default: SQLite in output_dir)
            output_dir: Output directory for results (default: benchmark_results/optimization)

        Returns:
            OptimizationResult with best parameters and score
        """
        import time

        # Setup output directory
        output_dir = output_dir or DEFAULT_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup Optuna storage
        if storage is None:
            db_path = output_dir / "studies.db"
            storage = f"sqlite:///{db_path}"

        # Create study
        study_name = f"{self.vad_type}_{self.language}"
        sampler = TPESampler(seed=seed)

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            storage=storage,
            study_name=study_name,
            load_if_exists=True,
        )

        # Create objective function
        objective = VADObjective(
            vad_type=self.vad_type,
            language=self.language,
            engine=self.engine,
            dataset=self.dataset,
        )

        # Run optimization
        logger.info(
            f"Starting optimization: {self.vad_type} x {self.language} "
            f"({n_trials} trials)"
        )
        start_time = time.time()

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        duration_s = time.time() - start_time

        # Create result
        result = OptimizationResult(
            vad_type=self.vad_type,
            language=self.language,
            best_params=study.best_params,
            best_score=study.best_value,
            n_trials=len(study.trials),
            duration_s=duration_s,
            study_name=study_name,
        )

        logger.info(
            f"Optimization complete: best_score={result.best_score:.4f}, "
            f"duration={duration_s:.1f}s"
        )

        return result

    def cleanup(self) -> None:
        """Release resources (engine, GPU memory)."""
        if self._engine is not None:
            logger.info("Releasing engine resources")
            del self._engine
            self._engine = None

            # Clear CUDA cache if available
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


__all__ = ["VADOptimizer", "OptimizationResult"]
