"""VAD Parameter Optimization module.

Provides Bayesian optimization for VAD parameters using Optuna.

Usage:
    python -m benchmarks.optimization --vad silero --language ja

Example:
    from benchmarks.optimization import VADOptimizer

    optimizer = VADOptimizer(vad_type="silero", language="ja")
    result = optimizer.optimize(n_trials=50)
    print(f"Best score: {result.best_score}")
    print(f"Best params: {result.best_params}")
"""

from __future__ import annotations

from .param_spaces import PARAM_SPACES, suggest_params
from .objective import VADObjective
from .vad_optimizer import VADOptimizer, OptimizationResult

__all__ = [
    "PARAM_SPACES",
    "suggest_params",
    "VADObjective",
    "VADOptimizer",
    "OptimizationResult",
]
