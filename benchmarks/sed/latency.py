"""5-axis runtime measurement (Issue #305 v3 Dimension 3).

The decision document reports these axes independently so PyTorch-native
vs TensorFlow-dependent candidates can be compared apples-to-apples:

- **Checkpoint size** — bytes on disk of the ``.pt`` file
- **Installed dependency delta** — bytes added vs the ``engines-torch`` baseline
  (PR-D0 expects 0 because EfficientAT is loaded from a manual clone, not pip)
- **Runtime peak memory** — bytes resident during a single inference
  (measured via :mod:`tracemalloc` — a conservative lower bound that
  excludes torch CUDA allocator overhead)
- **CPU p50 / p95 latency** — milliseconds per 1-second window inference
- **GPU p50 / p95 latency** — same, on CUDA when available
- **Cold start** — wall-clock for the first inference (model load + warmup)

The orchestrator writes one ``latency.csv`` row per device + axis so the
decision document can render a clean table.
"""

from __future__ import annotations

import os
import statistics
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from benchmarks.sed.inference import (
    LoadedSED,
    SED_WINDOW_SECONDS,
    compute_window_probs,
    resolve_device,
    load_model,
)


WARMUP_ITERATIONS = 3
"""How many warmup inferences to run before recording timing samples."""

DEFAULT_LATENCY_ITERATIONS = 100
"""How many samples feed each percentile measurement by default."""


@dataclass(frozen=True)
class LatencyMeasurement:
    """All five axes plus reproducibility metadata."""

    variant: str
    device: str
    checkpoint_path: str
    checkpoint_size_bytes: int
    installed_dependency_delta_bytes: int
    parameter_bytes: int
    runtime_peak_memory_bytes: int
    cold_start_seconds: float
    iterations: int
    cpu_p50_ms: float | None
    cpu_p95_ms: float | None
    gpu_p50_ms: float | None
    gpu_p95_ms: float | None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_csv_row(self) -> dict[str, str]:
        """Return a stringified row suited for ``csv.DictWriter``."""

        row = asdict(self)
        row["notes"] = " | ".join(self.notes)
        return {k: ("" if v is None else str(v)) for k, v in row.items()}

    @staticmethod
    def csv_fieldnames() -> tuple[str, ...]:
        return (
            "variant",
            "device",
            "checkpoint_path",
            "checkpoint_size_bytes",
            "installed_dependency_delta_bytes",
            "parameter_bytes",
            "runtime_peak_memory_bytes",
            "cold_start_seconds",
            "iterations",
            "cpu_p50_ms",
            "cpu_p95_ms",
            "gpu_p50_ms",
            "gpu_p95_ms",
            "notes",
        )


# ---------------------------------------------------------------------------
# Per-axis measurement primitives
# ---------------------------------------------------------------------------


def discover_checkpoint(efficientat_path: Path, variant: str) -> Path | None:
    """Locate the cached pretrained ``.pt`` for ``variant``.

    EfficientAT downloads checkpoints into ``<efficientat_path>/resources/``;
    file names embed the validation mAP, so we glob by prefix.
    """

    resources = efficientat_path / "resources"
    if not resources.is_dir():
        return None
    for candidate in sorted(resources.glob(f"{variant}_*.pt")):
        return candidate
    return None


def measure_parameter_bytes(model: object) -> int:
    """Sum ``parameter.numel() * element_size()`` over the model's parameters."""

    import torch  # type: ignore[import-not-found]

    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"model must be torch.nn.Module, got {type(model)!r}")
    total = 0
    for parameter in model.parameters():
        total += parameter.numel() * parameter.element_size()
    return int(total)


def _percentile_ms(samples: list[float], q: float) -> float:
    return float(np.percentile(samples, q) * 1000.0)


def _time_one_inference(bundle: LoadedSED, sample_16k: np.ndarray) -> float:
    start = time.perf_counter()
    compute_window_probs(sample_16k, bundle)
    return time.perf_counter() - start


def measure_latency(
    bundle: LoadedSED,
    sample_16k: np.ndarray,
    iterations: int = DEFAULT_LATENCY_ITERATIONS,
) -> tuple[float, float]:
    """Return ``(p50_ms, p95_ms)`` over ``iterations`` runs after warmup."""

    if iterations < 5:
        raise ValueError("iterations must be at least 5 for percentile stability")

    for _ in range(WARMUP_ITERATIONS):
        compute_window_probs(sample_16k, bundle)

    samples = [_time_one_inference(bundle, sample_16k) for _ in range(iterations)]
    return _percentile_ms(samples, 50), _percentile_ms(samples, 95)


def measure_runtime_peak_memory(
    bundle: LoadedSED, sample_16k: np.ndarray
) -> int:
    """Measure peak Python-side allocation during one inference.

    Uses :mod:`tracemalloc` (already in the standard library) so we do not
    add psutil as a dependency. Reported value is a conservative lower
    bound — it does not include CUDA allocator overhead, which the decision
    document should call out explicitly.
    """

    tracemalloc.start()
    try:
        compute_window_probs(sample_16k, bundle)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return int(peak)


def measure_cold_start(
    variant: str,
    efficientat_path: Path,
    device: str = "cpu",
    sample_16k: np.ndarray | None = None,
) -> float:
    """Wall-clock for ``load_model + first inference`` on ``device``.

    Distinct from :func:`measure_latency` which discards warmup samples;
    this captures the "first call after a cold process" cost that matters
    for one-shot CLI users.
    """

    if sample_16k is None:
        sample_16k = np.zeros(16_000, dtype=np.float32)

    start = time.perf_counter()
    bundle = load_model(variant, device=device, efficientat_path=efficientat_path)
    compute_window_probs(sample_16k, bundle)
    return time.perf_counter() - start


# ---------------------------------------------------------------------------
# Orchestration helper used by the pipeline runner (Phase G)
# ---------------------------------------------------------------------------


def measure_all_axes(
    variant: str,
    efficientat_path: Path,
    sample_16k: np.ndarray,
    iterations: int = DEFAULT_LATENCY_ITERATIONS,
    measure_gpu: bool = True,
) -> LatencyMeasurement:
    """Collect every axis required by Issue #305 v3 Dimension 3."""

    import torch  # type: ignore[import-not-found]

    notes: list[str] = []

    checkpoint_path = discover_checkpoint(efficientat_path, variant)
    if checkpoint_path is None:
        # The first load_model() call materialises the cache; the path
        # becomes available afterwards. Caller is expected to have loaded
        # the model at least once via load_model.
        notes.append(
            f"Checkpoint not yet cached for {variant}; size reported as 0."
        )
        checkpoint_size = 0
        checkpoint_repr = ""
    else:
        checkpoint_size = checkpoint_path.stat().st_size
        checkpoint_repr = str(checkpoint_path)

    # CPU measurements (always)
    cpu_bundle = load_model(variant, device="cpu", efficientat_path=efficientat_path)
    cpu_p50, cpu_p95 = measure_latency(cpu_bundle, sample_16k, iterations)
    runtime_peak = measure_runtime_peak_memory(cpu_bundle, sample_16k)
    param_bytes = measure_parameter_bytes(cpu_bundle.model)

    # Cold start uses a fresh load_model + inference call (still cheap).
    cold_start = measure_cold_start(variant, efficientat_path, device="cpu", sample_16k=sample_16k)

    # Optional GPU measurements.
    gpu_p50: float | None = None
    gpu_p95: float | None = None
    if measure_gpu and torch.cuda.is_available():
        gpu_bundle = load_model(
            variant, device="cuda", efficientat_path=efficientat_path
        )
        gpu_p50, gpu_p95 = measure_latency(gpu_bundle, sample_16k, iterations)
    elif measure_gpu:
        notes.append("CUDA unavailable; GPU latency not measured.")

    # PR-D0: EfficientAT is a manual clone, not a pip package, so the
    # installed-dependency delta vs the engines-torch baseline is 0.
    installed_delta_bytes = 0
    notes.append(
        "Installed dependency delta = 0 bytes: EfficientAT is loaded from a "
        "manual clone; PR-D1 will revisit when extras are added to pyproject.toml."
    )

    return LatencyMeasurement(
        variant=variant,
        device=resolve_device("auto"),
        checkpoint_path=checkpoint_repr,
        checkpoint_size_bytes=int(checkpoint_size),
        installed_dependency_delta_bytes=int(installed_delta_bytes),
        parameter_bytes=int(param_bytes),
        runtime_peak_memory_bytes=int(runtime_peak),
        cold_start_seconds=float(cold_start),
        iterations=int(iterations),
        cpu_p50_ms=float(cpu_p50),
        cpu_p95_ms=float(cpu_p95),
        gpu_p50_ms=None if gpu_p50 is None else float(gpu_p50),
        gpu_p95_ms=None if gpu_p95 is None else float(gpu_p95),
        notes=tuple(notes),
    )


def write_latency_csv(measurements: list[LatencyMeasurement], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = LatencyMeasurement.csv_fieldnames()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for measurement in measurements:
            writer.writerow(measurement.to_csv_row())


__all__ = [
    "WARMUP_ITERATIONS",
    "DEFAULT_LATENCY_ITERATIONS",
    "LatencyMeasurement",
    "discover_checkpoint",
    "measure_parameter_bytes",
    "measure_latency",
    "measure_runtime_peak_memory",
    "measure_cold_start",
    "measure_all_axes",
    "write_latency_csv",
]


# statistics is imported even though we use numpy percentiles; keep it
# available for callers that prefer stdlib quantiles.
_ = statistics  # noqa: F841 - intentional re-export hook
_ = os  # noqa: F841 - reserved for future env-driven config
