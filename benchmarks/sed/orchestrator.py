"""Full evaluation pipeline orchestration (Issue #305 PR-D0 Phase G).

Pulls together the four PR-D0 deliverable axes:

1. Corpus loading (reuses the PR-B harness via
   :func:`benchmarks.non_speech_filter.pipeline.load_real_corpus_items`)
2. EfficientAT inference over 1-second windows
3. Class-level + reject-signal-level metric scoring
4. 5-axis latency / memory measurement

Outputs (committed per Issue #305 v3 artifact policy):

- ``<out>/probabilities.csv`` — clip-window summary with target / speech-like
  per-class max + 3 policy scores
- ``<out>/probabilities_full.npz`` — raw per-window 527-vector tensors
- ``<out>/latency.csv`` — per-axis runtime measurement
- ``<out>/metadata.json`` — run metadata for reproducibility
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.sed.class_mapping import (
    POLICIES,
    SPEECH_LIKE_INDICES,
    TARGET_INDICES,
    load_class_names,
)
from benchmarks.sed.inference import (
    SED_WINDOW_SECONDS,
    compute_window_probs,
    load_model,
    resolve_device,
    resolve_efficientat_path,
)
from benchmarks.sed.latency import measure_all_axes, write_latency_csv
from benchmarks.sed.metrics import PerClipResult


def _resolve_corpus_dir(override: Path | None) -> Path:
    if override is not None:
        path = Path(override)
    else:
        env_value = os.environ.get("LIVECAP_NON_SPEECH_CORPUS_DIR")
        if not env_value:
            raise FileNotFoundError(
                "Set LIVECAP_NON_SPEECH_CORPUS_DIR or pass --corpus-dir; see "
                "benchmarks/sed/README.md."
            )
        path = Path(env_value)
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {path}")
    return path


def _git_commit(path: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _utc_timestamp() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _hardware_summary() -> dict[str, Any]:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        torch = None  # type: ignore[assignment]
    summary: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    if torch is not None:
        summary["torch"] = torch.__version__
        summary["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            summary["cuda_device_name"] = torch.cuda.get_device_name(0)
    return summary


def _build_probabilities_rows(
    results: list[PerClipResult],
    class_names: tuple[str, ...] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return one row per (clip, window) with explicit target/speech columns."""

    target_names: list[str] = []
    for idx in TARGET_INDICES:
        target_names.append(
            class_names[idx] if class_names is not None else f"target_idx_{idx}"
        )
    speech_names: list[str] = []
    for idx in SPEECH_LIKE_INDICES:
        speech_names.append(
            class_names[idx] if class_names is not None else f"speech_idx_{idx}"
        )

    fieldnames = (
        [
            "clip_label",
            "kind",
            "is_short_utterance",
            "window_index",
            "window_start_seconds",
            "max_overall_prob",
            "policy_max",
            "policy_sum",
            "policy_target_minus_speech",
        ]
        + [f"target__{name}" for name in target_names]
        + [f"speech__{name}" for name in speech_names]
    )

    rows: list[dict[str, Any]] = []
    for clip in results:
        for w_idx in range(clip.per_window_probs.shape[0]):
            probs_vec = clip.per_window_probs[w_idx]
            row: dict[str, Any] = {
                "clip_label": clip.label,
                "kind": clip.kind,
                "is_short_utterance": str(bool(clip.is_short_utterance)).lower(),
                "window_index": w_idx,
                "window_start_seconds": round(w_idx * SED_WINDOW_SECONDS, 3),
                "max_overall_prob": round(float(probs_vec.max()), 6),
                "policy_max": round(float(POLICIES["max"](probs_vec)), 6),
                "policy_sum": round(float(POLICIES["sum"](probs_vec)), 6),
                "policy_target_minus_speech": round(
                    float(POLICIES["target_minus_speech"](probs_vec)), 6
                ),
            }
            for name, idx in zip(target_names, TARGET_INDICES):
                row[f"target__{name}"] = round(float(probs_vec[idx]), 6)
            for name, idx in zip(speech_names, SPEECH_LIKE_INDICES):
                row[f"speech__{name}"] = round(float(probs_vec[idx]), 6)
            rows.append(row)

    return rows, fieldnames


def _write_probabilities(
    output_dir: Path,
    results: list[PerClipResult],
    class_names: tuple[str, ...] | None,
) -> None:
    rows, fieldnames = _build_probabilities_rows(results, class_names)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "probabilities.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    npz_path = output_dir / "probabilities_full.npz"
    # Use fixed-width Unicode dtypes (not object) so the analyse step can read
    # the archive with allow_pickle=False — pickle deserialisation of an
    # externally generated artefact is a security smell we want to avoid even
    # for research outputs (codex-review on #306, low-severity).
    labels = np.array([clip.label for clip in results], dtype=np.str_)
    kinds = np.array([clip.kind for clip in results], dtype=np.str_)
    short_flags = np.array(
        [bool(clip.is_short_utterance) for clip in results], dtype=bool
    )
    np.savez(
        npz_path,
        labels=labels,
        kinds=kinds,
        is_short_utterance=short_flags,
        **{f"probs__{clip.label}": clip.per_window_probs for clip in results},
    )


def _write_metadata(
    output_dir: Path,
    *,
    variant: str,
    device: str,
    corpus_dir: Path,
    efficientat_path: Path,
    n_clips: int,
    n_windows: int,
    has_latency: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "issue": "https://github.com/Mega-Gorilla/livecap-cli/issues/305",
        "phase": "PR-D0 (off-line evaluation)",
        "timestamp_utc": _utc_timestamp(),
        "model": {
            "variant": variant,
            "device": device,
            "efficientat_clone_path": str(efficientat_path),
            "efficientat_commit": _git_commit(efficientat_path),
        },
        "corpus": {
            "path": str(corpus_dir),
            "n_clips": n_clips,
            "total_windows": n_windows,
            "window_seconds": SED_WINDOW_SECONDS,
        },
        "hardware": _hardware_summary(),
        "outputs": {
            "probabilities_csv": "probabilities.csv",
            "probabilities_full_npz": "probabilities_full.npz",
            "latency_csv": "latency.csv" if has_latency else None,
            "metadata_json": "metadata.json",
        },
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run_evaluation(args: argparse.Namespace) -> int:
    # Late import keeps --help working without the EfficientAT clone.
    from benchmarks.non_speech_filter.pipeline import load_real_corpus_items

    efficientat_path = resolve_efficientat_path(args.efficientat_path)
    corpus_dir = _resolve_corpus_dir(args.corpus_dir)
    device = resolve_device(args.device)

    print(
        f"[sed] variant={args.model} device={device} "
        f"efficientat={efficientat_path} corpus={corpus_dir}"
    )

    print("[sed] loading model + corpus")
    bundle = load_model(args.model, device=device, efficientat_path=efficientat_path)
    corpus_items = load_real_corpus_items(corpus_dir)
    print(f"[sed] loaded {len(corpus_items)} corpus clips")

    class_names: tuple[str, ...] | None
    try:
        class_names = load_class_names(
            efficientat_path / "metadata" / "class_labels_indices.csv"
        )
    except FileNotFoundError:
        class_names = None

    results: list[PerClipResult] = []
    total_windows = 0
    for clip in corpus_items:
        probs = compute_window_probs(clip.audio, bundle)
        print(
            f"  - {clip.label:35s} kind={clip.kind:8s} "
            f"windows={probs.shape[0]:>3d}"
        )
        results.append(
            PerClipResult(
                label=clip.label,
                kind=clip.kind,  # type: ignore[arg-type]
                is_short_utterance=bool(clip.is_short_utterance),
                per_window_probs=probs,
            )
        )
        total_windows += probs.shape[0]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sed] writing probabilities to {output_dir}")
    _write_probabilities(output_dir, results, class_names)

    has_latency = not args.skip_latency
    if has_latency:
        print("[sed] measuring latency (CPU + GPU if available)")
        sample = np.zeros(int(SED_WINDOW_SECONDS * 16_000), dtype=np.float32)
        measurement = measure_all_axes(
            args.model,
            efficientat_path,
            sample,
            iterations=args.latency_iters,
        )
        write_latency_csv([measurement], output_dir / "latency.csv")
    else:
        print("[sed] --skip-latency set; latency.csv will not be written")

    _write_metadata(
        output_dir,
        variant=args.model,
        device=device,
        corpus_dir=corpus_dir,
        efficientat_path=efficientat_path,
        n_clips=len(results),
        n_windows=total_windows,
        has_latency=has_latency,
    )
    print("[sed] done")
    return 0
