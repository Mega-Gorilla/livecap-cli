"""Post-hoc analysis of the ``benchmarks/sed`` raw outputs.

Reads ``probabilities_full.npz`` + ``metadata.json`` from an orchestrator
run, computes class-level + reject-signal-level precision / recall sweeps,
applies the Issue #305 v3 provisional gate, and emits both a JSON summary
and a human-readable Markdown table for the decision document.

CLI::

    python -m benchmarks.sed.analyze --input-dir benchmark_results/sed/2026-06-10/

Outputs (written next to the input):

- ``analysis.json`` — machine-readable verdict + sweep numbers
- ``analysis.md`` — Markdown tables ready to paste into the decision doc
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from benchmarks.sed.class_mapping import (
    POLICIES,
    SPEECH_LIKE_INDICES,
    TARGET_INDICES,
    load_class_names,
)
from benchmarks.sed.metrics import (
    PROVISIONAL_PRECISION_FLOOR,
    PROVISIONAL_RECALL_FLOOR,
    PerClipResult,
    compute_class_level_metrics,
    compute_reject_signal_curve,
    provisional_gate_verdict,
)


DEFAULT_THRESHOLDS: tuple[float, ...] = (
    0.01,
    0.02,
    0.03,
    0.05,
    0.08,
    0.10,
    0.15,
    0.20,
    0.30,
    0.50,
)
"""Threshold sweep tuned for the empirically observed mn04_as score range."""


def load_results(npz_path: Path) -> list[PerClipResult]:
    data = np.load(npz_path, allow_pickle=True)
    labels: Sequence[str] = list(data["labels"])
    kinds: Sequence[str] = list(data["kinds"])
    short: Sequence[bool] = list(data["is_short_utterance"])

    results: list[PerClipResult] = []
    for label, kind, is_short in zip(labels, kinds, short):
        probs = data[f"probs__{label}"]
        results.append(
            PerClipResult(
                label=str(label),
                kind=str(kind),  # type: ignore[arg-type]
                is_short_utterance=bool(is_short),
                per_window_probs=probs.astype(np.float32, copy=False),
            )
        )
    return results


def _summarise_curve(curve, thresholds: Sequence[float]) -> list[dict]:
    return [
        {
            "threshold": round(float(t), 4),
            "precision": round(float(curve.precision[i]), 4),
            "recall": round(float(curve.recall[i]), 4),
        }
        for i, t in enumerate(thresholds)
    ]


def _markdown_curve_table(
    curves: list, policy_names: Sequence[str], thresholds: Sequence[float]
) -> str:
    header = (
        "| Threshold | "
        + " | ".join(f"P ({p})/R ({p})" for p in policy_names)
        + " |"
    )
    sep = "|---" * (1 + len(policy_names)) + "|"
    rows: list[str] = [header, sep]
    for i, t in enumerate(thresholds):
        cells = [f"{float(t):.2f}"]
        for curve in curves:
            cells.append(
                f"{float(curve.precision[i]):.2f}/{float(curve.recall[i]):.2f}"
            )
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _markdown_clip_scores_table(curves: list, policy_names: Sequence[str]) -> str:
    """Per-clip max reject score across all policies (the decision unit)."""

    clip_order: list[str] = []
    for curve in curves:
        for label in curve.per_clip_scores:
            if label not in clip_order:
                clip_order.append(label)

    header = "| Clip | " + " | ".join(policy_names) + " |"
    sep = "|---" * (1 + len(policy_names)) + "|"
    rows = [header, sep]
    for label in clip_order:
        cells = [label]
        for curve in curves:
            score = curve.per_clip_scores.get(label, float("nan"))
            cells.append(f"{score:.4f}")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _markdown_class_table(class_metrics: list, threshold: float) -> str:
    header = (
        f"| Class | Index | TP | FP | FN | TN | Precision | Recall | "
        f"(threshold {threshold:.2f}) |"
    )
    sep = "|---" * 9 + "|"
    rows = [header, sep]
    for m in class_metrics:
        rows.append(
            f"| {m.class_name} | {m.class_index} | {m.counts.tp} | "
            f"{m.counts.fp} | {m.counts.fn} | {m.counts.tn} | "
            f"{m.precision:.2f} | {m.recall:.2f} | |"
        )
    return "\n".join(rows)


def run_analysis(args: argparse.Namespace) -> int:
    input_dir: Path = args.input_dir
    npz_path = input_dir / "probabilities_full.npz"
    metadata_path = input_dir / "metadata.json"
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing {npz_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    efficientat_clone = Path(metadata["model"]["efficientat_clone_path"])

    class_names: tuple[str, ...] | None
    try:
        class_names = load_class_names(
            efficientat_clone / "metadata" / "class_labels_indices.csv"
        )
    except FileNotFoundError:
        class_names = None

    results = load_results(npz_path)

    # 1. Reject-signal curves per policy
    thresholds = list(args.thresholds)
    policy_names = list(POLICIES.keys())
    curves = [
        compute_reject_signal_curve(results, name, thresholds) for name in policy_names
    ]

    # 2. Class-level metrics for TARGET + SPEECH_LIKE classes at the gate threshold
    target_names = (
        [class_names[i] for i in TARGET_INDICES]
        if class_names is not None
        else [f"target_{i}" for i in TARGET_INDICES]
    )
    speech_names = (
        [class_names[i] for i in SPEECH_LIKE_INDICES]
        if class_names is not None
        else [f"speech_{i}" for i in SPEECH_LIKE_INDICES]
    )
    target_class_metrics = compute_class_level_metrics(
        results, TARGET_INDICES, target_names, threshold=args.class_threshold
    )
    speech_class_metrics = compute_class_level_metrics(
        results, SPEECH_LIKE_INDICES, speech_names, threshold=args.class_threshold
    )

    # 3. Provisional gate
    verdict = provisional_gate_verdict(curves, target_label=args.target_label)

    summary = {
        "metadata": metadata,
        "thresholds": thresholds,
        "policies": policy_names,
        "verdict": {
            "target_label": verdict.target_label,
            "precision_floor": verdict.precision_floor,
            "recall_floor": verdict.recall_floor,
            "is_provisional": verdict.is_provisional,
            "passed": verdict.passed,
            "passing_policies": list(verdict.passing_policies),
            "chosen_policy": verdict.chosen_policy,
            "chosen_threshold": verdict.chosen_threshold,
            "chosen_precision": verdict.chosen_precision,
            "chosen_recall": verdict.chosen_recall,
            "target_flagged_at_chosen_threshold": verdict.target_flagged_at_chosen_threshold,
            "notes": list(verdict.notes),
        },
        "reject_signal_curves": {
            curve.policy: _summarise_curve(curve, thresholds) for curve in curves
        },
        "clip_max_scores_per_policy": {
            curve.policy: {
                label: round(float(score), 6)
                for label, score in curve.per_clip_scores.items()
            }
            for curve in curves
        },
        "target_class_metrics": [
            {
                "class_name": m.class_name,
                "class_index": m.class_index,
                "tp": m.counts.tp,
                "fp": m.counts.fp,
                "fn": m.counts.fn,
                "tn": m.counts.tn,
                "precision": round(m.precision, 4),
                "recall": round(m.recall, 4),
                "threshold": m.threshold,
            }
            for m in target_class_metrics
        ],
        "speech_class_metrics": [
            {
                "class_name": m.class_name,
                "class_index": m.class_index,
                "tp": m.counts.tp,
                "fp": m.counts.fp,
                "fn": m.counts.fn,
                "tn": m.counts.tn,
                "precision": round(m.precision, 4),
                "recall": round(m.recall, 4),
                "threshold": m.threshold,
            }
            for m in speech_class_metrics
        ],
    }

    json_path = input_dir / "analysis.json"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md_lines = [
        f"# SED PR-D0 analysis — {metadata['model']['variant']}",
        "",
        f"Generated from `{npz_path.name}` (commit "
        f"`{metadata['model']['efficientat_commit']}`).",
        "",
        "## 1. Provisional gate (Issue #305 v3)",
        "",
        f"- Target clip: `{verdict.target_label}`",
        f"- Precision floor: {verdict.precision_floor:.2f}",
        f"- Recall floor: {verdict.recall_floor:.2f}",
        f"- **Result: {'PASS' if verdict.passed else 'FAIL'}**",
        f"- Passing policies: {', '.join(verdict.passing_policies) if verdict.passing_policies else '—'}",
        f"- Chosen (policy, threshold): "
        f"({verdict.chosen_policy or '—'}, "
        f"{verdict.chosen_threshold if verdict.chosen_threshold is not None else '—'})",
        f"- Chosen (precision, recall): "
        f"({verdict.chosen_precision if verdict.chosen_precision is not None else '—'}, "
        f"{verdict.chosen_recall if verdict.chosen_recall is not None else '—'})",
        "",
    ]
    for note in verdict.notes:
        md_lines.append(f"> {note}")
        md_lines.append("")

    md_lines.extend(
        [
            "## 2. Reject-signal-level P/R sweep",
            "",
            _markdown_curve_table(curves, policy_names, thresholds),
            "",
            "## 3. Clip-level max reject scores",
            "",
            _markdown_clip_scores_table(curves, policy_names),
            "",
            f"## 4. Class-level metrics @ threshold {args.class_threshold:.2f}",
            "",
            "### Target classes",
            "",
            _markdown_class_table(target_class_metrics, args.class_threshold),
            "",
            "### Speech-like classes (should ideally have low TP)",
            "",
            _markdown_class_table(speech_class_metrics, args.class_threshold),
            "",
        ]
    )

    md_path = input_dir / "analysis.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(
        f"[sed analyze] verdict={'PASS' if verdict.passed else 'FAIL'}; wrote "
        f"{json_path.name} + {md_path.name}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmarks.sed.analyze",
        description="Post-hoc analysis of SED PR-D0 evaluation outputs",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing probabilities_full.npz + metadata.json",
    )
    parser.add_argument(
        "--target-label",
        default="desk_tap",
        help="Clip label that must be flagged at the chosen threshold (default: desk_tap)",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLDS),
        help="Threshold values for the P/R sweep",
    )
    parser.add_argument(
        "--class-threshold",
        type=float,
        default=0.05,
        help="Threshold used for the per-class metric table",
    )
    args = parser.parse_args(argv)
    return run_analysis(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
