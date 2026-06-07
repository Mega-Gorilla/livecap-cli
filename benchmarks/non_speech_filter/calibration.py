"""Calibration analysis for the PR-B threshold sweep (Issue #295).

Reads the CSV emitted by :mod:`benchmarks.non_speech_filter.sweep` and
produces a Markdown report that turns the raw cell table into the four
artefacts the PR-B follow-up needs:

1. **Per-engine hallucination delta** — for each engine, how much does
   each on-mode preset move ``non_empty_hallucination_rate`` versus the
   ``baseline_off`` reference (segmented by corpus).
2. **Recall guard** — flags any (preset, backend, engine, corpus) cell
   where ``speech_recall`` or ``short_utterance_recall`` regressed
   below the baseline value.
3. **Pareto frontier** — preset list ordered by
   ``mean false_asr_trigger_rate ↓`` with the corresponding recall
   floor; presets that strictly dominate others are starred.
4. **Recommendation** — one of three structured verdicts driven by the
   thresholds documented in the plan file (Issue #295 PR-B calibration
   follow-up, decision rule D4).

The script is deliberately a *post-hoc* analysis tool: it does not
re-run the sweep. Re-running is the user's choice via
``python -m benchmarks.non_speech_filter.sweep``.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# ---------- Data ---------------------------------------------------------


@dataclass(frozen=True)
class SweepCell:
    """One row from the sweep CSV — kept in a frozen dataclass so the
    analysis surface is the same whether we read CSV or (future) JSON."""

    preset: str
    backend: str
    engine: str
    corpus: str
    mode: str
    false_asr_trigger_rate: float | None
    speech_recall: float | None
    short_utterance_recall: float | None
    non_empty_hallucination_rate: float | None
    added_latency_p50_ms: float
    added_latency_p95_ms: float


@dataclass
class PerEngineDelta:
    """Hallucination delta for one (engine, corpus, backend) trajectory."""

    engine: str
    corpus: str
    backend: str
    baseline_off_rate: float | None
    per_preset_rate: dict[str, float | None] = field(default_factory=dict)

    def best_preset(self) -> tuple[str, float] | None:
        """Return (preset, hallucination_rate) for the lowest hallucination."""
        candidates = [(name, rate) for name, rate in self.per_preset_rate.items() if rate is not None]
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[1])


@dataclass
class RecallRegression:
    preset: str
    backend: str
    engine: str
    corpus: str
    metric: str  # 'speech_recall' or 'short_utterance_recall'
    baseline_value: float | None
    preset_value: float | None
    delta: float | None  # baseline - preset (positive = regression)


@dataclass
class CalibrationReport:
    """Aggregated calibration findings for one sweep CSV."""

    source_csv: Path
    generated_at: str
    cells: list[SweepCell] = field(default_factory=list)
    per_engine_deltas: list[PerEngineDelta] = field(default_factory=list)
    recall_regressions: list[RecallRegression] = field(default_factory=list)
    pareto_summary: list[dict] = field(default_factory=list)
    recommendation: str = ""

    # ---- Markdown rendering --------------------------------------------

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append("# Transient Detector Calibration Report")
        lines.append("")
        lines.append(f"- **Generated**: {self.generated_at}")
        lines.append(f"- **Source**: `{self.source_csv}`")
        lines.append(f"- **Cells parsed**: {len(self.cells)}")
        presets = sorted({c.preset for c in self.cells})
        backends = sorted({c.backend for c in self.cells})
        engines = sorted({c.engine for c in self.cells})
        corpora = sorted({c.corpus for c in self.cells})
        lines.append(
            f"- **Presets**: {', '.join(presets)} ({len(presets)} total)"
        )
        lines.append(f"- **Backends**: {', '.join(backends)}")
        lines.append(f"- **Engines**: {', '.join(engines)}")
        lines.append(f"- **Corpora**: {', '.join(corpora)}")
        lines.append("")
        lines.append("---")
        lines.append("")

        lines.extend(self._render_per_engine_deltas())
        lines.extend(self._render_recall_regressions())
        lines.extend(self._render_pareto_summary())
        lines.extend(self._render_recommendation())
        return "\n".join(lines)

    def _render_per_engine_deltas(self) -> Iterable[str]:
        yield "## 1. Per-engine hallucination delta vs `baseline_off`"
        yield ""
        if not self.per_engine_deltas:
            yield "_no engine ran with `non_empty_hallucination_rate` data; possibly mock-only sweep._"
            yield ""
            return
        yield "Each row is one (engine, backend, corpus) trajectory across the on-mode"
        yield "presets. **Lower is better**. `Δ` columns subtract `baseline_off`."
        yield ""
        # Group by (engine, corpus) for readability
        by_ec: dict[tuple[str, str], list[PerEngineDelta]] = defaultdict(list)
        for d in self.per_engine_deltas:
            by_ec[(d.engine, d.corpus)].append(d)

        for (engine, corpus), entries in sorted(by_ec.items()):
            yield f"### {engine} × {corpus}"
            yield ""
            on_presets = sorted({
                name
                for entry in entries
                for name in entry.per_preset_rate
                if name != "baseline_off"
            })
            header = ["Backend", "baseline_off"] + on_presets
            yield "| " + " | ".join(header) + " |"
            yield "|" + "|".join(["---"] * len(header)) + "|"
            for entry in sorted(entries, key=lambda e: e.backend):
                cells = [entry.backend, _fmt_pct(entry.baseline_off_rate)]
                for p in on_presets:
                    rate = entry.per_preset_rate.get(p)
                    rate_str = _fmt_pct(rate)
                    if rate is not None and entry.baseline_off_rate is not None:
                        delta = rate - entry.baseline_off_rate
                        rate_str = f"{rate_str} ({delta:+.1%})"
                    cells.append(rate_str)
                yield "| " + " | ".join(cells) + " |"
            yield ""

    def _render_recall_regressions(self) -> Iterable[str]:
        yield "## 2. Recall regressions (vs `baseline_off`)"
        yield ""
        if not self.recall_regressions:
            yield "**No recall regression detected** across any (preset, backend, engine, corpus) cell."
            yield "All positive items remain triggered at or above the baseline rate."
            yield ""
            return
        yield "Cells where an on-mode preset dropped speech recall below the baseline."
        yield "**Any non-trivial regression here means that preset must not become a default.**"
        yield ""
        header = ["Preset", "Backend", "Engine", "Corpus", "Metric", "Baseline", "On-mode", "Δ"]
        yield "| " + " | ".join(header) + " |"
        yield "|" + "|".join(["---"] * len(header)) + "|"
        for r in sorted(self.recall_regressions, key=lambda x: (x.preset, x.backend, x.engine)):
            yield "| " + " | ".join([
                r.preset,
                r.backend,
                r.engine,
                r.corpus,
                r.metric,
                _fmt_pct(r.baseline_value),
                _fmt_pct(r.preset_value),
                f"{r.delta:+.1%}" if r.delta is not None else "-",
            ]) + " |"
        yield ""

    def _render_pareto_summary(self) -> Iterable[str]:
        yield "## 3. Pareto summary — false_trigger ↓ vs recall ≥"
        yield ""
        if not self.pareto_summary:
            yield "_insufficient data to compute pareto._"
            yield ""
            return
        yield "Each preset is summarised by its **mean** metrics across all"
        yield "(backend x engine x corpus) cells. Pareto-dominant presets are flagged."
        yield ""
        header = [
            "Preset",
            "Mean False Trigger",
            "Mean Speech Recall",
            "Mean Short Recall",
            "Mean Hallucination",
            "Dominates?",
        ]
        yield "| " + " | ".join(header) + " |"
        yield "|" + "|".join(["---"] * len(header)) + "|"
        for row in self.pareto_summary:
            yield "| " + " | ".join([
                row["preset"],
                _fmt_pct(row.get("mean_false_trigger")),
                _fmt_pct(row.get("mean_speech_recall")),
                _fmt_pct(row.get("mean_short_recall")),
                _fmt_pct(row.get("mean_hallucination")),
                "yes" if row.get("pareto_dominant") else "",
            ]) + " |"
        yield ""

    def _render_recommendation(self) -> Iterable[str]:
        yield "## 4. Recommendation"
        yield ""
        yield self.recommendation or "_no recommendation generated._"
        yield ""

    # ---- I/O -----------------------------------------------------------

    def save_markdown(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")


# ---------- Helpers ------------------------------------------------------


def _fmt_pct(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "-"


def _parse_float(value: str) -> float | None:
    if value == "" or value.lower() == "none":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_float_required(value: str) -> float:
    """Parse a column that the schema guarantees is numeric (latencies)."""
    try:
        return float(value)
    except ValueError:
        return 0.0


def load_cells_from_csv(path: Path) -> list[SweepCell]:
    rows: list[SweepCell] = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(
                SweepCell(
                    preset=r["preset"],
                    backend=r["backend"],
                    engine=r["engine"],
                    corpus=r["corpus"],
                    mode=r["mode"],
                    false_asr_trigger_rate=_parse_float(r["false_asr_trigger_rate"]),
                    speech_recall=_parse_float(r["speech_recall"]),
                    short_utterance_recall=_parse_float(r["short_utterance_recall"]),
                    non_empty_hallucination_rate=_parse_float(
                        r["non_empty_hallucination_rate"]
                    ),
                    added_latency_p50_ms=_parse_float_required(r["added_latency_p50_ms"]),
                    added_latency_p95_ms=_parse_float_required(r["added_latency_p95_ms"]),
                )
            )
    return rows


# ---------- Analysis -----------------------------------------------------


# Decision thresholds from the plan (D4): hallucination must drop by at
# least this much vs baseline_off, and recall floors must hold.
HALLUCINATION_DROP_REQUIRED = 0.30
SPEECH_RECALL_FLOOR = 0.95
SHORT_UTTERANCE_RECALL_FLOOR = 1.0
# The cell the PR-B AC singled out: webrtc x parakeet_ja x real.
TARGET_BACKEND = "webrtc"
TARGET_ENGINE = "parakeet_ja"
TARGET_CORPUS = "real"


def _index_by_key(cells: Iterable[SweepCell]) -> dict[tuple, SweepCell]:
    return {
        (c.preset, c.backend, c.engine, c.corpus): c
        for c in cells
    }


def _compute_per_engine_deltas(cells: list[SweepCell]) -> list[PerEngineDelta]:
    by_key = _index_by_key(cells)
    deltas: list[PerEngineDelta] = []
    by_ebc: dict[tuple[str, str, str], list[SweepCell]] = defaultdict(list)
    for c in cells:
        by_ebc[(c.engine, c.backend, c.corpus)].append(c)
    for (engine, backend, corpus), group in by_ebc.items():
        baseline = by_key.get(("baseline_off", backend, engine, corpus))
        if baseline is None:
            continue
        entry = PerEngineDelta(
            engine=engine,
            corpus=corpus,
            backend=backend,
            baseline_off_rate=baseline.non_empty_hallucination_rate,
        )
        for c in group:
            if c.preset == "baseline_off":
                continue
            entry.per_preset_rate[c.preset] = c.non_empty_hallucination_rate
        if entry.per_preset_rate:
            deltas.append(entry)
    return deltas


def _compute_recall_regressions(cells: list[SweepCell]) -> list[RecallRegression]:
    """Flag any cell where speech_recall or short_utterance_recall regressed.

    "Regressed" means below the baseline value for the same (backend,
    engine, corpus). A 0.0 vs 0.0 tie is not a regression.
    """
    by_key = _index_by_key(cells)
    regressions: list[RecallRegression] = []
    for c in cells:
        if c.preset == "baseline_off":
            continue
        baseline = by_key.get(("baseline_off", c.backend, c.engine, c.corpus))
        if baseline is None:
            continue
        for metric in ("speech_recall", "short_utterance_recall"):
            base_val = getattr(baseline, metric)
            preset_val = getattr(c, metric)
            if base_val is None or preset_val is None:
                continue
            if preset_val < base_val - 1e-9:
                regressions.append(
                    RecallRegression(
                        preset=c.preset,
                        backend=c.backend,
                        engine=c.engine,
                        corpus=c.corpus,
                        metric=metric,
                        baseline_value=base_val,
                        preset_value=preset_val,
                        delta=base_val - preset_val,
                    )
                )
    return regressions


def _safe_mean(values: list[float]) -> float | None:
    finite = [v for v in values if v is not None]
    return mean(finite) if finite else None


def _compute_pareto_summary(cells: list[SweepCell]) -> list[dict]:
    """Per-preset aggregates + naive Pareto-dominance flag.

    A preset is Pareto-dominant if no other preset has *strictly lower*
    mean false_trigger AND *no worse* mean speech_recall / short recall.
    """
    by_preset: dict[str, list[SweepCell]] = defaultdict(list)
    for c in cells:
        by_preset[c.preset].append(c)
    rows: list[dict] = []
    for preset, group in by_preset.items():
        rows.append(
            {
                "preset": preset,
                "mean_false_trigger": _safe_mean(
                    [c.false_asr_trigger_rate for c in group]
                ),
                "mean_speech_recall": _safe_mean(
                    [c.speech_recall for c in group]
                ),
                "mean_short_recall": _safe_mean(
                    [c.short_utterance_recall for c in group]
                ),
                "mean_hallucination": _safe_mean(
                    [c.non_empty_hallucination_rate for c in group]
                ),
            }
        )
    # Pareto dominance (lower false_trigger is better; higher recall is better)
    for row in rows:
        row["pareto_dominant"] = _is_pareto_dominant(row, rows)
    # Sort by mean false_trigger ascending.
    rows.sort(key=lambda r: (r["mean_false_trigger"] if r["mean_false_trigger"] is not None else 1.0))
    return rows


def _is_pareto_dominant(candidate: dict, all_rows: list[dict]) -> bool:
    cf = candidate["mean_false_trigger"]
    cs = candidate["mean_speech_recall"]
    ch = candidate["mean_short_recall"]
    if cf is None:
        return False
    for other in all_rows:
        if other["preset"] == candidate["preset"]:
            continue
        of = other["mean_false_trigger"]
        os = other["mean_speech_recall"]
        oh = other["mean_short_recall"]
        if of is None:
            continue
        # Candidate is dominated by `other` if other is strictly lower on
        # false_trigger AND at least as good on both recall metrics.
        if of < cf and (os is None or cs is None or os >= cs) and (
            oh is None or ch is None or oh >= ch
        ):
            return False
    return True


def _build_recommendation(
    deltas: list[PerEngineDelta],
    regressions: list[RecallRegression],
) -> str:
    """Render the structured verdict required by plan D4."""

    # Identify presets that satisfy the target cell criterion.
    target_winners: list[tuple[str, float, float]] = []
    for d in deltas:
        if d.backend != TARGET_BACKEND or d.engine != TARGET_ENGINE or d.corpus != TARGET_CORPUS:
            continue
        if d.baseline_off_rate is None or d.baseline_off_rate <= 0:
            continue
        for preset, rate in d.per_preset_rate.items():
            if rate is None:
                continue
            drop = d.baseline_off_rate - rate
            relative = drop / d.baseline_off_rate
            if relative >= HALLUCINATION_DROP_REQUIRED:
                target_winners.append((preset, rate, relative))

    # Filter out winners that triggered any recall regression.
    bad_presets = {r.preset for r in regressions}
    safe_target_winners = [w for w in target_winners if w[0] not in bad_presets]

    lines: list[str] = []
    if safe_target_winners:
        safe_target_winners.sort(key=lambda x: x[1])
        winner, rate, drop = safe_target_winners[0]
        lines.append(
            f"**Recommended action: promote `{winner}` toward production.** "
            f"It cuts `{TARGET_BACKEND} × {TARGET_ENGINE} × {TARGET_CORPUS}` "
            f"hallucination from baseline to {rate:.1%} "
            f"({drop:.0%} relative drop) without triggering any recall "
            f"regression in the sweep."
        )
        lines.append("")
        lines.append(
            "Decision still belongs to the maintainer: changing the CLI "
            "default to `on` with this preset also requires updating "
            "`BASELINE_INVARIANTS` and the Issue #295 AC."
        )
    elif target_winners:
        lines.append(
            "**No safe winner.** "
            f"{len(target_winners)} preset(s) hit the "
            f"≥{HALLUCINATION_DROP_REQUIRED:.0%} hallucination drop target, "
            "but they all triggered a recall regression somewhere in the "
            "sweep. See section 2 for details."
        )
        lines.append("")
        lines.append(
            "Recommended next step: keep `--transient-filter=off` as the CLI "
            "default; document the best-effort recall-safe preset as a calibration "
            "option in the docs; consider Phase 2 SED for `desk_tap`-style "
            "low-frequency transients that the AND design cannot catch."
        )
    else:
        lines.append(
            "**DSP detector cannot meet the AC target with the current "
            "candidate presets.** "
            f"No preset achieved ≥{HALLUCINATION_DROP_REQUIRED:.0%} "
            f"hallucination drop on `{TARGET_BACKEND} × {TARGET_ENGINE} × "
            f"{TARGET_CORPUS}`."
        )
        lines.append("")
        lines.append("Recommended next step:")
        lines.append("")
        lines.append("1. Keep `--transient-filter=off` as the CLI default.")
        lines.append(
            "2. Reframe the Issue #295 PR-B AC to record the empirical "
            "achievable bound rather than the original `50 % → 0 %` target."
        )
        lines.append(
            "3. Open a Phase 2 SED epic for low-frequency / non-broadband "
            "transient detection — the 6-feature AND design is structurally "
            "unable to fire on clips where centroid / flatness sit at 0 %."
        )
    return "\n".join(lines)


def analyze_sweep(csv_path: Path) -> CalibrationReport:
    cells = load_cells_from_csv(csv_path)
    deltas = _compute_per_engine_deltas(cells)
    regressions = _compute_recall_regressions(cells)
    pareto = _compute_pareto_summary(cells)
    recommendation = _build_recommendation(deltas, regressions)
    return CalibrationReport(
        source_csv=csv_path,
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
        cells=cells,
        per_engine_deltas=deltas,
        recall_regressions=regressions,
        pareto_summary=pareto,
        recommendation=recommendation,
    )


# ---------- CLI ----------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.non_speech_filter.calibration",
        description=(
            "Read a sweep CSV emitted by "
            "`python -m benchmarks.non_speech_filter.sweep` and produce a "
            "calibration Markdown report (Issue #295 PR-B follow-up)."
        ),
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to transient_sweep_<timestamp>.csv from the sweep harness.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output Markdown path (default: print to stdout).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.csv_path.exists():
        print(f"sweep CSV not found: {args.csv_path}", file=sys.stderr)
        return 1
    report = analyze_sweep(args.csv_path)
    if args.output:
        report.save_markdown(args.output)
        print(f"wrote calibration report: {args.output}")
    else:
        print(report.to_markdown())
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
