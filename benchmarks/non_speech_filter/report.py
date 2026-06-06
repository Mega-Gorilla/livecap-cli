"""Reporting for the non-speech filter benchmark runner.

A lightweight, domain-specific report (rather than reusing BenchmarkResult
from ``benchmarks.common``) so per-corpus-item metrics, hallucination text,
and per-backend × per-engine matrices can be persisted without forcing a
schema mismatch with ASR/VAD benchmarks.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class NonSpeechFilterRunRecord:
    """One (backend, engine, corpus, run_index) combination."""

    backend: str
    engine: str
    corpus: str  # "synthetic" | "real"
    run_index: int

    negative_total: int
    positive_total: int
    short_total: int

    false_asr_trigger_rate: float | None
    speech_recall: float | None
    short_utterance_recall: float | None
    non_empty_hallucination_rate: float | None

    added_latency_p50_ms: float
    added_latency_p95_ms: float

    per_label: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NonSpeechFilterReport:
    """Top-level report aggregating all benchmark runs."""

    timestamp: str
    mode: str
    device: str
    runs: int
    records: list[NonSpeechFilterRunRecord] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def add_record(self, record: NonSpeechFilterRunRecord) -> None:
        self.records.append(record)

    def add_skip(self, reason: str) -> None:
        self.skipped.append(reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": {
                "timestamp": self.timestamp,
                "mode": self.mode,
                "device": self.device,
                "runs": self.runs,
            },
            "records": [r.to_dict() for r in self.records],
            "skipped": list(self.skipped),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append("# Non-Speech Filter Benchmark Report")
        lines.append("")
        lines.append(f"- **Date:** {self.timestamp}")
        lines.append(f"- **Mode:** {self.mode}")
        lines.append(f"- **Device:** {self.device}")
        lines.append(f"- **Runs per cell:** {self.runs}")
        lines.append("")

        if not self.records:
            lines.append("No records.")
            return "\n".join(lines)

        # Aggregate by (backend, engine, corpus).
        from statistics import mean

        groups: dict[tuple[str, str, str], list[NonSpeechFilterRunRecord]] = {}
        for r in self.records:
            key = (r.backend, r.engine, r.corpus)
            groups.setdefault(key, []).append(r)

        lines.append("## Summary by Backend × Engine × Corpus")
        lines.append("")
        headers = (
            "Backend",
            "Engine",
            "Corpus",
            "False Trigger",
            "Speech Recall",
            "Short Recall",
            "Hallucination",
            "P50 ms",
            "P95 ms",
        )
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for key in sorted(groups.keys()):
            backend, engine, corpus = key
            records = groups[key]
            ft = [r.false_asr_trigger_rate for r in records if r.false_asr_trigger_rate is not None]
            sr = [r.speech_recall for r in records if r.speech_recall is not None]
            su = [r.short_utterance_recall for r in records if r.short_utterance_recall is not None]
            hl = [
                r.non_empty_hallucination_rate
                for r in records
                if r.non_empty_hallucination_rate is not None
            ]
            p50 = [r.added_latency_p50_ms for r in records]
            p95 = [r.added_latency_p95_ms for r in records]
            row = (
                backend,
                engine,
                corpus,
                f"{mean(ft):.1%}" if ft else "-",
                f"{mean(sr):.1%}" if sr else "-",
                f"{mean(su):.1%}" if su else "-",
                f"{mean(hl):.1%}" if hl else "-",
                f"{mean(p50):.2f}" if p50 else "-",
                f"{mean(p95):.2f}" if p95 else "-",
            )
            lines.append("| " + " | ".join(row) + " |")

        if self.skipped:
            lines.append("")
            lines.append("## Skipped")
            lines.append("")
            for s in self.skipped:
                lines.append(f"- {s}")
        lines.append("")
        return "\n".join(lines)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    def save_markdown(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")


def new_report(mode: str, device: str, runs: int) -> NonSpeechFilterReport:
    return NonSpeechFilterReport(
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        mode=mode,
        device=device,
        runs=runs,
    )
