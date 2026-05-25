"""Reporting for the speaker-embedding benchmark.

Self-contained (the shared ``BenchmarkResult`` is ASR/transcript shaped, which
does not fit speaker metrics). Produces JSON + Markdown + console output.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from tabulate import tabulate

    TABULATE_AVAILABLE = True
except ImportError:  # pragma: no cover
    TABULATE_AVAILABLE = False


@dataclass
class SpeakerBenchmarkResult:
    """One backend's measured results."""

    backend: str
    device: str
    status: str = "ok"  # ok | skipped | failed
    detail: str = ""  # reason when skipped/failed

    # Dataset context
    num_segments: int | None = None
    audio_duration_s: float | None = None
    embedding_dim: int | None = None

    # Performance
    load_s: float | None = None
    embed_latency_ms_p50: float | None = None
    embed_latency_ms_p95: float | None = None
    embed_latency_ms_mean: float | None = None
    rtf: float | None = None  # total embedding time / audio duration

    # Memory
    gpu_model_mb: float | None = None
    gpu_peak_mb: float | None = None
    ram_peak_mb: float | None = None

    # Label-free accuracy proxies
    silhouette: float | None = None
    cluster_sizes: list[int] = field(default_factory=list)
    target_sim_mean: float | None = None
    target_sim_std: float | None = None
    target_cluster_mean_gap: float | None = None

    # ASR co-residency (optional)
    coresidency_combined_gpu_mb: float | None = None
    coresidency_oom: bool | None = None

    # Threshold calibration (optional): FAR/FRR/EER for a target-speaker gate.
    eer: float | None = None
    eer_threshold: float | None = None
    cal_label_source: str | None = None  # gold | silver | self
    cal_n_target: int | None = None
    cal_n_impostor: int | None = None
    cal_far_frr: list = field(default_factory=list)  # [{threshold, far, frr}]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SpeakerBenchmarkReporter:
    """Collects results and writes JSON / Markdown / console output."""

    def __init__(self, device: str, audio_source: str = "") -> None:
        self.device = device
        self.audio_source = audio_source
        self.results: list[SpeakerBenchmarkResult] = []
        self.created_at = datetime.now().isoformat(timespec="seconds")

    def add_result(self, result: SpeakerBenchmarkResult) -> None:
        self.results.append(result)

    # --- serialization -------------------------------------------------

    def _payload(self) -> dict[str, Any]:
        return {
            "benchmark_type": "speaker",
            "created_at": self.created_at,
            "device": self.device,
            "audio_source": self.audio_source,
            "results": [r.to_dict() for r in self.results],
        }

    def save_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self._payload(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _table_rows(self) -> list[list[Any]]:
        rows: list[list[Any]] = []
        for r in self.results:
            if r.status != "ok":
                rows.append([r.backend, r.status, r.detail] + [""] * 8)
                continue
            rows.append(
                [
                    r.backend,
                    r.status,
                    _fmt(r.load_s, "{:.2f}s"),
                    _fmt(r.embed_latency_ms_p50, "{:.1f}"),
                    _fmt(r.embed_latency_ms_p95, "{:.1f}"),
                    _fmt(r.rtf, "{:.4f}"),
                    _fmt(r.gpu_model_mb, "{:.0f}"),
                    _fmt(r.gpu_peak_mb, "{:.0f}"),
                    _fmt(r.silhouette, "{:.3f}"),
                    _fmt(r.eer, "{:.3f}"),
                    _fmt(r.eer_threshold, "{:.3f}"),
                ]
            )
        return rows

    _HEADERS = [
        "backend",
        "status",
        "load",
        "lat_p50_ms",
        "lat_p95_ms",
        "rtf",
        "gpu_model_mb",
        "gpu_peak_mb",
        "silhouette",
        "eer",
        "eer_thr",
    ]

    def save_markdown(self, path: Path) -> None:
        lines = [
            "# Speaker Embedding Benchmark",
            "",
            f"- Created: {self.created_at}",
            f"- Device: {self.device}",
            f"- Audio source: {self.audio_source or 'n/a'}",
            "",
        ]
        rows = self._table_rows()
        if TABULATE_AVAILABLE:
            lines.append(tabulate(rows, headers=self._HEADERS, tablefmt="github"))
        else:
            lines.append("| " + " | ".join(self._HEADERS) + " |")
            lines.append("|" + "|".join(["---"] * len(self._HEADERS)) + "|")
            for row in rows:
                lines.append("| " + " | ".join(str(c) for c in row) + " |")
        lines.append("")
        lines.append("## Notes")
        lines.append(
            "- `rtf` is embedding-only (lower is faster). `silhouette` in [-1,1]: "
            "higher = better 2-speaker separation."
        )
        lines.append(
            "- TitaNet weights are CC-BY-4.0 (attribution); pyannote is gated; "
            "Sortformer is NVIDIA Open Model License."
        )
        path.write_text("\n".join(lines), encoding="utf-8")

    def to_console(self) -> None:
        rows = self._table_rows()
        print("\n=== Speaker Embedding Benchmark ===")
        print(f"device={self.device} source={self.audio_source or 'n/a'}")
        if TABULATE_AVAILABLE:
            print(tabulate(rows, headers=self._HEADERS, tablefmt="simple"))
        else:
            print("\t".join(self._HEADERS))
            for row in rows:
                print("\t".join(str(c) for c in row))


def _fmt(value: float | None, fmt: str) -> str:
    return fmt.format(value) if value is not None else "-"


__all__ = ["SpeakerBenchmarkResult", "SpeakerBenchmarkReporter"]
