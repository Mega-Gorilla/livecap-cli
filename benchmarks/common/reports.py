"""Report generation for benchmarks.

Provides:
- BenchmarkReporter: Generate reports in various formats (JSON, Markdown, Console)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Optional imports
try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False


__all__ = ["BenchmarkReporter", "BenchmarkResult", "BenchmarkSummary"]


@dataclass
class BenchmarkResult:
    """Single benchmark result."""

    engine: str
    language: str
    audio_file: str

    # Transcription
    transcript: str
    reference: str

    # Metrics
    wer: float | None = None
    cer: float | None = None
    rtf: float | None = None
    audio_duration_s: float | None = None
    processing_time_s: float | None = None

    # Memory
    memory_peak_mb: float | None = None
    gpu_memory_model_mb: float | None = None
    gpu_memory_peak_mb: float | None = None

    # Optional VAD info
    vad: str | None = None
    segments: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "engine": self.engine,
            "language": self.language,
            "audio_file": self.audio_file,
            "transcript": self.transcript,
            "reference": self.reference,
            "metrics": {
                "wer": self.wer,
                "cer": self.cer,
                "rtf": self.rtf,
                "audio_duration_s": self.audio_duration_s,
                "processing_time_s": self.processing_time_s,
                "memory_peak_mb": self.memory_peak_mb,
                "gpu_memory_model_mb": self.gpu_memory_model_mb,
                "gpu_memory_peak_mb": self.gpu_memory_peak_mb,
            },
            "vad": self.vad,
            "segments": self.segments,
        }


@dataclass
class BenchmarkSummary:
    """Summary of benchmark results."""

    best_by_language: dict[str, dict[str, Any]] = field(default_factory=dict)
    fastest: dict[str, Any] | None = None
    lowest_vram: dict[str, Any] | None = None


class BenchmarkReporter:
    """Generate benchmark reports in various formats.

    Usage:
        reporter = BenchmarkReporter()
        reporter.add_result(result)

        # Output formats
        print(reporter.to_json())
        print(reporter.to_markdown())
        reporter.to_console()
    """

    def __init__(
        self,
        benchmark_type: str = "asr",
        mode: str = "standard",
        device: str = "cuda",
    ) -> None:
        """Initialize reporter.

        Args:
            benchmark_type: Type of benchmark ('asr', 'vad', 'both')
            mode: Execution mode ('quick', 'standard', 'full')
            device: Device used ('cuda', 'cpu')
        """
        self.benchmark_type = benchmark_type
        self.mode = mode
        self.device = device
        self.results: list[BenchmarkResult] = []
        self.timestamp = datetime.utcnow().isoformat() + "Z"

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result."""
        self.results.append(result)

    def add_results(self, results: list[BenchmarkResult]) -> None:
        """Add multiple benchmark results."""
        self.results.extend(results)

    def to_json(self, indent: int = 2) -> str:
        """Generate JSON report.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string
        """
        report = {
            "metadata": {
                "timestamp": self.timestamp,
                "device": self.device,
                "benchmark_type": self.benchmark_type,
                "mode": self.mode,
            },
            "results": [r.to_dict() for r in self.results],
            "summary": self._generate_summary(),
        }
        return json.dumps(report, indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        """Generate Markdown report.

        Returns:
            Markdown string
        """
        lines = [
            f"# Benchmark Results",
            "",
            f"**Type:** {self.benchmark_type}",
            f"**Mode:** {self.mode}",
            f"**Device:** {self.device}",
            f"**Timestamp:** {self.timestamp}",
            "",
        ]

        # Group results by language
        by_language = self._group_by_language()

        for lang, results in by_language.items():
            lines.append(f"## {lang.upper()} Results")
            lines.append("")

            # Build table
            headers = ["Engine", "WER", "CER", "RTF", "VRAM (Peak)"]
            rows = []
            for r in results:
                row = [
                    r.engine,
                    f"{r.wer:.1%}" if r.wer is not None else "-",
                    f"{r.cer:.1%}" if r.cer is not None else "-",
                    f"{r.rtf:.3f}" if r.rtf is not None else "-",
                    f"{r.gpu_memory_peak_mb:.0f} MB" if r.gpu_memory_peak_mb else "-",
                ]
                rows.append(row)

            if TABULATE_AVAILABLE:
                lines.append(tabulate(rows, headers=headers, tablefmt="pipe"))
            else:
                # Simple fallback
                lines.append("| " + " | ".join(headers) + " |")
                lines.append("|" + "|".join(["---"] * len(headers)) + "|")
                for row in rows:
                    lines.append("| " + " | ".join(str(c) for c in row) + " |")

            lines.append("")

        # Summary
        summary = self._generate_summary()
        if summary:
            lines.append("## Summary")
            lines.append("")
            if summary.get("best_by_language"):
                for lang, best in summary["best_by_language"].items():
                    lines.append(f"- **Best for {lang}:** {best.get('engine', 'N/A')}")
            if summary.get("fastest"):
                lines.append(f"- **Fastest:** {summary['fastest'].get('engine', 'N/A')}")
            if summary.get("lowest_vram"):
                lines.append(f"- **Lowest VRAM:** {summary['lowest_vram'].get('engine', 'N/A')}")
            lines.append("")

        return "\n".join(lines)

    def to_console(self) -> None:
        """Print report to console.

        Uses tabulate for nice formatting if available.
        """
        print(f"\n=== Benchmark Results ===")
        print(f"Type: {self.benchmark_type}")
        print(f"Mode: {self.mode}")
        print(f"Device: {self.device}")
        print()

        by_language = self._group_by_language()

        for lang, results in by_language.items():
            print(f"--- {lang.upper()} ---")

            headers = ["Engine", "WER", "CER", "RTF", "VRAM"]
            rows = []
            for r in results:
                row = [
                    r.engine,
                    f"{r.wer:.1%}" if r.wer is not None else "-",
                    f"{r.cer:.1%}" if r.cer is not None else "-",
                    f"{r.rtf:.3f}" if r.rtf is not None else "-",
                    f"{r.gpu_memory_peak_mb:.0f}MB" if r.gpu_memory_peak_mb else "-",
                ]
                rows.append(row)

            if TABULATE_AVAILABLE:
                print(tabulate(rows, headers=headers, tablefmt="simple"))
            else:
                # Simple fallback
                print(" | ".join(headers))
                print("-" * 60)
                for row in rows:
                    print(" | ".join(str(c) for c in row))

            print()

        # Summary
        summary = self._generate_summary()
        if summary:
            print("=== Summary ===")
            if summary.get("best_by_language"):
                for lang, best in summary["best_by_language"].items():
                    metric = "CER" if lang == "ja" else "WER"
                    value = best.get("cer" if lang == "ja" else "wer", "N/A")
                    if isinstance(value, float):
                        value = f"{value:.1%}"
                    print(f"Best for {lang}: {best.get('engine', 'N/A')} ({metric}: {value})")
            if summary.get("fastest"):
                rtf = summary["fastest"].get("rtf", "N/A")
                if isinstance(rtf, float):
                    rtf = f"{rtf:.3f}"
                print(f"Fastest: {summary['fastest'].get('engine', 'N/A')} (RTF: {rtf})")
            if summary.get("lowest_vram"):
                vram = summary["lowest_vram"].get("gpu_memory_peak_mb", "N/A")
                if isinstance(vram, float):
                    vram = f"{vram:.0f} MB"
                print(f"Lowest VRAM: {summary['lowest_vram'].get('engine', 'N/A')} ({vram})")
            print()

    def save_json(self, path: Path | str) -> None:
        """Save JSON report to file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.write_text(self.to_json(), encoding="utf-8")

    def save_markdown(self, path: Path | str) -> None:
        """Save Markdown report to file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.write_text(self.to_markdown(), encoding="utf-8")

    def _group_by_language(self) -> dict[str, list[BenchmarkResult]]:
        """Group results by language."""
        by_lang: dict[str, list[BenchmarkResult]] = {}
        for r in self.results:
            if r.language not in by_lang:
                by_lang[r.language] = []
            by_lang[r.language].append(r)
        return by_lang

    def _generate_summary(self) -> dict[str, Any]:
        """Generate summary statistics."""
        if not self.results:
            return {}

        summary: dict[str, Any] = {}

        # Best by language
        by_lang = self._group_by_language()
        best_by_lang: dict[str, dict[str, Any]] = {}

        for lang, results in by_lang.items():
            # For Japanese, use CER; for others, use WER
            if lang == "ja":
                valid = [r for r in results if r.cer is not None]
                if valid:
                    best = min(valid, key=lambda r: r.cer or float("inf"))
                    best_by_lang[lang] = {
                        "engine": best.engine,
                        "cer": best.cer,
                    }
            else:
                valid = [r for r in results if r.wer is not None]
                if valid:
                    best = min(valid, key=lambda r: r.wer or float("inf"))
                    best_by_lang[lang] = {
                        "engine": best.engine,
                        "wer": best.wer,
                    }

        if best_by_lang:
            summary["best_by_language"] = best_by_lang

        # Fastest (lowest RTF)
        valid_rtf = [r for r in self.results if r.rtf is not None]
        if valid_rtf:
            fastest = min(valid_rtf, key=lambda r: r.rtf or float("inf"))
            summary["fastest"] = {
                "engine": fastest.engine,
                "rtf": fastest.rtf,
            }

        # Lowest VRAM
        valid_vram = [r for r in self.results if r.gpu_memory_peak_mb is not None]
        if valid_vram:
            lowest = min(valid_vram, key=lambda r: r.gpu_memory_peak_mb or float("inf"))
            summary["lowest_vram"] = {
                "engine": lowest.engine,
                "gpu_memory_peak_mb": lowest.gpu_memory_peak_mb,
            }

        return summary
