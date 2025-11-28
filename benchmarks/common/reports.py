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

    # VAD info (for VAD benchmark)
    vad: str | None = None
    vad_config: dict | None = None  # VAD parameters for reproducibility
    vad_rtf: float | None = None  # VAD processing RTF
    segments_count: int | None = None  # Number of detected segments
    avg_segment_duration_s: float | None = None  # Average segment duration
    speech_ratio: float | None = None  # Speech ratio (0.0-1.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
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
        }

        # Add VAD info if present
        if self.vad is not None:
            result["vad"] = {
                "name": self.vad,
                "config": self.vad_config,
                "vad_rtf": self.vad_rtf,
                "segments_count": self.segments_count,
                "avg_segment_duration_s": self.avg_segment_duration_s,
                "speech_ratio": self.speech_ratio,
            }

        return result


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
        self.skipped: list[str] = []
        self.timestamp = datetime.utcnow().isoformat() + "Z"

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result."""
        self.results.append(result)

    def add_results(self, results: list[BenchmarkResult]) -> None:
        """Add multiple benchmark results."""
        self.results.extend(results)

    def add_skipped(self, reason: str) -> None:
        """Record a skipped item.

        Args:
            reason: Description of what was skipped and why
        """
        self.skipped.append(reason)

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
        """Generate Markdown report with aggregated statistics.

        Shows average WER/CER/RTF per engine×language (ASR) or
        engine×vad×language (VAD) combination.

        Returns:
            Markdown string
        """
        if self.benchmark_type == "vad":
            return self._to_markdown_vad()
        return self._to_markdown_asr()

    def _to_markdown_asr(self) -> str:
        """Generate Markdown report for ASR benchmark."""
        lines = [
            f"# ASR Benchmark Report",
            "",
            f"**Date:** {self.timestamp}",
            f"**Mode:** {self.mode}",
            f"**Device:** {self.device}",
            "",
        ]

        # Compute aggregated stats
        aggregated = self._aggregate_by_engine_language()

        # Group by language for display
        by_language: dict[str, list[dict[str, Any]]] = {}
        for (engine, lang), stats in aggregated.items():
            if lang not in by_language:
                by_language[lang] = []
            by_language[lang].append({"engine": engine, **stats})

        # Results by language
        lines.append("## Results by Language")
        lines.append("")

        for lang in sorted(by_language.keys()):
            engine_stats = by_language[lang]
            lines.append(f"### {lang.upper()}")
            lines.append("")

            # Build table
            headers = ["Engine", "CER", "WER", "RTF", "VRAM", "Files"]
            rows = []
            for stats in sorted(engine_stats, key=lambda x: x.get("cer_mean", float("inf"))):
                row = [
                    stats["engine"],
                    f"{stats['cer_mean']:.1%}" if stats.get("cer_mean") is not None else "-",
                    f"{stats['wer_mean']:.1%}" if stats.get("wer_mean") is not None else "-",
                    f"{stats['rtf_mean']:.3f}" if stats.get("rtf_mean") is not None else "-",
                    f"{stats['gpu_memory_peak_mb']:.0f}MB" if stats.get("gpu_memory_peak_mb") else "-",
                    str(stats.get("file_count", 0)),
                ]
                rows.append(row)

            if TABULATE_AVAILABLE:
                lines.append(tabulate(rows, headers=headers, tablefmt="pipe"))
            else:
                lines.append("| " + " | ".join(headers) + " |")
                lines.append("|" + "|".join(["---"] * len(headers)) + "|")
                for row in rows:
                    lines.append("| " + " | ".join(str(c) for c in row) + " |")

            # Best for this language
            if engine_stats:
                metric_key = "cer_mean" if lang == "ja" else "wer_mean"
                metric_name = "CER" if lang == "ja" else "WER"
                valid = [s for s in engine_stats if s.get(metric_key) is not None]
                if valid:
                    best = min(valid, key=lambda x: x[metric_key])
                    lines.append("")
                    lines.append(f"**Best {metric_name}:** {best['engine']} ({best[metric_key]:.1%})")

            lines.append("")

        # Overall summary
        lines.append("## Summary")
        lines.append("")

        total_files = sum(s.get("file_count", 0) for s in aggregated.values())
        total_duration = sum(s.get("total_duration", 0) for s in aggregated.values())
        lines.append(f"- **Total files:** {total_files}")
        lines.append(f"- **Total duration:** {total_duration:.1f} sec")

        # Skipped info
        if self.skipped:
            lines.append(f"- **Skipped:** {len(self.skipped)}")
            lines.append("")
            lines.append("### Skipped Items")
            lines.append("")
            for item in self.skipped:
                lines.append(f"- {item}")
        lines.append("")

        return "\n".join(lines)

    def _to_markdown_vad(self) -> str:
        """Generate Markdown report for VAD benchmark."""
        lines = [
            f"# VAD Benchmark Report",
            "",
            f"**Date:** {self.timestamp}",
            f"**Mode:** {self.mode}",
            f"**Device:** {self.device}",
            "",
        ]

        # Compute aggregated stats by engine×vad×language
        aggregated = self._aggregate_by_engine_vad_language()

        # Group by language for display
        by_language: dict[str, list[dict[str, Any]]] = {}
        for (engine, vad, lang), stats in aggregated.items():
            if lang not in by_language:
                by_language[lang] = []
            by_language[lang].append({"engine": engine, "vad": vad, **stats})

        # Results by language
        lines.append("## Results by Language")
        lines.append("")

        for lang in sorted(by_language.keys()):
            combo_stats = by_language[lang]
            lines.append(f"### {lang.upper()}")
            lines.append("")

            # Build table with VAD columns
            headers = ["Engine", "VAD", "CER", "WER", "RTF", "VAD RTF", "Seg", "Speech%", "Files"]
            rows = []
            for stats in sorted(combo_stats, key=lambda x: x.get("cer_mean", float("inf"))):
                row = [
                    stats["engine"],
                    stats["vad"],
                    f"{stats['cer_mean']:.1%}" if stats.get("cer_mean") is not None else "-",
                    f"{stats['wer_mean']:.1%}" if stats.get("wer_mean") is not None else "-",
                    f"{stats['rtf_mean']:.3f}" if stats.get("rtf_mean") is not None else "-",
                    f"{stats['vad_rtf_mean']:.3f}" if stats.get("vad_rtf_mean") is not None else "-",
                    f"{stats['segments_mean']:.0f}" if stats.get("segments_mean") is not None else "-",
                    f"{stats['speech_ratio_mean']:.0%}" if stats.get("speech_ratio_mean") is not None else "-",
                    str(stats.get("file_count", 0)),
                ]
                rows.append(row)

            if TABULATE_AVAILABLE:
                lines.append(tabulate(rows, headers=headers, tablefmt="pipe"))
            else:
                lines.append("| " + " | ".join(headers) + " |")
                lines.append("|" + "|".join(["---"] * len(headers)) + "|")
                for row in rows:
                    lines.append("| " + " | ".join(str(c) for c in row) + " |")

            # Best for this language (by CER for ja, WER for others)
            if combo_stats:
                metric_key = "cer_mean" if lang == "ja" else "wer_mean"
                metric_name = "CER" if lang == "ja" else "WER"
                valid = [s for s in combo_stats if s.get(metric_key) is not None]
                if valid:
                    best = min(valid, key=lambda x: x[metric_key])
                    lines.append("")
                    lines.append(
                        f"**Best {metric_name}:** {best['engine']}+{best['vad']} "
                        f"({best[metric_key]:.1%})"
                    )

            lines.append("")

        # Overall summary
        lines.append("## Summary")
        lines.append("")

        total_files = sum(s.get("file_count", 0) for s in aggregated.values())
        total_duration = sum(s.get("total_duration", 0) for s in aggregated.values())
        lines.append(f"- **Total combinations:** {len(aggregated)}")
        lines.append(f"- **Total files:** {total_files}")
        lines.append(f"- **Total duration:** {total_duration:.1f} sec")

        # Skipped info
        if self.skipped:
            lines.append(f"- **Skipped:** {len(self.skipped)}")
            lines.append("")
            lines.append("### Skipped Items")
            lines.append("")
            for item in self.skipped:
                lines.append(f"- {item}")
        lines.append("")

        return "\n".join(lines)

    def _aggregate_by_engine_language(self) -> dict[tuple[str, str], dict[str, Any]]:
        """Aggregate results by engine×language.

        Returns:
            Dictionary mapping (engine, language) to aggregated statistics.
        """
        from statistics import mean

        groups: dict[tuple[str, str], list[BenchmarkResult]] = {}
        for r in self.results:
            key = (r.engine, r.language)
            if key not in groups:
                groups[key] = []
            groups[key].append(r)

        aggregated: dict[tuple[str, str], dict[str, Any]] = {}
        for key, results in groups.items():
            wers = [r.wer for r in results if r.wer is not None]
            cers = [r.cer for r in results if r.cer is not None]
            rtfs = [r.rtf for r in results if r.rtf is not None]
            durations = [r.audio_duration_s for r in results if r.audio_duration_s is not None]

            # Use max GPU peak memory across all files (captures worst-case VRAM usage)
            gpu_mems = [r.gpu_memory_peak_mb for r in results if r.gpu_memory_peak_mb is not None]
            gpu_mem = max(gpu_mems) if gpu_mems else None

            aggregated[key] = {
                "wer_mean": mean(wers) if wers else None,
                "cer_mean": mean(cers) if cers else None,
                "rtf_mean": mean(rtfs) if rtfs else None,
                "gpu_memory_peak_mb": gpu_mem,
                "file_count": len(results),
                "total_duration": sum(durations) if durations else 0,
            }

        return aggregated

    def _aggregate_by_engine_vad_language(
        self,
    ) -> dict[tuple[str, str, str], dict[str, Any]]:
        """Aggregate results by engine×vad×language for VAD benchmarks.

        Returns:
            Dictionary mapping (engine, vad, language) to aggregated statistics.
        """
        from statistics import mean

        groups: dict[tuple[str, str, str], list[BenchmarkResult]] = {}
        for r in self.results:
            vad = r.vad or "unknown"
            key = (r.engine, vad, r.language)
            if key not in groups:
                groups[key] = []
            groups[key].append(r)

        aggregated: dict[tuple[str, str, str], dict[str, Any]] = {}
        for key, results in groups.items():
            wers = [r.wer for r in results if r.wer is not None]
            cers = [r.cer for r in results if r.cer is not None]
            rtfs = [r.rtf for r in results if r.rtf is not None]
            vad_rtfs = [r.vad_rtf for r in results if r.vad_rtf is not None]
            segments = [r.segments_count for r in results if r.segments_count is not None]
            speech_ratios = [r.speech_ratio for r in results if r.speech_ratio is not None]
            durations = [r.audio_duration_s for r in results if r.audio_duration_s is not None]

            # Use max GPU peak memory across all files
            gpu_mems = [r.gpu_memory_peak_mb for r in results if r.gpu_memory_peak_mb is not None]
            gpu_mem = max(gpu_mems) if gpu_mems else None

            aggregated[key] = {
                "wer_mean": mean(wers) if wers else None,
                "cer_mean": mean(cers) if cers else None,
                "rtf_mean": mean(rtfs) if rtfs else None,
                "vad_rtf_mean": mean(vad_rtfs) if vad_rtfs else None,
                "segments_mean": mean(segments) if segments else None,
                "speech_ratio_mean": mean(speech_ratios) if speech_ratios else None,
                "gpu_memory_peak_mb": gpu_mem,
                "file_count": len(results),
                "total_duration": sum(durations) if durations else 0,
            }

        return aggregated

    def to_console(self) -> None:
        """Print report to console.

        Uses tabulate for nice formatting if available.
        """
        if self.benchmark_type == "vad":
            self._to_console_vad()
        else:
            self._to_console_asr()

    def _to_console_asr(self) -> None:
        """Print ASR report to console."""
        print(f"\n=== ASR Benchmark Results ===")
        print(f"Mode: {self.mode}")
        print(f"Device: {self.device}")
        print()

        # Use aggregated stats
        aggregated = self._aggregate_by_engine_language()
        by_language: dict[str, list[dict[str, Any]]] = {}
        for (engine, lang), stats in aggregated.items():
            if lang not in by_language:
                by_language[lang] = []
            by_language[lang].append({"engine": engine, **stats})

        for lang in sorted(by_language.keys()):
            print(f"--- {lang.upper()} ---")
            engine_stats = by_language[lang]

            headers = ["Engine", "CER", "WER", "RTF", "VRAM", "Files"]
            rows = []
            for stats in sorted(engine_stats, key=lambda x: x.get("cer_mean", float("inf"))):
                row = [
                    stats["engine"],
                    f"{stats['cer_mean']:.1%}" if stats.get("cer_mean") is not None else "-",
                    f"{stats['wer_mean']:.1%}" if stats.get("wer_mean") is not None else "-",
                    f"{stats['rtf_mean']:.3f}" if stats.get("rtf_mean") is not None else "-",
                    f"{stats['gpu_memory_peak_mb']:.0f}MB" if stats.get("gpu_memory_peak_mb") else "-",
                    str(stats.get("file_count", 0)),
                ]
                rows.append(row)

            if TABULATE_AVAILABLE:
                print(tabulate(rows, headers=headers, tablefmt="simple"))
            else:
                print(" | ".join(headers))
                print("-" * 70)
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

    def _to_console_vad(self) -> None:
        """Print VAD report to console."""
        print(f"\n=== VAD Benchmark Results ===")
        print(f"Mode: {self.mode}")
        print(f"Device: {self.device}")
        print()

        # Use aggregated stats by engine×vad×language
        aggregated = self._aggregate_by_engine_vad_language()
        by_language: dict[str, list[dict[str, Any]]] = {}
        for (engine, vad, lang), stats in aggregated.items():
            if lang not in by_language:
                by_language[lang] = []
            by_language[lang].append({"engine": engine, "vad": vad, **stats})

        for lang in sorted(by_language.keys()):
            print(f"--- {lang.upper()} ---")
            combo_stats = by_language[lang]

            headers = ["Engine", "VAD", "CER", "WER", "RTF", "VAD RTF", "Seg", "Speech%", "Files"]
            rows = []
            for stats in sorted(combo_stats, key=lambda x: x.get("cer_mean", float("inf"))):
                row = [
                    stats["engine"],
                    stats["vad"],
                    f"{stats['cer_mean']:.1%}" if stats.get("cer_mean") is not None else "-",
                    f"{stats['wer_mean']:.1%}" if stats.get("wer_mean") is not None else "-",
                    f"{stats['rtf_mean']:.3f}" if stats.get("rtf_mean") is not None else "-",
                    f"{stats['vad_rtf_mean']:.3f}" if stats.get("vad_rtf_mean") is not None else "-",
                    f"{stats['segments_mean']:.0f}" if stats.get("segments_mean") is not None else "-",
                    f"{stats['speech_ratio_mean']:.0%}" if stats.get("speech_ratio_mean") is not None else "-",
                    str(stats.get("file_count", 0)),
                ]
                rows.append(row)

            if TABULATE_AVAILABLE:
                print(tabulate(rows, headers=headers, tablefmt="simple"))
            else:
                print(" | ".join(headers))
                print("-" * 100)
                for row in rows:
                    print(" | ".join(str(c) for c in row))

            # Best for this language
            if combo_stats:
                metric_key = "cer_mean" if lang == "ja" else "wer_mean"
                metric_name = "CER" if lang == "ja" else "WER"
                valid = [s for s in combo_stats if s.get(metric_key) is not None]
                if valid:
                    best = min(valid, key=lambda x: x[metric_key])
                    print(f"\nBest {metric_name}: {best['engine']}+{best['vad']} ({best[metric_key]:.1%})")

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

    def to_csv(self, engine: str, language: str) -> str:
        """Generate CSV content for a specific engine and language.

        CSV columns: file_id,reference,transcript,cer,wer,rtf,duration_sec

        Args:
            engine: Engine identifier
            language: Language code

        Returns:
            CSV string with header and data rows
        """
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "file_id", "reference", "transcript", "cer", "wer", "rtf", "duration_sec"
        ])

        # Filter results for this engine+language
        for r in self.results:
            if r.engine == engine and r.language == language:
                writer.writerow([
                    r.audio_file,
                    r.reference,
                    r.transcript,
                    f"{r.cer:.4f}" if r.cer is not None else "",
                    f"{r.wer:.4f}" if r.wer is not None else "",
                    f"{r.rtf:.4f}" if r.rtf is not None else "",
                    f"{r.audio_duration_s:.2f}" if r.audio_duration_s is not None else "",
                ])

        return output.getvalue()

    def save_csv(self, directory: Path | str, engine: str, language: str) -> Path:
        """Save CSV file for a specific engine and language.

        Args:
            directory: Output directory (will create raw/ subdirectory)
            engine: Engine identifier
            language: Language code

        Returns:
            Path to the created CSV file
        """
        directory = Path(directory)
        raw_dir = directory / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{engine}_{language}.csv"
        path = raw_dir / filename
        path.write_text(self.to_csv(engine, language), encoding="utf-8")
        return path

    def save_all_csv(self, directory: Path | str) -> list[Path]:
        """Save CSV files for all engine+language combinations.

        Args:
            directory: Output directory

        Returns:
            List of paths to created CSV files
        """
        paths = []
        for (engine, language) in self._get_engine_language_pairs():
            path = self.save_csv(directory, engine, language)
            paths.append(path)
        return paths

    def _get_engine_language_pairs(self) -> list[tuple[str, str]]:
        """Get unique (engine, language) pairs from results."""
        pairs: set[tuple[str, str]] = set()
        for r in self.results:
            pairs.add((r.engine, r.language))
        return sorted(pairs)

    def _group_by_language(self) -> dict[str, list[BenchmarkResult]]:
        """Group results by language."""
        by_lang: dict[str, list[BenchmarkResult]] = {}
        for r in self.results:
            if r.language not in by_lang:
                by_lang[r.language] = []
            by_lang[r.language].append(r)
        return by_lang

    def _generate_summary(self) -> dict[str, Any]:
        """Generate summary statistics from aggregated per-engine results.

        Uses _aggregate_by_engine_language() to compute mean metrics per engine,
        then finds the best engine for each language based on those aggregates.
        """
        if not self.results:
            return {}

        summary: dict[str, Any] = {}

        # Get aggregated stats per engine×language
        aggregated = self._aggregate_by_engine_language()

        # Best by language (using aggregated means, not per-file results)
        best_by_lang: dict[str, dict[str, Any]] = {}

        # Group aggregated results by language
        by_lang: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for (engine, lang), stats in aggregated.items():
            if lang not in by_lang:
                by_lang[lang] = []
            by_lang[lang].append((engine, stats))

        for lang, engine_stats in by_lang.items():
            # For Japanese, use CER; for others, use WER
            if lang == "ja":
                valid = [(e, s) for e, s in engine_stats if s.get("cer_mean") is not None]
                if valid:
                    best_engine, best_stats = min(
                        valid, key=lambda x: x[1]["cer_mean"] or float("inf")
                    )
                    best_by_lang[lang] = {
                        "engine": best_engine,
                        "cer": best_stats["cer_mean"],
                    }
            else:
                valid = [(e, s) for e, s in engine_stats if s.get("wer_mean") is not None]
                if valid:
                    best_engine, best_stats = min(
                        valid, key=lambda x: x[1]["wer_mean"] or float("inf")
                    )
                    best_by_lang[lang] = {
                        "engine": best_engine,
                        "wer": best_stats["wer_mean"],
                    }

        if best_by_lang:
            summary["best_by_language"] = best_by_lang

        # Fastest (lowest mean RTF across all engine×language)
        valid_rtf = [
            (engine, stats) for (engine, _), stats in aggregated.items()
            if stats.get("rtf_mean") is not None
        ]
        if valid_rtf:
            fastest_engine, fastest_stats = min(
                valid_rtf, key=lambda x: x[1]["rtf_mean"] or float("inf")
            )
            summary["fastest"] = {
                "engine": fastest_engine,
                "rtf": fastest_stats["rtf_mean"],
            }

        # Lowest VRAM (using aggregated peak, not per-file)
        valid_vram = [
            (engine, stats) for (engine, _), stats in aggregated.items()
            if stats.get("gpu_memory_peak_mb") is not None
        ]
        if valid_vram:
            lowest_engine, lowest_stats = min(
                valid_vram, key=lambda x: x[1]["gpu_memory_peak_mb"] or float("inf")
            )
            summary["lowest_vram"] = {
                "engine": lowest_engine,
                "gpu_memory_peak_mb": lowest_stats["gpu_memory_peak_mb"],
            }

        return summary
