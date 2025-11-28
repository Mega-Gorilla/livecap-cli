"""Progress reporting for benchmark execution.

Provides real-time progress visibility in GitHub Actions and local environments.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["ProgressReporter", "EngineProgress"]


@dataclass
class EngineProgress:
    """Progress data for a single engine benchmark."""

    engine_id: str
    language: str
    files_total: int
    files_completed: int = 0
    wer: float | None = None
    cer: float | None = None
    rtf: float | None = None
    elapsed_s: float = 0.0
    status: str = "pending"  # pending, running, completed, skipped, failed
    # VAD benchmark fields
    vad_name: str | None = None
    vad_rtf: float | None = None
    segments_count: int | None = None
    speech_ratio: float | None = None


@dataclass
class BenchmarkProgress:
    """Overall benchmark progress state."""

    benchmark_type: str  # "asr" or "vad"
    mode: str
    languages: list[str]
    engines_total: int
    engines_completed: int = 0
    start_time: float = field(default_factory=time.time)
    engine_progress: list[EngineProgress] = field(default_factory=list)


class ProgressReporter:
    """Reports benchmark progress to GitHub Actions and local console.

    Features:
    - GitHub Actions: Step Summary 逐次追記、Annotations
    - Local: tqdm-style progress bar、logger output

    Usage:
        reporter = ProgressReporter(
            benchmark_type="asr",
            mode="full",
            languages=["ja", "en"],
            total_engines=10,
        )

        for engine_id in engines:
            reporter.engine_started(engine_id, "ja", files_count=100)
            # ... run benchmark ...
            reporter.engine_completed(engine_id, wer=0.15, cer=0.08, rtf=0.05)

        reporter.benchmark_completed()
    """

    def __init__(
        self,
        benchmark_type: str,
        mode: str,
        languages: list[str],
        total_engines: int,
    ) -> None:
        """Initialize progress reporter.

        Args:
            benchmark_type: Type of benchmark ("asr" or "vad")
            mode: Benchmark mode ("quick", "standard", "full")
            languages: Languages being benchmarked
            total_engines: Total number of engines to benchmark
        """
        self._benchmark_type = benchmark_type
        self._mode = mode
        self._languages = languages
        self._total_engines = total_engines

        # Environment detection
        self._is_github_actions = bool(os.environ.get("GITHUB_ACTIONS"))
        self._step_summary_path = os.environ.get("GITHUB_STEP_SUMMARY")

        # Progress tracking
        self._progress = BenchmarkProgress(
            benchmark_type=benchmark_type,
            mode=mode,
            languages=languages,
            engines_total=total_engines,
        )
        self._current_engine: EngineProgress | None = None
        self._engine_start_time: float = 0.0

        # Initialize Step Summary header if in GitHub Actions
        if self._is_github_actions and self._step_summary_path:
            self._init_step_summary()

    def _init_step_summary(self) -> None:
        """Initialize GitHub Step Summary with header."""
        if self._benchmark_type == "vad":
            header = f"""## VAD Benchmark Progress

**Mode:** {self._mode} | **Languages:** {', '.join(self._languages)} | **Engines:** {self._total_engines}

| # | Engine | VAD | Language | Files | WER | CER | RTF | VAD RTF | Segments | Speech% | Time | Status |
|---|--------|-----|----------|-------|-----|-----|-----|---------|----------|---------|------|--------|
"""
        else:
            header = f"""## {self._benchmark_type.upper()} Benchmark Progress

**Mode:** {self._mode} | **Languages:** {', '.join(self._languages)} | **Engines:** {self._total_engines}

| # | Engine | Language | Files | WER | CER | RTF | Time | Status |
|---|--------|----------|-------|-----|-----|-----|------|--------|
"""
        self._write_step_summary(header)

    def _write_step_summary(self, content: str) -> None:
        """Append content to GitHub Step Summary."""
        if not self._step_summary_path:
            return
        try:
            with open(self._step_summary_path, "a", encoding="utf-8") as f:
                f.write(content)
        except OSError as e:
            logger.debug(f"Failed to write to step summary: {e}")

    def _emit_notice(self, message: str) -> None:
        """Emit GitHub Actions notice annotation."""
        if self._is_github_actions:
            # GitHub Actions workflow command
            print(f"::notice::{message}")
        logger.info(message)

    def _emit_warning(self, message: str) -> None:
        """Emit GitHub Actions warning annotation."""
        if self._is_github_actions:
            print(f"::warning::{message}")
        logger.warning(message)

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m{secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h{mins}m"

    def _get_elapsed(self) -> float:
        """Get elapsed time since benchmark start."""
        return time.time() - self._progress.start_time

    def _estimate_remaining(self) -> float | None:
        """Estimate remaining time based on completed engines."""
        if self._progress.engines_completed == 0:
            return None
        avg_time = self._get_elapsed() / self._progress.engines_completed
        remaining = self._total_engines - self._progress.engines_completed
        return avg_time * remaining

    def benchmark_started(self) -> None:
        """Called when benchmark execution starts."""
        self._progress.start_time = time.time()
        message = (
            f"Starting {self._benchmark_type.upper()} benchmark "
            f"(mode={self._mode}, engines={self._total_engines})"
        )
        self._emit_notice(message)

    def engine_started(
        self,
        engine_id: str,
        language: str,
        files_count: int,
        vad_name: str | None = None,
    ) -> None:
        """Called when an engine benchmark starts.

        Args:
            engine_id: Engine identifier
            language: Language being benchmarked
            files_count: Number of files to process
            vad_name: VAD identifier (for VAD benchmarks)
        """
        self._engine_start_time = time.time()
        self._current_engine = EngineProgress(
            engine_id=engine_id,
            language=language,
            files_total=files_count,
            status="running",
            vad_name=vad_name,
        )

        idx = self._progress.engines_completed + 1
        vad_part = f", vad={vad_name}" if vad_name else ""
        message = (
            f"[{idx}/{self._total_engines}] Starting {engine_id} "
            f"({language}{vad_part}, {files_count} files)"
        )
        logger.info(message)

    def file_completed(self) -> None:
        """Called when a single file is processed."""
        if self._current_engine:
            self._current_engine.files_completed += 1

    def engine_completed(
        self,
        engine_id: str,
        wer: float | None = None,
        cer: float | None = None,
        rtf: float | None = None,
        vad_rtf: float | None = None,
        segments_count: int | None = None,
        speech_ratio: float | None = None,
    ) -> None:
        """Called when an engine benchmark completes.

        Args:
            engine_id: Engine identifier
            wer: Word Error Rate (0.0-1.0)
            cer: Character Error Rate (0.0-1.0)
            rtf: Real-Time Factor (ASR processing)
            vad_rtf: VAD Real-Time Factor (VAD benchmark only)
            segments_count: Number of detected segments (VAD benchmark only)
            speech_ratio: Speech ratio 0.0-1.0 (VAD benchmark only)
        """
        elapsed = time.time() - self._engine_start_time
        self._progress.engines_completed += 1
        idx = self._progress.engines_completed

        if self._current_engine:
            self._current_engine.wer = wer
            self._current_engine.cer = cer
            self._current_engine.rtf = rtf
            self._current_engine.vad_rtf = vad_rtf
            self._current_engine.segments_count = segments_count
            self._current_engine.speech_ratio = speech_ratio
            self._current_engine.elapsed_s = elapsed
            self._current_engine.status = "completed"
            self._progress.engine_progress.append(self._current_engine)

        # Format metrics
        wer_str = f"{wer:.1%}" if wer is not None else "-"
        cer_str = f"{cer:.1%}" if cer is not None else "-"
        rtf_str = f"{rtf:.2f}x" if rtf is not None else "-"
        time_str = self._format_time(elapsed)

        # Console message (use ASCII for Windows cp932 compatibility)
        remaining = self._estimate_remaining()
        remaining_str = f", ETA: {self._format_time(remaining)}" if remaining else ""

        if self._benchmark_type == "vad" and self._current_engine:
            vad_name = self._current_engine.vad_name or "-"
            message = (
                f"[{idx}/{self._total_engines}] [OK] {engine_id}+{vad_name} completed - "
                f"WER: {wer_str}, RTF: {rtf_str}, Time: {time_str}{remaining_str}"
            )
        else:
            message = (
                f"[{idx}/{self._total_engines}] [OK] {engine_id} completed - "
                f"WER: {wer_str}, RTF: {rtf_str}, Time: {time_str}{remaining_str}"
            )
        self._emit_notice(message)

        # GitHub Step Summary row
        if self._is_github_actions and self._current_engine:
            files_str = f"{self._current_engine.files_completed}/{self._current_engine.files_total}"

            if self._benchmark_type == "vad":
                # VAD benchmark row format
                vad_name = self._current_engine.vad_name or "-"
                vad_rtf_str = f"{vad_rtf:.3f}x" if vad_rtf is not None else "-"
                seg_str = str(segments_count) if segments_count is not None else "-"
                speech_str = f"{speech_ratio:.0%}" if speech_ratio is not None else "-"
                row = (
                    f"| {idx} | {engine_id} | {vad_name} | {self._current_engine.language} | "
                    f"{files_str} | {wer_str} | {cer_str} | {rtf_str} | {vad_rtf_str} | "
                    f"{seg_str} | {speech_str} | {time_str} | ✅ |\n"
                )
            else:
                # ASR benchmark row format
                row = (
                    f"| {idx} | {engine_id} | {self._current_engine.language} | "
                    f"{files_str} | {wer_str} | {cer_str} | {rtf_str} | {time_str} | ✅ |\n"
                )
            self._write_step_summary(row)

        self._current_engine = None

    def engine_skipped(
        self,
        engine_id: str,
        reason: str,
        vad_name: str | None = None,
    ) -> None:
        """Called when an engine is skipped.

        Args:
            engine_id: Engine identifier
            reason: Skip reason
            vad_name: VAD identifier (for VAD benchmarks)
        """
        self._progress.engines_completed += 1
        idx = self._progress.engines_completed

        # Console message (use ASCII for Windows cp932 compatibility)
        if vad_name:
            message = f"[{idx}/{self._total_engines}] [SKIP] {engine_id}+{vad_name} skipped: {reason}"
        else:
            message = f"[{idx}/{self._total_engines}] [SKIP] {engine_id} skipped: {reason}"
        self._emit_warning(message)

        # GitHub Step Summary row
        if self._is_github_actions:
            if self._benchmark_type == "vad":
                vad_str = vad_name or "-"
                row = f"| {idx} | {engine_id} | {vad_str} | - | - | - | - | - | - | - | - | - | ⏭️ |\n"
            else:
                row = f"| {idx} | {engine_id} | - | - | - | - | - | - | ⏭️ |\n"
            self._write_step_summary(row)

    def engine_failed(
        self,
        engine_id: str,
        error: str,
        vad_name: str | None = None,
    ) -> None:
        """Called when an engine benchmark fails.

        Args:
            engine_id: Engine identifier
            error: Error message
            vad_name: VAD identifier (for VAD benchmarks)
        """
        elapsed = time.time() - self._engine_start_time
        self._progress.engines_completed += 1
        idx = self._progress.engines_completed

        # Console message (use ASCII for Windows cp932 compatibility)
        if vad_name:
            message = f"[{idx}/{self._total_engines}] [FAIL] {engine_id}+{vad_name} failed: {error}"
        else:
            message = f"[{idx}/{self._total_engines}] [FAIL] {engine_id} failed: {error}"
        self._emit_warning(message)

        # GitHub Step Summary row
        if self._is_github_actions:
            time_str = self._format_time(elapsed)
            if self._benchmark_type == "vad":
                vad_str = vad_name or "-"
                row = f"| {idx} | {engine_id} | {vad_str} | - | - | - | - | - | - | - | - | {time_str} | ❌ |\n"
            else:
                row = f"| {idx} | {engine_id} | - | - | - | - | - | {time_str} | ❌ |\n"
            self._write_step_summary(row)

        self._current_engine = None

    def benchmark_completed(self) -> None:
        """Called when benchmark execution completes."""
        total_elapsed = self._get_elapsed()
        completed = self._progress.engines_completed

        message = (
            f"Benchmark completed: {completed}/{self._total_engines} engines, "
            f"Total time: {self._format_time(total_elapsed)}"
        )
        self._emit_notice(message)

        # GitHub Step Summary footer
        if self._is_github_actions:
            footer = f"\n**Total time:** {self._format_time(total_elapsed)}\n"
            self._write_step_summary(footer)
