"""ASR Benchmark Runner.

Provides ASRBenchmarkRunner for evaluating ASR engine performance.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Optional tqdm for progress display
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from benchmarks.common import (
    AudioFile,
    BenchmarkEngineManager,
    BenchmarkReporter,
    BenchmarkResult,
    Dataset,
    DatasetManager,
    GPUMemoryTracker,
    ProgressReporter,
    TranscriptionEngine,
    calculate_cer,
    calculate_rtf,
    calculate_wer,
)

__all__ = ["ASRBenchmarkRunner", "ASRBenchmarkConfig"]

logger = logging.getLogger(__name__)


# Default engines for debug/quick/standard modes
DEFAULT_MODE_ENGINES = {
    "ja": ["parakeet_ja", "whispers2t_large_v3"],
    "en": ["parakeet", "whispers2t_large_v3"],
}


@dataclass
class ASRBenchmarkConfig:
    """Configuration for ASR benchmark execution."""

    # Execution mode
    mode: str = "quick"  # debug, quick, standard, full

    # Target languages
    languages: list[str] = field(default_factory=lambda: ["ja", "en"])

    # Target engines (None = use mode defaults)
    engines: list[str] | None = None

    # Number of runs per file for RTF measurement
    runs: int = 1

    # Device
    device: str = "cuda"

    # Output directory (None = project_root/benchmark_results/)
    output_dir: Path | None = None

    def get_engines_for_language(self, language: str) -> list[str]:
        """Get engines to benchmark for a language.

        Args:
            language: Language code

        Returns:
            List of engine IDs
        """
        if self.engines:
            return self.engines

        if self.mode in ("debug", "quick", "standard"):
            return DEFAULT_MODE_ENGINES.get(language, [])

        # full: use all engines for this language
        return BenchmarkEngineManager.get_engines_for_language(language)


class ASRBenchmarkRunner:
    """Runs ASR benchmarks across engines and languages.

    Usage:
        config = ASRBenchmarkConfig(mode="quick")
        runner = ASRBenchmarkRunner(config)
        output_dir = runner.run()
        print(f"Results saved to: {output_dir}")
    """

    def __init__(self, config: ASRBenchmarkConfig) -> None:
        """Initialize benchmark runner.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.dataset_manager = DatasetManager()
        self.engine_manager = BenchmarkEngineManager()
        self.gpu_tracker = GPUMemoryTracker()
        self.reporter = BenchmarkReporter(
            benchmark_type="asr",
            mode=config.mode,
            device=config.device,
        )

        # Output directory
        if config.output_dir:
            self.output_dir = config.output_dir
        else:
            project_root = Path(__file__).resolve().parents[2]
            self.output_dir = project_root / "benchmark_results"

        # Progress reporter (initialized in run())
        self.progress: ProgressReporter | None = None

    def _count_total_engines(self) -> int:
        """Count total engines to benchmark across all languages."""
        total = 0
        for language in self.config.languages:
            engines = self.config.get_engines_for_language(language)
            total += len(engines)
        return total

    def run(self) -> Path:
        """Execute the benchmark.

        Returns:
            Path to the output directory
        """
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = self.output_dir / f"{timestamp}_{self.config.mode}"
        result_dir.mkdir(parents=True, exist_ok=True)

        # Initialize progress reporter
        total_engines = self._count_total_engines()
        self.progress = ProgressReporter(
            benchmark_type="asr",
            mode=self.config.mode,
            languages=self.config.languages,
            total_engines=total_engines,
        )

        logger.info(f"Starting ASR benchmark (mode={self.config.mode})")
        logger.info(f"Output directory: {result_dir}")
        self.progress.benchmark_started()

        try:
            # Run benchmarks for each language
            for language in self.config.languages:
                self._benchmark_language(language)
        finally:
            # Clean up engines
            self.engine_manager.clear_cache()

        # Save results
        self._save_results(result_dir)

        # Report completion
        if self.progress:
            self.progress.benchmark_completed()

        logger.info(f"Benchmark complete. Results saved to: {result_dir}")
        return result_dir

    def _benchmark_language(self, language: str) -> None:
        """Benchmark all engines for a language.

        Args:
            language: Language code
        """
        logger.info(f"Benchmarking language: {language}")

        # Load dataset
        try:
            dataset = self.dataset_manager.get_dataset(language, mode=self.config.mode)
        except Exception as e:
            reason = f"{language}: Failed to load dataset - {e}"
            logger.warning(reason)
            self.reporter.add_skipped(reason)
            return

        if len(dataset) == 0:
            reason = f"{language}: No audio files found"
            logger.warning(reason)
            self.reporter.add_skipped(reason)
            return

        logger.info(f"Loaded {len(dataset)} files for {language}")

        # Get engines for this language
        engines = self.config.get_engines_for_language(language)

        for engine_id in engines:
            # Check compatibility
            if not dataset.is_compatible_with_engine(engine_id):
                reason = f"Does not support {language}"
                logger.debug(f"{engine_id}: {reason}")
                self.reporter.add_skipped(f"{engine_id}: {reason}")
                if self.progress:
                    self.progress.engine_skipped(engine_id, reason)
                continue

            self._benchmark_engine(engine_id, dataset)

    def _benchmark_engine(self, engine_id: str, dataset: Dataset) -> None:
        """Benchmark a single engine on a dataset.

        Args:
            engine_id: Engine identifier
            dataset: Dataset to benchmark
        """
        logger.info(f"  Benchmarking engine: {engine_id}")

        # Get files first to know count for progress reporting
        files = list(dataset.get_files_for_engine(engine_id))
        if not files:
            reason = "No compatible files"
            logger.warning(f"{engine_id}: {reason}")
            self.reporter.add_skipped(f"{engine_id}: {reason}")
            if self.progress:
                self.progress.engine_skipped(engine_id, reason)
            return

        # Report engine start
        if self.progress:
            self.progress.engine_started(engine_id, dataset.language, len(files))

        # Load engine
        try:
            engine = self.engine_manager.get_engine(
                engine_id,
                device=self.config.device,
                language=dataset.language,
            )
        except Exception as e:
            reason = f"Failed to load - {e}"
            logger.warning(f"{engine_id}: {reason}")
            self.reporter.add_skipped(f"{engine_id}: {reason}")
            if self.progress:
                self.progress.engine_failed(engine_id, reason)
            return

        # Get GPU memory after model load
        gpu_memory_model = self.engine_manager.get_model_memory(
            engine_id, self.config.device, dataset.language
        )

        # Warm-up (once per engine)
        logger.debug(f"  Running warm-up for {engine_id}")
        try:
            first_file = files[0]
            engine.transcribe(first_file.audio, first_file.sample_rate)
        except Exception as e:
            reason = f"Warm-up failed - {e}"
            logger.warning(f"{engine_id}: {reason}")
            self.reporter.add_skipped(f"{engine_id}: {reason}")
            if self.progress:
                self.progress.engine_failed(engine_id, reason)
            return

        # Reset GPU peak memory for inference measurement
        if self.gpu_tracker.available:
            self.gpu_tracker.reset_peak()

        # Benchmark each file with progress bar
        file_iter = files
        if TQDM_AVAILABLE and self.config.mode != "quick":
            file_iter = tqdm(
                files,
                desc=f"  {engine_id}",
                leave=False,
                unit="file",
            )

        # Collect results for engine-level metrics
        engine_results: list[BenchmarkResult] = []

        for audio_file in file_iter:
            result = self._benchmark_file(engine, engine_id, audio_file, gpu_memory_model)
            if result:
                self.reporter.add_result(result)
                engine_results.append(result)
            # Report file completion for progress tracking
            if self.progress:
                self.progress.file_completed()

        # Unload audio data to free memory
        for audio_file in files:
            audio_file.unload()

        # Calculate and report engine-level metrics
        if engine_results:
            from statistics import mean
            avg_wer = mean(r.wer for r in engine_results)
            avg_cer = mean(r.cer for r in engine_results)
            avg_rtf = mean(r.rtf for r in engine_results)
            if self.progress:
                self.progress.engine_completed(
                    engine_id, wer=avg_wer, cer=avg_cer, rtf=avg_rtf
                )
        else:
            # No successful results
            if self.progress:
                self.progress.engine_completed(engine_id)

    def _benchmark_file(
        self,
        engine: TranscriptionEngine,
        engine_id: str,
        audio_file: AudioFile,
        gpu_memory_model: float | None,
    ) -> BenchmarkResult | None:
        """Benchmark a single file.

        Args:
            engine: Loaded engine
            engine_id: Engine identifier
            audio_file: Audio file to transcribe
            gpu_memory_model: GPU memory after model load

        Returns:
            BenchmarkResult or None if failed
        """
        try:
            # Measure processing time (multiple runs if configured)
            times = []
            transcript = ""

            for _ in range(self.config.runs):
                start = time.perf_counter()
                transcript, _ = engine.transcribe(audio_file.audio, audio_file.sample_rate)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            # Calculate metrics
            from statistics import mean
            processing_time = mean(times)
            rtf = calculate_rtf(audio_file.duration, processing_time)

            # Calculate WER/CER
            reference = audio_file.transcript
            wer = calculate_wer(reference, transcript, lang=audio_file.language)
            cer = calculate_cer(reference, transcript, lang=audio_file.language)

            # Get GPU peak memory
            gpu_peak = self.gpu_tracker.get_peak() if self.gpu_tracker.available else None

            return BenchmarkResult(
                engine=engine_id,
                language=audio_file.language,
                audio_file=audio_file.stem,
                transcript=transcript,
                reference=reference,
                wer=wer,
                cer=cer,
                rtf=rtf,
                audio_duration_s=audio_file.duration,
                processing_time_s=processing_time,
                gpu_memory_model_mb=gpu_memory_model,
                gpu_memory_peak_mb=gpu_peak,
            )

        except Exception as e:
            reason = f"{engine_id}/{audio_file.stem}: Transcription failed - {e}"
            logger.warning(reason)
            self.reporter.add_skipped(reason)
            return None

    def _save_results(self, result_dir: Path) -> None:
        """Save benchmark results.

        Args:
            result_dir: Output directory
        """
        # Save summary markdown
        summary_path = result_dir / "summary.md"
        self.reporter.save_markdown(summary_path)
        logger.info(f"  Saved summary: {summary_path}")

        # Save CSV files for each engine×language
        csv_paths = self.reporter.save_all_csv(result_dir)
        for path in csv_paths:
            logger.info(f"  Saved CSV: {path}")

        # Print to console
        self.reporter.to_console()
