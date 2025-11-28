"""VAD Benchmark Runner.

Provides VADBenchmarkRunner for evaluating VAD + ASR pipeline performance.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean

import numpy as np

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
from benchmarks.vad.backends import VADBenchmarkBackend
from benchmarks.vad.factory import create_vad, get_all_vad_ids, get_vad_config

__all__ = ["VADBenchmarkRunner", "VADBenchmarkConfig"]

logger = logging.getLogger(__name__)


# Default engines/VADs for quick mode
QUICK_MODE_ENGINES = {
    "ja": ["parakeet_ja", "whispers2t_large_v3"],
    "en": ["parakeet", "whispers2t_large_v3"],
}

QUICK_MODE_VADS = ["silero", "webrtc_mode3"]


@dataclass
class VADBenchmarkConfig:
    """Configuration for VAD benchmark execution."""

    # Execution mode
    mode: str = "quick"  # quick, standard, full

    # Target languages
    languages: list[str] = field(default_factory=lambda: ["ja", "en"])

    # Target engines (None = use mode defaults)
    engines: list[str] | None = None

    # Target VADs (None = use mode defaults)
    vads: list[str] | None = None

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

        if self.mode == "quick":
            return QUICK_MODE_ENGINES.get(language, [])

        # standard/full: use all engines for this language
        return BenchmarkEngineManager.get_engines_for_language(language)

    def get_vads(self) -> list[str]:
        """Get VADs to benchmark.

        Returns:
            List of VAD IDs
        """
        if self.vads:
            return self.vads

        if self.mode == "quick":
            return QUICK_MODE_VADS

        # standard/full: use all VADs
        return get_all_vad_ids()


class VADBenchmarkRunner:
    """Runs VAD + ASR benchmarks across engines, VADs and languages.

    Evaluates VAD quality indirectly through downstream ASR performance.
    For each (engine, VAD, language) combination:
    1. VAD segments the audio into speech regions
    2. ASR transcribes each segment
    3. Metrics (WER, CER, RTF, VAD RTF) are calculated

    Usage:
        config = VADBenchmarkConfig(mode="quick")
        runner = VADBenchmarkRunner(config)
        output_dir = runner.run()
        print(f"Results saved to: {output_dir}")
    """

    def __init__(self, config: VADBenchmarkConfig) -> None:
        """Initialize benchmark runner.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.dataset_manager = DatasetManager()
        self.engine_manager = BenchmarkEngineManager()
        self.gpu_tracker = GPUMemoryTracker()
        self.reporter = BenchmarkReporter(
            benchmark_type="vad",
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

    def _count_total_runs(self) -> int:
        """Count total (engine × VAD) combinations to benchmark."""
        total = 0
        vads = self.config.get_vads()
        for language in self.config.languages:
            engines = self.config.get_engines_for_language(language)
            total += len(engines) * len(vads)
        return total

    def run(self) -> Path:
        """Execute the benchmark.

        Returns:
            Path to the output directory
        """
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = self.output_dir / f"vad_{timestamp}_{self.config.mode}"
        result_dir.mkdir(parents=True, exist_ok=True)

        # Initialize progress reporter
        total_runs = self._count_total_runs()
        self.progress = ProgressReporter(
            benchmark_type="vad",
            mode=self.config.mode,
            languages=self.config.languages,
            total_engines=total_runs,
        )

        logger.info(f"Starting VAD benchmark (mode={self.config.mode})")
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
        """Benchmark all engines × VADs for a language.

        Args:
            language: Language code
        """
        logger.info(f"Benchmarking language: {language}")

        # Load dataset
        try:
            dataset = self.dataset_manager.get_dataset(
                language, mode=self.config.mode
            )
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

        # Get engines and VADs
        engines = self.config.get_engines_for_language(language)
        vads = self.config.get_vads()

        for engine_id in engines:
            # Check engine compatibility
            if not dataset.is_compatible_with_engine(engine_id):
                for vad_id in vads:
                    reason = f"Does not support {language}"
                    logger.debug(f"{engine_id}+{vad_id}: {reason}")
                    self.reporter.add_skipped(f"{engine_id}+{vad_id}: {reason}")
                    if self.progress:
                        self.progress.engine_skipped(engine_id, reason, vad_name=vad_id)
                continue

            # Benchmark this engine with each VAD
            for vad_id in vads:
                self._benchmark_engine_vad(engine_id, vad_id, dataset)

    def _benchmark_engine_vad(
        self,
        engine_id: str,
        vad_id: str,
        dataset: Dataset,
    ) -> None:
        """Benchmark a single engine + VAD combination.

        Args:
            engine_id: Engine identifier
            vad_id: VAD identifier
            dataset: Dataset to benchmark
        """
        logger.info(f"  Benchmarking: {engine_id} + {vad_id}")

        # Get files
        files = list(dataset.get_files_for_engine(engine_id))
        if not files:
            reason = "No compatible files"
            logger.warning(f"{engine_id}+{vad_id}: {reason}")
            self.reporter.add_skipped(f"{engine_id}+{vad_id}: {reason}")
            if self.progress:
                self.progress.engine_skipped(engine_id, reason, vad_name=vad_id)
            return

        # Report start
        if self.progress:
            self.progress.engine_started(
                engine_id, dataset.language, len(files), vad_name=vad_id
            )

        # Load engine
        try:
            engine = self.engine_manager.get_engine(
                engine_id,
                device=self.config.device,
                language=dataset.language,
            )
        except Exception as e:
            reason = f"Failed to load engine - {e}"
            logger.warning(f"{engine_id}+{vad_id}: {reason}")
            self.reporter.add_skipped(f"{engine_id}+{vad_id}: {reason}")
            if self.progress:
                self.progress.engine_failed(engine_id, reason, vad_name=vad_id)
            return

        # Create VAD
        try:
            vad = create_vad(vad_id)
            vad_config = get_vad_config(vad_id)
        except Exception as e:
            reason = f"Failed to create VAD - {e}"
            logger.warning(f"{engine_id}+{vad_id}: {reason}")
            self.reporter.add_skipped(f"{engine_id}+{vad_id}: {reason}")
            if self.progress:
                self.progress.engine_failed(engine_id, reason, vad_name=vad_id)
            return

        # Get GPU memory after model load
        gpu_memory_model = self.engine_manager.get_model_memory(
            engine_id, self.config.device, dataset.language
        )

        # Warm-up
        logger.debug(f"  Running warm-up for {engine_id}+{vad_id}")
        try:
            first_file = files[0]
            segments = vad.process_audio(first_file.audio, first_file.sample_rate)
            if segments:
                start_s, end_s = segments[0]
                start_sample = int(start_s * first_file.sample_rate)
                end_sample = int(end_s * first_file.sample_rate)
                segment_audio = first_file.audio[start_sample:end_sample]
                engine.transcribe(segment_audio, first_file.sample_rate)
        except Exception as e:
            reason = f"Warm-up failed - {e}"
            logger.warning(f"{engine_id}+{vad_id}: {reason}")
            self.reporter.add_skipped(f"{engine_id}+{vad_id}: {reason}")
            if self.progress:
                self.progress.engine_failed(engine_id, reason, vad_name=vad_id)
            return

        # Reset GPU peak memory for inference measurement
        if self.gpu_tracker.available:
            self.gpu_tracker.reset_peak()

        # Benchmark each file
        file_iter = files
        if TQDM_AVAILABLE and self.config.mode != "quick":
            file_iter = tqdm(
                files,
                desc=f"  {engine_id}+{vad_id}",
                leave=False,
                unit="file",
            )

        # Collect results
        run_results: list[BenchmarkResult] = []

        for audio_file in file_iter:
            result = self._benchmark_file(
                engine=engine,
                engine_id=engine_id,
                vad=vad,
                vad_id=vad_id,
                vad_config=vad_config,
                audio_file=audio_file,
                gpu_memory_model=gpu_memory_model,
            )
            if result:
                self.reporter.add_result(result)
                run_results.append(result)
            if self.progress:
                self.progress.file_completed()

        # Unload audio data
        for audio_file in files:
            audio_file.unload()

        # Report completion
        if run_results:
            avg_wer = mean(r.wer for r in run_results if r.wer is not None)
            avg_cer = mean(r.cer for r in run_results if r.cer is not None)
            avg_rtf = mean(r.rtf for r in run_results if r.rtf is not None)
            avg_vad_rtf = mean(r.vad_rtf for r in run_results if r.vad_rtf is not None)
            avg_segments = int(mean(r.segments_count for r in run_results if r.segments_count is not None))
            avg_speech_ratio = mean(r.speech_ratio for r in run_results if r.speech_ratio is not None)

            if self.progress:
                self.progress.engine_completed(
                    engine_id,
                    wer=avg_wer,
                    cer=avg_cer,
                    rtf=avg_rtf,
                    vad_rtf=avg_vad_rtf,
                    segments_count=avg_segments,
                    speech_ratio=avg_speech_ratio,
                )
        else:
            if self.progress:
                self.progress.engine_completed(engine_id)

    def _benchmark_file(
        self,
        engine: TranscriptionEngine,
        engine_id: str,
        vad: VADBenchmarkBackend,
        vad_id: str,
        vad_config: dict,
        audio_file: AudioFile,
        gpu_memory_model: float | None,
    ) -> BenchmarkResult | None:
        """Benchmark a single file with VAD + ASR.

        Args:
            engine: Loaded ASR engine
            engine_id: Engine identifier
            vad: VAD backend
            vad_id: VAD identifier
            vad_config: VAD configuration
            audio_file: Audio file to process
            gpu_memory_model: GPU memory after model load

        Returns:
            BenchmarkResult or None if failed
        """
        try:
            audio = audio_file.audio
            sample_rate = audio_file.sample_rate
            audio_duration = audio_file.duration

            # VAD processing
            vad_start = time.perf_counter()
            segments = vad.process_audio(audio, sample_rate)
            vad_elapsed = time.perf_counter() - vad_start

            # Calculate VAD metrics
            vad_rtf = calculate_rtf(audio_duration, vad_elapsed)
            segments_count = len(segments)

            # Calculate speech ratio
            total_speech_duration = sum(end - start for start, end in segments)
            speech_ratio = total_speech_duration / audio_duration if audio_duration > 0 else 0.0

            # Average segment duration
            avg_segment_duration = (
                total_speech_duration / segments_count if segments_count > 0 else 0.0
            )

            # ASR processing (transcribe each segment)
            transcripts = []
            asr_times = []

            for run_idx in range(self.config.runs):
                run_transcripts = []
                run_start = time.perf_counter()

                for start_s, end_s in segments:
                    start_sample = int(start_s * sample_rate)
                    end_sample = int(end_s * sample_rate)
                    segment_audio = audio[start_sample:end_sample]

                    if len(segment_audio) > 0:
                        transcript, _ = engine.transcribe(segment_audio, sample_rate)
                        run_transcripts.append(transcript)

                run_elapsed = time.perf_counter() - run_start
                asr_times.append(run_elapsed)

                if run_idx == 0:
                    transcripts = run_transcripts

            # Concatenate transcripts
            full_transcript = " ".join(transcripts)

            # Calculate ASR metrics
            asr_processing_time = mean(asr_times) if asr_times else 0.0
            total_processing_time = vad_elapsed + asr_processing_time
            rtf = calculate_rtf(audio_duration, total_processing_time)

            # Calculate WER/CER
            reference = audio_file.transcript
            wer = calculate_wer(reference, full_transcript, lang=audio_file.language)
            cer = calculate_cer(reference, full_transcript, lang=audio_file.language)

            # Get GPU peak memory
            gpu_peak = self.gpu_tracker.get_peak() if self.gpu_tracker.available else None

            return BenchmarkResult(
                engine=engine_id,
                language=audio_file.language,
                audio_file=audio_file.stem,
                transcript=full_transcript,
                reference=reference,
                wer=wer,
                cer=cer,
                rtf=rtf,
                audio_duration_s=audio_duration,
                processing_time_s=total_processing_time,
                gpu_memory_model_mb=gpu_memory_model,
                gpu_memory_peak_mb=gpu_peak,
                # VAD info
                vad=vad_id,
                vad_config=vad_config,
                vad_rtf=vad_rtf,
                segments_count=segments_count,
                avg_segment_duration_s=avg_segment_duration,
                speech_ratio=speech_ratio,
            )

        except Exception as e:
            reason = f"{engine_id}+{vad_id}/{audio_file.stem}: Processing failed - {e}"
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

        # Save JSON
        json_path = result_dir / "results.json"
        self.reporter.save_json(json_path)
        logger.info(f"  Saved JSON: {json_path}")

        # Print to console
        self.reporter.to_console()
