"""Tests for ASR benchmark runner."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from benchmarks.asr.runner import ASRBenchmarkConfig, ASRBenchmarkRunner, QUICK_MODE_ENGINES
from benchmarks.common import AudioFile, BenchmarkResult, Dataset


class TestASRBenchmarkConfig:
    """Tests for ASRBenchmarkConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ASRBenchmarkConfig()
        assert config.mode == "quick"
        assert config.languages == ["ja", "en"]
        assert config.engines is None
        assert config.runs == 1
        assert config.device == "cuda"
        assert config.output_dir is None

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ASRBenchmarkConfig(
            mode="standard",
            languages=["ja"],
            engines=["parakeet_ja"],
            runs=3,
            device="cpu",
            output_dir=Path("/tmp/results"),
        )
        assert config.mode == "standard"
        assert config.languages == ["ja"]
        assert config.engines == ["parakeet_ja"]
        assert config.runs == 3
        assert config.device == "cpu"
        assert config.output_dir == Path("/tmp/results")

    def test_get_engines_for_language_quick_mode_ja(self) -> None:
        """Test engine selection for quick mode - Japanese."""
        config = ASRBenchmarkConfig(mode="quick")
        engines = config.get_engines_for_language("ja")
        assert engines == QUICK_MODE_ENGINES["ja"]
        assert "parakeet_ja" in engines
        assert "whispers2t_large_v3" in engines

    def test_get_engines_for_language_quick_mode_en(self) -> None:
        """Test engine selection for quick mode - English."""
        config = ASRBenchmarkConfig(mode="quick")
        engines = config.get_engines_for_language("en")
        assert engines == QUICK_MODE_ENGINES["en"]
        assert "parakeet" in engines
        assert "whispers2t_large_v3" in engines

    def test_get_engines_for_language_quick_mode_unknown(self) -> None:
        """Test engine selection for quick mode - unknown language."""
        config = ASRBenchmarkConfig(mode="quick")
        engines = config.get_engines_for_language("zh")
        assert engines == []

    def test_get_engines_for_language_with_explicit_engines(self) -> None:
        """Test engine selection when explicit engines are configured."""
        config = ASRBenchmarkConfig(
            mode="quick",
            engines=["reazonspeech", "parakeet_ja"],
        )
        # Should return explicit engines regardless of language
        assert config.get_engines_for_language("ja") == ["reazonspeech", "parakeet_ja"]
        assert config.get_engines_for_language("en") == ["reazonspeech", "parakeet_ja"]

    @patch("benchmarks.asr.runner.BenchmarkEngineManager.get_engines_for_language")
    def test_get_engines_for_language_standard_mode(self, mock_get_engines: MagicMock) -> None:
        """Test engine selection for standard mode uses all engines for language."""
        mock_get_engines.return_value = ["engine_a", "engine_b", "engine_c"]

        config = ASRBenchmarkConfig(mode="standard")
        engines = config.get_engines_for_language("ja")

        mock_get_engines.assert_called_once_with("ja")
        assert engines == ["engine_a", "engine_b", "engine_c"]


class TestASRBenchmarkRunner:
    """Tests for ASRBenchmarkRunner."""

    @pytest.fixture
    def mock_dataset_manager(self) -> MagicMock:
        """Create mock dataset manager."""
        manager = MagicMock()
        return manager

    @pytest.fixture
    def mock_engine_manager(self) -> MagicMock:
        """Create mock engine manager."""
        manager = MagicMock()
        return manager

    @pytest.fixture
    def mock_gpu_tracker(self) -> MagicMock:
        """Create mock GPU tracker."""
        tracker = MagicMock()
        tracker.available = False
        return tracker

    @pytest.fixture
    def sample_audio_file(self, tmp_path: Path) -> AudioFile:
        """Create a sample AudioFile for testing."""
        audio_path = tmp_path / "test.wav"
        transcript_path = tmp_path / "test.txt"

        # Write dummy files
        audio_path.write_bytes(b"dummy")
        transcript_path.write_text("テスト", encoding="utf-8")

        audio_file = AudioFile(
            path=audio_path,
            transcript_path=transcript_path,
            language="ja",
        )
        # Pre-populate audio data to avoid file reading
        audio_file._audio = np.zeros(16000, dtype=np.float32)
        audio_file._sample_rate = 16000
        audio_file._transcript = "テスト"

        return audio_file

    @pytest.fixture
    def sample_dataset(self, sample_audio_file: AudioFile) -> Dataset:
        """Create a sample Dataset for testing."""
        dataset = Dataset(
            language="ja",
            files=[sample_audio_file],
        )
        return dataset

    def test_runner_initialization(self) -> None:
        """Test runner initialization with default config."""
        config = ASRBenchmarkConfig()
        runner = ASRBenchmarkRunner(config)

        assert runner.config == config
        assert runner.dataset_manager is not None
        assert runner.engine_manager is not None
        assert runner.gpu_tracker is not None
        assert runner.reporter is not None

    def test_runner_initialization_with_custom_output_dir(self, tmp_path: Path) -> None:
        """Test runner initialization with custom output directory."""
        config = ASRBenchmarkConfig(output_dir=tmp_path / "results")
        runner = ASRBenchmarkRunner(config)

        assert runner.output_dir == tmp_path / "results"

    @patch.object(ASRBenchmarkRunner, "_benchmark_language")
    @patch.object(ASRBenchmarkRunner, "_save_results")
    def test_run_creates_output_directory(
        self,
        mock_save: MagicMock,
        mock_benchmark: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that run() creates timestamped output directory."""
        config = ASRBenchmarkConfig(
            output_dir=tmp_path,
            languages=["ja"],
        )
        runner = ASRBenchmarkRunner(config)

        result_dir = runner.run()

        assert result_dir.exists()
        assert result_dir.parent == tmp_path
        assert "quick" in result_dir.name  # mode is in directory name
        mock_benchmark.assert_called_once_with("ja")

    @patch("benchmarks.asr.runner.DatasetManager")
    @patch("benchmarks.asr.runner.BenchmarkEngineManager")
    def test_benchmark_language_handles_dataset_error(
        self,
        mock_engine_manager_cls: MagicMock,
        mock_dataset_manager_cls: MagicMock,
    ) -> None:
        """Test that dataset load errors are handled gracefully."""
        # Setup mocks
        mock_dataset_manager = MagicMock()
        mock_dataset_manager.get_dataset.side_effect = Exception("Dataset not found")
        mock_dataset_manager_cls.return_value = mock_dataset_manager

        mock_engine_manager = MagicMock()
        mock_engine_manager_cls.return_value = mock_engine_manager

        config = ASRBenchmarkConfig(languages=["unknown_lang"])
        runner = ASRBenchmarkRunner(config)

        # Should not raise
        runner._benchmark_language("unknown_lang")

        # Should have added to skipped list
        assert len(runner.reporter.skipped) == 1
        assert "unknown_lang" in runner.reporter.skipped[0]

    @patch("benchmarks.asr.runner.DatasetManager")
    @patch("benchmarks.asr.runner.BenchmarkEngineManager")
    def test_benchmark_language_handles_empty_dataset(
        self,
        mock_engine_manager_cls: MagicMock,
        mock_dataset_manager_cls: MagicMock,
    ) -> None:
        """Test that empty datasets are handled gracefully."""
        # Setup mocks
        mock_dataset_manager = MagicMock()
        empty_dataset = Dataset(language="ja", files=[])
        mock_dataset_manager.get_dataset.return_value = empty_dataset
        mock_dataset_manager_cls.return_value = mock_dataset_manager

        mock_engine_manager = MagicMock()
        mock_engine_manager_cls.return_value = mock_engine_manager

        config = ASRBenchmarkConfig(languages=["ja"])
        runner = ASRBenchmarkRunner(config)

        # Should not raise
        runner._benchmark_language("ja")

        # Should have added to skipped list
        assert len(runner.reporter.skipped) == 1
        assert "No audio files" in runner.reporter.skipped[0]

    @patch("benchmarks.asr.runner.DatasetManager")
    @patch("benchmarks.asr.runner.BenchmarkEngineManager")
    def test_benchmark_engine_handles_load_error(
        self,
        mock_engine_manager_cls: MagicMock,
        mock_dataset_manager_cls: MagicMock,
        sample_dataset: Dataset,
    ) -> None:
        """Test that engine load errors are handled gracefully."""
        # Setup mocks
        mock_dataset_manager = MagicMock()
        mock_dataset_manager.get_dataset.return_value = sample_dataset
        mock_dataset_manager_cls.return_value = mock_dataset_manager

        mock_engine_manager = MagicMock()
        mock_engine_manager.get_engine.side_effect = ImportError("Engine not available")
        mock_engine_manager_cls.return_value = mock_engine_manager

        config = ASRBenchmarkConfig(
            languages=["ja"],
            engines=["unavailable_engine"],
        )
        runner = ASRBenchmarkRunner(config)

        # Should not raise
        runner._benchmark_engine("unavailable_engine", sample_dataset)

        # Should have added to skipped list
        assert len(runner.reporter.skipped) == 1
        assert "unavailable_engine" in runner.reporter.skipped[0]

    @patch("benchmarks.asr.runner.calculate_cer")
    @patch("benchmarks.asr.runner.calculate_wer")
    def test_benchmark_file_returns_result(
        self,
        mock_wer: MagicMock,
        mock_cer: MagicMock,
        sample_audio_file: AudioFile,
    ) -> None:
        """Test that benchmark_file returns proper result."""
        # Mock WER/CER calculations (jiwer may not be installed in CI)
        mock_wer.return_value = 0.0
        mock_cer.return_value = 0.0

        config = ASRBenchmarkConfig()
        runner = ASRBenchmarkRunner(config)

        # Create mock engine
        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = ("テスト", 0.95)

        result = runner._benchmark_file(
            engine=mock_engine,
            engine_id="test_engine",
            audio_file=sample_audio_file,
            gpu_memory_model=100.0,
        )

        assert result is not None
        assert isinstance(result, BenchmarkResult)
        assert result.engine == "test_engine"
        assert result.language == "ja"
        assert result.transcript == "テスト"
        assert result.reference == "テスト"
        assert result.wer == 0.0
        assert result.cer == 0.0
        assert result.rtf is not None
        mock_wer.assert_called_once()
        mock_cer.assert_called_once()

    def test_benchmark_file_handles_transcription_error(
        self,
        sample_audio_file: AudioFile,
    ) -> None:
        """Test that transcription errors are handled gracefully."""
        config = ASRBenchmarkConfig()
        runner = ASRBenchmarkRunner(config)

        # Create mock engine that raises
        mock_engine = MagicMock()
        mock_engine.transcribe.side_effect = RuntimeError("CUDA out of memory")

        result = runner._benchmark_file(
            engine=mock_engine,
            engine_id="test_engine",
            audio_file=sample_audio_file,
            gpu_memory_model=100.0,
        )

        # Should return None and add to skipped
        assert result is None
        assert len(runner.reporter.skipped) == 1
        assert "Transcription failed" in runner.reporter.skipped[0]

    @patch("benchmarks.asr.runner.calculate_cer")
    @patch("benchmarks.asr.runner.calculate_wer")
    def test_benchmark_file_multiple_runs(
        self,
        mock_wer: MagicMock,
        mock_cer: MagicMock,
        sample_audio_file: AudioFile,
    ) -> None:
        """Test that multiple runs are averaged correctly."""
        # Mock WER/CER calculations (jiwer may not be installed in CI)
        mock_wer.return_value = 0.1
        mock_cer.return_value = 0.05

        config = ASRBenchmarkConfig(runs=3)
        runner = ASRBenchmarkRunner(config)

        # Create mock engine
        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = ("テスト", 0.95)

        result = runner._benchmark_file(
            engine=mock_engine,
            engine_id="test_engine",
            audio_file=sample_audio_file,
            gpu_memory_model=100.0,
        )

        # Engine should be called runs times
        assert mock_engine.transcribe.call_count == 3
        assert result is not None
        assert result.wer == 0.1
        assert result.cer == 0.05

    @patch.object(ASRBenchmarkRunner, "_benchmark_language")
    def test_run_clears_engine_cache(
        self,
        mock_benchmark: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that engine cache is cleared after run."""
        config = ASRBenchmarkConfig(
            output_dir=tmp_path,
            languages=["ja"],
        )
        runner = ASRBenchmarkRunner(config)

        # Mock the engine manager
        mock_engine_mgr = MagicMock()
        runner.engine_manager = mock_engine_mgr

        runner.run()

        mock_engine_mgr.clear_cache.assert_called_once()

    def test_save_results_creates_files(self, tmp_path: Path) -> None:
        """Test that save_results creates summary.md and CSV files."""
        config = ASRBenchmarkConfig()
        runner = ASRBenchmarkRunner(config)

        # Add a sample result
        runner.reporter.add_result(BenchmarkResult(
            engine="test_engine",
            language="ja",
            audio_file="test",
            transcript="テスト",
            reference="テスト",
            wer=0.0,
            cer=0.0,
            rtf=0.5,
            audio_duration_s=1.0,
            processing_time_s=0.5,
        ))

        result_dir = tmp_path / "results"
        result_dir.mkdir()

        runner._save_results(result_dir)

        # Check summary.md exists
        assert (result_dir / "summary.md").exists()

        # Check CSV exists in raw/ subdirectory
        csv_files = list((result_dir / "raw").glob("*.csv"))
        assert len(csv_files) == 1

    def test_save_results_includes_summary_content(self, tmp_path: Path) -> None:
        """Test that summary.md contains expected content sections."""
        config = ASRBenchmarkConfig()
        runner = ASRBenchmarkRunner(config)

        # Add multiple results for aggregation testing
        for i, (engine, lang) in enumerate([
            ("engine_a", "ja"),
            ("engine_a", "ja"),
            ("engine_b", "ja"),
            ("engine_b", "en"),
        ]):
            runner.reporter.add_result(BenchmarkResult(
                engine=engine,
                language=lang,
                audio_file=f"test_{i}",
                transcript="テスト" if lang == "ja" else "test",
                reference="テスト" if lang == "ja" else "test",
                wer=0.1 + i * 0.01,
                cer=0.05 + i * 0.005,
                rtf=0.3 + i * 0.1,
                audio_duration_s=1.0,
                processing_time_s=0.3 + i * 0.1,
                gpu_memory_peak_mb=1000.0 + i * 100,
            ))

        result_dir = tmp_path / "results"
        result_dir.mkdir()

        runner._save_results(result_dir)

        # Check summary.md content
        summary_content = (result_dir / "summary.md").read_text(encoding="utf-8")

        # Should contain header
        assert "Benchmark Report" in summary_content or "ASR" in summary_content

        # Should contain engine names
        assert "engine_a" in summary_content
        assert "engine_b" in summary_content

    def test_reporter_add_skipped_tracking(self) -> None:
        """Test that skipped items are tracked correctly."""
        config = ASRBenchmarkConfig()
        runner = ASRBenchmarkRunner(config)

        # Add skipped items
        runner.reporter.add_skipped("engine_x: Not available")
        runner.reporter.add_skipped("ja: Dataset not found")

        assert len(runner.reporter.skipped) == 2
        assert "engine_x" in runner.reporter.skipped[0]
        assert "ja" in runner.reporter.skipped[1]

    def test_reporter_skipped_appears_in_summary(self, tmp_path: Path) -> None:
        """Test that skipped items appear in summary output."""
        config = ASRBenchmarkConfig()
        runner = ASRBenchmarkRunner(config)

        # Add a result and some skipped items
        runner.reporter.add_result(BenchmarkResult(
            engine="test_engine",
            language="ja",
            audio_file="test",
            transcript="テスト",
            reference="テスト",
            wer=0.0,
            cer=0.0,
            rtf=0.5,
            audio_duration_s=1.0,
            processing_time_s=0.5,
        ))
        runner.reporter.add_skipped("unavailable_engine: ImportError")
        runner.reporter.add_skipped("zh: No dataset")

        result_dir = tmp_path / "results"
        result_dir.mkdir()

        runner._save_results(result_dir)

        # Check summary includes skipped section
        summary_content = (result_dir / "summary.md").read_text(encoding="utf-8")
        assert "unavailable_engine" in summary_content or "Skipped" in summary_content
