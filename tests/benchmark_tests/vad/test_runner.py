"""Tests for VAD benchmark runner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from benchmarks.vad.runner import (
    VADBenchmarkConfig,
    VADBenchmarkRunner,
    QUICK_MODE_ENGINES,
    QUICK_MODE_VADS,
)
from benchmarks.common import AudioFile, BenchmarkResult, Dataset


class TestVADBenchmarkConfig:
    """Tests for VADBenchmarkConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = VADBenchmarkConfig()
        assert config.mode == "quick"
        assert config.languages == ["ja", "en"]
        assert config.engines is None
        assert config.vads is None
        assert config.runs == 1
        assert config.device == "cuda"
        assert config.output_dir is None

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = VADBenchmarkConfig(
            mode="standard",
            languages=["ja"],
            engines=["parakeet_ja"],
            vads=["silero", "webrtc_mode3"],
            runs=3,
            device="cpu",
            output_dir=Path("/tmp/results"),
        )
        assert config.mode == "standard"
        assert config.languages == ["ja"]
        assert config.engines == ["parakeet_ja"]
        assert config.vads == ["silero", "webrtc_mode3"]
        assert config.runs == 3
        assert config.device == "cpu"
        assert config.output_dir == Path("/tmp/results")

    def test_get_engines_for_language_quick_mode_ja(self) -> None:
        """Test engine selection for quick mode - Japanese."""
        config = VADBenchmarkConfig(mode="quick")
        engines = config.get_engines_for_language("ja")
        assert engines == QUICK_MODE_ENGINES["ja"]
        assert "parakeet_ja" in engines
        assert "whispers2t_large_v3" in engines

    def test_get_engines_for_language_quick_mode_en(self) -> None:
        """Test engine selection for quick mode - English."""
        config = VADBenchmarkConfig(mode="quick")
        engines = config.get_engines_for_language("en")
        assert engines == QUICK_MODE_ENGINES["en"]
        assert "parakeet" in engines
        assert "whispers2t_large_v3" in engines

    def test_get_engines_for_language_quick_mode_unknown(self) -> None:
        """Test engine selection for quick mode - unknown language."""
        config = VADBenchmarkConfig(mode="quick")
        engines = config.get_engines_for_language("zh")
        assert engines == []

    def test_get_engines_for_language_with_explicit_engines(self) -> None:
        """Test engine selection when explicit engines are configured."""
        config = VADBenchmarkConfig(
            mode="quick",
            engines=["reazonspeech", "parakeet_ja"],
        )
        # Should return explicit engines regardless of language
        assert config.get_engines_for_language("ja") == ["reazonspeech", "parakeet_ja"]
        assert config.get_engines_for_language("en") == ["reazonspeech", "parakeet_ja"]

    def test_get_vads_quick_mode(self) -> None:
        """Test VAD selection for quick mode."""
        config = VADBenchmarkConfig(mode="quick")
        vads = config.get_vads()
        assert vads == QUICK_MODE_VADS

    def test_get_vads_with_explicit_vads(self) -> None:
        """Test VAD selection when explicit VADs are configured."""
        config = VADBenchmarkConfig(
            mode="quick",
            vads=["silero", "tenvad"],
        )
        assert config.get_vads() == ["silero", "tenvad"]

    @patch("benchmarks.vad.runner.get_all_vad_ids")
    def test_get_vads_standard_mode_returns_all(self, mock_get_vads: MagicMock) -> None:
        """Test VAD selection for standard/full mode uses all VADs."""
        mock_get_vads.return_value = ["vad_a", "vad_b", "vad_c"]

        config = VADBenchmarkConfig(mode="standard")
        vads = config.get_vads()

        mock_get_vads.assert_called_once()
        assert vads == ["vad_a", "vad_b", "vad_c"]

    @patch("benchmarks.asr.runner.BenchmarkEngineManager.get_engines_for_language")
    def test_get_engines_for_language_standard_mode(self, mock_get_engines: MagicMock) -> None:
        """Test engine selection for standard mode uses all engines for language."""
        mock_get_engines.return_value = ["engine_a", "engine_b", "engine_c"]

        config = VADBenchmarkConfig(mode="standard")
        engines = config.get_engines_for_language("ja")

        mock_get_engines.assert_called_once_with("ja")
        assert engines == ["engine_a", "engine_b", "engine_c"]


class TestVADBenchmarkRunner:
    """Tests for VADBenchmarkRunner."""

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
        config = VADBenchmarkConfig()
        runner = VADBenchmarkRunner(config)

        assert runner.config == config
        assert runner.dataset_manager is not None
        assert runner.engine_manager is not None
        assert runner.gpu_tracker is not None
        assert runner.reporter is not None
        assert runner.reporter.benchmark_type == "vad"

    def test_runner_initialization_with_custom_output_dir(self, tmp_path: Path) -> None:
        """Test runner initialization with custom output directory."""
        config = VADBenchmarkConfig(output_dir=tmp_path / "results")
        runner = VADBenchmarkRunner(config)

        assert runner.output_dir == tmp_path / "results"

    def test_count_total_runs(self) -> None:
        """Test counting total (engine × VAD) combinations."""
        config = VADBenchmarkConfig(
            mode="quick",
            languages=["ja"],
            engines=["engine_a", "engine_b"],
            vads=["vad_1", "vad_2", "vad_3"],
        )
        runner = VADBenchmarkRunner(config)

        # 2 engines × 3 VADs = 6
        assert runner._count_total_runs() == 6

    def test_count_total_runs_multiple_languages(self) -> None:
        """Test counting total runs across multiple languages."""
        config = VADBenchmarkConfig(
            mode="quick",
            languages=["ja", "en"],
            engines=["engine_a"],
            vads=["vad_1", "vad_2"],
        )
        runner = VADBenchmarkRunner(config)

        # (1 engine × 2 VADs) × 2 languages = 4
        assert runner._count_total_runs() == 4

    @patch.object(VADBenchmarkRunner, "_benchmark_language")
    @patch.object(VADBenchmarkRunner, "_save_results")
    def test_run_creates_output_directory(
        self,
        mock_save: MagicMock,
        mock_benchmark: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that run() creates timestamped output directory."""
        config = VADBenchmarkConfig(
            output_dir=tmp_path,
            languages=["ja"],
        )
        runner = VADBenchmarkRunner(config)

        result_dir = runner.run()

        assert result_dir.exists()
        assert result_dir.parent == tmp_path
        assert "vad_" in result_dir.name
        assert "quick" in result_dir.name
        mock_benchmark.assert_called_once_with("ja")

    @patch("benchmarks.vad.runner.DatasetManager")
    @patch("benchmarks.vad.runner.BenchmarkEngineManager")
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

        config = VADBenchmarkConfig(languages=["unknown_lang"])
        runner = VADBenchmarkRunner(config)

        # Should not raise
        runner._benchmark_language("unknown_lang")

        # Should have added to skipped list
        assert len(runner.reporter.skipped) == 1
        assert "unknown_lang" in runner.reporter.skipped[0]

    @patch("benchmarks.vad.runner.DatasetManager")
    @patch("benchmarks.vad.runner.BenchmarkEngineManager")
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

        config = VADBenchmarkConfig(languages=["ja"])
        runner = VADBenchmarkRunner(config)

        # Should not raise
        runner._benchmark_language("ja")

        # Should have added to skipped list
        assert len(runner.reporter.skipped) == 1
        assert "No audio files" in runner.reporter.skipped[0]

    @patch("benchmarks.vad.runner.create_vad")
    @patch("benchmarks.vad.runner.get_vad_config")
    @patch("benchmarks.vad.runner.calculate_cer")
    @patch("benchmarks.vad.runner.calculate_wer")
    def test_benchmark_file_returns_result_with_vad_info(
        self,
        mock_wer: MagicMock,
        mock_cer: MagicMock,
        mock_get_vad_config: MagicMock,
        mock_create_vad: MagicMock,
        sample_audio_file: AudioFile,
    ) -> None:
        """Test that benchmark_file returns proper result with VAD info."""
        # Mock WER/CER calculations
        mock_wer.return_value = 0.0
        mock_cer.return_value = 0.0

        # Mock VAD
        mock_vad = MagicMock()
        mock_vad.process_audio.return_value = [(0.0, 0.5), (0.6, 1.0)]
        mock_vad.name = "test_vad"
        mock_create_vad.return_value = mock_vad
        mock_get_vad_config.return_value = {"type": "protocol", "params": {}}

        config = VADBenchmarkConfig()
        runner = VADBenchmarkRunner(config)

        # Create mock engine
        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = ("テスト", 0.95)

        result = runner._benchmark_file(
            engine=mock_engine,
            engine_id="test_engine",
            vad=mock_vad,
            vad_id="test_vad",
            vad_config={"type": "protocol"},
            audio_file=sample_audio_file,
            gpu_memory_model=100.0,
        )

        assert result is not None
        assert isinstance(result, BenchmarkResult)
        assert result.engine == "test_engine"
        assert result.language == "ja"
        assert result.vad == "test_vad"
        assert result.vad_config == {"type": "protocol"}
        assert result.segments_count == 2
        assert result.speech_ratio is not None
        assert result.vad_rtf is not None

    def test_benchmark_file_handles_vad_error(
        self,
        sample_audio_file: AudioFile,
    ) -> None:
        """Test that VAD errors are handled gracefully."""
        config = VADBenchmarkConfig()
        runner = VADBenchmarkRunner(config)

        # Create mock VAD that raises
        mock_vad = MagicMock()
        mock_vad.process_audio.side_effect = RuntimeError("VAD error")

        # Create mock engine
        mock_engine = MagicMock()

        result = runner._benchmark_file(
            engine=mock_engine,
            engine_id="test_engine",
            vad=mock_vad,
            vad_id="test_vad",
            vad_config={},
            audio_file=sample_audio_file,
            gpu_memory_model=100.0,
        )

        # Should return None and add to skipped
        assert result is None
        assert len(runner.reporter.skipped) == 1
        assert "Processing failed" in runner.reporter.skipped[0]

    @patch.object(VADBenchmarkRunner, "_benchmark_language")
    def test_run_clears_engine_cache(
        self,
        mock_benchmark: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that engine cache is cleared after run."""
        config = VADBenchmarkConfig(
            output_dir=tmp_path,
            languages=["ja"],
        )
        runner = VADBenchmarkRunner(config)

        # Mock the engine manager
        mock_engine_mgr = MagicMock()
        runner.engine_manager = mock_engine_mgr

        runner.run()

        mock_engine_mgr.clear_cache.assert_called_once()

    def test_save_results_creates_files(self, tmp_path: Path) -> None:
        """Test that save_results creates summary.md and JSON files."""
        config = VADBenchmarkConfig()
        runner = VADBenchmarkRunner(config)

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
            vad="test_vad",
            vad_config={"type": "protocol"},
            vad_rtf=0.01,
            segments_count=2,
            speech_ratio=0.5,
        ))

        result_dir = tmp_path / "results"
        result_dir.mkdir()

        runner._save_results(result_dir)

        # Check summary.md exists
        assert (result_dir / "summary.md").exists()

        # Check JSON exists
        assert (result_dir / "results.json").exists()

    def test_benchmark_result_contains_vad_fields(self) -> None:
        """Test that BenchmarkResult includes VAD-specific fields."""
        result = BenchmarkResult(
            engine="test_engine",
            language="ja",
            audio_file="test",
            transcript="テスト",
            reference="テスト",
            wer=0.1,
            cer=0.05,
            rtf=0.5,
            audio_duration_s=1.0,
            processing_time_s=0.5,
            vad="silero",
            vad_config={"threshold": 0.5, "onnx": True},
            vad_rtf=0.01,
            segments_count=5,
            avg_segment_duration_s=0.5,
            speech_ratio=0.75,
        )

        # Check VAD fields
        assert result.vad == "silero"
        assert result.vad_config == {"threshold": 0.5, "onnx": True}
        assert result.vad_rtf == 0.01
        assert result.segments_count == 5
        assert result.avg_segment_duration_s == 0.5
        assert result.speech_ratio == 0.75

        # Check to_dict includes VAD info
        result_dict = result.to_dict()
        assert "vad" in result_dict
        assert result_dict["vad"]["name"] == "silero"
        assert result_dict["vad"]["config"] == {"threshold": 0.5, "onnx": True}
        assert result_dict["vad"]["vad_rtf"] == 0.01
        assert result_dict["vad"]["segments_count"] == 5
