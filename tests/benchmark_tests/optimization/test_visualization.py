"""Tests for VAD optimization visualization module."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Skip all tests if optuna is not available
optuna = pytest.importorskip("optuna")

from benchmarks.optimization.visualization import (
    OptimizationReport,
    ComparisonCSVExporter,
    ReportPaths,
    generate_report,
)


class MockTrial:
    """Mock Optuna trial for testing."""

    def __init__(
        self,
        number: int = 0,
        value: float = 0.05,
        params: dict | None = None,
    ):
        self.number = number
        self.value = value
        self.params = params or {
            "threshold": 0.42,
            "min_silence_ms": 85,
            "speech_pad_ms": 120,
        }


class MockStudy:
    """Mock Optuna study for testing."""

    def __init__(
        self,
        study_name: str = "silero_ja",
        best_value: float = 0.0523,
        best_params: dict | None = None,
        n_trials: int = 50,
        user_attrs: dict | None = None,
    ):
        self.study_name = study_name
        # 実 study と同じく評価条件は user_attrs に入る (#480)。既定は記録の無い旧 study
        self.user_attrs = user_attrs or {}
        self.best_value = best_value
        self.best_params = best_params or {
            "threshold": 0.42,
            "min_silence_ms": 85,
            "speech_pad_ms": 120,
        }

        # Create mock trials
        self.trials = [
            MockTrial(i, best_value + (0.1 - i * 0.002), self.best_params)
            for i in range(n_trials)
        ]

        # Best trial
        self._best_trial = MockTrial(37, best_value, self.best_params)

    @property
    def best_trial(self) -> MockTrial:
        return self._best_trial


class TestOptimizationReport:
    """Tests for OptimizationReport class."""

    def test_init(self):
        """Test initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            study = MockStudy()

            report = OptimizationReport(
                study=study,
                output_dir=output_dir,
                vad_type="silero",
                language="ja",
            )

            assert report.study == study
            assert report.output_dir == output_dir
            assert report.vad_type == "silero"
            assert report.language == "ja"

    def test_study_name_property(self):
        """Test study_name property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            study = MockStudy(study_name="test_study")

            report = OptimizationReport(
                study=study,
                output_dir=output_dir,
                vad_type="silero",
                language="ja",
            )

            assert report.study_name == "test_study"

    def test_export_json(self):
        """Test JSON export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            study = MockStudy()

            report = OptimizationReport(
                study=study,
                output_dir=output_dir,
                vad_type="silero",
                language="ja",
            )

            json_path = report.export_json()

            assert json_path.exists()
            assert json_path.suffix == ".json"

            # Verify content
            with open(json_path) as f:
                data = json.load(f)

            assert data["study_name"] == "silero_ja"
            assert data["vad_type"] == "silero"
            assert data["language"] == "ja"
            assert "best_cer" in data  # Japanese uses CER
            assert data["best_cer"] == 0.0523
            assert data["best_params"]["threshold"] == 0.42
            assert data["n_trials"] == 50

    def test_export_json_english(self):
        """Test JSON export for English (uses WER)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            study = MockStudy(study_name="silero_en")

            report = OptimizationReport(
                study=study,
                output_dir=output_dir,
                vad_type="silero",
                language="en",
            )

            json_path = report.export_json()

            with open(json_path) as f:
                data = json.load(f)

            assert "best_wer" in data  # English uses WER
            assert "best_cer" not in data

    def test_write_step_summary_no_env(self):
        """Test Step Summary when GITHUB_STEP_SUMMARY is not set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            study = MockStudy()

            report = OptimizationReport(
                study=study,
                output_dir=output_dir,
                vad_type="silero",
                language="ja",
            )

            # Ensure env var is not set
            with patch.dict(os.environ, {}, clear=True):
                result = report.write_step_summary()

            assert result is False

    def test_write_step_summary_with_env(self):
        """Test Step Summary when GITHUB_STEP_SUMMARY is set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            summary_file = Path(tmpdir) / "step_summary.md"
            study = MockStudy()

            report = OptimizationReport(
                study=study,
                output_dir=output_dir,
                vad_type="silero",
                language="ja",
            )

            with patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(summary_file)}):
                result = report.write_step_summary()

            assert result is True
            assert summary_file.exists()

            content = summary_file.read_text()
            assert "silero_ja" in content
            assert "0.0523" in content
            assert "threshold" in content
            assert "Best CER" in content  # Japanese uses CER

    def test_generate_html_report_basic(self):
        """Test HTML report generation with mocked Plotly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            study = MockStudy()

            report = OptimizationReport(
                study=study,
                output_dir=output_dir,
                vad_type="silero",
                language="ja",
            )

            # Mock optuna.visualization
            mock_fig = MagicMock()
            mock_fig.to_html.return_value = "<div>Mock Chart</div>"

            with patch("optuna.visualization.plot_optimization_history", return_value=mock_fig):
                with patch("optuna.visualization.plot_param_importances", return_value=mock_fig):
                    with patch("optuna.visualization.plot_contour", return_value=mock_fig):
                        with patch("optuna.visualization.plot_parallel_coordinate", return_value=mock_fig):
                            html_path = report.generate_html_report()

            assert html_path.exists()
            assert html_path.suffix == ".html"

            content = html_path.read_text()
            assert "silero_ja" in content
            assert "0.0523" in content
            assert "threshold" in content

    def test_generate_all(self):
        """Test generate_all method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            study = MockStudy()

            report = OptimizationReport(
                study=study,
                output_dir=output_dir,
                vad_type="silero",
                language="ja",
            )

            # Mock HTML generation to avoid Plotly dependency
            with patch.object(report, "generate_html_report", return_value=output_dir / "test.html"):
                paths = report.generate_all()

            assert isinstance(paths, ReportPaths)
            assert paths.json is not None
            assert paths.json.exists()


class TestComparisonCSVExporter:
    """Tests for ComparisonCSVExporter class."""

    def test_add_result_and_save(self):
        """Test adding results and saving CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "comparison.csv"

            exporter = ComparisonCSVExporter(csv_path)

            # Add multiple results
            exporter.add_result(
                vad_type="silero",
                language="ja",
                best_score=0.0523,
                best_params={"threshold": 0.42, "min_silence_ms": 85},
                n_trials=50,
            )
            exporter.add_result(
                vad_type="silero",
                language="en",
                best_score=0.032,
                best_params={"threshold": 0.52, "min_silence_ms": 95},
                n_trials=50,
            )

            saved_path = exporter.save()

            assert saved_path.exists()
            assert saved_path == csv_path

            # Verify content
            content = csv_path.read_text()
            assert "silero" in content
            assert "ja" in content
            assert "en" in content
            assert "0.0523" in content
            assert "0.032" in content

    def test_add_study(self):
        """Test adding study directly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "comparison.csv"

            exporter = ComparisonCSVExporter(csv_path)

            study = MockStudy(study_name="silero_ja")
            exporter.add_study(study, vad_type="silero", language="ja")

            assert len(exporter.rows) == 1
            assert exporter.rows[0]["vad_type"] == "silero"
            assert exporter.rows[0]["language"] == "ja"

    def test_save_empty_raises(self):
        """Test that saving with no rows raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "comparison.csv"

            exporter = ComparisonCSVExporter(csv_path)

            with pytest.raises(ValueError, match="No studies added"):
                exporter.save()


class TestGenerateReport:
    """Tests for generate_report convenience function."""

    def test_generate_report_basic(self):
        """Test basic report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            study = MockStudy(study_name="silero_ja")

            # Mock HTML generation
            with patch(
                "benchmarks.optimization.visualization.OptimizationReport.generate_html_report",
                return_value=output_dir / "test.html",
            ):
                paths = generate_report(
                    study=study,
                    output_dir=output_dir,
                    vad_type="silero",
                    language="ja",
                )

            assert isinstance(paths, ReportPaths)

    def test_generate_report_extracts_from_study_name(self):
        """Test that vad_type and language are extracted from study name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            study = MockStudy(study_name="tenvad_en")

            with patch(
                "benchmarks.optimization.visualization.OptimizationReport.generate_html_report",
                return_value=output_dir / "test.html",
            ):
                # Don't pass vad_type and language - should extract from study_name
                paths = generate_report(
                    study=study,
                    output_dir=output_dir,
                )

            # Verify JSON was created with extracted values
            assert paths.json is not None


class TestReportPaths:
    """Tests for ReportPaths dataclass."""

    def test_default_values(self):
        """Test default values."""
        paths = ReportPaths()

        assert paths.html is None
        assert paths.json is None
        assert paths.csv is None
        assert paths.step_summary is False

    def test_with_values(self):
        """Test with values set."""
        paths = ReportPaths(
            html=Path("/tmp/test.html"),
            json=Path("/tmp/test.json"),
            step_summary=True,
        )

        assert paths.html == Path("/tmp/test.html")
        assert paths.json == Path("/tmp/test.json")
        assert paths.step_summary is True
