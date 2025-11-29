"""Visualization and reporting for VAD parameter optimization.

Provides report generation in multiple formats:
- Step Summary (GitHub Actions): Text tables
- HTML Report: Interactive Plotly graphs
- JSON: Best parameters export
- CSV: Comparison across multiple studies
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)


@dataclass
class ReportPaths:
    """Paths to generated report files."""

    html: Path | None = None
    json: Path | None = None
    csv: Path | None = None
    step_summary: bool = False


class OptimizationReport:
    """Generate optimization reports in various formats.

    Args:
        study: Optuna study object
        output_dir: Directory to save reports

    Example:
        report = OptimizationReport(study, output_dir=Path("reports"))
        paths = report.generate_all()
        print(f"HTML report: {paths.html}")
    """

    def __init__(
        self,
        study: "optuna.Study",
        output_dir: Path,
        vad_type: str | None = None,
        language: str | None = None,
    ) -> None:
        self.study = study
        self.output_dir = output_dir
        self.vad_type = vad_type
        self.language = language

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def study_name(self) -> str:
        """Get study name for file naming."""
        return self.study.study_name or f"{self.vad_type}_{self.language}"

    def generate_all(self) -> ReportPaths:
        """Generate all report formats.

        Returns:
            ReportPaths with paths to generated files
        """
        paths = ReportPaths()

        # Generate HTML report
        try:
            paths.html = self.generate_html_report()
            logger.info(f"HTML report generated: {paths.html}")
        except Exception as e:
            logger.warning(f"Failed to generate HTML report: {e}")

        # Export JSON
        try:
            paths.json = self.export_json()
            logger.info(f"JSON exported: {paths.json}")
        except Exception as e:
            logger.warning(f"Failed to export JSON: {e}")

        # Write Step Summary (if in GitHub Actions)
        try:
            paths.step_summary = self.write_step_summary()
            if paths.step_summary:
                logger.info("Step Summary written")
        except Exception as e:
            logger.warning(f"Failed to write Step Summary: {e}")

        return paths

    def generate_html_report(self) -> Path:
        """Generate interactive HTML report with Plotly.

        Returns:
            Path to generated HTML file
        """
        import optuna.visualization as vis

        html_path = self.output_dir / f"{self.study_name}.html"

        # Generate individual figures
        figures = {}

        # 1. Optimization History (always available)
        try:
            figures["history"] = vis.plot_optimization_history(self.study)
        except Exception as e:
            logger.warning(f"Failed to generate optimization history plot: {e}")

        # 2. Parameter Importance (needs enough trials)
        if len(self.study.trials) >= 10:
            try:
                figures["importance"] = vis.plot_param_importances(self.study)
            except Exception as e:
                logger.warning(f"Failed to generate param importance plot: {e}")

        # 3. Contour Plot (needs enough trials and parameters)
        if len(self.study.trials) >= 10 and len(self.study.best_params) >= 2:
            try:
                # Select two most important parameters for contour
                from optuna.importance import get_param_importances
                importances = get_param_importances(self.study)
                # Get top 2 parameters by importance
                top_params = list(importances.keys())[:2]
                figures["contour"] = vis.plot_contour(self.study, params=top_params)
            except Exception as e:
                logger.warning(f"Failed to generate contour plot: {e}")

        # 4. Parallel Coordinate (needs multiple parameters)
        if len(self.study.best_params) >= 2:
            try:
                figures["parallel"] = vis.plot_parallel_coordinate(self.study)
            except Exception as e:
                logger.warning(f"Failed to generate parallel coordinate plot: {e}")

        # Combine into single HTML
        html_content = self._build_html_report(figures)
        html_path.write_text(html_content, encoding="utf-8")

        return html_path

    def _build_html_report(self, figures: dict[str, Any]) -> str:
        """Build combined HTML report from multiple Plotly figures.

        Args:
            figures: Dictionary of figure names to Plotly figure objects

        Returns:
            Complete HTML string
        """
        # Get best trial info
        best_trial = self.study.best_trial
        metric_name = "CER" if self.language == "ja" else "WER"

        # Build params table rows
        params_rows = "\n".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>"
            for k, v in sorted(best_trial.params.items())
        )

        # Build figure HTML sections
        figure_sections = []
        figure_titles = {
            "history": "Optimization History",
            "importance": "Parameter Importance",
            "contour": "Contour Plot",
            "parallel": "Parallel Coordinate",
        }

        first_figure = True
        for name, fig in figures.items():
            if fig is not None:
                title = figure_titles.get(name, name.title())
                # Embed Plotly.js in the first figure for offline support
                fig_html = fig.to_html(
                    full_html=False,
                    include_plotlyjs=True if first_figure else False
                )
                first_figure = False
                figure_sections.append(f"""
                <div class="figure-section">
                    <h2>{title}</h2>
                    {fig_html}
                </div>
                """)

        figures_html = "\n".join(figure_sections)

        # Build complete HTML (Plotly.js is embedded in the first figure)
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Optimization Report: {self.study_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .summary-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .metric-box {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 4px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: 600;
        }}
        .figure-section {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .timestamp {{
            color: #999;
            font-size: 12px;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <h1>Optimization Report: {self.study_name}</h1>

    <div class="summary-card">
        <h2>Summary</h2>
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-value">{best_trial.value:.4f}</div>
                <div class="metric-label">Best {metric_name}</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{len(self.study.trials)}</div>
                <div class="metric-label">Total Trials</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">#{best_trial.number}</div>
                <div class="metric-label">Best Trial</div>
            </div>
        </div>

        <h3>Best Parameters</h3>
        <table>
            <thead>
                <tr><th>Parameter</th><th>Value</th></tr>
            </thead>
            <tbody>
                {params_rows}
            </tbody>
        </table>
    </div>

    {figures_html}

    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</body>
</html>"""

        return html

    def export_json(self) -> Path:
        """Export best parameters and metrics to JSON.

        Returns:
            Path to generated JSON file
        """
        json_path = self.output_dir / f"{self.study_name}.json"

        best_trial = self.study.best_trial
        metric_name = "cer" if self.language == "ja" else "wer"

        data = {
            "study_name": self.study_name,
            "vad_type": self.vad_type,
            "language": self.language,
            "best_trial": best_trial.number,
            f"best_{metric_name}": best_trial.value,
            "best_params": best_trial.params,
            "n_trials": len(self.study.trials),
            "created_at": datetime.now().isoformat(),
        }

        json_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return json_path

    def write_step_summary(self) -> bool:
        """Write to GitHub Actions Step Summary.

        Returns:
            True if Step Summary was written, False otherwise
        """
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if not summary_path:
            logger.debug("GITHUB_STEP_SUMMARY not set, skipping Step Summary")
            return False

        best_trial = self.study.best_trial
        metric_name = "CER" if self.language == "ja" else "WER"

        # Build parameters table
        params_rows = "\n".join(
            f"| {k} | {v} |" for k, v in sorted(best_trial.params.items())
        )

        summary = f"""
## 🎯 Optimization Results: {self.study_name}

| Metric | Value |
|--------|-------|
| Status | ✅ Completed |
| Best {metric_name} | {best_trial.value:.4f} |
| Total Trials | {len(self.study.trials)} |
| Best Trial | #{best_trial.number} |

### Best Parameters

| Parameter | Value |
|-----------|-------|
{params_rows}

📊 **詳細レポート**: Artifacts から `{self.study_name}.html` をダウンロードしてください

"""

        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(summary)

        return True


class ComparisonCSVExporter:
    """Export comparison CSV for multiple optimization studies.

    Args:
        output_path: Path to save CSV file

    Example:
        exporter = ComparisonCSVExporter(Path("results/comparison.csv"))
        exporter.add_study(study1, vad_type="silero", language="ja")
        exporter.add_study(study2, vad_type="silero", language="en")
        exporter.save()
    """

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.rows: list[dict[str, Any]] = []

    def add_study(
        self,
        study: "optuna.Study",
        vad_type: str,
        language: str,
    ) -> None:
        """Add a study to the comparison.

        Args:
            study: Optuna study object
            vad_type: VAD type name
            language: Language code
        """
        best_trial = study.best_trial
        metric_name = "cer" if language == "ja" else "wer"

        row = {
            "vad_type": vad_type,
            "language": language,
            "study_name": study.study_name,
            f"best_{metric_name}": best_trial.value,
            "best_trial": best_trial.number,
            "n_trials": len(study.trials),
            **{f"param_{k}": v for k, v in best_trial.params.items()},
        }
        self.rows.append(row)

    def add_result(
        self,
        vad_type: str,
        language: str,
        best_score: float,
        best_params: dict[str, Any],
        n_trials: int,
        study_name: str = "",
    ) -> None:
        """Add a result directly (without study object).

        Args:
            vad_type: VAD type name
            language: Language code
            best_score: Best score achieved
            best_params: Best parameters
            n_trials: Number of trials
            study_name: Optional study name
        """
        metric_name = "cer" if language == "ja" else "wer"

        row = {
            "vad_type": vad_type,
            "language": language,
            "study_name": study_name or f"{vad_type}_{language}",
            f"best_{metric_name}": best_score,
            "n_trials": n_trials,
            **{f"param_{k}": v for k, v in best_params.items()},
        }
        self.rows.append(row)

    def save(self) -> Path:
        """Save comparison to CSV file.

        Returns:
            Path to saved CSV file
        """
        import csv

        if not self.rows:
            raise ValueError("No studies added to comparison")

        # Collect all column names
        all_columns = set()
        for row in self.rows:
            all_columns.update(row.keys())

        # Define column order (fixed columns first, then params)
        fixed_columns = [
            "vad_type",
            "language",
            "study_name",
            "best_cer",
            "best_wer",
            "best_trial",
            "n_trials",
        ]
        param_columns = sorted(
            [c for c in all_columns if c.startswith("param_")]
        )
        columns = [c for c in fixed_columns if c in all_columns] + param_columns

        # Write CSV
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)

        logger.info(f"Comparison CSV saved: {self.output_path}")
        return self.output_path


def generate_report(
    study: "optuna.Study",
    output_dir: Path,
    vad_type: str | None = None,
    language: str | None = None,
) -> ReportPaths:
    """Convenience function to generate all reports for a study.

    Args:
        study: Optuna study object
        output_dir: Directory to save reports
        vad_type: VAD type name (optional, extracted from study name if not provided)
        language: Language code (optional, extracted from study name if not provided)

    Returns:
        ReportPaths with paths to generated files
    """
    # Try to extract vad_type and language from study name if not provided
    if (vad_type is None or language is None) and study.study_name:
        parts = study.study_name.split("_")
        if len(parts) >= 2:
            vad_type = vad_type or parts[0]
            language = language or parts[-1]

    report = OptimizationReport(
        study=study,
        output_dir=output_dir,
        vad_type=vad_type,
        language=language,
    )

    return report.generate_all()


__all__ = [
    "OptimizationReport",
    "ComparisonCSVExporter",
    "ReportPaths",
    "generate_report",
]
