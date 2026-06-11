"""PR-A.3 (Issue #308 v3.2) — confidence_filter sweep 軸の pin test。

``benchmarks/non_speech_filter/sweep.py`` の ``run_sweep()`` が
``filter_mode ∈ {off, observe, on}`` を直交軸として展開し、各 cell に
``filter_mode`` が記録されることを軽量 minimal sweep で verify する。
実機 ASR engine は使わず、MockEngine + silero VAD + synthetic corpus
のみで動作 (CI でも数秒で完走)。
"""
from __future__ import annotations

import pytest

from benchmarks.non_speech_filter.sweep import (
    SweepCellResult,
    SweepReport,
    default_named_presets,
    run_sweep,
)


@pytest.mark.evaluation_harness
class TestFilterModeSweepAxis:
    """``run_sweep`` が filter_mode 3 値を nested loop で回すことを pin。"""

    def test_minimal_sweep_emits_three_filter_modes_per_preset(self) -> None:
        """1 preset × 1 backend × 1 engine (mock) × synthetic corpus × 3 mode で
        最低 3 cell が出力される (synthetic corpus は固定 1 件)。
        """
        # default_named_presets() は 8 件、先頭 1 件だけに絞って minimal sweep
        single_preset = [default_named_presets()[0]]
        report: SweepReport = run_sweep(
            backends=["silero"],
            engines=[],          # 空 list → MockEngine 1 種のみ
            corpus_dir=None,     # synthetic corpus のみ (real corpus 不要)
            device="cpu",
            presets=single_preset,
        )

        assert isinstance(report, SweepReport)
        # 1 preset × 1 backend × 1 engine (mock) × synthetic corpus × 3 mode = 3 cell
        # synthetic corpus は build_synthetic_corpus() で複数 item を 1 corpus に
        # まとめているため、cell 単位では 1 corpus として扱われる。
        assert len(report.cells) == 3, (
            f"3 filter_mode で 3 cell 出力されること、actual={len(report.cells)}"
        )

        filter_modes = {cell.filter_mode for cell in report.cells}
        assert filter_modes == {"off", "observe", "on"}, (
            f"3 mode 完全網羅、actual={filter_modes}"
        )

    def test_filter_mode_appears_in_config_summary(self) -> None:
        """``config_summary`` 文字列に ``confidence_filter=<mode>`` が含まれ、
        decision document 解析時に trace 可能であること。
        """
        single_preset = [default_named_presets()[0]]
        report = run_sweep(
            backends=["silero"],
            engines=[],
            corpus_dir=None,
            device="cpu",
            presets=single_preset,
        )

        for cell in report.cells:
            assert f"confidence_filter={cell.filter_mode}" in cell.config_summary, (
                f"config_summary に filter_mode の trace が含まれること、"
                f"cell={cell.filter_mode}, summary={cell.config_summary!r}"
            )


@pytest.mark.evaluation_harness
class TestSweepReportSerialisation:
    """CSV / Markdown 出力に ``filter_mode`` 列が含まれることを pin。"""

    def test_csv_header_includes_filter_mode(self) -> None:
        single_preset = [default_named_presets()[0]]
        report = run_sweep(
            backends=["silero"],
            engines=[],
            corpus_dir=None,
            device="cpu",
            presets=single_preset,
        )
        csv_text = report.to_csv()
        # 1 行目が header
        header = csv_text.splitlines()[0]
        assert "filter_mode" in header, f"CSV header に filter_mode 列、actual={header}"

    def test_markdown_table_includes_filter_mode_column(self) -> None:
        single_preset = [default_named_presets()[0]]
        report = run_sweep(
            backends=["silero"],
            engines=[],
            corpus_dir=None,
            device="cpu",
            presets=single_preset,
        )
        md = report.to_markdown()
        assert "FilterMode" in md, "Markdown table header に FilterMode 列があること"

    def test_sweep_cell_result_includes_filter_mode_field(self) -> None:
        """``SweepCellResult`` dataclass に ``filter_mode`` field が存在すること。"""
        import dataclasses

        fields = {f.name for f in dataclasses.fields(SweepCellResult)}
        assert "filter_mode" in fields
        assert "mode" in fields  # transient_filter mode と直交する
