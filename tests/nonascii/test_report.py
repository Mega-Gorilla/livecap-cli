"""Report rendering regressions for the non-ASCII boundary inventory."""

from __future__ import annotations

from .report import render_summary, render_table
from .registry import BOUNDARIES, Method


def test_not_applicable_rows_are_not_reported_as_unverified():
    table = render_table([])

    logging_row = next(
        line for line in table.splitlines() if "ログファイルの出力先パス" in line
    )
    assert "— 対象外" in logging_row
    assert "— 未確定" not in logging_row


def test_summary_excludes_not_applicable_rows_from_runtime_denominator():
    summary = render_summary([])
    applicable = [b for b in BOUNDARIES if b.candidate_method is not Method.NOT_APPLICABLE]
    verified = sum(1 for b in applicable if b.verified_method)
    not_applicable = len(BOUNDARIES) - len(applicable)

    assert f"applicable 行: **{verified} / {len(applicable)}**" in summary
    assert f"**非該当**: **{not_applicable} 行**" in summary
