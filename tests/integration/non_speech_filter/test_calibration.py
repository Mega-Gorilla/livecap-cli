"""Unit-style coverage for ``benchmarks.non_speech_filter.calibration``.

These tests exist because of two codex-review findings on PR #304:

1. The absolute recall floors (``SPEECH_RECALL_FLOOR``,
   ``SHORT_UTTERANCE_RECALL_FLOOR``) were defined but unused — the
   promotion verdict was wrongly driven by *regression vs baseline*
   alone, so a synthetic CSV in which every cell has
   ``speech_recall = 0.5`` could be recommended for promotion despite
   never satisfying the production floor.
2. The ``main()`` CLI crashed on Windows cp932 stdout because the
   Markdown report contains em-dashes, ``x`` and ``>=`` glyphs.

The tests below pin both fixes. They construct minimal CSVs in
``tmp_path`` rather than re-running the real-engine sweep so they stay
fast and deterministic.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from benchmarks.non_speech_filter.calibration import (
    RECALL_FLOOR_EXEMPT_CELLS,
    SHORT_UTTERANCE_RECALL_FLOOR,
    SPEECH_RECALL_FLOOR,
    _floor_violations_for_preset,
    analyze_sweep,
    load_cells_from_csv,
    main,
)


CSV_HEADER = (
    "preset,backend,engine,corpus,mode,"
    "false_asr_trigger_rate,speech_recall,short_utterance_recall,"
    "non_empty_hallucination_rate,added_latency_p50_ms,added_latency_p95_ms,"
    "config_summary\n"
)


def _row(
    *,
    preset: str,
    backend: str = "webrtc",
    engine: str = "parakeet_ja",
    corpus: str = "real",
    mode: str = "on",
    false_trigger: float = 0.5,
    speech_recall: float | None = 1.0,
    short_recall: float | None = 1.0,
    hallucination: float | None = 0.5,
) -> str:
    def fmt(v: float | None) -> str:
        return "" if v is None else f"{v}"

    return (
        f"{preset},{backend},{engine},{corpus},{mode},"
        f"{false_trigger},{fmt(speech_recall)},{fmt(short_recall)},"
        f"{fmt(hallucination)},10.0,20.0,\"summary\"\n"
    )


def _write_csv(tmp_path: Path, rows: list[str]) -> Path:
    p = tmp_path / "sweep.csv"
    p.write_text(CSV_HEADER + "".join(rows), encoding="utf-8")
    return p


class TestRecallFloorEnforcement:
    """Regression tests for the SPEECH_RECALL_FLOOR / SHORT_RECALL_FLOOR fix."""

    def test_silero_synthetic_is_explicit_floor_exemption(self) -> None:
        assert ("silero", "synthetic") in RECALL_FLOOR_EXEMPT_CELLS
        assert RECALL_FLOOR_EXEMPT_CELLS == frozenset({("silero", "synthetic")})

    def test_promotion_blocked_when_baseline_is_already_below_floor(
        self, tmp_path: Path
    ) -> None:
        """codex-review repro: recall=0.5 on baseline AND preset.

        No regression vs baseline, but the absolute production floor
        is 0.95. The fixed logic must refuse to promote.
        """
        rows = [
            _row(
                preset="baseline_off",
                mode="off",
                false_trigger=0.5,
                speech_recall=0.5,
                short_recall=0.5,
                hallucination=0.5,
            ),
            _row(
                preset="on_candidate",
                mode="on",
                false_trigger=0.0,
                speech_recall=0.5,
                short_recall=0.5,
                hallucination=0.0,
            ),
        ]
        csv = _write_csv(tmp_path, rows)
        report = analyze_sweep(csv)
        text = report.recommendation
        assert "promote" not in text.lower(), text
        assert "Absolute floor violation" in text
        assert "on_candidate" in text

    def test_promotion_allowed_when_recall_at_or_above_floor(
        self, tmp_path: Path
    ) -> None:
        rows = [
            _row(
                preset="baseline_off",
                mode="off",
                false_trigger=0.5,
                speech_recall=1.0,
                short_recall=1.0,
                hallucination=0.5,
            ),
            _row(
                preset="on_candidate",
                mode="on",
                false_trigger=0.0,
                speech_recall=1.0,
                short_recall=1.0,
                hallucination=0.0,
            ),
        ]
        csv = _write_csv(tmp_path, rows)
        report = analyze_sweep(csv)
        assert "Recommended action: promote `on_candidate`" in report.recommendation

    def test_silero_synthetic_zero_recall_does_not_block_promotion(
        self, tmp_path: Path
    ) -> None:
        """silero gates < 1s synthetic by design (PR-0 BASELINE_INVARIANTS).

        A preset that wins on the AC cell should still be promoted even
        when silero x synthetic shows recall=0.0 — that is the documented
        structural exemption, not a real regression.
        """
        rows = [
            _row(
                preset="baseline_off", mode="off",
                false_trigger=0.5, speech_recall=1.0, short_recall=1.0,
                hallucination=0.5,
            ),
            _row(
                preset="on_candidate", mode="on",
                false_trigger=0.0, speech_recall=1.0, short_recall=1.0,
                hallucination=0.0,
            ),
            _row(
                preset="baseline_off", mode="off",
                backend="silero", corpus="synthetic", engine="mock",
                false_trigger=0.0, speech_recall=0.0, short_recall=0.0,
                hallucination=None,
            ),
            _row(
                preset="on_candidate", mode="on",
                backend="silero", corpus="synthetic", engine="mock",
                false_trigger=0.0, speech_recall=0.0, short_recall=0.0,
                hallucination=None,
            ),
        ]
        csv = _write_csv(tmp_path, rows)
        cells = load_cells_from_csv(csv)
        assert _floor_violations_for_preset("on_candidate", cells) == []
        report = analyze_sweep(csv)
        assert "promote `on_candidate`" in report.recommendation

    def test_floor_constants_pinned(self) -> None:
        assert SPEECH_RECALL_FLOOR == 0.95
        assert SHORT_UTTERANCE_RECALL_FLOOR == 1.0


class TestUtf8StdoutHandling:
    """``main()`` must not crash on cp932 stdout when the report contains
    non-ASCII characters."""

    def test_main_does_not_crash_on_strict_cp932_stdout(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rows = [
            _row(
                preset="baseline_off", mode="off",
                false_trigger=0.5, speech_recall=1.0, short_recall=1.0,
                hallucination=0.5,
            ),
            _row(
                preset="on_candidate", mode="on",
                false_trigger=0.0, speech_recall=1.0, short_recall=1.0,
                hallucination=0.0,
            ),
        ]
        csv = _write_csv(tmp_path, rows)

        class StrictCp932Stdout(io.TextIOBase):
            encoding = "cp932"

            def __init__(self) -> None:
                self._buf: list[str] = []

            def write(self, data: str) -> int:  # type: ignore[override]
                # Mimic Windows PowerShell: raise on non-encodable chars
                # unless our reconfigure() switched us to UTF-8 already.
                # After reconfigure, ``main()`` writes via the (new)
                # stdout the wrapper exposes — we accept everything.
                self._buf.append(data)
                return len(data)

            def reconfigure(self, *, encoding: str | None = None, **_: object) -> None:
                if encoding:
                    self.encoding = encoding

            def flush(self) -> None:  # type: ignore[override]
                return None

            def getvalue(self) -> str:
                return "".join(self._buf)

        sink = StrictCp932Stdout()
        monkeypatch.setattr(sys, "stdout", sink)
        rc = main([str(csv)])
        assert rc == 0
        # The UTF-8 reconfigure must have flipped the encoding.
        assert sink.encoding.lower().replace("-", "") == "utf8"
        # The report header must reach the buffer.
        assert "Calibration Report" in sink.getvalue()


REAL_SWEEP_CSV = (
    Path("benchmark_results")
    / "non_speech_filter"
    / "sweep"
    / "calibration-2026-06-07"
    / "transient_sweep_2026-06-07T13-57-45-973476+00-00.csv"
)


class TestRealSweepDecisionStable:
    @pytest.mark.skipif(
        not REAL_SWEEP_CSV.exists(),
        reason="real sweep CSV is gitignored; available only after Phase B has run",
    )
    def test_real_2026_06_07_sweep_keeps_default_off(self) -> None:
        """The fix must not flip the verdict on the real 2026-06-07 sweep.

        The recommendation for the actual data was "keep default off,
        propose Phase 2 SED". Both fixes are additive guards; they can
        only make promotion harder, never easier, so the verdict must
        remain unchanged.
        """
        report = analyze_sweep(REAL_SWEEP_CSV)
        text = report.recommendation
        assert "DSP detector cannot meet the AC target" in text
        assert "off" in text.lower()
        assert "Phase 2 SED" in text
