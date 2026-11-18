"""長尺の評価の採点 (benchmarks/longform/scoring.py、#488)。"""

from __future__ import annotations

import pytest

pytest.importorskip("jiwer")

from benchmarks.common.error_breakdown import ErrorCounts, count_errors  # noqa: E402
from benchmarks.common.yodas2 import Yodas2Segment, Yodas2Video  # noqa: E402
from benchmarks.longform.scoring import (  # noqa: E402
    CONDITIONS,
    MAIN_CONDITION,
    ScoreCondition,
    hypothesis_text,
    score_engine,
)


def _video(audio_id: str, speaker: str, segments: list[tuple[float, float, str, str]]) -> Yodas2Video:
    return Yodas2Video(
        audio_id, "test", speaker, 600.0, "other", (), -20.0, "ja000", "00000001",
        tuple(Yodas2Segment(s, e, t, st) for s, e, t, st in segments),
    )


def test_main_condition_is_fillers_without_notation_and_listed_first() -> None:
    assert MAIN_CONDITION == ScoreCondition("fillers", include_notation=False)
    assert CONDITIONS[0] == MAIN_CONDITION
    assert len(set(CONDITIONS)) == len(CONDITIONS) == 8


def test_hypothesis_drops_asr_segments_whose_middle_is_in_an_unscored_span() -> None:
    segments = [(0.0, 2.0, "はい"), (2.0, 6.0, "外す"), (5.5, 9.0, "残す"), (9.0, 10.0, "")]
    assert hypothesis_text(segments, [(2.5, 5.0)]) == "はい 残す"
    assert hypothesis_text(segments, []) == "はい 外す 残す"


def test_excluded_caption_and_matching_asr_output_both_leave_the_score() -> None:
    video = _video("A", "1", [(0, 2, "こんにちは", "ok"), (2, 4, "聞き取れない", "excluded"), (4, 6, "さようなら", "ok")])
    hyp = {"A": [(0.0, 2.0, "こんにちは"), (2.0, 4.0, "ききとれない"), (4.0, 6.0, "さようなら")]}
    score = score_engine([video], hyp, ScoreCondition("plain", include_notation=True))
    assert score.pooled == ErrorCounts(reference_length=10)


def test_notation_segments_are_scored_only_when_included() -> None:
    video = _video("A", "1", [(0, 2, "SELECTする", "notation"), (2, 4, "はい", "ok")])
    hyp = {"A": [(0.0, 2.0, "セレクトする"), (2.0, 4.0, "はい")]}
    with_notation = score_engine([video], hyp, ScoreCondition("plain", include_notation=True))
    without = score_engine([video], hyp, ScoreCondition("plain", include_notation=False))
    assert with_notation.pooled.errors > 0
    assert without.pooled == ErrorCounts(reference_length=2)


def test_pooled_and_speaker_mean() -> None:
    short = _video("S", "1", [(0, 2, "あいうえお", "ok")])
    long_a = _video("L1", "2", [(0, 2, "かきくけこさしすせそ", "ok")])
    long_b = _video("L2", "2", [(0, 2, "たちつてとなにぬねの", "ok")])
    hyp = {"S": [(0.0, 2.0, "あいうえ")], "L1": [(0.0, 2.0, "かきくけこさしすせそ")], "L2": [(0.0, 2.0, "たちつてとなにぬねの")]}
    score = score_engine([short, long_a, long_b], hyp, ScoreCondition("plain", True))
    assert score.pooled == ErrorCounts(deletions=1, reference_length=25)
    assert score.by_speaker["1"].rate == pytest.approx(0.2)
    assert score.by_speaker["2"].rate == 0.0
    assert score.speaker_mean == pytest.approx(0.1)
    assert set(score.by_video) == {"S", "L1", "L2"}
    assert score.to_dict()["condition"] == "plain"


def test_main_condition_forgives_fillers_on_either_side() -> None:
    video = _video("A", "1", [(0, 3, "九重で雪山", "ok")])
    hyp = {"A": [(0.0, 3.0, "えっと九重で雪山")]}
    assert score_engine([video], hyp, MAIN_CONDITION).pooled.errors == 0
    assert score_engine([video], hyp, ScoreCondition("plain", True)).pooled == count_errors("九重で雪山", "えっと九重で雪山", lang="ja")


def test_missing_transcription_is_an_error_not_a_skip() -> None:
    video = _video("A", "1", [(0, 1, "はい", "ok")])
    with pytest.raises(KeyError):
        score_engine([video], {}, MAIN_CONDITION)
    with pytest.raises(ValueError):
        score_engine([], {}, MAIN_CONDITION)
