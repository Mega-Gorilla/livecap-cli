"""confidence filter の部分ごとの判定 (#518)。

engine が 1 つの区間を 2 つ以上の部分に分けて書き起こしたとき (``TranscriptionResult.parts``)、filter は
部分ごとに判定し、捨てる部分の文だけを除く。区間全体の値 (部分の長さで重み付けした平均) で判定すると、
非発話の長い部分に引きずられて、はっきりした発話の部分まで捨てるため。
"""

from __future__ import annotations

import json
import logging

import pytest

from livecap_cli.engines.base_engine import (
    EngineConfidence,
    TranscriptionPart,
    TranscriptionResult,
    combine_part_confidences,
)
from livecap_cli.transcription.confidence_filter import FilterConfig, apply_filter, filter_result

_LOGGER = "livecap_cli.transcription.confidence_filter"


def _whisper_result(*parts: tuple[str, float, float, float], separator: str = " ") -> TranscriptionResult:
    """WhisperS2T engine が返す形 (部分ごとの文・時刻・no_speech_prob)。"""
    built = tuple(
        TranscriptionPart(text, EngineConfidence(no_speech_prob=nsp), start=start, end=end)
        for text, start, end, nsp in parts
    )
    return TranscriptionResult(
        text=separator.join(p.text for p in built),
        confidence=1.0,
        engine_confidence=combine_part_confidences(built),
        parts=built,
        part_separator=separator,
    )


def _apply(result, mode="on"):
    return apply_filter(result, FilterConfig(mode=mode), source_id="test", engine_name="whispers2t")


class TestPerPartDecision:
    def test_clear_speech_survives_a_long_non_speech_part(self):
        """29 秒の非発話らしい部分 (0.98) と 4 秒のはっきりした発話 (0.05)。区間全体の値は 0.867 (> 0.71) で、
        全体で判定すると発話まで捨てる。部分ごとなら、発話の部分だけが残る。"""
        result = _whisper_result(("Thanks for watching!", 0.0, 29.0, 0.98), ("Hello everyone.", 29.0, 33.0, 0.05))
        assert result.engine_confidence.no_speech_prob == pytest.approx(0.867, abs=1e-3)

        kept = _apply(result)

        assert kept is not None
        assert kept.text == "Hello everyone."
        assert kept.parts == () and kept.part_separator == ""
        assert kept.engine_confidence.no_speech_prob == pytest.approx(0.05)

    def test_all_parts_rejected_drops_the_segment(self):
        result = _whisper_result(("a", 0.0, 29.0, 0.9), ("b", 29.0, 40.0, 0.8))
        assert _apply(result) is None

    def test_no_part_rejected_keeps_the_result_as_is(self):
        result = _whisper_result(("a", 0.0, 29.0, 0.1), ("b", 29.0, 40.0, 0.2))
        assert _apply(result) is result

    def test_kept_parts_are_rejoined_with_the_separator(self):
        result = _whisper_result(
            ("今日は", 0.0, 29.0, 0.1), ("ご視聴ありがとうございました", 29.0, 58.0, 0.95), ("晴れです", 58.0, 60.0, 0.2),
            separator="",
        )
        kept = _apply(result)
        assert kept.text == "今日は晴れです"
        assert [p.text for p in kept.parts] == ["今日は", "晴れです"]
        assert kept.engine_confidence.no_speech_prob == pytest.approx((0.1 * 29 + 0.2 * 2) / 31)

    def test_a_result_without_parts_is_judged_as_a_whole(self):
        """部分の無い結果は今までと同じ (結果全体の値で判定)。"""
        reject = TranscriptionResult("x", 1.0, EngineConfidence(no_speech_prob=0.9))
        keep = TranscriptionResult("y", 1.0, EngineConfidence(no_speech_prob=0.1))
        assert _apply(reject) is None
        assert _apply(keep) is keep

    def test_a_part_without_signals_is_kept(self):
        """値の無い部分は、結果全体のときと同じく fail-open (捨てない)。"""
        built = (
            TranscriptionPart("a", EngineConfidence(), start=0.0, end=29.0),
            TranscriptionPart("b", EngineConfidence(no_speech_prob=0.9), start=29.0, end=40.0),
        )
        result = TranscriptionResult("a b", 1.0, combine_part_confidences(built), parts=built, part_separator=" ")
        assert _apply(result).text == "a"


class TestModes:
    def test_observe_logs_every_part_and_keeps_the_result(self, caplog):
        result = _whisper_result(("a", 0.0, 29.0, 0.98), ("b", 29.0, 33.0, 0.05))
        with caplog.at_level(logging.INFO, logger=_LOGGER):
            out = _apply(result, mode="observe")
        assert out is result
        payloads = [json.loads(r.getMessage().split(": ", 1)[1]) for r in caplog.records]
        assert [(p["part"], p["text"], p["decision"]) for p in payloads] == [("1/2", "a", "reject"), ("2/2", "b", "pass")]

    def test_on_logs_only_the_rejected_parts(self, caplog):
        result = _whisper_result(("a", 0.0, 29.0, 0.98), ("b", 29.0, 33.0, 0.05))
        with caplog.at_level(logging.INFO, logger=_LOGGER):
            _apply(result)
        payloads = [json.loads(r.getMessage().split(": ", 1)[1]) for r in caplog.records]
        assert [(p["part"], p["decision"]) for p in payloads] == [("1/2", "reject")]

    def test_a_whole_result_log_has_no_part_key(self, caplog):
        with caplog.at_level(logging.INFO, logger=_LOGGER):
            _apply(TranscriptionResult("x", 1.0, EngineConfidence(no_speech_prob=0.9)))
        payload = json.loads(caplog.records[0].getMessage().split(": ", 1)[1])
        assert "part" not in payload

    def test_off_returns_the_result_as_is(self):
        result = _whisper_result(("a", 0.0, 29.0, 0.98), ("b", 29.0, 33.0, 0.05))
        assert _apply(result, mode="off") is result


class TestFilterResult:
    def test_matches_apply_filter_without_logging(self, caplog):
        result = _whisper_result(("a", 0.0, 29.0, 0.98), ("b", 29.0, 33.0, 0.05))
        with caplog.at_level(logging.INFO, logger=_LOGGER):
            kept = filter_result(result, FilterConfig(), engine_name="whispers2t")
        assert kept.text == "b"
        assert not caplog.records

    def test_non_on_modes_return_the_result(self):
        result = _whisper_result(("a", 0.0, 29.0, 0.98), ("b", 29.0, 33.0, 0.05))
        assert filter_result(result, FilterConfig(mode="observe"), engine_name="whispers2t") is result
