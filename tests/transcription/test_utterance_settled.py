"""Tests for ``on_utterance_settled`` callback (Issue #332).

Covers Tier 1 (7 hook point) + delivery ordering (callback path: result-then-
settled, async generator path: settled-before-yield) + coalescer 0-2 件 emit
+ engine_error fallback + reason enumeration.

既存 ``MockEngine`` / ``MockVADProcessor`` / ``MockAudioSource`` /
``FilteringMockEngine`` (test_stream.py) を import 再利用。
"""

from __future__ import annotations

import asyncio
from typing import Any, List

import numpy as np
import pytest

from livecap_cli import (
    REASON_EMPTY_AUDIO,
    REASON_ENERGY_GATE,
    REASON_ENGINE_EMPTY,
    REASON_FILTER_REJECT,
    StreamTranscriber,
    UtteranceSettledEvent,
)
from livecap_cli.engines.base_engine import (
    EngineConfidence,
    TranscriptionResult as EngineTranscriptionResult,
)
from livecap_cli.transcription.confidence_filter import FilterConfig
from livecap_cli.transcription.stream import EngineError
from livecap_cli.vad import VADSegment

from tests.transcription.test_stream import (
    FilteringMockEngine,
    MockAudioSource,
    MockEngine,
    MockVADProcessor,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_segment(
    audio: np.ndarray | None = None,
    start: float = 0.0,
    end: float = 1.0,
    is_final: bool = True,
) -> VADSegment:
    """Convenience VADSegment factory (non-silent default audio)."""
    if audio is None:
        audio = np.full(16000, 0.5, dtype=np.float32)
    return VADSegment(
        audio=audio, start_time=start, end_time=end, is_final=is_final
    )


def _new_transcriber(engine: Any, **kwargs: Any) -> StreamTranscriber:
    """StreamTranscriber + ``MockVADProcessor`` factory with permissive defaults."""
    vad = kwargs.pop("vad_processor", MockVADProcessor())
    defaults = dict(
        engine_min_rms_dbfs=float("-inf"),  # EnergyGate opt-out unless overridden
        filter_config=FilterConfig(mode="off"),  # filter opt-out unless overridden
    )
    defaults.update(kwargs)
    return StreamTranscriber(engine=engine, vad_processor=vad, **defaults)


# ---------------------------------------------------------------------------
# Tier 1 path tests (drop paths)
# ---------------------------------------------------------------------------


class TestSettledOnDropPaths:
    """Issue #332 Tier 1 path 1-5 (drop reasons)."""

    def test_settled_on_empty_audio(self) -> None:
        engine = MockEngine()
        seg = _make_segment(audio=np.array([], dtype=np.float32))
        transcriber = _new_transcriber(
            engine, vad_processor=MockVADProcessor(segments=[seg])
        )
        events: List[UtteranceSettledEvent] = []
        transcriber.set_callbacks(on_utterance_settled=events.append)

        transcriber.feed_audio(np.zeros(160, dtype=np.float32), 16000)

        assert len(events) == 1
        assert events[0].emitted is False
        assert events[0].reason == REASON_EMPTY_AUDIO
        assert events[0].utterance_start_time == 0.0
        assert events[0].utterance_end_time == 1.0

    def test_settled_on_energy_gate(self) -> None:
        engine = MockEngine()
        seg = _make_segment(audio=np.zeros(16000, dtype=np.float32))
        transcriber = _new_transcriber(
            engine,
            vad_processor=MockVADProcessor(segments=[seg]),
            engine_min_rms_dbfs=-10.0,  # 高すぎる threshold で確実に drop
        )
        events: List[UtteranceSettledEvent] = []
        transcriber.set_callbacks(on_utterance_settled=events.append)

        transcriber.feed_audio(np.zeros(160, dtype=np.float32), 16000)

        assert len(events) == 1
        assert events[0].emitted is False
        assert events[0].reason == REASON_ENERGY_GATE
        assert engine.call_count == 0  # engine 呼ばれず

    def test_settled_on_filter_reject(self) -> None:
        """GUI #362 主因: confidence_filter mode=on で reject."""
        engine = FilteringMockEngine(
            return_text="ノイズ", no_speech_prob=0.8  # threshold 0.5 を超え reject
        )
        seg = _make_segment()
        transcriber = _new_transcriber(
            engine,
            vad_processor=MockVADProcessor(segments=[seg]),
            filter_config=FilterConfig(mode="on"),
        )
        events: List[UtteranceSettledEvent] = []
        transcriber.set_callbacks(on_utterance_settled=events.append)

        transcriber.feed_audio(np.zeros(160, dtype=np.float32), 16000)

        assert len(events) == 1
        assert events[0].emitted is False
        assert events[0].reason == REASON_FILTER_REJECT

    def test_settled_on_engine_empty(self) -> None:
        engine = MockEngine(return_text="   ")  # whitespace only → engine empty
        seg = _make_segment()
        transcriber = _new_transcriber(
            engine, vad_processor=MockVADProcessor(segments=[seg])
        )
        events: List[UtteranceSettledEvent] = []
        transcriber.set_callbacks(on_utterance_settled=events.append)

        transcriber.feed_audio(np.zeros(160, dtype=np.float32), 16000)

        assert len(events) == 1
        assert events[0].emitted is False
        assert events[0].reason == REASON_ENGINE_EMPTY


class TestSettledOnEngineError:
    """Issue #332 Tier 1 path 5 (engine_error) + __cause__ fallback (rev5)."""

    def test_settled_on_engine_error_with_cause(self) -> None:
        """engine.transcribe() throws RuntimeError → chain で __cause__ = RuntimeError."""
        engine = MockEngine(should_fail=True)
        seg = _make_segment()
        transcriber = _new_transcriber(
            engine, vad_processor=MockVADProcessor(segments=[seg])
        )
        events: List[UtteranceSettledEvent] = []
        transcriber.set_callbacks(on_utterance_settled=events.append)

        transcriber.feed_audio(np.zeros(160, dtype=np.float32), 16000)

        assert len(events) == 1
        assert events[0].emitted is False
        assert events[0].reason == "engine_error:RuntimeError"

    def test_engine_error_reason_no_cause_fallback(self) -> None:
        """__cause__ is None でも "engine_error:NoneType" にならず EngineError 型名を使う."""
        engine = MockEngine()
        transcriber = _new_transcriber(engine)
        # raise EngineError(...) を直接 (from なし、__cause__ = None)
        err = EngineError("standalone")
        assert err.__cause__ is None
        reason = transcriber._engine_error_reason(err)
        assert reason == "engine_error:EngineError"


# ---------------------------------------------------------------------------
# Tier 1 success path + coalescer 0-2 emission
# ---------------------------------------------------------------------------


class TestSettledOnCoalescerEmissions:
    """Issue #332 Tier 1 path 6-7 (coalescer push / flush)."""

    def test_settled_on_coalescer_emit_single(self) -> None:
        """長い text → 即時 emit、settled(True, None) 1 回。"""
        engine = MockEngine(return_text="今日は良い天気ですね")  # long → emit
        seg = _make_segment()
        transcriber = _new_transcriber(
            engine, vad_processor=MockVADProcessor(segments=[seg])
        )
        events: List[UtteranceSettledEvent] = []
        transcriber.set_callbacks(on_utterance_settled=events.append)

        transcriber.feed_audio(np.zeros(160, dtype=np.float32), 16000)

        assert len(events) == 1
        assert events[0].emitted is True
        assert events[0].reason is None

    def test_settled_on_coalescer_emit_dual(self) -> None:
        """coalescer push が 2 件 emit (pending 窓外 flush + 新 emit) で settled 2 回。

        result_coalescer.py:75-98 の窓外 flush path を pin。
        """
        engine_short = MockEngine(return_text="は")  # 1 char → pending
        engine_long = MockEngine(return_text="こんにちは、良い天気ですね")  # long → emit

        # 1 番目: 短 text、merge_window=5s default、start=0/end=1
        seg1 = _make_segment(start=0.0, end=1.0)
        # 2 番目: 長 text、gap > 5s で窓外 → pending を単独 flush + 新 emit
        seg2 = _make_segment(start=10.0, end=11.0)

        vad = MockVADProcessor(segments=[seg1, seg2])

        # 短 text → 長 text を切替える MultiEngine 簡易代替: VAD return order に合わせ
        # call_count で切替
        class SeqEngine:
            def __init__(self):
                self.call_count = 0

            def transcribe(self, audio, sample_rate):
                self.call_count += 1
                if self.call_count == 1:
                    return EngineTranscriptionResult(
                        text="は", confidence=0.9,
                    )
                return EngineTranscriptionResult(
                    text="こんにちは、良い天気ですね", confidence=0.9,
                )

            def get_required_sample_rate(self):
                return 16000

        engine = SeqEngine()
        transcriber = _new_transcriber(engine, vad_processor=vad)
        events: List[UtteranceSettledEvent] = []
        transcriber.set_callbacks(on_utterance_settled=events.append)

        # 2 chunk を feed (MockVAD は 1 chunk あたり 1 segment 返す)
        transcriber.feed_audio(np.zeros(160, dtype=np.float32), 16000)
        transcriber.feed_audio(np.zeros(160, dtype=np.float32), 16000)

        # 期待: settled 2 件 (pending "は" の単独 flush + 新 "こんにちは..." emit)
        success_events = [e for e in events if e.emitted]
        assert len(success_events) == 2
        for ev in success_events:
            assert ev.reason is None

    def test_settled_on_coalescer_pending_no_event(self) -> None:
        """短 text → pending のみ → settled 発火しない。"""
        engine = MockEngine(return_text="は")  # 1 char → pending
        seg = _make_segment()
        transcriber = _new_transcriber(
            engine, vad_processor=MockVADProcessor(segments=[seg])
        )
        events: List[UtteranceSettledEvent] = []
        transcriber.set_callbacks(on_utterance_settled=events.append)

        transcriber.feed_audio(np.zeros(160, dtype=np.float32), 16000)

        # pending のまま、まだ utterance 終了せず → settled 0 件
        assert len(events) == 0

    def test_settled_on_finalize(self) -> None:
        """finalize() で coalescer force flush → settled(True, None) 発火。"""
        engine = MockEngine(return_text="は")  # 短 text → pending
        seg = _make_segment()
        transcriber = _new_transcriber(
            engine, vad_processor=MockVADProcessor(segments=[seg])
        )
        events: List[UtteranceSettledEvent] = []
        transcriber.set_callbacks(on_utterance_settled=events.append)

        transcriber.feed_audio(np.zeros(160, dtype=np.float32), 16000)
        assert len(events) == 0  # pending 中

        results = transcriber.finalize()
        # finalize 内の force flush で emit
        assert len(results) == 1
        assert len(events) == 1
        assert events[0].emitted is True
        assert events[0].reason is None


# ---------------------------------------------------------------------------
# No-event paths
# ---------------------------------------------------------------------------


class TestNoSettledForInterim:
    """Interim path で reject されても settled 発火しない (utterance ongoing)."""

    def test_no_settled_for_interim_reject(self) -> None:
        engine = FilteringMockEngine(return_text="ノイズ", no_speech_prob=0.8)
        # is_final=False の interim segment を投入
        seg = _make_segment(is_final=False)
        transcriber = _new_transcriber(
            engine,
            vad_processor=MockVADProcessor(segments=[seg]),
            filter_config=FilterConfig(mode="on"),
        )
        settled_events: List[UtteranceSettledEvent] = []
        interim_events: List[Any] = []
        transcriber.set_callbacks(
            on_interim=interim_events.append,
            on_utterance_settled=settled_events.append,
        )

        transcriber.feed_audio(np.zeros(160, dtype=np.float32), 16000)

        # interim path での filter reject では settled 発火しない
        assert len(settled_events) == 0


# ---------------------------------------------------------------------------
# Delivery ordering
# ---------------------------------------------------------------------------


class TestDeliveryOrdering:
    """rev5/rev6: callback path = result-after-settled-before-result NG、
    generator path = settled-before-yield."""

    def test_settled_fires_after_emit_result_callback(self) -> None:
        """feed_audio path: on_result → on_utterance_settled の順。"""
        engine = MockEngine(return_text="今日は良い天気ですね")
        seg = _make_segment()
        transcriber = _new_transcriber(
            engine, vad_processor=MockVADProcessor(segments=[seg])
        )
        observations: List[str] = []
        transcriber.set_callbacks(
            on_result=lambda r: observations.append(f"result:{r.text}"),
            on_utterance_settled=lambda e: observations.append(
                f"settled:{e.emitted}"
            ),
        )

        transcriber.feed_audio(np.zeros(160, dtype=np.float32), 16000)

        # ordering: result first, settled after
        assert len(observations) == 2
        assert observations[0].startswith("result:")
        assert observations[1] == "settled:True"

    def test_settled_fires_before_yield_async(self) -> None:
        """transcribe_async path: yield 直前に settled 発火。

        Generator から 1 件取り出して即 break しても settled は発火済。
        """
        engine = MockEngine(return_text="今日は良い天気ですね")
        seg = _make_segment()
        transcriber = _new_transcriber(
            engine, vad_processor=MockVADProcessor(segments=[seg])
        )
        events: List[UtteranceSettledEvent] = []
        transcriber.set_callbacks(on_utterance_settled=events.append)

        async def run() -> None:
            source = MockAudioSource(
                chunks=[np.zeros(160, dtype=np.float32)]
            )
            async for result in transcriber.transcribe_async(source):
                # 1 件取り出して即 break (settled が yield 後に置かれていたら
                # 永久未発火になる、rev5 reviewer 指摘 bug)
                _ = result
                break

        asyncio.run(run())

        # settled が yield 前に発火していれば observed
        assert len(events) >= 1
        assert events[0].emitted is True


# ---------------------------------------------------------------------------
# REASON_* enumeration pin
# ---------------------------------------------------------------------------


class TestReasonConstantsEnumeration:
    def test_static_reason_set_matches_tier1(self) -> None:
        """``REASON_*`` 定数 set が Tier 1 4 reason と完全一致。

        Typecheck job 不在 (rev4 で撤回) の代替: unit test で拡張時の漏れを検知。
        """
        static_reasons = {
            REASON_EMPTY_AUDIO,
            REASON_ENERGY_GATE,
            REASON_FILTER_REJECT,
            REASON_ENGINE_EMPTY,
        }
        expected = {
            "segment:empty_audio",
            "energy_gate:low_rms",
            "confidence_filter:reject",
            "engine:empty_text",
        }
        assert static_reasons == expected
