"""Tests for Qwen3-ASR auto-detect fail-open warning (Issue #334 Finding 6).

``StreamTranscriber.__init__`` が ``filter_config.mode != "off"`` かつ engine が
Qwen3-ASR + auto-detect (``engine._asr_language is None``) の組合せの時、1 回
``logger.warning`` を出すことを pin する。

設計理由 (Issue #334 Finding 6 + reviewer 2nd round 指摘):

- ``Qwen3ASREngine.__init__`` は ``FilterConfig`` を受けないため、engine 層で
  warn するのは architectural separation 違反。両方を知る
  ``StreamTranscriber.__init__`` で警告するのが正しい。
- Duck typing (``engine.engine_name == "qwen3asr"`` string compare +
  ``engine._asr_language is None``) で検出、``isinstance`` は循環 import /
  Mock false negative を回避。
"""

from __future__ import annotations

import logging
from typing import Optional

from livecap_cli.transcription.confidence_filter import FilterConfig
from livecap_cli.transcription.stream import StreamTranscriber

from tests.transcription.test_stream import MockEngine, MockVADProcessor


class MockQwen3LikeEngine(MockEngine):
    """``Qwen3ASREngine`` の識別 attribute を模擬する Mock。

    実際の ``Qwen3ASREngine`` は ``__init__`` で以下を設定する:

    - ``self.engine_name = engine_id`` (default ``"qwen3asr"``、
      ``qwen3asr_engine.py:244``)
    - ``self._asr_language = self._resolve_language(language)``
      (``qwen3asr_engine.py:248``、auto-detect 時は ``None``)

    本 Mock はその 2 attribute だけ持つ最小実装で、``MockEngine`` の transcribe
    動作はそのまま継承する。
    """

    def __init__(self, language: Optional[str] = None) -> None:
        super().__init__()
        self.engine_name = "qwen3asr"
        self._asr_language = language


def _make_transcriber(engine, filter_mode: str = "on") -> StreamTranscriber:
    return StreamTranscriber(
        engine=engine,
        vad_processor=MockVADProcessor(),
        engine_min_rms_dbfs=float("-inf"),
        filter_config=FilterConfig(mode=filter_mode),
    )


WARN_FRAGMENT = "Qwen3-ASR auto-detect mode"
LOGGER_NAME = "livecap_cli.transcription.stream"


class TestQwen3AutoDetectWarn:
    """Issue #334 Finding 6 — warn 発火条件 matrix を pin。"""

    def test_warns_when_qwen3_auto_detect_with_filter_on(self, caplog) -> None:
        """qwen3asr + ``language=None`` + ``filter="on"`` → warn 1 回。"""
        engine = MockQwen3LikeEngine(language=None)
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            _make_transcriber(engine, filter_mode="on")
        warns = [r for r in caplog.records if WARN_FRAGMENT in r.getMessage()]
        assert len(warns) == 1

    def test_no_warn_when_filter_off(self, caplog) -> None:
        """``filter="off"`` では Qwen3 + auto-detect でも warn なし。"""
        engine = MockQwen3LikeEngine(language=None)
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            _make_transcriber(engine, filter_mode="off")
        warns = [r for r in caplog.records if WARN_FRAGMENT in r.getMessage()]
        assert warns == []

    def test_no_warn_when_language_specified(self, caplog) -> None:
        """qwen3asr + ``language="Japanese"`` + ``filter="on"`` → warn なし
        (auto-detect path に入らないため filter active)。"""
        engine = MockQwen3LikeEngine(language="Japanese")
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            _make_transcriber(engine, filter_mode="on")
        warns = [r for r in caplog.records if WARN_FRAGMENT in r.getMessage()]
        assert warns == []

    def test_no_warn_for_other_engines(self, caplog) -> None:
        """非 qwen3asr engine (``engine_name`` 属性なし) → warn なし。

        ``MockEngine`` は ``engine_name`` / ``_asr_language`` を持たないため、
        ``getattr(..., "")`` で sentinel default を返し、early return する。
        """
        engine = MockEngine()  # No engine_name / _asr_language
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            _make_transcriber(engine, filter_mode="on")
        warns = [r for r in caplog.records if WARN_FRAGMENT in r.getMessage()]
        assert warns == []

    def test_observe_mode_also_warns(self, caplog) -> None:
        """``observe`` mode (filter active、reject はしないが判定 + log) でも
        warn する (mode != "off" の判定 logic を pin)。"""
        engine = MockQwen3LikeEngine(language=None)
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            _make_transcriber(engine, filter_mode="observe")
        warns = [r for r in caplog.records if WARN_FRAGMENT in r.getMessage()]
        assert len(warns) == 1
