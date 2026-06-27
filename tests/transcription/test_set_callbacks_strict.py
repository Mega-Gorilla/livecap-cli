"""Tests for ``set_callbacks`` strict signature (Issue #332).

Policy: 「不要な後方互換は廃する」 (CLAUDE.md pre-1.0)。
``**kwargs`` swallow なし、未知 kwarg は ``TypeError`` で即時 fail。
"""

from __future__ import annotations

import pytest

from livecap_cli import StreamTranscriber

from tests.transcription.test_stream import MockEngine, MockVADProcessor


def _new_transcriber() -> StreamTranscriber:
    return StreamTranscriber(
        engine=MockEngine(),
        vad_processor=MockVADProcessor(),
        engine_min_rms_dbfs=float("-inf"),
    )


class TestSetCallbacksStrict:
    def test_optional_settled_callback(self) -> None:
        """``on_utterance_settled`` を渡さなくても動く (default = None)。"""
        transcriber = _new_transcriber()
        transcriber.set_callbacks(on_result=lambda r: None)
        # default is None, no error
        assert transcriber._on_utterance_settled is None

    def test_unknown_kwarg_raises_typeerror(self) -> None:
        """未知 kwarg を渡すと ``TypeError`` (policy: 不要な後方互換は廃する)。

        Python 標準の argument binding に依存。``**kwargs`` swallow を
        追加すると本 test が壊れる (regression guard)。
        """
        transcriber = _new_transcriber()
        with pytest.raises(TypeError):
            transcriber.set_callbacks(on_unknown_kwarg=lambda x: None)  # type: ignore[call-arg]
