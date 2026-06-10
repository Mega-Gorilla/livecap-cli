"""TranscriptionEngine Protocol の runtime type 解決を pin する test
(codex-review on #309)。

`typing.get_type_hints()` で `EngineTranscriptionResult` alias が
`livecap_cli.engines.base_engine.TranscriptionResult` に正しく解決される
ことを verify する。alias が `TYPE_CHECKING` ブロック内だけで import
されていると `NameError` になるため、将来の refactor で抜けないよう pin する。
"""
import typing

import pytest

from livecap_cli.engines.base_engine import TranscriptionResult as EngineDataclass
from livecap_cli.transcription.stream import TranscriptionEngine


class TestTranscribeReturnTypeIsRuntimeResolvable:
    def test_get_type_hints_resolves_engine_transcription_result(self):
        """`typing.get_type_hints(TranscriptionEngine.transcribe)` が
        `NameError` を起こさず、`engines.base_engine.TranscriptionResult` を
        返すこと。"""
        hints = typing.get_type_hints(TranscriptionEngine.transcribe)
        assert "return" in hints, "Protocol method must declare a return type hint"
        assert hints["return"] is EngineDataclass, (
            "TranscriptionEngine.transcribe の return type は "
            "livecap_cli.engines.base_engine.TranscriptionResult を指すこと "
            "(transcription.result.TranscriptionResult ではない)"
        )

    def test_alias_points_to_engines_module_not_transcription_module(self):
        """Protocol return type と transcription.result の同名 class が
        混同されていないことを pin する。"""
        from livecap_cli.transcription.result import (
            TranscriptionResult as CoalescerDataclass,
        )

        # 2 つの TranscriptionResult は別の class object
        assert EngineDataclass is not CoalescerDataclass

        hints = typing.get_type_hints(TranscriptionEngine.transcribe)
        assert hints["return"] is EngineDataclass
        assert hints["return"] is not CoalescerDataclass
