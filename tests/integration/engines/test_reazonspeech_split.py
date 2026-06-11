"""Regression test for `ReazonSpeechEngine._transcribe_with_split` (Issue #317 / PR-A.5.1).

PR #314 で ``TranscriptionResult.__iter__`` (旧 ``Tuple[str, float]`` 戻り値の
後方互換 shim) が削除された。それに伴い ``reazonspeech_engine.py:430`` の
``text, confidence = self._transcribe_single(...)`` unpack が **TypeError**
を投げるようになったが、その exception は外側の ``except Exception`` で
silent swallow されていたため、**長尺音声 (>auto_split_duration、default
30s) で全 segment が silently dropped されていた production critical bug**
が production reach していた。

本 PR (Issue #317 PR-A.5.1) で bug を修正、本 test で長尺音声を mock し:
1. ``_transcribe_with_split`` が segment を silently drop しないこと
2. 各 segment の ``TranscriptionResult.text`` が combined_text に積まれること
3. ``_transcribe_single`` の return type が ``TranscriptionResult`` で
   attribute access パターンが正しく動くこと
を pin する。
"""
from __future__ import annotations

import numpy as np
import pytest

from livecap_cli.engines.base_engine import TranscriptionResult


@pytest.fixture
def fake_engine_with_split():
    """``ReazonSpeechEngine`` を ``__new__`` で生成、必要 attribute だけ差し込む fixture。

    実 sherpa-onnx を load せず ``_transcribe_with_split`` の制御 path だけを
    test 可能にする。
    """
    from livecap_cli.engines.reazonspeech_engine import ReazonSpeechEngine

    engine = ReazonSpeechEngine.__new__(ReazonSpeechEngine)
    engine.engine_name = "reazonspeech"
    engine.auto_split_duration = 30.0
    engine.progress_callback = None
    engine._initialized = True
    engine.model = object()  # placeholder、_transcribe_single を mock する前提
    return engine


def test_transcribe_with_split_does_not_silently_drop_segments(
    fake_engine_with_split, monkeypatch
):
    """旧 bug: TranscriptionResult unpack が TypeError → 全 segment が silently dropped。

    本 test は ``_transcribe_single`` を mock し ``_transcribe_with_split`` が:
    - 各 segment を call すること (35s audio = 2 segments)
    - segment.text を正しく `results` に積めること
    - 最終的に combined_text が空でないこと
    を verify する。
    """
    calls = []

    def fake_transcribe_single(audio_data, sample_rate):
        calls.append((len(audio_data), sample_rate))
        # 各 segment が TranscriptionResult を返す (旧 bug の発生条件と同じ)
        return TranscriptionResult(text=f"segment_{len(calls)}", confidence=1.0)

    monkeypatch.setattr(
        fake_engine_with_split, "_transcribe_single", fake_transcribe_single
    )

    # 35 秒 audio @16kHz → auto_split (30s) で 2 segments (30s + 5s)
    sample_rate = 16000
    audio = np.zeros(35 * sample_rate, dtype=np.float32)

    result = fake_engine_with_split._transcribe_with_split(audio, sample_rate)

    # 旧 bug の場合、calls == [] (全 TypeError + silent swallow + continue)
    # 修正後は 2 segments call、text が積まれる
    assert len(calls) == 2, (
        f"_transcribe_single は 2 segments で call されるはず ({len(calls)} 件のみ)。"
        "旧 bug の場合 0 件 (silently drop) になる。"
    )
    assert result.text == "segment_1segment_2", (
        f"combined_text='{result.text}'。旧 bug の場合は空文字になる。"
    )


def test_transcribe_with_split_handles_empty_segments(
    fake_engine_with_split, monkeypatch
):
    """空 text を返した segment は results に積まない (text 評価で skip)。"""
    def fake_transcribe_single(audio_data, sample_rate):
        return TranscriptionResult(text="", confidence=1.0)

    monkeypatch.setattr(
        fake_engine_with_split, "_transcribe_single", fake_transcribe_single
    )

    sample_rate = 16000
    audio = np.zeros(35 * sample_rate, dtype=np.float32)

    result = fake_engine_with_split._transcribe_with_split(audio, sample_rate)

    # 全 segment が空 text → combined_text は空、confidence=0.0 path に進む
    assert result.text == ""
    assert result.confidence == 0.0


def test_transcribe_with_split_propagates_segment_exceptions_silently(
    fake_engine_with_split, monkeypatch
):
    """segment 例外は log 後 continue、他 segment は drop されない。

    旧 bug の場合: TypeError で全 segment が except 経由で drop されていた。
    修正後: 個別 segment 例外は log + continue するが、正常 segment は積まれる。
    """
    call_count = {"n": 0}

    def fake_transcribe_single(audio_data, sample_rate):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated single segment failure")
        return TranscriptionResult(text=f"segment_{call_count['n']}", confidence=1.0)

    monkeypatch.setattr(
        fake_engine_with_split, "_transcribe_single", fake_transcribe_single
    )

    sample_rate = 16000
    audio = np.zeros(35 * sample_rate, dtype=np.float32)

    result = fake_engine_with_split._transcribe_with_split(audio, sample_rate)

    # 1 件目: 例外 → swallow & continue
    # 2 件目: 正常 → 積まれる
    assert call_count["n"] == 2
    assert result.text == "segment_2"
