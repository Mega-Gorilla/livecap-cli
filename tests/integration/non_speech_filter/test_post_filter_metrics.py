"""Regression tests for post-filter metric semantics (Issue #308 PR-A.3).

codex-review on #312 3rd round Item 3 で要求された 3 件の意味論固定テスト:

1. ``finalize()`` 戻り値だけに final result がある場合、post-filter metric が拾う
2. queue が ``[InterimResult, TranscriptionResult]`` の順でも final result を拾う
3. positive speech が filter 後に消えた場合、``post_filter_speech_recall`` が下がる

実 ``StreamTranscriber`` の pipeline は通さず、``evaluate_pipeline`` の
metric 計算 path だけを単離テストする。これにより以下 2 件の過去 bug が
再発した時に CI で確実に止まる:

- 1st round (HIGH): ``finalize()`` 戻り値の取り逃がし
- 2nd round (MED): queue drain が ``InterimResult`` 先頭で停止
"""

from __future__ import annotations

import queue

import numpy as np
import pytest

from benchmarks.non_speech_filter.corpus import CorpusItem
from benchmarks.non_speech_filter.metrics import evaluate_pipeline
from livecap_cli.transcription.result import InterimResult, TranscriptionResult


# ---- Fake StreamTranscriber / Engine -----------------------------------------


class _FakeEngine:
    """``evaluate_pipeline`` が要求する surface を最小実装する fake。

    ``transcribe_count`` は ``_FakeTranscriber`` 側から bump される
    (start=0 → end=1 の差分で ``triggered=True`` 判定にするため)。
    """

    def __init__(self, last_texts: list[str] | None = None) -> None:
        self.transcribe_count = 0
        self.last_texts: list[str] = list(last_texts or [])


class _FakeTranscriber:
    """``feed_audio`` で engine.transcribe_count を bump、``finalize()`` と
    ``_result_queue`` の中身を test 側で制御。

    実 ``StreamTranscriber`` の I/F を模倣 (``_result_queue`` も含む)。
    """

    def __init__(
        self,
        engine: _FakeEngine,
        *,
        finalize_results: list[TranscriptionResult] | None = None,
        queue_items: list[TranscriptionResult | InterimResult] | None = None,
    ) -> None:
        self._engine = engine
        self._finalize_results = list(finalize_results or [])
        self._result_queue: queue.Queue = queue.Queue()
        for qi in queue_items or []:
            self._result_queue.put(qi)

    def feed_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> None:
        # engine の transcribe_count を bump して triggered=True 判定にする
        self._engine.transcribe_count += 1

    def finalize(self) -> list[TranscriptionResult]:
        return list(self._finalize_results)


def _mk_positive_item(label: str = "speech_a") -> CorpusItem:
    """Positive speech item (1 秒の dummy audio)。"""
    return CorpusItem(
        label=label,
        kind="positive",
        is_short_utterance=False,
        audio=np.zeros(16000, dtype=np.float32),
    )


def _mk_short_item(label: str = "speech_short") -> CorpusItem:
    return CorpusItem(
        label=label,
        kind="positive",
        is_short_utterance=True,
        audio=np.zeros(8000, dtype=np.float32),
    )


def _final(text: str) -> TranscriptionResult:
    return TranscriptionResult(text=text, start_time=0.0, end_time=1.0)


def _interim(text: str) -> InterimResult:
    return InterimResult(text=text, accumulated_time=0.5)


# ---- Test 1: finalize() 戻り値だけに final がある case --------------------


def test_post_filter_metric_collects_from_finalize_return_when_queue_empty() -> None:
    """1st round HIGH bug の regression test。

    ``finalize()`` が non-empty TranscriptionResult を返し、``_result_queue``
    は空。``post_filter_hallucination_rate`` と ``post_filter_speech_recall``
    の両方が ``finalize()`` 戻り値を拾えること。
    """
    item = _mk_positive_item()

    def factory():
        eng = _FakeEngine(last_texts=["legit speech"])
        return (
            _FakeTranscriber(
                eng,
                finalize_results=[_final("legit speech")],
                queue_items=[],
            ),
            eng,
        )

    eval_ = evaluate_pipeline(
        factory,
        [item],
        measure_hallucination=True,
    )

    # finalize() 戻り値から拾えれば post_filter_speech_recall = 1.0
    assert eval_.post_filter_speech_recall == 1.0, (
        "finalize() 戻り値の取り逃がし bug が再発している可能性"
    )
    # negative item がないため hallucination rate は None
    assert eval_.post_filter_hallucination_rate is None
    # pre-filter recall も 1.0 (engine call で計測)
    assert eval_.speech_recall == 1.0


# ---- Test 2: queue が [InterimResult, TranscriptionResult] の順 --------------


def test_post_filter_metric_collects_final_after_interim_in_queue() -> None:
    """2nd round MED bug の regression test。

    queue が ``[InterimResult, TranscriptionResult]`` の順でも final result を
    拾えること。旧 ``get_result(timeout=0)`` の drain だと interim を消費
    した時点で None 返却 → 早期 exit で final を取り逃がす bug。
    """
    item = _mk_positive_item()

    def factory():
        eng = _FakeEngine(last_texts=["legit speech"])
        return (
            _FakeTranscriber(
                eng,
                finalize_results=[],  # finalize() は何も返さない
                queue_items=[_interim("partial..."), _final("legit speech")],
            ),
            eng,
        )

    eval_ = evaluate_pipeline(
        factory,
        [item],
        measure_hallucination=True,
    )

    # queue 先頭の Interim を skip して TranscriptionResult を拾う
    assert eval_.post_filter_speech_recall == 1.0, (
        "queue drain が InterimResult で停止する bug が再発している可能性"
    )


# ---- Test 3: post-filter で speech が消えると recall が下がる ----------------


def test_post_filter_speech_recall_drops_when_filter_removes_all_speech() -> None:
    """post_filter_speech_recall の semantic pin test。

    confidence filter が legit speech を全 drop した想定 (post-filter empty)
    で、``speech_recall = 1.0`` (engine call は発生) のまま
    ``post_filter_speech_recall = 0.0`` に下がることを確認。
    既存 ``speech_recall`` が engine call で計測される問題点 (#312 3rd round
    Item 1 HIGH) を将来 metric 修正で逆戻りさせないための pin。
    """
    item = _mk_positive_item()

    def factory():
        eng = _FakeEngine(last_texts=["legit speech"])  # engine 自体は emit
        return (
            _FakeTranscriber(
                eng,
                # filter が drop した想定: finalize() も queue も空
                finalize_results=[],
                queue_items=[],
            ),
            eng,
        )

    eval_ = evaluate_pipeline(
        factory,
        [item],
        measure_hallucination=True,
    )

    # pre-filter (engine call で計測) は 1.0 のまま
    assert eval_.speech_recall == 1.0, (
        "engine call の counter は filter とは独立 (pre-filter recall)"
    )
    # post-filter (user の字幕 stream で計測) は 0.0 に下がる
    assert eval_.post_filter_speech_recall == 0.0, (
        "filter が legit speech を drop した時に post-filter recall が下がら"
        "ないと H3 claim が空 hypothesis になる"
    )


# ---- Bonus: short_utterance_recall も同じ semantic で動くこと ----------------


def test_post_filter_short_utterance_recall_tracks_post_filter_output() -> None:
    """short utterance も post-filter で空なら post_filter_short_utterance_recall=0.0。"""
    item = _mk_short_item()

    def factory():
        eng = _FakeEngine(last_texts=["short speech"])
        return (
            _FakeTranscriber(eng, finalize_results=[], queue_items=[]),
            eng,
        )

    eval_ = evaluate_pipeline(factory, [item], measure_hallucination=True)

    assert eval_.short_utterance_recall == 1.0
    assert eval_.post_filter_short_utterance_recall == 0.0


# ---- Bonus: measure_hallucination=False では post-filter recall=None --------


def test_post_filter_recall_is_none_when_measure_hallucination_is_false() -> None:
    """MockEngine 経路 (measure_hallucination=False) では post-filter recall を
    None にする。0.0 と区別することで「測定していない」と「測定して 0%」が
    報告で混同されないようにする。"""
    item = _mk_positive_item()

    def factory():
        eng = _FakeEngine()
        return (_FakeTranscriber(eng), eng)

    eval_ = evaluate_pipeline(factory, [item], measure_hallucination=False)

    assert eval_.post_filter_speech_recall is None
    assert eval_.post_filter_short_utterance_recall is None
    assert eval_.post_filter_hallucination_rate is None
