"""StreamTranscriber 翻訳統合のユニットテスト"""

from __future__ import annotations

import time
from collections import deque
from typing import List, Optional

from livecap_cli.engines.base_engine import TranscriptionResult as EngineTranscriptionResult
from unittest.mock import MagicMock, patch

import numpy as np
import asyncio
import threading
import pytest

from livecap_cli.transcription.stream import (
    MAX_CONTEXT_BUFFER,
    TRANSLATION_TIMEOUT,
    _DEFAULT_TRANSLATION_TIMEOUT,
    _get_translation_timeout,
    StreamTranscriber,
    TranscriptionEngine,
)
from livecap_cli.transcription.result import TranscriptionResult
from livecap_cli.translation.base import BaseTranslator
from livecap_cli.translation.result import TranslationResult
from livecap_cli.vad import VADSegment, VADState


class MockEngine:
    """テスト用のモックエンジン"""

    def __init__(self):
        self.transcribe_calls = []

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> EngineTranscriptionResult:
        self.transcribe_calls.append((audio, sample_rate))
        return EngineTranscriptionResult(text="テスト文字起こし", confidence=0.95)

    def get_required_sample_rate(self) -> int:
        return 16000

    def get_engine_name(self) -> str:
        return "mock_engine"

    def cleanup(self) -> None:
        pass


class MockVADProcessor:
    """テスト用モックVADプロセッサ（silero-vad 不要）"""

    def __init__(self, segments: list[VADSegment] | None = None):
        self._segments = segments or []
        self._segment_index = 0
        self._state = VADState.SILENCE
        self._finalize_segment: VADSegment | None = None
        self._current_time: float = 0.0

    def process_chunk(
        self, audio: np.ndarray, sample_rate: int
    ) -> list[VADSegment]:
        if self._segment_index < len(self._segments):
            segment = self._segments[self._segment_index]
            self._segment_index += 1
            return [segment]
        return []

    def finalize(self) -> VADSegment | None:
        return self._finalize_segment

    def reset(self) -> None:
        self._segment_index = 0
        self._state = VADState.SILENCE

    @property
    def state(self) -> VADState:
        return self._state

    @property
    def current_time(self) -> float:
        return self._current_time


class MockTranslator(BaseTranslator):
    """テスト用のモックTranslator"""

    def __init__(
        self,
        initialized: bool = True,
        translation_text: str = "Mock translation",
        default_context_sentences: int = 2,
    ):
        super().__init__(default_context_sentences=default_context_sentences)
        self._initialized = initialized
        self._translation_text = translation_text
        self.translate_calls: List[Tuple[str, str, str, Optional[List[str]]]] = []

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Optional[List[str]] = None,
    ) -> TranslationResult:
        self.translate_calls.append((text, source_lang, target_lang, context))
        return TranslationResult(
            text=self._translation_text,
            original_text=text,
            source_lang=source_lang,
            target_lang=target_lang,
        )

    def get_supported_pairs(self) -> List[Tuple[str, str]]:
        return [("ja", "en"), ("en", "ja")]

    def get_translator_name(self) -> str:
        return "mock_translator"


class TestStreamTranscriberInit:
    """StreamTranscriber 初期化のテスト"""

    def test_init_without_translator(self):
        """translator なしの初期化（後方互換）"""
        engine = MockEngine()
        vad = MockVADProcessor()
        transcriber = StreamTranscriber(engine=engine, vad_processor=vad)

        assert transcriber._translator is None
        assert transcriber._source_lang is None
        assert transcriber._target_lang is None
        assert isinstance(transcriber._context_buffer, deque)
        assert transcriber._context_buffer.maxlen == MAX_CONTEXT_BUFFER

    def test_init_with_translator(self):
        """translator ありの初期化"""
        engine = MockEngine()
        translator = MockTranslator()
        vad = MockVADProcessor()

        transcriber = StreamTranscriber(
            engine=engine,
            translator=translator,
            source_lang="ja",
            target_lang="en",
            vad_processor=vad,
        )

        assert transcriber._translator is translator
        assert transcriber._source_lang == "ja"
        assert transcriber._target_lang == "en"

    def test_init_translator_not_initialized_raises(self):
        """未初期化の translator でエラー"""
        engine = MockEngine()
        translator = MockTranslator(initialized=False)
        vad = MockVADProcessor()

        with pytest.raises(ValueError, match="not initialized"):
            StreamTranscriber(
                engine=engine,
                translator=translator,
                source_lang="ja",
                target_lang="en",
                vad_processor=vad,
            )

    def test_init_translator_without_source_lang_raises(self):
        """source_lang なしでエラー"""
        engine = MockEngine()
        translator = MockTranslator()
        vad = MockVADProcessor()

        with pytest.raises(ValueError, match="source_lang and target_lang are required"):
            StreamTranscriber(
                engine=engine,
                translator=translator,
                target_lang="en",
                vad_processor=vad,
            )

    def test_init_translator_without_target_lang_raises(self):
        """target_lang なしでエラー"""
        engine = MockEngine()
        translator = MockTranslator()
        vad = MockVADProcessor()

        with pytest.raises(ValueError, match="source_lang and target_lang are required"):
            StreamTranscriber(
                engine=engine,
                translator=translator,
                source_lang="ja",
                vad_processor=vad,
            )

    def test_init_unsupported_pair_warns(self, caplog):
        """未サポートの言語ペアで警告"""
        engine = MockEngine()
        translator = MockTranslator()  # supports ja-en, en-ja
        vad = MockVADProcessor()

        with caplog.at_level("WARNING"):
            StreamTranscriber(
                engine=engine,
                translator=translator,
                source_lang="fr",  # 未サポート
                target_lang="de",
                vad_processor=vad,
            )

        assert "may not be supported" in caplog.text


class TestStreamTranscriberTranslation:
    """StreamTranscriber 翻訳処理のテスト"""

    def test_translate_text_with_translator(self):
        """翻訳処理が正しく呼ばれる"""
        engine = MockEngine()
        translator = MockTranslator(translation_text="Hello")
        vad = MockVADProcessor()

        transcriber = StreamTranscriber(
            engine=engine,
            translator=translator,
            source_lang="ja",
            target_lang="en",
            vad_processor=vad,
        )

        outcome = transcriber._translate_text("こんにちは")

        assert outcome.state == "translated"
        assert outcome.translated_text == "Hello"
        assert outcome.target_language == "en"
        assert len(translator.translate_calls) == 1
        assert translator.translate_calls[0][0] == "こんにちは"
        assert translator.translate_calls[0][1] == "ja"
        assert translator.translate_calls[0][2] == "en"

    def test_translate_text_without_translator(self):
        """translator なしで翻訳なし"""
        engine = MockEngine()
        vad = MockVADProcessor()
        transcriber = StreamTranscriber(engine=engine, vad_processor=vad)

        outcome = transcriber._translate_text("こんにちは")

        assert outcome.state == "not_requested"
        assert outcome.translated_text is None

    def test_context_buffer_accumulation(self):
        """文脈バッファが蓄積される"""
        engine = MockEngine()
        translator = MockTranslator(default_context_sentences=2)
        vad = MockVADProcessor()

        transcriber = StreamTranscriber(
            engine=engine,
            translator=translator,
            source_lang="ja",
            target_lang="en",
            vad_processor=vad,
        )

        # 3回翻訳
        transcriber._translate_text("文1")
        transcriber._translate_text("文2")
        transcriber._translate_text("文3")

        assert len(transcriber._context_buffer) == 3
        assert list(transcriber._context_buffer) == ["文1", "文2", "文3"]

        # 3回目の翻訳では文脈として ["文1", "文2"] が渡される
        # （default_context_sentences=2 なので直近2文）
        assert translator.translate_calls[2][3] == ["文1", "文2"]

    def test_context_buffer_max_size(self):
        """文脈バッファの最大サイズ制限"""
        engine = MockEngine()
        translator = MockTranslator()
        vad = MockVADProcessor()

        transcriber = StreamTranscriber(
            engine=engine,
            translator=translator,
            source_lang="ja",
            target_lang="en",
            vad_processor=vad,
        )

        # MAX_CONTEXT_BUFFER + 10 回翻訳
        for i in range(MAX_CONTEXT_BUFFER + 10):
            transcriber._translate_text(f"文{i}")

        # maxlen=MAX_CONTEXT_BUFFER なので、最大数に制限される
        assert len(transcriber._context_buffer) == MAX_CONTEXT_BUFFER

    def test_translation_failure_returns_none(self, caplog):
        """翻訳失敗時は None を返す"""
        engine = MockEngine()
        translator = MockTranslator()
        vad = MockVADProcessor()

        # translate メソッドをモックして例外を発生させる
        def raise_error(*args, **kwargs):
            raise Exception("Translation API error")

        translator.translate = raise_error  # type: ignore

        transcriber = StreamTranscriber(
            engine=engine,
            translator=translator,
            source_lang="ja",
            target_lang="en",
            vad_processor=vad,
        )

        outcome = transcriber._translate_text("こんにちは")

        # 失敗は潰さず理由ごと返す (Issue #402 D1)。ログではなく型で運ぶ。
        assert outcome.state == "failed"
        assert outcome.translated_text is None
        assert outcome.error_type in ("network", "fatal")
        assert outcome.message

    def test_translation_failure_still_adds_to_context(self):
        """翻訳失敗しても文脈バッファには追加"""
        engine = MockEngine()
        translator = MockTranslator()
        vad = MockVADProcessor()

        def raise_error(*args, **kwargs):
            raise Exception("Error")

        translator.translate = raise_error  # type: ignore

        transcriber = StreamTranscriber(
            engine=engine,
            translator=translator,
            source_lang="ja",
            target_lang="en",
            vad_processor=vad,
        )

        transcriber._translate_text("こんにちは")

        # 翻訳失敗しても文脈バッファには追加される
        assert "こんにちは" in transcriber._context_buffer


class TestStreamTranscriberTimeout:
    """StreamTranscriber 翻訳タイムアウトのテスト"""

    @patch("livecap_cli.transcription.stream.TRANSLATION_TIMEOUT", 0.1)
    def test_translation_timeout_returns_none(self, caplog):
        """翻訳がタイムアウトした場合は None を返す"""
        engine = MockEngine()
        translator = MockTranslator()
        vad = MockVADProcessor()

        # translate をスリープさせてタイムアウトを発生させる
        def slow_translate(*args, **kwargs):
            time.sleep(0.5)  # 0.1秒より長くスリープ
            return TranslationResult(
                text="Should not reach here",
                original_text=args[0],
                source_lang=args[1],
                target_lang=args[2],
            )

        translator.translate = slow_translate  # type: ignore

        transcriber = StreamTranscriber(
            engine=engine,
            translator=translator,
            source_lang="ja",
            target_lang="en",
            vad_processor=vad,
        )

        outcome = transcriber._translate_text("こんにちは")

        assert outcome.state == "failed"
        assert outcome.error_type == "timeout"

    @patch("livecap_cli.transcription.stream.TRANSLATION_TIMEOUT", 0.1)
    def test_translation_timeout_still_adds_to_context(self):
        """翻訳がタイムアウトしても文脈バッファには追加"""
        engine = MockEngine()
        translator = MockTranslator()
        vad = MockVADProcessor()

        def slow_translate(*args, **kwargs):
            time.sleep(0.5)  # 0.1秒より長くスリープ
            return TranslationResult(
                text="Should not reach here",
                original_text=args[0],
                source_lang=args[1],
                target_lang=args[2],
            )

        translator.translate = slow_translate  # type: ignore

        transcriber = StreamTranscriber(
            engine=engine,
            translator=translator,
            source_lang="ja",
            target_lang="en",
            vad_processor=vad,
        )

        transcriber._translate_text("こんにちは")

        # タイムアウトしても文脈バッファには追加される
        assert "こんにちは" in transcriber._context_buffer

    def test_translation_timeout_default_is_two_seconds(self):
        """既定は 2 秒 (Issue #402 D10 で 10.0 から変更)。

        リアルタイム字幕では 10 秒待つのは明確に誤りだった — 遅れて届いた翻訳は
        今話している内容と重なって出るだけで、字幕として価値が無い。
        """
        assert _DEFAULT_TRANSLATION_TIMEOUT == 2.0

    def test_get_translation_timeout_without_env(self, monkeypatch):
        """環境変数未設定時はデフォルト値を返す"""
        monkeypatch.delenv("LIVECAP_TRANSLATION_TIMEOUT", raising=False)
        assert _get_translation_timeout() == 2.0

    def test_get_translation_timeout_with_valid_env(self, monkeypatch):
        """有効な環境変数が設定されている場合はその値を使用"""
        monkeypatch.setenv("LIVECAP_TRANSLATION_TIMEOUT", "20.0")
        assert _get_translation_timeout() == 20.0

    def test_get_translation_timeout_with_invalid_env(self, monkeypatch, caplog):
        """無効な環境変数（非数値）はデフォルトにフォールバック"""
        monkeypatch.setenv("LIVECAP_TRANSLATION_TIMEOUT", "invalid")
        with caplog.at_level("WARNING"):
            result = _get_translation_timeout()
        assert result == 2.0
        assert "Invalid LIVECAP_TRANSLATION_TIMEOUT" in caplog.text

    def test_get_translation_timeout_with_zero(self, monkeypatch, caplog):
        """0 はデフォルトにフォールバック"""
        monkeypatch.setenv("LIVECAP_TRANSLATION_TIMEOUT", "0")
        with caplog.at_level("WARNING"):
            result = _get_translation_timeout()
        assert result == 2.0
        assert "must be positive" in caplog.text

    def test_get_translation_timeout_with_negative(self, monkeypatch, caplog):
        """負の値はデフォルトにフォールバック"""
        monkeypatch.setenv("LIVECAP_TRANSLATION_TIMEOUT", "-5")
        with caplog.at_level("WARNING"):
            result = _get_translation_timeout()
        assert result == 2.0
        assert "must be positive" in caplog.text


class TestDoTranslateDirect:
    """StreamTranscriber._do_translate_direct のテスト"""

    def test_do_translate_direct_success(self):
        """正常に翻訳が実行される"""
        engine = MockEngine()
        translator = MockTranslator(translation_text="Hello")
        vad = MockVADProcessor()

        transcriber = StreamTranscriber(
            engine=engine,
            translator=translator,
            source_lang="ja",
            target_lang="en",
            vad_processor=vad,
        )

        outcome = transcriber._do_translate_direct("こんにちは")

        assert outcome.state == "translated"
        assert outcome.translated_text == "Hello"
        assert outcome.target_language == "en"
        assert "こんにちは" in transcriber._context_buffer

    def test_do_translate_direct_without_translator(self):
        """translator なしで None を返す"""
        engine = MockEngine()
        vad = MockVADProcessor()
        transcriber = StreamTranscriber(engine=engine, vad_processor=vad)

        outcome = transcriber._do_translate_direct("こんにちは")

        assert outcome.state == "not_requested"
        assert outcome.translated_text is None

    def test_do_translate_direct_failure_returns_none(self, caplog):
        """翻訳失敗時は None を返し、文脈バッファには追加"""
        engine = MockEngine()
        translator = MockTranslator()
        vad = MockVADProcessor()

        def raise_error(*args, **kwargs):
            raise Exception("Translation API error")

        translator.translate = raise_error  # type: ignore

        transcriber = StreamTranscriber(
            engine=engine,
            translator=translator,
            source_lang="ja",
            target_lang="en",
            vad_processor=vad,
        )

        outcome = transcriber._do_translate_direct("こんにちは")

        # worker 内では通知しない。理由は outcome で caller へ渡す (Issue #402 D1)。
        assert outcome.state == "failed"
        assert outcome.error_type in ("network", "fatal")
        assert outcome.message
        # 失敗しても文脈バッファには追加
        assert "こんにちは" in transcriber._context_buffer

    def test_do_translate_direct_no_executor_submission(self):
        """_do_translate_direct は executor に提出しない（デッドロック回避）"""
        engine = MockEngine()
        translator = MockTranslator(translation_text="Hello")
        vad = MockVADProcessor()

        transcriber = StreamTranscriber(
            engine=engine,
            translator=translator,
            source_lang="ja",
            target_lang="en",
            vad_processor=vad,
        )

        # executor.submit をモックしてカウント
        original_submit = transcriber._executor.submit
        submit_count = [0]

        def counting_submit(*args, **kwargs):
            submit_count[0] += 1
            return original_submit(*args, **kwargs)

        transcriber._executor.submit = counting_submit  # type: ignore

        # _do_translate_direct は executor を使わない
        transcriber._do_translate_direct("テスト")

        assert submit_count[0] == 0, "_do_translate_direct should not use executor"


class TestStreamTranscriberReset:
    """StreamTranscriber reset のテスト"""

    def test_reset_clears_context_buffer(self):
        """reset で文脈バッファがクリアされる"""
        engine = MockEngine()
        translator = MockTranslator()
        vad = MockVADProcessor()

        transcriber = StreamTranscriber(
            engine=engine,
            translator=translator,
            source_lang="ja",
            target_lang="en",
            vad_processor=vad,
        )

        # 文脈を蓄積
        transcriber._translate_text("文1")
        transcriber._translate_text("文2")
        assert len(transcriber._context_buffer) == 2

        # リセット
        transcriber.reset()

        assert len(transcriber._context_buffer) == 0


class TestBaseTranslatorProperty:
    """BaseTranslator.default_context_sentences プロパティのテスト"""

    def test_default_context_sentences_property(self):
        """プロパティが正しく値を返す"""
        translator = MockTranslator(default_context_sentences=5)
        assert translator.default_context_sentences == 5

    def test_default_context_sentences_default_value(self):
        """デフォルト値は 2"""
        translator = MockTranslator()
        assert translator.default_context_sentences == 2


# ===========================================================================
# Issue #402 PR 2: 翻訳失敗を見えるようにする
# ===========================================================================


class FlakyTranslator(MockTranslator):
    """失敗させたり遅くしたりできる translator。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.failing = False
        self.delay = 0.0

    def translate(self, text, source_lang, target_lang, context=None):
        if self.delay:
            time.sleep(self.delay)
        if self.failing:
            from livecap_cli.translation.exceptions import TranslationNetworkError

            raise TranslationNetworkError("HTTP 503", provider="mock")
        return super().translate(text, source_lang, target_lang, context)


def _transcriber(translator=None, **kwargs):
    return StreamTranscriber(
        engine=MockEngine(),
        translator=translator,
        source_lang="ja" if translator else None,
        target_lang="en" if translator else None,
        vad_processor=MockVADProcessor(),
        **kwargs,
    )


def _result(text="こんにちは"):
    return TranscriptionResult(text=text, start_time=0.0, end_time=1.0)


class TestTranslationStatusNotification:
    """失敗を黙って原文にしない (Issue #402 D1)。

    これが本 issue の核心。以前は 3 箇所の ``except Exception: logger.warning(...)``
    が失敗を飲み込み、ユーザには「日本語→日本語になった」としか見えなかった。
    """

    def test_failure_fires_once_and_stays_quiet(self):
        translator = FlakyTranslator()
        transcriber = _transcriber(translator)
        events = []
        transcriber.set_callbacks(on_translation_status=events.append)

        translator.failing = True
        for _ in range(4):
            transcriber._apply_translation_sync(_result())

        assert len(events) == 1, "segment ごとに連打してはいけない"
        assert events[0].status == "failed"
        assert events[0].translator == "mock_translator"
        assert events[0].error_type == "network"
        assert events[0].recoverable is True  # error_type から導出される
        transcriber.close()

    def test_recovery_is_announced(self):
        translator = FlakyTranslator()
        transcriber = _transcriber(translator)
        events = []
        transcriber.set_callbacks(on_translation_status=events.append)

        translator.failing = True
        transcriber._apply_translation_sync(_result())
        translator.failing = False
        transcriber._apply_translation_sync(_result())

        assert [e.status for e in events] == ["failed", "recovered"]
        transcriber.close()

    def test_healthy_run_says_nothing(self):
        transcriber = _transcriber(FlakyTranslator())
        events = []
        transcriber.set_callbacks(on_translation_status=events.append)

        for _ in range(3):
            transcriber._apply_translation_sync(_result())

        assert events == []
        transcriber.close()

    def test_async_path_uses_the_same_funnel(self):
        """3 つの swallow 経路が 1 つの通知経路へ集約されている。"""
        translator = FlakyTranslator()
        transcriber = _transcriber(translator)
        events = []
        transcriber.set_callbacks(on_translation_status=events.append)
        translator.failing = True

        async def run():
            for _ in range(3):
                await transcriber._apply_translation_async(_result())

        asyncio.run(run())

        assert len(events) == 1
        assert events[0].status == "failed"
        transcriber.close()

    def test_timeout_is_reported_as_a_failure(self):
        """timeout も通常例外も同じ silent fallback へ落ちないこと。"""
        translator = FlakyTranslator()
        translator.delay = 5.0
        transcriber = _transcriber(translator)
        events = []
        transcriber.set_callbacks(on_translation_status=events.append)

        with patch("livecap_cli.transcription.stream.TRANSLATION_TIMEOUT", 0.05):
            transcriber._apply_translation_sync(_result())

        assert len(events) == 1
        assert events[0].error_type == "timeout"
        transcriber.close()

    def test_callback_exception_does_not_break_the_pipeline(self):
        """通知の失敗で文字起こしが止まるのは本末転倒。"""
        translator = FlakyTranslator()
        transcriber = _transcriber(translator)

        def boom(event):
            raise RuntimeError("consumer bug")

        transcriber.set_callbacks(on_translation_status=boom)
        translator.failing = True

        result = transcriber._apply_translation_sync(_result())  # must not raise

        assert result.translation_state == "failed"
        transcriber.close()

    def test_no_callback_registered_is_a_no_op(self):
        translator = FlakyTranslator()
        translator.failing = True
        transcriber = _transcriber(translator)

        result = transcriber._apply_translation_sync(_result())

        assert result.translation_state == "failed"
        transcriber.close()

    def test_speech_never_reaches_the_event(self):
        """イベントは GUI まで届く。発話が載ってはいけない (Issue #402 D8)。"""
        secret = "来期の人員削減について田中部長と話しました"
        translator = FlakyTranslator()
        transcriber = _transcriber(translator)
        events = []
        transcriber.set_callbacks(on_translation_status=events.append)
        translator.failing = True

        transcriber._apply_translation_sync(_result(secret))

        assert secret not in (events[0].message or "")
        transcriber.close()


class TestTranslationState:
    """個々の字幕が原文のままである理由 (Issue #402 D10)。"""

    def test_translated(self):
        transcriber = _transcriber(FlakyTranslator(translation_text="Hello"))
        result = transcriber._apply_translation_sync(_result())
        assert result.translation_state == "translated"
        assert result.translated_text == "Hello"
        transcriber.close()

    def test_not_requested_without_a_translator(self):
        transcriber = _transcriber()
        result = transcriber._apply_translation_sync(_result())
        assert result.translation_state == "not_requested"
        assert result.translated_text is None
        transcriber.close()

    def test_failed(self):
        translator = FlakyTranslator()
        translator.failing = True
        transcriber = _transcriber(translator)
        result = transcriber._apply_translation_sync(_result())
        assert result.translation_state == "failed"
        transcriber.close()

    def test_empty(self):
        transcriber = _transcriber(FlakyTranslator(translation_text="   "))
        result = transcriber._apply_translation_sync(_result())
        assert result.translation_state == "empty"
        assert result.translated_text is None
        transcriber.close()

    def test_skipped_busy(self):
        translator = FlakyTranslator()
        translator.delay = 3.0
        transcriber = _transcriber(translator)

        with patch("livecap_cli.transcription.stream.TRANSLATION_TIMEOUT", 0.05):
            first = transcriber._apply_translation_sync(_result())
            second = transcriber._apply_translation_sync(_result())

        assert first.translation_state == "failed"         # timed out
        assert second.translation_state == "skipped_busy"  # previous still running
        transcriber.close()

    def test_default_on_a_bare_result(self):
        assert _result().translation_state == "not_requested"


class TestTranslationSingleFlight:
    """輻輳しても backlog を積まない (Issue #402 D10)。

    順番を守って遅れて全部出すより落とす方が良い — 数秒前の発話に対する字幕が
    今の音声に重なって出るくらいなら、その分は原文のままにする。
    """

    def test_only_one_translation_runs_at_a_time(self):
        translator = FlakyTranslator()
        translator.delay = 0.3
        transcriber = _transcriber(translator)

        active, peak, lock = [0], [0], threading.Lock()
        original = translator.translate

        def counting(*args, **kwargs):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            try:
                return original(*args, **kwargs)
            finally:
                with lock:
                    active[0] -= 1

        translator.translate = counting  # type: ignore[method-assign]

        with patch("livecap_cli.transcription.stream.TRANSLATION_TIMEOUT", 0.02):
            for _ in range(5):
                transcriber._apply_translation_sync(_result())

        assert peak[0] == 1
        transcriber.close()

    def test_skips_are_counted(self):
        translator = FlakyTranslator()
        translator.delay = 3.0
        transcriber = _transcriber(translator)

        with patch("livecap_cli.transcription.stream.TRANSLATION_TIMEOUT", 0.02):
            for _ in range(4):
                transcriber._apply_translation_sync(_result())

        assert transcriber._translation_skips >= 2
        transcriber.close()

    def test_stale_result_is_discarded(self):
        """timeout した翻訳が後から完了しても、その結果は誰も読まない。"""
        translator = FlakyTranslator(translation_text="STALE")
        translator.delay = 0.3
        transcriber = _transcriber(translator)

        with patch("livecap_cli.transcription.stream.TRANSLATION_TIMEOUT", 0.02):
            first = transcriber._apply_translation_sync(_result())

        assert first.translated_text is None
        assert first.translation_state == "failed"
        time.sleep(0.5)  # 裏で完了させる
        # 完了しても result には反映されない (誰も future を読まないため)
        assert first.translated_text is None
        transcriber.close()


class TestTranslationExecutorSeparation:
    """翻訳が文字起こしを止めない (Issue #402 D2)。"""

    def test_translation_uses_its_own_worker(self):
        transcriber = _transcriber(FlakyTranslator())
        transcriber._apply_translation_sync(_result())

        assert transcriber._translation_executor is not None
        assert transcriber._translation_executor is not transcriber._executor
        transcriber.close()

    def test_asr_executor_is_free_while_translation_hangs(self):
        """既定 max_workers=1 を共用していた頃は、ここが詰まっていた。"""
        translator = FlakyTranslator()
        translator.delay = 1.0
        transcriber = _transcriber(translator)

        with patch("livecap_cli.transcription.stream.TRANSLATION_TIMEOUT", 0.02):
            transcriber._apply_translation_sync(_result())

            started = time.perf_counter()
            future = transcriber._executor.submit(lambda: "asr work")
            assert future.result(timeout=1.0) == "asr work"
            assert time.perf_counter() - started < 0.5

        transcriber.close()

    def test_close_shuts_down_both_executors(self):
        transcriber = _transcriber(FlakyTranslator())
        transcriber._apply_translation_sync(_result())
        assert transcriber._translation_executor is not None

        transcriber.close()

        assert transcriber._translation_executor is None


class TestTranslationLifecycle:
    """`close()` が返った時点で translator は使われていない (Issue #402)。

    translator は呼び出し側 (CLI / GUI) が所有し、``close()`` の直後に
    ``translator.cleanup()`` が呼ばれる。待たずに返すと、**借りている
    requests.Session を使っている最中に閉じられる**ことになる。
    ``cancel_futures=True`` は実行中の future を止めないので、それだけでは足りない。
    """

    def test_close_waits_for_the_in_flight_translation(self):
        translator = FlakyTranslator()
        translator.delay = 0.6
        transcriber = _transcriber(translator)

        finished = threading.Event()
        original = translator.translate

        def marking(*args, **kwargs):
            try:
                return original(*args, **kwargs)
            finally:
                finished.set()

        translator.translate = marking  # type: ignore[method-assign]

        with patch("livecap_cli.transcription.stream.TRANSLATION_TIMEOUT", 0.05):
            transcriber._apply_translation_sync(_result())

        assert not finished.is_set(), "前提: timeout 時点で worker はまだ動いている"

        transcriber.close()

        assert finished.is_set(), "close() が返った時点で translator が使用中"

    def test_close_gives_up_after_the_drain_timeout(self):
        """待つが、無限には待たない。"""
        translator = FlakyTranslator()
        translator.delay = 5.0
        transcriber = _transcriber(translator)

        with patch("livecap_cli.transcription.stream.TRANSLATION_TIMEOUT", 0.05):
            transcriber._apply_translation_sync(_result())

        with patch("livecap_cli.transcription.stream.TRANSLATION_DRAIN_TIMEOUT", 0.3):
            started = time.perf_counter()
            transcriber.close()
            elapsed = time.perf_counter() - started

        assert 0.2 < elapsed < 2.0, f"{elapsed:.2f}s"

    def test_close_is_immediate_when_nothing_is_running(self):
        transcriber = _transcriber(FlakyTranslator())
        transcriber._apply_translation_sync(_result())

        started = time.perf_counter()
        transcriber.close()

        assert time.perf_counter() - started < 0.2

    def test_destructor_does_not_block(self):
        """GC 中にブロックするのは危険。厳密さより安全に抜けることを優先する。"""
        translator = FlakyTranslator()
        translator.delay = 3.0
        transcriber = _transcriber(translator)

        with patch("livecap_cli.transcription.stream.TRANSLATION_TIMEOUT", 0.05):
            transcriber._apply_translation_sync(_result())

        started = time.perf_counter()
        transcriber.__del__()

        assert time.perf_counter() - started < 0.5


class TestTranslationReset:
    """`reset()` は新セッションとして扱う (Issue #402)。"""

    def test_failure_state_does_not_survive_reset(self):
        """持ち越すと次の障害が通知されない。"""
        translator = FlakyTranslator()
        transcriber = _transcriber(translator)
        events = []
        transcriber.set_callbacks(on_translation_status=events.append)

        translator.failing = True
        transcriber._apply_translation_sync(_result())
        assert [e.status for e in events] == ["failed"]

        transcriber.reset()
        transcriber._apply_translation_sync(_result())

        assert [e.status for e in events] == ["failed", "failed"]
        transcriber.close()

    def test_first_success_after_reset_is_not_a_recovery(self):
        """前セッションの障害に対する recovered が出てはいけない。"""
        translator = FlakyTranslator()
        transcriber = _transcriber(translator)
        events = []
        transcriber.set_callbacks(on_translation_status=events.append)

        translator.failing = True
        transcriber._apply_translation_sync(_result())
        transcriber.reset()
        translator.failing = False
        transcriber._apply_translation_sync(_result())

        assert [e.status for e in events] == ["failed"]
        transcriber.close()

    def test_counters_are_cleared(self):
        translator = FlakyTranslator()
        translator.failing = True
        transcriber = _transcriber(translator)
        transcriber._apply_translation_sync(_result())
        assert transcriber._translation_failures == 1

        transcriber.reset()

        assert transcriber._translation_failures == 0
        assert transcriber._translation_skips == 0
        assert transcriber._translation_healthy is True
        transcriber.close()

    def test_reset_keeps_single_flight(self):
        """in-flight の参照を捨てると、走っている worker と新しい翻訳が
        同じ translator を並行利用してしまう。"""
        translator = FlakyTranslator()
        translator.delay = 1.0
        transcriber = _transcriber(translator)

        with patch("livecap_cli.transcription.stream.TRANSLATION_TIMEOUT", 0.05):
            transcriber._apply_translation_sync(_result())
            transcriber.reset()
            after = transcriber._apply_translation_sync(_result())

        assert after.translation_state == "skipped_busy"
        transcriber.close()

    def test_stale_worker_does_not_pollute_the_new_context(self):
        """reset を跨いで完了した翻訳が、新セッションの文脈へ書き戻さない。"""
        translator = FlakyTranslator(default_context_sentences=3)
        translator.delay = 0.4
        transcriber = _transcriber(translator)

        with patch("livecap_cli.transcription.stream.TRANSLATION_TIMEOUT", 0.05):
            transcriber._apply_translation_sync(_result("前セッションの発話"))

        transcriber.reset()
        time.sleep(0.8)  # 裏で完了させる

        assert "前セッションの発話" not in transcriber._context_buffer
        transcriber.close()
