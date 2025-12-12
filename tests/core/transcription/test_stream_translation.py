"""StreamTranscriber 翻訳統合のユニットテスト"""

from __future__ import annotations

import time
from collections import deque
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from livecap_core.transcription.stream import (
    MAX_CONTEXT_BUFFER,
    TRANSLATION_TIMEOUT,
    _DEFAULT_TRANSLATION_TIMEOUT,
    _get_translation_timeout,
    StreamTranscriber,
    TranscriptionEngine,
)
from livecap_core.translation.base import BaseTranslator
from livecap_core.translation.result import TranslationResult
from livecap_core.vad import VADSegment, VADState


class MockEngine:
    """テスト用のモックエンジン"""

    def __init__(self):
        self.transcribe_calls = []

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Tuple[str, float]:
        self.transcribe_calls.append((audio, sample_rate))
        return "テスト文字起こし", 0.95

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

        translated, target_lang = transcriber._translate_text("こんにちは")

        assert translated == "Hello"
        assert target_lang == "en"
        assert len(translator.translate_calls) == 1
        assert translator.translate_calls[0][0] == "こんにちは"
        assert translator.translate_calls[0][1] == "ja"
        assert translator.translate_calls[0][2] == "en"

    def test_translate_text_without_translator(self):
        """translator なしで翻訳なし"""
        engine = MockEngine()
        vad = MockVADProcessor()
        transcriber = StreamTranscriber(engine=engine, vad_processor=vad)

        translated, target_lang = transcriber._translate_text("こんにちは")

        assert translated is None
        assert target_lang is None

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

        with caplog.at_level("WARNING"):
            translated, target_lang = transcriber._translate_text("こんにちは")

        assert translated is None
        assert target_lang is None
        assert "Translation failed" in caplog.text

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

    @patch("livecap_core.transcription.stream.TRANSLATION_TIMEOUT", 0.1)
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

        with caplog.at_level("WARNING"):
            translated, target_lang = transcriber._translate_text("こんにちは")

        assert translated is None
        assert target_lang is None
        assert "timed out" in caplog.text

    @patch("livecap_core.transcription.stream.TRANSLATION_TIMEOUT", 0.1)
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

    def test_translation_timeout_default_is_10_seconds(self):
        """デフォルトタイムアウトが10秒であることを確認"""
        assert _DEFAULT_TRANSLATION_TIMEOUT == 10.0

    def test_get_translation_timeout_without_env(self, monkeypatch):
        """環境変数未設定時はデフォルト値を返す"""
        monkeypatch.delenv("LIVECAP_TRANSLATION_TIMEOUT", raising=False)
        assert _get_translation_timeout() == 10.0

    def test_get_translation_timeout_with_valid_env(self, monkeypatch):
        """有効な環境変数が設定されている場合はその値を使用"""
        monkeypatch.setenv("LIVECAP_TRANSLATION_TIMEOUT", "20.0")
        assert _get_translation_timeout() == 20.0

    def test_get_translation_timeout_with_invalid_env(self, monkeypatch, caplog):
        """無効な環境変数（非数値）はデフォルトにフォールバック"""
        monkeypatch.setenv("LIVECAP_TRANSLATION_TIMEOUT", "invalid")
        with caplog.at_level("WARNING"):
            result = _get_translation_timeout()
        assert result == 10.0
        assert "Invalid LIVECAP_TRANSLATION_TIMEOUT" in caplog.text

    def test_get_translation_timeout_with_zero(self, monkeypatch, caplog):
        """0 はデフォルトにフォールバック"""
        monkeypatch.setenv("LIVECAP_TRANSLATION_TIMEOUT", "0")
        with caplog.at_level("WARNING"):
            result = _get_translation_timeout()
        assert result == 10.0
        assert "must be positive" in caplog.text

    def test_get_translation_timeout_with_negative(self, monkeypatch, caplog):
        """負の値はデフォルトにフォールバック"""
        monkeypatch.setenv("LIVECAP_TRANSLATION_TIMEOUT", "-5")
        with caplog.at_level("WARNING"):
            result = _get_translation_timeout()
        assert result == 10.0
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

        translated, target_lang = transcriber._do_translate_direct("こんにちは")

        assert translated == "Hello"
        assert target_lang == "en"
        assert "こんにちは" in transcriber._context_buffer

    def test_do_translate_direct_without_translator(self):
        """translator なしで None を返す"""
        engine = MockEngine()
        vad = MockVADProcessor()
        transcriber = StreamTranscriber(engine=engine, vad_processor=vad)

        translated, target_lang = transcriber._do_translate_direct("こんにちは")

        assert translated is None
        assert target_lang is None

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

        with caplog.at_level("WARNING"):
            translated, target_lang = transcriber._do_translate_direct("こんにちは")

        assert translated is None
        assert target_lang is None
        assert "Translation failed" in caplog.text
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
