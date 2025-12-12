"""FileTranscriptionPipeline 翻訳統合のユニットテスト (Phase 6a)"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from livecap_core.transcription.file_pipeline import (
    MAX_CONTEXT_BUFFER,
    FileSubtitleSegment,
    FileTranscriptionPipeline,
)
from livecap_core.translation.base import BaseTranslator
from livecap_core.translation.result import TranslationResult


class MockTranslator(BaseTranslator):
    """テスト用のモック Translator"""

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


class TestFileSubtitleSegmentTranslationFields:
    """FileSubtitleSegment 翻訳フィールドのテスト"""

    def test_default_translation_fields_are_none(self):
        """デフォルトでは翻訳フィールドは None"""
        segment = FileSubtitleSegment(
            index=1,
            start=0.0,
            end=1.0,
            text="テスト",
        )
        assert segment.translated_text is None
        assert segment.target_language is None

    def test_translation_fields_can_be_set(self):
        """翻訳フィールドを設定できる"""
        segment = FileSubtitleSegment(
            index=1,
            start=0.0,
            end=1.0,
            text="こんにちは",
            translated_text="Hello",
            target_language="en",
        )
        assert segment.translated_text == "Hello"
        assert segment.target_language == "en"

    def test_backward_compatible_with_existing_code(self):
        """既存コード（位置引数）との後方互換性"""
        # 既存の位置引数での作成（翻訳フィールドなし）
        segment = FileSubtitleSegment(1, 0.0, 1.0, "テスト")
        assert segment.index == 1
        assert segment.start == 0.0
        assert segment.end == 1.0
        assert segment.text == "テスト"
        assert segment.translated_text is None
        assert segment.target_language is None


class TestValidateTranslatorParams:
    """_validate_translator_params のテスト"""

    def test_no_translator_passes(self):
        """translator=None の場合はバリデーション通過"""
        # Should not raise
        FileTranscriptionPipeline._validate_translator_params(None, None, None)

    def test_uninitialized_translator_raises(self):
        """未初期化の translator でエラー"""
        translator = MockTranslator(initialized=False)

        with pytest.raises(ValueError, match="not initialized"):
            FileTranscriptionPipeline._validate_translator_params(
                translator, "ja", "en"
            )

    def test_missing_source_lang_raises(self):
        """source_lang なしでエラー"""
        translator = MockTranslator()

        with pytest.raises(ValueError, match="source_lang and target_lang are required"):
            FileTranscriptionPipeline._validate_translator_params(
                translator, None, "en"
            )

    def test_missing_target_lang_raises(self):
        """target_lang なしでエラー"""
        translator = MockTranslator()

        with pytest.raises(ValueError, match="source_lang and target_lang are required"):
            FileTranscriptionPipeline._validate_translator_params(
                translator, "ja", None
            )

    def test_unsupported_pair_warns(self, caplog):
        """未サポートの言語ペアで警告"""
        translator = MockTranslator()  # supports ja-en, en-ja

        with caplog.at_level("WARNING"):
            FileTranscriptionPipeline._validate_translator_params(
                translator, "fr", "de"  # 未サポート
            )

        assert "may not be supported" in caplog.text

    def test_empty_source_lang_raises(self):
        """空文字列の source_lang でエラー（"required" エラーでキャッチ）"""
        translator = MockTranslator()

        # 空文字列は falsy なので "required" エラーになる
        with pytest.raises(ValueError, match="required"):
            FileTranscriptionPipeline._validate_translator_params(
                translator, "", "en"
            )

    def test_empty_target_lang_raises(self):
        """空文字列の target_lang でエラー（"required" エラーでキャッチ）"""
        translator = MockTranslator()

        # 空文字列は falsy なので "required" エラーになる
        with pytest.raises(ValueError, match="required"):
            FileTranscriptionPipeline._validate_translator_params(
                translator, "ja", ""
            )

    def test_whitespace_only_lang_raises(self):
        """空白のみの言語コードでエラー"""
        translator = MockTranslator()

        with pytest.raises(ValueError, match="cannot be empty or whitespace"):
            FileTranscriptionPipeline._validate_translator_params(
                translator, "  ", "en"
            )


class TestTranslateText:
    """_translate_text のテスト"""

    def test_translate_text_success(self):
        """翻訳が正常に動作"""
        pipeline = FileTranscriptionPipeline()
        translator = MockTranslator(translation_text="Hello")
        context_buffer: deque[str] = deque(maxlen=MAX_CONTEXT_BUFFER)

        translated, target_lang = pipeline._translate_text(
            text="こんにちは",
            translator=translator,
            source_lang="ja",
            target_lang="en",
            context_buffer=context_buffer,
        )

        assert translated == "Hello"
        assert target_lang == "en"
        assert len(translator.translate_calls) == 1
        pipeline.close()

    def test_translate_text_with_context(self):
        """文脈が正しく渡される"""
        pipeline = FileTranscriptionPipeline()
        translator = MockTranslator(default_context_sentences=2)
        context_buffer: deque[str] = deque(["前文1", "前文2", "前文3"], maxlen=MAX_CONTEXT_BUFFER)

        pipeline._translate_text(
            text="現在の文",
            translator=translator,
            source_lang="ja",
            target_lang="en",
            context_buffer=context_buffer,
        )

        # default_context_sentences=2 なので直近2文が渡される
        assert translator.translate_calls[0][3] == ["前文2", "前文3"]
        pipeline.close()

    def test_translate_text_failure_returns_none(self, caplog):
        """翻訳失敗時は (None, None) を返す"""
        pipeline = FileTranscriptionPipeline()
        translator = MockTranslator()

        def raise_error(*args, **kwargs):
            raise Exception("API Error")

        translator.translate = raise_error  # type: ignore
        context_buffer: deque[str] = deque(maxlen=MAX_CONTEXT_BUFFER)

        with caplog.at_level("WARNING"):
            translated, target_lang = pipeline._translate_text(
                text="テスト",
                translator=translator,
                source_lang="ja",
                target_lang="en",
                context_buffer=context_buffer,
            )

        assert translated is None
        assert target_lang is None
        assert "Translation failed" in caplog.text
        pipeline.close()

    def test_translate_text_with_timeout(self):
        """タイムアウト設定時の正常動作"""
        pipeline = FileTranscriptionPipeline()
        translator = MockTranslator(translation_text="Translated")
        context_buffer: deque[str] = deque(maxlen=MAX_CONTEXT_BUFFER)

        translated, target_lang = pipeline._translate_text(
            text="テスト",
            translator=translator,
            source_lang="ja",
            target_lang="en",
            context_buffer=context_buffer,
            timeout=5.0,  # 十分長いタイムアウト
        )

        assert translated == "Translated"
        assert target_lang == "en"
        pipeline.close()

    def test_translate_text_timeout_returns_none(self, caplog):
        """タイムアウト時は (None, None) を返す"""
        pipeline = FileTranscriptionPipeline()
        translator = MockTranslator()

        def slow_translate(*args, **kwargs):
            time.sleep(1.0)  # 遅い翻訳
            return TranslationResult(
                text="Should not reach",
                original_text=args[0],
                source_lang=args[1],
                target_lang=args[2],
            )

        translator.translate = slow_translate  # type: ignore
        context_buffer: deque[str] = deque(maxlen=MAX_CONTEXT_BUFFER)

        with caplog.at_level("WARNING"):
            translated, target_lang = pipeline._translate_text(
                text="テスト",
                translator=translator,
                source_lang="ja",
                target_lang="en",
                context_buffer=context_buffer,
                timeout=0.1,  # 短いタイムアウト
            )

        assert translated is None
        assert target_lang is None
        assert "timed out" in caplog.text
        pipeline.close()

    def test_translate_text_zero_timeout_treated_as_no_timeout(self):
        """timeout=0 はタイムアウトなしとして扱われる"""
        pipeline = FileTranscriptionPipeline()
        translator = MockTranslator(translation_text="Translated")
        context_buffer: deque[str] = deque(maxlen=MAX_CONTEXT_BUFFER)

        # timeout=0 は無効値として扱われ、タイムアウトなしで実行
        translated, target_lang = pipeline._translate_text(
            text="テスト",
            translator=translator,
            source_lang="ja",
            target_lang="en",
            context_buffer=context_buffer,
            timeout=0,  # 0 は無効 → タイムアウトなし
        )

        assert translated == "Translated"
        assert target_lang == "en"
        pipeline.close()

    def test_translate_text_negative_timeout_treated_as_no_timeout(self):
        """timeout=-1 はタイムアウトなしとして扱われる"""
        pipeline = FileTranscriptionPipeline()
        translator = MockTranslator(translation_text="Translated")
        context_buffer: deque[str] = deque(maxlen=MAX_CONTEXT_BUFFER)

        # 負のタイムアウトも無効値として扱われる
        translated, target_lang = pipeline._translate_text(
            text="テスト",
            translator=translator,
            source_lang="ja",
            target_lang="en",
            context_buffer=context_buffer,
            timeout=-1.0,  # 負の値は無効 → タイムアウトなし
        )

        assert translated == "Translated"
        assert target_lang == "en"
        pipeline.close()


class TestWriteTranslatedSrt:
    """_write_translated_srt のテスト"""

    def test_write_translated_srt(self, tmp_path):
        """翻訳済み SRT ファイルが正しく出力される"""
        pipeline = FileTranscriptionPipeline()
        source = tmp_path / "test.wav"
        source.touch()

        subtitles = [
            FileSubtitleSegment(
                index=1, start=0.0, end=1.0, text="こんにちは",
                translated_text="Hello", target_language="en",
            ),
            FileSubtitleSegment(
                index=2, start=1.0, end=2.0, text="さようなら",
                translated_text="Goodbye", target_language="en",
            ),
        ]

        output_path = pipeline._write_translated_srt(source, subtitles, "en")

        assert output_path is not None
        assert output_path.name == "test_en.srt"
        assert output_path.exists()

        content = output_path.read_text()
        assert "Hello" in content
        assert "Goodbye" in content
        pipeline.close()

    def test_write_translated_srt_no_translations(self, tmp_path, caplog):
        """翻訳がない場合は None を返す"""
        pipeline = FileTranscriptionPipeline()
        source = tmp_path / "test.wav"
        source.touch()

        subtitles = [
            FileSubtitleSegment(
                index=1, start=0.0, end=1.0, text="こんにちは",
                translated_text=None, target_language=None,
            ),
        ]

        with caplog.at_level("WARNING"):
            output_path = pipeline._write_translated_srt(source, subtitles, "en")

        assert output_path is None
        assert "No translated segments" in caplog.text
        pipeline.close()


class TestProcessFileWithTranslation:
    """process_file の翻訳統合テスト"""

    def test_process_file_without_translator(self, tmp_path):
        """translator なしの後方互換動作"""
        # Create a test audio file
        audio_path = tmp_path / "test.wav"
        # Write a minimal WAV file
        import wave
        with wave.open(str(audio_path), "w") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(np.zeros(16000, dtype=np.int16).tobytes())

        def mock_transcriber(audio: np.ndarray, sample_rate: int) -> str:
            return "Transcribed text"

        pipeline = FileTranscriptionPipeline()
        result = pipeline.process_file(
            audio_path,
            segment_transcriber=mock_transcriber,
            write_subtitles=False,
        )

        assert result.success
        assert len(result.subtitles) == 1
        assert result.subtitles[0].text == "Transcribed text"
        assert result.subtitles[0].translated_text is None
        assert result.subtitles[0].target_language is None
        pipeline.close()

    def test_process_file_with_translator(self, tmp_path):
        """translator ありの翻訳処理"""
        # Create a test audio file
        audio_path = tmp_path / "test.wav"
        import wave
        with wave.open(str(audio_path), "w") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(np.zeros(16000, dtype=np.int16).tobytes())

        def mock_transcriber(audio: np.ndarray, sample_rate: int) -> str:
            return "こんにちは"

        translator = MockTranslator(translation_text="Hello")

        pipeline = FileTranscriptionPipeline()
        result = pipeline.process_file(
            audio_path,
            segment_transcriber=mock_transcriber,
            translator=translator,
            source_lang="ja",
            target_lang="en",
            write_subtitles=False,
        )

        assert result.success
        assert len(result.subtitles) == 1
        assert result.subtitles[0].text == "こんにちは"
        assert result.subtitles[0].translated_text == "Hello"
        assert result.subtitles[0].target_language == "en"
        assert result.metadata.get("translation_enabled") is True
        pipeline.close()

    def test_process_file_translator_not_initialized_raises(self, tmp_path):
        """未初期化の translator でエラー"""
        audio_path = tmp_path / "test.wav"
        audio_path.touch()

        translator = MockTranslator(initialized=False)

        pipeline = FileTranscriptionPipeline()
        with pytest.raises(ValueError, match="not initialized"):
            pipeline.process_file(
                audio_path,
                segment_transcriber=lambda a, s: "text",
                translator=translator,
                source_lang="ja",
                target_lang="en",
            )
        pipeline.close()

    def test_process_file_missing_lang_raises(self, tmp_path):
        """言語パラメータなしでエラー"""
        audio_path = tmp_path / "test.wav"
        audio_path.touch()

        translator = MockTranslator()

        pipeline = FileTranscriptionPipeline()
        with pytest.raises(ValueError, match="source_lang and target_lang"):
            pipeline.process_file(
                audio_path,
                segment_transcriber=lambda a, s: "text",
                translator=translator,
                # source_lang と target_lang がない
            )
        pipeline.close()


class TestContextBufferFileScope:
    """文脈バッファのファイルスコープテスト"""

    def test_context_buffer_resets_between_files(self, tmp_path):
        """ファイル間で文脈バッファがリセットされる"""
        # Create two test audio files
        for name in ["test1.wav", "test2.wav"]:
            audio_path = tmp_path / name
            import wave
            with wave.open(str(audio_path), "w") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(np.zeros(16000, dtype=np.int16).tobytes())

        call_count = [0]

        def mock_transcriber(audio: np.ndarray, sample_rate: int) -> str:
            call_count[0] += 1
            return f"Text {call_count[0]}"

        translator = MockTranslator()

        pipeline = FileTranscriptionPipeline()

        # Process first file
        result1 = pipeline.process_file(
            tmp_path / "test1.wav",
            segment_transcriber=mock_transcriber,
            translator=translator,
            source_lang="ja",
            target_lang="en",
            write_subtitles=False,
        )

        # 最初のファイルの翻訳呼び出しでは文脈なし
        first_file_context = translator.translate_calls[0][3]
        assert first_file_context is None or first_file_context == []

        # Process second file
        translator.translate_calls.clear()
        result2 = pipeline.process_file(
            tmp_path / "test2.wav",
            segment_transcriber=mock_transcriber,
            translator=translator,
            source_lang="ja",
            target_lang="en",
            write_subtitles=False,
        )

        # 2番目のファイルでも文脈なし（リセットされている）
        second_file_context = translator.translate_calls[0][3]
        assert second_file_context is None or second_file_context == []

        pipeline.close()
