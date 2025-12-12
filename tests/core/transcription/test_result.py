"""TranscriptionResult のユニットテスト"""

from __future__ import annotations

import pytest

from livecap_core.transcription.result import TranscriptionResult, InterimResult


class TestTranscriptionResult:
    """TranscriptionResult のテスト"""

    def test_basic_creation(self):
        """基本的なインスタンス生成"""
        result = TranscriptionResult(
            text="こんにちは",
            start_time=0.0,
            end_time=1.5,
        )
        assert result.text == "こんにちは"
        assert result.start_time == 0.0
        assert result.end_time == 1.5
        assert result.is_final is True
        assert result.confidence == 1.0
        assert result.language == ""
        assert result.source_id == "default"
        # Phase 5: 翻訳フィールドはデフォルトで None
        assert result.translated_text is None
        assert result.target_language is None

    def test_with_all_fields(self):
        """全フィールドを指定"""
        result = TranscriptionResult(
            text="こんにちは",
            start_time=0.0,
            end_time=1.5,
            is_final=True,
            confidence=0.95,
            language="ja",
            source_id="mic-1",
            translated_text="Hello",
            target_language="en",
        )
        assert result.text == "こんにちは"
        assert result.confidence == 0.95
        assert result.language == "ja"
        assert result.source_id == "mic-1"
        assert result.translated_text == "Hello"
        assert result.target_language == "en"

    def test_duration_property(self):
        """duration プロパティ"""
        result = TranscriptionResult(
            text="test",
            start_time=1.0,
            end_time=3.5,
        )
        assert result.duration == 2.5

    def test_to_srt_entry(self):
        """SRT 形式への変換"""
        result = TranscriptionResult(
            text="テスト字幕",
            start_time=0.0,
            end_time=2.0,
        )
        srt = result.to_srt_entry(1)
        assert "1\n" in srt
        assert "00:00:00,000 --> 00:00:02,000" in srt
        assert "テスト字幕" in srt

    def test_immutable(self):
        """frozen=True でイミュータブル"""
        result = TranscriptionResult(
            text="test",
            start_time=0.0,
            end_time=1.0,
        )
        with pytest.raises(AttributeError):
            result.text = "modified"  # type: ignore

    def test_translation_fields_optional(self):
        """翻訳フィールドはオプショナル"""
        # 翻訳なしの結果
        result_no_trans = TranscriptionResult(
            text="テスト",
            start_time=0.0,
            end_time=1.0,
        )
        assert result_no_trans.translated_text is None
        assert result_no_trans.target_language is None

        # 翻訳ありの結果
        result_with_trans = TranscriptionResult(
            text="テスト",
            start_time=0.0,
            end_time=1.0,
            translated_text="Test",
            target_language="en",
        )
        assert result_with_trans.translated_text == "Test"
        assert result_with_trans.target_language == "en"

    def test_translation_failure_representation(self):
        """翻訳失敗時の表現（translated_text=None）"""
        result = TranscriptionResult(
            text="こんにちは",
            start_time=0.0,
            end_time=1.0,
            language="ja",
            translated_text=None,  # 翻訳失敗
            target_language=None,
        )
        assert result.text == "こんにちは"
        assert result.language == "ja"
        assert result.translated_text is None
        assert result.target_language is None


class TestInterimResult:
    """InterimResult のテスト"""

    def test_basic_creation(self):
        """基本的なインスタンス生成"""
        interim = InterimResult(
            text="こんに",
            accumulated_time=0.5,
        )
        assert interim.text == "こんに"
        assert interim.accumulated_time == 0.5
        assert interim.source_id == "default"

    def test_with_source_id(self):
        """source_id を指定"""
        interim = InterimResult(
            text="test",
            accumulated_time=1.0,
            source_id="mic-1",
        )
        assert interim.source_id == "mic-1"
