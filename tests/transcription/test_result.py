"""TranscriptionResult / InterimResult のユニットテスト"""

import pytest
from dataclasses import FrozenInstanceError

from livecap_cli.transcription.result import (
    TranscriptionResult,
    InterimResult,
    _format_srt_time,
)


class TestTranscriptionResult:
    """TranscriptionResult のテスト"""

    def test_create_basic(self):
        """基本的な生成テスト"""
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

    def test_create_with_all_fields(self):
        """全フィールド指定のテスト"""
        result = TranscriptionResult(
            text="Hello",
            start_time=10.5,
            end_time=12.0,
            is_final=False,
            confidence=0.95,
            language="en",
            source_id="mic1",
        )

        assert result.text == "Hello"
        assert result.start_time == 10.5
        assert result.end_time == 12.0
        assert result.is_final is False
        assert result.confidence == 0.95
        assert result.language == "en"
        assert result.source_id == "mic1"

    def test_duration_property(self):
        """duration プロパティのテスト"""
        result = TranscriptionResult(
            text="test",
            start_time=5.0,
            end_time=8.5,
        )

        assert result.duration == 3.5

    def test_frozen_immutable(self):
        """frozen=True によりイミュータブルであることを確認"""
        result = TranscriptionResult(
            text="test",
            start_time=0.0,
            end_time=1.0,
        )

        with pytest.raises(FrozenInstanceError):
            result.text = "modified"  # type: ignore

    def test_to_srt_entry_basic(self):
        """to_srt_entry の基本テスト"""
        result = TranscriptionResult(
            text="テスト文章です。",
            start_time=0.0,
            end_time=2.5,
        )

        srt = result.to_srt_entry(1)

        assert srt == (
            "1\n"
            "00:00:00,000 --> 00:00:02,500\n"
            "テスト文章です。\n"
        )

    def test_to_srt_entry_long_time(self):
        """1時間を超える時刻のテスト"""
        result = TranscriptionResult(
            text="Long video content",
            start_time=3661.5,  # 1:01:01.500
            end_time=3665.123,  # 1:01:05.123
        )

        srt = result.to_srt_entry(42)

        assert srt == (
            "42\n"
            "01:01:01,500 --> 01:01:05,123\n"
            "Long video content\n"
        )

    def test_equality(self):
        """等価性テスト"""
        result1 = TranscriptionResult(
            text="same",
            start_time=0.0,
            end_time=1.0,
        )
        result2 = TranscriptionResult(
            text="same",
            start_time=0.0,
            end_time=1.0,
        )

        assert result1 == result2

    def test_hashable(self):
        """ハッシュ可能であることを確認（セットや辞書キーに使用可能）"""
        result = TranscriptionResult(
            text="test",
            start_time=0.0,
            end_time=1.0,
        )

        # ハッシュが計算できることを確認
        hash_value = hash(result)
        assert isinstance(hash_value, int)

        # セットに追加できることを確認
        result_set = {result}
        assert result in result_set


class TestInterimResult:
    """InterimResult のテスト"""

    def test_create_basic(self):
        """基本的な生成テスト"""
        interim = InterimResult(
            text="話している途中",
            accumulated_time=2.5,
        )

        assert interim.text == "話している途中"
        assert interim.accumulated_time == 2.5
        assert interim.source_id == "default"

    def test_create_with_source_id(self):
        """source_id 指定のテスト"""
        interim = InterimResult(
            text="test",
            accumulated_time=1.0,
            source_id="system_audio",
        )

        assert interim.source_id == "system_audio"

    def test_frozen_immutable(self):
        """frozen=True によりイミュータブルであることを確認"""
        interim = InterimResult(
            text="test",
            accumulated_time=1.0,
        )

        with pytest.raises(FrozenInstanceError):
            interim.text = "modified"  # type: ignore


class TestFormatSrtTime:
    """_format_srt_time 関数のテスト"""

    def test_zero(self):
        """0秒のテスト"""
        assert _format_srt_time(0.0) == "00:00:00,000"

    def test_milliseconds(self):
        """ミリ秒のテスト"""
        assert _format_srt_time(0.123) == "00:00:00,123"
        assert _format_srt_time(0.999) == "00:00:00,999"

    def test_seconds(self):
        """秒のテスト"""
        assert _format_srt_time(45.0) == "00:00:45,000"
        assert _format_srt_time(59.999) == "00:00:59,999"

    def test_minutes(self):
        """分のテスト"""
        assert _format_srt_time(60.0) == "00:01:00,000"
        assert _format_srt_time(125.5) == "00:02:05,500"

    def test_hours(self):
        """時間のテスト"""
        assert _format_srt_time(3600.0) == "01:00:00,000"
        assert _format_srt_time(7325.750) == "02:02:05,750"

    def test_negative_clamped_to_zero(self):
        """負の値は0にクランプされる"""
        assert _format_srt_time(-5.0) == "00:00:00,000"
