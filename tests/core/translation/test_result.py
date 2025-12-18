"""
TranslationResult のテスト
"""

from __future__ import annotations

import pytest

from livecap_cli.translation.result import TranslationResult


class TestTranslationResult:
    """TranslationResult のテスト"""

    def test_basic_creation(self):
        """基本的な作成"""
        result = TranslationResult(
            text="Hello",
            original_text="こんにちは",
            source_lang="ja",
            target_lang="en",
        )
        assert result.text == "Hello"
        assert result.original_text == "こんにちは"
        assert result.source_lang == "ja"
        assert result.target_lang == "en"
        assert result.confidence is None
        assert result.source_id == "default"

    def test_with_confidence(self):
        """信頼度付き"""
        result = TranslationResult(
            text="Hello",
            original_text="こんにちは",
            source_lang="ja",
            target_lang="en",
            confidence=0.95,
        )
        assert result.confidence == 0.95

    def test_with_source_id(self):
        """ソース ID 付き"""
        result = TranslationResult(
            text="Hello",
            original_text="こんにちは",
            source_lang="ja",
            target_lang="en",
            source_id="microphone_1",
        )
        assert result.source_id == "microphone_1"

    def test_to_event_dict(self):
        """イベント辞書への変換"""
        result = TranslationResult(
            text="Hello",
            original_text="こんにちは",
            source_lang="ja",
            target_lang="en",
            confidence=0.9,
            source_id="test_source",
        )
        event = result.to_event_dict()

        assert event["event_type"] == "translation_result"
        assert event["translated_text"] == "Hello"
        assert event["original_text"] == "こんにちは"
        assert event["source_language"] == "ja"
        assert event["target_language"] == "en"
        assert event["confidence"] == 0.9
        assert event["source_id"] == "test_source"

    def test_to_event_dict_without_confidence(self):
        """信頼度なしのイベント変換"""
        result = TranslationResult(
            text="Hello",
            original_text="こんにちは",
            source_lang="ja",
            target_lang="en",
        )
        event = result.to_event_dict()

        # confidence が None の場合も含まれる（TypedDict の定義による）
        assert event["event_type"] == "translation_result"
        assert event["confidence"] is None
