"""
言語コード正規化のテスト
"""

from __future__ import annotations

import pytest

from livecap_cli.translation.lang_codes import (
    get_language_name,
    get_opus_mt_model_name,
    normalize_for_google,
    normalize_for_opus_mt,
    to_iso639_1,
)


class TestToISO639_1:
    """to_iso639_1 のテスト"""

    def test_simple_code(self):
        assert to_iso639_1("ja") == "ja"
        assert to_iso639_1("en") == "en"

    def test_with_region(self):
        assert to_iso639_1("ja-JP") == "ja"
        assert to_iso639_1("en-US") == "en"
        assert to_iso639_1("zh-CN") == "zh"

    def test_uppercase_normalized(self):
        assert to_iso639_1("ZH-TW") == "zh"
        assert to_iso639_1("JA") == "ja"


class TestNormalizeForGoogle:
    """normalize_for_google のテスト"""

    def test_simple_code(self):
        assert normalize_for_google("ja") == "ja"
        assert normalize_for_google("en") == "en"

    def test_chinese_simplified(self):
        assert normalize_for_google("zh") == "zh-CN"
        assert normalize_for_google("zh-CN") == "zh-CN"

    def test_chinese_traditional_preserved(self):
        assert normalize_for_google("zh-TW") == "zh-TW"
        assert normalize_for_google("zh-Hant") == "zh-TW"


class TestNormalizeForOpusMT:
    """normalize_for_opus_mt のテスト"""

    def test_returns_iso639_1(self):
        assert normalize_for_opus_mt("ja") == "ja"
        assert normalize_for_opus_mt("ja-JP") == "ja"
        assert normalize_for_opus_mt("zh-CN") == "zh"


class TestGetLanguageName:
    """get_language_name のテスト"""

    def test_known_languages(self):
        assert get_language_name("ja") == "Japanese"
        assert get_language_name("en") == "English"
        assert get_language_name("zh") == "Simplified Chinese"

    def test_traditional_chinese(self):
        assert get_language_name("zh-TW") == "Traditional Chinese"
        assert get_language_name("zh-Hant") == "Traditional Chinese"


class TestGetOpusMTModelName:
    """get_opus_mt_model_name のテスト"""

    def test_ja_en(self):
        assert get_opus_mt_model_name("ja", "en") == "Helsinki-NLP/opus-mt-ja-en"

    def test_en_ja(self):
        assert get_opus_mt_model_name("en", "ja") == "Helsinki-NLP/opus-mt-en-ja"

    def test_with_region_codes(self):
        # Region codes should be normalized
        assert get_opus_mt_model_name("ja-JP", "en-US") == "Helsinki-NLP/opus-mt-ja-en"
