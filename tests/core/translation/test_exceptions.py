"""
翻訳例外クラスのテスト
"""

from __future__ import annotations

import pytest

from livecap_cli.translation.exceptions import (
    TranslationError,
    TranslationModelError,
    TranslationNetworkError,
    UnsupportedLanguagePairError,
)


class TestExceptionHierarchy:
    """例外クラス階層のテスト"""

    def test_translation_error_is_exception(self):
        """TranslationError は Exception のサブクラス"""
        assert issubclass(TranslationError, Exception)

    def test_network_error_is_translation_error(self):
        """TranslationNetworkError は TranslationError のサブクラス"""
        assert issubclass(TranslationNetworkError, TranslationError)

    def test_model_error_is_translation_error(self):
        """TranslationModelError は TranslationError のサブクラス"""
        assert issubclass(TranslationModelError, TranslationError)

    def test_unsupported_language_pair_is_translation_error(self):
        """UnsupportedLanguagePairError は TranslationError のサブクラス"""
        assert issubclass(UnsupportedLanguagePairError, TranslationError)


class TestUnsupportedLanguagePairError:
    """UnsupportedLanguagePairError のテスト"""

    def test_error_message(self):
        """エラーメッセージの形式"""
        error = UnsupportedLanguagePairError("ja", "ja", "google")
        assert "ja -> ja" in str(error)
        assert "google" in str(error)

    def test_attributes(self):
        """エラー属性"""
        error = UnsupportedLanguagePairError("en", "fr", "opus_mt")
        assert error.source == "en"
        assert error.target == "fr"
        assert error.translator == "opus_mt"

    def test_can_be_caught_as_translation_error(self):
        """TranslationError としてキャッチ可能"""
        with pytest.raises(TranslationError):
            raise UnsupportedLanguagePairError("ja", "en", "test")


class TestNetworkError:
    """TranslationNetworkError のテスト"""

    def test_with_message(self):
        """メッセージ付きエラー"""
        error = TranslationNetworkError("API request failed")
        assert "API request failed" in str(error)


class TestModelError:
    """TranslationModelError のテスト"""

    def test_with_message(self):
        """メッセージ付きエラー"""
        error = TranslationModelError("Model not loaded")
        assert "Model not loaded" in str(error)
