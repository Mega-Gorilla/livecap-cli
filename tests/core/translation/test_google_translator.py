"""
GoogleTranslator のテスト
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from livecap_cli.translation.exceptions import (
    TranslationError,
    TranslationNetworkError,
    UnsupportedLanguagePairError,
)
from livecap_cli.translation.impl.google import GoogleTranslator


class TestGoogleTranslatorBasic:
    """GoogleTranslator の基本テスト"""

    def test_initialization(self):
        """初期化テスト"""
        translator = GoogleTranslator()
        assert translator.is_initialized() is True
        assert translator.get_translator_name() == "google"

    def test_supported_pairs_empty(self):
        """全言語ペア対応（空リスト）"""
        translator = GoogleTranslator()
        pairs = translator.get_supported_pairs()
        assert pairs == []

    def test_default_context_sentences(self):
        """デフォルト文脈数"""
        translator = GoogleTranslator()
        assert translator._default_context_sentences == 2

    def test_custom_context_sentences(self):
        """カスタム文脈数"""
        translator = GoogleTranslator(default_context_sentences=5)
        assert translator._default_context_sentences == 5


class TestGoogleTranslatorMocked:
    """モックを使用した GoogleTranslator テスト"""

    def test_translate_basic(self):
        """基本翻訳テスト"""
        with patch(
            "livecap_cli.translation.impl.google.DeepGoogleTranslator"
        ) as mock_gt:
            mock_gt.return_value.translate.return_value = "こんにちは"
            translator = GoogleTranslator()
            result = translator.translate("Hello", "en", "ja")

            assert result.text == "こんにちは"
            assert result.original_text == "Hello"
            assert result.source_lang == "en"
            assert result.target_lang == "ja"

    def test_translate_with_context(self):
        """文脈付き翻訳テスト"""
        with patch(
            "livecap_cli.translation.impl.google.DeepGoogleTranslator"
        ) as mock_gt:
            # 文脈を含めた入力に対して複数行の結果を返す
            mock_gt.return_value.translate.return_value = (
                "Yesterday I played with friends.\nI am tired today."
            )
            translator = GoogleTranslator()
            context = ["昨日は友達と遊んだ。"]
            result = translator.translate("今日は疲れている。", "ja", "en", context=context)

            # 最後の文が抽出される
            assert result.text == "I am tired today."
            assert result.original_text == "今日は疲れている。"

    def test_translate_with_long_context(self):
        """長い文脈は制限される"""
        with patch(
            "livecap_cli.translation.impl.google.DeepGoogleTranslator"
        ) as mock_gt:
            mock_gt.return_value.translate.return_value = "line1\nline2\nline3\nresult"
            translator = GoogleTranslator(default_context_sentences=2)
            context = ["文1", "文2", "文3", "文4"]  # 4文だが2文のみ使用
            result = translator.translate("テスト", "ja", "en", context=context)

            # 呼び出し時の引数を確認
            call_args = mock_gt.return_value.translate.call_args[0][0]
            # 最後の2文 + 現在のテキストが渡される
            assert "文3" in call_args
            assert "文4" in call_args
            assert "テスト" in call_args
            # 最初の文は含まれない
            assert "文1" not in call_args

    def test_translate_empty_text(self):
        """空文字列の翻訳"""
        translator = GoogleTranslator()
        result = translator.translate("", "en", "ja")
        assert result.text == ""
        assert result.original_text == ""

    def test_translate_whitespace_only(self):
        """空白のみの翻訳"""
        translator = GoogleTranslator()
        result = translator.translate("   ", "en", "ja")
        assert result.text == ""
        assert result.original_text == "   "

    def test_translate_same_language_raises(self):
        """同一言語でエラー"""
        translator = GoogleTranslator()
        with pytest.raises(UnsupportedLanguagePairError) as exc_info:
            translator.translate("Hello", "en", "en")
        assert exc_info.value.source == "en"
        assert exc_info.value.target == "en"
        assert exc_info.value.translator == "google"

    def test_translate_same_language_normalized(self):
        """正規化後に同一言語でもエラー"""
        translator = GoogleTranslator()
        with pytest.raises(UnsupportedLanguagePairError):
            # ja-JP と ja は正規化後に同じ
            translator.translate("こんにちは", "ja-JP", "ja")


class TestGoogleTranslatorExceptions:
    """例外処理のテスト"""

    def test_rate_limited_raises_network_error(self):
        """レート制限時に TranslationNetworkError"""
        from deep_translator.exceptions import TooManyRequests

        with patch(
            "livecap_cli.translation.impl.google.DeepGoogleTranslator"
        ) as mock_gt:
            mock_gt.return_value.translate.side_effect = TooManyRequests()
            translator = GoogleTranslator()

            with pytest.raises(TranslationNetworkError, match="Rate limited"):
                translator.translate("Hello", "en", "ja")

    def test_request_error_raises_network_error(self):
        """リクエストエラー時に TranslationNetworkError"""
        from deep_translator.exceptions import RequestError

        with patch(
            "livecap_cli.translation.impl.google.DeepGoogleTranslator"
        ) as mock_gt:
            mock_gt.return_value.translate.side_effect = RequestError()
            translator = GoogleTranslator()

            with pytest.raises(TranslationNetworkError, match="API request failed"):
                translator.translate("Hello", "en", "ja")

    def test_translation_not_found_raises_error(self):
        """翻訳なし時に TranslationError"""
        from deep_translator.exceptions import TranslationNotFound

        with patch(
            "livecap_cli.translation.impl.google.DeepGoogleTranslator"
        ) as mock_gt:
            mock_gt.return_value.translate.side_effect = TranslationNotFound("test")
            translator = GoogleTranslator()

            with pytest.raises(TranslationError, match="Translation not found"):
                translator.translate("Hello", "en", "ja")

    def test_unexpected_error_raises_error(self):
        """予期しないエラー時に TranslationError"""
        with patch(
            "livecap_cli.translation.impl.google.DeepGoogleTranslator"
        ) as mock_gt:
            mock_gt.return_value.translate.side_effect = RuntimeError("Unexpected")
            translator = GoogleTranslator()

            with pytest.raises(TranslationError, match="Unexpected error"):
                translator.translate("Hello", "en", "ja")


class TestGoogleTranslatorRetry:
    """リトライ機能のテスト"""

    def test_retry_on_network_error(self):
        """ネットワークエラー時にリトライ"""
        from deep_translator.exceptions import RequestError

        with patch(
            "livecap_cli.translation.impl.google.DeepGoogleTranslator"
        ) as mock_gt:
            # 2回失敗して3回目に成功
            mock_gt.return_value.translate.side_effect = [
                RequestError(),
                RequestError(),
                "成功",
            ]
            translator = GoogleTranslator()

            # リトライ時間を短縮するためにパッチ
            with patch("livecap_cli.translation.retry.time.sleep"):
                result = translator.translate("test", "en", "ja")

            assert result.text == "成功"
            assert mock_gt.return_value.translate.call_count == 3


class TestGoogleTranslatorExtractLastSentence:
    """_extract_last_sentence のテスト"""

    def test_single_line(self):
        """単一行"""
        translator = GoogleTranslator()
        result = translator._extract_last_sentence("Hello world")
        assert result == "Hello world"

    def test_multiple_lines(self):
        """複数行"""
        translator = GoogleTranslator()
        result = translator._extract_last_sentence("Line 1\nLine 2\nLine 3")
        assert result == "Line 3"

    def test_with_trailing_whitespace(self):
        """末尾空白"""
        translator = GoogleTranslator()
        result = translator._extract_last_sentence("Line 1\nLine 2\n  ")
        assert result == "Line 2"

    def test_empty_string(self):
        """空文字列"""
        translator = GoogleTranslator()
        result = translator._extract_last_sentence("")
        assert result == ""


class TestGoogleTranslatorAsync:
    """非同期翻訳のテスト"""

    def test_translate_async(self):
        """非同期翻訳テスト"""
        import asyncio

        async def run_test():
            with patch(
                "livecap_cli.translation.impl.google.DeepGoogleTranslator"
            ) as mock_gt:
                mock_gt.return_value.translate.return_value = "Hello"
                translator = GoogleTranslator()
                result = await translator.translate_async("こんにちは", "ja", "en")
                return result

        result = asyncio.run(run_test())
        assert result.text == "Hello"
        assert result.original_text == "こんにちは"


@pytest.mark.network
class TestGoogleTranslatorNetwork:
    """実ネットワークを使用したテスト（CI ではスキップ）"""

    def test_translate_ja_to_en(self):
        """日本語→英語の実翻訳"""
        translator = GoogleTranslator()
        result = translator.translate("こんにちは", "ja", "en")
        # 何らかの翻訳が返る
        assert result.text
        assert len(result.text) > 0
        # "Hello" または類似の挨拶が含まれる可能性が高い
        assert result.original_text == "こんにちは"

    def test_translate_en_to_ja(self):
        """英語→日本語の実翻訳"""
        translator = GoogleTranslator()
        result = translator.translate("Hello", "en", "ja")
        assert result.text
        assert len(result.text) > 0

    def test_translate_with_context_real(self):
        """文脈付き翻訳の実テスト"""
        translator = GoogleTranslator()
        context = ["昨日は友達と遊んだ。"]
        result = translator.translate("今日は疲れている。", "ja", "en", context=context)
        assert result.text
        # tired/exhausted などが含まれる可能性
        assert result.original_text == "今日は疲れている。"
