"""
OpusMTTranslator のテスト
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# Check if dependencies are available
try:
    import ctranslate2
    import transformers

    HAS_OPUS_MT_DEPS = True
except ImportError:
    HAS_OPUS_MT_DEPS = False

# Skip all tests if dependencies not available
pytestmark = pytest.mark.skipif(
    not HAS_OPUS_MT_DEPS,
    reason="OPUS-MT dependencies (ctranslate2, transformers) not installed",
)

from livecap_core.translation.exceptions import (
    TranslationModelError,
    UnsupportedLanguagePairError,
)
from livecap_core.translation.impl.opus_mt import OpusMTTranslator


class TestOpusMTTranslatorBasic:
    """OpusMTTranslator の基本テスト"""

    def test_initialization_default(self):
        """デフォルト初期化"""
        translator = OpusMTTranslator()
        assert translator.source_lang == "ja"
        assert translator.target_lang == "en"
        assert translator.model_name == "Helsinki-NLP/opus-mt-ja-en"
        assert translator.device == "cpu"
        assert translator.compute_type == "int8"
        assert translator.is_initialized() is False

    def test_initialization_custom_langs(self):
        """カスタム言語ペアで初期化"""
        translator = OpusMTTranslator(source_lang="en", target_lang="ja")
        assert translator.source_lang == "en"
        assert translator.target_lang == "ja"
        assert translator.model_name == "Helsinki-NLP/opus-mt-en-ja"

    def test_initialization_custom_model_name(self):
        """カスタムモデル名で初期化"""
        translator = OpusMTTranslator(
            source_lang="ja",
            target_lang="en",
            model_name="custom/my-model",
        )
        assert translator.model_name == "custom/my-model"

    def test_initialization_custom_device(self):
        """カスタムデバイスで初期化"""
        translator = OpusMTTranslator(device="cuda", compute_type="float16")
        assert translator.device == "cuda"
        assert translator.compute_type == "float16"

    def test_get_translator_name(self):
        """翻訳エンジン名"""
        translator = OpusMTTranslator()
        assert translator.get_translator_name() == "opus_mt"

    def test_get_supported_pairs(self):
        """サポート言語ペア"""
        translator = OpusMTTranslator(source_lang="ja", target_lang="en")
        pairs = translator.get_supported_pairs()
        assert pairs == [("ja", "en")]

    def test_default_context_sentences(self):
        """デフォルト文脈数"""
        translator = OpusMTTranslator()
        assert translator._default_context_sentences == 2

    def test_custom_context_sentences(self):
        """カスタム文脈数"""
        translator = OpusMTTranslator(default_context_sentences=5)
        assert translator._default_context_sentences == 5


class TestOpusMTTranslatorNotLoaded:
    """モデル未ロード時のテスト"""

    def test_translate_without_load_raises(self):
        """モデル未ロードで翻訳するとエラー"""
        translator = OpusMTTranslator()
        with pytest.raises(TranslationModelError, match="Model not loaded"):
            translator.translate("Hello", "en", "ja")


class TestOpusMTTranslatorMocked:
    """モックを使用した OpusMTTranslator テスト"""

    @pytest.fixture
    def mock_translator(self):
        """モック済みトランスレータ"""
        translator = OpusMTTranslator(source_lang="ja", target_lang="en")

        # モックモデルとトークナイザー
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # トークナイザーの動作を設定
        mock_tokenizer.encode.return_value = [1, 2, 3, 4]
        mock_tokenizer.convert_ids_to_tokens.return_value = ["▁Hello", "▁world"]
        mock_tokenizer.convert_tokens_to_ids.return_value = [10, 20]
        mock_tokenizer.decode.return_value = "Hello world"

        # モデルの翻訳結果
        mock_result = MagicMock()
        mock_result.hypotheses = [["▁Hello", "▁world"]]
        mock_model.translate_batch.return_value = [mock_result]

        translator._model = mock_model
        translator._tokenizer = mock_tokenizer
        translator._initialized = True

        return translator

    def test_translate_basic(self, mock_translator):
        """基本翻訳テスト"""
        result = mock_translator.translate("こんにちは", "ja", "en")

        assert result.text == "Hello world"
        assert result.original_text == "こんにちは"
        assert result.source_lang == "ja"
        assert result.target_lang == "en"

    def test_translate_empty_text(self, mock_translator):
        """空文字列の翻訳"""
        result = mock_translator.translate("", "ja", "en")
        assert result.text == ""
        assert result.original_text == ""

    def test_translate_whitespace_only(self, mock_translator):
        """空白のみの翻訳"""
        result = mock_translator.translate("   ", "ja", "en")
        assert result.text == ""
        assert result.original_text == "   "

    def test_translate_same_language_raises(self, mock_translator):
        """同一言語でエラー"""
        with pytest.raises(UnsupportedLanguagePairError) as exc_info:
            mock_translator.translate("Hello", "en", "en")
        assert exc_info.value.source == "en"
        assert exc_info.value.target == "en"
        assert exc_info.value.translator == "opus_mt"

    def test_translate_with_context(self, mock_translator):
        """文脈付き翻訳"""
        mock_translator._tokenizer.decode.return_value = "Line1\nLine2\nHello world"
        context = ["前の文。"]
        result = mock_translator.translate("こんにちは", "ja", "en", context=context)

        # 最後の行が抽出される
        assert result.text == "Hello world"

    def test_translate_with_long_context(self, mock_translator):
        """長い文脈は制限される"""
        mock_translator._default_context_sentences = 2
        context = ["文1", "文2", "文3", "文4"]
        mock_translator.translate("テスト", "ja", "en", context=context)

        # encode が呼ばれた引数を確認
        call_args = mock_translator._tokenizer.encode.call_args[0][0]
        # 最後の2文 + 現在のテキストが渡される
        assert "文3" in call_args
        assert "文4" in call_args
        assert "テスト" in call_args
        # 最初の文は含まれない
        assert "文1" not in call_args


class TestOpusMTTranslatorExtractRelevantPart:
    """_extract_relevant_part のテスト"""

    def test_single_line(self):
        """単一行"""
        translator = OpusMTTranslator()
        result = translator._extract_relevant_part("Hello world")
        assert result == "Hello world"

    def test_multiple_lines(self):
        """複数行"""
        translator = OpusMTTranslator()
        result = translator._extract_relevant_part("Line 1\nLine 2\nLine 3")
        assert result == "Line 3"

    def test_with_trailing_whitespace(self):
        """末尾空白"""
        translator = OpusMTTranslator()
        result = translator._extract_relevant_part("Line 1\nLine 2\n  ")
        assert result == "Line 2"

    def test_empty_string(self):
        """空文字列"""
        translator = OpusMTTranslator()
        result = translator._extract_relevant_part("")
        assert result == ""


class TestOpusMTTranslatorCleanup:
    """cleanup のテスト"""

    def test_cleanup(self):
        """クリーンアップ"""
        translator = OpusMTTranslator()
        translator._model = MagicMock()
        translator._tokenizer = MagicMock()
        translator._initialized = True

        translator.cleanup()

        assert translator._model is None
        assert translator._tokenizer is None
        assert translator._initialized is False

    def test_cleanup_when_not_initialized(self):
        """未初期化でもクリーンアップ可能"""
        translator = OpusMTTranslator()
        # エラーなく実行できる
        translator.cleanup()
        assert translator._initialized is False


class TestOpusMTTranslatorAsync:
    """非同期翻訳のテスト"""

    def test_translate_async(self):
        """非同期翻訳テスト"""
        import asyncio

        translator = OpusMTTranslator()

        # モック設定
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.convert_ids_to_tokens.return_value = ["▁Hello"]
        mock_tokenizer.convert_tokens_to_ids.return_value = [10]
        mock_tokenizer.decode.return_value = "Hello"

        mock_result = MagicMock()
        mock_result.hypotheses = [["▁Hello"]]
        mock_model.translate_batch.return_value = [mock_result]

        translator._model = mock_model
        translator._tokenizer = mock_tokenizer
        translator._initialized = True

        async def run_test():
            return await translator.translate_async("こんにちは", "ja", "en")

        result = asyncio.run(run_test())
        assert result.text == "Hello"
        assert result.original_text == "こんにちは"


@pytest.mark.slow
class TestOpusMTTranslatorIntegration:
    """統合テスト（実モデルロード、要 translation-local extra）"""

    def test_load_model_ja_en(self):
        """日本語→英語モデルのロード"""
        translator = OpusMTTranslator(source_lang="ja", target_lang="en")
        translator.load_model()

        assert translator.is_initialized() is True
        assert translator._model is not None
        assert translator._tokenizer is not None

        translator.cleanup()

    def test_translate_ja_to_en(self):
        """日本語→英語の翻訳"""
        translator = OpusMTTranslator(source_lang="ja", target_lang="en")
        translator.load_model()

        result = translator.translate("こんにちは", "ja", "en")

        assert result.text
        assert len(result.text) > 0
        assert result.original_text == "こんにちは"

        translator.cleanup()

    def test_translate_with_context_real(self):
        """文脈付き翻訳（実モデル）"""
        translator = OpusMTTranslator(source_lang="ja", target_lang="en")
        translator.load_model()

        context = ["昨日は友達と遊んだ。"]
        result = translator.translate("今日は疲れている。", "ja", "en", context=context)

        assert result.text
        assert result.original_text == "今日は疲れている。"

        translator.cleanup()
