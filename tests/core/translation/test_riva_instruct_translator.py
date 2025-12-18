"""
RivaInstructTranslator のテスト
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# Check if dependencies are available
try:
    import torch
    import transformers

    HAS_RIVA_DEPS = True
except ImportError:
    HAS_RIVA_DEPS = False

# Skip all tests if dependencies not available
pytestmark = pytest.mark.skipif(
    not HAS_RIVA_DEPS,
    reason="Riva dependencies (torch, transformers) not installed",
)

from livecap_cli.translation.exceptions import (
    TranslationModelError,
    UnsupportedLanguagePairError,
)
from livecap_cli.translation.impl.riva_instruct import RivaInstructTranslator


class TestRivaInstructTranslatorBasic:
    """RivaInstructTranslator の基本テスト"""

    def test_initialization_default(self):
        """デフォルト初期化"""
        translator = RivaInstructTranslator()
        assert translator.device == "cuda"
        assert translator.max_new_tokens == 256
        assert translator.is_initialized() is False

    def test_initialization_custom_device(self):
        """カスタムデバイスで初期化"""
        translator = RivaInstructTranslator(device="cpu")
        assert translator.device == "cpu"

    def test_initialization_custom_max_tokens(self):
        """カスタム最大トークン数で初期化"""
        translator = RivaInstructTranslator(max_new_tokens=512)
        assert translator.max_new_tokens == 512

    def test_get_translator_name(self):
        """翻訳エンジン名"""
        translator = RivaInstructTranslator()
        assert translator.get_translator_name() == "riva_instruct"

    def test_get_supported_pairs(self):
        """サポート言語ペア"""
        translator = RivaInstructTranslator()
        pairs = translator.get_supported_pairs()
        # 10言語 × 9 = 90 ペア
        assert len(pairs) == 90
        assert ("ja", "en") in pairs
        assert ("en", "ja") in pairs
        assert ("zh", "en") in pairs
        # 同一言語は含まれない
        assert ("ja", "ja") not in pairs

    def test_default_context_sentences(self):
        """デフォルト文脈数"""
        translator = RivaInstructTranslator()
        assert translator._default_context_sentences == 2

    def test_custom_context_sentences(self):
        """カスタム文脈数"""
        translator = RivaInstructTranslator(default_context_sentences=5)
        assert translator._default_context_sentences == 5


class TestRivaInstructTranslatorNotLoaded:
    """モデル未ロード時のテスト"""

    def test_translate_without_load_raises(self):
        """モデル未ロードで翻訳するとエラー"""
        translator = RivaInstructTranslator()
        with pytest.raises(TranslationModelError, match="Model not loaded"):
            translator.translate("Hello", "en", "ja")


class TestRivaInstructTranslatorMocked:
    """モックを使用した RivaInstructTranslator テスト"""

    @pytest.fixture
    def mock_translator(self):
        """モック済みトランスレータ"""
        translator = RivaInstructTranslator(device="cuda")

        # モックモデルとトークナイザー
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # デバイス設定
        mock_model.device = "cuda:0"

        # トークナイザーの動作を設定
        mock_input = MagicMock()
        mock_input.shape = (1, 10)  # batch_size=1, seq_len=10
        mock_input.to.return_value = mock_input
        mock_tokenizer.apply_chat_template.return_value = mock_input
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.decode.return_value = "Hello world"

        # モデルの生成結果
        mock_output = MagicMock()
        mock_output.__getitem__ = lambda self, idx: MagicMock()
        mock_model.generate.return_value = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]

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
        assert exc_info.value.translator == "riva_instruct"

    def test_translate_with_context(self, mock_translator):
        """文脈付き翻訳"""
        context = ["前の文。"]
        result = mock_translator.translate("こんにちは", "ja", "en", context=context)

        # apply_chat_template が呼ばれた引数を確認
        call_args = mock_translator._tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]

        # system メッセージに文脈が含まれている
        assert "Previous context for reference" in messages[0]["content"]
        assert "前の文。" in messages[0]["content"]

    def test_translate_with_long_context(self, mock_translator):
        """長い文脈は制限される"""
        mock_translator._default_context_sentences = 2
        context = ["文1", "文2", "文3", "文4"]
        mock_translator.translate("テスト", "ja", "en", context=context)

        # apply_chat_template が呼ばれた引数を確認
        call_args = mock_translator._tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]

        # 最後の2文のみが含まれる
        assert "文3" in messages[0]["content"]
        assert "文4" in messages[0]["content"]
        # 最初の文は含まれない
        assert "文1" not in messages[0]["content"]


class TestRivaInstructTranslatorPrompt:
    """プロンプト構築のテスト"""

    @pytest.fixture
    def mock_translator(self):
        """モック済みトランスレータ"""
        translator = RivaInstructTranslator(device="cuda")

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.device = "cuda:0"

        mock_input = MagicMock()
        mock_input.shape = (1, 10)
        mock_input.to.return_value = mock_input
        mock_tokenizer.apply_chat_template.return_value = mock_input
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.decode.return_value = "Translation"

        mock_model.generate.return_value = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]

        translator._model = mock_model
        translator._tokenizer = mock_tokenizer
        translator._initialized = True

        return translator

    def test_prompt_contains_language_names(self, mock_translator):
        """プロンプトに言語名が含まれる"""
        mock_translator.translate("テスト", "ja", "en")

        call_args = mock_translator._tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]

        # system メッセージに言語名が含まれる
        assert "Japanese" in messages[0]["content"]
        assert "English" in messages[0]["content"]

    def test_prompt_user_message_format(self, mock_translator):
        """ユーザーメッセージの形式"""
        mock_translator.translate("こんにちは", "ja", "en")

        call_args = mock_translator._tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]

        # user メッセージの形式を確認
        assert messages[1]["role"] == "user"
        assert "こんにちは" in messages[1]["content"]
        assert "English translation" in messages[1]["content"]


class TestRivaInstructTranslatorCleanup:
    """cleanup のテスト"""

    def test_cleanup(self):
        """クリーンアップ"""
        translator = RivaInstructTranslator(device="cpu")
        translator._model = MagicMock()
        translator._tokenizer = MagicMock()
        translator._initialized = True

        translator.cleanup()

        assert translator._model is None
        assert translator._tokenizer is None
        assert translator._initialized is False

    def test_cleanup_when_not_initialized(self):
        """未初期化でもクリーンアップ可能"""
        translator = RivaInstructTranslator()
        # エラーなく実行できる
        translator.cleanup()
        assert translator._initialized is False


class TestRivaInstructTranslatorAsync:
    """非同期翻訳のテスト"""

    def test_translate_async(self):
        """非同期翻訳テスト"""
        import asyncio

        translator = RivaInstructTranslator(device="cuda")

        # モック設定
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.device = "cuda:0"

        mock_input = MagicMock()
        mock_input.shape = (1, 10)
        mock_input.to.return_value = mock_input
        mock_tokenizer.apply_chat_template.return_value = mock_input
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.decode.return_value = "Hello"

        mock_model.generate.return_value = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

        translator._model = mock_model
        translator._tokenizer = mock_tokenizer
        translator._initialized = True

        async def run_test():
            return await translator.translate_async("こんにちは", "ja", "en")

        result = asyncio.run(run_test())
        assert result.text == "Hello"
        assert result.original_text == "こんにちは"


class TestRivaInstructTranslatorVRAMCheck:
    """VRAM チェックのテスト"""

    @patch("livecap_cli.translation.impl.riva_instruct.transformers")
    @patch("livecap_cli.utils.get_available_vram")
    def test_vram_warning_when_insufficient(self, mock_vram, mock_transformers, caplog):
        """VRAM 不足時に警告"""
        import logging

        mock_vram.return_value = 4000  # 4GB (insufficient)

        # transformers のモック設定
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model

        translator = RivaInstructTranslator(device="cuda")

        # load_model() を呼んで警告が出ることを確認
        with caplog.at_level(logging.WARNING):
            translator.load_model()

        # 警告メッセージを確認
        assert any("Riva-4B requires" in record.message for record in caplog.records)
        assert any("4000MB" in record.message for record in caplog.records)

    @patch("livecap_cli.translation.impl.riva_instruct.transformers")
    @patch("livecap_cli.utils.get_available_vram")
    def test_vram_check_skipped_when_none(self, mock_vram, mock_transformers, caplog):
        """VRAM が None の場合はチェックスキップ"""
        import logging

        mock_vram.return_value = None

        # transformers のモック設定
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model

        translator = RivaInstructTranslator(device="cuda")

        # load_model() を呼んで警告が出ないことを確認
        with caplog.at_level(logging.DEBUG):
            translator.load_model()

        # VRAM 不足警告は出ない（スキップのデバッグログは出る）
        assert not any(
            "Riva-4B requires" in record.message for record in caplog.records
        )

    @patch("livecap_cli.translation.impl.riva_instruct.transformers")
    @patch("livecap_cli.utils.get_available_vram")
    def test_no_vram_warning_when_sufficient(self, mock_vram, mock_transformers, caplog):
        """VRAM 十分な場合は警告なし"""
        import logging

        mock_vram.return_value = 10000  # 10GB (sufficient)

        # transformers のモック設定
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model

        translator = RivaInstructTranslator(device="cuda")

        # load_model() を呼んで警告が出ないことを確認
        with caplog.at_level(logging.WARNING):
            translator.load_model()

        # VRAM 不足警告は出ない
        assert not any(
            "Riva-4B requires" in record.message for record in caplog.records
        )


@pytest.mark.gpu
@pytest.mark.slow
class TestRivaInstructTranslatorIntegration:
    """統合テスト（実モデルロード、要 translation-riva extra + GPU）"""

    def test_load_model(self):
        """モデルのロード"""
        translator = RivaInstructTranslator(device="cuda")
        translator.load_model()

        assert translator.is_initialized() is True
        assert translator._model is not None
        assert translator._tokenizer is not None

        translator.cleanup()

    def test_translate_ja_to_en(self):
        """日本語→英語の翻訳"""
        translator = RivaInstructTranslator(device="cuda")
        translator.load_model()

        result = translator.translate("こんにちは", "ja", "en")

        assert result.text
        assert len(result.text) > 0
        assert result.original_text == "こんにちは"

        translator.cleanup()

    def test_translate_with_context_real(self):
        """文脈付き翻訳（実モデル）"""
        translator = RivaInstructTranslator(device="cuda")
        translator.load_model()

        context = ["昨日は友達と遊んだ。"]
        result = translator.translate("今日は疲れている。", "ja", "en", context=context)

        assert result.text
        assert result.original_text == "今日は疲れている。"

        translator.cleanup()
