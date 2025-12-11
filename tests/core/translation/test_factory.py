"""
TranslatorFactory のテスト
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from livecap_core.translation.factory import TranslatorFactory
from livecap_core.translation.impl.google import GoogleTranslator

# Check if OPUS-MT dependencies are available
try:
    import ctranslate2
    import transformers

    HAS_OPUS_MT_DEPS = True
except ImportError:
    HAS_OPUS_MT_DEPS = False


class TestTranslatorFactory:
    """TranslatorFactory のテスト"""

    def test_create_google_translator(self):
        """Google Translator の作成"""
        translator = TranslatorFactory.create_translator("google")
        assert isinstance(translator, GoogleTranslator)
        assert translator.is_initialized() is True
        assert translator.get_translator_name() == "google"

    def test_create_google_with_custom_context(self):
        """カスタム文脈数で Google Translator を作成"""
        translator = TranslatorFactory.create_translator(
            "google",
            default_context_sentences=5,
        )
        assert translator._default_context_sentences == 5

    def test_create_unknown_translator_raises(self):
        """不明な翻訳エンジンでエラー"""
        with pytest.raises(ValueError, match="Unknown translator type"):
            TranslatorFactory.create_translator("unknown_engine")

    def test_list_available_translators(self):
        """利用可能な翻訳エンジンのリスト"""
        translators = TranslatorFactory.list_available_translators()
        assert "google" in translators
        assert "opus_mt" in translators
        assert "riva_instruct" in translators


class TestTranslatorFactoryIntegration:
    """TranslatorFactory の統合テスト"""

    def test_google_translator_translate(self):
        """Factory で作成した Translator で翻訳"""
        with patch(
            "livecap_core.translation.impl.google.DeepGoogleTranslator"
        ) as mock_gt:
            mock_gt.return_value.translate.return_value = "Hello"

            translator = TranslatorFactory.create_translator("google")
            result = translator.translate("こんにちは", "ja", "en")

            assert result.text == "Hello"
            assert result.original_text == "こんにちは"

    def test_default_params_from_metadata(self):
        """メタデータからのデフォルトパラメータ"""
        translator = TranslatorFactory.create_translator("google")
        # Google の default_context_sentences は 2
        assert translator._default_context_sentences == 2


class TestTranslatorFactoryOpusMT:
    """OPUS-MT Factory テスト"""

    @pytest.mark.skipif(not HAS_OPUS_MT_DEPS, reason="OPUS-MT deps not installed")
    def test_create_opus_mt_translator(self):
        """OPUS-MT Translator の作成（依存関係インストール済み）"""
        from livecap_core.translation.impl.opus_mt import OpusMTTranslator

        translator = TranslatorFactory.create_translator(
            "opus_mt",
            source_lang="ja",
            target_lang="en",
        )
        assert isinstance(translator, OpusMTTranslator)
        assert translator.is_initialized() is False  # load_model() 必要
        assert translator.get_translator_name() == "opus_mt"
        assert translator.source_lang == "ja"
        assert translator.target_lang == "en"

    @pytest.mark.skipif(not HAS_OPUS_MT_DEPS, reason="OPUS-MT deps not installed")
    def test_create_opus_mt_with_default_params(self):
        """メタデータからのデフォルトパラメータ"""
        translator = TranslatorFactory.create_translator("opus_mt")
        # metadata の default_params: device="cpu", compute_type="int8"
        assert translator.device == "cpu"
        assert translator.compute_type == "int8"

    @pytest.mark.skipif(HAS_OPUS_MT_DEPS, reason="OPUS-MT deps installed")
    def test_opus_mt_not_implemented_without_deps(self):
        """依存関係未インストールで NotImplementedError"""
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            TranslatorFactory.create_translator("opus_mt")


class TestTranslatorFactoryFuture:
    """未実装エンジンのテスト"""

    def test_riva_instruct_not_implemented_yet(self):
        """Riva Instruct は Phase 4 で実装予定"""
        # 明確なエラーメッセージで NotImplementedError が発生する
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            TranslatorFactory.create_translator("riva_instruct")

    def test_not_implemented_error_message_shows_available(self):
        """未実装エンジンのエラーメッセージに利用可能なエンジンが表示される"""
        with pytest.raises(NotImplementedError) as exc_info:
            TranslatorFactory.create_translator("riva_instruct")
        assert "google" in str(exc_info.value)
        assert "Currently available" in str(exc_info.value)
