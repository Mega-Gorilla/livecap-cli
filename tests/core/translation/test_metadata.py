"""
TranslatorMetadata のテスト
"""

from __future__ import annotations

import pytest

from livecap_cli.translation.metadata import TranslatorInfo, TranslatorMetadata


class TestTranslatorMetadata:
    """TranslatorMetadata のテスト"""

    def test_get_existing_translator(self):
        """存在する翻訳エンジンを取得"""
        info = TranslatorMetadata.get("google")
        assert info is not None
        assert info.translator_id == "google"
        assert info.display_name == "Google Translate"
        assert info.requires_model_load is False

    def test_get_nonexistent_translator(self):
        """存在しない翻訳エンジンは None"""
        info = TranslatorMetadata.get("nonexistent")
        assert info is None

    def test_get_all_translators(self):
        """全ての翻訳エンジンを取得"""
        all_translators = TranslatorMetadata.get_all()
        assert "google" in all_translators
        assert "opus_mt" in all_translators
        assert "riva_instruct" in all_translators

    def test_get_translators_for_pair_ja_en(self):
        """ja→en をサポートする翻訳エンジン"""
        translators = TranslatorMetadata.get_translators_for_pair("ja", "en")
        assert "google" in translators
        assert "opus_mt" in translators
        assert "riva_instruct" in translators

    def test_get_translators_for_pair_unsupported(self):
        """サポートされていない言語ペア"""
        # Google は全ペア対応なので常に含まれる
        translators = TranslatorMetadata.get_translators_for_pair("xyz", "abc")
        assert "google" in translators
        assert "opus_mt" not in translators

    def test_list_translator_ids(self):
        """翻訳エンジン ID のリスト"""
        ids = TranslatorMetadata.list_translator_ids()
        assert isinstance(ids, list)
        assert "google" in ids
        assert "opus_mt" in ids
        assert "riva_instruct" in ids


class TestTranslatorInfo:
    """TranslatorInfo のテスト"""

    def test_google_info(self):
        """Google Translate のメタデータ"""
        info = TranslatorMetadata.get("google")
        assert info.requires_model_load is False
        assert info.requires_gpu is False
        assert info.default_context_sentences == 2

    def test_opus_mt_info(self):
        """OPUS-MT のメタデータ"""
        info = TranslatorMetadata.get("opus_mt")
        assert info.requires_model_load is True
        assert info.requires_gpu is False
        assert info.default_params.get("device") == "cpu"
        assert info.default_params.get("compute_type") == "int8"

    def test_riva_instruct_info(self):
        """Riva Instruct のメタデータ"""
        info = TranslatorMetadata.get("riva_instruct")
        assert info.requires_model_load is True
        assert info.requires_gpu is True
        assert info.default_context_sentences == 5  # LLM なので多め
        assert info.default_params.get("device") == "cuda"
