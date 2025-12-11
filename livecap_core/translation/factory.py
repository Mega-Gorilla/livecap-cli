"""
翻訳エンジンのファクトリー

TranslatorFactory は翻訳エンジンを作成するためのファクトリークラス。
EngineFactory と同様のパターンで実装。
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .metadata import TranslatorMetadata

if TYPE_CHECKING:
    from .base import BaseTranslator


class TranslatorFactory:
    """翻訳エンジンを作成するファクトリークラス"""

    @classmethod
    def create_translator(
        cls,
        translator_type: str,
        **translator_options,
    ) -> BaseTranslator:
        """
        指定されたタイプの翻訳エンジンを作成

        Args:
            translator_type: 翻訳エンジンタイプ
                利用可能: google, opus_mt, riva_instruct
            **translator_options: エンジン固有のパラメータ

        Returns:
            BaseTranslator のインスタンス

        Raises:
            ValueError: 不明な翻訳エンジンタイプが指定された場合

        Examples:
            # Google Translate
            >>> translator = TranslatorFactory.create_translator("google")

            # OPUS-MT (CPU)
            >>> translator = TranslatorFactory.create_translator(
            ...     "opus_mt",
            ...     source_lang="ja",
            ...     target_lang="en",
            ...     device="cpu"
            ... )

            # Riva 4B Instruct (GPU)
            >>> translator = TranslatorFactory.create_translator(
            ...     "riva_instruct",
            ...     device="cuda"
            ... )
        """
        metadata = TranslatorMetadata.get(translator_type)
        if metadata is None:
            available = list(TranslatorMetadata.get_all().keys())
            raise ValueError(
                f"Unknown translator type: {translator_type}. " f"Available: {available}"
            )

        # default_params と options をマージ
        params = {**metadata.default_params, **translator_options}

        # default_context_sentences をメタデータから注入
        if "default_context_sentences" not in params:
            params["default_context_sentences"] = metadata.default_context_sentences

        # 動的インポート
        try:
            module = importlib.import_module(metadata.module, package="livecap_core.translation")
            translator_class = getattr(module, metadata.class_name)
        except ModuleNotFoundError as e:
            raise NotImplementedError(
                f"Translator '{translator_type}' is registered but not yet implemented. "
                f"This translator will be available in a future release. "
                f"Currently available: {cls._get_implemented_translators()}"
            ) from e

        return translator_class(**params)

    @classmethod
    def _get_implemented_translators(cls) -> list[str]:
        """
        実装済みの翻訳エンジンIDのリストを取得

        Returns:
            実装済み翻訳エンジンIDのリスト
        """
        implemented = []
        for translator_id in TranslatorMetadata.list_translator_ids():
            metadata = TranslatorMetadata.get(translator_id)
            if metadata is None:
                continue
            try:
                importlib.import_module(metadata.module, package="livecap_core.translation")
                implemented.append(translator_id)
            except ModuleNotFoundError:
                pass
        return implemented

    @classmethod
    def list_available_translators(cls) -> list[str]:
        """
        利用可能な翻訳エンジンのリストを取得

        Returns:
            翻訳エンジンIDのリスト
        """
        return TranslatorMetadata.list_translator_ids()
