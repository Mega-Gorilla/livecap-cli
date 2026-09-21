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
            ImportError: extra が提供する module (``TranslatorInfo.required_modules``) が未導入の場合。
                メッセージに必要な extra 名を含む
            NotImplementedError: metadata に登録されているが実装 module 自体が無い場合
            ModuleNotFoundError: 上記以外 (実装内部の import の欠落など) — 原因を隠さずそのまま送出

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

        # 動的インポート。**translator ごとに遅延**させる (Issue #454): ``impl/__init__`` は何も
        # import しないので、Google だけを使う場合に torch / transformers / ctranslate2 は読み込まれない
        module_name = "livecap_cli.translation" + metadata.module
        try:
            module = importlib.import_module(metadata.module, package="livecap_cli.translation")
            translator_class = getattr(module, metadata.class_name)
        except ModuleNotFoundError as e:
            missing = e.name or ""
            if missing == module_name or module_name.startswith(missing + "."):
                # 実装 module 自体が無い。案内の一覧は metadata から作る (他の translator を import しない)
                raise NotImplementedError(
                    f"Translator '{translator_type}' is registered but not yet implemented. "
                    f"Registered translators: {TranslatorMetadata.list_translator_ids()}"
                ) from e
            if metadata.extra and missing in metadata.required_modules:
                # 宣言済みの optional dependency (top-level module そのもの) が無い → extra の案内。
                # `transformers.some_internal` のような配下の欠落は実装 / 配布物の不整合なので包まない
                raise ImportError(
                    f"Translator '{translator_type}' requires the '{metadata.extra}' extra "
                    f"(missing module: {missing}). Run: pip install livecap-cli[{metadata.extra}]"
                ) from e
            # それ以外 (実装内部の import typo / 欠落) は原因を隠さない
            raise

        return translator_class(**params)

    @classmethod
    def list_available_translators(cls) -> list[str]:
        """
        利用可能な翻訳エンジンのリストを取得

        Returns:
            翻訳エンジンIDのリスト
        """
        return TranslatorMetadata.list_translator_ids()
