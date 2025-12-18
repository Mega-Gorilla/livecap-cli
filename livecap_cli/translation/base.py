"""
翻訳エンジンの抽象基底クラス

全ての翻訳エンジン実装はこの基底クラスを継承する。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from .result import TranslationResult


class BaseTranslator(ABC):
    """翻訳エンジンの抽象基底クラス"""

    def __init__(self, default_context_sentences: int = 2, **kwargs):
        """
        翻訳エンジンを初期化

        Args:
            default_context_sentences: 文脈として使用するデフォルトの文数
            **kwargs: サブクラス固有のパラメータ
        """
        self._initialized = False
        self._default_context_sentences = default_context_sentences

    @abstractmethod
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Optional[List[str]] = None,
    ) -> TranslationResult:
        """
        テキストを翻訳

        Args:
            text: 翻訳対象テキスト
            source_lang: ソース言語コード (BCP-47)
            target_lang: ターゲット言語コード (BCP-47)
            context: 過去の文脈（直近N文）

        Returns:
            TranslationResult
        """
        ...

    async def translate_async(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Optional[List[str]] = None,
    ) -> TranslationResult:
        """
        非同期翻訳（デフォルト実装）

        同期メソッドを asyncio.to_thread でラップ。
        サブクラスで真の非同期実装が必要な場合はオーバーライド可能。

        Args:
            text: 翻訳対象テキスト
            source_lang: ソース言語コード (BCP-47)
            target_lang: ターゲット言語コード (BCP-47)
            context: 過去の文脈（直近N文）

        Returns:
            TranslationResult
        """
        import asyncio

        return await asyncio.to_thread(
            self.translate, text, source_lang, target_lang, context
        )

    @abstractmethod
    def get_supported_pairs(self) -> List[Tuple[str, str]]:
        """
        サポートする言語ペアを取得

        Returns:
            言語ペアのリスト。空リストは全ペア対応を意味する。
        """
        ...

    @abstractmethod
    def get_translator_name(self) -> str:
        """
        翻訳エンジン名を取得

        Returns:
            翻訳エンジンの識別子（例: "google", "opus_mt"）
        """
        ...

    @property
    def default_context_sentences(self) -> int:
        """
        文脈として使用するデフォルトの文数

        StreamTranscriber が翻訳時に使用する文脈の文数を取得するために使用。

        Returns:
            デフォルトの文脈文数
        """
        return self._default_context_sentences

    def is_initialized(self) -> bool:
        """
        初期化済みかどうか

        Returns:
            初期化済みなら True
        """
        return self._initialized

    def load_model(self) -> None:
        """
        モデルをロード（ローカルモデルの場合）

        クラウド API の場合はオーバーライド不要。
        ローカルモデルを使用するエンジンはこのメソッドをオーバーライドする。
        """
        pass  # クラウド API はオーバーライド不要

    def cleanup(self) -> None:
        """
        リソースのクリーンアップ

        GPU メモリの解放などを行う。
        サブクラスでオーバーライドして具体的な処理を実装する。
        """
        pass
