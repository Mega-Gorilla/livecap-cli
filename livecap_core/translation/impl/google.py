"""
Google Translate 実装

deep-translator ライブラリを使用した Google Translate API のラッパー。
無料枠で動作し、ほぼ全言語ペアに対応。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from deep_translator import GoogleTranslator as DeepGoogleTranslator
from deep_translator.exceptions import (
    RequestError,
    TooManyRequests,
    TranslationNotFound,
)

from ..base import BaseTranslator
from ..exceptions import (
    TranslationError,
    TranslationNetworkError,
    UnsupportedLanguagePairError,
)
from ..lang_codes import normalize_for_google, to_iso639_1
from ..result import TranslationResult
from ..retry import with_retry


class GoogleTranslator(BaseTranslator):
    """
    Google Translate (via deep-translator)

    無料の Google Translate API を使用した翻訳エンジン。
    ほぼ全言語ペアに対応し、初期化不要で即座に使用可能。

    Examples:
        >>> translator = GoogleTranslator()
        >>> result = translator.translate("こんにちは", "ja", "en")
        >>> print(result.text)
        "Hello"
    """

    def __init__(self, **kwargs):
        """
        GoogleTranslator を初期化

        Args:
            **kwargs: BaseTranslator に渡すパラメータ
                - default_context_sentences: 文脈として使用するデフォルトの文数
        """
        super().__init__(**kwargs)
        self._initialized = True  # クラウド API なので初期化不要

    @with_retry(max_retries=3, base_delay=1.0)
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
            context: 過去の文脈（直近N文）。翻訳精度向上のため使用。

        Returns:
            TranslationResult

        Raises:
            UnsupportedLanguagePairError: 同一言語が指定された場合
            TranslationNetworkError: API リクエスト失敗、レート制限
            TranslationError: その他の翻訳エラー
        """
        # 入力バリデーション: 空文字列
        if not text or not text.strip():
            return TranslationResult(
                text="",
                original_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
            )

        # 入力バリデーション: 同一言語
        if to_iso639_1(source_lang) == to_iso639_1(target_lang):
            raise UnsupportedLanguagePairError(
                source_lang, target_lang, self.get_translator_name()
            )

        # 文脈をパラグラフとして連結
        if context:
            ctx = context[-self._default_context_sentences :]
            full_text = "\n".join(ctx) + "\n" + text
        else:
            full_text = text

        try:
            translator = DeepGoogleTranslator(
                source=normalize_for_google(source_lang),
                target=normalize_for_google(target_lang),
            )
            result = translator.translate(full_text)
        except TooManyRequests as e:
            raise TranslationNetworkError(f"Rate limited: {e}") from e
        except RequestError as e:
            raise TranslationNetworkError(f"API request failed: {e}") from e
        except TranslationNotFound as e:
            raise TranslationError(f"Translation not found: {e}") from e
        except Exception as e:
            raise TranslationError(f"Unexpected error: {e}") from e

        # 文脈を含めた場合、最後の文を抽出
        if context:
            result = self._extract_last_sentence(result)

        return TranslationResult(
            text=result,
            original_text=text,
            source_lang=source_lang,
            target_lang=target_lang,
        )

    def _extract_last_sentence(self, text: str) -> str:
        """
        翻訳結果から最後の文を抽出

        文脈を連結して翻訳した場合、最後の文のみを抽出する。

        Args:
            text: 翻訳結果テキスト

        Returns:
            最後の文
        """
        lines = text.strip().split("\n")
        return lines[-1] if lines else text

    def get_translator_name(self) -> str:
        """翻訳エンジン名を取得"""
        return "google"

    def get_supported_pairs(self) -> List[Tuple[str, str]]:
        """
        サポートする言語ペアを取得

        Returns:
            空リスト（全言語ペア対応を意味する）
        """
        return []
