"""
翻訳プラグインシステム

複数の翻訳エンジン（Google Translate, OPUS-MT, Riva-4B-Instruct）をサポートする
プラグイン可能な翻訳システムを提供する。

Usage:
    from livecap_cli.translation import TranslatorFactory, TranslationResult

    # Google Translate
    translator = TranslatorFactory.create_translator("google")
    result = translator.translate("こんにちは", "ja", "en")
    print(result.text)  # "Hello"

    # OPUS-MT (ローカル)
    translator = TranslatorFactory.create_translator(
        "opus_mt",
        source_lang="ja",
        target_lang="en",
    )
    translator.load_model()
    result = translator.translate("こんにちは", "ja", "en")
"""

from __future__ import annotations

from .base import BaseTranslator
from .exceptions import (
    TranslationError,
    TranslationModelError,
    TranslationNetworkError,
    UnsupportedLanguagePairError,
)
from .factory import TranslatorFactory
from .lang_codes import (
    get_language_name,
    get_opus_mt_model_name,
    normalize_for_google,
    normalize_for_opus_mt,
    to_iso639_1,
)
from .metadata import TranslatorInfo, TranslatorMetadata
from .result import TranslationResult
from .retry import with_retry

__all__ = [
    # Core classes
    "BaseTranslator",
    "TranslationResult",
    "TranslatorFactory",
    "TranslatorMetadata",
    "TranslatorInfo",
    # Exceptions
    "TranslationError",
    "TranslationNetworkError",
    "TranslationModelError",
    "UnsupportedLanguagePairError",
    # Language code utilities
    "to_iso639_1",
    "normalize_for_google",
    "normalize_for_opus_mt",
    "get_language_name",
    "get_opus_mt_model_name",
    # Retry decorator
    "with_retry",
]
