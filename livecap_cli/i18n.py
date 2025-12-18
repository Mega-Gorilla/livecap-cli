"""軽量な翻訳サービスのためのユーティリティ"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranslatorDetails:
    registered: bool
    name: Optional[str] = None
    extras: Tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class I18nDiagnostics:
    translator: TranslatorDetails
    fallback_count: int
    fallback_keys_sample: Tuple[str, ...] = ()


class I18nManager:
    """コア層で使用する簡易的な翻訳マネージャ"""

    def __init__(self) -> None:
        self._translator: Optional[Callable[..., str]] = None
        self._fallbacks: Dict[str, str] = {}
        self._translator_details = TranslatorDetails(registered=False)

    @contextmanager
    def preserve_state(self):
        """テスト用途などで現在の登録状態を一時退避する"""
        translator = self._translator
        fallbacks = dict(self._fallbacks)
        details = self._translator_details
        try:
            yield self
        finally:
            self._translator = translator
            self._fallbacks = fallbacks
            self._translator_details = details

    def register_translator(
        self,
        translator: Callable[..., str],
        *,
        name: Optional[str] = None,
        extras: Optional[Iterable[str]] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        """翻訳関数を登録"""
        self._translator = translator

        if name is None:
            qualname = getattr(translator, "__qualname__", None)
            module = getattr(translator, "__module__", None)
            if module and qualname:
                name = f"{module}.{qualname}"
            else:
                name = qualname or getattr(translator, "__name__", None)

        extras_tuple = tuple(str(item) for item in extras or ())
        metadata_dict = {str(key): str(value) for key, value in (metadata or {}).items()}

        self._translator_details = TranslatorDetails(
            registered=True,
            name=name,
            extras=extras_tuple,
            metadata=metadata_dict,
        )

    def clear_translator(self) -> None:
        """登録済み翻訳関数を解除"""
        self._translator = None
        self._translator_details = TranslatorDetails(registered=False)

    def register_fallbacks(self, mapping: Mapping[str, str], *, namespace: str | None = None) -> None:
        """フォールバック用メッセージを登録"""
        if namespace:
            for key, value in mapping.items():
                qualified = f"{namespace}.{key}" if key else namespace
                self._fallbacks[qualified] = value
        else:
            self._fallbacks.update(mapping)

    def clear_fallbacks(self, *, prefix: str | None = None) -> None:
        """登録済みフォールバックを削除"""
        if prefix is None:
            self._fallbacks.clear()
            return
        keys = [key for key in self._fallbacks if key.startswith(prefix)]
        for key in keys:
            self._fallbacks.pop(key, None)

    def get_fallback(self, key: str) -> Optional[str]:
        """フォールバックメッセージを取得"""
        return self._fallbacks.get(key)

    def fallback_keys(self) -> Tuple[str, ...]:
        """登録済みフォールバックキーの一覧"""
        return tuple(self._fallbacks.keys())

    def translate(self, key: str, *, default: Optional[str] = None, **kwargs) -> str:
        """
        翻訳文字列を取得

        translatorが登録されていない、もしくは翻訳に失敗した場合は
        フォールバック値を使用する。
        """
        if self._translator:
            try:
                return self._translator(key, **kwargs)
            except Exception as exc:  # pragma: no cover - ログのみ
                logger.debug("Translator failed for key '%s': %s", key, exc)

        template = self._fallbacks.get(key, default or key)

        if kwargs:
            try:
                return template.format(**kwargs)
            except (KeyError, IndexError, ValueError):
                return template

        return template

    def diagnostics(self, *, sample_size: int = 5) -> I18nDiagnostics:
        """登録状態を診断用に返す"""
        sample = tuple(list(self._fallbacks.keys())[:sample_size])
        return I18nDiagnostics(
            translator=self._translator_details,
            fallback_count=len(self._fallbacks),
            fallback_keys_sample=sample,
        )


i18n = I18nManager()

translate = i18n.translate
register_translator = i18n.register_translator
register_fallbacks = i18n.register_fallbacks
diagnose = i18n.diagnostics

__all__ = [
    "I18nDiagnostics",
    "I18nManager",
    "TranslatorDetails",
    "translate",
    "register_translator",
    "register_fallbacks",
    "diagnose",
]
