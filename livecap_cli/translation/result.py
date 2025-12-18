"""
翻訳結果のデータクラス

翻訳処理の結果を格納し、既存のイベント型への変換機能を提供。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from livecap_cli.transcription_types import TranslationResultEventDict


@dataclass
class TranslationResult:
    """翻訳結果"""

    text: str  # 翻訳テキスト
    original_text: str  # 原文（イベント型との整合性）
    source_lang: str  # ソース言語
    target_lang: str  # ターゲット言語
    confidence: Optional[float] = None  # 信頼度（LLMの場合）
    source_id: str = "default"  # ソース識別子

    def to_event_dict(self) -> TranslationResultEventDict:
        """既存の TranslationResultEventDict に変換"""
        from livecap_cli.transcription_types import create_translation_result_event

        return create_translation_result_event(
            original_text=self.original_text,
            translated_text=self.text,
            source_id=self.source_id,
            source_language=self.source_lang,
            target_language=self.target_lang,
            confidence=self.confidence,
        )
