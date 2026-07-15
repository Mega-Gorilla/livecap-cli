"""Qwen3-ASR の言語データ正本 (Issue #230)。

ISO 639-1/3 コード → qwen-asr API が期待する言語名の map。
`EngineMetadata` (supported_languages / quality_tier) と
`Qwen3ASREngine` (言語コード変換・検証) の**両方がこの module から派生**する。
`whisper_languages.py` と同じ data-only pattern — heavy import 禁止
(metadata.py が module level で import するため軽量性を維持すること)。

key の順序は公開 API (`EngineInfo.supported_languages`) の順序として
そのまま観測されるため、変更しないこと。
"""

from __future__ import annotations

from typing import Dict

QWEN_ASR_LANGUAGE_NAMES: Dict[str, str] = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese",
    "ar": "Arabic", "de": "German", "fr": "French",
    "es": "Spanish", "pt": "Portuguese", "id": "Indonesian",
    "it": "Italian", "ko": "Korean", "ru": "Russian",
    "th": "Thai", "vi": "Vietnamese", "ja": "Japanese",
    "tr": "Turkish", "hi": "Hindi", "ms": "Malay",
    "nl": "Dutch", "sv": "Swedish", "da": "Danish",
    "fi": "Finnish", "pl": "Polish", "cs": "Czech",
    "fil": "Filipino", "fa": "Persian", "el": "Greek",
    "hu": "Hungarian", "mk": "Macedonian", "ro": "Romanian",
}

__all__ = ["QWEN_ASR_LANGUAGE_NAMES"]
