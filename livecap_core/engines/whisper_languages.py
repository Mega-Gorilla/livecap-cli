"""Whisper supported languages (100 languages from OpenAI Whisper tokenizer.py)

This module provides the complete list of languages supported by OpenAI Whisper.
The language codes are ISO 639-1 two-letter codes.

Reference: https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
"""

from __future__ import annotations

# ISO 639-1 codes supported by Whisper (100 languages)
# Ordered as defined in OpenAI Whisper's tokenizer.py
WHISPER_LANGUAGES = (
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl",
    "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk",
    "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr",
    "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn",
    "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne",
    "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn",
    "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi",
    "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my",
    "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
    "yue",
)

# Frozen set for O(1) lookup during language validation
WHISPER_LANGUAGES_SET = frozenset(WHISPER_LANGUAGES)
