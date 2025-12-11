"""
翻訳エンジン実装

各翻訳エンジンの実装を格納するサブパッケージ。

- google.py: Google Translate (Phase 2)
- opus_mt.py: OPUS-MT via CTranslate2 (Phase 3)
- riva_instruct.py: Riva-Translate-4B-Instruct (Phase 4)
"""

from __future__ import annotations

from .google import GoogleTranslator

__all__ = [
    "GoogleTranslator",
]

# Optional: OPUS-MT (requires translation-local extra)
try:
    from .opus_mt import OpusMTTranslator

    __all__.append("OpusMTTranslator")
except ImportError:
    pass  # ctranslate2/transformers not installed
