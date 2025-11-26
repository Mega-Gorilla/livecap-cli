"""Text normalization for benchmark evaluation.

Provides language-specific text preprocessing to ensure fair WER/CER comparison.
This module is independent from tests/utils/text_normalization.py to keep
benchmarks self-contained.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["normalize_en", "normalize_ja", "normalize_text"]


def _collapse_spaces(text: str) -> str:
    """Collapse multiple whitespace characters into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_en(text: str, *, keep_apostrophes: bool = False) -> str:
    """Normalize English transcripts for WER calculation.

    Processing steps:
    - Convert to lowercase
    - Remove punctuation (optionally keep apostrophes as spaces)
    - Collapse whitespace

    Args:
        text: Input transcript
        keep_apostrophes: If True, replace apostrophes with spaces instead of removing

    Returns:
        Normalized text
    """
    cleaned = text.lower().strip()
    if keep_apostrophes:
        cleaned = re.sub(r"[^a-z0-9\s']", " ", cleaned)
        cleaned = cleaned.replace("'", " ")
    else:
        cleaned = cleaned.replace("'", "")
        cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    return _collapse_spaces(cleaned)


def normalize_ja(
    text: str,
    *,
    strip_punctuation: bool = True,
    normalize_width: bool = False,
) -> str:
    """Normalize Japanese transcripts for CER calculation.

    Processing steps:
    - Trim surrounding whitespace
    - Optionally normalize full/half-width characters (NFKC)
    - Optionally strip common punctuation (。、．，)
    - Remove all whitespace (for character-level comparison)

    Args:
        text: Input transcript
        strip_punctuation: If True, remove common Japanese punctuation
        normalize_width: If True, apply NFKC normalization

    Returns:
        Normalized text
    """
    cleaned = text.strip()
    if normalize_width:
        cleaned = unicodedata.normalize("NFKC", cleaned)
    if strip_punctuation:
        # Remove common Japanese punctuation (full-width and half-width)
        cleaned = (
            cleaned.replace("。", "")
            .replace("、", "")
            .replace("．", "")
            .replace("，", "")
            .replace("！", "")
            .replace("？", "")
            .replace("・", "")
            .replace("「", "")
            .replace("」", "")
            .replace("『", "")
            .replace("』", "")
            .replace("（", "")
            .replace("）", "")
        )
    # Remove any spacing (half- or full-width) for character-level comparison
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned


def _normalize_generic(text: str) -> str:
    """Generic normalization for languages without specific rules.

    Applies basic normalization:
    - Lowercase
    - Remove Unicode punctuation (category P)
    - Collapse whitespace

    Args:
        text: Input transcript

    Returns:
        Normalized text
    """
    cleaned = text.lower().strip()
    # Remove all Unicode punctuation (category P)
    cleaned = "".join(c for c in cleaned if not unicodedata.category(c).startswith("P"))
    return _collapse_spaces(cleaned)


def normalize_text(text: str, *, lang: str) -> str:
    """Dispatch to language-specific normalization.

    For languages without specific normalization rules, applies generic
    normalization (lowercase, remove punctuation, collapse whitespace).

    Args:
        text: Input transcript
        lang: Language code ('en', 'ja', etc.)

    Returns:
        Normalized text
    """
    if lang == "en":
        return normalize_en(text)
    if lang == "ja":
        return normalize_ja(text)
    # Fallback: generic normalization for other languages (de, fr, es, etc.)
    return _normalize_generic(text)
