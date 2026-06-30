"""Japanese kana-level normalization for calibration alignment metrics.

This helper is **dev / benchmark-only**. ``pykakasi`` (``GPL-3.0-or-later``) is
a development dependency declared in ``pyproject.toml`` under
``[project.optional-dependencies] dev``. The livecap-cli production runtime
must never import this module — that invariant is statically enforced by
``tests/test_production_no_pykakasi.py`` (Issue #338 PR-γ).

Purpose
-------
The PR-β active calibration harness aligns ASR output against a reference
transcript by string coverage. For Japanese, text-level coverage conflates
two unrelated effects:

* **Acoustic confidence** — does the ASR's phoneme sequence match the
  reference? This is what threshold calibration actually wants to evaluate.
* **Lexical surface form** — does the ASR's kanji/katakana/digit choice
  match the reference's surface form? This is independent of acoustic
  confidence (e.g. "1人で" vs "一人で" share the same reading "ひとりで").

This module normalises both sides of the comparison to a kana-only form so
the kana-level alignment score isolates acoustic confidence from surface
form noise.

Normalization pipeline (``normalize_for_alignment``)
----------------------------------------------------
1. NFKC: 全角 → 半角 ASCII, 全角 punctuation → half-width
2. Digit masking: ASCII digit runs AND 漢数字 runs → ``"#"`` (single token)
3. pykakasi: kanji → hiragana reading, katakana → hiragana, ASCII pass-through
4. Strip punctuation + whitespace

Step 2 (digit mask) must run **before** step 3 (pykakasi) so that
``"一人"`` is not first collapsed to ``"ひとり"`` by pykakasi — see PR-γ
plan D2.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache


@lru_cache(maxsize=1)
def _kakasi():
    """Return a memoised pykakasi.kakasi() converter instance.

    Lazy import keeps pykakasi out of module top-level so that mere import of
    this file does not require the dev dependency. Raises ``ModuleNotFoundError``
    on first call if pykakasi is missing — pip install ``livecap-cli[dev]``.
    """
    import pykakasi  # noqa: PLC0415 — intentional lazy import (dev-only dep)

    return pykakasi.kakasi()


# ASCII digit runs (e.g. "1000")
_ASCII_DIGIT_RE = re.compile(r"\d+")

# 漢数字 runs: 〇/零/一-九/十/百/千/万/億/兆 (no 京 — virtually unused in calibration corpora)
_KANJI_NUM_RUN_RE = re.compile(r"[〇零一二三四五六七八九十百千万億兆]+")

# Punctuation + whitespace to strip after normalization (alignment 比較用 noise).
# Note: step 1 (NFKC) folds 全角 punctuation (．，（）！？) into half-width ASCII,
# so the half-width forms must be present here. Full-width forms are kept too
# as a safety net in case NFKC misses anything (e.g. 「」『』 are not folded).
_STRIP_RE = re.compile(
    r"[、。，．・〜「」『』（）\(\)！？!?\s　.,;:\"'\-]"
)


def to_hiragana(text: str) -> str:
    """Convert mixed-script Japanese to all-hiragana via pykakasi.

    - 漢字 → hiragana reading (e.g. "砂漠" → "さばく")
    - katakana → hiragana (e.g. "サハラ" → "さはら")
    - hiragana → unchanged
    - ASCII / digits / symbols → unchanged (passes through pykakasi)

    Empty input returns "" without invoking pykakasi.
    """
    if not text:
        return ""
    return "".join(item["hira"] for item in _kakasi().convert(text))


def normalize_for_alignment(text: str) -> str:
    """Normalize text for kana-level alignment comparison.

    Pipeline: NFKC → digit mask → hiragana → strip punctuation+whitespace.
    See module docstring for rationale.

    Safe to call on EN text — pykakasi passes ASCII through unchanged, so the
    result is the input with NFKC + punctuation strip applied (no kana effect).
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _ASCII_DIGIT_RE.sub("#", text)
    text = _KANJI_NUM_RUN_RE.sub("#", text)
    text = to_hiragana(text)
    text = _STRIP_RE.sub("", text)
    return text


__all__ = ["to_hiragana", "normalize_for_alignment"]
