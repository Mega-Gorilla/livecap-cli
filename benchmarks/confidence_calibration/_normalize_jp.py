"""Japanese kana-level normalization for calibration alignment metrics.

This helper is **dev / benchmark-only**. ``pykakasi`` (``GPL-3.0-or-later``)
and ``kanjize`` (``MIT``) are development dependencies declared in
``pyproject.toml`` under ``[project.optional-dependencies] dev``. The
livecap-cli production runtime must never import this module — that
invariant is statically enforced by ``tests/test_production_no_pykakasi.py``
(Issue #338 PR-γ).

Purpose
-------
The PR-β active calibration harness aligns ASR output against a reference
transcript by string coverage. For Japanese, text-level coverage conflates
two unrelated effects:

* **Acoustic confidence** — does the ASR's phoneme sequence match the
  reference? This is what threshold calibration actually wants to evaluate.
* **Lexical surface form** — does the ASR's kanji / katakana / digit choice
  match the reference's surface form? This is independent of acoustic
  confidence (e.g. ``1人で`` vs ``一人で`` share the same reading ``ひとりで``).

This module normalises both sides of the comparison to a kana-only form so
the kana-level alignment score isolates acoustic confidence from surface
form noise.

Normalization pipeline (``normalize_for_alignment``)
----------------------------------------------------
1. NFKC: 全角 → 半角 ASCII, 全角 punctuation → half-width.
2. **Arabic-to-kanji numeral conversion**: digit runs adjacent to a CJK
   character (hiragana / katakana / kanji) are converted to kanji compound
   numerals via :func:`kanjize.number2kanji` (e.g. ``"1人"`` → ``"一人"``,
   ``"1200マイル"`` → ``"千二百マイル"``). Digit runs not adjacent to CJK
   (e.g. ``"Chapter 1"`` in EN-only text) are left as-is.
3. pykakasi: kanji → hiragana reading, katakana → hiragana, ASCII pass-through.
4. Strip punctuation + whitespace.

Step 2 must run **before** step 3 so that pykakasi's natural compound rules
apply consistently regardless of whether the input was originally written
in kanji or algebraic form:

* ``"1人"`` (algebraic)   → ``"一人"`` (step 2) → ``"ひとり"`` (pykakasi)
* ``"一人"`` (kanji)      → ``"一人"`` (no-op)  → ``"ひとり"`` (pykakasi)
* ``"一緒に"`` (idiom)    → ``"一緒に"`` (no-op) → ``"いっしょに"`` (pykakasi)
* ``"千二百マイル"``       → unchanged          → ``"せんにひゃくまいる"``
* ``"1200マイル"``        → ``"千二百マイル"``    → ``"せんにひゃくまいる"``

Idiomatic non-numeric compounds (``一緒``, ``十分``, ``一番``, ``一人``…)
that contain a kanji digit character are preserved as-is — they never have
an Arabic counterpart in mixed-form ASR output, so the inverse direction
(arabic → kanji) avoids breaking pykakasi's compound dictionary.

EN text containing numbers (``"Chapter 1"``, ``"It was about 100 years ago."``)
is also unaffected because the Arabic-to-kanji conversion only triggers
when at least one neighbour is a CJK character.

Design history (PR #341)
------------------------
* **v1 (blanket mask, reverted)** — collapsed all digits to ``"#"``. Failed
  ``一人`` (1 person) vs ``二人`` (2 people) by producing the same string,
  hiding real ASR errors as false-high coverage.
* **v2 (per-char kanji → arabic, reverted)** — substituted each kanji digit
  by its algebraic value individually. Correct for single-char compounds
  (``一人 ↔ 1人``) but broke compound numerals (``千二百`` → ``"10002100"``
  instead of ``"1200"``).
* **v3 (kanjize kanji → int, reverted)** — used :func:`kanjize.kanji2number`
  to convert kanji compound runs to integers. Fixed compound numerals but
  broke common idiomatic compounds: pykakasi's natural reading of
  ``一緒`` / ``十分`` / ``一番`` / ``一人`` was destroyed because ``一`` was
  first replaced with ``"1"`` and the remaining word (``"1緒"``, ``"10分"``,
  ``"1番"``, ``"1人"``) could no longer be looked up.
* **v4 (current, arabic → kanji via kanjize, CJK-adjacent only)** — reverses
  direction. Arabic digits in Japanese context become kanji compounds, then
  pykakasi handles the resulting all-kanji text with its native compound
  rules. Idiomatic compounds and EN text are preserved. Conversion failures
  (e.g. negative numbers, extreme magnitudes outside kanjize's range) fall
  back to leaving the digits unchanged.
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


@lru_cache(maxsize=1)
def _number2kanji():
    """Return :func:`kanjize.number2kanji` (memoised lazy import).

    Same lazy-import rationale as :func:`_kakasi`. Raises
    ``ModuleNotFoundError`` on first call if kanjize is missing.
    """
    from kanjize import number2kanji  # noqa: PLC0415 — intentional lazy import

    return number2kanji


# CJK script range: hiragana (U+3040-309F), katakana (U+30A0-30FF), and the
# CJK Unified Ideographs main block (U+4E00-9FFF). Used to decide whether an
# Arabic digit run is in a Japanese context (and therefore should be converted
# to a kanji compound for symmetric alignment).
_CJK_CLASS = r"[぀-ヿ一-鿿]"

# Match Arabic digit runs that have at least one CJK character as an
# immediate neighbour (either before or after). Pure-English contexts like
# ``"Chapter 1"`` are not matched.
_ARABIC_NEAR_CJK_RE = re.compile(
    rf"(?:(?<={_CJK_CLASS})\d+|\d+(?={_CJK_CLASS}))"
)

# Punctuation + whitespace to strip after normalization (alignment 比較用 noise).
# Note: step 1 (NFKC) folds 全角 punctuation (．，（）！？) into half-width ASCII,
# so the half-width forms must be present here. Full-width forms are kept too
# as a safety net in case NFKC misses anything (e.g. 「」『』 are not folded).
_STRIP_RE = re.compile(
    r"[、。，．・〜「」『』（）\(\)！？!?\s　.,;:\"'\-]"
)


def _replace_arabic_digit_run(match: "re.Match[str]") -> str:
    """Convert an Arabic digit run to a kanji compound numeral.

    Uses :func:`kanjize.number2kanji` so that pykakasi can subsequently read
    the result with its natural compound rules. Falls back to the original
    digit run if conversion fails (e.g. very large integers outside kanjize's
    range or unexpected input). The fallback preserves deterministic output
    so that the alignment metric remains consistent.
    """
    digits = match.group(0)
    try:
        return _number2kanji()(int(digits))
    except (ValueError, OverflowError):
        return digits


def to_hiragana(text: str) -> str:
    """Convert mixed-script Japanese to all-hiragana via pykakasi.

    - 漢字 → hiragana reading (e.g. ``"砂漠"`` → ``"さばく"``)
    - katakana → hiragana (e.g. ``"サハラ"`` → ``"さはら"``)
    - hiragana → unchanged
    - ASCII / digits / symbols → unchanged (passes through pykakasi)

    Empty input returns ``""`` without invoking pykakasi.
    """
    if not text:
        return ""
    return "".join(item["hira"] for item in _kakasi().convert(text))


def normalize_for_alignment(text: str) -> str:
    """Normalize text for kana-level alignment comparison.

    Pipeline: NFKC → arabic-to-kanji (in CJK context) → hiragana → strip
    punctuation+whitespace. See module docstring for the full rationale and
    the design history v1/v2/v3/v4.

    Safe to call on EN text — pykakasi passes ASCII through unchanged, and
    the Arabic-to-kanji conversion only triggers when digits are adjacent
    to CJK characters, so pure-English text with numbers is unaffected.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _ARABIC_NEAR_CJK_RE.sub(_replace_arabic_digit_run, text)
    text = to_hiragana(text)
    text = _STRIP_RE.sub("", text)
    return text


__all__ = ["to_hiragana", "normalize_for_alignment"]
