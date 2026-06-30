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
* **Lexical surface form** — does the ASR's kanji/katakana/digit choice
  match the reference's surface form? This is independent of acoustic
  confidence (e.g. "1人で" vs "一人で" share the same reading "ひとりで").

This module normalises both sides of the comparison to a kana-only form so
the kana-level alignment score isolates acoustic confidence from surface
form noise.

Normalization pipeline (``normalize_for_alignment``)
----------------------------------------------------
1. NFKC: 全角 → 半角 ASCII, 全角 punctuation → half-width
2. **Kanji numeral canonicalisation**: pure kanji numeral runs are matched
   by regex and parsed to their integer value via :mod:`kanjize`
   (e.g. ``千二百`` → ``"1200"``, ``千二百三十四`` → ``"1234"``,
   ``一万二千三百四十五`` → ``"12345"``). Single-char digits work the same
   (``一`` → ``"1"``, ``千`` → ``"1000"``). If kanjize cannot parse a run
   (rare invalid composition like ``千千``), we fall back to per-character
   substitution. ASCII digits are unchanged (algebraic representation is
   preserved).
3. pykakasi: kanji → hiragana reading, katakana → hiragana, ASCII pass-through
4. Strip punctuation + whitespace

Step 2 must run **before** step 3. Otherwise pykakasi would resolve
compound kanji numerals like ``一人 → ひとり`` (compound reading) while
``1人 → 1にん`` (separate tokens), breaking the surface-form match that
motivates the kana metric. By substituting ``一 → 1`` first, both forms
become ``1人`` and pykakasi produces the same kana on both sides.

Design history (PR #341)
------------------------
* **v1 (blanket mask, reverted)** — collapsed all digits to ``#``. Failed
  ``一人`` (1 person) vs ``二人`` (2 people) by producing the same string,
  hiding real ASR errors as false-high coverage.
* **v2 (per-char substitution)** — substituted each kanji digit by its
  algebraic value individually (``千`` → ``"1000"`` etc.). Correct for
  single-char compounds (``一人 ↔ 1人``) but broke compound numerals
  (``千二百`` → ``"10002100"`` instead of ``"1200"``).
* **v3 (current, kanjize-powered)** — matches kanji numeral runs by regex
  and delegates parsing to :func:`kanjize.kanji2number`, falling back to
  per-char on parsing failure. Compound numerals like ``千二百`` are
  correctly resolved to ``"1200"`` so cross-form (kanji compound vs
  algebraic) alignment works.

The per-char fallback preserves graceful degradation for kanjize's failure
cases (``千千`` and other invalid compositions raise ``ValueError`` —
extremely rare in real corpora, but defensive coding).
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
def _kanji2number():
    """Return :func:`kanjize.kanji2number` (memoised lazy import).

    Same lazy-import rationale as :func:`_kakasi`. Raises
    ``ModuleNotFoundError`` on first call if kanjize is missing.
    """
    from kanjize import kanji2number  # noqa: PLC0415 — intentional lazy import

    return kanji2number


# Pure kanji numeral runs (no non-digit kanji). Used to identify substrings
# that can be parsed by kanjize.kanji2number().
_KANJI_NUM_RUN_RE = re.compile(r"[〇零一二三四五六七八九十百千万億兆]+")

# Per-character kanji-numeral fallback (only used if kanjize raises on a
# specific run, e.g. ``千千`` and other invalid compositions). Documented
# in the module docstring as v2 design retained for graceful degradation.
_KANJI_DIGIT_TRANSLATE = str.maketrans(
    {
        "〇": "0",
        "零": "0",
        "一": "1",
        "二": "2",
        "三": "3",
        "四": "4",
        "五": "5",
        "六": "6",
        "七": "7",
        "八": "8",
        "九": "9",
        "十": "10",
        "百": "100",
        "千": "1000",
        "万": "10000",
        "億": "100000000",
        "兆": "1000000000000",
    }
)

# Punctuation + whitespace to strip after normalization (alignment 比較用 noise).
# Note: step 1 (NFKC) folds 全角 punctuation (．，（）！？) into half-width ASCII,
# so the half-width forms must be present here. Full-width forms are kept too
# as a safety net in case NFKC misses anything (e.g. 「」『』 are not folded).
_STRIP_RE = re.compile(
    r"[、。，．・〜「」『』（）\(\)！？!?\s　.,;:\"'\-]"
)


def _replace_kanji_numeral_run(match: "re.Match[str]") -> str:
    """Replace a pure kanji numeral run with its integer string.

    Uses kanjize to handle compound numerals correctly (``千二百`` → ``"1200"``).
    Falls back to per-character substitution if kanjize raises ``ValueError``
    on a rare invalid composition such as ``千千`` (multiple thousands).
    Both paths produce deterministic output, so the alignment metric remains
    consistent across runs.
    """
    run = match.group(0)
    try:
        return str(_kanji2number()(run))
    except ValueError:
        return run.translate(_KANJI_DIGIT_TRANSLATE)


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

    Pipeline: NFKC → kanji numeral run → integer (kanjize) → hiragana →
    strip punctuation+whitespace. See module docstring for the full
    rationale and the design-history v1/v2/v3 evolution.

    Safe to call on EN text — pykakasi passes ASCII through unchanged, so the
    result is the input with NFKC + punctuation strip applied (no kana effect).
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _KANJI_NUM_RUN_RE.sub(_replace_kanji_numeral_run, text)
    text = to_hiragana(text)
    text = _STRIP_RE.sub("", text)
    return text


__all__ = ["to_hiragana", "normalize_for_alignment"]
