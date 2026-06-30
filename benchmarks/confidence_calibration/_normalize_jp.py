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
2. **Kanji numeral canonicalisation**: per-character translation
   ``一 → 1, 二 → 2, ... 千 → 1000`` etc. (digits themselves preserved,
   not masked). This unifies surface form (``千`` vs ``1000``) while
   keeping the numeric value distinguishable (``一`` vs ``二`` remain
   different).
3. pykakasi: kanji → hiragana reading, katakana → hiragana, ASCII pass-through
4. Strip punctuation + whitespace

Step 2 must run **before** step 3. Otherwise pykakasi would resolve
compound kanji numerals like ``一人 → ひとり`` (compound reading) while
``1人 → 1にん`` (separate tokens), breaking the surface-form match that
motivates the kana metric. By substituting ``一 → 1`` first, both forms
become ``1人`` and pykakasi produces the same kana on both sides.

Trade-off (documented PR #341 review correction): per-character substitution
does not correctly parse compound kanji numerals (``千二百`` → ``10002100``
under per-char rules, semantically ``1200``). For Phase 4 朗読 corpus this
is acceptable because (a) both ASR-side and reference-side go through the
SAME deterministic transformation so alignment still works for same-surface
inputs, and (b) compound kanji numerals are rare in this corpus. A full
Japanese numeral parser (e.g. ``kanjize``) would be required to correctly
align ``千二百`` (kanji compound) against ``1200`` (algebraic) — left as
future work if Phase 4 measurements show this edge case matters.
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


# Per-character kanji-numeral canonicalisation map (PR #341 review fix).
#
# Pre-fix (blanket mask): both ``一`` and ``二`` collapsed to ``"#"``, so
# ``一人`` and ``二人`` were treated identical → false-high coverage for
# real ASR mistakes between different counts. This map preserves the
# numeric value by substituting each kanji digit with its algebraic
# equivalent (str.translate accepts multi-char string values per Python docs).
#
# Note: this is per-character, not a full numeral parser. ``千二百``
# becomes the deterministic-but-semantically-wrong string ``10002100``,
# which still differs from ``1200`` (algebraic). Both forms still match
# themselves on both sides of the alignment comparison so the metric
# works correctly for same-surface inputs (e.g. reference and ASR both
# use the kanji form). See module docstring for the full trade-off.
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

    Pipeline: NFKC → kanji-numeral canonicalisation → hiragana → strip
    punctuation+whitespace. See module docstring for rationale and
    trade-offs.

    Safe to call on EN text — pykakasi passes ASCII through unchanged, so the
    result is the input with NFKC + punctuation strip applied (no kana effect).
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_KANJI_DIGIT_TRANSLATE)
    text = to_hiragana(text)
    text = _STRIP_RE.sub("", text)
    return text


__all__ = ["to_hiragana", "normalize_for_alignment"]
