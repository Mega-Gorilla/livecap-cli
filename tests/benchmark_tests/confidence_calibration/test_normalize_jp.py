"""Tests for ``benchmarks.confidence_calibration._normalize_jp``.

Verifies the kana-level normalization pipeline used by PR-γ kana alignment
metric. Key invariants tested:

* digit mask (ASCII + 漢数字) runs BEFORE pykakasi (D2 in the PR-γ plan), so
  ``"1人で"`` and ``"一人で"`` collapse to the same normalised form.
* katakana / kanji / hiragana 表記揺れが kana 化で吸収されること.
* 真の音響誤認識 (e.g. ``"真っ先"`` vs ``"さっき"``) は kana 化しても異なる.
* EN passthrough — ASCII text is unchanged (pykakasi passes through).
"""

from __future__ import annotations

import pytest

from benchmarks.confidence_calibration._normalize_jp import (
    normalize_for_alignment,
    to_hiragana,
)


class TestToHiragana:
    def test_empty_returns_empty(self) -> None:
        assert to_hiragana("") == ""

    def test_hiragana_passes_through(self) -> None:
        assert to_hiragana("ひらがな") == "ひらがな"

    def test_katakana_to_hiragana(self) -> None:
        assert to_hiragana("サハラ") == "さはら"

    def test_kanji_to_hiragana_reading(self) -> None:
        # 砂漠 → さばく は pykakasi default 読み
        assert to_hiragana("砂漠") == "さばく"

    def test_mixed_kanji_kana(self) -> None:
        # サハラ砂漠 → さはらさばく
        assert to_hiragana("サハラ砂漠") == "さはらさばく"

    def test_ascii_passes_through(self) -> None:
        # English text is not converted by pykakasi
        assert to_hiragana("It was a picture") == "It was a picture"


class TestNormalizeForAlignmentBasics:
    def test_empty_returns_empty(self) -> None:
        assert normalize_for_alignment("") == ""

    def test_whitespace_only_returns_empty(self) -> None:
        assert normalize_for_alignment("   ") == ""
        assert normalize_for_alignment("　") == ""  # 全角空白

    def test_pure_hiragana_passes(self) -> None:
        assert normalize_for_alignment("あいうえお") == "あいうえお"

    def test_strips_japanese_punctuation(self) -> None:
        assert normalize_for_alignment("こんにちは、世界。") == "こんにちはせかい"


class TestNormalizeForAlignmentDigitMask:
    """Digit mask must run BEFORE pykakasi (D2 in PR-γ plan).

    This is the key invariant for Phase 4 segment 0014:
        "1人で..." vs "一人で..."
    must collapse to the same normalized form so kana coverage is high.
    """

    def test_ascii_digits_masked(self) -> None:
        # 1000 → # so "1000マイル" → "#まいる"
        result = normalize_for_alignment("1000マイル")
        assert result == "#まいる"

    def test_kanji_digits_masked(self) -> None:
        # 千 → # so "千マイル" → "#まいる"
        result = normalize_for_alignment("千マイル")
        assert result == "#まいる"

    def test_digit_format_difference_normalized(self) -> None:
        # The Phase 4 0014 case (1 vs 一 prefix)
        a = normalize_for_alignment("1人でエンジンを修理")
        b = normalize_for_alignment("一人でエンジンを修理")
        assert a == b, f"Digit format diff not normalized: {a!r} vs {b!r}"


class TestNormalizeForAlignmentNFKC:
    def test_fullwidth_ascii_to_halfwidth(self) -> None:
        # ＡＢＣ１２３ → ABC123 → then digit mask → ABC#
        result = normalize_for_alignment("ＡＢＣ１２３")
        assert result == "ABC#"

    def test_fullwidth_punctuation_stripped(self) -> None:
        # ， → , (NFKC) → stripped
        assert normalize_for_alignment("あ，い．う") == "あいう"


class TestNormalizeForAlignmentRealCases:
    """Pinned behavior for Phase 4 raw data cases (smoke verify segments)."""

    def test_phase4_0010_katakana_kanji_aligns(self) -> None:
        # サハラ砂漠 (katakana+kanji) vs さはらさばく (pure hiragana reference)
        # → both collapse to "さはらさばく"
        asr = normalize_for_alignment("サハラ砂漠")
        reading = normalize_for_alignment("さはらさばく")
        assert asr == reading

    def test_phase4_0014_digit_kanji_align(self) -> None:
        # 1人で... vs 一人で... must produce identical kana
        asr = normalize_for_alignment("一人でエンジンを修理しなければならなかった")
        ref = normalize_for_alignment("1人でエンジンを修理しなければならなかった")
        assert asr == ref

    def test_phase4_0006_real_misrecognition_stays_different(self) -> None:
        # 真っ先 vs さっき are real ASR errors (same-sound confusion);
        # kana normalization must NOT make them match.
        asr = normalize_for_alignment("さっきに")
        ref = normalize_for_alignment("真っ先に")
        assert asr != ref


class TestNormalizeForAlignmentEnglish:
    """EN audio support — pykakasi passes ASCII through, so kana version
    degrades gracefully to (NFKC + strip) for EN text.
    """

    def test_pure_english_only_strip(self) -> None:
        # Period + spaces stripped; letters kept as-is
        result = normalize_for_alignment("It was a picture of a boa constrictor.")
        assert result == "Itwasapictureofaboaconstrictor"

    def test_english_with_apostrophe_stripped(self) -> None:
        assert normalize_for_alignment("don't") == "dont"

    def test_phase4_en_equivalence(self) -> None:
        # Two identical EN strings (modulo whitespace + punctuation)
        a = normalize_for_alignment("It was a picture.")
        b = normalize_for_alignment("It was a picture .")
        assert a == b


class TestKakasiCacheStable:
    """Single kakasi instance is reused across calls (lru_cache(1))."""

    def test_repeated_calls_consistent(self) -> None:
        # Same input → same output across repeated calls
        text = "サハラ砂漠で1人"
        results = [normalize_for_alignment(text) for _ in range(3)]
        assert len(set(results)) == 1


@pytest.mark.parametrize(
    "text",
    [
        "",
        "あ",
        "サハラ砂漠",
        "1000マイル",
        "千マイル",
        "It was a picture.",
    ],
)
def test_normalize_is_idempotent(text: str) -> None:
    """normalize(normalize(x)) == normalize(x) — should be stable."""
    once = normalize_for_alignment(text)
    twice = normalize_for_alignment(once)
    assert once == twice
