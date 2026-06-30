"""Tests for ``benchmarks.confidence_calibration._normalize_jp``.

Verifies the kana-level normalization pipeline used by PR-γ kana alignment
metric (v4 design — see ``_normalize_jp.py`` docstring). Key invariants:

* **Numeric canonicalisation** runs BEFORE pykakasi so that algebraic and
  kanji representations of the same value collapse to the same kana (e.g.
  ``"1人で"`` and ``"一人で"`` both become ``"ひとりで"``).
* Compound numerals match across surface form (``"千二百マイル"`` ↔
  ``"1200マイル"`` both → ``"せんにひゃくまいる"``).
* Idiomatic compounds with kanji digit characters are preserved (``"一緒"``
  ↔ ``"いっしょ"``, ``"十分"`` ↔ ``"じゅうぶん"``, ``"一番"`` ↔ ``"いちばん"``).
* Different counts remain distinguishable (``"一人"`` → ``"ひとり"`` vs
  ``"二人"`` → ``"ふたり"``).
* katakana / kanji / hiragana 表記揺れが kana 化で吸収されること.
* 真の音響誤認識 (e.g. ``"真っ先"`` vs ``"さっき"``) は kana 化しても異なる.
* EN passthrough — ASCII digits are only converted when adjacent to CJK
  characters, so ``"Chapter 1"`` is unchanged.
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


class TestNormalizeForAlignmentDigitCanonicalisation:
    """Numeric canonicalisation must run BEFORE pykakasi (v4, PR #341).

    v4 converts Arabic digit runs adjacent to a CJK character into kanji
    compound numerals via :func:`kanjize.number2kanji`. pykakasi then reads
    the result with its native compound rules. This unifies surface form
    across algebraic and kanji representations while preserving value
    distinctions and idiomatic compound readings.

    Key invariants:

    1. Same value, different surface form → MATCH (1人 == 一人, 千 == 1000)
    2. Different values → DIFFER (一人 != 二人, 千マイル != 一マイル)
    3. Idiomatic compounds preserved (一緒 → いっしょ, 十分 → じゅうぶん)

    Earlier design history (reverted): v1 blanket mask, v2 per-char
    canonicalisation, v3 kanji-to-arabic via :func:`kanjize.kanji2number`.
    See module docstring for details.
    """

    # ---- legitimate matches: same value, different surface ----

    def test_ascii_digit_adjacent_to_cjk_becomes_kanji(self) -> None:
        """Arabic digits adjacent to a CJK char are converted to kanji,
        then read by pykakasi natural compound rules."""
        # 1000マイル → 千マイル → せんまいる
        assert normalize_for_alignment("1000マイル") == "せんまいる"

    def test_kanji_thousand_read_by_pykakasi(self) -> None:
        """Kanji digits stay as-is; pykakasi reads compound naturally."""
        # 千マイル → せんまいる (no Arabic-to-kanji conversion needed)
        assert normalize_for_alignment("千マイル") == "せんまいる"

    def test_thousand_matches_across_surface(self) -> None:
        # The Phase 4 motivating case: 千 vs 1000 must produce the same kana
        assert normalize_for_alignment("千マイル") == normalize_for_alignment(
            "1000マイル"
        )

    def test_1_person_matches_1_kanji(self) -> None:
        # Phase 4 segment 0014: 一人 vs 1人 — both should read "ひとり"
        # via pykakasi's compound rules (1 → 一 in JA context, then 一人 → ひとり)
        a = normalize_for_alignment("一人でエンジン")
        b = normalize_for_alignment("1人でエンジン")
        assert a == b, f"1 vs 一 surface diff should normalise: {a!r} vs {b!r}"

    # ---- value-preserving distinctions: different values → must DIFFER ----

    def test_one_person_differs_from_two_person(self) -> None:
        """1 vs 2 surface form must NOT collapse — this is the reviewer's case."""
        a = normalize_for_alignment("一人で")
        b = normalize_for_alignment("二人で")
        assert a != b, (
            f"一人 (1 person) and 二人 (2 people) must NOT match "
            f"(reviewer's case): {a!r} == {b!r}"
        )

    def test_1000_miles_differs_from_1_mile(self) -> None:
        """1000 vs 1 algebraic must NOT collapse."""
        a = normalize_for_alignment("1000マイル")
        b = normalize_for_alignment("1マイル")
        assert a != b, (
            f"1000マイル and 1マイル must NOT match (reviewer's case): "
            f"{a!r} == {b!r}"
        )

    def test_thousand_miles_differs_from_one_mile(self) -> None:
        """1000 (千) vs 1 (一) kanji must NOT collapse."""
        a = normalize_for_alignment("千マイル")
        b = normalize_for_alignment("一マイル")
        assert a != b, (
            f"千マイル and 一マイル must NOT match (reviewer's case): "
            f"{a!r} == {b!r}"
        )

    def test_single_kanji_digits_all_distinct(self) -> None:
        """Each kanji digit produces a distinct normalised form."""
        forms = {
            normalize_for_alignment("一個"),
            normalize_for_alignment("二個"),
            normalize_for_alignment("三個"),
            normalize_for_alignment("四個"),
            normalize_for_alignment("五個"),
        }
        assert len(forms) == 5, f"all 5 kanji digits should differ: {forms}"


class TestNormalizeForAlignmentCompoundNumerals:
    """Compound kanji numerals must round-trip through kanjize correctly.

    The per-char design (PR #341 review v2) failed for compound forms:
    ``千二百`` was rewritten as ``"10002100"`` (per-char) instead of
    ``"1200"`` (semantic value), so ``千二百マイル`` did not match
    ``1200マイル``. The kanjize-powered v3 implementation parses the
    full kanji numeral run and produces the integer value, fixing
    cross-form alignment for compounds.
    """

    def test_compound_1200_matches_algebraic(self) -> None:
        """千二百 (kanji compound) must match 1200 (algebraic)."""
        a = normalize_for_alignment("千二百マイル")
        b = normalize_for_alignment("1200マイル")
        assert a == b, (
            f"千二百 should resolve to 1200 (kanjize), got {a!r} vs {b!r}"
        )

    def test_compound_1234_matches_algebraic(self) -> None:
        """千二百三十四 must match 1234 (4-element compound)."""
        a = normalize_for_alignment("千二百三十四マイル")
        b = normalize_for_alignment("1234マイル")
        assert a == b, (
            f"千二百三十四 should resolve to 1234, got {a!r} vs {b!r}"
        )

    def test_compound_with_man_matches_algebraic(self) -> None:
        """一万二千三百 (with 万) must match 12300."""
        a = normalize_for_alignment("一万二千三百個")
        b = normalize_for_alignment("12300個")
        assert a == b, (
            f"一万二千三百 should resolve to 12300, got {a!r} vs {b!r}"
        )

    def test_compound_within_long_text(self) -> None:
        """Compound resolution works embedded in longer text (regex-based sub)."""
        a = normalize_for_alignment("私の故郷は千二百キロ離れた場所にある")
        b = normalize_for_alignment("私の故郷は1200キロ離れた場所にある")
        assert a == b, (
            f"compound within text should match: {a!r} vs {b!r}"
        )

    def test_compound_different_values_still_differ(self) -> None:
        """千二百 (1200) and 千二百三十四 (1234) must NOT match."""
        a = normalize_for_alignment("千二百マイル")
        b = normalize_for_alignment("千二百三十四マイル")
        assert a != b, (
            f"different compound values should differ: {a!r} vs {b!r}"
        )

    def test_invalid_compound_left_as_kanji(self) -> None:
        """Unusual kanji digit composition (千千) is left as-is for pykakasi.

        In v4 the Arabic-to-kanji direction means we only call
        :func:`kanjize.number2kanji` for **Arabic** digit runs. Kanji digit
        runs (including unusual compositions like ``千千``) simply pass
        through to pykakasi, which reads each char by its base kana
        (``千千`` → ``"せんせん"``). The metric remains deterministic and
        symmetric for any input.
        """
        result_a = normalize_for_alignment("千千マイル")
        result_b = normalize_for_alignment("千千マイル")
        assert result_a == result_b  # deterministic
        # No exception, output is a non-empty string
        assert isinstance(result_a, str) and result_a


class TestNormalizeForAlignmentReviewerCases:
    """Reviewer-provided regression cases for PR #341 v4 (Arabic→kanji).

    These 8 cases pin the invariants laid out in the 2nd-round review
    comment: idiomatic compounds preserve pykakasi's natural reading,
    cross-form alignment works for both single and compound numerals,
    and different values remain distinguishable.
    """

    def test_isshoni_idiom(self) -> None:
        """一緒に ↔ いっしょに must match (idiomatic compound, kanjize-side no-op)."""
        a = normalize_for_alignment("一緒に")
        b = normalize_for_alignment("いっしょに")
        assert a == b, f"一緒に ↔ いっしょに: {a!r} vs {b!r}"

    def test_juubun_idiom(self) -> None:
        """十分です ↔ じゅうぶんです must match."""
        a = normalize_for_alignment("十分です")
        b = normalize_for_alignment("じゅうぶんです")
        assert a == b, f"十分です ↔ じゅうぶんです: {a!r} vs {b!r}"

    def test_ichiban_idiom(self) -> None:
        """一番 ↔ いちばん must match."""
        a = normalize_for_alignment("一番")
        b = normalize_for_alignment("いちばん")
        assert a == b, f"一番 ↔ いちばん: {a!r} vs {b!r}"

    def test_hitori_compound(self) -> None:
        """一人で ↔ ひとりで must match (pykakasi compound dictionary)."""
        a = normalize_for_alignment("一人で")
        b = normalize_for_alignment("ひとりで")
        assert a == b, f"一人で ↔ ひとりで: {a!r} vs {b!r}"

    def test_hitori_cross_form_kanji_vs_arabic(self) -> None:
        """一人で ↔ 1人で must match (Arabic→kanji conversion bridges them)."""
        a = normalize_for_alignment("一人で")
        b = normalize_for_alignment("1人で")
        assert a == b, f"一人で ↔ 1人で: {a!r} vs {b!r}"

    def test_compound_1200_cross_form(self) -> None:
        """千二百マイル ↔ 1200マイル must match (compound numeral, both → せんにひゃく…)."""
        a = normalize_for_alignment("千二百マイル")
        b = normalize_for_alignment("1200マイル")
        assert a == b, f"千二百マイル ↔ 1200マイル: {a!r} vs {b!r}"

    def test_one_person_vs_two_person_differ(self) -> None:
        """一人 ↔ 二人 must differ (different counts → ひとり vs ふたり)."""
        a = normalize_for_alignment("一人")
        b = normalize_for_alignment("二人")
        assert a != b, f"一人 ↔ 二人 should differ: {a!r} vs {b!r}"

    def test_1000_vs_1_mile_differ(self) -> None:
        """1000マイル ↔ 1マイル must differ (different magnitudes)."""
        a = normalize_for_alignment("1000マイル")
        b = normalize_for_alignment("1マイル")
        assert a != b, f"1000マイル ↔ 1マイル should differ: {a!r} vs {b!r}"

    def test_english_chapter_1_preserved(self) -> None:
        """EN context: ``Chapter 1`` must NOT be Arabic→kanji converted.

        The ``1`` in ``Chapter 1`` is not adjacent to any CJK character
        (surrounded by Latin letter and EOL), so the conversion regex
        does not match it. pykakasi then leaves ASCII as-is.
        """
        result = normalize_for_alignment("Chapter 1")
        assert "1" in result, (
            f"ASCII digit in EN context must be preserved: {result!r}"
        )
        # Should not contain Japanese kanji
        assert not any(
            "一" <= ch <= "鿿" for ch in result
        ), f"EN text must not contain kanji: {result!r}"


class TestNormalizeForAlignmentNFKC:
    def test_fullwidth_ascii_to_halfwidth(self) -> None:
        # ＡＢＣ１２３ → ABC123 (NFKC), digits preserved (PR #341 review fix)
        result = normalize_for_alignment("ＡＢＣ１２３")
        assert result == "ABC123"

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
