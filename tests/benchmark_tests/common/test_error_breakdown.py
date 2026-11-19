"""誤りの内訳と、並べて出す値 (benchmarks/common/error_breakdown.py、#488) の test。

例は、最終評価用の字幕を聞いて確かめたときに実際に出た型から取っている。
"""

from __future__ import annotations

import pytest

pytest.importorskip("jiwer")

from benchmarks.common import calculate_cer, calculate_wer
from benchmarks.common.error_breakdown import (
    VARIANTS,
    ErrorCounts,
    aggregate,
    count_errors,
    count_errors_in_parts,
    error_breakdown,
    is_filler_run,
    macro_average,
)


class TestPlainMatchesTheMainMetric:
    """``plain`` は、主の値 (calculate_cer / calculate_wer) と同じ割合になる。"""

    @pytest.mark.parametrize(
        ("ref", "hyp"),
        [
            ("前回はですね。くじゅうで雪山", "前回はですね、えっと九重で雪山"),
            ("約45分です。ここからですね", "約四十五分ですここからですね"),
            ("テントの設営です", "テントの撮影です"),
            ("同じ文です。", "同じ文です"),
        ],
    )
    def test_ja(self, ref: str, hyp: str) -> None:
        assert count_errors(ref, hyp, lang="ja").rate == pytest.approx(calculate_cer(ref, hyp, lang="ja"))

    def test_en(self) -> None:
        ref, hyp = "He can't see twenty birds", "he can not see 20 bird today"
        counts = count_errors(ref, hyp, lang="en")
        assert counts.rate == pytest.approx(calculate_wer(ref, hyp, lang="en"))
        assert (counts.substitutions, counts.deletions, counts.insertions) == (1, 0, 1)


class TestBreakdown:
    def test_asr_dropping_words_is_a_deletion(self) -> None:
        counts = count_errors("山頂までピークまで目指す", "山頂まで目指す", lang="ja")
        assert (counts.substitutions, counts.deletions, counts.insertions) == (0, 5, 0)
        assert counts.reference_length == 12

    def test_summarised_caption_shows_up_as_insertions(self) -> None:
        """字幕が発話を短くまとめていると、ASR が正しく書いた言葉が挿入に数えられる。"""
        counts = count_errors("本日のテーマは痛みの仕組みです", "本日のテーマは痛みの仕組みといった内容です", lang="ja")
        assert counts.insertions == 6 and counts.deletions == 0 and counts.substitutions == 0

    def test_wrong_word_is_a_substitution(self) -> None:
        counts = count_errors("テントの設営です", "テントの撮影です", lang="ja")
        assert (counts.substitutions, counts.deletions, counts.insertions) == (2, 0, 0)


class TestFillers:
    """片方にしか無い口ぐせは数えない。両方にある語は、口ぐせと同じ綴りでも残る。"""

    @pytest.mark.parametrize(
        ("ref", "hyp"),
        [
            ("前回はですね。九重で雪山", "前回はですね、えっと九重で雪山"),
            ("というのがですね下山後の", "というのがですねあの下山後の"),
            ("それは次録りましょう", "それはまあ次録りましょう"),
            ("ここからですね", "えーここからですねえーっと"),
            ("ここからですね", "アノーここからですね"),
        ],
    )
    def test_filler_only_in_the_asr_output_is_not_an_insertion(self, ref: str, hyp: str) -> None:
        assert count_errors(ref, hyp, lang="ja").insertions > 0
        assert count_errors(ref, hyp, lang="ja", variant="fillers") == ErrorCounts(reference_length=len(ref.replace("。", "")))

    def test_filler_only_in_the_caption_is_not_a_deletion(self) -> None:
        """口ぐせを落とす ASR と、口ぐせまで書いた字幕。正解の長さからも外す。"""
        counts = count_errors("えっと九重で雪山", "九重で雪山", lang="ja", variant="fillers")
        assert counts == ErrorCounts(reference_length=5)

    def test_word_spelled_like_a_filler_is_kept_when_both_sides_have_it(self) -> None:
        """「あの人」の「あの」は口ぐせではない。両方にあるので一致として残り、落とせば脱落になる。"""
        assert count_errors("あの人が来た", "あの人が来た", lang="ja", variant="fillers") == ErrorCounts(reference_length=6)
        dropped = count_errors("あの人が来た", "人が来た", lang="ja", variant="fillers")
        # 片方にしか無い「あの」は口ぐせと区別できないので数えない。この限界は文書に書いてある
        assert dropped.errors == 0 and dropped.reference_length == 4

    def test_repeated_characters_do_not_hide_the_filler(self) -> None:
        """「あのあの人」は、突き合わせの切れ目が「あ|のあ|の人」になっても口ぐせとして外れる。"""
        assert count_errors("あの人が来た", "あのあの人が来た", lang="ja", variant="fillers").errors == 0

    def test_real_insertions_are_still_counted(self) -> None:
        counts = count_errors("山頂まで目指す", "山頂までえっとピークまで目指す", lang="ja", variant="fillers")
        # 「えっとピークまで」は口ぐせだけではないので、まとめて挿入に数える
        assert counts.insertions == 8

    def test_single_kana_and_ordinary_words_are_not_fillers(self) -> None:
        for text in ("え", "あ", "ま", "ん", "その", "こう", "ちょっと", "ピーク", ""):
            assert not is_filler_run(text)
        for text in ("えー", "えっと", "えーっと", "あの", "あのー", "まあ", "なんか", "うーん", "えーあのー", "エート"):
            assert is_filler_run(text)

    def test_other_languages_are_unchanged(self) -> None:
        ref, hyp = "we should go", "um we should uh go"
        assert count_errors(ref, hyp, lang="en", variant="fillers") == count_errors(ref, hyp, lang="en")


@pytest.fixture
def kana_deps() -> None:
    """``digits`` と ``kana`` が使う dev の依存。無ければ、その test だけを飛ばす (ほかの test は走らせる)。"""
    pytest.importorskip("kanjize")
    pytest.importorskip("pykakasi")


@pytest.mark.usefixtures("kana_deps")
class TestDigitsAndKana:
    def test_digits_unifies_numerals_only(self) -> None:
        ref, hyp = "約45分で3つ出来る", "約四十五分で三つできる"
        assert count_errors(ref, hyp, lang="ja").errors > 0
        digits = count_errors(ref, hyp, lang="ja", variant="digits")
        assert (digits.substitutions, digits.deletions, digits.insertions) == (2, 0, 0)  # 出来る / できる だけが残る
        assert count_errors(ref, hyp, lang="ja", variant="kana").errors == 0

    def test_kana_keeps_different_readings_as_errors(self) -> None:
        """字幕「米国」、発話「アメリカ」は、読みに直しても違う (字幕の側を直す型)。"""
        assert count_errors("米国の責任", "アメリカの責任", lang="ja", variant="kana").errors > 0

    def test_speaker_labels_are_removed_before_the_reading_is_taken(self) -> None:
        counts = count_errors("山田：出来る", "できる", lang="ja", variant="kana", speaker_labels=["山田"])
        assert counts.errors == 0


class TestEmptySides:
    def test_empty_reference(self) -> None:
        assert count_errors("", "", lang="ja").rate == 0.0
        counts = count_errors("[音楽]", "こんにちは", lang="ja")
        assert counts.insertions == 5 and counts.reference_length == 0 and counts.rate == 1.0

    def test_empty_hypothesis(self) -> None:
        counts = count_errors("こんにちは", "", lang="ja")
        assert counts.deletions == 5 and counts.rate == 1.0

    def test_fillers_against_an_empty_side(self) -> None:
        """片方が空でも、もう片方が口ぐせだけなら数えない。口ぐせ以外を含めば、これまでどおり数える。"""
        assert count_errors("えっと", "", lang="ja", variant="fillers") == ErrorCounts()
        assert count_errors("", "あのー", lang="ja", variant="fillers") == ErrorCounts()
        assert count_errors("", "あのー", lang="ja").insertions == 3
        assert count_errors("えっと九重", "", lang="ja", variant="fillers").deletions == 5

    def test_fillers_when_the_caption_is_only_markup(self) -> None:
        """``[音楽]`` は正規化で空になる。字幕の無い区間に ASR が口ぐせだけを出しても数えず、言葉を出せば数える。"""
        assert count_errors("[音楽]", "あのー", lang="ja", variant="fillers") == ErrorCounts()
        assert count_errors("[音楽]", "山", lang="ja", variant="fillers") == ErrorCounts(insertions=1)

    def test_ordinary_words_against_an_empty_side_still_count(self) -> None:
        assert count_errors("", "山", lang="ja", variant="fillers") == ErrorCounts(insertions=1)
        assert count_errors("山", "", lang="ja", variant="fillers") == ErrorCounts(deletions=1, reference_length=1)

    def test_empty_side_agrees_with_the_aligned_case_when_aggregated(self) -> None:
        """片方が空の区間と、ほかの区間に混ざった場合とで、口ぐせの扱いが同じになる。"""
        separate = aggregate([
            count_errors("", "えっと", lang="ja", variant="fillers"),
            count_errors("山", "山", lang="ja", variant="fillers"),
        ])
        joined = count_errors("山", "えっと山", lang="ja", variant="fillers")
        assert separate == joined == ErrorCounts(reference_length=1)


class TestAggregation:
    def test_aggregate_is_the_pooled_rate_not_the_mean_of_rates(self) -> None:
        short = ErrorCounts(substitutions=5, reference_length=10)  # 50%
        long = ErrorCounts(substitutions=10, reference_length=990)  # 約 1%
        assert aggregate([short, long]).rate == pytest.approx(15 / 1000)
        assert macro_average({"a": short, "b": long}) == pytest.approx((0.5 + 10 / 990) / 2)

    def test_aggregate_of_nothing_is_zero(self) -> None:
        assert aggregate([]) == ErrorCounts()

    def test_macro_average_needs_a_group(self) -> None:
        with pytest.raises(ValueError):
            macro_average({})

    def test_to_dict(self) -> None:
        assert ErrorCounts(1, 2, 3, 10).to_dict() == {
            "rate": 0.6, "substitutions": 1, "deletions": 2, "insertions": 3, "reference_length": 10,
        }


@pytest.mark.usefixtures("kana_deps")
def test_breakdown_returns_every_variant_for_japanese_only() -> None:
    assert tuple(error_breakdown("三つ", "3つ", lang="ja")) == VARIANTS
    assert tuple(error_breakdown("three", "3", lang="en")) == ("plain",)


def test_unknown_variant() -> None:
    with pytest.raises(ValueError, match="unknown variant"):
        count_errors("a", "a", lang="ja", variant="nope")


class TestCountErrorsInParts:
    """正解の一部を採点から外す数え方 (#502 F2)。ASR の出力は時刻で切らない。"""

    @pytest.mark.parametrize("variant", ["plain", "fillers"])
    def test_without_unscored_parts_it_matches_count_errors(self, variant: str) -> None:
        parts = [("前回はですね。", True), ("くじゅうで雪山", True)]
        hyp = "前回はですね、えっと九重で雪山"
        assert count_errors_in_parts(parts, hyp, lang="ja", variant=variant) == count_errors(
            "前回はですね。 くじゅうで雪山", hyp, lang="ja", variant=variant
        )

    def test_errors_on_unscored_characters_are_not_counted(self) -> None:
        parts = [("今日は", True), ("SELECT", False), ("します", True)]
        # 外す区間の置換 (SELECT → セレクト) と、それに続く挿入は数えない
        assert count_errors_in_parts(parts, "今日はセレクトします", lang="ja") == ErrorCounts(reference_length=6)
        # 外す区間の言葉が出なくても (脱落)、数えない
        assert count_errors_in_parts(parts, "今日はします", lang="ja") == ErrorCounts(reference_length=6)

    def test_errors_on_scored_characters_are_still_counted(self) -> None:
        parts = [("今日は", True), ("SELECT", False), ("します", True)]
        assert count_errors_in_parts(parts, "今日はセレクトしま", lang="ja") == ErrorCounts(deletions=1, reference_length=6)
        # 「日は」の 2 文字は、脱落にも置換 (後ろの言葉と対応) にもなりうる。どちらでも 2 件の誤り
        missing = count_errors_in_parts(parts, "今セレクトします", lang="ja")
        assert (missing.errors, missing.reference_length) == (2, 6)
        assert count_errors_in_parts(parts, "", lang="ja") == ErrorCounts(deletions=6, reference_length=6)

    def test_insertions_between_scored_characters_are_counted(self) -> None:
        parts = [("今日は", True), ("SELECT", False), ("します", True)]
        assert count_errors_in_parts(parts, "今日わはセレクトします", lang="ja").insertions == 1

    def test_only_unscored_reference(self) -> None:
        assert count_errors_in_parts([("聞き取れない", False)], "ききとれない", lang="ja") == ErrorCounts()
