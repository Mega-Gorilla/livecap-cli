"""採点の正規化 (benchmarks/common/text_normalization.py) の規則 (#488)。"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from benchmarks.common.metrics import calculate_cer, calculate_wer
from benchmarks.common.text_normalization import (
    normalize_en,
    normalize_ja,
    normalize_text,
    strip_caption_markup,
)

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "benchmarks" / "common" / "data"


class TestCaptionMarkup:
    """YODAS2 の手動字幕に出てくる書式を、どの言語でも同じに取り除く。"""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("[音楽] こんにちは", "こんにちは"),
            ("[Music] hello", "hello"),
            ("生まれ変わった部屋♪", "生まれ変わった部屋"),
            # 単語の区切りを守るため空白に置き換える (ja は採点のときに空白を消す)
            ("そういう風に（笑）僕みたいに", "そういう風に 僕みたいに"),
            ("a CV in Japanese (laughs) they have", "a CV in Japanese they have"),
            ("(F えー)それでは", "それでは"),
            ("(male) こんにちは", "こんにちは"),
            ("(Female Narrator) hello", "hello"),
            ("&gt;&gt; Hi.", "Hi."),
            ("- I see", "I see"),
            ("king&amp;prince", "king&prince"),
            ("ＡＢＣ１２３", "ABC123"),
        ],
    )
    def test_removes_caption_markup(self, text: str, expected: str) -> None:
        assert " ".join(strip_caption_markup(text).split()) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "10:30 に集合",
            "午前10:30に集合",
            "Answer: forty two",
            "注意:足元に気をつけて",
            "彼は(音楽理論)を学ぶ",
            "結(むすび)",
            "(laughing gas) is a gas",
            "A - B",
        ],
    )
    def test_keeps_what_was_spoken(self, text: str) -> None:
        """推定で消さない: 時刻、印の無い ``語:``、既知の語を含むだけの括弧、文中の ``-`` (レビュー F1)。"""
        assert strip_caption_markup(text) == text

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (">> こんにちは", "こんにちは"),
            ("- こんにちは", "こんにちは"),
            (">> - こんにちは", "こんにちは"),
            ("- >> こんにちは", "こんにちは"),
            ("- - こんにちは", "こんにちは"),
            ("&gt;&gt; Yeah. &gt;&gt; How do we do it?", "Yeah. How do we do it?"),
            ("- は〜い\n- どうも", "は〜い どうも"),
            # 印の直後でも、名前かどうか分からない ``語:`` は残す (再レビュー F1)
            (">> Answer: forty two", "Answer: forty two"),
            ("- 注意:足元に気をつけて", "注意:足元に気をつけて"),
            (">> トミー:こんにちは", "トミー:こんにちは"),
            (">> 10:30 に集合", "10:30 に集合"),
        ],
    )
    def test_removes_only_the_turn_marker(self, text: str, expected: str) -> None:
        """話者交代の印だけを消す。印の順序や重なりで結果は変わらない (再レビュー F2)。"""
        assert " ".join(strip_caption_markup(text).split()) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "トミー：こんにちは",
            ">> トミー：こんにちは",
            "- トミー：こんにちは",
            ">> - トミー：こんにちは",
            "- >> トミー：こんにちは",
            ">> - [拍手] トミー：こんにちは",
            "(male) >> ＨＥＸ: こんにちは",
            "[音楽] &gt;&gt; トミー：こん\n- ＨＥＸ: にちは",
            "トミー：こん &gt;&gt; HEX: にちは",
        ],
    )
    def test_known_speaker_names_are_removed_in_any_marker_order(self, text: str) -> None:
        stripped = strip_caption_markup(text, speaker_labels=["トミー", "HEX"])
        assert "".join(stripped.split()) == "こんにちは"

    def test_a_name_inside_an_utterance_is_kept(self) -> None:
        stripped = strip_caption_markup("トミー：こんにちは\nトミーは言った", speaker_labels=["トミー"])
        assert " ".join(stripped.split()) == "こんにちは トミーは言った"


class TestDifferentUtterancesStayDifferent:
    """違う発話が正規化で同じ文字列になり、誤りが 0 に見えてはいけない (レビュー F1)。

    正解にだけ印がある場合も含む: 印の直後の語を消すと、その語を落とした出力と一致してしまう。
    """

    @pytest.mark.parametrize(
        ("reference", "hypothesis"),
        [
            ("午前10:30に集合", "午後10:30に集合"),
            ("彼は(音楽理論)を学ぶ", "彼はを学ぶ"),
            ("注意:足元に気をつけて", "足元に気をつけて"),
            (">> 注意:足元に気をつけて", "足元に気をつけて"),
            ("- 注意:足元に気をつけて", "足元に気をつけて"),
            (">> - 注意:足元に気をつけて", "足元に気をつけて"),
        ],
    )
    def test_japanese(self, reference: str, hypothesis: str) -> None:
        assert calculate_cer(reference, hypothesis, lang="ja") > 0

    @pytest.mark.parametrize("reference", ["Answer: forty two", ">> Answer: forty two", "- Answer: forty two"])
    def test_english(self, reference: str) -> None:
        assert calculate_wer(reference, "forty two", lang="en") > 0


class TestMarkedCaptionsScoreAsCorrect:
    """書式つきの字幕と、正しい ASR の出力の誤りは 0 (レビュー F2)。"""

    @pytest.mark.parametrize(
        "reference",
        [">> こんにちは", "- こんにちは", ">> - こんにちは", "- >> こんにちは", "(male) こんにちは", "[音楽] こんにちは♪", "- こん\n- にちは"],
    )
    def test_japanese(self, reference: str) -> None:
        assert calculate_cer(reference, "こんにちは", lang="ja") == 0

    @pytest.mark.parametrize(
        "reference",
        ["&gt;&gt; Hello there.", "- Hello there", "&gt;&gt; - Hello there", "(male) Hello there (laughs)", "[Tony] Hello\n&gt;&gt; there"],
    )
    def test_english(self, reference: str) -> None:
        assert calculate_wer(reference, "hello there", lang="en") == 0

    def test_known_speaker_names(self) -> None:
        assert normalize_ja(">> - トミー：こんにちは", speaker_labels=["トミー"]) == "こんにちは"
        assert normalize_en("&gt;&gt; HEX: Hello there", speaker_labels=["HEX"]) == "hello there"
        assert normalize_text("Tom: Guten Tag", lang="de", speaker_labels=["Tom"]) == "guten tag"


class TestJapanese:
    def test_strips_punctuation_symbols_and_spaces(self) -> None:
        assert normalize_ja("水を マレーシアから 買わなくては ならないのです。") == "水をマレーシアから買わなくてはならないのです"
        assert normalize_ja("「は〜い」、今は【4K】で撮ってます！") == "はい今は4kで撮ってます"

    def test_width_and_case_do_not_count_as_errors(self) -> None:
        assert normalize_ja("ＮＨＫ") == normalize_ja("nhk") == "nhk"

    def test_numbers_are_not_converted(self) -> None:
        """漢数字と算用数字の揺れは誤りに数える (#488 の規則)。"""
        assert normalize_ja("三つ目") != normalize_ja("3つ目")

    def test_caption_and_asr_output_agree_after_normalization(self) -> None:
        caption = "[音楽] >> 今日は(笑)いい天気ですね。"
        asr_output = "今日はいい天気ですね"
        assert normalize_ja(caption) == normalize_ja(asr_output)


class TestEnglish:
    """Whisper 標準の英語の正規化 (transformers の EnglishTextNormalizer)。"""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("STUFF IT INTO YOU HIS BELLY COUNSELLED HIM", "stuff it into you his belly counseled him"),
            ("Stuff it, into you -- his belly counselled him.", "stuff it into you his belly counseled him"),
            ("&gt;&gt; Hi. Welcome to module 4.", "hi welcome to module 4"),
            ("[Music] (male) Hello, let's go.", "hello let us go"),
            ("Um, I can't pay $20, Mr. Smith.", "i can not pay $20 mister smith"),
            ("- I'm gonna go (laughs)", "i am going to go"),
        ],
    )
    def test_whisper_normalization(self, text: str, expected: str) -> None:
        assert normalize_en(text) == expected

    def test_spelled_and_digit_numbers_match(self) -> None:
        assert normalize_en("twenty five hundred dollars") == normalize_en("$2,500")

    def test_british_and_american_spellings_match(self) -> None:
        assert normalize_en("The colour of the theatre") == normalize_en("the color of the theater")


def test_dispatch() -> None:
    assert normalize_text("HELLO WORLD", lang="en") == "hello world"
    assert normalize_text("水を買う。", lang="ja") == "水を買う"
    assert normalize_text("Guten Tag, Herr Müller! ♪", lang="de") == "guten tag herr müller"


class TestBundledSpellingMap:
    """同梱した Whisper の綴りの対応表が、wheel にも入る。"""

    def test_map_and_its_license_are_bundled(self) -> None:
        mapping = json.loads((DATA / "whisper_english_spelling.json").read_text(encoding="utf-8"))
        assert mapping["colour"] == "color"
        assert "Copyright (c) 2022 OpenAI" in (DATA / "LICENSE.whisper").read_text(encoding="utf-8")

    def test_package_data_ships_them_in_the_wheel(self) -> None:
        config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
        shipped = config["tool"]["setuptools"]["package-data"]["benchmarks.common"]
        assert sorted(shipped) == ["data/LICENSE.whisper", "data/whisper_english_spelling.json"]
