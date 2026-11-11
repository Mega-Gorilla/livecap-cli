"""採点の正規化 (benchmarks/common/text_normalization.py) の規則 (#488)。"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

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
            ("トミー：は〜い", "は〜い"),
            ("1号：今は", "今は"),
            ("&gt;&gt; Hi.", "Hi."),
            ("- I see", "I see"),
            ("king&amp;prince", "king&prince"),
            ("ＡＢＣ１２３", "ABC123"),
        ],
    )
    def test_removes_caption_markup(self, text: str, expected: str) -> None:
        assert " ".join(strip_caption_markup(text).split()) == expected

    def test_keeps_a_time_at_the_start_of_a_line(self) -> None:
        """数字だけのもの (``10:30`` の ``10``) は話者名ではない。"""
        assert strip_caption_markup("10:30 に集合") == "10:30 に集合"

    def test_keeps_parentheses_that_are_not_sound_tags(self) -> None:
        assert strip_caption_markup("結（むすび）") == "結(むすび)"


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
        caption = "[音楽] トミー：今日は(笑)いい天気ですね。"
        asr_output = "今日はいい天気ですね"
        assert normalize_ja(caption) == normalize_ja(asr_output)


class TestEnglish:
    """Whisper の EnglishTextNormalizer (Open ASR Leaderboard と同じ)。"""

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
