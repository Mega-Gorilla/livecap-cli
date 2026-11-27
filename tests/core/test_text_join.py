"""言語に合わせて文をつなぐ規則 (字幕の結合と WhisperS2T engine が共有する、#518)。"""

from __future__ import annotations

import pytest

from livecap_cli.text_join import SPACE_DELIMITED_LANGS, join_texts


@pytest.mark.parametrize("language", ["en", "en-US", "EN", "fr", "ko"])
def test_space_delimited_languages_get_a_space(language):
    assert join_texts(["Hello", "world"], language) == "Hello world"


@pytest.mark.parametrize("language", ["ja", "zh", "zh-CN"])
def test_other_known_languages_get_no_space(language):
    assert join_texts(["こんにちは", "世界"], language) == "こんにちは世界"


def test_without_a_language_a_space_is_used_only_when_a_text_has_one():
    assert join_texts(["Hello there", "world"], None) == "Hello there world"
    assert join_texts(["こんにちは", "世界"], None) == "こんにちは世界"
    assert join_texts(["Hello", "world"], "") == "Helloworld"


def test_more_than_two_texts_are_joined_in_order():
    assert join_texts(["a", "b", "c"], "en") == "a b c"


def test_the_language_set_is_two_letter_codes():
    assert all(len(code) == 2 and code == code.lower() for code in SPACE_DELIMITED_LANGS)
