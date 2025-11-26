from __future__ import annotations

import pytest

from tests.utils.text_normalization import (
    get_language_from_path,
    normalize_en,
    normalize_ja,
    normalize_text,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("STUFF IT INTO YOU HIS BELLY COUNSELLED HIM", "stuff it into you his belly counselled him"),
        ("Stuff it, into you -- his belly counselled him.", "stuff it into you his belly counselled him"),
    ],
)
def test_normalize_en_basic(text: str, expected: str) -> None:
    assert normalize_en(text) == expected


def test_normalize_en_apostrophes_option() -> None:
    assert normalize_en("Can't stop, won't stop.") == "cant stop wont stop"
    assert normalize_en("Can't stop, won't stop.", keep_apostrophes=True) == "can t stop won t stop"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("水をマレーシアから買わなくてはならないのです。", "水をマレーシアから買わなくてはならないのです"),
        ("水を マレーシアから 買わなくては ならないのです。", "水をマレーシアから買わなくてはならないのです"),
    ],
)
def test_normalize_ja_basic(text: str, expected: str) -> None:
    assert normalize_ja(text) == expected


def test_normalize_ja_width_normalization() -> None:
    assert normalize_ja("ＡＢＣ１２３", normalize_width=True) == "ABC123"
    # default keeps width
    assert normalize_ja("ＡＢＣ１２３") == "ＡＢＣ１２３"


def test_normalize_dispatch() -> None:
    assert normalize_text("HELLO WORLD", lang="en") == "hello world"
    assert normalize_text("水をマレーシアから買わなくてはならないのです。", lang="ja") == "水をマレーシアから買わなくてはならないのです"
    with pytest.raises(ValueError):
        normalize_text("noop", lang="xx")


@pytest.mark.parametrize(
    ("file_path", "expected"),
    [
        ("tests/assets/audio/en/librispeech_1089-134686-0001.wav", "en"),
        ("tests/assets/audio/ja/jsut_basic5000_0001.txt", "ja"),
        ("tests/assets/audio/unknown_corpus_0001.wav", "unknown"),
        ("/path/to/audio/de/german_audio.wav", "de"),
    ],
)
def test_get_language_from_path(file_path: str, expected: str) -> None:
    assert get_language_from_path(file_path) == expected
