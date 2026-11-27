"""言語に合わせて、ASR の文をつなぐ (間に空白を入れるかどうか)。

ASR は、区間や部分ごとに前後の空白を取った文を返す。つなぐときに、語を空白で区切る言語かどうかで
間に空白を入れるかを決める。

- 字幕の結合 (:mod:`livecap_cli.transcription.result_coalescer`): :func:`separator_for` /
  :func:`join_texts`。空白で区切る言語の一覧 (:data:`SPACE_DELIMITED_LANGS`) にある言語だけ空白を入れる
- 1 つの区間を部分に分けて書き起こす engine の部分のつなぎ目 (WhisperS2T、#518): :func:`part_separator`。
  空白なしでつなぐ言語 (:data:`UNSPACED_LANGS`) 以外は空白を入れる。WhisperS2T は約 100 の言語に対応し、
  上の一覧に無い言語 (アラビア語・ヒンディー語など) も語を空白で区切るため
"""

from __future__ import annotations

from typing import Optional, Sequence

__all__ = ["SPACE_DELIMITED_LANGS", "UNSPACED_LANGS", "join_texts", "part_separator", "separator_for"]

#: 語を空白で区切らない言語 (ASR の部分のつなぎ目に空白を入れない)。日本語・中国語 (広東語を含む)・タイ語・
#: ラオ語・クメール語・ビルマ語・チベット語
UNSPACED_LANGS = frozenset({"ja", "zh", "yue", "th", "lo", "km", "my", "bo"})

#: 語を空白で区切る言語 (ISO 639-1 の 2 文字)。ここに無い言語は、言語が分かっていれば空白を入れない。
SPACE_DELIMITED_LANGS = frozenset(
    {
        "en",
        "fr",
        "de",
        "es",
        "it",
        "pt",
        "nl",
        "ru",
        "ko",
        "pl",
        "sv",
        "da",
        "no",
        "fi",
        "cs",
        "hu",
        "ro",
        "tr",
        "vi",
        "id",
        "ms",
    }
)


def separator_for(texts: Sequence[str], language: Optional[str]) -> str:
    """``texts`` をつなぐときの区切り (``" "`` か ``""``)。

    ``language`` が :data:`SPACE_DELIMITED_LANGS` にあれば空白。``language`` が無ければ、どれかの文に空白が
    あるときだけ空白 (翻訳を使わず、言語が分からないときのため)。それ以外は区切らない。
    """
    if language and language[:2].lower() in SPACE_DELIMITED_LANGS:
        return " "
    if not language and any(" " in t for t in texts):
        return " "
    return ""


def join_texts(texts: Sequence[str], language: Optional[str]) -> str:
    """``texts`` を、:func:`separator_for` の区切りで順につなぐ。空の文は呼ぶ側で除く。"""
    return separator_for(texts, language).join(texts)


def part_separator(texts: Sequence[str], language: Optional[str]) -> str:
    """1 つの区間を部分に分けて書き起こしたときの、部分のつなぎ目 (``" "`` か ``""``、#518)。

    ``language`` (``"ar"``、``"zh-CN"`` のような BCP-47 でもよい) が :data:`UNSPACED_LANGS` にあれば区切らず、
    それ以外は空白。``language`` が無ければ、:func:`separator_for` と同じく文の中身で決める。
    """
    if not language:
        return separator_for(texts, None)
    base = language.replace("_", "-").split("-")[0].lower()
    return "" if base in UNSPACED_LANGS else " "
