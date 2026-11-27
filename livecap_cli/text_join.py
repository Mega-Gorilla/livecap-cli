"""言語に合わせて、ASR の文をつなぐ (間に空白を入れるかどうか)。

ASR は、区間や部分ごとに前後の空白を取った文を返す。つなぐときに、語を空白で区切る言語かどうかで
間に空白を入れるかを決める。字幕の結合 (:mod:`livecap_cli.transcription.result_coalescer`) と、
1 つの区間を部分に分けて書き起こす engine (WhisperS2T、#518) が同じ規則を使う。
"""

from __future__ import annotations

from typing import Optional, Sequence

__all__ = ["SPACE_DELIMITED_LANGS", "join_texts"]

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


def join_texts(texts: Sequence[str], language: Optional[str]) -> str:
    """``texts`` を順につなぐ。

    ``language`` が :data:`SPACE_DELIMITED_LANGS` にあれば空白を入れる。``language`` が無ければ、
    どれかの文に空白があるときだけ入れる (翻訳を使わず、言語が分からないときのため)。それ以外は
    そのままつなぐ。空の文は呼ぶ側で除く。
    """
    if language and language[:2].lower() in SPACE_DELIMITED_LANGS:
        return " ".join(texts)
    if not language and any(" " in t for t in texts):
        return " ".join(texts)
    return "".join(texts)
