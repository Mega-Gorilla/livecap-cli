"""Text normalization for benchmark evaluation.

採点の前に、正解と ASR の出力の両方へ同じ規則をかけ、表記の揺れで WER / CER が動かないようにする。
規則は #488 で決めたもので、長尺の字幕 (YODAS2) にも短文の読み上げ (JSUT / LibriSpeech) にも同じものを使う。

1. 共通 (:func:`strip_caption_markup`)
   - HTML 実体参照を元の文字に戻す (``&amp;`` → ``&``、``&gt;&gt;`` → ``>>``)
   - NFKC で文字幅をそろえる (``ＡＢＣ１２３`` → ``ABC123``、``（笑）`` → ``(笑)``)
   - 話していない音の表記を消す: ``[...]`` と ``♪``、括弧は**中身が既知の語だけ**のもの
     (``(笑)`` ``(laughs)`` ``(male)``)。``(音楽理論)`` のような普通の括弧は残す
   - CSJ と同じ ``(F えー)`` 形式のフィラーを消す (test の正解を直すときに付ける、#488)
   - 話者交代の印 (``>>``、行頭の ``-``) と、**その直後の**話者名 (``>> トミー：``) を消す
   - 印の無い行頭の話者名 (``トミー：``) は、``speaker_labels`` で名前を渡したときだけ消す。
     名前を推定で消すと ``午前10:30`` や ``Answer: ...`` のような発話まで消え、違う発話が
     同じ文字列になって誤りが 0 に見える (PR #493 のレビュー F1)
2. ja: Unicode の句読点 (P) と記号 (S) と空白を消し、英字を小文字にする。**数字は変換しない**
   (漢数字と算用数字の揺れは誤りに数える。ReazonSpeech などの日本語 ASR の評価と同じ)
3. en: Whisper 標準の英語の正規化 (transformers の ``EnglishTextNormalizer``)。小文字化、括弧の中身と
   フィラー (um / uh / hmm) の除去、縮約 (``can't`` → ``can not``)・数字 (``twenty`` → ``20``)・
   英米の綴り (``colour`` → ``color``) の統一。綴りの対応表は Whisper の ``english.json`` を
   ``data/`` に同梱している (MIT、``data/LICENSE.whisper``)
4. その他の言語: 1 のあと、小文字にし、句読点と記号を消す

This module is independent from tests/utils/text_normalization.py to keep
benchmarks self-contained.
"""

from __future__ import annotations

import html
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

__all__ = ["normalize_en", "normalize_ja", "normalize_text", "strip_caption_markup"]

#: Whisper の英米の綴りの対応表 (openai/whisper ea1c266 の whisper/normalizers/english.json)
_SPELLING_MAP = Path(__file__).with_name("data") / "whisper_english_spelling.json"

# NFKC のあとにかけるので、全角の括弧・コロン・不等号は半角になっている
_BRACKET_TAG = re.compile(r"\[[^\]\n]{0,40}\]")
_MUSIC_NOTE = re.compile(r"[♩♪♫♬]")
_FILLER_TAG = re.compile(r"\(F [^()\n]{0,20}\)")
#: 括弧の中身が、音の表記か話者の属性を表す既知の語だけのときに消す (語を含むだけでは消さない)
_PAREN_WORD = (
    r"(?:笑|笑い|拍手|音楽|BGM|歓声|ため息|咳|laughs?|laughter|laughing|music|applause|inaudible|"
    r"cheers?|cheering|sighs?|coughs?|male|female|man|woman|narrator|speaker|crosstalk)"
)
_PAREN_TAG = re.compile(rf"\([ \t]*{_PAREN_WORD}(?:[ \t]+{_PAREN_WORD})*[ \t]*\)", re.IGNORECASE)
#: 話者名の形: 空白・コロンを含まない 12 文字まで。数字で終わるもの、コロンの直後が数字のもの
#: (``10:30`` ``午前10:30``) は時刻なので除く
_LABEL = r"[^\s:>]{1,12}(?<!\d)[ \t]*:(?!\d)[ \t]*(?=\S)"
#: 話者交代の印 (行頭の ``-``、``>>``) と、その直後の話者名
_TURN = re.compile(rf"(?:^[ \t]*[-‐–—][ \t]+|>{{2,}}[ \t]*)+(?:{_LABEL})?", re.MULTILINE)


def _known_label_pattern(speaker_labels: Iterable[str]) -> re.Pattern[str] | None:
    names = sorted({unicodedata.normalize("NFKC", n).strip() for n in speaker_labels} - {""}, key=len, reverse=True)
    if not names:
        return None
    return re.compile(rf"^[ \t]*(?:{'|'.join(map(re.escape, names))})[ \t]*:[ \t]*", re.MULTILINE)


def strip_caption_markup(text: str, *, speaker_labels: Iterable[str] = ()) -> str:
    """字幕の書式 (実体参照・文字幅・音の表記・フィラーの印・話者交代の印) を、どの言語でも同じに取り除く。

    Args:
        text: 字幕か ASR の出力
        speaker_labels: その字幕の話者名 (``["トミー", "HEX"]``)。行頭の ``名前:`` を消す。
            渡さなければ、印 (``>>`` / 行頭の ``-``) の無い話者名は消さない
    """
    cleaned = unicodedata.normalize("NFKC", html.unescape(text or ""))
    for pattern in (_BRACKET_TAG, _MUSIC_NOTE, _FILLER_TAG, _PAREN_TAG, _TURN):
        cleaned = pattern.sub(" ", cleaned)
    known = _known_label_pattern(speaker_labels)
    if known is not None:
        # 印や音の表記を消したあとなので、``>> [拍手] トミー：`` の名前も行頭として扱える
        cleaned = known.sub(" ", cleaned)
    return cleaned


def _drop_punctuation_and_symbols(text: str) -> str:
    return "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)


def _collapse_spaces(text: str) -> str:
    """Collapse multiple whitespace characters into a single space."""
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def _whisper_english_normalizer() -> Callable[[str], str]:
    # transformers は必須の依存。import が重いので、英語を採点するときだけ読む
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    mapping = json.loads(_SPELLING_MAP.read_text(encoding="utf-8"))
    return EnglishTextNormalizer(mapping)


def normalize_en(text: str, *, speaker_labels: Iterable[str] = ()) -> str:
    """英語の書き起こしを WER 用にそろえる (字幕の書式を除いてから、Whisper の正規化をかける)。"""
    stripped = strip_caption_markup(text, speaker_labels=speaker_labels)
    return _collapse_spaces(_whisper_english_normalizer()(stripped))


def normalize_ja(text: str, *, speaker_labels: Iterable[str] = ()) -> str:
    """日本語の書き起こしを CER 用にそろえる (字幕の書式・句読点・記号・空白を除き、英字を小文字にする)。"""
    stripped = strip_caption_markup(text, speaker_labels=speaker_labels)
    return re.sub(r"\s+", "", _drop_punctuation_and_symbols(stripped).lower())


def _normalize_generic(text: str, *, speaker_labels: Iterable[str] = ()) -> str:
    """規則を決めていない言語 (de / fr / es など): 字幕の書式を除き、小文字にして句読点と記号を消す。"""
    stripped = strip_caption_markup(text, speaker_labels=speaker_labels)
    return _collapse_spaces(_drop_punctuation_and_symbols(stripped).lower())


def normalize_text(text: str, *, lang: str, speaker_labels: Iterable[str] = ()) -> str:
    """Dispatch to language-specific normalization.

    Args:
        text: Input transcript
        lang: Language code ('en', 'ja', etc.)
        speaker_labels: その字幕の話者名 (:func:`strip_caption_markup` を参照)

    Returns:
        Normalized text
    """
    if lang == "en":
        return normalize_en(text, speaker_labels=speaker_labels)
    if lang == "ja":
        return normalize_ja(text, speaker_labels=speaker_labels)
    # Fallback: generic normalization for other languages (de, fr, es, etc.)
    return _normalize_generic(text, speaker_labels=speaker_labels)
