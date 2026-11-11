"""Text normalization for benchmark evaluation.

採点の前に、正解と ASR の出力の両方へ同じ規則をかけ、表記の揺れで WER / CER が動かないようにする。
規則は #488 で決めたもので、長尺の字幕 (YODAS2) にも短文の読み上げ (JSUT / LibriSpeech) にも同じものを使う。

1. 共通 (:func:`strip_caption_markup`)
   - HTML 実体参照を元の文字に戻す (``&amp;`` → ``&``、``&gt;&gt;`` → ``>>``)
   - NFKC で文字幅をそろえる (``ＡＢＣ１２３`` → ``ABC123``、``（笑）`` → ``(笑)``)
   - 話していない音の表記を消す (``[音楽]`` ``[Music]`` ``♪`` ``(笑)`` ``(laughs)``)
   - CSJ と同じ ``(F えー)`` 形式のフィラーを消す (test の正解を直すときに付ける、#488)
   - 行頭の話者名 (``トミー：`` ``1号:``)、``>>``、行頭の ``-`` を消す
2. ja: Unicode の句読点 (P) と記号 (S) と空白を消し、英字を小文字にする。**数字は変換しない**
   (漢数字と算用数字の揺れは誤りに数える。ReazonSpeech などの日本語 ASR の評価と同じ)
3. en: Whisper の ``EnglishTextNormalizer`` (Open ASR Leaderboard と同じ)。小文字化、括弧の中身と
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
from typing import Callable

__all__ = ["normalize_en", "normalize_ja", "normalize_text", "strip_caption_markup"]

#: Whisper の英米の綴りの対応表 (openai/whisper ea1c266 の whisper/normalizers/english.json)
_SPELLING_MAP = Path(__file__).with_name("data") / "whisper_english_spelling.json"

# NFKC のあとにかけるので、全角の括弧・コロン・不等号は半角になっている
_BRACKET_TAG = re.compile(r"\[[^\]\n]{0,40}\]")
_MUSIC_NOTE = re.compile(r"[♩♪♫♬]")
_FILLER_TAG = re.compile(r"\(F [^()\n]{0,20}\)")
_SOUND_TAG = re.compile(
    r"\([^()\n]{0,12}?"
    r"(?:笑|拍手|音楽|BGM|歓声|ため息|咳|laugh\w*|music|applause|inaudible|cheer\w*|sigh\w*|cough\w*)"
    r"[^()\n]{0,12}?\)",
    re.IGNORECASE,
)
#: 行頭の話者名。空白を含まない 12 文字までで、数字だけのもの (``10:30`` の ``10``) は除く
_SPEAKER_LABEL = re.compile(r"^[ \t]*(?!\d+[ \t]*:)[^\s:]{1,12}[ \t]*:[ \t]*(?=\S)", re.MULTILINE)
_CHEVRON = re.compile(r">{2,}")
_DASH_LEAD = re.compile(r"^[ \t]*[-‐–—][ \t]+", re.MULTILINE)


def strip_caption_markup(text: str) -> str:
    """字幕の書式 (実体参照・文字幅・音の表記・フィラーの印・話者名) を、どの言語でも同じに取り除く。"""
    cleaned = unicodedata.normalize("NFKC", html.unescape(text or ""))
    for pattern in (_BRACKET_TAG, _MUSIC_NOTE, _FILLER_TAG, _SOUND_TAG, _SPEAKER_LABEL, _CHEVRON, _DASH_LEAD):
        cleaned = pattern.sub(" ", cleaned)
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


def normalize_en(text: str) -> str:
    """英語の書き起こしを WER 用にそろえる (字幕の書式を除いてから、Whisper の正規化をかける)。"""
    return _collapse_spaces(_whisper_english_normalizer()(strip_caption_markup(text)))


def normalize_ja(text: str) -> str:
    """日本語の書き起こしを CER 用にそろえる (字幕の書式・句読点・記号・空白を除き、英字を小文字にする)。"""
    cleaned = _drop_punctuation_and_symbols(strip_caption_markup(text)).lower()
    return re.sub(r"\s+", "", cleaned)


def _normalize_generic(text: str) -> str:
    """規則を決めていない言語 (de / fr / es など): 字幕の書式を除き、小文字にして句読点と記号を消す。"""
    return _collapse_spaces(_drop_punctuation_and_symbols(strip_caption_markup(text)).lower())


def normalize_text(text: str, *, lang: str) -> str:
    """Dispatch to language-specific normalization.

    Args:
        text: Input transcript
        lang: Language code ('en', 'ja', etc.)

    Returns:
        Normalized text
    """
    if lang == "en":
        return normalize_en(text)
    if lang == "ja":
        return normalize_ja(text)
    # Fallback: generic normalization for other languages (de, fr, es, etc.)
    return _normalize_generic(text)
