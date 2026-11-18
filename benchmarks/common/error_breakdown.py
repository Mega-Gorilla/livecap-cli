"""誤りの内訳と、主の値に並べて出す値 (#488)。

:func:`~benchmarks.common.metrics.calculate_cer` は 1 つの割合しか返さない。長尺の字幕では、それだけでは
「正解の字幕が口ぐせを省いた」のか「ASR が言葉を落とした」のかが分からない。ここでは誤りを
置換・脱落・挿入の件数で数え、同じ文を次の 4 通りに比べる (``docs/guides/benchmark/yodas2-longform-dataset.md`` 5.6)。

==========  ==============================================================================
variant     何をそろえてから比べるか
==========  ==============================================================================
``plain``   採点の正規化だけ (``calculate_cer`` / ``calculate_wer`` と同じ値になる)
``fillers`` ``plain`` に加えて、**片方にしか無い口ぐせ** (えー、えっと、あのー、まあ など) を数えない
``digits``  ``plain`` に加えて、日本語の中の算用数字を漢数字に直す (``3つ`` と ``三つ``)
``kana``    ``digits`` に加えて、読み (ひらがな) に直す (``出来る`` と ``できる``)
==========  ==============================================================================

``fillers`` は、口ぐせの語を文から消すのではなく、**字幕と ASR を突き合わせたあとで**、片方にしか無い部分が
口ぐせだけでできているときに、その部分を数えない。「あの」「まあ」は口ぐせにも普通の言葉 (「あの人」) にもなるが、
両方にある「あの」は一致として残り、片方にしか無い「あの」だけが外れる。形態素解析は要らない。
人が付ける字幕は口ぐせを省くのが普通で、ASR は口ぐせを書くものと落とすものがある。``plain`` では口ぐせを
落とす ASR が有利になるので、長尺の字幕では ``fillers`` を主の値にする。

単位は、日本語が文字、それ以外が語。``fillers`` ``digits`` ``kana`` は日本語だけの区別で、ほかの言語では
``plain`` と同じ値になる (英語の口ぐせと数字は、採点の正規化が先にそろえている)。

``digits`` と ``kana`` は ``kanjize`` と ``pykakasi`` (どちらも dev の依存) を使う。使うときだけ読み込む。

正解の一部 (聞き取れない区間など) を採点から外すときは :func:`count_errors_in_parts` を使う。ASR の出力を
時刻で切らずに正解の全体と突き合わせ、外す区間に当たった誤りだけを数えない。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Sequence

from .text_normalization import normalize_text, strip_caption_markup

__all__ = [
    "VARIANTS",
    "ErrorCounts",
    "aggregate",
    "count_errors",
    "count_errors_in_parts",
    "error_breakdown",
    "is_filler_run",
    "macro_average",
]

#: :func:`count_errors` の ``variant`` に渡せる値
VARIANTS = ("plain", "fillers", "digits", "kana")

# 口ぐせ (採点の正規化のあとの形: 句読点は無く、カタカナはここでひらがなにそろえる)。
# 1 文字の「え」「あ」「ま」「ん」は入れない (ASR の 1 文字の誤りと区別できない)。「その」「こう」「ちょっと」も
# 入れない (片方にしか無くても、言葉として落ちた・足されたことが多い)。「そのー」のように伸ばしたものは入れる。
_FILLER = (
    r"(?:えーっと|えっとー*|えーと|えと|えー+|ええと|あのー+|あのう|あの|そのー+|まあ|まー+|なんか|"
    r"うーん+|んー+|あー+|うー+)"
)
_FILLER_RUN = re.compile(rf"{_FILLER}+")
_KATAKANA = {code: code - 0x60 for code in range(0x30A1, 0x30F7)}


@dataclass(frozen=True)
class ErrorCounts:
    """誤りの件数 (日本語は文字、それ以外は語)。足し合わせると、全体を合わせた値になる。"""

    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_length: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        """誤りの割合。正解が空のときは、出力も空なら 0、何か出していれば 1 (``calculate_cer`` と同じ)。"""
        if self.reference_length == 0:
            return 0.0 if self.insertions == 0 else 1.0
        return self.errors / self.reference_length

    def __add__(self, other: ErrorCounts) -> ErrorCounts:
        return ErrorCounts(
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
            self.reference_length + other.reference_length,
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "rate": self.rate,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "reference_length": self.reference_length,
        }


def is_filler_run(text: str) -> bool:
    """``text`` が口ぐせだけでできているか (「えっと」「あのーまあ」)。カタカナで書かれていても同じに扱う。"""
    return bool(text) and _FILLER_RUN.fullmatch(text.translate(_KATAKANA)) is not None


def _normalize_ja(text: str, *, variant: str, speaker_labels: Iterable[str]) -> str:
    if variant in ("plain", "fillers"):
        return normalize_text(text, lang="ja", speaker_labels=speaker_labels)
    # dev の依存 (kanjize / pykakasi) は、数字と読みをそろえるときだけ読み込む
    from benchmarks.confidence_calibration._normalize_jp import digits_to_kanji, to_hiragana

    # 話者名は ``名前:`` の形のうちに消す。読みに直したあとでは名前と一致しなくなる
    cleaned = digits_to_kanji(strip_caption_markup(text, speaker_labels=speaker_labels))
    if variant == "kana":
        cleaned = to_hiragana(cleaned)
    return normalize_text(cleaned, lang="ja")


def _is_unmatched_filler(units: list[str], start: int, end: int) -> bool:
    """``units[start:end]`` (片方にしか無い部分) が口ぐせか。

    文字単位の突き合わせは、同じ文字が並ぶと切れ目が 1〜2 文字ずれる (「あの|あの人」が「あ|のあ|の人」)。
    前後に同じ文字が続く分だけ窓を動かしても口ぐせになるなら、口ぐせとして扱う。
    """
    size = end - start
    if size == 0:
        return False
    if is_filler_run("".join(units[start:end])):
        return True
    for shift in range(1, size):
        if start - shift >= 0 and units[start - shift:start] == units[end - shift:end]:
            if is_filler_run("".join(units[start - shift:end - shift])):
                return True
        if end + shift <= len(units) and units[start:start + shift] == units[end:end + shift]:
            if is_filler_run("".join(units[start + shift:end + shift])):
                return True
    return False


def count_errors(
    reference: str,
    hypothesis: str,
    *,
    lang: str,
    variant: str = "plain",
    speaker_labels: Iterable[str] = (),
) -> ErrorCounts:
    """正解と ASR の出力を比べ、置換・脱落・挿入の件数を返す。

    Args:
        reference: 正解 (字幕)
        hypothesis: ASR の出力
        lang: 言語。``ja`` は文字単位、それ以外は語単位
        variant: :data:`VARIANTS` のどれか。日本語でなければ ``plain`` と同じ結果になる
        speaker_labels: その字幕の話者名 (:func:`~benchmarks.common.text_normalization.strip_caption_markup`)

    Raises:
        ValueError: ``variant`` が :data:`VARIANTS` に無い
        ImportError: jiwer が無い。``digits`` ``kana`` では kanjize / pykakasi が無い
    """
    return count_errors_in_parts([(reference, True)], hypothesis, lang=lang, variant=variant, speaker_labels=speaker_labels)


def count_errors_in_parts(
    reference_parts: Sequence[tuple[str, bool]],
    hypothesis: str,
    *,
    lang: str,
    variant: str = "plain",
    speaker_labels: Iterable[str] = (),
) -> ErrorCounts:
    """正解の一部を採点から外して数える :func:`count_errors`。

    正解は区間ごとの ``(文, 採点するか)``。ASR の出力は、外す区間も含めた正解の全体と区切らずに突き合わせ、
    外す区間に当たった誤りだけを数えない。

    - 置換・脱落: 外す区間の文字なら数えない。正解の長さにも入れない
    - 挿入: 前後どちらかの文字が外す区間なら数えない (外す区間の言葉が多めに出たものとみなす)

    ASR の出力を時刻で切って外すのと違い、ASR がどう区切ったか (VAD の区間の長さ) に結果が依らない。
    外す区間が無ければ :func:`count_errors` と同じ結果になる (区間ごとに正規化してつなぐ。字幕の区間の
    境目をまたぐ正規化は無い)。

    Raises:
        ValueError: ``variant`` が :data:`VARIANTS` に無い
        ImportError: jiwer が無い。``digits`` ``kana`` では kanjize / pykakasi が無い
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    import jiwer

    labels = tuple(speaker_labels)
    ref_units: list[str] = []
    scored: list[bool] = []
    for text, keep in reference_parts:
        units = _units(text, lang=lang, variant=variant, speaker_labels=labels)
        ref_units += units
        scored += [keep] * len(units)
    hyp_units = _units(hypothesis, lang=lang, variant=variant, speaker_labels=labels)
    sep = "" if lang == "ja" else " "
    ref, hyp = sep.join(ref_units), sep.join(hyp_units)

    forgive = lang == "ja" and variant == "fillers"
    if not ref_units:
        return ErrorCounts() if forgive and is_filler_run(hyp) else ErrorCounts(insertions=len(hyp_units))
    if hyp_units:
        out = jiwer.process_characters(ref, hyp) if lang == "ja" else jiwer.process_words(ref, hyp)
        chunks = [(c.type, c.ref_start_idx, c.ref_end_idx, c.hyp_start_idx, c.hyp_end_idx) for c in out.alignments[0]]
    else:  # 出力が空なら、正解の全体が 1 つの「片方にしか無い部分」になる
        chunks = [("delete", 0, len(ref_units), 0, 0)]

    subs = dels = ins = 0
    length = sum(scored)
    for kind, ref_start, ref_end, hyp_start, hyp_end in chunks:
        if kind == "substitute":
            subs += sum(scored[ref_start:ref_end])
        elif kind == "delete":
            for start, end in _scored_runs(scored, ref_start, ref_end):
                if forgive and _is_unmatched_filler(ref_units, start, end):
                    length -= end - start  # 字幕にだけある口ぐせは、正解の長さからも外す
                else:
                    dels += end - start
        elif kind == "insert":
            touches_unscored = (ref_start > 0 and not scored[ref_start - 1]) or (
                ref_start < len(ref_units) and not scored[ref_start]
            )
            if not touches_unscored and not (forgive and _is_unmatched_filler(hyp_units, hyp_start, hyp_end)):
                ins += hyp_end - hyp_start
    return ErrorCounts(subs, dels, ins, length)


def _units(text: str, *, lang: str, variant: str, speaker_labels: tuple[str, ...]) -> list[str]:
    """比べる単位 (日本語は文字、それ以外は語) に分ける。"""
    if lang == "ja":
        return list(_normalize_ja(text, variant=variant, speaker_labels=speaker_labels))
    return normalize_text(text, lang=lang, speaker_labels=speaker_labels).split()


def _scored_runs(scored: Sequence[bool], start: int, end: int) -> Iterator[tuple[int, int]]:
    """``scored[start:end]`` のうち、採点する単位が続くところ (開始, 終了) を順に返す。"""
    run_start = None
    for i in range(start, end):
        if scored[i] and run_start is None:
            run_start = i
        elif not scored[i] and run_start is not None:
            yield run_start, i
            run_start = None
    if run_start is not None:
        yield run_start, end


def error_breakdown(
    reference: str,
    hypothesis: str,
    *,
    lang: str,
    speaker_labels: Iterable[str] = (),
) -> dict[str, ErrorCounts]:
    """並べて出す値をまとめて数える。日本語は :data:`VARIANTS` の全部、それ以外は ``plain`` だけ。"""
    variants = VARIANTS if lang == "ja" else ("plain",)
    labels = tuple(speaker_labels)
    return {v: count_errors(reference, hypothesis, lang=lang, variant=v, speaker_labels=labels) for v in variants}


def aggregate(counts: Iterable[ErrorCounts]) -> ErrorCounts:
    """全体を合わせた値 (誤りの総数 ÷ 正解の総数) にするために、件数を足し合わせる。"""
    total = ErrorCounts()
    for item in counts:
        total = total + item
    return total


def macro_average(groups: Mapping[str, ErrorCounts]) -> float:
    """まとまり (話者など) ごとの割合の平均。量の多い話者に引きずられない値を、全体の値と並べて見るために使う。

    Raises:
        ValueError: ``groups`` が空
    """
    if not groups:
        raise ValueError("macro_average needs at least one group")
    return sum(c.rate for c in groups.values()) / len(groups)
