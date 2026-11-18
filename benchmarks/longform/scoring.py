"""長尺の評価の採点 (#488)。

1 本の動画の書き起こし (VAD で区切った区間ごとの文) を、字幕と比べる。

- 正解は :meth:`~benchmarks.common.yodas2.Yodas2Video.reference` (採点から外す区間を除いた字幕)
- **採点から外す区間は、ASR の出力の側でも外す。** 区間の中央の時刻が、外す字幕の区間に入る ASR の
  区間を除く。字幕の時刻は人が確かめていないので、近似である (外すのは最終評価用の 3 区間、表記を
  除く条件ではさらに 19 区間)
- 誤りは :func:`~benchmarks.common.error_breakdown.count_errors` で数える。全体の値は件数を足して出し
  (誤りの総数 ÷ 正解の総文字数)、話者ごとの値の平均も並べる

主の値は :data:`MAIN_CONDITION` (片方にしか無い口ぐせを数えない、表記だけ違う区間を除く)。人が付けた
字幕は口ぐせを省くのが普通で、そのまま比べると口ぐせを落とす ASR が有利になるため
(``docs/guides/benchmark/yodas2-longform-dataset.md`` 5.6)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from benchmarks.common.error_breakdown import VARIANTS, ErrorCounts, aggregate, count_errors
from benchmarks.common.yodas2 import Yodas2Video

__all__ = [
    "CONDITIONS",
    "MAIN_CONDITION",
    "EngineScore",
    "ScoreCondition",
    "hypothesis_text",
    "score_engine",
]

#: ASR の区間: (開始秒, 終了秒, 文)
HypSegment = tuple[float, float, str]


@dataclass(frozen=True)
class ScoreCondition:
    """比べ方。``variant`` は :data:`~benchmarks.common.error_breakdown.VARIANTS` のどれか。"""

    variant: str
    include_notation: bool

    @property
    def key(self) -> str:
        return f"{self.variant}{'' if self.include_notation else '-notation_excluded'}"


#: 主の値: 片方にしか無い口ぐせを数えない、表記だけ違う区間 (コードや URL の読み上げなど) を除く
MAIN_CONDITION = ScoreCondition("fillers", include_notation=False)
#: 並べて出す比べ方 (主の値を先頭に)
CONDITIONS: tuple[ScoreCondition, ...] = (MAIN_CONDITION,) + tuple(
    ScoreCondition(v, n) for n in (True, False) for v in VARIANTS if (v, n) != ("fillers", False)
)


@dataclass(frozen=True)
class EngineScore:
    """1 つの ASR の結果。``pooled`` は全体を合わせた値、``by_speaker`` は話者ごとに合わせた値。"""

    condition: ScoreCondition
    pooled: ErrorCounts
    by_speaker: Mapping[str, ErrorCounts]
    by_video: Mapping[str, ErrorCounts]

    @property
    def speaker_mean(self) -> float:
        """話者ごとの割合の平均 (量の多い話者に引きずられていないかを見る)。"""
        return sum(c.rate for c in self.by_speaker.values()) / len(self.by_speaker)

    def to_dict(self) -> dict:
        return {
            "condition": self.condition.key,
            "pooled": self.pooled.to_dict(),
            "speaker_mean_rate": self.speaker_mean,
            "by_speaker": {k: v.to_dict() for k, v in sorted(self.by_speaker.items())},
            "by_video": {k: v.to_dict() for k, v in sorted(self.by_video.items())},
        }


def _inside(t: float, spans: Iterable[tuple[float, float]]) -> bool:
    return any(start <= t < end for start, end in spans)


def hypothesis_text(segments: Sequence[HypSegment], unscored_spans: Sequence[tuple[float, float]]) -> str:
    """ASR の区間をつなげる。区間の中央が採点から外す字幕の区間に入るものは除く。"""
    return " ".join(text for start, end, text in segments if text and not _inside((start + end) / 2, unscored_spans))


def score_engine(
    videos: Sequence[Yodas2Video],
    hypotheses: Mapping[str, Sequence[HypSegment]],
    condition: ScoreCondition = MAIN_CONDITION,
) -> EngineScore:
    """1 つの ASR を採点する。

    Raises:
        KeyError: ``videos`` のどれかの書き起こしが ``hypotheses`` に無い (欠けたまま平均すると、
            難しい動画を落とした値になるため、黙って飛ばさない)
        ValueError: ``videos`` が空
    """
    if not videos:
        raise ValueError("score_engine needs at least one video")
    by_video: dict[str, ErrorCounts] = {}
    per_speaker: dict[str, list[ErrorCounts]] = {}
    for video in videos:
        ref = video.reference(include_notation=condition.include_notation)
        hyp = hypothesis_text(hypotheses[video.audio_id], video.unscored_spans(include_notation=condition.include_notation))
        counts = count_errors(ref, hyp, lang="ja", variant=condition.variant)
        by_video[video.audio_id] = counts
        per_speaker.setdefault(video.speaker, []).append(counts)
    by_speaker = {spk: aggregate(cs) for spk, cs in per_speaker.items()}
    return EngineScore(condition, aggregate(by_video.values()), by_speaker, by_video)
