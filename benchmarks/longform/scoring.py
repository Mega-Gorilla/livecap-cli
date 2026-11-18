"""長尺の評価の採点 (#488)。

1 本の動画の書き起こし (VAD で区切った区間ごとの文) を、字幕と比べる。

- 正解は字幕の全区間 (:meth:`~benchmarks.common.yodas2.Yodas2Video.reference_parts`)、ASR の出力は全区間を
  つなげた文。誤りは :func:`~benchmarks.common.error_breakdown.count_errors_in_parts` で数える
- **採点から外す字幕の区間** (最終評価用の 3 区間、表記を除く条件ではさらに 19 区間) は、ASR の出力を時刻で
  切らずに全体どうしを突き合わせ、外す区間に当たった誤りだけを数えない。ASR の区間の切り方 (VAD の区間の
  長さ) に結果が依らないので、区間の長さが違う ASR や VAD の設定を同じ物差しで比べられる。外す区間の境目に
  ついた挿入も数えないので、その分だけ近似である
- 全体の値は件数を足して出し (誤りの総数 ÷ 正解の総文字数)、話者ごとの値の平均も並べる

主の値は :data:`MAIN_CONDITION` (片方にしか無い口ぐせを数えない、表記だけ違う区間を除く)。人が付けた
字幕は口ぐせを省くのが普通で、そのまま比べると口ぐせを落とす ASR が有利になるため
(``docs/guides/benchmark/yodas2-longform-dataset.md`` 5.6)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from benchmarks.common.error_breakdown import VARIANTS, ErrorCounts, aggregate, count_errors_in_parts
from benchmarks.common.yodas2 import Yodas2Video

__all__ = [
    "CONDITIONS",
    "MAIN_CONDITION",
    "EngineScore",
    "ScoreCondition",
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
        hyp = " ".join(text for _, _, text in hypotheses[video.audio_id] if text)
        parts = video.reference_parts(include_notation=condition.include_notation)
        counts = count_errors_in_parts(parts, hyp, lang="ja", variant=condition.variant)
        by_video[video.audio_id] = counts
        per_speaker.setdefault(video.speaker, []).append(counts)
    by_speaker = {spk: aggregate(cs) for spk, cs in per_speaker.items()}
    return EngineScore(condition, aggregate(by_video.values()), by_speaker, by_video)
