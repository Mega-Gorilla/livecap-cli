"""長尺の評価 (YODAS2 の日本語データ、#488)。

- :mod:`benchmarks.longform.runner`: CLI の ``transcribe`` (ファイル) と同じ部品で書き起こす
- :mod:`benchmarks.longform.scoring`: 採点 (誤りの内訳、口ぐせ・表記の扱い、話者ごとの平均)

使い方は ``python -m benchmarks.longform --help`` と
``docs/guides/benchmark/yodas2-longform-dataset.md`` を参照。
"""

from __future__ import annotations

from .scoring import MAIN_CONDITION, ScoreCondition, score_engine

__all__ = ["MAIN_CONDITION", "ScoreCondition", "score_engine"]
