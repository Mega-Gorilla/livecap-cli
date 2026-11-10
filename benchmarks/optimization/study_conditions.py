"""Optuna study の評価条件 (#470 / PR #480 レビュー F1・F2)。

1 つの study の試行は **同じ条件で採点したもの**でなければ比べられない。``best_value`` /
``best_params`` は全試行から選ばれ、TPE も全履歴から学習するので、条件の違う試行が混ざると
最良値を取り違える (例: 後段なしの ``raw`` で 0.0 だった試行を ``production`` の最良値として
preset に書く)。そこで:

- study 名に条件を入れ、条件が違えば別の study にする (同じ ``studies.db`` に並べられる)
- 条件を study 自身 (``user_attrs``) に記録し、既存の study は記録が一致するときだけ再利用する
  (名前は人が読むための識別で、条件の正本は study に持たせる — report / DB だけからでも
  評価モードを判別できる)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    import optuna

#: 試行を比べてよい条件 (study 名もこの順で作る)
CONDITION_KEYS = ("vad_type", "language", "engine_id", "mode", "objective")

#: 条件を記録する ``study.user_attrs`` の key
STUDY_CONDITIONS_ATTR = "conditions"


def study_name_for(conditions: Mapping[str, str]) -> str:
    """``silero_ja_qwen3asr_large_standard_production`` の形の study 名。"""
    return "_".join(conditions[key] for key in CONDITION_KEYS)


def bind_study_conditions(study: "optuna.Study", conditions: Mapping[str, str]) -> None:
    """新しい study に条件を記録し、既存の study は記録が ``conditions`` と一致するときだけ通す。

    試行を足す**前**に呼ぶ (engine の読み込みより前に失敗させる)。

    Raises:
        ValueError: 既存の study の記録が違う、または記録の無い試行がある (#480 より前に作られた
            study 等) — 条件の違う試行と混ざり得るので再利用しない。
    """
    expected = dict(conditions)
    stored = study.user_attrs.get(STUDY_CONDITIONS_ATTR)
    if stored == expected:
        return
    if stored is None and not study.trials:
        study.set_user_attr(STUDY_CONDITIONS_ATTR, expected)
        return
    recorded = "記録なし" if stored is None else stored
    raise ValueError(
        f"study {study.study_name!r} は評価条件が違う (記録: {recorded}、今回: {expected})。"
        "条件の違う試行と混ぜると最良値を取り違えるので再利用しない。"
        "別の --storage / --output-dir を使うか、この study を削除すること"
    )


def read_study_conditions(study: "optuna.Study") -> dict[str, str]:
    """study に記録された条件。記録の無い study (#480 より前) は空 dict。"""
    stored = study.user_attrs.get(STUDY_CONDITIONS_ATTR)
    return dict(stored) if isinstance(stored, Mapping) else {}
