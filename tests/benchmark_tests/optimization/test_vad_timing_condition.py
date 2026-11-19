"""VAD の時間の設定の数え方の版を、Optuna の study の条件に入れる (#507、#508 レビュー F2)。

study は偽物を使う (照合は user_attrs / trials / study_name だけを見る)。

``benchmarks.optimization`` の ``__init__`` は optuna を import するので、CI (optimization の extra を
入れない) でも走るよう、条件の照合だけを持つ ``study_conditions.py`` はファイルから直接読む
(optuna を型注釈でしか使わない)。optimizer を作るテストだけ optuna を要る。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[3] / "benchmarks" / "optimization" / "study_conditions.py"
_spec = importlib.util.spec_from_file_location("_study_conditions_under_test", _PATH)
_sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sc)

CONDITION_KEYS = _sc.CONDITION_KEYS
STUDY_CONDITIONS_ATTR = _sc.STUDY_CONDITIONS_ATTR
VAD_TIMING = _sc.VAD_TIMING
bind_study_conditions = _sc.bind_study_conditions
study_name_for = _sc.study_name_for


class FakeStudy:
    def __init__(self, name: str, conditions=None, trials: int = 0) -> None:
        self.study_name = name
        self.user_attrs = {} if conditions is None else {STUDY_CONDITIONS_ATTR: dict(conditions)}
        self.trials = [object()] * trials

    def set_user_attr(self, key, value) -> None:
        self.user_attrs[key] = value


CURRENT = {
    "vad_type": "tenvad", "language": "ja", "engine_id": "qwen3asr", "mode": "standard",
    "objective": "production", "vad_timing": VAD_TIMING,
}


def test_the_timing_version_is_a_condition_and_part_of_the_name():
    assert "vad_timing" in CONDITION_KEYS and VAD_TIMING == "frame_ms"
    assert study_name_for(CURRENT) == "tenvad_ja_qwen3asr_standard_production_frame_ms"


def test_a_study_from_before_the_timing_fix_is_not_reused():
    """32 ms 換算の試行 (版の記録が無い) を持つ study は、同じ名前でも再利用しない。"""
    before = {k: v for k, v in CURRENT.items() if k != "vad_timing"}
    with pytest.raises(ValueError, match="評価条件が違う"):
        bind_study_conditions(FakeStudy(study_name_for(CURRENT), before, trials=100), CURRENT)


def test_a_new_study_records_the_timing_version():
    study = FakeStudy(study_name_for(CURRENT))
    bind_study_conditions(study, CURRENT)
    assert study.user_attrs[STUDY_CONDITIONS_ATTR]["vad_timing"] == "frame_ms"


def test_the_optimizer_conditions_carry_the_version():
    pytest.importorskip("optuna")  # benchmarks.optimization の import に要る
    pytest.importorskip("jiwer")
    from benchmarks.optimization.vad_optimizer import VADOptimizer

    optimizer = VADOptimizer(vad_type="tenvad", language="ja", engine_id="qwen3asr")
    assert optimizer.study_conditions["vad_timing"] == VAD_TIMING
    assert optimizer.study_name.endswith("_frame_ms")
