"""VAD 最適化の study は評価条件ごとに分け、report / DB に条件を残す (#470 / PR #480 レビュー F1・F2)。

1 つの study の試行は同じ条件で採点したものでなければ比べられない。以前は study 名が
``{vad_type}_{language}`` だけで ``load_if_exists=True`` だったので、同じ保存先で ``--objective raw``
から ``production`` へ切り替えると旧 raw の試行が残り、その最良値が ``objective=production`` の
結果・preset として出力されていた (TPE も別の目的関数の履歴を学習していた)。

VAD / ASR は目的関数ごと fake に置き換える (raw は 0.0、production は 1.0 を返す)。Optuna と
SQLite は実物。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

optuna = pytest.importorskip("optuna", reason="optuna not installed")

from benchmarks.optimization import vad_optimizer
from benchmarks.optimization.study_conditions import STUDY_CONDITIONS_ATTR, study_name_for
from benchmarks.optimization.vad_optimizer import VADOptimizer
from benchmarks.optimization.visualization import ComparisonCSVExporter, generate_report

#: fake の目的関数が返す score (raw と production で違う値にして混在を見分ける)
SCORES = {"raw": 0.0, "production": 1.0}


class _FakeObjective:
    def __init__(self, *, objective: str, **_kwargs) -> None:
        self.objective = objective

    def __call__(self, trial) -> float:
        # preset の schema が要求する vad_config を揃える (build_preset_dict を通すため)
        trial.suggest_float("vad_config_threshold", 0.1, 0.9)
        for name in ("min_speech_ms", "min_silence_ms", "speech_pad_ms"):
            trial.suggest_int(f"vad_config_{name}", 50, 500)
        return SCORES[self.objective]


@pytest.fixture(autouse=True)
def _fake_objective(monkeypatch: pytest.MonkeyPatch):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    monkeypatch.setattr(vad_optimizer, "VADObjective", _FakeObjective)


@pytest.fixture
def storage(tmp_path: Path) -> str:
    return f"sqlite:///{(tmp_path / 'studies.db').as_posix()}"


def _optimizer(objective: str = "production", **overrides) -> VADOptimizer:
    options = {"vad_type": "silero", "language": "en", "engine_id": "parakeet", "mode": "quick", **overrides}
    optimizer = VADOptimizer(objective=objective, **options)
    optimizer._engine = object()  # 実 engine は読まない
    optimizer._dataset = []
    return optimizer


def _run(optimizer: VADOptimizer, storage: str, tmp_path: Path, n_trials: int = 1):
    return optimizer.optimize(n_trials=n_trials, storage=storage, output_dir=tmp_path)


class TestStudiesAreSeparatedByConditions:
    def test_raw_then_production_in_the_same_storage_do_not_mix(self, storage, tmp_path):
        """レビュー F1 の再現手順。以前は production の結果が raw の試行 (0.0) を最良値として返していた。"""
        raw = _run(_optimizer("raw"), storage, tmp_path)
        production = _run(_optimizer("production"), storage, tmp_path)

        assert (raw.best_score, raw.n_trials, raw.objective) == (0.0, 1, "raw")
        assert (production.best_score, production.n_trials, production.objective) == (1.0, 1, "production")
        assert raw.study_name != production.study_name
        preset = production.build_preset_dict()["metadata"]
        assert (preset["score"], preset["trials"], preset["objective"]) == (1.0, 1, "production")

    def test_same_conditions_resume_the_study(self, storage, tmp_path):
        _run(_optimizer(), storage, tmp_path)
        resumed = _run(_optimizer(), storage, tmp_path, n_trials=2)

        assert resumed.n_trials == 3, "同じ条件なら試行を足す (resume は維持)"

    def test_legacy_study_is_not_reused(self, storage, tmp_path):
        """#480 より前の study (名前 ``{vad}_{lang}``、条件の記録なし) に production の試行を足さない。"""
        legacy = optuna.create_study(study_name="silero_en", storage=storage, direction="minimize")
        legacy.optimize(lambda trial: trial.suggest_float("vad_config_threshold", 0.1, 0.9) * 0.0, n_trials=1)

        result = _run(_optimizer("production"), storage, tmp_path)

        assert (result.best_score, result.n_trials) == (1.0, 1)
        assert len(optuna.load_study(study_name="silero_en", storage=storage).trials) == 1, "旧 study は触らない"

    @pytest.mark.parametrize(
        ("key", "value"),
        [("vad_type", "webrtc"), ("language", "ja"), ("engine_id", "canary"), ("mode", "standard"), ("objective", "raw")],
    )
    def test_every_condition_is_part_of_the_name(self, key, value):
        base = _optimizer()
        changed = _optimizer(**{"objective": base.objective, **{key: value}}) if key != "objective" else _optimizer(value)
        assert base.study_name != changed.study_name

    def test_name_lists_the_conditions(self):
        assert _optimizer(engine_id="qwen3asr_large", mode="standard").study_name == (
            "silero_en_qwen3asr_large_standard_production"
        )


class TestExistingStudyIsCheckedBeforeAddingTrials:
    def test_conditions_are_recorded_in_the_storage(self, storage, tmp_path):
        optimizer = _optimizer()
        _run(optimizer, storage, tmp_path)

        stored = optuna.load_study(study_name=optimizer.study_name, storage=storage).user_attrs
        assert stored[STUDY_CONDITIONS_ATTR] == optimizer.study_conditions

    def test_unrecorded_study_with_trials_fails_before_loading_the_engine(self, storage, tmp_path):
        optimizer = _optimizer()
        foreign = optuna.create_study(study_name=optimizer.study_name, storage=storage, direction="minimize")
        foreign.optimize(lambda trial: 0.0, n_trials=1)

        optimizer._engine = None

        def _no_engine():
            raise AssertionError("条件の照合より前に engine を読んだ")

        optimizer._load_engine = _no_engine

        with pytest.raises(ValueError, match="評価条件が違う"):
            _run(optimizer, storage, tmp_path)
        assert len(optuna.load_study(study_name=optimizer.study_name, storage=storage).trials) == 1

    def test_study_with_other_recorded_conditions_fails(self, storage, tmp_path):
        optimizer = _optimizer()
        study = optuna.create_study(study_name=optimizer.study_name, storage=storage, direction="minimize")
        study.set_user_attr(STUDY_CONDITIONS_ATTR, {**optimizer.study_conditions, "objective": "raw"})

        with pytest.raises(ValueError, match="評価条件が違う"):
            _run(optimizer, storage, tmp_path)

    def test_empty_unrecorded_study_is_adopted(self, storage, tmp_path):
        """作っただけで試行の無い study (前回が試行前に落ちた等) は条件を記録して使う。"""
        optimizer = _optimizer()
        optuna.create_study(study_name=optimizer.study_name, storage=storage, direction="minimize")

        assert _run(optimizer, storage, tmp_path).n_trials == 1


class TestReportsCarryTheConditions:
    @pytest.mark.parametrize("objective", ["raw", "production"])
    def test_json_html_and_step_summary(self, storage, tmp_path, monkeypatch, objective):
        """レビュー F2: ``--report`` の成果物から評価モードを判別できる。"""
        summary = tmp_path / "summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
        result = _run(_optimizer(objective, mode="standard"), storage, tmp_path)

        paths = result.generate_reports(output_dir=tmp_path)

        data = json.loads(paths.json.read_text(encoding="utf-8"))
        assert (data["objective"], data["engine_id"], data["mode"]) == (objective, "parakeet", "standard")
        assert (data["vad_type"], data["language"]) == ("silero", "en")
        assert f"<td>Objective</td><td>{objective}</td>" in paths.html.read_text(encoding="utf-8")
        assert f"| Objective | {objective} |" in summary.read_text(encoding="utf-8")

    def test_legacy_study_reports_unrecorded_conditions(self, tmp_path):
        legacy = optuna.create_study(study_name="silero_ja", direction="minimize")
        legacy.optimize(lambda trial: trial.suggest_float("vad_config_threshold", 0.1, 0.9), n_trials=1)

        paths = generate_report(legacy, tmp_path)

        data = json.loads(paths.json.read_text(encoding="utf-8"))
        assert (data["vad_type"], data["language"]) == ("silero", "ja"), "旧 study は名前から推す"
        assert data["objective"] is None and data["engine_id"] is None
        assert "<td>Objective</td><td>(not recorded)</td>" in paths.html.read_text(encoding="utf-8")

    def test_generate_report_takes_vad_and_language_from_the_conditions(self, storage, tmp_path):
        """記録のある study の名前は engine 等を含む — 名前を分割すると言語を取り違える。"""
        result = _run(_optimizer(language="ja", engine_id="qwen3asr_large"), storage, tmp_path)

        paths = generate_report(result.study, tmp_path / "standalone")

        data = json.loads(paths.json.read_text(encoding="utf-8"))
        assert (data["vad_type"], data["language"]) == ("silero", "ja")
        assert "best_cer" in data

    def test_comparison_csv_has_the_conditions(self, storage, tmp_path):
        raw = _run(_optimizer("raw"), storage, tmp_path)
        production = _run(_optimizer("production"), storage, tmp_path)
        exporter = ComparisonCSVExporter(tmp_path / "comparison.csv")
        for result in (raw, production):
            exporter.add_study(result.study, vad_type="silero", language="en")

        lines = exporter.save().read_text(encoding="utf-8").splitlines()

        assert lines[0].startswith("vad_type,language,engine_id,mode,objective,study_name")
        assert [line.split(",")[4] for line in lines[1:]] == ["raw", "production"]


def test_study_name_for_follows_the_condition_order():
    conditions = {"objective": "raw", "mode": "quick", "engine_id": "e", "language": "ja", "vad_type": "webrtc"}
    assert study_name_for(conditions) == "webrtc_ja_e_quick_raw"
