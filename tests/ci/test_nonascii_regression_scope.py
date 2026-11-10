"""Integration Tests の非 ASCII 実モデル回帰 (#377) を、PR では関係する変更のときだけ回す。

GPU job (約 28 分) の約半分 (13 分) がこの回帰だった。self-hosted の GPU runner は 1 台で、
PR が続くと待ち行列になる。実モデル tier が見るのはネイティブライブラリの非 ASCII path の
扱いで、変えるのは依存と境界の helper — 呼び出し側の配線は cheap tier (Core Tests) が毎 PR
固定する。PR では依存 / ``tests/nonascii`` / ``livecap_cli/paths`` / ``.github/actions`` /
workflow を変えたときだけ回し、週次の schedule と手動実行では必ず回す。
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "nonascii_regression_scope.py"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "integration-tests.yml"


def _load():
    spec = importlib.util.spec_from_file_location("_nonascii_regression_scope", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scope = _load()


class TestDecide:
    @pytest.mark.parametrize("event", ["schedule", "workflow_dispatch", "push"])
    def test_non_pr_events_always_run(self, event):
        run, _reason = scope.decide(event, [])
        assert run is True

    @pytest.mark.parametrize(
        "changed",
        [
            ["pyproject.toml"],
            ["uv.lock"],
            [".github/workflows/integration-tests.yml"],
            [".github/scripts/nonascii_regression_scope.py"],
            ["tests/nonascii/registry.py"],
            ["tests/nonascii/probes/native_models.py"],
            ["livecap_cli/paths/staging.py"],
            [".github/actions/setup-livecap-cuda-venv/action.yml"],
            ["livecap_cli/engines/parakeet_engine.py", "uv.lock"],
        ],
    )
    def test_pr_touching_a_trigger_runs(self, changed):
        run, reason = scope.decide("pull_request", changed)
        assert run is True
        assert "the PR changes" in reason

    @pytest.mark.parametrize(
        "changed",
        [
            [],
            ["livecap_cli/engines/parakeet_engine.py", "livecap_cli/engines/model_store.py"],
            ["livecap_cli/cli.py", "livecap_cli/transcription/stream.py", "CHANGELOG.md"],
            ["benchmarks/optimization/vad_optimizer.py", "docs/testing/README.md"],
            # 前方一致を dir 単位で取る (``tests/nonascii_x`` / ``livecap_cli/paths_x.py`` は別物)
            ["tests/nonascii_other/test_x.py", "livecap_cli/paths_compat.py"],
            [".github/workflows/core-tests.yml"],
        ],
    )
    def test_pr_without_a_trigger_skips(self, changed):
        run, reason = scope.decide("pull_request", changed)
        assert run is False
        assert "weekly schedule" in reason

    def test_reason_lists_the_triggering_files(self):
        _run, reason = scope.decide("pull_request", ["uv.lock", "livecap_cli/cli.py", "tests/nonascii/paths.py"])
        assert "`uv.lock`" in reason and "`tests/nonascii/paths.py`" in reason
        assert "cli.py" not in reason


class TestMain:
    def test_writes_github_output_and_summary(self, tmp_path, monkeypatch, capsys):
        output = tmp_path / "output"
        summary = tmp_path / "summary"
        monkeypatch.setenv("GITHUB_OUTPUT", str(output))
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

        assert scope.main(["scope", "schedule"]) == 0

        assert output.read_text(encoding="utf-8") == "run=true\n"
        assert "Non-ASCII real-model regression (#377)" in summary.read_text(encoding="utf-8")
        assert "runs" in capsys.readouterr().out

    @staticmethod
    def _merge_commit_repo(root: Path, pr_files: list[str]) -> None:
        """base → (pr: pr_files を追加) と (main: 別の変更) → merge commit、の実 git repo を作る。"""

        def git(*args):
            subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)

        git("init", "-q", "-b", "main")
        git("config", "user.email", "ci@example.invalid")
        git("config", "user.name", "ci")
        # 既定の core.quotePath=true のまま (runner と同じ)。ここを false にするとテストが意味を失う
        git("config", "core.quotePath", "true")
        (root / "base.txt").write_text("base", encoding="utf-8")
        git("add", ".")
        git("commit", "-q", "-m", "base")
        git("checkout", "-q", "-b", "pr")
        for name in pr_files:
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text("pr", encoding="utf-8")
        git("add", ".")
        git("commit", "-q", "-m", "pr")
        git("checkout", "-q", "main")
        (root / "main-only.txt").write_text("main", encoding="utf-8")  # base 側だけの変更は PR の変更ではない
        git("add", ".")
        git("commit", "-q", "-m", "main moves")
        git("merge", "-q", "--no-ff", "-m", "merge", "pr")

    def test_pr_reads_the_merge_commit_against_its_first_parent(self, tmp_path, monkeypatch):
        """checkout は PR の merge commit。第 1 親 (base の先端) との差分が PR の変更になる。"""
        self._merge_commit_repo(tmp_path, ["uv.lock"])
        monkeypatch.chdir(tmp_path)

        assert scope.pr_changed_files() == ["uv.lock"]

    def test_non_ascii_paths_come_back_unquoted(self, tmp_path, monkeypatch):
        """``core.quotePath`` は非 ASCII のパスを ``"tests/nonascii/\\350..."`` と引用する。

        引用されたままだと trigger の前方一致に当たらず、必要な回帰を skip していた (PR #484 レビュー F1)。
        """
        changed = ["tests/nonascii/probes/追加.py", "docs/日本語 の 説明.md"]
        self._merge_commit_repo(tmp_path, changed)
        monkeypatch.chdir(tmp_path)

        files = scope.pr_changed_files()

        assert sorted(files) == sorted(changed)
        run, reason = scope.decide("pull_request", files)
        assert run is True
        assert "`tests/nonascii/probes/追加.py`" in reason

    def test_usage_error(self):
        assert scope.main(["scope"]) == 2


class TestWorkflowWiring:
    @pytest.fixture(scope="class")
    def workflow(self):
        return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

    def _steps(self, workflow):
        return workflow["jobs"]["engine-smoke-gpu"]["steps"]

    def test_checkout_fetches_the_merge_parent(self, workflow):
        checkout = self._steps(workflow)[0]
        assert checkout["uses"].startswith("actions/checkout@")
        assert checkout["with"]["fetch-depth"] == 2

    def test_regression_step_is_gated_by_the_scope_step(self, workflow):
        steps = self._steps(workflow)
        names = [step.get("name", "") for step in steps]
        scope_index = next(i for i, step in enumerate(steps) if step.get("id") == "nonascii-scope")
        regression_index = names.index("Run non-ASCII real-model regression (#377)")

        assert scope_index < regression_index
        assert "nonascii_regression_scope.py" in steps[scope_index]["run"]
        assert "github.event_name" in steps[scope_index]["run"]
        assert "steps.nonascii-scope.outputs.run == 'true'" in steps[regression_index]["if"]

    def test_weekly_schedule_still_runs_the_regression(self, workflow):
        """PR で回さない分を schedule が拾う。schedule を消すと回帰が週次でも走らなくなる。"""
        triggers = workflow.get("on", workflow.get(True))
        assert triggers["schedule"], "週次の schedule が要る"
        assert "workflow_dispatch" in triggers

    def test_script_change_triggers_the_workflow(self, workflow):
        triggers = workflow.get("on", workflow.get(True))
        assert ".github/scripts/nonascii_regression_scope.py" in triggers["pull_request"]["paths"]


def test_script_runs_standalone(tmp_path):
    """runner は venv の python で直接実行する (標準ライブラリだけで動く)。"""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "workflow_dispatch"], capture_output=True, text=True, check=False, cwd=tmp_path
    )
    assert result.returncode == 0, result.stderr
    assert "runs" in result.stdout
