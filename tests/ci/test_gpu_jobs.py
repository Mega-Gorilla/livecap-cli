"""Integration Tests の GPU job を Windows と Linux の self-hosted runner で分けて流す。

self-hosted の GPU runner が Windows 1 台だけだった頃は、PR ごとの GPU job (約 28 分) が順番待ちに
なっていた (2026-11-10 実測)。PR では OS に依らない engine smoke を Linux (Ubuntu-Desktop、12 GB)
へ移し、Windows は Voxtral (16 GB 要) と Windows 固有の確認だけにする。Linux の job は repo 変数で
止められ、止めている間は Windows が全部を流す。warm と「PASSED の要求」は両 job で共有する。
"""

from __future__ import annotations

import fnmatch
import importlib.util
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / ".github" / "scripts"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "integration-tests.yml"
LINUX_VAR = "vars.LIVECAP_ENABLE_LINUX_GPU_SMOKE == '1'"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


warm = _load("warm_engine_caches")
required = _load("run_required_tests")


@pytest.fixture(scope="module")
def jobs():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]


def _step(job: dict, name: str) -> dict:
    return next(step for step in job["steps"] if step.get("name") == name)


class TestWarmEngineCaches:
    def test_all_is_every_target_in_order(self):
        assert warm.resolve_targets(["all"]) == list(warm.TARGETS)

    @pytest.mark.parametrize("names", [[], ["nope"], ["parakeet", "nope"]])
    def test_unknown_or_missing_targets_are_rejected(self, names):
        with pytest.raises(ValueError):
            warm.resolve_targets(names)

    def test_int8_reazonspeech_is_warmed(self):
        """#409 の cache identity は int8 / float32 の両方を要求する。int8 が無いと skip してゲートが効かない。"""
        assert warm.TARGETS["reazonspeech-int8"] == {"engine_type": "reazonspeech", "use_int8": True}

    def test_unknown_target_exits_2(self):
        assert warm.main(["warm", "nope"]) == 2


class TestRunRequiredTests:
    def test_missing_passes_are_reported(self):
        output = "test_a PASSED\ntest_b SKIPPED (no model)\n"
        assert required.missing_passes(output, ["test_a", "test_b", "test_c"]) == ["test_b", "test_c"]

    def _fake_pytest(self, monkeypatch, *, code: int, stdout: str):
        def run(command, **kwargs):
            assert command[1:3] == ["-m", "pytest"]
            return subprocess.CompletedProcess(command, code, stdout=stdout, stderr="")

        monkeypatch.setattr(required.subprocess, "run", run)

    def test_all_required_passed(self, monkeypatch):
        _label, _args, tests = required.SUITES["reazonspeech-cache-identity"]
        self._fake_pytest(monkeypatch, code=0, stdout="\n".join(f"{t} PASSED" for t in tests))
        assert required.main(["run", "reazonspeech-cache-identity"]) == 0

    def test_green_exit_with_a_skip_still_fails(self, monkeypatch):
        """exit 0 でも、要求した test が PASSED していなければ落とす (skip が黙って緑にならない)。"""
        self._fake_pytest(monkeypatch, code=0, stdout="1 skipped")
        assert required.main(["run", "reazonspeech-cache-identity"]) == 1

    def test_pytest_failure_is_propagated(self, monkeypatch):
        self._fake_pytest(monkeypatch, code=3, stdout="")
        assert required.main(["run", "reazonspeech-cache-identity"]) == 3

    def test_unknown_suite(self):
        assert required.main(["run", "nope"]) == 2


class TestLinuxJob:
    def test_runs_only_when_both_variables_enable_it(self, jobs):
        """runner を止めている間 / 外したときは変数を 1 以外にする。job が作られなければ queue に残らない。"""
        condition = jobs["engine-smoke-gpu-linux"]["if"]
        assert "vars.LIVECAP_ENABLE_GPU_SMOKE == '1'" in condition
        assert LINUX_VAR in condition

    def test_targets_the_linux_gpu_runner(self, jobs):
        assert jobs["engine-smoke-gpu-linux"]["runs-on"] == ["self-hosted", "linux", "gpu"]

    def test_skips_voxtral(self, jobs):
        """Voxtral の smoke は min_vram_gb=16。12 GB では skip になり、LIVECAP_REQUIRE_ENGINE_SMOKE で落ちる。"""
        job = jobs["engine-smoke-gpu-linux"]
        smoke = _step(job, "Run engine smoke tests (GPU / self-hosted, Linux)")["run"]
        assert '-k "not voxtral"' in smoke
        warmed = _step(job, "Warm engine caches (GPU / self-hosted, Linux)")["run"].split()
        assert "voxtral" not in warmed
        assert set(warmed[2:]) == set(warm.TARGETS) - {"voxtral"}

    def test_builds_the_venv_from_the_lock(self, jobs):
        sync = _step(jobs["engine-smoke-gpu-linux"], "Sync dependencies (Linux)")
        assert sync["uses"] == "./.github/actions/setup-livecap-cuda-venv"

    def test_runs_the_reazonspeech_cache_identity_gate(self, jobs):
        step = _step(jobs["engine-smoke-gpu-linux"], "Run ReazonSpeech cache identity check (#409)")
        assert "run_required_tests.py reazonspeech-cache-identity" in step["run"]


class TestWindowsJob:
    def test_linux_takes_the_smoke_only_on_prs_with_linux_enabled(self, jobs):
        expression = jobs["engine-smoke-gpu"]["env"]["LIVECAP_LINUX_TAKES_SMOKE"]
        assert "github.event_name == 'pull_request'" in expression
        assert LINUX_VAR in expression

    def test_smoke_narrows_to_voxtral_only_when_linux_takes_the_rest(self, jobs):
        run = _step(jobs["engine-smoke-gpu"], "Run engine smoke tests (GPU / self-hosted, Windows)")["run"]
        assert "LIVECAP_LINUX_TAKES_SMOKE -eq 'true'" in run
        assert "'-k', 'voxtral'" in run
        pytest_line = next(line for line in run.splitlines() if "python -m pytest" in line)
        assert "@select" in pytest_line, "絞り込みを pytest に渡していない"

    def test_reazonspeech_gate_moves_to_linux_with_the_smoke(self, jobs):
        step = _step(jobs["engine-smoke-gpu"], "Run ReazonSpeech cache identity check (#409)")
        assert "env.LIVECAP_LINUX_TAKES_SMOKE != 'true'" in step["if"]
        assert "run_required_tests.py reazonspeech-cache-identity" in step["run"]

    def test_warms_every_target(self, jobs):
        run = _step(jobs["engine-smoke-gpu"], "Warm engine caches (GPU / self-hosted, Windows)")["run"]
        assert "warm_engine_caches.py all" in run

    @pytest.mark.parametrize(
        "name",
        [
            "Run Voxtral language contract check (#418)",
            "Run PyTorch runtime GPU checks (#422 kernel cache / #462 CUDA memory release)",
            "Run NeMo non-ASCII %TEMP% mitigation check (#379)",
        ],
    )
    def test_windows_only_checks_stay_unconditional(self, jobs, name):
        """Voxtral (16 GB) と Windows 固有の確認は、Linux の有無に関係なく Windows で回す。"""
        assert _step(jobs["engine-smoke-gpu"], name)["if"] == "runner.os == 'Windows'"


def test_scripts_are_in_the_pull_request_paths():
    triggers = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    paths = (triggers.get("on") or triggers[True])["pull_request"]["paths"]
    for script in SCRIPTS.glob("*.py"):
        rel = script.relative_to(REPO_ROOT).as_posix()
        assert any(fnmatch.fnmatch(rel, pattern) for pattern in paths), rel
