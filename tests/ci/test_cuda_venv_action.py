"""``.github/actions/setup-livecap-cuda-venv`` の torch 版の読み取り (#470)。

action は ``uv sync --locked`` 後に、lock が入れた torch 一式と**同じ版**の CUDA build を
入れ直す。版の読み取りは ``torch_pins.py`` に置く — PowerShell の中で書いていた頃は、
未導入のパッケージ (例: torchvision を含まない extras) で Python が何も出力せずに終わると、
終了コードを見る前に ``$null.Trim()`` で止まった (PR #476 レビュー)。
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ACTION_DIR = Path(__file__).resolve().parents[2] / ".github" / "actions" / "setup-livecap-cuda-venv"


def _load_module():
    """Import by path: ``.github`` is not an importable package."""
    spec = importlib.util.spec_from_file_location("_torch_pins", ACTION_DIR / "torch_pins.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pins_mod = _load_module()


def _versions(**installed):
    def version_of(name):
        if name not in installed:
            raise importlib.metadata.PackageNotFoundError(name)
        return installed[name]

    return version_of


def test_pins_the_installed_family_without_the_local_version():
    pins = pins_mod.installed_pins(_versions(torch="2.9.1+cpu", torchaudio="2.9.1+cpu", torchvision="0.24.1+cpu"))

    assert pins == ["torch==2.9.1", "torchaudio==2.9.1", "torchvision==0.24.1"]


def test_skips_members_the_extras_did_not_install():
    """torchvision を入れない extras でも落ちずに、入っている分だけ返す (レビュー指摘のケース)。"""
    assert pins_mod.installed_pins(_versions(torch="2.9.1+cpu")) == ["torch==2.9.1"]


def test_main_fails_when_torch_itself_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(pins_mod, "installed_pins", lambda: ["torchaudio==2.9.1"])

    assert pins_mod.main() == pins_mod.EXIT_NO_TORCH
    assert capsys.readouterr().out == ""


def test_script_prints_one_pin_per_line_for_this_interpreter():
    """実際に subprocess で実行し、action が読む stdout の形 (1 行 1 pin) を固定する。"""
    try:
        importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("torch not installed in this environment")

    proc = subprocess.run(
        [sys.executable, str(ACTION_DIR / "torch_pins.py")], capture_output=True, text=True, check=False
    )

    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert lines[0].startswith("torch==") and "+" not in lines[0]
    assert all("==" in line for line in lines)


def _setup_module():
    spec = importlib.util.spec_from_file_location("_setup_cuda_venv", ACTION_DIR / "setup_cuda_venv.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup_mod = _setup_module()


class TestActionSteps:
    """Windows / Linux の step は shell を選ぶだけで、手順は setup_cuda_venv.py の 1 か所にある。"""

    @pytest.fixture(scope="class")
    def steps(self):
        return yaml.safe_load((ACTION_DIR / "action.yml").read_text(encoding="utf-8"))["runs"]["steps"]

    def test_one_step_per_os(self, steps):
        assert [(step["if"], step["shell"]) for step in steps] == [
            ("runner.os == 'Windows'", "pwsh"),
            ("runner.os != 'Windows'", "bash"),
        ]

    def test_both_run_the_shared_script_with_the_preinstalled_interpreter(self, steps):
        for step in steps:
            assert "setup_cuda_venv.py" in step["run"]
            assert "GITHUB_ACTION_PATH" in step["run"]
            assert "PYTHON_EXE" in step["run"]
            assert "uv sync" not in step["run"], "手順を shell 側へ戻さない (2 か所に分かれる)"
            assert ".Trim()" not in step["run"], "版の読み取りを PowerShell の文字列処理に戻さない"

    def test_windows_step_fails_on_a_nonzero_exit(self, steps):
        assert "$LASTEXITCODE" in steps[0]["run"]


class TestSetupCudaVenv:
    @pytest.mark.parametrize("text", ["", "   ", "Dev", "dev;rm", "../x", "-x"])
    def test_rejects_bad_extras(self, text):
        with pytest.raises(ValueError):
            setup_mod.parse_extras(text)

    def test_parses_space_separated_extras(self):
        assert setup_mod.parse_extras(" translation dev  engines-torch ") == ["translation", "dev", "engines-torch"]

    def test_installs_the_lock_without_re_resolving(self):
        assert setup_mod.sync_command(["dev", "engines-nemo"]) == [
            "uv", "sync", "--locked", "--extra", "dev", "--extra", "engines-nemo",
        ]

    def test_reinstalls_the_same_versions_from_the_cuda_index(self):
        """``torch==2.9.1`` は ``2.9.1+cpu`` で満たされるので ``--reinstall-package`` が要る。"""
        assert setup_mod.cuda_install_command(["torch==2.9.1", "torchaudio==2.9.1"], "https://idx") == [
            "uv", "pip", "install",
            "--reinstall-package", "torch", "--reinstall-package", "torchaudio",
            "torch==2.9.1", "torchaudio==2.9.1",
            "--index-url", "https://idx",
        ]

    @pytest.mark.parametrize(("windows", "bin_name", "python"), [(True, "Scripts", "python.exe"), (False, "bin", "python")])
    def test_venv_layout_per_os(self, tmp_path, windows, bin_name, python):
        assert setup_mod.venv_bin_dir(tmp_path, windows=windows) == tmp_path / bin_name
        assert setup_mod.venv_python(tmp_path, windows=windows) == tmp_path / bin_name / python

    def _fake_run(self, monkeypatch, *, pins="torch==2.9.1\n", fail=None):
        calls = []

        def run(command, **kwargs):
            calls.append(list(command))
            what = " ".join(str(c) for c in command)
            code = 1 if fail and fail in what else 0
            return subprocess.CompletedProcess(command, code, stdout=pins if "torch_pins.py" in what else "", stderr="")

        monkeypatch.setattr(setup_mod.subprocess, "run", run)
        return calls

    def test_setup_runs_the_steps_in_order_on_a_clean_venv(self, tmp_path, monkeypatch):
        stale = tmp_path / ".venv" / "stale"
        stale.mkdir(parents=True)
        calls = self._fake_run(monkeypatch)

        bin_dir = setup_mod.setup(["dev"], "https://idx", python_exe="/py3.12", workdir=tmp_path, windows=False)

        assert not stale.exists(), "前の job の venv を残さない"
        assert calls[0] == ["uv", "python", "pin", "/py3.12"]
        assert calls[1] == ["uv", "sync", "--locked", "--extra", "dev"]
        assert calls[2][1].endswith("torch_pins.py")
        assert calls[3] == ["uv", "pip", "install", "--reinstall-package", "torch", "torch==2.9.1", "--index-url", "https://idx"]
        assert calls[4][1] == "-c" and "cuda.is_available()" in calls[4][2]
        assert bin_dir == tmp_path / ".venv" / "bin"

    def test_cpu_only_env_is_an_error(self, tmp_path, monkeypatch):
        self._fake_run(monkeypatch, fail="cuda.is_available")
        with pytest.raises(setup_mod.SetupError, match="CUDA is not available"):
            setup_mod.setup(["dev"], "https://idx", python_exe="py", workdir=tmp_path, windows=False)

    def test_no_pins_is_an_error(self, tmp_path, monkeypatch):
        self._fake_run(monkeypatch, pins="")
        with pytest.raises(setup_mod.SetupError, match="torch versions"):
            setup_mod.setup(["dev"], "https://idx", python_exe="py", workdir=tmp_path, windows=False)

    def test_main_puts_the_venv_on_github_path(self, tmp_path, monkeypatch):
        github_path = tmp_path / "path"
        monkeypatch.setenv("GITHUB_PATH", str(github_path))
        monkeypatch.setenv("PYTHON_EXE", "py")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(setup_mod, "setup", lambda *a, **k: tmp_path / ".venv" / "bin")

        assert setup_mod.main(["setup", "dev", "https://idx"]) == 0
        assert github_path.read_text(encoding="utf-8") == f"{tmp_path / '.venv' / 'bin'}\n"

    def test_main_requires_the_preinstalled_interpreter(self, monkeypatch):
        monkeypatch.delenv("PYTHON_EXE", raising=False)
        assert setup_mod.main(["setup", "dev", "https://idx"]) == 2
