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


def test_action_reads_the_versions_through_the_script():
    run = yaml.safe_load((ACTION_DIR / "action.yml").read_text(encoding="utf-8"))["runs"]["steps"][0]["run"]

    assert "torch_pins.py" in run
    assert "GITHUB_ACTION_PATH" in run
    assert ".Trim()" not in run, "版の読み取りを PowerShell の文字列処理に戻さない"
