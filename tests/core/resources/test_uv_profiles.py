import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_system_python() -> str:
    env_value = os.environ.get("UV_SMOKE_PYTHON")
    if env_value:
        located = shutil.which(env_value)
        if located:
            path = Path(located)
            if not path.is_absolute():
                path = (Path.cwd() / path)
            return str(path)
        path = Path(env_value).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path)
        return str(path)

    venv_candidates = [
        REPO_ROOT / ".venv" / "bin" / "python",
        REPO_ROOT / ".venv" / "Scripts" / "python.exe",
    ]
    for candidate in venv_candidates:
        if candidate.exists():
            return str(candidate)

    if sys.executable:
        return str(sys.executable)

    for candidate in ("python3", "python"):
        located = shutil.which(candidate)
        if located:
            return str(Path(located).resolve())

    return sys.executable


SYSTEM_PYTHON = resolve_system_python()

PROFILE_COMMANDS = [
    {
        "cmd": [sys.executable, "-m", "pytest", "tests/core/config/test_config_defaults.py", "-q"],
    },
    {
        "cmd": [
            sys.executable,
            "-c",
            "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('deep_translator') else 1)",
        ],
    },
]

NEMO_SMOKE = REPO_ROOT / "tests" / "engines" / "test_nemo_smoke.py"
if NEMO_SMOKE.exists():
    PROFILE_COMMANDS.append(
        {
            "cmd": [SYSTEM_PYTHON, "-m", "pytest", "engines/test_nemo_smoke.py", "-q"],
            "cwd": REPO_ROOT / "tests",
            "env": {
                "PYTHONPATH": str(REPO_ROOT / "src"),
            },
        }
    )


@pytest.mark.parametrize("spec", PROFILE_COMMANDS)
def test_installed_extras(spec):
    cmd = spec["cmd"]
    cwd = spec.get("cwd", REPO_ROOT)
    env = os.environ.copy()
    overrides = spec.get("env")
    if overrides:
        overrides = overrides.copy()
        if "PYTHONPATH" in overrides:
            current = env.get("PYTHONPATH")
            parts = [overrides["PYTHONPATH"]]
            if current:
                parts.append(current)
            overrides["PYTHONPATH"] = os.pathsep.join(parts)
        env.update(overrides)

    completed = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if completed.returncode != 0:
        hint = ""
        if spec is PROFILE_COMMANDS[-1]:
            hint = (
                "\nHint: Ensure the engines-nemo extra is installed in the Python "
                "specified by UV_SMOKE_PYTHON (e.g. `python -m pip install "
                "'.[engines-nemo]' pytest`)."
            )
        pytest.fail(
            f"Command {' '.join(cmd)} failed with code {completed.returncode}:\n"
            f"{completed.stdout}\n{completed.stderr}{hint}"
        )
