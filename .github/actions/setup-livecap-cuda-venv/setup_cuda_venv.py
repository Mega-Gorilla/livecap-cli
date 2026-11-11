"""Create the GPU venv for a self-hosted runner from uv.lock (#379 / #470).

Run with the preinstalled interpreter (``PYTHON_EXE``) from the repository root::

    python setup_cuda_venv.py "<extras>" <torch index url>

Steps (the same on Windows and Linux; the action calls this from both):

1. Remove ``.venv`` (a non-ephemeral runner keeps the workspace between jobs).
2. ``uv python pin <PYTHON_EXE>`` and ``uv sync --locked --extra ...``: install
   exactly the lock, never re-resolve.
3. Swap the lock's CPU torch family for the CUDA build of the **same** versions
   (read back with ``torch_pins.py``; ``--reinstall-package`` because
   ``torch==2.9.1`` is already satisfied by ``2.9.1+cpu``).
4. Fail unless CUDA is usable (GPU smoke would skip everything and an
   optimization would crawl on CPU).
5. Put the venv's script directory on ``GITHUB_PATH``. Later steps use the plain
   ``python`` on PATH - **not** ``uv run``, which re-syncs to the lock and puts the
   CPU wheels back.

Exit codes: 0 = ready, 1 = a step failed, 2 = bad arguments.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

_EXTRA_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")

CUDA_CHECK = (
    "import sys, torch; "
    "print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda_available', torch.cuda.is_available()); "
    "print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'); "
    "sys.exit(0 if torch.cuda.is_available() else 1)"
)


class SetupError(RuntimeError):
    """A step failed; the message says which."""


def parse_extras(text: str) -> list[str]:
    """Space-separated extras -> names. Rejects empty input and anything that is not an extra name."""
    extras = text.split()
    if not extras:
        raise ValueError("extras is empty")
    for extra in extras:
        if not _EXTRA_NAME.match(extra):
            raise ValueError(f"invalid extra name: {extra!r}")
    return extras


def venv_bin_dir(venv: Path, *, windows: bool) -> Path:
    return venv / ("Scripts" if windows else "bin")


def venv_python(venv: Path, *, windows: bool) -> Path:
    return venv_bin_dir(venv, windows=windows) / ("python.exe" if windows else "python")


def sync_command(extras: Sequence[str]) -> list[str]:
    command = ["uv", "sync", "--locked"]
    for extra in extras:
        command += ["--extra", extra]
    return command


def cuda_install_command(pins: Sequence[str], index_url: str) -> list[str]:
    """``uv pip install --reinstall-package <name>... <name>==<ver>... --index-url <cuda index>``."""
    command = ["uv", "pip", "install"]
    for pin in pins:
        command += ["--reinstall-package", pin.split("==", 1)[0]]
    return command + list(pins) + ["--index-url", index_url]


def _run(command: Sequence[str], what: str, **kwargs) -> subprocess.CompletedProcess:
    print("+ " + " ".join(str(part) for part in command), flush=True)
    result = subprocess.run(list(command), **kwargs)
    if result.returncode != 0:
        raise SetupError(f"{what} failed (exit {result.returncode})")
    return result


def setup(extras: Sequence[str], index_url: str, *, python_exe: str, workdir: Path, windows: bool) -> Path:
    """Build ``workdir/.venv``; returns the directory to add to PATH."""
    venv = workdir / ".venv"
    shutil.rmtree(venv, ignore_errors=True)
    _run(["uv", "python", "pin", python_exe], "uv python pin", cwd=workdir)
    _run(sync_command(extras), "uv sync", cwd=workdir)

    python = venv_python(venv, windows=windows)
    pins_script = Path(__file__).with_name("torch_pins.py")
    pins = _run(
        [str(python), str(pins_script)], "reading the installed torch versions", cwd=workdir, capture_output=True, text=True
    ).stdout.split()
    if not pins:
        raise SetupError("could not read the installed torch versions (no pins printed)")
    print(f"Installing CUDA build: {' '.join(pins)} from {index_url}", flush=True)
    _run(cuda_install_command(pins, index_url), "CUDA torch install", cwd=workdir)

    try:
        _run([str(python), "-c", CUDA_CHECK], "CUDA check", cwd=workdir)
    except SetupError as exc:
        raise SetupError(
            "CUDA is not available after the swap - fix the install steps (GPU work would skip or fall back to CPU)"
        ) from exc
    return venv_bin_dir(venv, windows=windows)


def main(argv: Sequence[str]) -> int:
    if len(argv) != 3:
        print('usage: setup_cuda_venv.py "<extras>" <torch-index-url>', file=sys.stderr)
        return 2
    try:
        extras = parse_extras(argv[1])
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    python_exe = os.environ.get("PYTHON_EXE")
    if not python_exe:
        print("error: PYTHON_EXE (the preinstalled interpreter) is not set", file=sys.stderr)
        return 2
    try:
        bin_dir = setup(extras, argv[2], python_exe=python_exe, workdir=Path.cwd(), windows=os.name == "nt")
    except SetupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    github_path = os.environ.get("GITHUB_PATH")
    if github_path:
        with open(github_path, "a", encoding="utf-8") as fh:
            fh.write(f"{bin_dir}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
