"""License & dependency safety: dev-only libs must not leak into production code.

``pykakasi`` (``GPL-3.0-or-later``) and ``kanjize`` (``MIT``) are
development dependencies declared in ``pyproject.toml`` under
``[project.optional-dependencies] dev``, used only by
``benchmarks/confidence_calibration/_normalize_jp.py`` for calibration
alignment metrics (Issue #338 PR-γ).

The livecap-cli production runtime is licensed under ``AGPL-3.0-only``
(``pyproject.toml`` ``license = "AGPL-3.0-only"``). To keep production
distributions free of GPL-3.0-or-later coupling (pykakasi) and free of
unnecessary runtime dependencies (kanjize is a small benchmark-only
helper), no module under ``livecap_cli/`` may reference either library.

The tests enforce this invariant by static grep over the production
source tree. A static check is preferred over a dynamic ``sys.modules``
inspection because it remains correct even when imports are lazy or
guarded by ``try/except``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PRODUCTION_DIR = _REPO_ROOT / "livecap_cli"


@pytest.mark.parametrize(
    "library,license_label",
    [
        ("pykakasi", "GPL-3.0-or-later"),
        ("kanjize", "MIT (benchmark-only helper)"),
    ],
)
def test_no_dev_only_lib_in_production_code(
    library: str, license_label: str
) -> None:
    """Assert no ``.py`` file under ``livecap_cli/`` mentions the dev-only lib.

    If this test fails, the dev-only library has leaked into production
    code. Either:
    1. Move the usage to ``benchmarks/`` (dev-only path), or
    2. Re-evaluate dependency compatibility (AGPL-3.0-only + ``license_label``)
       and update this test only if the project's dependency posture changes.
    """
    assert _PRODUCTION_DIR.is_dir(), (
        f"production dir not found: {_PRODUCTION_DIR}"
    )
    offenders: list[Path] = []
    for py_file in _PRODUCTION_DIR.rglob("*.py"):
        # Skip __pycache__ and similar
        if "__pycache__" in py_file.parts:
            continue
        try:
            text = py_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if library in text:
            offenders.append(py_file.relative_to(_REPO_ROOT))
    assert not offenders, (
        f"{library} ({license_label}, dev-only) found in production code: "
        f"{offenders}. livecap-cli is AGPL-3.0-only; {library} belongs only "
        f"under benchmarks/."
    )
