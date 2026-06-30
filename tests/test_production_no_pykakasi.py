"""License safety: pykakasi must not be imported by livecap-cli production code.

``pykakasi`` is licensed under ``GPL-3.0-or-later`` and is used only by
``benchmarks/confidence_calibration/_normalize_jp.py`` for calibration
alignment metrics (Issue #338 PR-γ). It is declared in ``pyproject.toml``
under ``[project.optional-dependencies] dev`` and is a development /
benchmark-only dependency.

The livecap-cli production runtime is licensed under ``AGPL-3.0-only``
(``pyproject.toml`` ``license = "AGPL-3.0-only"``). To keep production
distributions free of GPL-3.0-or-later coupling, no module under
``livecap_cli/`` may reference ``pykakasi`` (import or otherwise).

This test enforces that invariant by static grep over the production
source tree. A static check is preferred over a dynamic ``sys.modules``
inspection because it remains correct even when imports are lazy or
guarded by ``try/except``.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PRODUCTION_DIR = _REPO_ROOT / "livecap_cli"


def test_no_pykakasi_in_production_code() -> None:
    """Assert no ``.py`` file under ``livecap_cli/`` mentions ``pykakasi``.

    If this test fails, pykakasi has leaked into production code. Either:
    1. Move the usage to ``benchmarks/`` (dev-only path), or
    2. Re-evaluate license compatibility (AGPL-3.0-only vs GPL-3.0-or-later)
       and update this test only if the project license posture changes.
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
        if "pykakasi" in text:
            offenders.append(py_file.relative_to(_REPO_ROOT))
    assert not offenders, (
        f"pykakasi (GPL-3.0-or-later, dev-only) found in production code: "
        f"{offenders}. livecap-cli is AGPL-3.0-only; pykakasi belongs only "
        f"under benchmarks/."
    )
