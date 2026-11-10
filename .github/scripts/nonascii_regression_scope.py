"""Decide whether Integration Tests runs the non-ASCII real-model regression (#377).

The real-model tier checks how the native libraries behave with non-ASCII paths
and real models (sentencepiece in NeMo's untar directory, sherpa-onnx, tokenizers,
the CUDA Jiterator kernel cache, ...). What changes that behaviour is the
dependency stack and the boundary helpers, not ordinary engine code: whether each
call site goes through the helpers is pinned by the cheap tier of
``tests/nonascii`` (the registry / AST checks), which Core Tests runs on every PR.

So a pull request runs the regression only when it touches one of the triggers
below. ``schedule`` (weekly) and ``workflow_dispatch`` always run it, so a
regression that slips past a PR is caught within a week.

Usage (GitHub Actions)::

    python .github/scripts/nonascii_regression_scope.py <event_name>

On ``pull_request`` the changed files are ``git diff --name-only HEAD^1 HEAD``:
the checkout is the PR merge commit, whose first parent is the base tip, so this
is exactly what the PR changes (it needs ``fetch-depth: 2``). Writes ``run=true``
/ ``run=false`` to ``$GITHUB_OUTPUT`` and the reason to ``$GITHUB_STEP_SUMMARY``
when they are set, and prints the decision.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Iterable, Sequence

#: Files whose change re-runs the regression on a PR.
TRIGGER_FILES = (
    "pyproject.toml",
    "uv.lock",
    ".github/workflows/integration-tests.yml",
    ".github/scripts/nonascii_regression_scope.py",
)

#: Directories (prefixes) whose change re-runs the regression on a PR.
#: ``livecap_cli/paths/`` holds the ASCII-safe staging / temp helpers, and
#: ``.github/actions/`` builds the GPU venv (the native stack under test).
TRIGGER_DIRS = (
    "tests/nonascii/",
    "livecap_cli/paths/",
    ".github/actions/",
)


def triggering_files(changed: Iterable[str]) -> list[str]:
    """The changed paths that require the regression (``/``-separated, repo-relative)."""
    return [path for path in changed if path in TRIGGER_FILES or path.startswith(TRIGGER_DIRS)]


def decide(event_name: str, changed: Sequence[str]) -> tuple[bool, str]:
    """``(run, reason)`` for this event."""
    if event_name != "pull_request":
        return True, f"`{event_name}` always runs it"
    hits = triggering_files(changed)
    if hits:
        shown = ", ".join(f"`{path}`" for path in hits[:5])
        more = f" (+{len(hits) - 5})" if len(hits) > 5 else ""
        return True, f"the PR changes {shown}{more}"
    return False, (
        "the PR changes none of the dependencies / `tests/nonascii/` / `livecap_cli/paths/` / "
        "`.github/actions/` / this workflow; the weekly schedule and workflow_dispatch run it"
    )


def pr_changed_files() -> list[str]:
    """Files the PR changes: the merge commit against its first parent (the base tip)."""
    out = subprocess.run(
        ["git", "diff", "--name-only", "HEAD^1", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def _append(env_var: str, text: str) -> None:
    path = os.environ.get(env_var)
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text)


def main(argv: Sequence[str]) -> int:
    if len(argv) != 2:
        print("usage: nonascii_regression_scope.py <event_name>", file=sys.stderr)
        return 2
    event_name = argv[1]
    changed = pr_changed_files() if event_name == "pull_request" else []
    run, reason = decide(event_name, changed)
    verdict = "runs" if run else "skipped"
    print(f"non-ASCII real-model regression: {verdict} ({reason})")
    _append("GITHUB_OUTPUT", f"run={'true' if run else 'false'}\n")
    _append("GITHUB_STEP_SUMMARY", f"**Non-ASCII real-model regression (#377)**: {verdict} — {reason}\n\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
