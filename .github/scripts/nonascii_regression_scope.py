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

On ``pull_request`` the changed files are ``git diff --name-only -z HEAD^1 HEAD``:
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
    triggers = ", ".join(f"`{path}`" for path in (*TRIGGER_FILES, *TRIGGER_DIRS))
    return False, f"the PR changes none of {triggers}; the weekly schedule and workflow_dispatch run it"


def pr_changed_files() -> list[str]:
    """Files the PR changes: the merge commit against its first parent (the base tip).

    ``-z`` (NUL-separated) is required: without it ``core.quotePath`` (on by default)
    prints a non-ASCII path quoted and escaped (``"tests/nonascii/\\350..."``), which
    then matches no trigger and silently skips the regression (PR #484 review F1).
    Git prints the path bytes as stored (UTF-8); undecodable bytes are kept as
    surrogates instead of failing, so the ASCII prefix still matches.
    """
    out = subprocess.run(
        ["git", "diff", "--name-only", "-z", "HEAD^1", "HEAD"],
        check=True,
        capture_output=True,
    ).stdout
    return [path for path in out.decode("utf-8", errors="surrogateescape").split("\0") if path]


def _append(env_var: str, text: str) -> None:
    """UTF-8 で追記する。``pr_changed_files`` が残した surrogate (UTF-8 でない path) は escape する。"""
    path = os.environ.get(env_var)
    if path:
        with open(path, "a", encoding="utf-8", errors="backslashreplace") as fh:
            fh.write(text)


def _ascii_log(text: str) -> str:
    """標準出力へ書く 1 行。

    Windows runner の標準出力は cp1252 などで、非 ASCII のファイル名をそのまま ``print``
    すると ``UnicodeEncodeError`` で落ちる (PR #484 レビュー F2、``tests/conftest.py`` の
    ``_annotation_safe`` と同じ扱い)。ログは ASCII へ escape し、原文は UTF-8 の Step Summary に残す。
    """
    return text.encode("ascii", "backslashreplace").decode("ascii")


def main(argv: Sequence[str]) -> int:
    if len(argv) != 2:
        print("usage: nonascii_regression_scope.py <event_name>", file=sys.stderr)
        return 2
    event_name = argv[1]
    changed = pr_changed_files() if event_name == "pull_request" else []
    run, reason = decide(event_name, changed)
    verdict = "runs" if run else "skipped"
    # 判定 (GITHUB_OUTPUT) を先に書く: 表示のログが何で落ちても、後続 step の判定は残す
    _append("GITHUB_OUTPUT", f"run={'true' if run else 'false'}\n")
    _append("GITHUB_STEP_SUMMARY", f"**Non-ASCII real-model regression (#377)**: {verdict} — {reason}\n\n")
    print(_ascii_log(f"non-ASCII real-model regression: {verdict} ({reason})"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
