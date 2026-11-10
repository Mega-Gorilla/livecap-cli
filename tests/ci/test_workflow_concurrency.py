"""GitHub Actions の concurrency 方針を固定する。

* **PR で起動する workflow**: 同じ PR に新しい push があったら古い HEAD の run を取り消す。
  self-hosted の GPU runner は 1 台だけで、置き換え済みの run (GPU job 20〜25 分) が新しい
  HEAD の GPU job を待たせていた (手で取り消した例: run 37278047769 / 42556993911)。
  PR 以外 (main への push / schedule / workflow_dispatch) は取り消さない
* **手動専用の GPU workflow** (VAD 最適化 / ベンチマーク): 夜間に複数投入して runner で順に
  流す使い方なので、取り消しを付けない
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"

PR_GROUP = "${{ github.workflow }}-${{ github.event.pull_request.number || github.run_id }}"
PR_ONLY_CANCEL = "${{ github.event_name == 'pull_request' }}"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _triggers(workflow: dict) -> set[str]:
    # PyYAML は `on:` を真偽値 True として読む
    on = workflow.get("on", workflow.get(True))
    if isinstance(on, str):
        return {on}
    if isinstance(on, list):
        return set(on)
    return set(on or {})


ALL = sorted(WORKFLOWS.glob("*.yml"))
PR_TRIGGERED = [p for p in ALL if "pull_request" in _triggers(_load(p))]
DISPATCH_ONLY = [p for p in ALL if _triggers(_load(p)) == {"workflow_dispatch"}]


def test_the_expected_workflows_are_covered():
    names = {p.name for p in PR_TRIGGERED}
    assert {"core-tests.yml", "core-tests-windows.yml", "integration-tests.yml"} <= names


@pytest.mark.parametrize("path", PR_TRIGGERED, ids=lambda p: p.name)
def test_pr_runs_cancel_superseded_runs_of_the_same_pr(path):
    concurrency = _load(path).get("concurrency")

    assert concurrency == {"group": PR_GROUP, "cancel-in-progress": PR_ONLY_CANCEL}, (
        f"{path.name}: PR の run は PR 番号ごとの group で取り消し、PR 以外は run_id で取り消さない"
    )


@pytest.mark.parametrize("path", DISPATCH_ONLY, ids=lambda p: p.name)
def test_dispatch_only_workflows_never_cancel_queued_runs(path):
    """夜間にまとめて投入した最適化 / ベンチマークが、後続の投入で取り消されない。"""
    concurrency = _load(path).get("concurrency")

    assert not (isinstance(concurrency, dict) and concurrency.get("cancel-in-progress")), path.name
