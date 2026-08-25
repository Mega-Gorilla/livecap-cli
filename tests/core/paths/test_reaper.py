"""staging root の孤児回収 (Issue #375 PR 2)。

`ascii_safe_temp_environment()` は**退出時に自分のディレクトリを消さない**
(消すと #386 のデータ消失が再発する)。その代わりに残骸が積み上がるので、
ここで TTL 回収する。

**使用中を消さないことが最重要。** PID 生存判定は使わない — 子プロセスは親の
TEMP を継承するがディレクトリ名は親 pid のままで、pid は再利用される。
代わりに **OS に判定させる** (掴まれていれば削除が失敗する)。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from livecap_cli.paths.reaper import (
    DEFAULT_TTL_HOURS,
    reap_staging_root,
    reset_reaper_state,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_reaper_state()
    yield
    reset_reaper_state()


def _entry(root: Path, purpose: str, name: str, *, age_hours: float) -> Path:
    path = root / purpose / name
    path.mkdir(parents=True)
    (path / "leftover.tmp").write_bytes(b"x")
    old = time.time() - age_hours * 3600
    os.utime(path, (old, old))
    return path


class TestReaping:
    def test_removes_entries_older_than_the_ttl(self, tmp_path: Path):
        stale = _entry(tmp_path, "runtime", "stale", age_hours=DEFAULT_TTL_HOURS + 1)

        assert reap_staging_root(tmp_path) == 1
        assert not stale.exists()

    def test_keeps_entries_within_the_ttl(self, tmp_path: Path):
        fresh = _entry(tmp_path, "runtime", "fresh", age_hours=1)

        assert reap_staging_root(tmp_path) == 0
        assert fresh.is_dir()

    def test_runs_once_per_root(self, tmp_path: Path):
        """境界呼び出しのたびに ``scandir`` しない。回収に緊急性は無い。"""
        _entry(tmp_path, "runtime", "stale", age_hours=DEFAULT_TTL_HOURS + 1)

        assert reap_staging_root(tmp_path) == 1

        _entry(tmp_path, "runtime", "stale2", age_hours=DEFAULT_TTL_HOURS + 1)
        assert reap_staging_root(tmp_path) == 0, "2 回目は走らないはず"
        assert reap_staging_root(tmp_path, force=True) == 1

    @pytest.mark.skipif(
        os.name != "nt", reason="保持ハンドルが削除を阻むのは Windows の挙動"
    )
    def test_does_not_remove_entries_that_are_in_use(self, tmp_path: Path):
        """**OS に判定させる。** 掴まれていれば削除が失敗し、そのエントリは残る。"""
        in_use = _entry(tmp_path, "runtime", "busy", age_hours=DEFAULT_TTL_HOURS + 1)

        handle = open(in_use / "leftover.tmp", "rb")
        try:
            removed = reap_staging_root(tmp_path)
        finally:
            handle.close()

        assert removed == 0
        assert in_use.is_dir(), "使用中のエントリを消してはいけない"

    def test_does_not_use_pid_liveness(self, tmp_path: Path):
        """**PID を見ない。**

        自分の pid を名前に持つ古いエントリでも、TTL を超えていれば消す。
        pid が生きているかは判定材料にしない (pid は再利用される)。
        """
        mine = _entry(
            tmp_path, "runtime", f"{os.getpid()}-abc", age_hours=DEFAULT_TTL_HOURS + 1
        )

        assert reap_staging_root(tmp_path) == 1
        assert not mine.exists()


class TestBestEffort:
    def test_missing_root_does_not_raise(self, tmp_path: Path):
        assert reap_staging_root(tmp_path / "does-not-exist") == 0

    def test_unreadable_purpose_is_skipped(self, tmp_path: Path, monkeypatch):
        """回収の失敗が本筋を止めてはいけない。"""
        _entry(tmp_path, "runtime", "stale", age_hours=DEFAULT_TTL_HOURS + 1)

        import livecap_cli.paths.reaper as reaper_mod

        real_rmtree = reaper_mod.shutil.rmtree

        def exploding(path, *args, **kwargs):
            raise OSError("filesystem said no")

        monkeypatch.setattr(reaper_mod.shutil, "rmtree", exploding)

        assert reap_staging_root(tmp_path) == 0  # 例外を出さない
        monkeypatch.setattr(reaper_mod.shutil, "rmtree", real_rmtree)

    def test_files_at_the_top_level_are_ignored(self, tmp_path: Path):
        """purpose 階層でないものを掃除対象にしない。"""
        stray = tmp_path / "not-a-purpose.txt"
        stray.write_text("x")

        assert reap_staging_root(tmp_path) == 0
        assert stray.exists()
