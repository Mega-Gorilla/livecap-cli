"""`git_dirty` が「記録した commit で再現できるか」を正しく答えること (Issue #377)。

この判定が守っているのは **evidence の再現可能性**である。未コミットの working
tree で測ると ``git_commit`` は 1 つ前の commit を指したまま、実際には手元の変更で
測ることになり、証拠だけを見ても実行コードを特定できない。

**偽の clean が最も危険**なので、そちらを重点的に固定する。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from .record import RunMetadata, _git_commit, _git_dirty
from .report import render_metadata

pytestmark = pytest.mark.nonascii_paths


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """commit が 1 つある clean な repository。"""
    if shutil.which("git") is None:  # pragma: no cover - CI には git がある
        pytest.skip("git が見つからない")

    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", ".")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "test")
    (root / "tracked.py").write_text("x = 1\n", encoding="utf-8")
    _git(root, "add", "tracked.py")
    _git(root, "commit", "-q", "-m", "init")
    return root


class TestGitDirty:
    def test_clean_tree(self, repo: Path):
        assert _git_dirty(repo) is False

    def test_tracked_modification(self, repo: Path):
        (repo / "tracked.py").write_text("x = 2\n", encoding="utf-8")
        assert _git_dirty(repo) is True

    def test_staged_change(self, repo: Path):
        (repo / "tracked.py").write_text("x = 3\n", encoding="utf-8")
        _git(repo, "add", "tracked.py")
        assert _git_dirty(repo) is True

    def test_untracked_file(self, repo: Path):
        (repo / "new_probe.py").write_text("y = 1\n", encoding="utf-8")
        assert _git_dirty(repo) is True

    def test_untracked_file_when_status_hides_untracked(self, repo: Path):
        """**偽の clean を防ぐ本丸。**

        ``git status --porcelain`` は ``status.showUntrackedFiles`` を尊重する。
        ``no`` を設定した環境では、未追跡の probe / helper / test が実行に使われても
        出力が空になり、素の実装では ``False`` (= 再現可能) と記録してしまう。
        """
        _git(repo, "config", "status.showUntrackedFiles", "no")
        (repo / "new_probe.py").write_text("y = 1\n", encoding="utf-8")

        # 前提: この設定では素の --porcelain が未追跡を隠す
        hidden = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            capture_output=True,
            text=True,
        ).stdout
        assert hidden.strip() == "", "前提が崩れている (設定が効いていない)"

        assert _git_dirty(repo) is True

    def test_ignored_file_is_not_dirty(self, repo: Path):
        """ignore 済みは dirty ではない。``.venv`` や ``__pycache__`` で常時 True に
        なると、判定そのものが無意味になる。"""
        (repo / ".gitignore").write_text("junk/\n", encoding="utf-8")
        _git(repo, "add", ".gitignore")
        _git(repo, "commit", "-q", "-m", "ignore")
        (repo / "junk").mkdir()
        (repo / "junk" / "tmp.bin").write_bytes(b"")

        assert _git_dirty(repo) is False

    def test_not_a_git_repository(self, tmp_path: Path):
        """**判定不能を clean と混同しない。**"""
        plain = tmp_path / "plain"
        plain.mkdir()
        assert _git_dirty(plain) is None

    def test_commit_is_reported_for_the_same_tree(self, repo: Path):
        assert _git_commit(repo) != "(unknown)"
        assert len(_git_commit(repo)) == 40


class TestMetadataRendering:
    """棚卸し表の §0 が 3 状態を書き分けること。"""

    @staticmethod
    def _run(dirty: bool | None) -> dict:
        """実物の :class:`RunMetadata` から作る。

        手書きの dict にすると、フィールドが増えたときに**このテストだけが
        古い形を検証し続ける**。見たいのは ``git_dirty`` の書き分けだけなので、
        それ以外は実物に任せる。
        """
        run = RunMetadata(run_id="run-test", measured_at="2026-08-25")
        payload = run.to_dict()
        payload["git_dirty"] = dirty
        payload["git_commit"] = "0" * 40
        return payload

    def test_clean_has_no_warning(self):
        out = render_metadata(self._run(False))
        assert "0" * 40 in out
        assert "dirty" not in out

    def test_dirty_is_called_out(self):
        out = render_metadata(self._run(True))
        assert "dirty tree" in out
        assert "再現できない" in out

    def test_unknown_is_distinguished_from_clean(self):
        out = render_metadata(self._run(None))
        assert "判定不能" in out
