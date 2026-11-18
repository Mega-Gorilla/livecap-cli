"""上限つきの一時ファイル / 一時ディレクトリの作成 (Issue #468 / #504)。

``tempfile.mkstemp`` / ``mkdtemp`` / ``TemporaryDirectory`` は、Windows で ACL により
書き込みを拒否されたディレクトリでは ``PermissionError`` を名前の衝突とみなして
``os.TMP_MAX`` 回まで再試行し、返らない。ここで固定するのは次の点:

- 再試行するのは名前の衝突 (``FileExistsError``、または候補の名前に実際に
  ディレクトリがあるときの ``PermissionError``) だけで、回数に上限がある
- 既存のファイルやディレクトリには触れない
- ``temporary_directory`` は ``TemporaryDirectory`` と同じく、抜けるときに中身ごと
  消す (読み取り専用のファイルも)
- Windows で deny ACE を付けた実ディレクトリに対し、``model_manager`` と
  ``ffmpeg_manager`` の 3 経路が有限時間で ``PermissionError`` になる
"""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path
from typing import Callable, List

import pytest

from livecap_cli.resources import bounded_tempfile, ffmpeg_pins
from livecap_cli.resources.bounded_tempfile import (
    NAME_ATTEMPTS,
    mkdtemp,
    mkstemp,
    temporary_directory,
)
from tests.core.resources.acl_support import (
    DENY_FORMS,
    PROBE_BUDGET_S,
    run_in_child,
    shadow_module,
    short_temp_dir,  # noqa: F401 - fixture
    write_denied,
)


def _make_file(directory: Path) -> str:
    handle, path = mkstemp(dir=directory, prefix="pre-", suffix=".tmp")
    os.close(handle)
    return path


#: ``(作る関数, それが使う os の関数名)``。どちらにも同じ再試行の規則が当てはまる。
_CREATORS = {
    "mkstemp": (_make_file, "open"),
    "mkdtemp": (lambda directory: mkdtemp(dir=directory, prefix="pre-", suffix=".tmp"), "mkdir"),
}


def _record(monkeypatch: pytest.MonkeyPatch, func_name: str, error: OSError | None) -> List[str]:
    """``bounded_tempfile`` が使う ``os.<func_name>`` を差し替え、呼ばれた path を記録する。"""
    calls: List[str] = []
    real = getattr(os, func_name)

    def fake(path, *args):
        calls.append(str(path))
        if error is not None:
            raise error
        return real(path, *args)

    shadow_module(monkeypatch, bounded_tempfile, "os", **{func_name: fake})
    return calls


class TestCreate:
    def test_mkstemp_returns_an_open_file_in_the_directory(self, tmp_path: Path):
        handle, path = mkstemp(dir=tmp_path, prefix="stamp.", suffix=".tmp")
        try:
            os.write(handle, b"payload")
        finally:
            os.close(handle)

        created = Path(path)
        assert created.is_absolute()
        assert created.parent == tmp_path
        assert created.name.startswith("stamp.") and created.name.endswith(".tmp")
        assert created.read_bytes() == b"payload"

    def test_mkdtemp_returns_an_empty_directory_in_the_directory(self, tmp_path: Path):
        created = Path(mkdtemp(dir=tmp_path, prefix="ffmpeg-install-"))
        assert created.is_absolute()
        assert created.parent == tmp_path
        assert created.name.startswith("ffmpeg-install-")
        assert created.is_dir() and list(created.iterdir()) == []

    @pytest.mark.parametrize("creator", sorted(_CREATORS))
    def test_missing_directory_fails_on_the_first_attempt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, creator: str
    ):
        create, func_name = _CREATORS[creator]
        calls = _record(monkeypatch, func_name, error=None)
        with pytest.raises(FileNotFoundError):
            create(tmp_path / "missing")
        assert len(calls) == 1

    @pytest.mark.parametrize("creator", sorted(_CREATORS))
    @pytest.mark.parametrize(
        "error",
        [
            PermissionError(errno.EACCES, "Permission denied"),
            OSError(errno.EROFS, "Read-only file system"),
        ],
        ids=["PermissionError", "EROFS"],
    )
    def test_other_errors_fail_on_the_first_attempt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, creator: str, error: OSError
    ):
        """**``PermissionError`` を名前の衝突と読まない** (候補の名前にディレクトリが無いとき)。"""
        create, func_name = _CREATORS[creator]
        calls = _record(monkeypatch, func_name, error=error)
        with pytest.raises(type(error)):
            create(tmp_path)
        assert len(calls) == 1

    @pytest.mark.parametrize("creator", sorted(_CREATORS))
    def test_name_collisions_retry_up_to_the_bound(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, creator: str
    ):
        create, func_name = _CREATORS[creator]
        calls = _record(monkeypatch, func_name, error=FileExistsError(errno.EEXIST, "File exists"))
        with pytest.raises(FileExistsError, match="no unused temporary name"):
            create(tmp_path)
        assert len(calls) == NAME_ATTEMPTS
        assert len(set(calls)) == NAME_ATTEMPTS, "再試行のたびに別の名前を使うこと"

    @pytest.mark.parametrize("creator", sorted(_CREATORS))
    @pytest.mark.parametrize("taken_kind", ["file", "dir"])
    def test_an_existing_entry_moves_to_the_next_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, creator: str, taken_kind: str
    ):
        """同じ名前のファイルかディレクトリがあれば次の名前で成功し、**既存のものは壊さない**。

        Windows でディレクトリと同じ名前のファイルを開こうとすると ``PermissionError``
        になる。これは本物の衝突なので、再試行してよい唯一の ``PermissionError``。
        """
        names = iter(["aaaaaaaa", "bbbbbbbb"])
        shadow_module(monkeypatch, bounded_tempfile, "secrets", token_hex=lambda nbytes: next(names))
        taken = tmp_path / "pre-aaaaaaaa.tmp"
        if taken_kind == "file":
            taken.write_text("do not touch")
        else:
            taken.mkdir()
            (taken / "inner.txt").write_text("do not touch")

        create, _ = _CREATORS[creator]
        created = Path(create(tmp_path))

        assert created.name == "pre-bbbbbbbb.tmp"
        if taken_kind == "file":
            assert taken.read_text() == "do not touch"
        else:
            assert (taken / "inner.txt").read_text() == "do not touch"


class TestTemporaryDirectory:
    def test_removes_the_directory_and_its_contents(self, tmp_path: Path):
        with temporary_directory(dir=tmp_path, prefix="ws-") as name:
            workspace = Path(name)
            assert workspace.parent == tmp_path and workspace.is_dir()
            (workspace / "sub").mkdir()
            (workspace / "sub" / "file.bin").write_bytes(b"x")
        assert list(tmp_path.iterdir()) == []

    def test_removes_the_directory_when_the_body_raises(self, tmp_path: Path):
        with pytest.raises(RuntimeError, match="boom"):
            with temporary_directory(dir=tmp_path) as name:
                Path(name, "file").write_text("x")
                raise RuntimeError("boom")
        assert list(tmp_path.iterdir()) == []

    def test_removes_read_only_contents(self, tmp_path: Path):
        """``TemporaryDirectory`` と同じく、権限のせいで消せないものは権限を戻して消す。"""
        with temporary_directory(dir=tmp_path) as name:
            locked = Path(name, "locked.bin")
            locked.write_bytes(b"x")
            os.chmod(locked, stat.S_IREAD)
        assert list(tmp_path.iterdir()) == []

    def test_a_directory_already_removed_by_the_body_is_fine(self, tmp_path: Path):
        with temporary_directory(dir=tmp_path) as name:
            os.rmdir(name)
        assert list(tmp_path.iterdir()) == []

    def test_creation_failure_propagates_without_retry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        calls = _record(monkeypatch, "mkdir", error=PermissionError(errno.EACCES, "Permission denied"))
        with pytest.raises(PermissionError):
            with temporary_directory(dir=tmp_path):
                pytest.fail("the body must not run")
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# Windows: deny ACE を付けた実ディレクトリ
# ---------------------------------------------------------------------------

_CHILD = r"""
import faulthandler, json, sys, time
from pathlib import Path

faulthandler.dump_traceback_later(float(sys.argv[1]), exit=True)
target, base = sys.argv[2], Path(sys.argv[3])

from livecap_cli.resources import ffmpeg_manager, ffmpeg_pins
from livecap_cli.resources.ffmpeg_manager import FFmpegManager
from livecap_cli.resources.model_manager import ModelManager
from livecap_cli.resources.resource_locator import ResourceLocator


def no_network(*args, **kwargs):
    raise RuntimeError("the download was reached; the test must not use the network")


ffmpeg_manager.download_with_retry = no_network
model_manager = ModelManager(models_root=base / "models", cache_root=base / "cache")


def run():
    if target == "model_manager_temporary_directory":
        with model_manager.temporary_directory("downloads"):
            return "created"
    ffmpeg = FFmpegManager(locator=ResourceLocator(search_roots=[]), model_manager=model_manager)
    spec = ffmpeg_pins.resolve_platform_spec()
    if target == "ffmpeg_install_workspace":
        ffmpeg._install_verified_pair(spec)
        return "installed"
    if target == "ffmpeg_write_stamp":
        ffmpeg._write_stamp(spec)
        return "written"
    raise SystemExit(f"unknown target: {target}")


start = time.monotonic()
try:
    outcome = run()
except Exception as error:
    outcome = f"{type(error).__name__}: {error}"
print(json.dumps({"outcome": outcome, "elapsed": time.monotonic() - start}))
"""


def _layout_downloads(base: Path) -> Path:
    """``<cache_root>/downloads`` を拒否する (``ModelManager.temporary_directory``)。"""
    downloads = base / "cache" / "downloads"
    downloads.mkdir(parents=True)
    return downloads


def _layout_install(base: Path) -> Path:
    """``<cache_root>`` を拒否する。``<cache_root>/ffmpeg`` は先に作っておく。

    無いと ``FFmpegManager.__init__`` の ``mkdir`` が先に失敗し、install の作業
    ディレクトリまで届かない。
    """
    cache = base / "cache"
    (cache / "ffmpeg").mkdir(parents=True)
    return cache


def _layout_stamp(base: Path) -> Path:
    """``<cache_root>/ffmpeg`` を拒否する。``_write_stamp`` が stat する binary は先に置く。"""
    ffmpeg_dir = base / "cache" / "ffmpeg"
    ffmpeg_dir.mkdir(parents=True)
    for binary in ffmpeg_pins.resolve_platform_spec().binaries.values():
        (ffmpeg_dir / binary.name).write_bytes(b"stub")
    return ffmpeg_dir


_LAYOUTS: dict[str, Callable[[Path], Path]] = {
    "model_manager_temporary_directory": _layout_downloads,
    "ffmpeg_install_workspace": _layout_install,
    "ffmpeg_write_stamp": _layout_stamp,
}


@pytest.mark.parametrize("form", sorted(DENY_FORMS))
@pytest.mark.parametrize("target", sorted(_LAYOUTS))
def test_write_denied_directory_is_reported_in_bounded_time(
    short_temp_dir: Path, target: str, form: str  # noqa: F811 - fixture
):
    """#504 の回帰。修正前は 3 つとも ``tempfile`` の中で返らなかった。"""
    if target != "model_manager_temporary_directory":
        try:
            ffmpeg_pins.resolve_platform_spec()
        except ffmpeg_pins.UnsupportedPlatformError:
            pytest.skip("このプラットフォームには pinned の FFmpeg が無い")

    denied = _LAYOUTS[target](short_temp_dir)
    with write_denied(denied, form):
        result = run_in_child(_CHILD, target, str(short_temp_dir))

    assert result["outcome"].startswith("PermissionError:"), result["outcome"]
    assert result["elapsed"] < PROBE_BUDGET_S, result
