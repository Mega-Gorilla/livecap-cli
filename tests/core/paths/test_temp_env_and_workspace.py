"""`ascii_safe_temp_environment()` と `ascii_safe_workspace()` (Issue #375 PR 2)。

**この 2 つの非対称が設計の要点**なので、対比で固定する:

====================================  ==============  ==========================
                                      env を変える    退出時に自分の dir を消す
====================================  ==============  ==========================
``ascii_safe_temp_environment()``     する            **しない**
``ascii_safe_workspace()``            しない          **する**
====================================  ==============  ==========================

同じ 1 つの事実から出る — プロセス全体の TEMP を向けている間は**無関係な
スレッドのファイルもそこへ落ちる**。向けていなければ自分のファイルしか無い。
前者で消すと #386 のデータ消失が再発する。
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

import pytest

from livecap_cli.paths import (
    TempEnvironmentConflictError,
    ascii_safe_temp_environment,
    ascii_safe_workspace,
)
from livecap_cli.paths import roots

BOUNDARY = "test.boundary"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from livecap_cli.resources import _reset_resources_for_tests
    from livecap_cli.resources.configuration import clear_staging_roots

    for name in ("ProgramData", "SystemDrive", "PUBLIC"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIVECAP_CORE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LIVECAP_CORE_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.delenv("LIVECAP_CORE_ASCII_STAGING_DIR", raising=False)

    _reset_resources_for_tests()
    clear_staging_roots()
    roots.reset_staging_root_cache()
    yield
    _reset_resources_for_tests()
    clear_staging_roots()
    roots.reset_staging_root_cache()


def _temp_env_snapshot() -> dict:
    return {
        "TEMP": os.environ.get("TEMP"),
        "TMP": os.environ.get("TMP"),
        "TMPDIR": os.environ.get("TMPDIR"),
        "tempdir": tempfile.tempdir,
    }


class TestTempEnvironment:
    def test_redirects_every_variable_to_ascii(self):
        with ascii_safe_temp_environment(boundary=BOUNDARY) as target:
            assert str(target).isascii()
            for key in ("TEMP", "TMP", "TMPDIR"):
                assert os.environ[key] == str(target)
            assert tempfile.tempdir == str(target)

    def test_tempfile_actually_lands_in_the_ascii_directory(self):
        """**#379 が必要としている性質そのもの。**

        ネイティブが自前で ``%TEMP%`` へ展開する経路を救うには、``tempfile`` の
        既定行き先が ASCII でなければ意味がない。
        """
        with ascii_safe_temp_environment(boundary=BOUNDARY) as target:
            created = Path(tempfile.mkdtemp())
            assert str(created).isascii()
            assert target in created.parents or created.parent == target

    def test_restores_on_success(self):
        before = _temp_env_snapshot()
        with ascii_safe_temp_environment(boundary=BOUNDARY):
            pass
        assert _temp_env_snapshot() == before

    def test_restores_on_exception(self):
        before = _temp_env_snapshot()
        with pytest.raises(RuntimeError, match="boom"):
            with ascii_safe_temp_environment(boundary=BOUNDARY):
                raise RuntimeError("boom")
        assert _temp_env_snapshot() == before

    def test_nesting_the_same_purpose_is_reentrant(self):
        with ascii_safe_temp_environment(boundary=BOUNDARY, purpose="runtime") as outer:
            with ascii_safe_temp_environment(boundary=BOUNDARY, purpose="runtime") as inner:
                # 内側は**外側と同じ path を返す** — 環境は外側を指したままなので、
                # 別 path を返したら呼び出し側に嘘をつくことになる。
                assert inner == outer
            # 内側の退出で環境が戻ってはいけない
            assert os.environ["TEMP"] == str(outer)

    def test_conflicting_purpose_raises(self):
        with ascii_safe_temp_environment(boundary=BOUNDARY, purpose="runtime"):
            with pytest.raises(TempEnvironmentConflictError, match="runtime"):
                with ascii_safe_temp_environment(boundary=BOUNDARY, purpose="downloads"):
                    pass

    def test_does_not_delete_its_directory_on_exit(self):
        """**#386 の回帰。**

        プロセス全体の TEMP を向けている間は無関係なスレッドの
        ``NamedTemporaryFile()`` もそこへ落ちる。「自分が作った dir だから」と
        消すと、他人のファイルを巻き込む。
        """
        with ascii_safe_temp_environment(boundary=BOUNDARY) as target:
            victim = target / "someone-elses-file.tmp"
            victim.write_text("data from another thread")

        assert target.is_dir(), "退出時にディレクトリを消してはいけない"
        assert victim.read_text() == "data from another thread"

    def test_concurrent_scopes_do_not_restore_early(self):
        """並行スコープの一方が終わっても、他方の移設を途中で戻さない。"""
        before = _temp_env_snapshot()
        observed: list[str] = []
        errors: list[BaseException] = []
        release = threading.Event()

        def worker() -> None:
            try:
                with ascii_safe_temp_environment(boundary=BOUNDARY) as target:
                    release.wait(5)
                    observed.append(os.environ["TEMP"])
                    assert os.environ["TEMP"] == str(target)
            except BaseException as error:  # pragma: no cover - 失敗時の診断用
                errors.append(error)

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for thread in threads:
            thread.start()
        release.set()
        for thread in threads:
            thread.join(10)

        assert not errors, errors
        assert len(observed) == 3
        assert _temp_env_snapshot() == before


class TestWorkspace:
    def test_returns_an_empty_ascii_directory(self):
        with ascii_safe_workspace(boundary=BOUNDARY) as work:
            assert str(work).isascii()
            assert work.is_dir()
            assert list(work.iterdir()) == []

    def test_does_not_touch_the_environment(self):
        """**env を触らないので自明にスレッド安全・ネスト可。**"""
        before = _temp_env_snapshot()
        with ascii_safe_workspace(boundary=BOUNDARY):
            assert _temp_env_snapshot() == before
        assert _temp_env_snapshot() == before

    def test_removes_its_directory_on_exit(self):
        with ascii_safe_workspace(boundary=BOUNDARY) as work:
            (work / "utterance.wav").write_bytes(b"RIFF")
        assert not work.exists()

    def test_removes_its_directory_on_exception(self):
        with pytest.raises(RuntimeError, match="boom"):
            with ascii_safe_workspace(boundary=BOUNDARY) as work:
                (work / "utterance.wav").write_bytes(b"RIFF")
                raise RuntimeError("boom")
        assert not work.exists()

    def test_nested_and_concurrent_scopes_are_isolated(self):
        with ascii_safe_workspace(boundary=BOUNDARY) as outer:
            with ascii_safe_workspace(boundary=BOUNDARY) as inner:
                assert inner != outer
                assert inner.is_dir() and outer.is_dir()
            assert not inner.exists()
            assert outer.is_dir(), "内側の退出が外側を消してはいけない"

    def test_many_threads_get_distinct_workspaces(self):
        seen: list[Path] = []
        lock = threading.Lock()

        def worker() -> None:
            with ascii_safe_workspace(boundary=BOUNDARY) as work:
                with lock:
                    seen.append(work)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)

        assert len(seen) == 8
        assert len(set(seen)) == 8, "workspace が共有された"


class TestAsymmetry:
    """2 つの API の削除挙動を**並べて**固定する。

    別々のテストに散らすと、片方だけ直したときに意図が崩れたことに気づけない。
    """

    def test_temp_env_keeps_and_workspace_removes(self):
        with ascii_safe_temp_environment(boundary=BOUNDARY) as env_dir:
            pass
        with ascii_safe_workspace(boundary=BOUNDARY) as work_dir:
            pass

        assert env_dir.is_dir(), (
            "temp environment は消してはいけない — プロセス全体の TEMP を向けて "
            "いた間に他スレッドのファイルが入り得る (#386)"
        )
        assert not work_dir.exists(), (
            "workspace は消すべき — env を触っていないので自分のファイルしか無い"
        )
