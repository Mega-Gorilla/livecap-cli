"""``unicode_safe_download_directory()`` のデータ消失回帰テスト (Issue #386)。

このヘルパは **プロセス全体**の ``TEMP`` / ``TMP`` / ``TMPDIR`` /
``tempfile.tempdir`` を書き換える。したがってスコープが開いている間は、
**無関係なスレッドの ``NamedTemporaryFile()`` もその移設先へ落ちる**。

以前はスコープ退出時に移設先を ``shutil.rmtree`` していたため、別処理が
使用中の一時ファイル (発話ごとの wav を含む) まで削除していた。

**「呼び出しごとの固有ディレクトリにすれば消してよい」は成立しない** —
固有ディレクトリにしても無関係なファイルはそこへ入るので、結果は同じである。
したがって修正の中核は **eager な再帰削除の廃止**であり、ここではそれを固定する。
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import threading
from pathlib import Path

import pytest

from livecap_cli.resources import _reset_resources_for_tests
from livecap_cli.utils import (
    TempEnvironmentConflictError,
    unicode_safe_download_directory,
    unicode_safe_temp_directory,
)
# TEMP 移設の状態は livecap_cli.paths.temp_env へ移った (Issue #375 PR 2)。
# utils 側は委譲だけなので、状態を見るテストは移設先を見る。
from livecap_cli.paths import temp_env as temp_env_mod

_ENV_KEYS = ("TEMP", "TMP", "TMPDIR")


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """cache root を tmp_path へ向け、プロセス全体の状態を必ず元へ戻す。

    このテスト群は本物の ``os.environ`` と ``tempfile.tempdir`` を触るので、
    後始末を fixture 側で保証する (テストが途中で落ちても他へ波及させない)。
    """
    _reset_resources_for_tests()
    monkeypatch.setenv("LIVECAP_CORE_CACHE_DIR", str(tmp_path / "cache"))

    saved_env = {key: os.environ.get(key) for key in _ENV_KEYS}
    saved_tempdir = tempfile.tempdir
    try:
        yield tmp_path / "cache"
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        tempfile.tempdir = saved_tempdir
        _reset_resources_for_tests()


@pytest.fixture(autouse=True)
def depth_is_balanced():
    """どのテストの後でもスコープ深度が 0 に戻っていること。

    深度が漏れると以降のテストが「ネスト」と誤認して全滅するため、
    不変条件として毎回検査する。
    """
    yield
    assert temp_env_mod._TEMP_ENV_STATE["depth"] == 0
    assert temp_env_mod._TEMP_ENV_STATE["path"] is None


def _snapshot() -> dict:
    snap = {key: os.environ.get(key) for key in _ENV_KEYS}
    snap["tempdir"] = tempfile.tempdir
    return snap


def test_outermost_scope_gets_a_unique_directory(isolated_cache):
    """最外周スコープごとに downloads 配下の固有ディレクトリを使う。"""
    with unicode_safe_download_directory() as first:
        pass
    with unicode_safe_download_directory() as second:
        pass

    downloads = isolated_cache / "downloads"
    assert first.parent == downloads
    assert second.parent == downloads
    assert first != second, "スコープごとに固有のディレクトリになっていない"
    assert first.is_dir() and second.is_dir()


def test_unrelated_thread_file_survives_scope_exit():
    """**本 issue の核心**: 別スレッドが作った一時ファイルが消されないこと。

    ``base_engine`` 経由の発話 wav (``dir=`` 未指定) がまさにこの経路に乗る。
    """
    victim: dict = {}
    entered = threading.Event()
    created = threading.Event()

    def unrelated_worker() -> None:
        entered.wait(timeout=10)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(b"utterance audio")
            victim["path"] = handle.name
        created.set()

    worker = threading.Thread(target=unrelated_worker, daemon=True)
    worker.start()

    with unicode_safe_download_directory() as scope_dir:
        entered.set()
        assert created.wait(timeout=10), "別スレッドが一時ファイルを作れなかった"
    worker.join(timeout=10)

    path = Path(victim["path"])
    # 前提の確認 — リダイレクトされていなければテストが無意味になる
    assert path.parent == scope_dir, "一時ファイルがスコープ配下に落ちていない"
    assert path.exists(), "スコープ退出で別スレッドのファイルが削除された (#386 の再発)"


def test_child_process_file_survives_scope_exit():
    """子プロセスが継承した TEMP に作ったファイルも消されないこと。

    子は親の ``os.environ`` を継承するので、移設先は親と同じになる。
    """
    script = textwrap.dedent(
        """
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as fh:
            fh.write(b'child payload')
            print(fh.name)
        """
    )
    with unicode_safe_download_directory() as scope_dir:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        child_path = Path(proc.stdout.strip())
        assert child_path.parent == scope_dir, "子プロセスが TEMP を継承していない"

    assert child_path.exists(), "スコープ退出で子プロセスのファイルが削除された"


def test_scope_exit_does_not_recursively_delete():
    """スコープ中に置いたものが、退出後もディレクトリごと残ること。"""
    with unicode_safe_download_directory() as scope_dir:
        marker = scope_dir / "keep-me.txt"
        marker.write_text("payload", encoding="utf-8")
        nested = scope_dir / "sub" / "deep.txt"
        nested.parent.mkdir(parents=True)
        nested.write_text("payload", encoding="utf-8")

    assert scope_dir.is_dir()
    assert marker.exists()
    assert nested.exists()


def test_nested_scope_reuses_outer_directory():
    """ネストは外側と同じ path を返し、実際の出力先と一致すること。

    内側が自前のディレクトリを作っても環境は外側を指したままなので、
    別 path を返すと**返却値と ``tempfile`` の出力先が食い違う**。
    """
    with unicode_safe_download_directory() as outer:
        with unicode_safe_download_directory() as inner:
            assert inner == outer
            assert Path(tempfile.gettempdir()) == outer


def test_inner_exit_does_not_restore_outer_environment():
    """内側の退出で環境を戻さず、最外周の退出でだけ元へ戻すこと。"""
    before = _snapshot()

    with unicode_safe_download_directory() as outer:
        with unicode_safe_download_directory():
            pass
        # 内側を抜けても外側の移設は生きている
        assert Path(tempfile.gettempdir()) == outer
        for key in _ENV_KEYS:
            assert os.environ[key] == str(outer)

    assert _snapshot() == before


def test_environment_is_restored_on_exception():
    """例外で抜けても TEMP 系 4 項目が元へ戻ること。"""
    before = _snapshot()

    with pytest.raises(RuntimeError, match="boom"):
        with unicode_safe_download_directory():
            raise RuntimeError("boom")

    assert _snapshot() == before


def test_concurrent_outermost_scopes_are_serialized():
    """別スレッドの最外周スコープが直列化されること。

    プロセス全体の状態を書き換える以上、並行実行は一貫させられない
    (内側の snapshot が外側の上書き済みの値を掴む)。
    """
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    dirs: dict = {}

    def hold_first() -> None:
        with unicode_safe_download_directory() as path:
            dirs["first"] = path
            first_entered.set()
            release_first.wait(timeout=10)

    def try_second() -> None:
        with unicode_safe_download_directory() as path:
            dirs["second"] = path
            second_entered.set()

    holder = threading.Thread(target=hold_first, daemon=True)
    holder.start()
    assert first_entered.wait(timeout=10)

    contender = threading.Thread(target=try_second, daemon=True)
    contender.start()

    # 1 本目が保持している間は 2 本目が入れない
    assert not second_entered.wait(timeout=1.0), "並行スコープが直列化されていない"

    release_first.set()
    holder.join(timeout=10)
    assert second_entered.wait(timeout=10), "解放後も 2 本目が入れない"
    contender.join(timeout=10)

    assert dirs["first"] != dirs["second"]


def test_conflicting_purpose_raises():
    """purpose が違うネストは、嘘の path を返さず失敗すること。"""
    with unicode_safe_download_directory():
        with pytest.raises(TempEnvironmentConflictError, match="downloads"):
            with unicode_safe_temp_directory():
                pass


def test_runtime_helper_keeps_its_directory(isolated_cache):
    """``unicode_safe_temp_directory()`` の返す path が変わっていないこと。"""
    with unicode_safe_temp_directory() as temp_dir:
        assert temp_dir == isolated_cache / "runtime"
        assert Path(tempfile.gettempdir()) == temp_dir
