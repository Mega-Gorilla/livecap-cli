"""Windows の ACL で書き込みを拒否したディレクトリを使うテストの共通部品 (Issue #468 / #504)。

``tempfile`` の再試行が ACL で拒否されたディレクトリで返らなくなる回帰を確かめる
テストが使う。退行したときにテスト全体を止めないよう、確かめる処理は別プロセスで、
締め切りつきで走らせる。
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest

import livecap_cli

#: ``icacls /deny`` に渡す権限。``WD_AD`` は #468 の報告で使われた形。
DENY_FORMS = {
    "WD_AD": "(OI)(CI)(WD,AD)",
    "W": "(W)",
}

#: 子プロセスが自分で打ち切る秒数。修正前はここで止まっていた (無限に再試行する)。
CHILD_DEADLINE_S = 30

#: 確かめる処理そのものにかかってよい時間。修正後は数 ms で返る。
PROBE_BUDGET_S = 5.0


def shadow_module(monkeypatch: pytest.MonkeyPatch, owner: types.ModuleType, name: str, **overrides: Any) -> None:
    """``owner`` が参照する module ``name`` を、``overrides`` だけ差し替えた複製にする。

    ``monkeypatch.setattr(owner.os, "open", ...)`` だと ``os`` module そのものを
    書き換え、プロセス全体の ``os.open`` が変わる。monkeypatch の後始末は autouse
    fixture の teardown より後に走るので、その間も偽物が残る。複製を ``owner``
    にだけ見せれば、他のコードには影響しない。
    """
    real = getattr(owner, name)
    shadow = types.ModuleType(real.__name__)
    shadow.__dict__.update(real.__dict__)
    shadow.__dict__.update(overrides)
    monkeypatch.setattr(owner, name, shadow)


def _current_user_sid() -> str:
    """現在のユーザーの SID。名前ではなく SID で指定する (ドメインや非 ASCII 名に依らない)。"""
    output = subprocess.run(
        ["whoami", "/user", "/fo", "csv", "/nh"],
        check=True,
        capture_output=True,
    ).stdout.decode("ascii", "replace")
    return next(csv.reader(output.splitlines()))[-1]


@pytest.fixture
def short_temp_dir() -> Iterator[Path]:
    """system temp の直下の一時ディレクトリ。終了時に消す。

    **tmp_path は使わない。** tmp_path は system temp の下に
    ``pytest-of-<user>\\pytest-<N>\\<テスト名>`` を重ねるので長くなり、staging root の
    判定 (``STAGING_ROOT_MAX_LEN``) が書き込み probe より前に "too long" で返ってしまう。
    """
    directory = Path(tempfile.mkdtemp(prefix="livecap-acl-"))
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


@contextmanager
def write_denied(directory: Path, form: str) -> Iterator[Path]:
    """``directory`` に現在のユーザーへの deny ACE (``DENY_FORMS[form]``) を付ける。

    抜けるときに ACE を外す。Windows 以外と、deny ACE が書き込みを止めない環境では skip。
    """
    if sys.platform != "win32":
        pytest.skip("ACL の deny ACE は Windows のみ")
    icacls = shutil.which("icacls")
    if icacls is None:
        pytest.skip("icacls が無い")

    sid = _current_user_sid()
    subprocess.run(
        [icacls, str(directory), "/deny", f"*{sid}:{DENY_FORMS[form]}"],
        check=True,
        capture_output=True,
    )
    try:
        # deny が実際に効いていることを確かめる。os.open は tempfile と違って 1 回で返る。
        try:
            handle = os.open(directory / "precheck", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except PermissionError:
            pass
        else:
            os.close(handle)
            pytest.skip("この環境では deny ACE が書き込みを止めない")
        yield directory
    finally:
        subprocess.run(
            [icacls, str(directory), "/remove:d", f"*{sid}"],
            check=False,
            capture_output=True,
        )


def run_in_child(code: str, *args: str) -> Dict[str, Any]:
    """``code`` を締め切りつきの別プロセスで走らせ、最後の行の JSON を返す。

    子は ``faulthandler.dump_traceback_later(<締め切り>, exit=True)`` で自分を
    打ち切るので、``code`` は ``sys.argv[1]`` に締め切りの秒数を受け取ること。
    締め切りに達したら、子の traceback を付けて assert で失敗させる。
    """
    package_parent = str(Path(livecap_cli.__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (package_parent, env.get("PYTHONPATH")) if p)
    proc = subprocess.run(
        [sys.executable, "-c", code, str(CHILD_DEADLINE_S), *args],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=CHILD_DEADLINE_S * 2,
    )
    assert proc.returncode == 0, (
        f"did not return within {CHILD_DEADLINE_S}s (exit {proc.returncode}):\n{proc.stderr}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])
