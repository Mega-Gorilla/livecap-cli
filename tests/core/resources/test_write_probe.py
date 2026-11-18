"""書き込み probe が有限回で答えを返すこと (Issue #468)。

以前の 3 箇所の probe は ``tempfile.mkstemp`` を使っていた。CPython の
``_mkstemp_inner`` は Windows で ``PermissionError`` を受けても、
``os.access(dir, W_OK)`` (ACL を見ない) が真なら名前の衝突とみなして再試行を
続けるので、**ACL で書き込みを拒否されたディレクトリでは返らなかった**
(``configure_resources()`` が CPU 100% のまま止まる)。

ここで固定するのは次の 3 点:

- helper は ``FileExistsError`` のときだけ再試行し、回数に上限がある
- 3 つの呼び出し側が helper を通り、それぞれの「書けない」を返す
- Windows で deny ACE を付けた実ディレクトリに対し、helper と 3 経路が有限時間で
  「書けない」を返す (別プロセスで締め切りつきで走らせる — 退行したときに
  テストの実行そのものを止めないため)
"""

from __future__ import annotations

import csv
import errno
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterator, List

import pytest

import livecap_cli
from livecap_cli.paths import roots
from livecap_cli.resources import ResourceConfigurationError, configure_resources
from livecap_cli.resources import configuration, write_probe
from livecap_cli.resources.configuration import STAGING_ROOT_MAX_LEN
from livecap_cli.resources.write_probe import PROBE_ATTEMPTS, probe_write_access
from livecap_cli.runtime import pytorch

PREFIX = ".livecap-test-probe-"


def _record_open(monkeypatch: pytest.MonkeyPatch, error: OSError | None) -> List[str]:
    """``os.open`` を差し替え、呼ばれた path を記録する。``error`` があれば毎回送出する。"""
    calls: List[str] = []
    real_open = os.open

    def fake_open(path, flags, mode=0o777):
        calls.append(str(path))
        if error is not None:
            raise error
        return real_open(path, flags, mode)

    monkeypatch.setattr(write_probe.os, "open", fake_open)
    return calls


class TestProbeWriteAccess:
    def test_writable_directory_passes_and_leaves_nothing(self, tmp_path: Path):
        probe_write_access(tmp_path, prefix=PREFIX)
        assert list(tmp_path.iterdir()) == []

    def test_existing_files_are_not_touched(self, tmp_path: Path):
        sentinel = tmp_path / "important.txt"
        sentinel.write_text("do not touch")

        probe_write_access(tmp_path, prefix=PREFIX)

        assert sentinel.read_text() == "do not touch"
        assert [p.name for p in tmp_path.iterdir()] == ["important.txt"]

    def test_missing_directory_fails_on_the_first_attempt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        calls = _record_open(monkeypatch, error=None)
        with pytest.raises(FileNotFoundError):
            probe_write_access(tmp_path / "missing", prefix=PREFIX)
        assert len(calls) == 1

    @pytest.mark.parametrize(
        "error",
        [
            PermissionError(errno.EACCES, "Permission denied"),
            OSError(errno.EROFS, "Read-only file system"),
        ],
        ids=["PermissionError", "EROFS"],
    )
    def test_other_errors_fail_on_the_first_attempt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: OSError
    ):
        """**``PermissionError`` を名前の衝突と読まない。** #468 の無限ループの原因そのもの。"""
        calls = _record_open(monkeypatch, error=error)
        with pytest.raises(type(error)):
            probe_write_access(tmp_path, prefix=PREFIX)
        assert len(calls) == 1

    def test_name_collisions_retry_up_to_the_bound(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        calls = _record_open(monkeypatch, error=FileExistsError(errno.EEXIST, "File exists"))
        with pytest.raises(FileExistsError, match="no unused probe file name"):
            probe_write_access(tmp_path, prefix=PREFIX)
        assert len(calls) == PROBE_ATTEMPTS
        assert len(set(calls)) == PROBE_ATTEMPTS, "再試行のたびに別の名前を使うこと"

    def test_collision_with_an_existing_file_moves_to_the_next_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """同名のファイルがあれば別の名前で成功し、**既存のファイルは壊さない**。"""
        names = iter(["a" * 16, "b" * 16])
        monkeypatch.setattr(secrets, "token_hex", lambda nbytes: next(names))
        taken = tmp_path / (PREFIX + "a" * 16)
        taken.write_text("do not touch")

        probe_write_access(tmp_path, prefix=PREFIX)

        assert taken.read_text() == "do not touch"
        assert [p.name for p in tmp_path.iterdir()] == [taken.name]


def _deny_probe(monkeypatch: pytest.MonkeyPatch, module) -> List[Path]:
    """呼び出し側の module が持つ ``probe_write_access`` を「書けない」に差し替える。"""
    calls: List[Path] = []

    def denied(directory: Path, *, prefix: str) -> None:
        calls.append(Path(directory))
        raise PermissionError(errno.EACCES, "Permission denied", str(directory))

    monkeypatch.setattr(module, "probe_write_access", denied)
    return calls


class TestCallSitesUseTheBoundedProbe:
    """3 つの呼び出し側が helper を通ること。**どの OS でも**配線を固定する。"""

    def test_configure_resources(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        calls = _deny_probe(monkeypatch, configuration)
        with pytest.raises(ResourceConfigurationError, match="is not writable"):
            configure_resources(models_dir=str(tmp_path / "models"))
        assert calls, "configure_resources() が書き込み probe を通らなかった"

    def test_staging_root_candidate(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        calls = _deny_probe(monkeypatch, roots)
        reason = roots._reject_reason(tmp_path / "root") or ""
        assert reason.startswith("not writable:")
        assert calls == [tmp_path / "root"]

    def test_pytorch_kernel_cache(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        calls = _deny_probe(monkeypatch, pytorch)
        reason = pytorch._reject_reason(tmp_path / "kernels") or ""
        assert reason.startswith("not writable:")
        assert calls == [tmp_path / "kernels"]


# ---------------------------------------------------------------------------
# Windows: deny ACE を付けた実ディレクトリ
# ---------------------------------------------------------------------------

#: ``icacls /deny`` に渡す権限。``WD_AD`` は #468 の報告で使われた形。
_DENY_FORMS = {
    "WD_AD": "(OI)(CI)(WD,AD)",
    "W": "(W)",
}

#: 子プロセスが自分で打ち切る秒数。修正前はここで止まっていた (無限に再試行する)。
_CHILD_DEADLINE_S = 30

#: probe そのものにかかってよい時間。修正後は数 ms で返る。
_PROBE_BUDGET_S = 5.0

_CHILD = r"""
import faulthandler, json, sys, time
from pathlib import Path

faulthandler.dump_traceback_later(float(sys.argv[3]), exit=True)
target, directory = sys.argv[1], Path(sys.argv[2])

if target == "helper":
    from livecap_cli.resources.write_probe import probe_write_access

    def run():
        try:
            probe_write_access(directory, prefix=".livecap-test-probe-")
            return "writable"
        except OSError as error:
            return f"{type(error).__name__}: {error}"
elif target == "configure_resources":
    from livecap_cli.resources import ResourceConfigurationError, configure_resources

    def run():
        try:
            configure_resources(models_dir=str(directory))
            return "configured"
        except ResourceConfigurationError as error:
            return f"ResourceConfigurationError: {error}"
elif target == "staging_root":
    from livecap_cli.paths import roots

    def run():
        return roots._reject_reason(directory) or "accepted"
elif target == "pytorch_kernel_cache":
    from livecap_cli.runtime import pytorch

    def run():
        return pytorch._reject_reason(directory) or "accepted"
else:
    raise SystemExit(f"unknown target: {target}")

start = time.monotonic()
outcome = run()
print(json.dumps({"outcome": outcome, "elapsed": time.monotonic() - start}))
"""

#: 子プロセスの結果が満たすべき接頭辞と部分文字列。
_EXPECTED = {
    "helper": ("PermissionError:", ""),
    "configure_resources": ("ResourceConfigurationError:", "is not writable"),
    "staging_root": ("not writable:", ""),
    "pytorch_kernel_cache": ("not writable:", ""),
}


def _current_user_sid() -> str:
    """現在のユーザーの SID。名前ではなく SID で指定する (ドメインや非 ASCII 名に依らない)。"""
    output = subprocess.run(
        ["whoami", "/user", "/fo", "csv", "/nh"],
        check=True,
        capture_output=True,
    ).stdout.decode("ascii", "replace")
    return next(csv.reader(output.splitlines()))[-1]


@pytest.fixture(params=sorted(_DENY_FORMS))
def write_denied_dir(request: pytest.FixtureRequest) -> Iterator[Path]:
    """現在のユーザーへの deny ACE を付けた一時ディレクトリ。終了時に ACE を外して消す。"""
    if sys.platform != "win32":
        pytest.skip("ACL の deny ACE は Windows のみ")
    icacls = shutil.which("icacls")
    if icacls is None:
        pytest.skip("icacls が無い")

    sid = _current_user_sid()
    # **tmp_path は使わない。** テスト名で長くなると STAGING_ROOT_MAX_LEN を超え、
    # staging の判定が書き込み probe より前に "too long" で返ってしまう。
    directory = Path(tempfile.mkdtemp(prefix="livecap-468-"))
    assert len(str(directory)) <= STAGING_ROOT_MAX_LEN
    subprocess.run(
        [icacls, str(directory), "/deny", f"*{sid}:{_DENY_FORMS[request.param]}"],
        check=True,
        capture_output=True,
    )
    try:
        # deny が実際に効いていることを確かめる。os.open は mkstemp と違って 1 回で返る。
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
        shutil.rmtree(directory, ignore_errors=True)


@pytest.mark.parametrize("target", sorted(_EXPECTED))
def test_write_denied_directory_is_reported_in_bounded_time(
    write_denied_dir: Path, target: str
):
    """#468 の回帰。修正前は 4 つとも ``_mkstemp_inner`` の中で返らなかった。"""
    package_parent = str(Path(livecap_cli.__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (package_parent, env.get("PYTHONPATH")) if p
    )
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, target, str(write_denied_dir), str(_CHILD_DEADLINE_S)],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_CHILD_DEADLINE_S * 2,
    )
    assert proc.returncode == 0, (
        f"{target} did not return within {_CHILD_DEADLINE_S}s "
        f"(exit {proc.returncode}):\n{proc.stderr}"
    )

    result = json.loads(proc.stdout.strip().splitlines()[-1])
    prefix, fragment = _EXPECTED[target]
    assert result["outcome"].startswith(prefix), result["outcome"]
    assert fragment in result["outcome"], result["outcome"]
    assert result["elapsed"] < _PROBE_BUDGET_S, result
