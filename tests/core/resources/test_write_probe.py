"""書き込み probe が有限回で答えを返すこと (Issue #468)。

以前の 3 箇所の probe は ``tempfile.mkstemp`` を使っていた。CPython の
``_mkstemp_inner`` は Windows で ``PermissionError`` を受けても、
``os.access(dir, W_OK)`` (ACL を見ない) が真なら名前の衝突とみなして再試行を
続けるので、**ACL で書き込みを拒否されたディレクトリでは返らなかった**
(``configure_resources()`` が CPU 100% のまま止まる)。

ここで固定するのは次の 3 点 (再試行の上限そのものは ``test_bounded_tempfile.py``):

- probe は ``PermissionError`` などを再試行せず、既存のファイルに触れない
- 3 つの呼び出し側が probe を通り、それぞれの「書けない」を返す
- Windows で deny ACE を付けた実ディレクトリに対し、probe と 3 経路が有限時間で
  「書けない」を返す (別プロセスで締め切りつきで走らせる — 退行したときに
  テストの実行そのものを止めないため)
"""

from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import List

import pytest

from livecap_cli.paths import roots
from livecap_cli.resources import (
    ResourceConfigurationError,
    bounded_tempfile,
    configuration,
    configure_resources,
)
from livecap_cli.resources.configuration import STAGING_ROOT_MAX_LEN
from livecap_cli.resources.write_probe import probe_write_access
from livecap_cli.runtime import pytorch
from tests.core.resources.acl_support import (
    DENY_FORMS,
    PROBE_BUDGET_S,
    run_in_child,
    shadow_module,
    short_temp_dir,  # noqa: F401 - fixture
    write_denied,
)

PREFIX = ".livecap-test-probe-"


def _record_open(monkeypatch: pytest.MonkeyPatch, error: OSError | None) -> List[str]:
    """probe が使う ``os.open`` を差し替え、呼ばれた path を記録する。

    ``error`` があれば毎回送出する。
    """
    calls: List[str] = []

    def fake_open(path, flags, mode=0o777):
        calls.append(str(path))
        if error is not None:
            raise error
        return os.open(path, flags, mode)

    shadow_module(monkeypatch, bounded_tempfile, "os", open=fake_open)
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


def _deny_probe(monkeypatch: pytest.MonkeyPatch, module) -> List[Path]:
    """呼び出し側の module が持つ ``probe_write_access`` を「書けない」に差し替える。"""
    calls: List[Path] = []

    def denied(directory: Path, *, prefix: str) -> None:
        calls.append(Path(directory))
        raise PermissionError(errno.EACCES, "Permission denied", str(directory))

    monkeypatch.setattr(module, "probe_write_access", denied)
    return calls


class TestCallSitesUseTheBoundedProbe:
    """3 つの呼び出し側が probe を通ること。**どの OS でも**配線を固定する。"""

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

_CHILD = r"""
import faulthandler, json, sys, time
from pathlib import Path

faulthandler.dump_traceback_later(float(sys.argv[1]), exit=True)
target, directory = sys.argv[2], Path(sys.argv[3])

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


def _rejected_before_probe(target: str, directory: Path) -> str | None:
    """``target`` が書き込み probe より前に path そのものを理由に拒否するなら、その理由。

    staging root と PyTorch の kernel cache は、まず ASCII か (staging は長さも) を見る。
    system temp が非 ASCII か長い環境では probe まで届かず、この回帰を確かめられない。
    """
    text = str(directory)
    if target in ("staging_root", "pytorch_kernel_cache") and not text.isascii():
        return "the temp directory is not ASCII"
    if target == "staging_root" and len(text) > STAGING_ROOT_MAX_LEN:
        return f"the temp directory is too long ({len(text)} > {STAGING_ROOT_MAX_LEN})"
    return None


@pytest.mark.parametrize("form", sorted(DENY_FORMS))
@pytest.mark.parametrize("target", sorted(_EXPECTED))
def test_write_denied_directory_is_reported_in_bounded_time(
    short_temp_dir: Path, target: str, form: str  # noqa: F811 - fixture
):
    """#468 の回帰。修正前は 4 つとも ``_mkstemp_inner`` の中で返らなかった。"""
    reason = _rejected_before_probe(target, short_temp_dir)
    if reason is not None:
        pytest.skip(f"{target} rejects the path before the write probe: {reason}")

    with write_denied(short_temp_dir, form):
        result = run_in_child(_CHILD, target, str(short_temp_dir))

    prefix, fragment = _EXPECTED[target]
    assert result["outcome"].startswith(prefix), result["outcome"]
    assert fragment in result["outcome"], result["outcome"]
    assert result["elapsed"] < PROBE_BUDGET_S, result
