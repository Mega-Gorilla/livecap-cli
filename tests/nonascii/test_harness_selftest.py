"""ハーネス自身の検証 (Issue #378)。

**検証されていないハーネスは、検証されていない証拠しか生まない。**
期待 verdict が既知の仕込みプローブを、モックではなく**実の runner** に通し、
分類が正しいことを assert する。

とりわけ ``selftest.silent_truncation`` を ``fail_silent`` と分類できない
ハーネスは証拠として使えない — それを CI 時点で言わせるのがこのモジュール。
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

from .record import Verdict
from .roots import is_usable, resolve_base_root
from .runner import HarnessError, run_probe

pytestmark = pytest.mark.nonascii_paths

_VARIANT = "cjk_kana"

_PLANTED = [
    ("selftest.pass", Verdict.PASS.value, "偽陽性が無いこと"),
    ("selftest.loud", Verdict.FAIL_LOUD.value, "パスを名指しする失敗を loud と分類"),
    (
        "selftest.silent_truncation",
        Verdict.FAIL_SILENT.value,
        "**中核**: 自分でエラーを握り潰す silent corruption を検出",
    ),
    ("selftest.silent_deferred", Verdict.FAIL_SILENT.value, "遅延失敗を検出"),
    ("selftest.silent_mangled", Verdict.FAIL_SILENT.value, "mangler 署名を検出"),
    ("selftest.crash", Verdict.FAIL_LOUD.value, "ネイティブ abort でも隔離が生き残る"),
]


@pytest.fixture(scope="module")
def selftest_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("nonascii-selftest")
    assert str(root).isascii(), "selftest の base root は ASCII でなければならない"
    return root


@pytest.mark.parametrize(
    ("probe_id", "expected", "why"),
    _PLANTED,
    ids=[p[0] for p in _PLANTED],
)
def test_planted_defect_is_classified(selftest_root, probe_id, expected, why):
    result = run_probe(probe_id, variant_id=_VARIANT, base_root=selftest_root, timeout_s=90)
    assert result.verdict == expected, (
        f"{probe_id}: {why} に失敗。expected={expected} actual={result.verdict} "
        f"criteria={result.silent_criteria_hit} exit={result.exit_code} "
        f"notes={result.notes}"
    )


def test_loud_failure_names_the_path(selftest_root):
    """loud 判定はパス言及の検出に基づいていること。"""
    result = run_probe("selftest.loud", variant_id=_VARIANT, base_root=selftest_root, timeout_s=90)
    assert result.error_mentions_path is True


def test_crash_records_exit_code(selftest_root):
    """ネイティブ abort 相当でも終了コードが証拠として残ること。"""
    result = run_probe("selftest.crash", variant_id=_VARIANT, base_root=selftest_root, timeout_s=90)
    assert result.exit_code == 3


@pytest.mark.slow
def test_timeout_is_contained(selftest_root):
    """ハングが run 全体を止めないこと。"""
    result = run_probe(
        "selftest.timeout", variant_id=_VARIANT, base_root=selftest_root, timeout_s=5
    )
    assert result.timed_out is True
    assert result.verdict == Verdict.FAIL_LOUD.value


def test_control_success_is_required(selftest_root):
    """control が通っていない結果は error_harness であり、バグの証拠にしない。"""
    result = run_probe("selftest.pass", variant_id=_VARIANT, base_root=selftest_root, timeout_s=90)
    assert result.control_verdict == Verdict.PASS.value


def test_nonascii_base_root_is_rejected(tmp_path):
    """base root 自体が非 ASCII だと variant を分離できないので拒否すること。"""
    bad = tmp_path / "ユーザー"
    bad.mkdir()
    with pytest.raises(HarnessError):
        run_probe("selftest.pass", variant_id=_VARIANT, base_root=bad, timeout_s=30)


def test_parent_process_state_is_untouched(selftest_root):
    """**ハーネスは親プロセスの env / tempdir を汚さない。**

    これは調査対象そのものの欠陥 (``utils/__init__.py`` の無ロック env 書き換え)
    をハーネスが踏まないことの固定。
    """
    before_env = dict(os.environ)
    before_tempdir = tempfile.tempdir

    for probe_id, _, _ in _PLANTED:
        run_probe(probe_id, variant_id=_VARIANT, base_root=selftest_root, timeout_s=90)

    assert tempfile.tempdir == before_tempdir
    changed = {
        k: (before_env.get(k), os.environ.get(k))
        for k in set(before_env) | set(os.environ)
        if before_env.get(k) != os.environ.get(k)
    }
    assert not changed, f"親プロセスの環境変数が変化した: {changed}"


def test_determinism(selftest_root):
    """同じプローブを 2 回走らせて同じ verdict になること。

    順序依存や状態漏れがあれば落ちる。
    """
    first = run_probe(
        "selftest.silent_deferred", variant_id=_VARIANT, base_root=selftest_root, timeout_s=90
    )
    second = run_probe(
        "selftest.silent_deferred", variant_id=_VARIANT, base_root=selftest_root, timeout_s=90
    )
    assert first.verdict == second.verdict
    assert first.silent_criteria_hit == second.silent_criteria_hit
    assert first.observation == second.observation


class TestBaseRootLadder:
    """ASCII 保証された base root の探索 (レビュー指摘 1)。

    **最も測りたい環境で ハーネスが動かない**、という状態を防ぐための回帰テスト。
    Windows ユーザー名が非 ASCII だと ``tempfile.gettempdir()`` が非 ASCII になり、
    base root もそれに引きずられて session ごと skip されてしまう。
    """

    def test_nonascii_temp_does_not_disable_the_harness(self, tmp_path, monkeypatch):
        """システム %TEMP% が非 ASCII でも ASCII な base root を見つけること。"""
        fake_temp = tmp_path / "ユーザー" / "Temp"
        fake_temp.mkdir(parents=True)
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_temp))

        ascii_home = tmp_path / "ascii_repo"
        ascii_home.mkdir()

        root, label, rejected = resolve_base_root(
            override=None, models_root=None, repo_root=ascii_home
        )
        assert str(root).isascii(), f"非 ASCII な base root が選ばれた: {root!r}"
        assert label, "採用した候補のラベルが記録されていない"

    def test_all_nonascii_candidates_are_rejected_with_reasons(self, tmp_path, monkeypatch):
        """ASCII な候補が一つも無い場合は、理由付きで失敗すること。

        黙って非 ASCII root を使うと「非 ASCII を試したつもりが base root ごと
        非 ASCII だった」という無意味な測定になる。
        """
        nonascii = tmp_path / "ユーザー"
        nonascii.mkdir()
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(nonascii))
        for var in ("ProgramData", "SystemDrive", "PUBLIC"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(sys, "platform", "win32", raising=False)

        with pytest.raises(RuntimeError) as excinfo:
            resolve_base_root(override=None, models_root=None, repo_root=nonascii)
        message = str(excinfo.value)
        assert "非 ASCII" in message, f"落ちた理由が記録されていない: {message}"

    def test_explicit_override_is_never_silently_ignored(self, tmp_path):
        """明示指定が使えない場合、黙って fallback しないこと。

        運用者の明示指示を無視するのは、本調査が問題視している
        silent degradation そのものである。
        """
        bad = tmp_path / "ユーザー"
        bad.mkdir()
        with pytest.raises(RuntimeError, match="LIVECAP_NONASCII_ROOT"):
            resolve_base_root(override=str(bad), models_root=None, repo_root=tmp_path)

    def test_predicate_rejects_nonascii_and_overlong(self, tmp_path):
        ok, reason = is_usable(tmp_path / "ユーザー")
        assert not ok and "非 ASCII" in reason
        ok, reason = is_usable(tmp_path / ("a" * 200))
        assert not ok and "長すぎる" in reason
