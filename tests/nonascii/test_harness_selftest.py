"""ハーネス自身の検証 (Issue #378)。

**検証されていないハーネスは、検証されていない証拠しか生まない。**
期待 verdict が既知の仕込みプローブを、モックではなく**実の runner** に通し、
分類が正しいことを assert する。

とりわけ ``selftest.silent_truncation`` を ``fail_silent`` と分類できない
ハーネスは証拠として使えない — それを CI 時点で言わせるのがこのモジュール。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from pathlib import Path

import pytest

from .record import Verdict
from . import runner as runner_mod
from .runner import _mentions_path, derive_verdict
from .registry import REPO_ROOT
from . import roots as roots_mod
from .roots import (
    MAX_PARENT_ROOT_LEN,
    MAX_SESSION_ROOT_LEN,
    SESSION_LOCK_NAME,
    SESSION_MARKER_NAME,
    SESSION_SUFFIX_LEN,
    create_session_root,
    is_usable,
    is_session_in_use,
    reap_stale_sessions,
    release_session_root,
    resolve_base_root,
)
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


class TestPathMentionNormalization:
    """``_mentions_path()`` が Unicode 正規化を跨いで一致すること。

    ``nfd`` variant では、ライブラリがエラーメッセージ中のパスを NFC 化して
    返し得る。素の部分文字列比較だと「パスを名指しした失敗」を「言及なし」と
    誤判定し、**fail_loud を fail_silent に落とす**。verdict の意味そのものが
    変わるので、``paths._roundtrip_ok()`` と同じく NFC に揃えて比較する。
    """

    # NFD (か + 濁点) と NFC (が) — ファイル名として同じものを指す
    NFD = "が"
    NFC = "が"

    def test_nfd_needle_matches_nfc_message(self):
        """探す側が NFD、メッセージが NFC でも言及と判定される。"""
        assert _mentions_path(f"cannot open C:/x/{self.NFC}/model.onnx", [self.NFD]) is True

    def test_nfc_needle_matches_nfd_message(self):
        """逆向き (探す側が NFC、メッセージが NFD) も対称に一致する。"""
        assert _mentions_path(f"cannot open C:/x/{self.NFD}/model.onnx", [self.NFC]) is True

    def test_identical_forms_still_match(self):
        """正規化を挟んでも従来の一致が壊れないこと。"""
        assert _mentions_path(f"cannot open C:/x/{self.NFD}/m.onnx", [self.NFD]) is True

    def test_escaped_form_still_matches(self):
        """cp932 経由で ascii() 表現に化けた場合も従来どおり拾う。"""
        escaped = ascii(self.NFC).strip("'")
        assert _mentions_path(f"cannot open C:/x/{escaped}/m.onnx", [self.NFC]) is True

    def test_unrelated_message_is_not_a_mention(self):
        """無関係なメッセージを言及と誤判定しないこと (偽陽性の防止)。"""
        assert _mentions_path("NVIDIA NeMo is not installed.", [self.NFD]) is False


def test_crash_records_exit_code(selftest_root):
    """ネイティブ abort 相当でも終了コードが証拠として残ること。"""
    result = run_probe("selftest.crash", variant_id=_VARIANT, base_root=selftest_root, timeout_s=90)
    assert result.exit_code == 3


class TestMaterializationIsDeterministic:
    """``materialize_file()`` の観測が**呼び出し回数に依存しない**こと。

    **ASCII control の root は variant を跨いで共有される。** 2 回目以降だけ答えが
    変わると、差分判定が「path と無関係な非決定性」を検出して **error_harness** に
    落ちる — 境界のバグでもないのに証拠が取れなくなる。

    #387 で `engine.voxtral.autoprocessor` の `required_variants` を 1 件から 2 件へ
    増やした瞬間に実際に踏んだ。旧実装は 2 回目に ``"existing"`` を返していた。
    """

    def test_second_call_reports_the_same_mechanism(self, tmp_path):
        from .artifacts import materialize_file

        src = tmp_path / "src.bin"
        src.write_bytes(b"x" * 32)
        dst = tmp_path / "out" / "src.bin"

        first = materialize_file(src, dst)
        second = materialize_file(src, dst)

        assert first == second, (
            "2 回目の観測が 1 回目と違う。共有 control root で variant を跨ぐと "
            f"error_harness になる: {first!r} -> {second!r}"
        )
        assert first in {"hardlink", "copy"}

    def test_tree_materialization_is_stable_across_calls(self, tmp_path):
        """``materialize_tree`` / ``dominant_mechanism`` まで通しても同じこと。"""
        from .artifacts import dominant_mechanism, materialize_tree

        src = tmp_path / "model"
        src.mkdir()
        for name in ("config.json", "weights.bin"):
            (src / name).write_bytes(b"y" * 16)
        dst = tmp_path / "copy"

        first = dominant_mechanism(materialize_tree(src, dst))
        second = dominant_mechanism(materialize_tree(src, dst))

        assert first == second != "mixed", (
            f"tree の実体化機構が呼び出し回数で変わる: {first!r} -> {second!r}"
        )

    @staticmethod
    def _pretend_inode_is_unavailable(monkeypatch) -> None:
        """``st_ino`` が 0 の FS を模擬する。

        Windows の ReFS/ネットワーク FS や一部の Docker volume では inode が
        取れない。そこでは hardlink かどうかを**判定できない**ので、実装は
        「常に copy と答える」契約になっている。
        """
        from . import artifacts

        real_stat = artifacts.os.stat

        def zero_ino(path, *args, **kwargs):
            st = real_stat(path, *args, **kwargs)
            fields = list(st)
            fields[1] = 0  # st_ino
            return os.stat_result(fields)

        monkeypatch.setattr(artifacts.os, "stat", zero_ino)

    def test_inode_unavailable_still_reports_one_answer(self, tmp_path, monkeypatch):
        """**inode が使えない FS でも 1 回目と 2 回目の答えが割れないこと。**

        `os.link` の成功だけで ``"hardlink"`` と答えると、2 回目は
        ``_is_hardlink_of`` が False になって ``"copy"`` へ変わる —
        本クラスが塞いだはずのドリフトが**別の環境で再発**する。
        """
        from .artifacts import materialize_file

        self._pretend_inode_is_unavailable(monkeypatch)

        src = tmp_path / "src.bin"
        src.write_bytes(b"w" * 32)
        dst = tmp_path / "out" / "src.bin"

        first = materialize_file(src, dst)
        second = materialize_file(src, dst)

        assert first == second == "copy", (
            "inode が使えない FS で観測が割れる (hardlink か判定できないので "
            f"常に copy と答える契約): {first!r} -> {second!r}"
        )

    def test_a_real_copy_is_not_reported_as_hardlink(self, tmp_path):
        """**陰性対照**: 中身だけ同じ別実体を hardlink と誤認しないこと。"""
        from .artifacts import materialize_file

        src = tmp_path / "src.bin"
        src.write_bytes(b"z" * 32)
        dst = tmp_path / "out" / "src.bin"
        dst.parent.mkdir(parents=True)
        dst.write_bytes(src.read_bytes())  # hardlink ではない別実体

        assert materialize_file(src, dst) == "copy"


class TestSpawnFailureIsNotEvidence:
    """worker を起動できなかった run が **境界のバグとして記録されない**こと。

    起動失敗は payload=None / exit_code=None / timed_out=False になるため、
    素朴に扱うと「exit 0 なのに成果物なし」= ``fail_silent`` に落ちる。
    それでは一時的なプロセス生成失敗や sandbox の実行制約が
    「非 ASCII path の fail_silent」という**誤った証拠**になってしまう。
    """

    CONTROL_OK = {"payload": {"ok": True, "observation": {"n": 1}}, "stdout": "", "stderr": ""}

    @staticmethod
    def _spawn_failed(message: str = "PermissionError: [Errno 13] Permission denied") -> dict:
        return {
            "payload": None,
            "timed_out": False,
            "exit_code": None,
            "duration_s": 0.01,
            "stdout": "",
            "stderr": message,
            "spawn_failed": True,
            "spawn_error": message,
            "invocation": {"command": ["python", "-m", "tests.nonascii.worker"], "cwd": "."},
        }

    def _verdict(self, control: dict, trial: dict):
        return derive_verdict(
            control=control, trial=trial, variant_id="cjk_kana", variant_segment="ユーザー"
        )

    def test_trial_spawn_failure_is_error_harness(self):
        """trial だけが起動失敗しても fail_silent にしない。"""
        verdict, criteria, detail = self._verdict(self.CONTROL_OK, self._spawn_failed())
        assert verdict == Verdict.ERROR_HARNESS.value
        assert criteria == []
        assert "trial" in detail["spawn_failures"]

    def test_control_spawn_failure_is_error_harness(self):
        """control 側の起動失敗も同じ扱い。"""
        verdict, _, detail = self._verdict(self._spawn_failed(), self.CONTROL_OK)
        assert verdict == Verdict.ERROR_HARNESS.value
        assert "control" in detail["spawn_failures"]

    def test_spawn_error_text_is_preserved(self):
        """切り分けのため、例外の内容が verdict の詳細に残ること。"""
        _, _, detail = self._verdict(self.CONTROL_OK, self._spawn_failed("OSError: [Errno 8] Exec format error"))
        assert "Exec format error" in detail["spawn_failures"]["trial"]

    def test_real_silent_failure_is_still_detected(self):
        """起動できた場合の本物の fail_silent は従来どおり検出されること。

        上の分岐が広すぎて fail_silent を飲み込んでいないことの negative control。
        """
        trial = {
            "payload": None,
            "timed_out": False,
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "invocation": {},
        }
        verdict, criteria, _ = self._verdict(self.CONTROL_OK, trial)
        assert verdict == Verdict.FAIL_SILENT.value
        assert "exit_zero_but_no_result" in criteria

    def test_run_child_records_spawn_failure(self, monkeypatch, tmp_path):
        """``_run_child()`` が ``OSError`` を握り潰さず spawn_failed を残すこと。"""
        def boom(*args, **kwargs):
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(runner_mod.subprocess, "run", boom)
        child = runner_mod._run_child(
            "selftest.pass",
            variant_id="control",
            root=tmp_path,
            is_control=True,
            payload=None,
            timeout_s=5,
            env_extra=None,
        )
        assert child["spawn_failed"] is True
        assert "PermissionError" in child["spawn_error"]
        assert child["payload"] is None

        diag = runner_mod.worker_diagnostics(child)
        assert diag["spawn_failed"] is True
        assert "Permission denied" in diag["spawn_error"]
        assert diag["command"][-2:] == ["-m", "tests.nonascii.worker"]


def test_timeout_leaves_actionable_diagnostics(selftest_root):
    """timeout した worker について、切り分けに足る情報が記録されること。

    **制限付き環境 (子プロセス実行やファイル作成に制約のある sandbox) で
    「90 秒 timeout した」以上のことが分からない状態を作らないための契約。**
    harness 側のバグなのか実行環境の制約なのかを、記録だけで判断できるようにする。
    """
    result = run_probe(
        "selftest.timeout", variant_id=_VARIANT, base_root=selftest_root, timeout_s=5
    )
    assert result.timed_out is True

    diag = result.worker_diagnostics
    assert diag is not None, "timeout したのに診断が残っていない"
    trial = diag["trial"]
    assert trial["timed_out"] is True
    assert trial["command"][-2:] == ["-m", "tests.nonascii.worker"]
    assert Path(trial["cwd"]).is_dir()
    assert trial["timeout_s"] == 5
    assert "exit_code" in trial and "stderr_tail" in trial
    assert trial["sentinel_seen"] is False


def test_healthy_run_carries_no_diagnostics(selftest_root):
    """正常時は診断を残さない (決定性比較・観測比較を汚さないこと)。"""
    result = run_probe("selftest.pass", variant_id=_VARIANT, base_root=selftest_root, timeout_s=90)
    assert result.verdict == Verdict.PASS.value
    assert result.worker_diagnostics is None


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


class TestSessionRootIsolation:
    """並行 run が互いのデータを壊さないこと (レビュー指摘: 再レビュー 1)。

    候補パスは固定名なので、共有親をそのまま base root にすると 2 つの run が
    同じ probe パスを読み書きし、片方の teardown がもう片方の実行中データを
    消してしまう。これは本調査が問題視している
    旧 ``unicode_safe_download_directory`` の「共有ディレクトリを rmtree する」欠陥
    (#386) と**同じ構造**なので、ハーネス自身が繰り返してはならない。
    """

    def test_concurrent_sessions_get_distinct_roots(self, tmp_path):
        parent = tmp_path / "shared-parent"
        parent.mkdir()
        roots = [create_session_root(parent) for _ in range(8)]
        assert len({str(r) for r in roots}) == len(roots), "session root が衝突した"
        for root in roots:
            assert root.parent == parent
            assert root.is_dir()

    def test_teardown_of_one_session_does_not_touch_another(self, tmp_path):
        """片方の後始末が、実行中のもう片方のデータを消さないこと。"""
        parent = tmp_path / "shared-parent"
        parent.mkdir()
        finished = create_session_root(parent)
        running = create_session_root(parent)

        (finished / "variant").mkdir()
        live_artifact = running / "variant" / "model.bin"
        live_artifact.parent.mkdir(parents=True)
        live_artifact.write_bytes(b"in use")

        # conftest の teardown と同じことをする:
        # 使用中ロックを手放してから、自分の session root だけ消す
        release_session_root(finished)
        shutil.rmtree(finished, ignore_errors=True)

        assert not finished.exists()
        assert live_artifact.exists(), "実行中 session のデータが消された"
        assert live_artifact.read_bytes() == b"in use"

    def test_stale_sessions_are_reaped_but_live_ones_are_not(self, tmp_path):
        """異常終了の残骸だけを回収し、生存中の session は残すこと。

        **目的は再現性ではなくディスクの衛生である。** session root は PID + UUID で
        分離され、``materialize_file()`` が参照するのは常に自分の session root 配下
        なので、**残骸が次の run に混入することはない** (``roots.reap_stale_sessions``
        の docstring と `README` を参照)。したがって回収は「あれば嬉しい」程度で、
        所有権と liveness を確認できた stale session だけを消す。
        """
        parent = tmp_path / "shared-parent"
        parent.mkdir()
        stale = create_session_root(parent)
        live = create_session_root(parent)
        _age_session(stale, hours=24)
        # 「異常終了した run」= プロセスが消えてロックが解放された状態
        release_session_root(stale)

        reaped = reap_stale_sessions(parent, max_age_hours=6.0)

        assert stale.name in reaped
        assert not stale.exists()
        assert live.exists(), "生存中の session root を消してはならない"

    def test_reaper_never_raises(self, tmp_path):
        """存在しない親に対しても例外にしない (best-effort)。"""
        assert reap_stale_sessions(tmp_path / "missing") == []


class TestRootFailureIsLoud:
    """root を確保できない状態を **skip で流さない** こと (再レビュー指摘 2)。

    cheap tier を既定スイートに載せている以上、「green = 実際に測った」で
    なければ意味がない。``LIVECAP_NONASCII_ROOT`` の typo・非 ASCII・権限不足が
    skip として流れると、CI green のまま何も測っていない状態になる。
    """

    def _run_pytest(self, env_extra: dict) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env.update(env_extra)
        env.pop("LIVECAP_NONASCII_REAL_MODELS", None)
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                # base root を実際に確保する cheap tier の 1 行なら何でもよい。
                # ここで見たいのは probe の中身ではなく **root 確保の失敗が
                # skip ではなく失敗になるか**である。
                # (以前は utils.download_dir_data_loss の専用テストを指していたが、
                #  同 probe は #375 PR 3 で helper ごと削除された。)
                "tests/nonascii/test_probes.py::test_cheap_boundary"
                "[resources.model_manager.urlretrieve]",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )

    def test_invalid_override_fails_instead_of_skipping(self, tmp_path):
        bad = tmp_path / "ユーザー"
        bad.mkdir()
        proc = self._run_pytest({"LIVECAP_NONASCII_ROOT": str(bad)})
        output = proc.stdout + proc.stderr

        assert proc.returncode != 0, (
            f"無効な LIVECAP_NONASCII_ROOT が失敗にならなかった:\n{output[-2000:]}"
        )
        assert "LIVECAP_NONASCII_ROOT" in output, (
            f"対処方法がメッセージに出ていない:\n{output[-2000:]}"
        )
        assert " skipped" not in output, (
            f"skip で流れている (green のまま未測定になる):\n{output[-2000:]}"
        )

    def test_valid_override_still_runs(self):
        """逆に、正しい override では普通に実行されること (偽陽性が無いこと)。

        **``tmp_path`` は使わない。** pytest の basetemp は host によっては
        100 文字を超え (CI の Windows runner で実測 113 文字)、長さ述語に
        引っかかって「正しい override」ではなくなってしまう。ここで測りたいのは
        override の扱いなので、ladder が返す短い親の下に自前で作る。
        """
        parent, _, _ = resolve_base_root(
            override=None, models_root=None, repo_root=REPO_ROOT
        )
        good = parent / f"override-check-{uuid.uuid4().hex[:8]}"
        good.mkdir(parents=True, exist_ok=False)
        try:
            proc = self._run_pytest({"LIVECAP_NONASCII_ROOT": str(good)})
            output = proc.stdout + proc.stderr
            assert proc.returncode == 0, (
                "正しい override で失敗した:" + chr(10) + output[-2000:]
            )
        finally:
            shutil.rmtree(good, ignore_errors=True)


def _age_session(session: Path, *, hours: float) -> None:
    """所有権マーカーの ``created_at`` を過去にずらす。

    reaper は dir の mtime ではなくマーカーの ``created_at`` を見るので、
    テストもそちらを操作する。
    """
    marker_path = session / SESSION_MARKER_NAME
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    payload["created_at"] = time.time() - hours * 3600
    marker_path.write_text(json.dumps(payload), encoding="utf-8")


class TestReaperOwnership:
    """reaper が**自分の生成物以外**を消さないこと (再レビュー指摘 1)。

    ``LIVECAP_NONASCII_ROOT`` には利用者が任意の既存ディレクトリを指定できる。
    「``run-*`` という名前で古いもの」だけを条件に再帰削除すると、
    利用者の作業ディレクトリを指定された場合に無関係な ``run-backup`` を消してしまう。
    """

    def _stale_foreign_dir(self, parent: Path, name: str) -> Path:
        """利用者の既存ディレクトリを模す (マーカー無し・十分古い)。"""
        target = parent / name
        target.mkdir()
        (target / "important.txt").write_text("do not delete", encoding="utf-8")
        old = time.time() - 90 * 24 * 3600
        os.utime(target, (old, old))
        return target

    @pytest.mark.parametrize("name", ["run-backup", "run-2025", "run-old-data", "runner"])
    def test_foreign_directories_are_never_deleted(self, tmp_path, name):
        parent = tmp_path / "user-workspace"
        parent.mkdir()
        victim = self._stale_foreign_dir(parent, name)

        reaped = reap_stale_sessions(parent, max_age_hours=0)

        assert victim.exists(), f"利用者のディレクトリ {name} が削除された"
        assert (victim / "important.txt").read_text(encoding="utf-8") == "do not delete"
        assert name not in reaped

    def test_name_matching_directory_without_marker_is_kept(self, tmp_path):
        """名前形式は合っていても、マーカーが無ければ削除しないこと。"""
        parent = tmp_path / "user-workspace"
        parent.mkdir()
        lookalike = parent / "run-999-cafebabe"
        lookalike.mkdir()
        (lookalike / "important.txt").write_text("mine", encoding="utf-8")
        old = time.time() - 90 * 24 * 3600
        os.utime(lookalike, (old, old))

        reaped = reap_stale_sessions(parent, max_age_hours=0)

        assert lookalike.exists(), "マーカー無しのディレクトリを削除してはならない"
        assert not reaped

    def test_foreign_marker_magic_is_kept(self, tmp_path):
        """他人が置いた別 magic のマーカーでは削除しないこと。"""
        parent = tmp_path / "user-workspace"
        parent.mkdir()
        lookalike = parent / "run-999-cafebabe"
        lookalike.mkdir()
        (lookalike / SESSION_MARKER_NAME).write_text(
            json.dumps({"magic": "some-other-tool", "schema": 1, "created_at": 0}),
            encoding="utf-8",
        )

        reaped = reap_stale_sessions(parent, max_age_hours=0)

        assert lookalike.exists()
        assert not reaped

    def test_own_marked_session_is_reaped(self, tmp_path):
        """逆に、自分が作った古い session はきちんと回収されること。"""
        parent = tmp_path / "shared-parent"
        parent.mkdir()
        stale = create_session_root(parent)
        _age_session(stale, hours=48)
        release_session_root(stale)  # 所有プロセスが終了した状態にする

        reaped = reap_stale_sessions(parent, max_age_hours=6.0)

        assert stale.name in reaped and not stale.exists()

    def test_marker_is_written_on_creation(self, tmp_path):
        parent = tmp_path / "shared-parent"
        parent.mkdir()
        session = create_session_root(parent)
        marker = json.loads((session / SESSION_MARKER_NAME).read_text(encoding="utf-8"))
        assert marker["magic"] == roots_mod.SESSION_MAGIC
        assert marker["session_id"] == session.name
        assert marker["pid"] == os.getpid()


class TestPathLengthBudget:
    """session suffix 分を予約すること (再レビュー指摘 2)。

    親だけを上限判定すると、後から付く ``/run-<pid>-<uuid>`` の分だけ
    実際の base root が超過し、MAX_PATH の予算保証が成立しない。
    """

    def test_parent_predicate_reserves_session_suffix(self, tmp_path):
        """予約なしなら通る長さの親が、予約ありでは弾かれること。"""
        # session root としては通るが、**親としては**弾かれる長さをちょうど構成する
        target_len = MAX_PARENT_ROOT_LEN + 1
        prefix_len = len(str(tmp_path)) + 1  # tmp_path + セパレータ
        if not (prefix_len < target_len <= MAX_SESSION_ROOT_LEN):
            pytest.skip(f"tmp_path ({prefix_len} 文字) ではこのケースを構成できない")
        long_parent = tmp_path / ("p" * (target_len - prefix_len))
        assert len(str(long_parent)) == target_len

        ok_as_session, _ = is_usable(long_parent)
        ok_as_parent, reason = is_usable(long_parent, limit=MAX_PARENT_ROOT_LEN)

        assert ok_as_session, "session root の上限には収まる長さであること (前提)"
        assert not ok_as_parent, "session suffix 分が親の上限に反映されていない"
        assert "長すぎる" in reason
        # 二重予約になっていないこと (親の上限は session 上限 - suffix ちょうど)
        assert MAX_PARENT_ROOT_LEN == MAX_SESSION_ROOT_LEN - SESSION_SUFFIX_LEN

    def test_session_root_stays_within_budget(self, tmp_path):
        """述語を通った親から作った session root は上限内に収まること。"""
        parent = tmp_path / "shared-parent"
        parent.mkdir()
        ok, reason = is_usable(parent, limit=MAX_PARENT_ROOT_LEN)
        if not ok:
            pytest.skip(f"tmp_path が長すぎる: {reason}")

        session = create_session_root(parent)
        assert len(str(session)) <= MAX_SESSION_ROOT_LEN

    def test_overlong_session_root_fails_loudly(self, tmp_path, monkeypatch):
        """万一 session root が超過したら黙って進まないこと。"""
        parent = tmp_path / "shared-parent"
        parent.mkdir()
        monkeypatch.setattr(roots_mod, "MAX_SESSION_ROOT_LEN", 1)
        with pytest.raises(RuntimeError, match="長すぎる"):
            create_session_root(parent)


_HOLDER_SOURCE = textwrap.dedent(
    """
    import json
    import sys
    from pathlib import Path

    from tests.nonascii.roots import SESSION_MARKER_NAME, create_session_root

    parent = Path(sys.argv[1])
    session = create_session_root(parent)

    # 「実行中だが created_at は十分古い」状態を作る
    marker_path = session / SESSION_MARKER_NAME
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    payload["created_at"] = 0.0
    marker_path.write_text(json.dumps(payload), encoding="utf-8")

    print(session.name, flush=True)
    sys.stdin.read()   # ロックを握ったまま待機する
    """
)


class TestReaperLiveness:
    """**実行中の session を古さだけで消さない** こと (再レビュー指摘)。

    heavy / real_model tier やモデル取得待ち、低速環境では run が閾値を超えて
    走り続け得る。``created_at`` の古さだけで削除すると、その間に始まった別 run が
    **生存中の session root を消してしまう**。生存判定は時間ではなくロックで行う。
    """

    def test_live_session_is_not_reaped_even_when_old(self, tmp_path):
        """別プロセスが保持している古い session を消さないこと。"""
        parent = tmp_path / "shared-parent"
        parent.mkdir()

        holder = subprocess.Popen(
            [sys.executable, "-c", _HOLDER_SOURCE, str(parent)],
            cwd=str(REPO_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        try:
            name = holder.stdout.readline().strip()
            assert name, "holder プロセスが session 名を返さなかった"
            session = parent / name
            assert session.is_dir()

            # created_at は 0 (十分古い) だが、holder が生きている
            reaped = reap_stale_sessions(parent, max_age_hours=0)

            assert session.exists(), (
                "**実行中の session root が削除された。** "
                "古さだけで判定してはならない。"
            )
            assert name not in reaped
        finally:
            holder.stdin.close()
            holder.terminate()
            holder.wait(timeout=30)

        # 所有プロセスが終了すれば残骸として回収できる。
        #
        # **Windows では process の終了と file handle の解放が同時ではない。**
        # `wait()` が返った直後に判定すると「まだ使用中」に見えることがあり、
        # 実際に CI (Windows / 3.12) で 1 度 flaky に落ちた。解放を待ってから判定する。
        # **判定そのものは緩めない** — 生存判定が壊れて常に「使用中」を返すなら、
        # ここは待ち切って下の assert で落ちる。
        deadline = time.monotonic() + 30.0
        while is_session_in_use(session) and time.monotonic() < deadline:
            time.sleep(0.1)

        reaped_after = reap_stale_sessions(parent, max_age_hours=0)
        assert name in reaped_after, f"終了後も回収されない: {reaped_after}"
        assert not (parent / name).exists()

    def test_liveness_check_is_non_destructive(self, tmp_path):
        """生存判定を繰り返しても答えが変わらないこと。

        「ロックファイルを削除できるか」で判定すると**判定自体が破壊的**になり、
        2 回目に「ロックが無い」状態を見て答えが変わってしまう。
        """
        parent = tmp_path / "shared-parent"
        parent.mkdir()
        session = create_session_root(parent)
        lock_path = session / SESSION_LOCK_NAME
        assert lock_path.is_file()

        release_session_root(session)

        first = is_session_in_use(session)
        second = is_session_in_use(session)
        assert first == second is False
        assert lock_path.is_file(), "判定でロックファイルが消えている (破壊的判定)"

    def test_session_without_lock_is_kept(self, tmp_path):
        """ロックが無い session は判断材料が無いので消さないこと。"""
        parent = tmp_path / "shared-parent"
        parent.mkdir()
        session = create_session_root(parent)
        release_session_root(session)
        (session / SESSION_LOCK_NAME).unlink()

        assert is_session_in_use(session) is True
        assert reap_stale_sessions(parent, max_age_hours=0) == []
        assert session.exists()
