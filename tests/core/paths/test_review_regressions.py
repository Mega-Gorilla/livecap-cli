"""PR 2 のレビューで見つかった契約違反の回帰テスト (Issue #375)。

いずれも「動いているように見えるが保証が破れている」類で、**ユニットテストが
通っている状態で見逃していた**ものである。
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pytest

from livecap_cli.paths import (
    AsciiStagingUnavailableError,
    TempEnvironmentConflictError,
    ascii_safe_temp_environment,
    ascii_safe_workspace,
    roots,
)
from livecap_cli.paths import AsciiPathError
from livecap_cli.paths.lease import fcntl as lease_fcntl
from livecap_cli.paths.lease import hold_lease, is_owned, marker_path
from livecap_cli.paths.reaper import DEFAULT_TTL_HOURS, reap_staging_root, reset_reaper_state
from livecap_cli.resources import (
    _reset_resources_for_tests,
    configure_resources,
    get_resource_configuration,
)
from livecap_cli.resources.configuration import clear_staging_roots

BOUNDARY = "test.boundary"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    for name in ("ProgramData", "SystemDrive", "PUBLIC"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIVECAP_CORE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LIVECAP_CORE_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.delenv("LIVECAP_CORE_ASCII_STAGING_DIR", raising=False)
    monkeypatch.setattr(roots, "STAGING_ROOT_MAX_LEN", 4096)

    _reset_resources_for_tests()
    clear_staging_roots()
    roots.reset_staging_root_cache()
    reset_reaper_state()
    yield
    _reset_resources_for_tests()
    clear_staging_roots()
    roots.reset_staging_root_cache()
    reset_reaper_state()


class TestStagingUseFreezesConfiguration:
    """**staging root を配る操作は configuration を確定させる。**

    preview を読んでいた頃は、初回利用の後に ``configure_resources(staging_root=...)``
    が成功してしまい、**既に配った root と食い違う設定が黙って受け入れられた**。
    """

    def test_first_use_freezes(self):
        assert get_resource_configuration().is_frozen is False

        with ascii_safe_workspace(boundary=BOUNDARY):
            pass

        assert get_resource_configuration().is_frozen is True

    def test_later_configure_is_rejected_instead_of_ignored(self, tmp_path: Path):
        """**黙って無視するのではなく落ちる。**"""
        from livecap_cli.resources.errors import ResourceConfigurationError

        with ascii_safe_workspace(boundary=BOUNDARY):
            pass

        with pytest.raises(ResourceConfigurationError, match="already configured"):
            configure_resources(staging_root=str(tmp_path / "explicit"))

    def test_configure_before_use_wins(self, tmp_path: Path):
        """先に設定していれば当然それが使われる (API > env > default)。"""
        explicit = tmp_path / "explicit"
        configure_resources(staging_root=str(explicit))

        with ascii_safe_workspace(boundary=BOUNDARY) as work:
            assert work.parents[2] == explicit

    def test_cache_follows_the_configuration(self, tmp_path: Path):
        """configuration が入れ替われば root も選び直す。

        素のキャッシュだと、``_reset_resources_for_tests()`` の後も古い root を
        返し続ける。
        """
        with ascii_safe_workspace(boundary=BOUNDARY) as first:
            first_root = first.parents[2]

        _reset_resources_for_tests()
        clear_staging_roots()
        explicit = tmp_path / "explicit"
        configure_resources(staging_root=str(explicit))

        with ascii_safe_workspace(boundary=BOUNDARY) as second:
            assert second.parents[2] == explicit != first_root


class TestExplicitRootFailsLoud:
    """**明示指定が使えなくなったら候補へ降りない** (R2)。"""

    def test_unusable_explicit_root_raises_instead_of_falling_back(self, tmp_path: Path):
        explicit = tmp_path / "explicit"
        configure_resources(staging_root=str(explicit))

        # configure 時には有効だったが、その後使えなくなった状況を作る。
        original = roots._reject_reason

        def broken(path: Path):
            if path == explicit:
                return "permission denied (simulated)"
            return original(path)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(roots, "_reject_reason", broken)
            roots.reset_staging_root_cache()

            with pytest.raises(AsciiStagingUnavailableError) as excinfo:
                roots.select_staging_root(boundary="engine.demo.load")

        message = str(excinfo.value)
        assert "no longer" in message
        assert "engine.demo.load" in message
        assert "permission denied (simulated)" in message
        assert excinfo.value.boundary == "engine.demo.load"

    def test_default_ladder_still_falls_through(self, tmp_path: Path):
        """明示指定が無いときは従来どおり次候補へ降りる。"""
        original = roots._reject_reason
        cache_candidate = None

        def reject_cache(path: Path):
            nonlocal cache_candidate
            if path.name == "ascii-staging":
                cache_candidate = path
                return "simulated failure"
            return original(path)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(roots, "_reject_reason", reject_cache)
            selected = roots.select_staging_root(boundary=BOUNDARY).path

        assert cache_candidate is not None, "前提: cache root 候補が評価された"
        assert selected != cache_candidate


class TestPurposeIsValidated:
    """``purpose`` は公開 API の入力なので、**保証を破れないよう強制する**。"""

    @pytest.mark.parametrize(
        "purpose",
        [
            "日本語",              # 完成した path が非 ASCII になる
            "../outside",          # staging root の外へ出る
            "..",
            "sub/dir",             # separator
            "sub\\dir",
            "",                    # 空
            ".",
            "a" * 17,              # 予算超過
            "has space",
        ],
    )
    def test_invalid_purpose_is_rejected(self, purpose: str):
        with pytest.raises(ValueError, match="purpose"):
            with ascii_safe_workspace(boundary=BOUNDARY, purpose=purpose):
                pass
        with pytest.raises(ValueError, match="purpose"):
            with ascii_safe_temp_environment(boundary=BOUNDARY, purpose=purpose):
                pass

    @pytest.mark.parametrize("purpose", ["runtime", "downloads", "a", "A-1_b", "a" * 16])
    def test_valid_purpose_is_accepted(self, purpose: str):
        with ascii_safe_workspace(boundary=BOUNDARY, purpose=purpose) as work:
            assert str(work).isascii()

    def test_rejection_happens_before_any_directory_is_created(self, tmp_path: Path):
        """**検証は staging root を触る前に行う。** 不正な入力で副作用を残さない。"""
        sandbox = tmp_path / "cache"
        with pytest.raises(ValueError):
            with ascii_safe_workspace(boundary=BOUNDARY, purpose="日本語"):
                pass
        assert not sandbox.exists() or not list(sandbox.rglob("日本語"))


class TestOwnership:
    """**reaper は自分が作った印のある entry にしか触らない。**

    明示 staging root には運用者が**既存のディレクトリ**を指定できる。TTL だけで
    回収すると、その配下の無関係なデータを消す — #386 のデータ消失そのもので、
    レビューで実測されるまでこの実装はまさにそれをしていた。
    """

    @staticmethod
    def _age(path: Path) -> None:
        old = time.time() - (DEFAULT_TTL_HOURS + 1) * 3600
        os.utime(path, (old, old))

    def test_unmarked_directory_is_left_alone(self, tmp_path: Path):
        """**印の無い古いディレクトリは、どれだけ古くても他人のもの。**"""
        root = tmp_path / "configured-staging"
        unrelated = root / "runtime" / "customer-data"
        unrelated.mkdir(parents=True)
        payload = unrelated / "important.db"
        payload.write_text("user data")
        self._age(unrelated)

        assert reap_staging_root(root, force=True) == 0
        assert unrelated.is_dir(), "LiveCap のものでないディレクトリを消してはいけない"
        assert payload.read_text() == "user data"

    def test_marked_directory_is_reaped(self, tmp_path: Path):
        """印があれば従来どおり回収する (上のテストが常に 0 なだけではない証明)。"""
        root = tmp_path / "configured-staging"
        entry = root / "runtime" / "ours"
        entry.mkdir(parents=True)
        marker_path(entry).write_bytes(b"")
        self._age(entry)

        assert reap_staging_root(root, force=True) == 1
        assert not entry.exists()

    def test_marker_outlives_the_scope(self, tmp_path: Path):
        """**所有権は entry と運命を共にする。**

        スコープ退出時にマーカーを消していた頃は、残骸が「印の無いディレクトリ」に
        なり、上の一次防御をすり抜けて回収対象から永久に外れていた。
        """
        entry = tmp_path / "runtime" / "kept"
        entry.mkdir(parents=True)

        with hold_lease(entry, boundary=BOUNDARY):
            assert is_owned(entry)

        assert is_owned(entry), "退出後も所有権の印は残る"


class TestLeaseFailsLoud:
    """**lease を確立できないまま進まない。**

    lease は「唯一の使用中証明」なので、無いまま進むと reaper から見て使用中と
    区別できない entry が生まれる。TTL は猶予であって安全性ではない。
    """

    def test_marker_creation_failure_raises_before_yield(self, tmp_path: Path):
        entry = tmp_path / "runtime" / "blocked"
        entry.mkdir(parents=True)
        # マーカー名をディレクトリにして open を失敗させる。
        marker_path(entry).mkdir()

        with pytest.raises(AsciiPathError) as excinfo:
            with hold_lease(entry, boundary="engine.demo.load"):
                pytest.fail("lease を取れていないのに yield している")

        assert "engine.demo.load" in str(excinfo.value)
        assert excinfo.value.boundary == "engine.demo.load"

    def test_public_api_fails_before_yield(self, tmp_path: Path, monkeypatch):
        """公開 API も同様に、保護なしのディレクトリを渡さない。"""

        def refuse(entry: Path, *, boundary: str):
            raise AsciiPathError(f"{boundary}: simulated", boundary=boundary)

        monkeypatch.setattr("livecap_cli.paths.workspace.hold_lease", refuse)

        with pytest.raises(AsciiPathError):
            with ascii_safe_workspace(boundary=BOUNDARY):
                pytest.fail("lease を取れていないのに yield している")

    @pytest.mark.skipif(lease_fcntl is None, reason="flock による排他判定は POSIX のみ")
    def test_second_holder_does_not_delete_the_first_holders_marker(self, tmp_path: Path):
        """**他人の lease を unlink しない。**

        以前は取得に失敗しても退出時に unlink していた。既存 holder は inode を
        保持したままでも **path が消えるので次の reaper には「印無し」に見え**、
        使用中の entry が回収されてしまう。
        """
        entry = tmp_path / "runtime" / "contended"
        entry.mkdir(parents=True)

        with hold_lease(entry, boundary=BOUNDARY):
            with pytest.raises(AsciiPathError, match="already leased"):
                with hold_lease(entry, boundary="second.holder"):
                    pytest.fail("共有してはいけない")

            assert marker_path(entry).is_file(), "既存 holder のマーカーが消えている"


class TestLease:
    """**TTL だけでは生存判定にならない。**

    このクラスの要点は ``_age()`` を **lease を保持したまま**呼ぶこと。
    ``hold_lease()`` は entry の中にマーカーを作るので、**その副作用で entry の
    mtime が現在時刻に更新される**。素直に「古くしてから lease して reap」と書くと、
    reaper は lease ではなく **TTL で飛ばしてしまい、lease 機構を一度も通らないまま
    緑になる** (実際そう書いていて、Linux CI が別の症状で露呈させた — Windows は
    NTFS がディレクトリ timestamp を遅延更新するためローカルでは気づけなかった)。
    """

    @staticmethod
    def _age(entry: Path) -> None:
        """entry を TTL 超過へ戻す。"""
        old = time.time() - (DEFAULT_TTL_HOURS + 1) * 3600
        os.utime(entry, (old, old))

    def _entry(self, root: Path, name: str) -> Path:
        entry = root / "runtime" / name
        entry.mkdir(parents=True)
        self._age(entry)
        return entry

    def test_leased_entry_survives_the_reaper(self, tmp_path: Path):
        """**lease 機構そのものを通す。**

        lease を保持した状態で TTL 超過にしてから reap する — こうしないと
        「TTL が新しいから飛ばされた」のか「lease が効いた」のか区別できない。
        """
        entry = self._entry(tmp_path, "busy")

        with hold_lease(entry, boundary=BOUNDARY):
            self._age(entry)  # lease は保持したまま TTL だけ超過させる
            assert time.time() - entry.stat().st_mtime > DEFAULT_TTL_HOURS * 3600, (
                "前提: TTL では飛ばされない状態になっている"
            )

            removed = reap_staging_root(tmp_path, force=True)

        assert removed == 0
        assert entry.is_dir(), "lease 中のエントリを消してはいけない"

    def test_unleased_stale_entry_is_removed(self, tmp_path: Path):
        entry = self._entry(tmp_path, "idle")
        marker_path(entry).write_bytes(b"")  # 我々のものである印
        self._age(entry)

        assert reap_staging_root(tmp_path, force=True) == 1
        assert not entry.exists()

    def test_lease_is_released_on_exit(self, tmp_path: Path):
        entry = self._entry(tmp_path, "released")

        with hold_lease(entry, boundary=BOUNDARY):
            pass

        # **マーカーは残る** (所有権)。残るのは lease ではなく印なので、回収を
        # 妨げないことをここで固定する。
        assert is_owned(entry)

        # マーカーの作成・close で mtime が更新されているので TTL を戻してから
        # 確かめる。見たいのは「lease が外れたら回収できる」ことである。
        self._age(entry)
        assert reap_staging_root(tmp_path, force=True) == 1
        assert not entry.exists()

    def test_ttl_refresh_is_not_relied_upon(self, tmp_path: Path):
        """**lease の mtime 副作用を保証として扱わない。**

        マーカーの作成は親ディレクトリの mtime を更新し得るが、**NTFS はディレクトリ
        timestamp を遅延更新する**ため当てにならない (同じコードが ext4 では更新され
        Windows では更新されないのを実測した)。したがって「使用中だから TTL が延びる」
        に寄りかからず、**保護は lease そのもの**で行う。
        """
        entry = self._entry(tmp_path, "recent")

        with hold_lease(entry, boundary=BOUNDARY):
            pass

        self._age(entry)
        assert reap_staging_root(tmp_path, force=True) == 1

    def test_workspace_stays_empty_while_the_entry_is_leased(self, tmp_path: Path):
        """**両立させる。** マーカーは entry の中、消費側にはその子を渡す。

        マーカーを entry の外へ出すと ``rmtree(entry)`` を妨げず、**Windows の保護
        そのものが消える**。かといって消費側のディレクトリに置くと「空を返す」
        契約が破れる。階層を 1 つ分けることで両方成立する。
        """
        with ascii_safe_workspace(boundary=BOUNDARY) as work:
            assert list(work.iterdir()) == [], "帳簿ファイルが混ざっている"
            entry = work.parent
            assert marker_path(entry).is_file(), "entry にマーカーが作られていない"

    def test_reaper_cleans_up_the_marker_too(self, tmp_path: Path):
        entry = self._entry(tmp_path, "orphan")
        marker_path(entry).write_bytes(b"")
        self._age(entry)  # マーカー作成で mtime が動いたので戻す

        reap_staging_root(tmp_path, force=True)

        assert not entry.exists()
        assert not marker_path(entry).exists()


class TestLeaseWrapsTheEnvironmentWindow:
    """**lease は env を書き換える前に確立し、復元し終わるまで保持する。**

    逆順だと「プロセス全体の TEMP が target を指しているのに lease が無い」区間が
    生まれ、その隙に別プロセスの reaper が消せてしまう。
    """

    def test_marker_exists_while_temp_points_at_the_target(self):
        seen = {}

        with ascii_safe_temp_environment(boundary=BOUNDARY) as target:
            seen["temp"] = os.environ["TEMP"]
            seen["owned"] = is_owned(target)

        assert seen["temp"] == str(target)
        assert seen["owned"], "TEMP が向いている間にマーカーが無い"

    def test_marker_is_created_before_the_override(self, monkeypatch):
        """順序そのものを固定する。"""
        from livecap_cli.paths import temp_env as temp_env_module

        order = []
        original_override = temp_env_module._override
        original_hold = temp_env_module.hold_lease

        def traced_override(target: Path):
            order.append("override")
            return original_override(target)

        def traced_hold(entry: Path, *, boundary: str):
            order.append("lease")
            return original_hold(entry, boundary=boundary)

        monkeypatch.setattr(temp_env_module, "_override", traced_override)
        monkeypatch.setattr(temp_env_module, "hold_lease", traced_hold)

        with ascii_safe_temp_environment(boundary=BOUNDARY):
            pass

        assert order == ["lease", "override"]


class TestStagingIsObservable:
    """**staging の発生と root 選定の理由が運用ログで観測できる** (Issue #375 の AC)。

    元の実装は 2 つの穴を持っていた:

    1. root が cache hit すると**即 return** するので、2 回目以降の staging には
       ログが 1 行も無かった
    2. 拒否された候補の理由は、**後続候補が成功した時点で消えていた** — 全滅時の
       例外メッセージにしか載らない

    運用者にとって重要なのは「cache root が選ばれた」ことではなく「``%ProgramData%``
    が長すぎたので cache root へ降りた」ことである。
    """

    def test_every_staging_call_logs_even_on_a_cached_root(self, caplog):
        """**2 回目の別 boundary でもログが出る。** cache hit で黙らない。"""
        caplog.set_level(logging.DEBUG, logger="livecap_cli.paths.roots")

        with ascii_safe_temp_environment(boundary="first.boundary"):
            pass
        first_root = roots._cached[1].path

        caplog.clear()
        with ascii_safe_workspace(boundary="second.boundary"):
            pass

        assert roots._cached[1].path == first_root, "前提: 同じ root を cache から使う"
        messages = [record.getMessage() for record in caplog.records]
        staging = [m for m in messages if "ASCII staging:" in m]
        assert staging, "cache hit でも staging 発生ログが要る"
        assert "boundary=second.boundary" in staging[0]
        assert f"resolved_root={ascii(str(first_root))}" in staging[0]

    def test_first_use_of_a_boundary_is_info(self, caplog):
        """**初回は INFO。** DEBUG だけだと通常の CLI / GUI ログで観測できない。"""
        caplog.set_level(logging.INFO, logger="livecap_cli.paths.roots")

        with ascii_safe_temp_environment(boundary="engine.demo.load"):
            pass

        info = [r for r in caplog.records if r.levelno == logging.INFO]
        assert any("boundary=engine.demo.load" in r.getMessage() for r in info)

    def test_repeats_drop_to_debug(self, caplog):
        """**発話ごとに INFO を出さない。**

        ``ascii_safe_workspace()`` は PR 4 で 5 engine が**発話ごと**に呼ぶ。
        1 発話 1 行 INFO を出すと realtime 転写でログが埋まり、肝心の 1 行が読めなくなる。
        """
        caplog.set_level(logging.DEBUG, logger="livecap_cli.paths.roots")

        for _ in range(3):
            with ascii_safe_workspace(boundary="parakeet.utterance_wav"):
                pass

        staging = [r for r in caplog.records if "ASCII staging:" in r.getMessage()]
        assert len(staging) == 3, "毎回何らかのログは出す"
        assert [r.levelno for r in staging] == [logging.INFO, logging.DEBUG, logging.DEBUG]

    def test_rejected_candidates_and_reasons_survive_a_later_success(self, caplog):
        """**優先候補を落とした理由がログに残る。**

        後続候補が成功すると、拒否理由は例外にも載らずどこにも出なくなっていた。
        """
        caplog.set_level(logging.INFO, logger="livecap_cli.paths.roots")
        original = roots._reject_reason
        rejected = []

        def reject_cache(path: Path):
            if path.name == "ascii-staging":
                rejected.append(path)
                return "too long (simulated)"
            return original(path)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(roots, "_reject_reason", reject_cache)
            with ascii_safe_temp_environment(boundary="engine.demo.load"):
                pass

        assert rejected, "前提: 優先候補を 1 つ落としている"
        message = next(
            r.getMessage() for r in caplog.records if "ASCII staging:" in r.getMessage()
        )
        assert "cache root" in message
        assert "too long (simulated)" in message
        assert "root_source=system temp" in message, "採用された候補も分かる"

    def test_readback_carries_the_source_and_the_fallbacks(self):
        """ログだけでなく **readback からも辿れる**。"""
        from livecap_cli.resources import get_resource_configuration

        original = roots._reject_reason

        def reject_cache(path: Path):
            if path.name == "ascii-staging":
                return "too long (simulated)"
            return original(path)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(roots, "_reject_reason", reject_cache)
            with ascii_safe_workspace(boundary=BOUNDARY):
                pass

        (status,) = get_resource_configuration().staging_roots
        assert status.root_source == "system temp"
        assert any("too long (simulated)" == why for _, why in status.fallbacks)

    def test_mechanism_names_the_api_not_the_root_source(self, caplog):
        """**mechanism と root_source を混ぜない。**

        本 repo では "mechanism" を hardlink / copy の materialization の意味で
        使っている (``tests/nonascii/artifacts.py``)。root の選択元をそこへ入れると
        読み手が誤解するので、ログでは別の key に出し、``StagingRootStatus`` にも
        ``mechanism`` を持たせない。
        """
        from livecap_cli.resources.configuration import StagingRootStatus

        assert not hasattr(StagingRootStatus, "mechanism")

        caplog.set_level(logging.INFO, logger="livecap_cli.paths.roots")
        with ascii_safe_temp_environment(boundary="a.boundary"):
            pass
        with ascii_safe_workspace(boundary="b.boundary"):
            pass

        messages = [r.getMessage() for r in caplog.records if "ASCII staging:" in r.getMessage()]
        assert any("mechanism=temp-environment" in m for m in messages)
        assert any("mechanism=workspace" in m for m in messages)


class TestSourceVolumeMeansTheSource:
    """**``source_volume`` は staging 元であって、採用された root の drive ではない。**

    元の実装は ``splitdrive(selection.path)`` を記録していたため、``D:`` から staging
    しようとして ``C:\\ProgramData\\...`` へ降りた場合に ``"C:"`` が入り、**入力が
    失われて fallback の関係が説明できなく**なっていた。現行 2 API は source を
    持たないのに Windows では ``"C:"`` 等が入る、という食い違いも起きていた。
    """

    def test_current_apis_record_none(self):
        """現行 2 API は source を持たない。**``None`` が正しい。**"""
        from livecap_cli.resources import get_resource_configuration

        with ascii_safe_temp_environment(boundary=BOUNDARY):
            pass

        (status,) = get_resource_configuration().staging_roots
        # **Windows で意味を持つ。** 旧実装は splitdrive(path) を記録していたので
        # ここに "C:" が入っていた (POSIX では空文字 -> None なので露呈しない)。
        assert status.source_volume is None

    def test_workspace_records_none_too(self):
        from livecap_cli.resources import get_resource_configuration

        with ascii_safe_workspace(boundary=BOUNDARY):
            pass

        (status,) = get_resource_configuration().staging_roots
        assert status.source_volume is None

    def test_input_survives_a_fallback_to_another_volume(self, tmp_path: Path):
        """**別ボリュームへ降りても入力が残る。**"""
        from livecap_cli.resources import get_resource_configuration

        original = roots._reject_reason
        rejected = []

        def reject_source_volume(path: Path):
            if path.name.endswith("Staging"):
                rejected.append(path)
                return "not writable (simulated)"
            return original(path)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(roots, "_reject_reason", reject_source_volume)
            selection = roots.select_staging_root(
                boundary=BOUNDARY, source_volume="D:"
            )

        assert rejected, "前提: 同一ボリューム候補を落としている"
        assert selection.source_volume == "D:", "入力が失われている"
        assert selection.root_source != "source volume"

        (status,) = get_resource_configuration().staging_roots
        assert status.source_volume == "D:"
        assert status.path != rejected[0], "前提: 別の root へ降りている"

    def test_same_root_from_different_sources_is_recorded_twice(self):
        """**重複判定は ``(path, source_volume)``。**

        同じ root でも staging 元が違えば別の関係であり、``D:`` からの staging と
        ``E:`` からの staging が同じ fallback 先へ降りたことは**どちらも観測できる
        べき**である。path だけで潰すと 2 本目が黙って消える。
        """
        from livecap_cli.resources import get_resource_configuration

        original = roots._reject_reason

        def reject_source_volume(path: Path):
            if path.name.endswith("Staging"):
                return "not writable (simulated)"
            return original(path)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(roots, "_reject_reason", reject_source_volume)
            first = roots.select_staging_root(boundary=BOUNDARY, source_volume="D:")
            second = roots.select_staging_root(boundary=BOUNDARY, source_volume="E:")

        assert first.path == second.path, "前提: 同じ fallback 先へ降りている"

        statuses = get_resource_configuration().staging_roots
        assert [s.source_volume for s in statuses] == ["D:", "E:"]

    def test_the_same_pair_is_not_duplicated(self):
        """同じ ``(path, source_volume)`` は 1 度だけ。"""
        from livecap_cli.resources import get_resource_configuration

        for _ in range(3):
            with ascii_safe_temp_environment(boundary=BOUNDARY):
                pass

        assert len(get_resource_configuration().staging_roots) == 1


class TestConflictErrorCarriesTheBoundary:
    """診断契約の 1 番目は**境界名**である (#378 §6.8)。"""

    def test_boundary_is_in_the_message_and_the_attribute(self):
        with ascii_safe_temp_environment(boundary="outer.boundary", purpose="runtime"):
            with pytest.raises(TempEnvironmentConflictError) as excinfo:
                with ascii_safe_temp_environment(
                    boundary="inner.boundary", purpose="downloads"
                ):
                    pass

        assert "inner.boundary" in str(excinfo.value)
        assert excinfo.value.boundary == "inner.boundary"
