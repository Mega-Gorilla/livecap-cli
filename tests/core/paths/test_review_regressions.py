"""PR 2 のレビューで見つかった契約違反の回帰テスト (Issue #375)。

いずれも「動いているように見えるが保証が破れている」類で、**ユニットテストが
通っている状態で見逃していた**ものである。
"""

from __future__ import annotations

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
from livecap_cli.paths.lease import hold_lease, lease_path
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
            selected = roots.select_staging_root(boundary=BOUNDARY)

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


class TestLease:
    """**TTL だけでは生存判定にならない。**

    このクラスの要点は ``_age()`` を **lease を保持したまま**呼ぶこと。
    ``hold_lease()`` は entry の中に lease ファイルを作るので、**その副作用で
    entry の mtime が現在時刻に更新される**。素直に「古くしてから lease して
    reap」と書くと、reaper は lease ではなく **TTL で飛ばしてしまい、lease 機構を
    一度も通らないまま緑になる** (実際そう書いていて、Linux CI が別の症状で
    露呈させた — Windows は NTFS がディレクトリ timestamp を遅延更新するため
    ローカルでは気づけなかった)。
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

        with hold_lease(entry):
            self._age(entry)  # lease は保持したまま TTL だけ超過させる
            assert time.time() - entry.stat().st_mtime > DEFAULT_TTL_HOURS * 3600, (
                "前提: TTL では飛ばされない状態になっている"
            )

            removed = reap_staging_root(tmp_path, force=True)

        assert removed == 0
        assert entry.is_dir(), "lease 中のエントリを消してはいけない"

    def test_unleased_stale_entry_is_removed(self, tmp_path: Path):
        entry = self._entry(tmp_path, "idle")

        assert reap_staging_root(tmp_path, force=True) == 1
        assert not entry.exists()

    def test_lease_is_released_on_exit(self, tmp_path: Path):
        entry = self._entry(tmp_path, "released")

        with hold_lease(entry):
            pass

        assert not lease_path(entry).exists(), "lease を残すと永久に消せなくなる"

        # **lease の作成・削除で mtime が更新されている**ので、TTL を戻してから
        # 確かめる。見たいのは「lease が外れたら回収できる」ことであって、
        # 「使ったばかりの entry が回収されない」ことではない (それは下のテスト)。
        self._age(entry)
        assert reap_staging_root(tmp_path, force=True) == 1
        assert not entry.exists()

    def test_ttl_refresh_is_not_relied_upon(self, tmp_path: Path):
        """**lease の mtime 副作用を保証として扱わない。**

        lease ファイルの作成・削除は親ディレクトリの mtime を更新し得るが、
        **NTFS はディレクトリ timestamp を遅延更新する**ため当てにならない
        (同じコードが ext4 では更新され Windows では更新されないのを実測した)。
        したがって「使用中だから TTL が延びる」に寄りかからず、**保護は lease
        そのもの**で行う。

        ここで固定するのは「TTL を戻せば必ず回収できる」= reaper が mtime の
        気まぐれに左右されないことである。
        """
        entry = self._entry(tmp_path, "recent")

        with hold_lease(entry):
            pass

        self._age(entry)
        assert reap_staging_root(tmp_path, force=True) == 1

    def test_workspace_stays_empty_while_the_entry_is_leased(self, tmp_path: Path):
        """**両立させる。** lease は entry の中、消費側にはその子を渡す。

        lease を entry の外へ出すと ``rmtree(entry)`` を妨げず、**Windows の保護
        そのものが消える**。かといって消費側のディレクトリに置くと「空を返す」
        契約が破れる。階層を 1 つ分けることで両方成立する。
        """
        with ascii_safe_workspace(boundary=BOUNDARY) as work:
            assert list(work.iterdir()) == [], "帳簿ファイルが混ざっている"
            entry = work.parent
            assert lease_path(entry).is_file(), "entry に lease が作られていない"

    def test_reaper_cleans_up_the_lease_file_too(self, tmp_path: Path):
        entry = self._entry(tmp_path, "orphan")
        lease_path(entry).write_bytes(b"")
        self._age(entry)  # lease ファイル作成で mtime が動いたので戻す

        reap_staging_root(tmp_path, force=True)

        assert not entry.exists()
        assert not lease_path(entry).exists()


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
