"""ASCII 保証された staging root の選定 (Issue #375 PR 2)。

守っているのは「**ネイティブ境界へ渡して安全だと保証できる場所を選ぶ**」こと。
既定の `cache_root` は `appdirs` 由来で**ユーザー名を含む**ため、Windows の
ユーザー名が非 ASCII だとそのままでは使えない。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from livecap_cli.paths import roots
from livecap_cli.resources import freeze_and_snapshot
from livecap_cli.paths.errors import AsciiStagingUnavailableError
from livecap_cli.resources.configuration import STAGING_ROOT_MAX_LEN

BOUNDARY = "test.boundary"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """候補 ladder を tmp_path 配下だけに閉じ込める。

    実環境の `%ProgramData%` などを触ると、テストがマシンの状態に依存する。
    """
    from livecap_cli.resources import _reset_resources_for_tests
    from livecap_cli.resources.configuration import clear_staging_roots

    for name in ("ProgramData", "SystemDrive", "PUBLIC", "TEMP", "TMP", "TMPDIR"):
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


class TestPredicates:
    """述語は ``_reject_reason()`` に 1 つだけ置く。

    以前は ``is_ascii_safe()`` という 1 行の公開 wrapper も並べていたが、
    **消費者が 0 件**なうえ ``_reject_reason()`` が ``.isascii()`` を直接呼んで
    いたので判定が 2 箇所にあった。さらに #378 §6.9 が要求する「``\?\`` 付き
    入力を ``ValueError`` で拒否する」を満たしておらず、**設計書を読んだホストの
    期待と食い違う**状態だった。`ascii_safe_path()` を実装するときに §6 のとおり
    作る。
    """

    def test_non_ascii_candidate_is_rejected(self, tmp_path: Path):
        assert "not ASCII" in (roots._reject_reason(tmp_path / "ユーザー") or "")

    def test_overlong_candidate_is_rejected(self, tmp_path: Path):
        deep = tmp_path / ("d" * (STAGING_ROOT_MAX_LEN + 20))
        reason = roots._reject_reason(deep) or ""
        assert "too long" in reason

    def test_uncreatable_candidate_is_rejected(self, tmp_path: Path):
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        assert "cannot create" in (roots._reject_reason(blocker / "under") or "")

    def test_write_probe_does_not_disturb_existing_files(self, tmp_path: Path):
        """probe が既存ファイルを壊さないこと。

        固定名にすると同名ファイルを truncate してから消すことになる — PR 1 の
        レビューで実際に指摘された経路と同じ。
        """
        candidate = tmp_path / "root"
        candidate.mkdir()
        sentinel = candidate / "important.txt"
        sentinel.write_text("do not touch")

        assert roots._reject_reason(candidate) is None

        assert sentinel.read_text() == "do not touch"
        assert sorted(p.name for p in candidate.iterdir()) == ["important.txt"]


class TestLadderOrder:
    def test_explicit_root_wins(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """明示指定は最優先。**PR 1 が freeze 時に検証済み**なのでここでは信じてよい。"""
        from livecap_cli.resources import configure_resources

        explicit = tmp_path / "explicit"
        configure_resources(staging_root=str(explicit))

        assert roots.select_staging_root(boundary=BOUNDARY).path == explicit

    def test_candidate_order_is_the_contract(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """**順序そのもの**を検証する。

        「実際にどれが選ばれるか」で見ると、テスト環境の path 長 (pytest の
        tmp_path は深い) が述語に引っかかって順序と無関係に結果が変わる。
        守りたいのは ladder の順序なので、候補列で見る。
        """
        monkeypatch.setenv("ProgramData", str(tmp_path / "ProgramData"))
        monkeypatch.setenv("SystemDrive", "C:")
        monkeypatch.setenv("PUBLIC", str(tmp_path / "Public"))

        config = freeze_and_snapshot()
        labels = [label for label, _ in roots._candidates(config, None, boundary=BOUNDARY)]

        assert labels == [
            "%ProgramData%",
            "%SystemDrive%",
            "%PUBLIC%",
            "cache root",
            "system temp",
        ]

    def test_explicit_root_is_first_and_source_volume_second(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """明示指定 -> ソースボリューム -> OS 共有領域 の順。

        ソースボリューム候補は現行の 2 API では使わない (source が無い) が、
        **ladder は順序が契約**なので席を空けてある — 後から先頭へ差し込むと
        既存環境の staging root が黙って移動する。
        """
        from livecap_cli.resources import configure_resources

        configure_resources(staging_root=str(tmp_path / "explicit"))
        monkeypatch.setenv("ProgramData", str(tmp_path / "ProgramData"))

        config = freeze_and_snapshot()
        labels = [label for label, _ in roots._candidates(config, "D:", boundary=BOUNDARY)]

        assert labels[0].startswith("explicit staging root")
        assert labels[1] == "source volume"
        assert labels[2] == "%ProgramData%"

    def test_first_passing_candidate_is_selected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """先頭から順に述語を当て、**最初に通ったもの**を採る。"""
        good = tmp_path / "good"
        seen: list[str] = []
        real = roots._reject_reason

        def only_good(path: Path):
            seen.append(str(path))
            return None if path == good else "rejected for test"

        monkeypatch.setattr(roots, "_candidates", lambda config, sv, *, boundary: [
            ("first", tmp_path / "bad1"),
            ("second", good),
            ("third", tmp_path / "bad2"),
        ])
        monkeypatch.setattr(roots, "_reject_reason", only_good)

        assert roots.select_staging_root(boundary=BOUNDARY).path == good
        # 通ったところで打ち切る — 後続候補は評価しない
        assert seen == [str(tmp_path / "bad1"), str(good)]

    def test_program_data_candidate_never_contains_the_username(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """**ユーザー名そのものを候補 path に混ぜない。**

        非 ASCII なユーザー名を入れたら、ASCII 保証という目的自体が壊れる。
        """
        program_data = tmp_path / "ProgramData"
        program_data.mkdir()
        monkeypatch.setenv("ProgramData", str(program_data))
        monkeypatch.setenv("USERNAME", "ユーザー名")

        selected = roots.select_staging_root(boundary=BOUNDARY).path

        assert "ユーザー名" not in str(selected)
        assert str(selected).isascii()

    def test_falls_through_to_cache_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """OS 共有候補が無ければ cache root へ降りる (述語を通った場合のみ)。

        **長さ述語を緩めてから確かめる。** pytest の ``tmp_path`` は深く
        (CI では ``<tmp>/cache/ascii-staging`` が 121 字になった)、素のままだと
        長さで弾かれて system temp まで落ちる — 見たいのは ladder の降り方で
        あって、テスト環境の path 長ではない。実際の予算が足りることは
        :meth:`TestPathBudget.test_production_shaped_cache_root_fits` が見る。
        """
        monkeypatch.setattr(roots, "STAGING_ROOT_MAX_LEN", 4096)

        selected = roots.select_staging_root(boundary=BOUNDARY).path

        assert selected.name == "ascii-staging"

    def test_non_ascii_cache_root_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """**動機そのもの。** cache root が非 ASCII なら次の候補へ降りる。"""
        monkeypatch.setenv("LIVECAP_CORE_CACHE_DIR", str(tmp_path / "ユーザー" / "cache"))
        system_temp = tmp_path / "systemp"
        system_temp.mkdir()
        monkeypatch.setenv("TEMP", str(system_temp))

        from livecap_cli.resources import _reset_resources_for_tests

        _reset_resources_for_tests()
        roots.reset_staging_root_cache()

        selected = roots.select_staging_root(boundary=BOUNDARY).path

        assert str(selected).isascii()
        assert "ユーザー" not in str(selected)


class TestPathBudget:
    """長さ述語 120 が**実運用の形で足りる**こと。

    120 は #378 §6.5 の設計値で、``\?\`` 接頭辞を一切使わずに Windows の
    MAX_PATH 260 に収めるためのもの。PR 1 は staged path の形が未定だったため
    160 を暫定値にしており、PR 2 で形が確定したので締め直した。
    """

    def test_production_shaped_cache_root_fits(self):
        """既定の cache root は余裕で収まる。"""
        sep = "\\"
        production = sep.join([
            "C:", "Users", "a-fairly-long-username", "AppData", "Local",
            "PineLab", "LiveCap", "Cache", "cache", "ascii-staging",
        ])
        assert len(production) <= STAGING_ROOT_MAX_LEN, (
            f"既定 cache root ({len(production)} 字) が予算を超えている"
        )

    def test_budget_leaves_room_for_the_consumer_subtree(self):
        """消費側のサブツリーに十分残る。

        staged path の形は ``<root>\<purpose>\<uuid12>\...``。NeMo の untar は
        この下へ入れ子を作るので、``root`` を使い切ってはいけない。
        """
        overhead = len("runtime") + 12 + 3  # purpose + uuid12 + 区切り
        remaining = 260 - STAGING_ROOT_MAX_LEN - overhead

        assert remaining >= 100, (
            f"消費側に {remaining} 字しか残らない — 入れ子の展開に足りない"
        )


class TestExhaustion:
    def test_all_candidates_rejected_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """**元の非 ASCII path へ黙って fallback しない。**"""
        monkeypatch.setattr(roots, "_reject_reason", lambda path: "forced failure")

        with pytest.raises(AsciiStagingUnavailableError) as excinfo:
            roots.select_staging_root(boundary="engine.demo.load").path

        error = excinfo.value
        message = str(error)
        assert "engine.demo.load" in message, "境界名がメッセージに無い"
        assert "forced failure" in message, "何を試して何故失敗したかが無い"
        assert "LIVECAP_CORE_ASCII_STAGING_DIR" in message, "対処の env var 名が無い"
        assert error.boundary == "engine.demo.load"
        assert error.attempts, "attempts が構造化されていない"

    def test_error_is_not_an_oserror(self, monkeypatch: pytest.MonkeyPatch):
        """``except OSError`` で握り潰されないこと (#378 §6.8)。"""
        monkeypatch.setattr(roots, "_reject_reason", lambda path: "forced failure")

        with pytest.raises(AsciiStagingUnavailableError) as excinfo:
            roots.select_staging_root(boundary=BOUNDARY).path

        assert not isinstance(excinfo.value, OSError)


class TestSingleExceptionDefinition:
    """同じ条件に**クラスは 1 つだけ**であること。

    PR 2 の実装中に ``resources`` 側と ``paths`` 側へ別々に定義してしまい、
    ``configure_resources()`` が送出したものを ``paths`` 由来の except が
    捕まえられない状態になった (実地確認で発覚)。捕捉側がどちらを掴めばよいか
    分からないのは、silent degradation の温床になる。
    """

    def test_paths_and_resources_export_the_same_class(self):
        from livecap_cli.paths import AsciiStagingUnavailableError as from_paths
        from livecap_cli.resources import AsciiStagingUnavailableError as from_resources

        assert from_paths is from_resources

    def test_configure_time_failure_is_catchable_from_paths(self, tmp_path: Path):
        """**PR 1 が送出したものを PR 2 側の except で拾える。**"""
        from livecap_cli.paths import AsciiStagingUnavailableError
        from livecap_cli.resources import configure_resources

        with pytest.raises(AsciiStagingUnavailableError):
            configure_resources(staging_root=str(tmp_path / "ステージング"))

    def test_both_bases_catch_it(self, monkeypatch: pytest.MonkeyPatch):
        """configuration の失敗としても ASCII path の失敗としても拾える。

        実際にこの例外は**両方**である — configure 時は前者、境界呼び出し時に
        候補が全滅したときは後者。
        """
        from livecap_cli.paths import AsciiPathError
        from livecap_cli.resources.errors import ResourceConfigurationError

        monkeypatch.setattr(roots, "_reject_reason", lambda path: "forced failure")

        with pytest.raises(ResourceConfigurationError):
            roots.select_staging_root(boundary=BOUNDARY).path
        with pytest.raises(AsciiPathError):
            roots.select_staging_root(boundary=BOUNDARY).path


class TestRuntimeStatus:
    def test_selected_root_appears_in_readback(self, tmp_path: Path):
        """AC「staging root が遅延決定・複数 root を表現できる」。"""
        from livecap_cli.resources import get_resource_configuration

        assert get_resource_configuration().staging_roots == ()

        selected = roots.select_staging_root(boundary=BOUNDARY).path
        statuses = get_resource_configuration().staging_roots

        assert [s.path for s in statuses] == [selected]
        assert statuses[0].root_source, "どの候補が採用されたかを readback で辿れる"

        # **拒否の「有無」を assert しない。** どの候補が落ちるかは環境の path 長に
        # 依存する (CI の tmp_path は 120 文字を超えることがある)。固定するのは形と
        # 整合性だけで、中身は拒否を自分で起こすテストが見る
        # (test_review_regressions.py::TestStagingIsObservable)。
        assert all(
            isinstance(where, str) and isinstance(why, str)
            for where, why in statuses[0].fallbacks
        )
        assert statuses[0].root_source not in {
            where.split(":")[0] for where, _ in statuses[0].fallbacks
        }, "採用された候補が拒否リストにも載っている"

    def test_readback_does_not_create_directories(self, tmp_path: Path):
        """**preview が filesystem を触らない契約を壊していない。**"""
        from livecap_cli.resources import get_resource_configuration

        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        before = sorted(p.name for p in sandbox.iterdir())

        get_resource_configuration()

        assert sorted(p.name for p in sandbox.iterdir()) == before == []
