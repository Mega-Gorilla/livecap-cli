"""`configure_resources()` / `get_resource_configuration()` の契約 (Issue #375)。

ホストが root を設定しても黙って効かない、という不具合が動機なので、ここで
固定するのは主に**「効いたことを観測できる」**性質である — 優先順位、上書きの
可視化、freeze の境界、そして **preview が副作用を持たないこと**。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import pytest

from livecap_cli.resources import (
    AsciiStagingUnavailableError,
    ResourceConfigurationError,
    _reset_resources_for_tests,
    configure_resources,
    get_model_manager,
    get_resource_configuration,
)
from livecap_cli.resources.configuration import (
    ENV_ASCII_STAGING_DIR,
    ENV_CACHE_DIR,
    ENV_MODELS_DIR,
    ENV_RESOURCE_ROOT,
    normalize_path,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """既定値の挙動を見るテストがあるので、env を明示的に空にする。

    ``tests/core/resources/conftest.py`` の autouse fixture が models/cache を
    tmp_path へ向けるが、本 module は優先順位そのものを検証するため、各テストが
    必要な env だけを立てる。
    """
    for name in (ENV_MODELS_DIR, ENV_CACHE_DIR, ENV_RESOURCE_ROOT, ENV_ASCII_STAGING_DIR):
        monkeypatch.delenv(name, raising=False)


class TestPrecedence:
    """R1 — API > env > built-in default。"""

    def test_api_individual_beats_data_root(self, tmp_path: Path):
        config = configure_resources(
            data_root=str(tmp_path / "data"),
            models_dir=str(tmp_path / "explicit-models"),
        )
        assert config.models_root == tmp_path / "explicit-models"
        # cache は data_root からの派生が残る
        assert config.cache_root == tmp_path / "data" / "cache"

    def test_data_root_derives_models_and_cache_only(self, tmp_path: Path):
        config = configure_resources(data_root=str(tmp_path / "data"))
        assert config.models_root == tmp_path / "data" / "models"
        assert config.cache_root == tmp_path / "data" / "cache"
        # 静的 resource の検索 root は派生しない
        assert (tmp_path / "data") not in config.resource_search.effective_roots

    def test_api_beats_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(ENV_MODELS_DIR, str(tmp_path / "from-env"))
        config = configure_resources(models_dir=str(tmp_path / "from-api"))
        assert config.models_root == tmp_path / "from-api"
        assert config.models.source == "api"

    def test_env_beats_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(ENV_MODELS_DIR, str(tmp_path / "from-env"))
        config = get_resource_configuration()
        assert config.models_root == tmp_path / "from-env"
        assert config.models.source == "env"

    def test_default_when_nothing_is_set(self):
        config = get_resource_configuration()
        assert config.models.source in ("default", "fallback")
        assert config.models.configured is None

    def test_configured_keeps_the_raw_value(self, tmp_path: Path):
        """``configured`` は**正規化前**。ホストが書いた文字列がそのまま返る。"""
        raw = str(tmp_path / "data" / "." / "models")
        config = configure_resources(models_dir=raw)
        assert config.models.configured == Path(raw)
        assert config.models.resolved == tmp_path / "data" / "models"

    def test_configured_keeps_the_raw_env_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        raw = str(tmp_path / "env" / "." / "models")
        monkeypatch.setenv(ENV_MODELS_DIR, raw)
        config = get_resource_configuration()
        assert config.models.configured == Path(raw)
        assert config.models.resolved == tmp_path / "env" / "models"


class TestResourceSearchOrder:
    """API と env は**混在しない** (Issue #375 の 2 分岐契約)。"""

    def test_env_branch(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        env_root = tmp_path / "env-root"
        env_root.mkdir()
        monkeypatch.setenv(ENV_RESOURCE_ROOT, str(env_root))

        roots = get_resource_configuration().resource_search.effective_roots
        assert roots[0] == env_root

    def test_api_branch(self, tmp_path: Path):
        api_root = tmp_path / "api-root"
        api_root.mkdir()
        config = configure_resources(resource_root=str(api_root))
        assert config.resource_search.effective_roots[0] == api_root
        assert config.resource_search.source == "api"

    def test_api_excludes_the_env_root_entirely(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """**env root を fallback として残さない。**

        残すとそれは「上書き」ではなく「優先 fallback」であり、R3 の
        ``overridden_env`` が嘘になる。
        """
        env_root = tmp_path / "env-root"
        env_root.mkdir()
        api_root = tmp_path / "api-root"
        api_root.mkdir()
        monkeypatch.setenv(ENV_RESOURCE_ROOT, str(env_root))

        search = configure_resources(resource_root=str(api_root)).resource_search

        assert search.effective_roots[0] == api_root
        assert env_root not in search.effective_roots
        assert [o.name for o in search.overridden_env] == [ENV_RESOURCE_ROOT]
        assert search.overridden_env[0].value == str(env_root)

    def test_extra_roots_come_after_project_and_source(self, tmp_path: Path):
        extra = tmp_path / "extra"
        extra.mkdir()
        roots = configure_resources(
            extra_resource_roots=[str(extra)]
        ).resource_search.effective_roots
        assert roots[-1] == extra
        assert len(roots) == 3  # project, source, extra

    def test_package_fallback_keys_are_reported(self):
        keys = get_resource_configuration().resource_search.package_fallback_keys
        assert "languages" in keys and "fonts" in keys


class TestFailLoud:
    """R2 — 明示指定が使えないときは候補へ落ちず送出する。"""

    def test_unwritable_models_root_raises(self, tmp_path: Path):
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        with pytest.raises(ResourceConfigurationError, match="models root"):
            configure_resources(models_dir=str(blocker / "models"))

    def test_missing_resource_root_raises(self, tmp_path: Path):
        with pytest.raises(ResourceConfigurationError, match="resource root"):
            configure_resources(resource_root=str(tmp_path / "does-not-exist"))

    def test_missing_extra_resource_root_raises(self, tmp_path: Path):
        with pytest.raises(ResourceConfigurationError, match="extra resource root"):
            configure_resources(extra_resource_roots=[str(tmp_path / "nope")])

    def test_non_ascii_staging_root_raises(self, tmp_path: Path):
        with pytest.raises(AsciiStagingUnavailableError, match="not ASCII"):
            configure_resources(staging_root=str(tmp_path / "ステージング"))

    def test_non_ascii_staging_env_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """env も API と等しく fail loud する (R2 は両方に適用される)。"""
        monkeypatch.setenv(ENV_ASCII_STAGING_DIR, str(tmp_path / "ステージング"))
        with pytest.raises(AsciiStagingUnavailableError, match="not ASCII"):
            configure_resources()

    @pytest.mark.skipif(
        not __import__("sys").platform.startswith("win"),
        reason="長さ制限は Windows の MAX_PATH 予算に基づく",
    )
    def test_overlong_staging_root_raises(self, tmp_path: Path):
        deep = tmp_path / ("d" * 200)
        with pytest.raises(AsciiStagingUnavailableError, match="too long"):
            configure_resources(staging_root=str(deep))

    def test_ascii_staging_root_is_accepted(self, tmp_path: Path):
        staging = tmp_path / "staging"
        policy = configure_resources(staging_root=str(staging)).staging_policy
        assert policy.source == "api"
        assert policy.configured_root == staging

    def test_no_explicit_staging_root_is_not_an_error(self):
        """明示指定が無いことは失敗ではない — 候補 ladder は PR 2 の責務。"""
        policy = configure_resources().staging_policy
        assert policy.source is None
        assert policy.configured_root is None

    def test_failed_configure_does_not_freeze(self, tmp_path: Path):
        """送出した設定が残ると、次の正しい呼び出しが「再設定」で弾かれる。"""
        with pytest.raises(AsciiStagingUnavailableError):
            configure_resources(staging_root=str(tmp_path / "日本語"))
        config = configure_resources(data_root=str(tmp_path / "data"))
        assert config.models_root == tmp_path / "data" / "models"


class TestWriteProbeIsNonDestructive:
    """検証用の書き込み probe が既存ファイルを壊さないこと。

    以前は ``.livecap-write-probe`` という固定名へ空バイト列を書いて削除して
    いた。同名のファイルがあれば **truncate したうえで消す**ことになり、
    symlink ならリンク先まで巻き込む。複数プロセスの同時 configure も同じ
    probe を奪い合う。
    """

    def test_existing_file_with_the_probe_name_survives(self, tmp_path: Path):
        models = tmp_path / "models"
        models.mkdir()
        sentinel = models / ".livecap-write-probe"
        sentinel.write_text("do not touch")

        configure_resources(models_dir=str(models))

        assert sentinel.exists(), "probe が既存ファイルを削除した"
        assert sentinel.read_text() == "do not touch", "probe が既存ファイルを truncate した"

    def test_probe_leaves_nothing_behind(self, tmp_path: Path):
        models = tmp_path / "models"
        configure_resources(models_dir=str(models))
        leftovers = [p.name for p in models.iterdir()]
        assert leftovers == [], f"probe が残骸を残した: {leftovers}"


class TestEnvResourceRootIsValidated:
    """R2 は env にも等しく適用される。

    env だけ素通しにすると、存在しない ``LIVECAP_RESOURCE_ROOT`` を設定しても
    configure が成功し、``ResourceLocator.resolve()`` が project/source root へ
    黙って落ちる — **本 PR が防ごうとしている silent degradation そのもの**。
    """

    def test_missing_env_root_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(ENV_RESOURCE_ROOT, str(tmp_path / "does-not-exist"))
        with pytest.raises(ResourceConfigurationError, match=ENV_RESOURCE_ROOT):
            configure_resources()

    def test_env_root_that_is_a_file_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        not_a_dir = tmp_path / "file.txt"
        not_a_dir.write_text("x")
        monkeypatch.setenv(ENV_RESOURCE_ROOT, str(not_a_dir))
        with pytest.raises(ResourceConfigurationError, match="not an existing directory"):
            configure_resources()

    def test_valid_env_root_is_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        env_root = tmp_path / "env-root"
        env_root.mkdir()
        monkeypatch.setenv(ENV_RESOURCE_ROOT, str(env_root))
        search = configure_resources().resource_search
        assert search.source == "env"
        assert search.effective_roots[0] == env_root

    def test_preview_still_does_not_validate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """検証するのは freeze 経路だけ。preview は未検証のまま返す。"""
        monkeypatch.setenv(ENV_RESOURCE_ROOT, str(tmp_path / "does-not-exist"))
        config = get_resource_configuration()
        assert config.is_frozen is False
        assert config.resource_search.source == "env"

    def test_env_root_is_not_validated_when_api_overrides_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """API が勝つとき env root は検索順から外れる。**使わないものは検証しない。**"""
        api_root = tmp_path / "api-root"
        api_root.mkdir()
        monkeypatch.setenv(ENV_RESOURCE_ROOT, str(tmp_path / "does-not-exist"))

        search = configure_resources(resource_root=str(api_root)).resource_search

        assert search.effective_roots[0] == api_root
        assert [o.name for o in search.overridden_env] == [ENV_RESOURCE_ROOT]


class TestOverrideIsObservable:
    """R3 — API が設定済み env を上書きするときは黙って行わない。"""

    def test_warns_and_records(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        env_value = str(tmp_path / "from-env")
        monkeypatch.setenv(ENV_MODELS_DIR, env_value)

        with caplog.at_level(logging.WARNING, logger="livecap_cli.resources.configuration"):
            config = configure_resources(models_dir=str(tmp_path / "from-api"))

        assert [o.name for o in config.models.overridden_env] == [ENV_MODELS_DIR]
        assert config.models.overridden_env[0].value == env_value

        message = caplog.text
        assert ENV_MODELS_DIR in message
        assert env_value in message
        assert str(tmp_path / "from-api") in message

    def test_data_root_override_is_recorded_for_both_roots(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """``data_root`` 経由でも記録する。

        「``LIVECAP_CORE_MODELS_DIR`` で非 ASCII 問題を回避しているユーザーの
        ホストが ``data_root`` を渡すと数 GB の再ダウンロードが起きる」が動機。
        """
        monkeypatch.setenv(ENV_MODELS_DIR, str(tmp_path / "env-models"))
        monkeypatch.setenv(ENV_CACHE_DIR, str(tmp_path / "env-cache"))

        config = configure_resources(data_root=str(tmp_path / "data"))

        assert [o.name for o in config.models.overridden_env] == [ENV_MODELS_DIR]
        assert [o.name for o in config.cache.overridden_env] == [ENV_CACHE_DIR]

    def test_no_warning_when_env_is_unset(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        with caplog.at_level(logging.WARNING, logger="livecap_cli.resources.configuration"):
            config = configure_resources(models_dir=str(tmp_path / "models"))
        assert config.models.overridden_env == ()
        assert caplog.text == ""

    def test_readback_does_not_repeat_the_warning(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        """preview は警告を出さない。readback のたびに積み上がると読めなくなる。"""
        monkeypatch.setenv(ENV_MODELS_DIR, str(tmp_path / "from-env"))
        configure_resources(models_dir=str(tmp_path / "from-api"))

        caplog.clear()  # freeze 時の 1 回目は出て当然。見たいのは readback の分。
        with caplog.at_level(logging.WARNING, logger="livecap_cli.resources.configuration"):
            config = get_resource_configuration()

        assert caplog.text == ""
        # 記録の方は消えない
        assert [o.name for o in config.models.overridden_env] == [ENV_MODELS_DIR]


class TestPreviewHasNoSideEffects:
    """``get_resource_configuration()`` は参照系である。"""

    def test_preview_creates_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """**呼んだだけで root が実体化してはいけない。**

        起動ログに readback を出すホストが、意図せずディレクトリを作ってしまう。
        """
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        monkeypatch.setenv(ENV_MODELS_DIR, str(sandbox / "models"))
        monkeypatch.setenv(ENV_CACHE_DIR, str(sandbox / "cache"))

        before = sorted(p.name for p in sandbox.iterdir())
        config = get_resource_configuration()
        after = sorted(p.name for p in sandbox.iterdir())

        assert before == after == []
        assert config.models_root == sandbox / "models"

    def test_preview_does_not_validate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """使えない root でも preview は送出しない — 未検証だからである。"""
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        monkeypatch.setenv(ENV_MODELS_DIR, str(blocker / "models"))

        config = get_resource_configuration()

        assert config.is_frozen is False
        assert config.models_root == blocker / "models"

    def test_preview_does_not_freeze(self, tmp_path: Path):
        get_resource_configuration()
        # freeze していれば「再設定」で弾かれるはず
        config = configure_resources(data_root=str(tmp_path / "data"))
        assert config.is_frozen is True


class TestFreeze:
    def test_configure_freezes(self, tmp_path: Path):
        assert configure_resources(data_root=str(tmp_path / "d")).is_frozen is True

    def test_manager_access_freezes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(ENV_MODELS_DIR, str(tmp_path / "models"))
        monkeypatch.setenv(ENV_CACHE_DIR, str(tmp_path / "cache"))

        assert get_resource_configuration().is_frozen is False
        get_model_manager()
        assert get_resource_configuration().is_frozen is True

    def test_env_is_pinned_at_freeze(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """freeze 後の env 変更は無視される。"""
        monkeypatch.setenv(ENV_MODELS_DIR, str(tmp_path / "first"))
        configure_resources()

        monkeypatch.setenv(ENV_MODELS_DIR, str(tmp_path / "second"))

        assert get_resource_configuration().models_root == tmp_path / "first"
        assert get_model_manager().models_root == tmp_path / "first"


class TestReconfiguration:
    def test_identical_request_is_a_no_op(self, tmp_path: Path):
        first = configure_resources(data_root=str(tmp_path / "data"))
        second = configure_resources(data_root=str(tmp_path / "data"))
        assert first.models_root == second.models_root

    def test_equivalent_after_normalization_is_a_no_op(self, tmp_path: Path):
        """正規化して同じなら同じ設定。``~`` 展開や ``.`` の有無で弾かない。"""
        configure_resources(data_root=str(tmp_path / "data"))
        second = configure_resources(data_root=str(tmp_path / "data" / "." ))
        assert second.models_root == tmp_path / "data" / "models"

    def test_different_request_raises(self, tmp_path: Path):
        configure_resources(data_root=str(tmp_path / "one"))
        with pytest.raises(ResourceConfigurationError, match="already configured"):
            configure_resources(data_root=str(tmp_path / "two"))

    def test_same_paths_but_different_inputs_raises(self, tmp_path: Path):
        """**path だけで判定しない。**

        ``data_root`` を渡すのと ``models_dir`` / ``cache_dir`` を個別に渡すのは、
        結果の path が同じでもホストの意図が違う。
        """
        data = tmp_path / "data"
        configure_resources(data_root=str(data))
        with pytest.raises(ResourceConfigurationError, match="already configured"):
            configure_resources(
                models_dir=str(data / "models"), cache_dir=str(data / "cache")
            )


class TestNormalization:
    def test_expands_user(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        assert normalize_path("~/models") == tmp_path / "models"

    def test_normalizes_dot_segments(self, tmp_path: Path):
        assert normalize_path(tmp_path / "a" / ".." / "b") == tmp_path / "b"

    def test_makes_relative_absolute(self):
        assert normalize_path("models").is_absolute()

    def test_does_not_follow_symlinks(self, tmp_path: Path):
        """``Path.resolve()`` を使わない理由そのもの。

        ホストが symlink を渡したら、その symlink を指したままにする — 実体を
        追いかけると readback が「渡していない path」を返すことになる。
        """
        target = tmp_path / "real"
        target.mkdir()
        link = tmp_path / "link"
        try:
            link.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlink を作成できない環境")

        assert normalize_path(link) == link
        assert normalize_path(link) != target


class TestSnapshotIsImmutable:
    def test_cannot_be_mutated(self, tmp_path: Path):
        config = configure_resources(data_root=str(tmp_path / "data"))
        with pytest.raises(Exception):
            config.is_frozen = False  # type: ignore[misc]

    def test_readback_returns_a_fresh_instance(self, tmp_path: Path):
        """固定した 1 つを返し続けない。

        configuration の freeze と runtime status の更新は別概念で、PR 2 が加える
        ``staging_roots`` のように**後から変わる情報**を載せられる必要がある。
        """
        configure_resources(data_root=str(tmp_path / "data"))
        first = get_resource_configuration()
        second = get_resource_configuration()
        assert first is not second
        assert first == second

    def test_is_ascii_is_reported_per_root(self, tmp_path: Path):
        config = configure_resources(data_root=str(tmp_path / "data"))
        assert config.models.is_ascii == str(config.models_root).isascii()


class TestConcurrency:
    def test_only_one_configuration_wins(self, tmp_path: Path):
        """configure と初期アクセスの競合で、部分生成された graph を公開しない。"""
        results: list[object] = []
        barrier = threading.Barrier(2)

        def configure() -> None:
            barrier.wait()
            try:
                results.append(configure_resources(data_root=str(tmp_path / "data")))
            except ResourceConfigurationError as error:
                results.append(error)

        def access() -> None:
            barrier.wait()
            try:
                results.append(get_model_manager())
            except Exception as error:  # pragma: no cover - 失敗時の診断用
                results.append(error)

        threads = [threading.Thread(target=configure), threading.Thread(target=access)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)

        assert len(results) == 2
        # どちらの順序でも、確定した configuration は 1 つだけ
        final = get_resource_configuration()
        assert final.is_frozen is True
        assert get_model_manager().models_root == final.models_root

    def test_preview_during_configure_is_consistent(self, tmp_path: Path):
        """preview は freeze しない別経路なので、上の競合テストではカバーされない。"""
        observed: list[bool] = []
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                observed.append(get_resource_configuration().models_root.is_absolute())

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        try:
            configure_resources(data_root=str(tmp_path / "data"))
        finally:
            stop.set()
            thread.join(5)

        assert observed and all(observed)


def test_reset_helpers_differ(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """production reset は configuration を維持し、test helper は消す。"""
    from livecap_cli.resources import reset_resource_graph

    monkeypatch.setenv(ENV_MODELS_DIR, str(tmp_path / "first"))
    first_manager = get_model_manager()

    monkeypatch.setenv(ENV_MODELS_DIR, str(tmp_path / "second"))

    reset_resource_graph()
    rebuilt = get_model_manager()
    assert rebuilt is not first_manager
    assert rebuilt.models_root == tmp_path / "first"  # configuration は動かない

    _reset_resources_for_tests()
    assert get_model_manager().models_root == tmp_path / "second"
