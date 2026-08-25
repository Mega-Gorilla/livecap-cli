"""共有 resource graph の契約 (Issue #375)。

不具合の核心は「``FFmpegManager`` が使う cache root と ``get_model_manager()``
の cache root が別物になり得る」ことだった。ここで固定するのは **1 つの
configuration から 1 つの graph が組み上がり、全 consumer がそれを見る**こと。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from livecap_cli.resources import (
    _reset_resources_for_tests,
    configure_resources,
    get_ffmpeg_manager,
    get_model_manager,
    get_resource_configuration,
    get_resource_locator,
    reset_resource_graph,
)

MANAGED_TYPES = {"ModelManager", "FFmpegManager", "ResourceLocator"}

#: 唯一の構築点。ここ以外で構築すると、その instance だけが frozen
#: configuration の外側に立つ。
FACTORY = Path("livecap_cli/resources/graph.py")


class TestSharedConfiguration:
    def test_ffmpeg_and_model_manager_share_the_cache_root(self, tmp_path: Path):
        """以前は別々の ``ModelManager()`` を持ち、root が食い違い得た。"""
        configure_resources(data_root=str(tmp_path / "data"))

        assert get_ffmpeg_manager()._model_manager is get_model_manager()
        assert get_ffmpeg_manager()._model_manager.cache_root == get_model_manager().cache_root

    def test_ffmpeg_uses_the_shared_locator(self, tmp_path: Path):
        configure_resources(data_root=str(tmp_path / "data"))
        assert get_ffmpeg_manager()._locator is get_resource_locator()

    def test_all_three_reflect_the_configuration(self, tmp_path: Path):
        config = configure_resources(data_root=str(tmp_path / "data"))

        assert get_model_manager().models_root == config.models_root
        assert get_model_manager().cache_root == config.cache_root
        assert (
            tuple(get_resource_locator()._search_roots)
            == config.resource_search.effective_roots
        )
        assert get_ffmpeg_manager()._cache_dir == config.cache_root / "ffmpeg"

    def test_getters_are_stable(self, tmp_path: Path):
        configure_resources(data_root=str(tmp_path / "data"))
        assert get_model_manager() is get_model_manager()
        assert get_ffmpeg_manager() is get_ffmpeg_manager()
        assert get_resource_locator() is get_resource_locator()


class TestNoPrivateConstruction:
    """runtime code が manager を直接構築していないこと。

    これを許すと、その instance だけが frozen configuration の外側に立ち、
    「ホストが設定したのに効かない」が再発する。**本 PR が直した不具合そのもの**
    なので、コメントではなくテストで固定する。
    """

    def test_only_the_factory_constructs_managers(self):
        offenders: list[str] = []

        for path in sorted(Path("livecap_cli").rglob("*.py")):
            if path == FACTORY:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = (
                    func.id
                    if isinstance(func, ast.Name)
                    else func.attr
                    if isinstance(func, ast.Attribute)
                    else None
                )
                if name in MANAGED_TYPES:
                    offenders.append(f"{path}:{node.lineno} {name}()")

        assert offenders == [], (
            "manager は livecap_cli/resources/graph.py の "
            "build_resource_graph() でのみ構築すること (Issue #375): "
            + ", ".join(offenders)
        )

    def test_the_factory_actually_constructs_them(self):
        """上のテストが「誰も構築していない」で通ってしまわないようにする。"""
        source = FACTORY.read_text(encoding="utf-8")
        for name in MANAGED_TYPES:
            assert f"{name}(" in source, f"{FACTORY} が {name} を構築していない"


class TestDependenciesAreRequired:
    """``FFmpegManager`` は依存を**必須注入**で受け取る。

    既定値を与えて暗黙に shared graph から取れるようにすると、その instance は
    :func:`reset_resource_graph` の管理外に残り、reset 後も古い manager を掴み
    続ける。片方だけ注入すれば hybrid graph も作れてしまう。
    """

    def test_no_argument_construction_is_rejected(self):
        from livecap_cli.resources import FFmpegManager

        with pytest.raises(TypeError):
            FFmpegManager()

    def test_partial_injection_is_rejected(self, tmp_path: Path):
        from livecap_cli.resources import FFmpegManager

        configure_resources(data_root=str(tmp_path / "data"))
        with pytest.raises(TypeError):
            FFmpegManager(locator=get_resource_locator())  # type: ignore[call-arg]


class TestReset:
    def test_production_reset_keeps_the_configuration(self, tmp_path: Path):
        config = configure_resources(data_root=str(tmp_path / "data"))
        before = get_model_manager()

        reset_resource_graph()
        after = get_model_manager()

        assert after is not before
        assert after.models_root == config.models_root
        assert get_resource_configuration().is_frozen is True

    def test_production_reset_rebuilds_every_manager(self, tmp_path: Path):
        configure_resources(data_root=str(tmp_path / "data"))
        before = (get_model_manager(), get_ffmpeg_manager(), get_resource_locator())

        reset_resource_graph()
        after = (get_model_manager(), get_ffmpeg_manager(), get_resource_locator())

        assert all(a is not b for a, b in zip(after, before))
        # 作り直した後も 3 者は互いに整合している
        assert get_ffmpeg_manager()._model_manager is get_model_manager()

    def test_test_helper_clears_the_configuration(self, tmp_path: Path):
        configure_resources(data_root=str(tmp_path / "data"))
        _reset_resources_for_tests()
        assert get_resource_configuration().is_frozen is False


def test_graph_is_not_published_when_construction_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """部分生成された graph を次の呼び出しへ漏らさない。"""
    import livecap_cli.resources as resources

    def boom(_configuration):
        raise RuntimeError("construction failed")

    monkeypatch.setattr(resources, "build_resource_graph", boom)
    with pytest.raises(RuntimeError, match="construction failed"):
        get_model_manager()

    monkeypatch.undo()
    # 壊れた graph が残っていれば、ここで同じ例外が返る
    assert get_model_manager().models_root.is_dir()


def test_failed_graph_construction_does_not_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """**失敗した getter は freeze を成立させない。**

    先に空 request を publish していた頃は、構築に失敗しても「設定済み」に
    なってしまい、その後の ``configure_resources(data_root=...)`` が
    「different settings」で拒否された。**プロセス再起動以外に復旧できない。**
    """
    import livecap_cli.resources as resources

    def boom(_configuration):
        raise RuntimeError("construction failed")

    monkeypatch.setattr(resources, "build_resource_graph", boom)
    with pytest.raises(RuntimeError, match="construction failed"):
        get_model_manager()
    monkeypatch.undo()

    assert get_resource_configuration().is_frozen is False

    # 正しい設定で復旧できる
    config = configure_resources(data_root=str(tmp_path / "data"))
    assert config.models_root == tmp_path / "data" / "models"
    assert get_model_manager().models_root == tmp_path / "data" / "models"
