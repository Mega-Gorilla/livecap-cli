"""Resource graph の central factory (Issue #375)。

**このモジュールが `ModelManager` / `ResourceLocator` / `FFmpegManager` を構築
する唯一の場所である。** 他所で直接構築すると、その instance だけが frozen
configuration の外側に立ち、「設定したのに効かない」が再発する。
``tests/core/resources/test_resource_graph.py`` が AST で検査している。

以前は ``FFmpegManager.__init__`` が private に ``ResourceLocator()`` と
``ModelManager()`` を作っており、``get_model_manager()`` が返す instance とは
**別の root を持ち得た**。ここで 1 度だけ組み立てて注入することで解消する。
"""
from __future__ import annotations

from dataclasses import dataclass

from .configuration import ResourceConfiguration
from .ffmpeg_manager import FFmpegManager
from .model_manager import ModelManager
from .resource_locator import ResourceLocator

__all__ = ["ResourceGraph", "build_resource_graph"]


@dataclass(frozen=True, slots=True)
class ResourceGraph:
    """1 つの frozen configuration から作られた manager 一式。

    graph 全体が 1 つの configuration に対応するため、``reset_resource_graph()``
    は「graph を捨てて作り直す」だけで済み、configuration は動かない。
    """

    configuration: ResourceConfiguration
    model_manager: ModelManager
    resource_locator: ResourceLocator
    ffmpeg_manager: FFmpegManager


def build_resource_graph(configuration: ResourceConfiguration) -> ResourceGraph:
    """解決済み configuration から manager 一式を組み立てる。

    構築順が意味を持つ: ``FFmpegManager`` は locator と model manager を**注入
    される**ので、先に 2 つを作る。ここで渡さないと ``FFmpegManager`` 側の
    fallback (shared graph の getter) が働き、graph 構築中の再入になる。

    既定 root の作成はここで起きる。``configure_resources()`` は明示指定 root しか
    検証せず、preview は filesystem を触らないため、**実際に使う段階まで作成を
    遅らせる**のが本 issue の契約である。
    """
    model_manager = ModelManager(
        models_root=configuration.models.resolved,
        cache_root=configuration.cache.resolved,
    )
    resource_locator = ResourceLocator(
        search_roots=configuration.resource_search.effective_roots
    )
    ffmpeg_manager = FFmpegManager(
        locator=resource_locator, model_manager=model_manager
    )
    return ResourceGraph(
        configuration=configuration,
        model_manager=model_manager,
        resource_locator=resource_locator,
        ffmpeg_manager=ffmpeg_manager,
    )
