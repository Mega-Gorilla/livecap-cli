"""Resource management helpers for Live Cap core.

ホスト向けの入口は :func:`configure_resources` / :func:`get_resource_configuration` /
:func:`reset_resource_graph` の **3 つ** (Issue #375)。本 module は他にも manager の
getter や :func:`freeze_and_snapshot` を公開しているが、前者は内部配線、後者は
staging core が freeze と snapshot を不可分に行うための内部接続で、ホストが直接
呼ぶ必要はない (``docs/reference/api.md`` の「ホスト向けの入口」節と同一の範囲)。

:func:`reset_resource_graph` は **frozen configuration を維持したまま** graph 全体を
作り直す — root 設定を変更する API ではない。env を読み直す完全 reset は private な
:func:`_reset_resources_for_tests` に限定する。

freeze / reset の契約
--------------------
==================================  ========  ==================================
操作                                freeze    filesystem
==================================  ========  ==================================
``configure_resources()`` 成功      する      **明示指定 root のみ**検証 (作成+probe)
manager getter による graph 初期化  する      既定 root を作成
``get_resource_configuration()``    しない    **一切触らない** (preview)
==================================  ========  ==================================

すべて単一の ``RLock`` 下で行い、**部分生成された graph を公開しない**。
env は freeze 時点で写しを取り、以後の変更は無視する — manager は env を読まない。
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, Optional, Sequence

from .configuration import (
    ENV_ASCII_STAGING_DIR,
    ENV_CACHE_DIR,
    ENV_MODELS_DIR,
    ENV_RESOURCE_ROOT,
    ConfiguredPath,
    OverriddenEnv,
    ResourceConfiguration,
    ResourceRequest,
    ResourceSearchResolution,
    RootResolution,
    StagingPolicy,
    StagingRootStatus,
    resolve_configuration,
)
from .errors import AsciiStagingUnavailableError, ResourceConfigurationError
from .ffmpeg_manager import FFmpegManager, FFmpegNotFoundError, FFmpegUpstreamUnavailable
from .graph import ResourceGraph, build_resource_graph
from .model_manager import ModelManager
from .resource_locator import ResourceLocator

# configure / reset / getter がすべてこの 1 本を共有する。**部分生成された graph
# を公開しない**ための唯一の同期点。``RLock`` は将来の入れ子に対する保険で、
# 現状 graph 構築中に getter が再入することはない (依存は必須注入なので、
# ``FFmpegManager`` が getter を呼ぶ経路が無い)。
_lock = threading.RLock()

_request: Optional[ResourceRequest] = None
_frozen_env: Optional[Dict[str, str]] = None
_graph: Optional[ResourceGraph] = None


def configure_resources(
    *,
    data_root: Optional[str | Path] = None,
    models_dir: Optional[str | Path] = None,
    cache_dir: Optional[str | Path] = None,
    resource_root: Optional[str | Path] = None,
    extra_resource_roots: Optional[Sequence[str | Path]] = None,
    staging_root: Optional[str | Path] = None,
) -> ResourceConfiguration:
    """resource root を設定して configuration を freeze する。

    優先順位は **API > env > built-in default** (R1)。``data_root`` から派生する
    のは ``data_root/"models"`` と ``data_root/"cache"`` だけで、静的 resource の
    検索 root は派生しない。個別指定と ``data_root`` を併用した場合は個別指定が
    勝つ (エラーにはしない)。

    **明示された入力が使えないときは候補へ黙って落ちず送出する** (R2)。API が
    設定済みの env を上書きするときは ``WARNING`` を出し、readback の
    ``overridden_env`` にも載せる (R3)。

    Returns:
        freeze された snapshot (``is_frozen=True``)。

    Raises:
        ResourceConfigurationError: 明示指定 root が使えない、または**すでに
            別の設定で freeze 済み**のとき。同一の入力での再呼び出しは no-op と
            して成功する。
        AsciiStagingUnavailableError: 明示された staging root が ASCII / 長さ /
            書き込み可能の述語を満たさないとき。
    """
    request = ResourceRequest.from_arguments(
        data_root=data_root,
        models_dir=models_dir,
        cache_dir=cache_dir,
        resource_root=resource_root,
        extra_resource_roots=extra_resource_roots,
        staging_root=staging_root,
    )

    with _lock:
        if _request is not None:
            # 一致判定は **静的 configuration 全体**で行う。resolved path だけを
            # 比べると「data_root を渡した」と「models/cache を個別に渡した」が
            # 同じ結果になったときに区別できず、意図が違うのに no-op 成功する。
            if request == _request:
                return _snapshot_locked()
            raise ResourceConfigurationError(
                "resources are already configured with different settings; "
                "configure_resources() must be called once before the resource "
                "graph is used."
            )

        env = dict(os.environ)
        configuration = resolve_configuration(
            request, env, enforce=True, frozen=True
        )
        _set_frozen(request, env)
        return configuration


def get_resource_configuration() -> ResourceConfiguration:
    """現在の解決結果を返す。**freeze しない。**

    まだ freeze されていなければ、env と既定値から組み立てた
    ``is_frozen=False`` の **preview** を返す。

    Note:
        preview は **directory 作成も書き込み probe も行わない**。参照しただけで
        root が実体化するのは副作用として不適切で、起動ログに readback を出す
        ホストが意図せず root を作ってしまうため。したがって preview の
        ``resolved`` が**実際に使えるかは未検証**である。
    """
    with _lock:
        if _request is not None:
            return _snapshot_locked()
        return resolve_configuration(
            ResourceRequest(), dict(os.environ), enforce=False, frozen=False
        )


def freeze_and_snapshot() -> ResourceConfiguration:
    """configuration を **freeze して** snapshot を返す。

    ``get_resource_configuration()`` (freeze しない preview) との違いが要点である。
    **configuration を「使う」操作は freeze しなければならない** — freeze せずに
    使うと、その後の ``configure_resources()`` が成功してしまい、**既に使った値と
    食い違う設定が黙って受け入れられる**。

    manager getter と ``livecap_cli.paths`` の staging root 選定が呼ぶ。どちらも
    「resolved 値に依存して動き出す」操作だからである。
    """
    with _lock:
        return _freeze_locked()


def get_model_manager() -> ModelManager:
    """共有 graph の ``ModelManager``。初回呼び出しで configuration を freeze する。"""
    with _lock:
        return _ensure_graph().model_manager


def get_ffmpeg_manager() -> FFmpegManager:
    """共有 graph の ``FFmpegManager``。初回呼び出しで configuration を freeze する。"""
    with _lock:
        return _ensure_graph().ffmpeg_manager


def get_resource_locator() -> ResourceLocator:
    """共有 graph の ``ResourceLocator``。初回呼び出しで configuration を freeze する。"""
    with _lock:
        return _ensure_graph().resource_locator


def reset_resource_graph() -> None:
    """graph 全体を捨てて作り直す。**frozen configuration は維持する。**

    manager の状態 (FFmpeg の解決キャッシュ等) を捨てたいときに使う。設定を
    変えるためのものではない — 設定を変えるには新しいプロセスを起動すること。

    3 つを個別に作り直す手段は用意しない。一部だけ差し替えられると、graph の
    一部が古い configuration を参照する状態が作れてしまうため。
    """
    global _graph
    with _lock:
        _graph = None


def _reset_resources_for_tests() -> None:
    """configuration も含めて完全に初期化する。**テスト専用。**

    env を読み直したい場合はこちらを使う。production 用の
    :func:`reset_resource_graph` は frozen configuration を維持するため、
    ``monkeypatch.setenv`` の効果が反映されない。
    """
    global _request, _frozen_env, _graph
    with _lock:
        _request = None
        _frozen_env = None
        _graph = None


# ---------------------------------------------------------------------------
# 内部 (すべて _lock を保持した状態で呼ぶこと)
# ---------------------------------------------------------------------------


def _set_frozen(request: ResourceRequest, env: Dict[str, str]) -> None:
    global _request, _frozen_env
    _request = request
    _frozen_env = env


def _snapshot_locked() -> ResourceConfiguration:
    """freeze 済みの入力から snapshot を作り直す。

    毎回新しい instance を返す。configuration の freeze と runtime status の更新
    は別概念であり、PR 2 が加える ``staging_roots`` のように**後から変わる情報**
    を載せられる必要があるため、固定した 1 つを返し続けない。

    ``enforce=False`` なので filesystem は触らず、R3 の警告も再送出しない。
    """
    assert _request is not None and _frozen_env is not None
    return resolve_configuration(
        _request, _frozen_env, enforce=False, frozen=True
    )


def _freeze_locked() -> ResourceConfiguration:
    """まだ freeze されていなければ freeze し、snapshot を返す。**要 lock。**"""
    if _request is None:
        # ホストが configure_resources() を呼ばなかった場合。env と既定値で
        # freeze する。以後の env 変更は無視される。
        request, env = ResourceRequest(), dict(os.environ)
        configuration = resolve_configuration(
            request, env, enforce=True, frozen=True
        )
        _set_frozen(request, env)
        return configuration
    return _snapshot_locked()


def _ensure_graph() -> ResourceGraph:
    """graph を返す。まだ無ければ freeze して構築する。

    **freeze は構築が成功してから publish する。** 先に freeze しておくと、構築が
    失敗したときに空の request だけが確定して残り、その後の
    ``configure_resources(data_root=...)`` が「別の設定」として拒否される —
    プロセスを再起動する以外に復旧できなくなる。**失敗した getter は freeze を
    成立させない。**

    :func:`freeze_and_snapshot` が即座に freeze するのとは commit の意味が違う。
    あちらは「resolved 値を配ってしまう」操作なので、その時点で確定させるのが
    正しい — 後から取り消せる余地が無い。
    """
    global _graph
    if _graph is not None:
        return _graph

    if _request is None:
        request, env = ResourceRequest(), dict(os.environ)
        configuration = resolve_configuration(
            request, env, enforce=True, frozen=True
        )
    else:
        request, env, configuration = _request, _frozen_env, _snapshot_locked()

    graph = build_resource_graph(configuration)
    assert env is not None
    _set_frozen(request, env)
    _graph = graph
    return graph


__all__ = [
    # Configuration API
    "configure_resources",
    "get_resource_configuration",
    "freeze_and_snapshot",
    "reset_resource_graph",
    # Snapshot types
    "ResourceConfiguration",
    "RootResolution",
    "ResourceSearchResolution",
    "StagingPolicy",
    "StagingRootStatus",
    "ConfiguredPath",
    "OverriddenEnv",
    # Errors
    "ResourceConfigurationError",
    "AsciiStagingUnavailableError",
    "FFmpegNotFoundError",
    "FFmpegUpstreamUnavailable",
    # Managers (構築は graph.build_resource_graph() のみ)
    "ModelManager",
    "FFmpegManager",
    "ResourceLocator",
    "get_model_manager",
    "get_ffmpeg_manager",
    "get_resource_locator",
    # Environment variable names
    "ENV_MODELS_DIR",
    "ENV_CACHE_DIR",
    "ENV_RESOURCE_ROOT",
    "ENV_ASCII_STAGING_DIR",
]
