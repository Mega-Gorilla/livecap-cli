"""境界ごとのプローブ実装。

各プローブは ``ProbeContext`` を受け取り、JSON 化可能な観測 dict を返す。
戻り値は ASCII control 実行との**比較**にのみ使われるので、golden 値を
持たない (モデル/ライブラリ更新に耐えるため)。

プローブは必ず子プロセス (``tests.nonascii.worker``) で実行される。
親プロセスの ``os.environ`` / ``tempfile.tempdir`` を書き換えてはならない。
"""

from __future__ import annotations

from typing import Callable, Dict

from ..record import ProbeContext

PROBE_IMPLS: Dict[str, Callable[[ProbeContext], dict]] = {}


def probe(probe_id: str):
    """プローブ実装を ``PROBE_IMPLS`` に登録する decorator。"""

    def _register(fn: Callable[[ProbeContext], dict]) -> Callable[[ProbeContext], dict]:
        if probe_id in PROBE_IMPLS:
            raise RuntimeError(f"probe id が重複している: {probe_id}")
        PROBE_IMPLS[probe_id] = fn
        return fn

    return _register


def load_all() -> Dict[str, Callable[[ProbeContext], dict]]:
    """全プローブモジュールを import して ``PROBE_IMPLS`` を埋める。"""
    from . import (  # noqa: F401
        archives,
        audio_io,
        ffmpeg_pipeline,
        hf_stack,
        native_models,
        pytorch_runtime,
        resources_stdio,
        utterance_wav,
        selftest,
        win32,
    )

    return PROBE_IMPLS


__all__ = ["PROBE_IMPLS", "load_all", "probe"]
