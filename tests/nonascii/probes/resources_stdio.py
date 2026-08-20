"""リソース root 解決と stdio の境界プローブ (Issue #378)。

``model_manager.roots`` はハーネス自身の**前提条件テスト**も兼ねる —
env 注入が効いていなければ、他の全プローブは非 ASCII を一度も試していない
ことになる。
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

from ..record import ProbeContext, ProbeSkipped
from . import probe


@probe("model_manager.roots")
def model_manager_roots(ctx: ProbeContext) -> dict:
    """``LIVECAP_CORE_MODELS_DIR`` / ``LIVECAP_CORE_CACHE_DIR`` の注入と派生パス。

    ``get_temp_dir`` / ``huggingface_cache`` まで通して、root 注入が
    後続の全境界へ正しく伝播することを確認する。
    """
    try:
        from livecap_cli.resources import get_model_manager
    except ImportError as exc:
        raise ProbeSkipped(f"livecap_cli.resources 未 import: {exc}") from exc

    manager = get_model_manager(force_reset=True)
    ctx.stage("construct")

    models_root = Path(manager.models_root)
    cache_root = Path(manager.cache_root)
    engine_dir = Path(manager.get_models_dir("probe-engine"))
    temp_dir = Path(manager.get_temp_dir("runtime"))
    ctx.stage("resolve_roots")

    with manager.huggingface_cache() as hf_cache:
        hf_exists = Path(hf_cache).exists()
    ctx.stage("huggingface_cache")

    # パスそのものは返さない。「注入した root 配下にあるか」「作成できたか」だけ。
    return {
        "models_root_under_probe_root": str(models_root).startswith(str(ctx.root)),
        "cache_root_under_probe_root": str(cache_root).startswith(str(ctx.root)),
        "models_root_exists": models_root.exists(),
        "cache_root_exists": cache_root.exists(),
        "engine_dir_created": engine_dir.exists(),
        "engine_dir_leaf": engine_dir.name,
        "temp_dir_created": temp_dir.exists(),
        "temp_dir_leaf": temp_dir.name,
        "hf_cache_exists": hf_exists,
    }


@probe("resource_locator.env_root")
def resource_locator_env_root(ctx: ProbeContext) -> dict:
    """``LIVECAP_RESOURCE_ROOT`` 経由の同梱リソース解決。"""
    try:
        from livecap_cli.resources import get_resource_locator
    except ImportError as exc:
        raise ProbeSkipped(f"livecap_cli.resources 未 import: {exc}") from exc

    root = ctx.root / "resources"
    target = root / "probe-assets"
    target.mkdir(parents=True, exist_ok=True)
    (target / "marker.txt").write_text("resource payload", encoding="utf-8")
    ctx.stage("prepare_resource")

    locator = get_resource_locator(force_reset=True)
    resolved = locator.resolve("probe-assets")
    ctx.stage("resolve")

    marker = Path(resolved) / "marker.txt"
    return {
        "resolved_under_probe_root": str(Path(resolved)).startswith(str(ctx.root)),
        "marker_readable": marker.is_file()
        and marker.read_text(encoding="utf-8") == "resource payload",
    }


_EMIT_SOURCE = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    target = Path(sys.argv[1])
    stream = getattr(sys, sys.argv[2])
    print("Transcribing: {}".format(target), file=stream)
    sys.stderr.write("DONE" + chr(10))
    """
)


def _emit_probe(ctx: ProbeContext, stream: str) -> dict:
    """孫プロセスを起動し、非 ASCII パスを ``stream`` へ書かせる。

    ``capture_output=True`` によって孫の stdio は**パイプ**になる。これが肝で、
    コンソール接続時 (``WriteConsoleW``) とは挙動が変わる。

    実測 (Windows 11 / ACP=932 / Python 3.11、stdio がパイプの場合):

    - ``sys.stdout``: cp932 / ``surrogateescape``
    - ``sys.stderr``: cp932 / ``backslashreplace``

    つまり **stderr は落ちないが stdout は落ちる**。この差は「非 ASCII パス」
    という括りとは独立した encoding の failure family であり、
    ASCII staging では直らない。
    """
    # **variant セグメント以外は常に ASCII に保つ。** ここでファイル名を
    # 決め打ちで非 ASCII にすると、ASCII-only variant (space_paren) にまで
    # encoding の要因が混入し、variant ごとの機構分離が壊れる。
    target = ctx.root / "audio.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"")
    ctx.stage("prepare")

    proc = subprocess.run(
        [sys.executable, "-c", _EMIT_SOURCE, str(target), stream],
        capture_output=True,  # ← パイプ = locale エンコーダになる
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    ctx.stage("run_child")

    stderr = proc.stderr or ""
    if proc.returncode != 0:
        # 「落ちる」ことが観測結果。パスを名指しして raise することで
        # runner に fail_loud と分類させる (黙って差分を返すのではない)。
        raise RuntimeError(
            f"{stream} へのパス出力が失敗した (exit={proc.returncode}): "
            f"path={target} / stderr_tail={stderr[-400:]}"
        )
    return {
        "exit_code": proc.returncode,
        "reached_done": "DONE" in stderr,
        "raised_unicode_error": "UnicodeEncodeError" in stderr,
    }


@probe("stdio.stderr_path")
def stdio_stderr_path(ctx: ProbeContext) -> dict:
    """``cli.py`` の ``print(f"Transcribing: {...}", file=sys.stderr)``。

    stderr は ``backslashreplace`` なので、ACP で表現できない文字でも
    エスケープされるだけで**落ちない**。
    """
    return _emit_probe(ctx, "stderr")


@probe("stdio.stdout_path")
def stdio_stdout_path(ctx: ProbeContext) -> dict:
    """``cli.py`` の ``sys.stdout.write(build_srt(...))`` 側。

    stdout は ``surrogateescape`` であって ``backslashreplace`` ではないため、
    ACP に無い文字を書くと ``UnicodeEncodeError`` で**落ちる**。
    SRT 本文には認識結果 (任意言語) が乗るので、これは実害のある経路である。
    """
    return _emit_probe(ctx, "stdout")
