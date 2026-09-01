"""リソース root 解決と stdio の境界プローブ (Issue #378)。

``model_manager.roots`` はハーネス自身の**前提条件テスト**も兼ねる —
env 注入が効いていなければ、他の全プローブは非 ASCII を一度も試していない
ことになる。
"""

from __future__ import annotations

import json
import os
import shutil
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
        from livecap_cli.resources import (
            _reset_resources_for_tests,
            get_model_manager,
        )
    except ImportError as exc:
        raise ProbeSkipped(f"livecap_cli.resources 未 import: {exc}") from exc

    _reset_resources_for_tests()

    manager = get_model_manager()
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
        from livecap_cli.resources import (
            _reset_resources_for_tests,
            get_resource_locator,
        )
    except ImportError as exc:
        raise ProbeSkipped(f"livecap_cli.resources 未 import: {exc}") from exc

    root = ctx.root / "resources"
    target = root / "probe-assets"
    target.mkdir(parents=True, exist_ok=True)
    (target / "marker.txt").write_text("resource payload", encoding="utf-8")
    ctx.stage("prepare_resource")

    _reset_resources_for_tests()

    locator = get_resource_locator()
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


#: ``livecap_cli/`` を非 ASCII 側へ置いた孫プロセスで、``Path(__file__).resolve()``
#: 由来の探索 root がどう解決されるかを見る。**親プロセスでは測れない** —
#: worker 自身は ASCII の repo から import 済みで、``source_root`` は確定済みである。
_SOURCE_ROOT_SOURCE = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path

    expected_root = Path(sys.argv[1])

    import livecap_cli

    module_path = Path(livecap_cli.__file__).resolve()
    if not module_path.is_relative_to(expected_root):
        # **「pass したが元 package を読んでいた」を排除する。**
        # PYTHONPATH より editable install が勝つと、非 ASCII を一度も
        # 通さないまま緑になる。
        sys.stderr.write(
            "imported the original package instead of the copy: "
            + ascii(str(module_path)) + chr(10)
        )
        raise SystemExit(3)

    from livecap_cli.resources import get_resource_configuration, get_resource_locator

    roots = tuple(get_resource_configuration().resource_search.effective_roots)
    resolved = Path(get_resource_locator().resolve("probe-assets"))
    marker = resolved / "marker.txt"

    print(json.dumps({
        "module_under_expected_root": True,
        "effective_root_count": len(roots),
        "effective_roots_under_expected_root": all(
            Path(r).resolve().is_relative_to(expected_root.parent) for r in roots
        ),
        "resolved_under_expected_root": resolved.resolve().is_relative_to(expected_root),
        "marker_readable": marker.is_file()
        and marker.read_text(encoding="utf-8") == "resource payload",
    }))
    """
)


@probe("resources.source_root")
def resources_source_root(ctx: ProbeContext) -> dict:
    """**インストール先が非 ASCII のとき**の ``source_root`` 由来の resource 探索。

    ``livecap_cli/resources/configuration.py`` の ``Path(__file__).resolve()`` から
    ``(project_root, source_root)`` が導かれ、静的 resource の探索 root になる。
    非 ASCII なディレクトリへインストールすると、ここから非 ASCII が流入する。

    **site-packages を丸ごと複製する必要は無い。** ``livecap_cli/`` (2.9 MB) だけを
    ``ctx.root`` 配下へ置き、``PYTHONPATH`` 経由で孫プロセスに import させれば足りる。
    依存は venv の site-packages にそのまま残る。

    **symlink ではなく物理コピーにする。** ``resolve()`` が ASCII 側へ戻ってしまい、
    測ったつもりで何も測っていない状態になる。

    cwd は ASCII scratch に固定する — 非 ASCII にすると「package の場所」と
    「cwd」の 2 つが同時に動き、失敗したときどちらが原因か切り分けられない。
    """
    repo_pkg = Path(__file__).resolve().parents[3] / "livecap_cli"
    if not repo_pkg.is_dir():
        raise ProbeSkipped(f"livecap_cli/ が見つからない: {ascii(str(repo_pkg))}")

    pkg_root = ctx.root / "pkgroot"
    # **control root は variant 間で共有される。** 素の copytree は 2 回目に
    # FileExistsError で落ちるので、完了マーカーを見て冪等にする。中途半端に
    # 残ったツリーを使い回さないよう、マーカーが無ければ作り直す。
    done = pkg_root / ".copy_complete"
    if not done.is_file():
        shutil.rmtree(pkg_root, ignore_errors=True)
        shutil.copytree(
            repo_pkg,
            pkg_root / "livecap_cli",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        # source_root (= pkg_root) 配下に置く。**LIVECAP_RESOURCE_ROOT は子で外す**ので、
        # 探索は project_root -> source_root へ落ちてここに当たる。
        assets = pkg_root / "probe-assets"
        assets.mkdir(parents=True, exist_ok=True)
        (assets / "marker.txt").write_text("resource payload", encoding="utf-8")
        done.write_text("ok", encoding="utf-8")
    ctx.stage("copy_package")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(pkg_root)
    # env root が刺さっていると探索の先頭に載り、source_root 経路を測れない。
    env.pop("LIVECAP_RESOURCE_ROOT", None)

    proc = subprocess.run(
        [sys.executable, "-c", _SOURCE_ROOT_SOURCE, str(pkg_root)],
        cwd=ctx.payload.get("ascii_scratch") or str(pkg_root),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    ctx.stage("run_child")

    if proc.returncode != 0:
        raise RuntimeError(
            f"非 ASCII コピーからの import / resource 解決が失敗した "
            f"(exit={proc.returncode}): pkg_root={pkg_root} / "
            f"stderr_tail={(proc.stderr or '')[-400:]}"
        )
    return json.loads((proc.stdout or "").strip().splitlines()[-1])
