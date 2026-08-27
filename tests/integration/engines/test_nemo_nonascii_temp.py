"""**mitigated track** — 非 ASCII `%TEMP%` でも production 経路で復元できること (Issue #379)。

`tests/nonascii/probes/native_models.py` の heavy probe は **raw track** である —
NeMo を**直接**呼んで「NeMo 単体では非 ASCII `%TEMP%` で失敗する」という基準データを取る。
あれを対策済み helper で上書きしてはならない (基準を失う)。

こちらは **production の `_load_model_from_path()`** を非 ASCII `%TEMP%` の下で走らせて
**成功する**ことを見る。両方あって初めて「欠陥は実在し、我々の経路では起きない」と言える。

**最重要の条件は「モデルファイルあり・メモリキャッシュなし」** である。同一プロセスでは
`ModelMemoryCache` hit で `restore_from()` を通らないし、`from_pretrained()` (ダウンロード経路、
#375 PR 3 で対策済み) だけ成功しても `restore_from()` の欠陥は見逃す。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = [pytest.mark.engine_smoke, pytest.mark.slow]

#: engine ごとの (import path, class 名, models root からの相対 .nemo)
_CASES = {
    "parakeet": (
        "livecap_cli.engines.parakeet_engine",
        "ParakeetEngine",
        "nvidia--parakeet-tdt-0.6b-v2.nemo",
    ),
    "canary": (
        "livecap_cli.engines.canary_engine",
        "CanaryEngine",
        "nvidia--canary-1b-flash.nemo",
    ),
}

#: 非 ASCII segment。cp932 の内側にある実世界ケース。
_NONASCII = "ユーザー"

_SENTINEL = "---LIVECAP-379-JSON---"

_CHILD = textwrap.dedent(
    """
    import json, sys, tempfile
    from pathlib import Path

    module_name, class_name, model_path = sys.argv[1], sys.argv[2], sys.argv[3]
    out = {"tempdir": tempfile.gettempdir()}
    try:
        from livecap_cli.engines.model_memory_cache import ModelMemoryCache
        # **in-memory cold** — cache hit だと restore_from() を通らない
        if hasattr(ModelMemoryCache, "clear"):
            ModelMemoryCache.clear()

        module = __import__(module_name, fromlist=[class_name])
        engine = getattr(module, class_name)(device="cpu")
        model = engine._load_model_from_path(Path(model_path))
        out["ok"] = model is not None
        out["model_class"] = type(model).__name__
    except BaseException as exc:
        out["ok"] = False
        out["error_type"] = type(exc).__name__
        out["error"] = str(exc)[:400]

    sys.stdout.write("\\n{s}\\n".format(s="---LIVECAP-379-JSON---"))
    sys.stdout.write(json.dumps(out, ensure_ascii=True, default=str))
    sys.stdout.write("\\n{s}\\n".format(s="---LIVECAP-379-JSON---"))
    """
)


def _models_root() -> Path:
    from livecap_cli.resources import get_model_manager

    return get_model_manager().get_models_dir()


def _resolve_nemo(relative: str) -> Path | None:
    source = _models_root() / relative
    if source.is_dir():
        # 実環境では ``<name>.nemo/<name>.nemo`` と入れ子になっていることがある
        nested = source / source.name
        if nested.is_file():
            return nested
        return None
    return source if source.is_file() else None


def _run_child(module_name: str, class_name: str, model_path: Path, temp_root: Path) -> dict:
    env = dict(os.environ)
    env["TEMP"] = str(temp_root)
    env["TMP"] = str(temp_root)
    env["TMPDIR"] = str(temp_root)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, module_name, class_name, str(model_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=1800,
    )
    chunks = proc.stdout.split(_SENTINEL)
    assert len(chunks) >= 3, (
        "子プロセスが結果 JSON を出さなかった:\n"
        f"stdout tail: {proc.stdout[-2000:]}\nstderr tail: {proc.stderr[-2000:]}"
    )
    return json.loads(chunks[1])


@pytest.mark.parametrize("case_key", sorted(_CASES))
def test_restore_succeeds_with_non_ascii_temp(case_key: str, tmp_path: Path) -> None:
    """**on-disk warm / in-memory cold** で、非 ASCII `%TEMP%` から復元できる。"""
    pytest.importorskip("nemo", reason="nemo-toolkit 未導入 (engines-nemo extra)")

    module_name, class_name, relative = _CASES[case_key]
    model_path = _resolve_nemo(relative)
    if model_path is None:
        pytest.skip(f".nemo が存在しない: {relative} (先にモデルを取得すること)")

    temp_root = tmp_path / _NONASCII / "temp"
    try:
        temp_root.mkdir(parents=True)
    except (OSError, UnicodeError) as exc:  # pragma: no cover - FS が variant を拒否
        pytest.skip(f"非 ASCII の TEMP を作れない: {exc}")

    result = _run_child(module_name, class_name, model_path, temp_root)

    # 前提の確認 — 注入が効いていなければテストが無意味になる
    assert not str(result["tempdir"]).isascii(), (
        f"子プロセスの %TEMP% が ASCII のまま: {result['tempdir']!r}。"
        "注入が効いておらず、このテストは何も検証していない"
    )
    assert result["ok"], (
        f"非 ASCII %TEMP% で {case_key} の復元が失敗した (#379 の再発): "
        f"{result.get('error_type')}: {result.get('error')}"
    )
