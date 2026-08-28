"""実モデルで cache identity v2 が機能すること (Issue #409)。

**ゲートは mock 側にある。** identity の変異 (root 違い / ファイル差し替え /
構築パラメータ / native 版) は `tests/core/engines/test_reazonspeech_cache_key.py` が
mock で網羅する。ここは「実 recognizer でも 2 回目が同一オブジェクトとして返る」ことを
int8 / float32 の両方で確かめるだけで、**全組合せを実モデルで回さない**。

モデルが無ければ skip する。**実モデルが常駐する self-hosted runner でだけ CI が走らせ、
そこでは PASSED を要求する** (`engine-smoke-gpu` の「Run ReazonSpeech cache identity check」)。
engine_smoke + slow なので通常の smoke step では収集されず、ゲートを置かないと
**どこでも走らないテスト**になる。**`sherpa-onnx` の importorskip は置かない** —
`pyproject.toml` の `dependencies` に入っているコア依存なので、skip できる状況は
「壊れている」ときだけであり、guard は breakage を隠すことにしかならない。
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.engine_smoke, pytest.mark.slow]

#: (id, use_int8, models root からの相対ディレクトリ)
_CASES = [
    ("int8", True, "reazonspeech/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01"),
    ("float32", False, "reazonspeech/reazon-research--reazonspeech-k2-v2"),
]


def _model_dir(relative: str) -> Path | None:
    from livecap_cli.resources import get_model_manager

    path = get_model_manager().get_models_dir() / relative
    return path if path.is_dir() else None


@pytest.mark.parametrize(("case_id", "use_int8", "relative"), _CASES, ids=[c[0] for c in _CASES])
def test_second_load_returns_the_cached_recognizer(
    case_id: str, use_int8: bool, relative: str
) -> None:
    from livecap_cli.engines.model_memory_cache import ModelMemoryCache
    from livecap_cli.engines.reazonspeech_engine import ReazonSpeechEngine

    model_dir = _model_dir(relative)
    if model_dir is None:
        pytest.skip(f"実モデルが無い: {relative}")

    ModelMemoryCache.clear()
    try:
        engine = ReazonSpeechEngine(device="cpu", use_int8=use_int8)
        first = engine._load_model_from_path(model_dir)
        assert first is not None

        # **別インスタンスから引く** — cache が engine ではなくキーで効いていること
        second = ReazonSpeechEngine(device="cpu", use_int8=use_int8)._load_model_from_path(model_dir)
        assert second is first, f"{case_id}: 2 回目が cache hit していない"

        keys = ModelMemoryCache.get_stats()["cache_keys"]
        assert any(k.startswith("reazonspeech:v2:") for k in keys), (
            f"v2 のキーで保存されていない: {keys}"
        )
        assert not any(k.startswith("reazonspeech_") for k in keys), (
            f"legacy な v1 形式のキーを書いている: {keys}"
        )
    finally:
        ModelMemoryCache.clear()


@pytest.mark.parametrize(("case_id", "use_int8", "relative"), _CASES, ids=[c[0] for c in _CASES])
def test_changing_num_threads_rebuilds(case_id: str, use_int8: bool, relative: str) -> None:
    """構築パラメータが違えば別 recognizer になる (実モデルでの確認)。"""
    from livecap_cli.engines.model_memory_cache import ModelMemoryCache
    from livecap_cli.engines.reazonspeech_engine import ReazonSpeechEngine

    model_dir = _model_dir(relative)
    if model_dir is None:
        pytest.skip(f"実モデルが無い: {relative}")

    ModelMemoryCache.clear()
    try:
        a = ReazonSpeechEngine(
            device="cpu", use_int8=use_int8, num_threads=2
        )._load_model_from_path(model_dir)
        b = ReazonSpeechEngine(
            device="cpu", use_int8=use_int8, num_threads=4
        )._load_model_from_path(model_dir)
        assert a is not b, f"{case_id}: num_threads を変えても同じ recognizer が返っている"
    finally:
        ModelMemoryCache.clear()
