"""torch で推論する engine の dtype 解決 (#470)。

``dtype="auto"`` は CUDA かつ bf16 をネイティブに計算できる GPU (Ampere 以降) で bf16、それ以外
(CPU / それより前の GPU) で fp32。Qwen3-ASR / Parakeet / Canary が同じ helper
(:func:`livecap_cli.utils.resolve_torch_dtype`) を使う。core CI には torch が無いので、GPU の
世代は ``torch`` の stand-in で与える。
"""

from __future__ import annotations

import logging
import sys
import types

import pytest

from livecap_cli.engines.engine_factory import EngineFactory
from livecap_cli.engines.model_memory_cache import ModelMemoryCache
from livecap_cli.utils import TORCH_DTYPES, resolve_torch_dtype

#: dtype を受ける engine (``EngineFactory`` の engine ID)
TORCH_ENGINES = ("qwen3asr", "qwen3asr_large", "parakeet", "parakeet_ja", "canary")


def _gpu_with_capability(monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int]) -> None:
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(get_device_capability=lambda: capability)
    monkeypatch.setitem(sys.modules, "torch", torch)


def _without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """``import torch`` が ImportError になる状態 (torch を触らない経路であることの確認)。"""
    monkeypatch.setitem(sys.modules, "torch", None)


class TestResolveTorchDtype:
    @pytest.mark.parametrize(
        ("capability", "expected"),
        [
            ((8, 0), "bfloat16"),  # Ampere (A100)
            ((8, 9), "bfloat16"),  # Ada (RTX 40)
            ((12, 0), "bfloat16"),  # Blackwell (RTX 50)
            ((7, 5), "float32"),  # Turing — bf16 は模擬実行のみ
            ((6, 1), "float32"),  # Pascal
        ],
    )
    def test_auto_on_cuda_follows_native_bf16_support(
        self, monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int], expected: str
    ):
        _gpu_with_capability(monkeypatch, capability)
        assert resolve_torch_dtype("auto", "cuda", "Engine") == expected

    def test_auto_on_cpu_is_float32_without_touching_torch(self, monkeypatch: pytest.MonkeyPatch):
        _without_torch(monkeypatch)
        assert resolve_torch_dtype("auto", "cpu", "Engine") == "float32"

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize("requested", [d for d in TORCH_DTYPES if d != "auto"])
    def test_explicit_value_is_used_as_is(self, monkeypatch: pytest.MonkeyPatch, requested: str, device: str):
        _without_torch(monkeypatch)
        assert resolve_torch_dtype(requested, device, "Engine") == requested

    @pytest.mark.parametrize("requested", ["bf16", "fp16", "half", "float64", "", "BFLOAT16"])
    def test_unknown_value_is_rejected(self, requested: str):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            resolve_torch_dtype(requested, "cpu", "Engine")

    def test_log_names_the_resolved_dtype_and_the_reason(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        _gpu_with_capability(monkeypatch, (7, 5))
        with caplog.at_level(logging.INFO, logger="livecap_cli.utils"):
            resolve_torch_dtype("auto", "cuda", "Parakeet")
        assert "Using float32 for Parakeet" in caplog.text
        assert "7.5" in caplog.text


class TestEnginesResolveDtype:
    @pytest.mark.parametrize("engine_id", TORCH_ENGINES)
    def test_default_is_auto(self, monkeypatch: pytest.MonkeyPatch, engine_id: str):
        _without_torch(monkeypatch)
        engine = EngineFactory.create_engine(engine_id, device="cpu")
        assert engine.dtype == "float32"

    @pytest.mark.parametrize("engine_id", TORCH_ENGINES)
    def test_explicit_dtype_reaches_the_engine(self, engine_id: str):
        engine = EngineFactory.create_engine(engine_id, device="cpu", dtype="bfloat16")
        assert engine.dtype == "bfloat16"

    @pytest.mark.parametrize("engine_id", TORCH_ENGINES)
    def test_unknown_dtype_fails_at_construction(self, engine_id: str):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            EngineFactory.create_engine(engine_id, device="cpu", dtype="bf16")

    @pytest.mark.parametrize("engine_id", TORCH_ENGINES)
    def test_memory_cache_key_separates_dtypes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, engine_id: str
    ):
        """fp32 で読んだ model を bf16 の要求に返さない (逆も)。"""
        keys: list[str] = []
        sentinel = object()

        def _get(key, *args, **kwargs):
            keys.append(key)
            return sentinel

        monkeypatch.setattr(ModelMemoryCache, "get", staticmethod(_get))
        for dtype in ("float32", "bfloat16"):
            engine = EngineFactory.create_engine(engine_id, device="cpu", dtype=dtype)
            assert engine._load_model_from_path(tmp_path / "model") is sentinel

        assert len(keys) == 2
        assert keys[0] != keys[1]
        assert keys[0].endswith("_float32") and keys[1].endswith("_bfloat16")
