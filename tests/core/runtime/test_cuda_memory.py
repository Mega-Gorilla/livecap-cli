"""``release_idle_cuda_memory()`` の手順・fallback・no-op を固定する (Issue #462)。

**実 GPU は使わない。** ここで守りたいのは「何をどの順で呼ぶか」「private API が無い /
失敗したときに silent success にしないか」「CUDA が無い・未初期化のときに CUDA context を
作らないか」であり、いずれも fake な ``torch`` で決定的に確認できる。実 GPU で reserved が
baseline へ戻ることは ``tests/integration/runtime/test_cuda_memory_release.py`` が見る。
"""

from __future__ import annotations

import logging
import types
from typing import List, Optional

import pytest

from livecap_cli.runtime import cuda_memory
from livecap_cli.runtime.cuda_memory import CudaMemoryRelease, release_idle_cuda_memory


class _FakeTorch:
    """必要な口だけ持つ ``torch`` の代役。呼ばれた順を ``calls`` に記録する。"""

    def __init__(
        self,
        *,
        available: bool = True,
        initialized: bool = True,
        clear_names: tuple[str, ...] = ("torch._C._cuda_clearCublasWorkspaces",),
        clear_error: Optional[Exception] = None,
        synchronize_error: Optional[Exception] = None,
        allocated: tuple[int, int] = (8_519_680, 0),
        reserved: tuple[int, int] = (8_359_247_872, 0),
    ) -> None:
        self.calls: List[str] = []
        self._allocated = list(allocated)
        self._reserved = list(reserved)
        self._clear_error = clear_error

        def _clear() -> None:
            self.calls.append("clear_cublas")
            if self._clear_error is not None:
                raise self._clear_error

        def _synchronize() -> None:
            self.calls.append("synchronize")
            if synchronize_error is not None:
                raise synchronize_error

        def _empty_cache() -> None:
            self.calls.append("empty_cache")

        def _memory_allocated() -> int:
            return self._allocated[0] if "empty_cache" not in self.calls else self._allocated[-1]

        def _memory_reserved() -> int:
            return self._reserved[0] if "empty_cache" not in self.calls else self._reserved[-1]

        self.cuda = types.SimpleNamespace(
            is_available=lambda: available,
            is_initialized=lambda: initialized,
            synchronize=_synchronize,
            empty_cache=_empty_cache,
            memory_allocated=_memory_allocated,
            memory_reserved=_memory_reserved,
        )
        self._C = types.SimpleNamespace()
        if "torch.cuda._clear_cublas_workspaces" in clear_names:
            self.cuda._clear_cublas_workspaces = _clear
        if "torch._C._cuda_clearCublasWorkspaces" in clear_names:
            self._C._cuda_clearCublasWorkspaces = _clear


@pytest.fixture
def fake_torch(monkeypatch):
    """``_import_torch`` を差し替えるファクトリ。``gc.collect`` も記録する。"""

    def _install(**kwargs) -> _FakeTorch:
        torch = _FakeTorch(**kwargs)
        monkeypatch.setattr(cuda_memory, "_import_torch", lambda: torch)
        monkeypatch.setattr(cuda_memory.gc, "collect", lambda: torch.calls.append("gc.collect") or 0)
        return torch

    return _install


class TestOrder:
    def test_fixed_order_gc_synchronize_clear_empty_cache(self, fake_torch):
        """順序は契約である — clear の前に synchronize しないと in-flight な cuBLAS と競合し、
        clear の後に empty_cache しないと segment が返らない。"""
        torch = fake_torch()

        result = release_idle_cuda_memory()

        assert torch.calls == ["gc.collect", "synchronize", "clear_cublas", "gc.collect", "empty_cache"]
        assert result.attempted and result.cublas_cleared
        assert result.reason == cuda_memory.REASON_RELEASED

    def test_prefers_the_public_api_when_both_exist(self, fake_torch):
        """将来 PyTorch が公開 API を足したら、private binding より先に使う。"""
        torch = fake_torch(
            clear_names=("torch.cuda._clear_cublas_workspaces", "torch._C._cuda_clearCublasWorkspaces")
        )

        result = release_idle_cuda_memory()

        assert result.cublas_api == "torch.cuda._clear_cublas_workspaces"
        assert torch.calls.count("clear_cublas") == 1

    def test_reports_the_private_api_on_torch_2_9(self, fake_torch):
        fake_torch(clear_names=("torch._C._cuda_clearCublasWorkspaces",))

        assert release_idle_cuda_memory().cublas_api == "torch._C._cuda_clearCublasWorkspaces"


class TestMeasurements:
    def test_before_and_after_are_reported(self, fake_torch):
        fake_torch(allocated=(8_519_680, 0), reserved=(8_359_247_872, 0))

        result = release_idle_cuda_memory()

        assert (result.allocated_before, result.allocated_after) == (8_519_680, 0)
        assert (result.reserved_before, result.reserved_after) == (8_359_247_872, 0)
        assert result.released_bytes == 8_359_247_872

    def test_released_bytes_is_never_negative(self, fake_torch):
        """別スレッドが確保して reserved が増えていても、負の「解放量」を報告しない。"""
        fake_torch(reserved=(1_000, 2_000))

        assert release_idle_cuda_memory().released_bytes == 0

    def test_to_dict_is_json_friendly(self, fake_torch):
        import json

        fake_torch()

        payload = release_idle_cuda_memory().to_dict()

        assert json.loads(json.dumps(payload))["reason"] == cuda_memory.REASON_RELEASED
        assert payload["released_bytes"] > 0 and isinstance(payload["warnings"], list)


class TestFallback:
    """**silent success にしない。** 解放しきれていないことを呼び出し側が判定できること。"""

    def test_missing_private_api_still_empties_cache_but_reports_it(self, fake_torch, caplog):
        torch = fake_torch(clear_names=())

        with caplog.at_level(logging.WARNING, logger="livecap_cli.runtime.cuda_memory"):
            result = release_idle_cuda_memory()

        assert torch.calls == ["gc.collect", "synchronize", "gc.collect", "empty_cache"], "empty_cache は続行する"
        assert result.attempted and not result.cublas_cleared
        assert result.cublas_api is None
        assert result.reason == cuda_memory.REASON_CUBLAS_API_MISSING
        assert any("cuBLAS" in w for w in result.warnings)
        assert any("cuBLAS" in r.getMessage() for r in caplog.records)

    def test_failing_private_api_is_reported(self, fake_torch, caplog):
        torch = fake_torch(clear_error=RuntimeError("boom"))

        with caplog.at_level(logging.WARNING, logger="livecap_cli.runtime.cuda_memory"):
            result = release_idle_cuda_memory()

        assert "empty_cache" in torch.calls
        assert not result.cublas_cleared and result.cublas_api is None
        assert result.reason == cuda_memory.REASON_CUBLAS_API_FAILED
        assert any("boom" in w for w in result.warnings)

    def test_synchronize_failure_does_not_abort_the_release(self, fake_torch):
        torch = fake_torch(synchronize_error=RuntimeError("driver hiccup"))

        result = release_idle_cuda_memory()

        assert torch.calls == ["gc.collect", "synchronize", "clear_cublas", "gc.collect", "empty_cache"]
        assert result.cublas_cleared, "synchronize の失敗で解放そのものを諦めない"
        assert any("synchronize" in w for w in result.warnings)


class TestNoOp:
    """CUDA が無い / 未初期化なら**何も呼ばない** — context を新しく作らないため。"""

    def test_torch_missing(self, monkeypatch):
        monkeypatch.setattr(cuda_memory, "_import_torch", lambda: None)

        result = release_idle_cuda_memory()

        assert result == CudaMemoryRelease(
            attempted=False,
            cublas_cleared=False,
            cublas_api=None,
            allocated_before=0,
            allocated_after=0,
            reserved_before=0,
            reserved_after=0,
            reason=cuda_memory.REASON_TORCH_MISSING,
        )

    def test_cuda_unavailable(self, fake_torch):
        torch = fake_torch(available=False)

        result = release_idle_cuda_memory()

        assert torch.calls == [], "CPU 環境で gc / synchronize すら走らせない"
        assert not result.attempted and result.reason == cuda_memory.REASON_CUDA_UNAVAILABLE

    def test_cuda_not_initialized_does_not_create_a_context(self, fake_torch):
        """``torch.cuda.synchronize()`` は CUDA context を作る。未初期化なら解放すべき
        メモリも無いので、**触らない**のが正しい。"""
        torch = fake_torch(initialized=False)

        result = release_idle_cuda_memory()

        assert torch.calls == []
        assert not result.attempted and result.reason == cuda_memory.REASON_CUDA_NOT_INITIALIZED


class TestIdempotent:
    def test_calling_twice_is_safe(self, fake_torch):
        torch = fake_torch(allocated=(8_519_680, 0), reserved=(20_971_520, 0))

        first = release_idle_cuda_memory()
        torch.calls.clear()
        second = release_idle_cuda_memory()

        assert first.cublas_cleared and second.cublas_cleared
        assert torch.calls == ["gc.collect", "synchronize", "clear_cublas", "gc.collect", "empty_cache"]
        assert second.reason == cuda_memory.REASON_RELEASED
