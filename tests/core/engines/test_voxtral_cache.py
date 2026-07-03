"""Issue #198: Voxtral cache が weakref 可能な container で env var 制御可能なことを pin。

`(model, processor)` tuple は ``weakref.ref()`` 不可のため ``ModelMemoryCache``
の weak-cache path で強参照に fallback し、 ``LIVECAP_ENGINE_STRONG_CACHE`` の
設定に関わらず常に VRAM を保持していた。 ``VoxtralModelContainer`` (dataclass、
weakref 可能) に置換したことで env var 通りの weak / strong cache 挙動になる。

実 Voxtral model は load せず、 container + ModelMemoryCache の参照挙動のみ検証。
"""

from __future__ import annotations

import gc
import weakref

import pytest

from livecap_cli.engines.model_memory_cache import ModelMemoryCache
from livecap_cli.engines.voxtral_engine import VoxtralModelContainer


@pytest.fixture(autouse=True)
def _clear_cache():
    """各 test 前後で global cache をクリア (test 間の汚染防止)。"""
    ModelMemoryCache.clear()
    yield
    ModelMemoryCache.clear()


class TestVoxtralModelContainerWeakref:
    """root-cause の pin: tuple は weakref 不可、 container は weakref 可能。"""

    def test_tuple_is_not_weakref_able(self):
        """(bug の根本原因) tuple は ``weakref.ref()`` できず TypeError。"""
        model, processor = object(), object()
        with pytest.raises(TypeError):
            weakref.ref((model, processor))

    def test_container_is_weakref_able(self):
        """VoxtralModelContainer (dataclass) は weakref 可能。"""
        container = VoxtralModelContainer(model=object(), processor=object())
        ref = weakref.ref(container)
        assert ref() is container

    def test_container_holds_model_and_processor(self):
        model, processor = object(), object()
        container = VoxtralModelContainer(model=model, processor=processor)
        assert container.model is model
        assert container.processor is processor


class TestVoxtralCacheEnvVarSemantics:
    """``LIVECAP_ENGINE_STRONG_CACHE`` (strong flag) が container で機能することを pin。"""

    def test_weak_cache_gc_after_holder_dropped(self):
        """strong=False: holder が消えると GC され、 cache miss になる
        (旧 tuple 実装では強参照 fallback で GC されなかった)。"""
        container = VoxtralModelContainer(model=object(), processor=object())
        ModelMemoryCache.set("voxtral_test", container, strong=False)

        # holder が生きている間は hit
        assert ModelMemoryCache.get("voxtral_test") is container

        # holder を手放して GC → weak ref が切れ cache miss
        del container
        gc.collect()
        assert ModelMemoryCache.get("voxtral_test") is None

    def test_strong_cache_survives_holder_dropped(self):
        """strong=True: holder が消えても cache が強参照で保持し続ける。"""
        container = VoxtralModelContainer(model=object(), processor=object())
        ModelMemoryCache.set("voxtral_test", container, strong=True)

        del container
        gc.collect()
        # strong cache なので生存
        got = ModelMemoryCache.get("voxtral_test")
        assert got is not None
        assert isinstance(got, VoxtralModelContainer)

    def test_engine_holder_keeps_weak_cache_alive(self):
        """engine 側 (別 holder) が container を掴んでいれば weak-cache は生存。

        _configure_model が ``self._model_container = container`` で保持する
        挙動を模擬。
        """
        container = VoxtralModelContainer(model=object(), processor=object())
        ModelMemoryCache.set("voxtral_test", container, strong=False)

        # engine が別 ref で保持 (self._model_container 相当)
        engine_held = container
        del container
        gc.collect()

        # engine が掴んでいる間は hit
        assert ModelMemoryCache.get("voxtral_test") is engine_held

        # engine も手放すと GC → miss
        del engine_held
        gc.collect()
        assert ModelMemoryCache.get("voxtral_test") is None
