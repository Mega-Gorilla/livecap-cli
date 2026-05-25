"""Tests for speaker benchmark factory and mock backend."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.speaker.factory import (
    SPEAKER_REGISTRY,
    create_embedding_backend,
    get_all_backend_ids,
    get_backend_info,
)


class TestRegistry:
    def test_registry_has_expected_backends(self) -> None:
        assert {"titanet", "ecapa", "pyannote", "mock"} <= set(SPEAKER_REGISTRY)

    def test_entries_have_required_fields(self) -> None:
        for spec in SPEAKER_REGISTRY.values():
            assert "module" in spec
            assert "class" in spec
            assert "license" in spec

    def test_get_all_backend_ids(self) -> None:
        ids = get_all_backend_ids()
        assert isinstance(ids, list)
        assert set(ids) == set(SPEAKER_REGISTRY)

    def test_get_backend_info_copy(self) -> None:
        info1 = get_backend_info("titanet")
        info2 = get_backend_info("titanet")
        assert info1 == info2
        assert info1 is not info2

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend_info("does-not-exist")


class TestCreateBackend:
    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            create_embedding_backend("nope")

    def test_create_mock(self) -> None:
        backend = create_embedding_backend("mock")
        assert backend.name == "mock"
        assert hasattr(backend, "load")
        assert hasattr(backend, "extract_embedding")


class TestMockBackend:
    def test_requires_load(self) -> None:
        backend = create_embedding_backend("mock")
        with pytest.raises(RuntimeError):
            backend.extract_embedding(np.zeros(1600, dtype=np.float32))

    def test_embedding_shape_and_norm(self) -> None:
        backend = create_embedding_backend("mock")
        backend.load("cpu")
        rng = np.random.default_rng(0)
        emb = backend.extract_embedding(rng.standard_normal(1600).astype(np.float32))
        assert emb.shape == (backend.embedding_dim,)
        assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-5)

    def test_distinct_tones_are_separable(self) -> None:
        backend = create_embedding_backend("mock")
        backend.load("cpu")
        sr = 16000
        t = np.arange(sr) / sr
        low = np.sin(2 * np.pi * 200 * t).astype(np.float32)
        high = np.sin(2 * np.pi * 3000 * t).astype(np.float32)
        e_low = backend.extract_embedding(low)
        e_high = backend.extract_embedding(high)
        # Different dominant frequencies -> low cosine similarity.
        cos = float(np.dot(e_low, e_high))
        assert cos < 0.5
