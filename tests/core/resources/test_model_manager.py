from pathlib import Path
import asyncio
import hashlib
import os

import pytest

from livecap_core.resources import get_model_manager, reset_resource_managers


@pytest.fixture(autouse=True)
def reset_managers():
    reset_resource_managers()
    yield
    reset_resource_managers()


def test_models_dir_env_override(tmp_path, monkeypatch):
    models_root = tmp_path / "custom-models"
    cache_root = tmp_path / "custom-cache"
    monkeypatch.setenv("LIVECAP_CORE_MODELS_DIR", str(models_root))
    monkeypatch.setenv("LIVECAP_CORE_CACHE_DIR", str(cache_root))

    manager = get_model_manager()
    engine_dir = manager.get_models_dir("engine-a")

    assert engine_dir == models_root / "engine-a"
    assert engine_dir.exists()
    assert manager.cache_root == cache_root


def test_temp_dir_created(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("LIVECAP_CORE_CACHE_DIR", str(cache_root))

    manager = get_model_manager()
    temp_dir = manager.get_temp_dir("downloads")

    assert temp_dir == cache_root / "downloads"
    assert temp_dir.exists()


def test_download_file_from_local_source(tmp_path):
    manager = get_model_manager()
    source = tmp_path / "sample.bin"
    payload = b"hello world"
    source.write_bytes(payload)

    checksum = hashlib.sha256(payload).hexdigest()

    downloaded = manager.download_file(source.as_uri(), expected_sha256=checksum)

    assert downloaded.read_bytes() == payload


def test_download_file_async_from_local_source(tmp_path):
    manager = get_model_manager()
    source = tmp_path / "sample_async.bin"
    payload = b"async hello"
    source.write_bytes(payload)

    checksum = hashlib.sha256(payload).hexdigest()

    async def run() -> None:
        downloaded = await manager.download_file_async(
            source.as_uri(),
            expected_sha256=checksum,
        )
        assert downloaded.read_bytes() == payload

    asyncio.run(run())


def test_huggingface_cache_context(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    manager = get_model_manager()

    with manager.huggingface_cache() as cache_dir:
        assert Path(os.environ["HF_HOME"]) == cache_dir

    assert "HF_HOME" not in os.environ


def test_models_root_and_temporary_directory(tmp_path, monkeypatch):
    models_root = tmp_path / "models-root"
    cache_root = tmp_path / "cache-root"
    monkeypatch.setenv("LIVECAP_CORE_MODELS_DIR", str(models_root))
    monkeypatch.setenv("LIVECAP_CORE_CACHE_DIR", str(cache_root))

    manager = get_model_manager(force_reset=True)

    assert manager.models_root == models_root
    assert manager.cache_root == cache_root
    assert manager.models_root.is_dir()
    assert manager.cache_root.is_dir()

    base_temp_dir = manager.get_temp_dir("phase1-spec")
    assert base_temp_dir.exists()
    assert base_temp_dir.parent == cache_root

    with manager.temporary_directory("phase1-spec") as temp_dir:
        assert temp_dir.is_dir()
        assert temp_dir.parent == base_temp_dir
        created_path = temp_dir

    # TemporaryDirectory cleans up automatically.
    assert not created_path.exists()
