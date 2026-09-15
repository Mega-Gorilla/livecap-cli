from pathlib import Path
import asyncio
import hashlib
import os

import pytest

from livecap_cli.resources import get_model_manager, _reset_resources_for_tests


@pytest.fixture(autouse=True)
def reset_managers():
    _reset_resources_for_tests()
    yield
    _reset_resources_for_tests()


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


def test_huggingface_cache_dir_is_under_cache_root(tmp_path, monkeypatch):
    """#428: `cache_dir=` に渡す階層は `<cache_root>/huggingface/hub`。

    `models--org--name/{blobs,refs,snapshots}` がこの直下にできるので、
    `huggingface_hub` の既定 (`~/.cache/huggingface/hub`) と同じ深さである。
    """
    cache_root = tmp_path / "cache-root"
    monkeypatch.setenv("LIVECAP_CORE_CACHE_DIR", str(cache_root))
    manager = get_model_manager()

    hf_cache = manager.get_huggingface_cache_dir()

    assert hf_cache == cache_root / "huggingface" / "hub"
    assert hf_cache.is_dir()


def test_huggingface_cache_dir_does_not_touch_env(tmp_path, monkeypatch):
    """#428: 環境変数経由は効かない (huggingface_hub は import 時に確定) ので、
    `HF_HOME` / `HF_HUB_CACHE` を**書き換えない**。呼び出し側が `cache_dir=` で渡す。"""
    monkeypatch.setenv("LIVECAP_CORE_CACHE_DIR", str(tmp_path / "cache-root"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "sentinel-home"))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    manager = get_model_manager()

    manager.get_huggingface_cache_dir()

    assert os.environ["HF_HOME"] == str(tmp_path / "sentinel-home")
    assert "HF_HUB_CACHE" not in os.environ
    assert not hasattr(manager, "huggingface_cache"), "旧 API は削除済み (#428)"


def test_models_root_and_temporary_directory(tmp_path, monkeypatch):
    models_root = tmp_path / "models-root"
    cache_root = tmp_path / "cache-root"
    monkeypatch.setenv("LIVECAP_CORE_MODELS_DIR", str(models_root))
    monkeypatch.setenv("LIVECAP_CORE_CACHE_DIR", str(cache_root))

    _reset_resources_for_tests()

    manager = get_model_manager()

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
