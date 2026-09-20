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
    models_dir = manager.get_models_dir()

    assert models_dir == models_root
    assert models_dir.exists()
    assert manager.cache_root == cache_root


def test_models_dir_has_no_engine_subdir_scope():
    """engine subdir (``<models_root>/<engine>/``) は #456 で廃止 — 正本は root 直下だけ。"""
    import inspect

    from livecap_cli.resources.model_manager import ModelManager

    assert list(inspect.signature(ModelManager.get_models_dir).parameters) == ["self"]


def test_temp_dir_created(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("LIVECAP_CORE_CACHE_DIR", str(cache_root))

    manager = get_model_manager()
    temp_dir = manager.get_temp_dir("downloads")

    assert temp_dir == cache_root / "downloads"
    assert temp_dir.exists()


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
