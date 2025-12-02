from typing import Any, Dict, Optional

import pytest

from engines.engine_factory import EngineFactory
from engines.metadata import EngineInfo, EngineMetadata


class DummyEngine:
    def __init__(
        self,
        device: Optional[str] = None,
        language: str = "ja",
        model_size: str = "base",
        **kwargs,
    ) -> None:
        self.device = device
        self.language = language
        self.model_size = model_size
        self.extra_kwargs = kwargs


@pytest.fixture(autouse=True)
def reset_engine_factory_and_metadata(monkeypatch):
    """Ensure tests do not leak state into EngineFactory/EngineMetadata caches."""
    # Reset EngineFactory cache
    monkeypatch.setattr(EngineFactory, "_ENGINES", None, raising=False)
    # Store original metadata
    original_engines = EngineMetadata._ENGINES.copy()
    yield
    # Restore
    monkeypatch.setattr(EngineFactory, "_ENGINES", None, raising=False)
    EngineMetadata._ENGINES = original_engines


def _add_stub_engine_to_metadata():
    """Add a stub engine to EngineMetadata for testing."""
    EngineMetadata._ENGINES["stub"] = EngineInfo(
        id="stub",
        display_name="Stub Engine",
        description="Stub description",
        supported_languages=["ja"],
        default_params={
            "language": "ja",
            "model_size": "base",
        }
    )


def test_create_engine_uses_defaults(monkeypatch):
    """Test that create_engine uses default_params from EngineMetadata."""
    _add_stub_engine_to_metadata()
    monkeypatch.setattr(EngineFactory, "_get_engine_class", lambda *_: DummyEngine)

    engine = EngineFactory.create_engine(
        engine_type="stub",
        device="cpu",
    )

    assert isinstance(engine, DummyEngine)
    assert engine.device == "cpu"
    # Defaults from EngineMetadata.default_params should be applied
    assert engine.language == "ja"
    assert engine.model_size == "base"


def test_create_engine_allows_overriding_defaults(monkeypatch):
    """Test that engine_options can override default_params."""
    _add_stub_engine_to_metadata()
    monkeypatch.setattr(EngineFactory, "_get_engine_class", lambda *_: DummyEngine)

    engine = EngineFactory.create_engine(
        engine_type="stub",
        device="cuda",
        language="en",
        model_size="large",
    )

    assert isinstance(engine, DummyEngine)
    assert engine.device == "cuda"
    # Overridden values
    assert engine.language == "en"
    assert engine.model_size == "large"


def test_create_engine_rejects_auto():
    """Test that engine_type='auto' raises ValueError."""
    with pytest.raises(ValueError, match="engine_type='auto' is deprecated"):
        EngineFactory.create_engine(engine_type="auto", device="cpu")


def test_create_engine_rejects_unknown_engine():
    """Test that unknown engine types raise ValueError."""
    with pytest.raises(ValueError, match="Unknown engine type: nonexistent"):
        EngineFactory.create_engine(engine_type="nonexistent", device="cpu")


def test_get_engine_info_returns_metadata():
    """Test that get_engine_info returns correct metadata."""
    info = EngineFactory.get_engine_info("whispers2t_base")
    assert info is not None
    assert info["supported_languages"]
    assert "batch_size" in info["default_params"]


def test_get_available_engines():
    """Test that get_available_engines returns all engines."""
    engines = EngineFactory.get_available_engines()
    assert "whispers2t_base" in engines
    assert "reazonspeech" in engines
    assert engines["whispers2t_base"]["name"]
    assert engines["whispers2t_base"]["description"]


def test_get_engines_for_language():
    """Test that get_engines_for_language filters correctly."""
    ja_engines = EngineFactory.get_engines_for_language("ja")
    assert "reazonspeech" in ja_engines
    assert "parakeet_ja" in ja_engines

    en_engines = EngineFactory.get_engines_for_language("en")
    assert "parakeet" in en_engines
    assert "whispers2t_base" in en_engines
