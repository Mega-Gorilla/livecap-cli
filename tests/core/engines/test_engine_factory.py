from typing import Any, Dict

import pytest

from engines.engine_factory import EngineFactory


class DummyEngine:
    def __init__(self, *, device: str | None = None, config: Dict[str, Any] | None = None) -> None:
        self.device = device
        self.config = config or {}


@pytest.fixture(autouse=True)
def reset_engine_factory(monkeypatch):
    # Ensure tests do not leak state into EngineFactory cache
    monkeypatch.setattr(EngineFactory, "_ENGINES", None, raising=False)
    yield
    monkeypatch.setattr(EngineFactory, "_ENGINES", None, raising=False)


def _stub_registry() -> Dict[str, Dict[str, Any]]:
    return {
        "stub": {
            "module": "tests.stub",
            "class_name": "DummyEngine",
            "name": "Stub Engine",
            "description": "Stub description",
            "supported_languages": ["ja"],
            "model_name": None,
            "model_size": None,
        }
    }


def test_create_engine_uses_defaults(monkeypatch):
    monkeypatch.setattr(EngineFactory, "_ENGINES", _stub_registry())
    monkeypatch.setattr(EngineFactory, "_get_engine_class", lambda *_: DummyEngine)

    engine = EngineFactory.create_engine(
        engine_type="stub",
        device="cpu",
        config={
            "transcription": {
                "engine": "stub",
                "input_language": "ja",
            }
        },
    )

    assert isinstance(engine, DummyEngine)
    # Defaults should be merged so audio settings are present
    assert engine.config["audio"]["sample_rate"] == 16000


def test_create_engine_resolves_auto(monkeypatch):
    monkeypatch.setattr(EngineFactory, "_ENGINES", _stub_registry())
    monkeypatch.setattr(EngineFactory, "_get_engine_class", lambda *_: DummyEngine)

    engine = EngineFactory.create_engine(
        engine_type="auto",
        device=None,
        config={
            "transcription": {
                "engine": "auto",
                "input_language": "ja",
                "language_engines": {
                    "ja": "stub",
                    "default": "stub",
                },
            }
        },
    )

    assert isinstance(engine, DummyEngine)
    assert engine.config["transcription"]["engine"] == "auto"


def test_create_engine_ignores_gui_specific_sections(monkeypatch):
    monkeypatch.setattr(EngineFactory, "_ENGINES", _stub_registry())
    monkeypatch.setattr(EngineFactory, "_get_engine_class", lambda *_: DummyEngine)

    engine = EngineFactory.create_engine(
        engine_type="stub",
        device="cpu",
        config={
            "transcription": {
                "engine": "stub",
                "input_language": "ja",
            },
            "subtitle": {"enabled": True},
            "translation": {
                "enable_translation": True,
                "translation_service": "google",
            },
        },
    )

    assert isinstance(engine, DummyEngine)
    assert "subtitle" not in engine.config
    assert engine.config["translation"]["enabled"] is True


def test_get_engine_info_prefers_override(monkeypatch):
    monkeypatch.setattr(EngineFactory, "_ENGINES", _stub_registry())

    override = {
        "engines": {
            "stub": {
                "display_name": "Custom Stub",
                "description": "Custom description",
            }
        }
    }

    info = EngineFactory.get_engine_info("stub", config=override)
    assert info is not None
    assert info["name"] == "Custom Stub"
    assert info["description"] == "Custom description"
