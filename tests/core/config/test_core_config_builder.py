from config.core_config_builder import build_core_config


def test_translation_key_mapping():
    config = build_core_config(
        {
            "translation": {
                "enable_translation": True,
                "translation_service": "riva",
                "target_language": "fr",
                "show_original_with_translation": True,  # GUI 専用キーは無視
            }
        }
    )

    assert config["translation"]["enabled"] is True
    assert config["translation"]["service"] == "riva"
    assert config["translation"]["target_language"] == "fr"
    assert "show_original_with_translation" not in config["translation"]


def test_multi_source_sequence_normalization():
    raw = {
        "multi_source": {
            "sources": [
                {"id": "mic-1", "gain": 1.2},
                {"id": "mic-2"},
            ]
        }
    }

    config = build_core_config(raw)
    sources = config["multi_source"]["sources"]

    assert set(sources.keys()) == {"mic-1", "mic-2"}
    assert sources["mic-1"]["id"] == "mic-1"
    assert sources["mic-1"]["gain"] == 1.2
    assert sources["mic-2"]["id"] == "mic-2"


def test_unknown_gui_keys_are_removed():
    config = build_core_config(
        {
            "subtitle": {"enabled": True},
            "gui": {"language": "ja"},
        }
    )

    assert set(config.keys()) == {
        "audio",
        "multi_source",
        "silence_detection",
        "transcription",
        "translation",
        "engines",
        "logging",
        "queue",
        "debug",
        "file_mode",
    }
