from livecap_core.config.defaults import DEFAULT_CONFIG, get_default_config, merge_config
from livecap_core.config.validator import ConfigValidator


def test_get_default_config_returns_deep_copy():
    config_a = get_default_config()
    config_b = get_default_config()

    assert config_a is not config_b
    config_a["audio"]["sample_rate"] = 8000
    assert DEFAULT_CONFIG["audio"]["sample_rate"] == 16000


def test_default_config_contains_only_core_sections():
    config = get_default_config()
    expected_top_level = {
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
    assert set(config.keys()) == expected_top_level


def test_merge_config_overrides_nested_dicts():
    base = {"audio": {"sample_rate": 16000, "chunk_duration": 0.25}}
    override = {"audio": {"chunk_duration": 0.5}}

    merged = merge_config(base, override)
    assert merged["audio"]["sample_rate"] == 16000
    assert merged["audio"]["chunk_duration"] == 0.5


def test_config_validator_accepts_default_config():
    config = get_default_config()
    errors = ConfigValidator.validate(config)
    assert errors == []


def test_config_validator_flags_missing_section():
    config = {}
    errors = ConfigValidator.validate(config)
    assert errors
    assert any(err.path == "audio" for err in errors)


def test_config_validator_rejects_unknown_top_level_key():
    config = get_default_config()
    config["subtitle"] = {}

    errors = ConfigValidator.validate(config)
    assert errors
    assert any(err.path == "subtitle" for err in errors)


def test_config_validator_rejects_invalid_type_in_queue() -> None:
    config = get_default_config()
    config["queue"]["timeout"] = "fast"

    errors = ConfigValidator.validate(config)
    assert errors
    assert any(err.path == "queue.timeout" for err in errors)
