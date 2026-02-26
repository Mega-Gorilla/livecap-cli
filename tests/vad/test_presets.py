"""Tests for VAD optimized presets (JSON-backed)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from livecap_cli.vad import VADConfig
from livecap_cli.vad.presets import (
    get_available_presets,
    get_best_vad_for_language,
    get_optimized_preset,
)


# =========================================================================
# Existing API compatibility tests
# =========================================================================


class TestPresetStructure:
    """Test preset structure via public API."""

    def test_presets_have_required_vad_types(self):
        """All expected VAD types should be present."""
        presets = get_available_presets()
        vad_types = set(vad_type for vad_type, _, _ in presets)
        expected_vads = {"silero", "tenvad", "webrtc"}
        assert vad_types == expected_vads

    def test_presets_have_both_languages(self):
        """Each VAD should have both JA and EN presets."""
        presets = get_available_presets()
        vad_langs: dict[str, set[str]] = {}
        for vad_type, lang, _ in presets:
            vad_langs.setdefault(vad_type, set()).add(lang)
        for vad_type, langs in vad_langs.items():
            assert "ja" in langs, f"{vad_type} missing 'ja' preset"
            assert "en" in langs, f"{vad_type} missing 'en' preset"

    def test_preset_has_vad_config_and_metadata(self):
        """Each preset should have vad_config and metadata."""
        for vad_type, lang, engine in get_available_presets():
            preset = get_optimized_preset(vad_type, lang, engine)
            assert preset is not None, f"Missing {vad_type}/{lang}/{engine}"
            assert "vad_config" in preset, f"{vad_type}/{lang}/{engine} missing vad_config"
            assert "metadata" in preset, f"{vad_type}/{lang}/{engine} missing metadata"
            assert "score" in preset["metadata"], f"{vad_type}/{lang}/{engine} missing score"

    def test_vad_config_can_be_created(self):
        """VADConfig should be creatable from preset vad_config."""
        for vad_type, lang, engine in get_available_presets():
            preset = get_optimized_preset(vad_type, lang, engine)
            config = VADConfig.from_dict(preset["vad_config"])
            assert isinstance(config, VADConfig)
            assert config.min_speech_ms > 0
            assert config.min_silence_ms > 0
            assert config.speech_pad_ms >= 0


class TestGetOptimizedPreset:
    """Test get_optimized_preset function."""

    def test_get_existing_preset_with_engine(self):
        """Should return preset for valid vad_type, language, and engine."""
        preset = get_optimized_preset("silero", "ja", "parakeet_ja")
        assert preset is not None
        assert "vad_config" in preset
        assert isinstance(preset["vad_config"]["threshold"], float)

    def test_get_nonexistent_vad_type(self):
        """Should return None for unknown VAD type."""
        preset = get_optimized_preset("unknown_vad", "ja")
        assert preset is None

    def test_get_nonexistent_language(self):
        """Should return None for unknown language."""
        preset = get_optimized_preset("silero", "zh")
        assert preset is None

    def test_get_nonexistent_engine(self):
        """Should return None for unknown engine."""
        preset = get_optimized_preset("silero", "ja", "nonexistent_engine")
        assert preset is None

    def test_get_without_engine_returns_best(self):
        """engine=None should return the preset with the lowest score."""
        preset = get_optimized_preset("silero", "ja")
        assert preset is not None
        # parakeet_ja has the best (lowest) score among silero/ja engines
        all_presets = get_available_presets()
        silero_ja_engines = [e for v, l, e in all_presets if v == "silero" and l == "ja"]
        assert len(silero_ja_engines) > 1  # multiple engines exist

        # Verify it picked the best
        best_score = preset["metadata"]["score"]
        for engine in silero_ja_engines:
            other = get_optimized_preset("silero", "ja", engine)
            assert other["metadata"]["score"] >= best_score

    def test_get_all_engine_combinations(self):
        """Should return presets for all known (vad_type, lang, engine) combinations."""
        for vad_type, lang, engine in get_available_presets():
            preset = get_optimized_preset(vad_type, lang, engine)
            assert preset is not None, f"Missing preset: {vad_type}/{lang}/{engine}"


class TestGetAvailablePresets:
    """Test get_available_presets function."""

    def test_returns_27_combinations(self):
        """Should return all 27 combinations (3 VADs x 2 languages x multiple engines)."""
        presets = get_available_presets()
        assert len(presets) == 27

    def test_returns_3_tuples(self):
        """Should return list of (vad_type, language, engine) tuples."""
        presets = get_available_presets()
        for item in presets:
            assert isinstance(item, tuple)
            assert len(item) == 3
            vad_type, language, engine = item
            assert isinstance(vad_type, str)
            assert isinstance(language, str)
            assert isinstance(engine, str)

    def test_contains_expected_combinations(self):
        """Should contain expected combinations."""
        presets = get_available_presets()
        assert ("silero", "ja", "parakeet_ja") in presets
        assert ("silero", "en", "parakeet") in presets
        assert ("webrtc", "ja", "qwen3asr") in presets
        assert ("tenvad", "en", "canary") in presets


class TestGetBestVadForLanguage:
    """Test get_best_vad_for_language function."""

    def test_best_vad_for_japanese(self):
        """TenVAD should be best for Japanese (lowest CER)."""
        result = get_best_vad_for_language("ja")
        assert result is not None
        vad_type, preset = result
        assert vad_type == "tenvad"
        assert preset["metadata"]["score"] < 0.08

    def test_best_vad_for_english(self):
        """WebRTC/canary should be best for English (lowest WER)."""
        result = get_best_vad_for_language("en")
        assert result is not None
        vad_type, preset = result
        assert vad_type == "webrtc"
        assert preset["metadata"]["engine"] == "canary"
        assert preset["metadata"]["score"] == pytest.approx(0.0166, rel=0.01)

    def test_best_vad_for_japanese_with_engine(self):
        """Should filter by engine when specified."""
        result = get_best_vad_for_language("ja", engine="qwen3asr")
        assert result is not None
        vad_type, preset = result
        assert preset["metadata"]["engine"] == "qwen3asr"

    def test_best_vad_for_english_with_engine(self):
        """Should filter by engine when specified."""
        result = get_best_vad_for_language("en", engine="parakeet")
        assert result is not None
        vad_type, preset = result
        assert vad_type == "webrtc"
        assert preset["metadata"]["engine"] == "parakeet"
        assert preset["metadata"]["score"] < 0.04

    def test_unknown_language(self):
        """Should return None for unknown language."""
        result = get_best_vad_for_language("zh")
        assert result is None

    def test_unknown_engine(self):
        """Should return None for unknown engine."""
        result = get_best_vad_for_language("ja", engine="nonexistent")
        assert result is None


class TestPresetMetadata:
    """Test preset metadata correctness."""

    def test_ja_tenvad_parakeet_is_best(self):
        """TenVAD/parakeet_ja should have lowest score for JA."""
        silero = get_optimized_preset("silero", "ja", "parakeet_ja")
        tenvad = get_optimized_preset("tenvad", "ja", "parakeet_ja")
        webrtc = get_optimized_preset("webrtc", "ja", "parakeet_ja")

        assert tenvad["metadata"]["score"] < silero["metadata"]["score"]
        assert tenvad["metadata"]["score"] < webrtc["metadata"]["score"]

    def test_en_webrtc_canary_is_best(self):
        """WebRTC/canary should have lowest score for EN."""
        silero = get_optimized_preset("silero", "en", "canary")
        tenvad = get_optimized_preset("tenvad", "en", "canary")
        webrtc = get_optimized_preset("webrtc", "en", "canary")

        assert webrtc["metadata"]["score"] < tenvad["metadata"]["score"]
        assert webrtc["metadata"]["score"] < silero["metadata"]["score"]

    def test_all_scores_are_valid(self):
        """All scores should be between 0 and 1."""
        for vad_type, lang, engine in get_available_presets():
            preset = get_optimized_preset(vad_type, lang, engine)
            score = preset["metadata"]["score"]
            assert 0 < score < 1, f"{vad_type}/{lang}/{engine} has invalid score: {score}"


# =========================================================================
# JSON loading and validation tests
# =========================================================================


class TestJSONPresetLoading:
    """Test that presets are loaded from JSON files via importlib.resources."""

    def test_loads_from_importlib_resources(self):
        """Presets should be loadable via importlib.resources.files()."""
        from importlib.resources import files

        package = files("livecap_cli.vad.presets")
        json_files = [
            r.name for r in package.iterdir()
            if hasattr(r, "name") and r.name.endswith(".json")
        ]
        assert len(json_files) == 27

    def test_all_json_files_are_valid_json(self):
        """All JSON files should parse without errors."""
        from importlib.resources import files

        package = files("livecap_cli.vad.presets")
        for resource in package.iterdir():
            if not hasattr(resource, "name") or not resource.name.endswith(".json"):
                continue
            raw = resource.read_text(encoding="utf-8")
            data = json.loads(raw)
            assert isinstance(data, dict), f"{resource.name} is not a JSON object"

    def test_json_metadata_has_extended_fields(self):
        """JSON presets should have engine, metric, and created_at in metadata."""
        for vad_type, lang, engine in get_available_presets():
            preset = get_optimized_preset(vad_type, lang, engine)
            meta = preset["metadata"]
            assert "metric" in meta, f"{vad_type}/{lang}/{engine} missing metric"
            assert "engine" in meta, f"{vad_type}/{lang}/{engine} missing engine"
            assert "created_at" in meta, f"{vad_type}/{lang}/{engine} missing created_at"

    def test_ja_presets_have_cer_metric(self):
        """Japanese presets should use CER metric."""
        for vad_type in ["silero", "tenvad", "webrtc"]:
            preset = get_optimized_preset(vad_type, "ja", "parakeet_ja")
            assert preset["metadata"]["metric"] == "cer"

    def test_en_presets_have_wer_metric(self):
        """English presets should use WER metric."""
        for vad_type in ["silero", "tenvad", "webrtc"]:
            preset = get_optimized_preset(vad_type, "en", "parakeet")
            assert preset["metadata"]["metric"] == "wer"


class TestPresetValidation:
    """Test JSON preset validation logic."""

    def _make_valid_preset(self) -> dict[str, Any]:
        """Create a minimal valid preset dict."""
        return {
            "vad_type": "silero",
            "language": "ja",
            "vad_config": {
                "threshold": 0.5,
                "min_speech_ms": 250,
                "min_silence_ms": 100,
                "speech_pad_ms": 100,
            },
            "metadata": {
                "score": 0.1,
                "metric": "cer",
                "trials": 10,
                "engine": "test_engine",
                "created_at": "2025-01-01T00:00:00Z",
            },
        }

    def test_valid_preset_passes(self):
        """A fully valid preset should pass validation."""
        from livecap_cli.vad.presets import _validate_preset

        data = self._make_valid_preset()
        _validate_preset(data, "test.json")  # Should not raise

    def test_missing_top_level_key(self):
        """Missing top-level key should raise ValueError with key name."""
        from livecap_cli.vad.presets import _validate_preset

        data = self._make_valid_preset()
        del data["vad_type"]

        with pytest.raises(ValueError, match="missing required key 'vad_type'"):
            _validate_preset(data, "test.json")

    def test_missing_vad_config_key(self):
        """Missing vad_config key should raise ValueError with dotted path."""
        from livecap_cli.vad.presets import _validate_preset

        data = self._make_valid_preset()
        del data["vad_config"]["min_speech_ms"]

        with pytest.raises(ValueError, match="vad_config.min_speech_ms"):
            _validate_preset(data, "test.json")

    def test_missing_metadata_key(self):
        """Missing metadata key should raise ValueError with dotted path."""
        from livecap_cli.vad.presets import _validate_preset

        data = self._make_valid_preset()
        del data["metadata"]["engine"]

        with pytest.raises(ValueError, match="metadata.engine"):
            _validate_preset(data, "test.json")

    def test_wrong_type_top_level(self):
        """Wrong type for top-level key should raise ValueError."""
        from livecap_cli.vad.presets import _validate_preset

        data = self._make_valid_preset()
        data["vad_type"] = 123  # should be str

        with pytest.raises(ValueError, match="must be str"):
            _validate_preset(data, "test.json")

    def test_wrong_type_vad_config(self):
        """Wrong type for vad_config value should raise ValueError."""
        from livecap_cli.vad.presets import _validate_preset

        data = self._make_valid_preset()
        data["vad_config"]["min_speech_ms"] = "not_a_number"

        with pytest.raises(ValueError, match="must be int/float"):
            _validate_preset(data, "test.json")

    def test_wrong_type_metadata(self):
        """Wrong type for metadata value should raise ValueError."""
        from livecap_cli.vad.presets import _validate_preset

        data = self._make_valid_preset()
        data["metadata"]["score"] = "not_a_number"

        with pytest.raises(ValueError, match="must be int/float"):
            _validate_preset(data, "test.json")

    def test_invalid_metric_value(self):
        """Unsupported metric should raise ValueError with supported list."""
        from livecap_cli.vad.presets import _validate_preset

        data = self._make_valid_preset()
        data["metadata"]["metric"] = "bleu"

        with pytest.raises(ValueError, match=r"must be one of \['cer', 'wer'\]"):
            _validate_preset(data, "test.json")

    def test_error_message_includes_source(self):
        """Error messages should include the source file name."""
        from livecap_cli.vad.presets import _validate_preset

        data = self._make_valid_preset()
        del data["language"]

        with pytest.raises(ValueError, match="my_preset.json"):
            _validate_preset(data, "my_preset.json")

    def test_backend_wrong_type_raises(self):
        """Non-dict backend should raise ValueError."""
        from livecap_cli.vad.presets import _validate_preset

        data = self._make_valid_preset()
        data["backend"] = ["not", "a", "dict"]

        with pytest.raises(ValueError, match="'backend' must be dict"):
            _validate_preset(data, "test.json")

    def test_backend_string_raises(self):
        """String backend should raise ValueError."""
        from livecap_cli.vad.presets import _validate_preset

        data = self._make_valid_preset()
        data["backend"] = "invalid"

        with pytest.raises(ValueError, match="'backend' must be dict"):
            _validate_preset(data, "test.json")

    def test_optional_threshold_wrong_type_raises(self):
        """Non-numeric threshold should raise ValueError."""
        from livecap_cli.vad.presets import _validate_preset

        data = self._make_valid_preset()
        data["vad_config"]["threshold"] = "high"

        with pytest.raises(ValueError, match="vad_config.threshold"):
            _validate_preset(data, "test.json")

    def test_optional_neg_threshold_wrong_type_raises(self):
        """Non-numeric neg_threshold should raise ValueError."""
        from livecap_cli.vad.presets import _validate_preset

        data = self._make_valid_preset()
        data["vad_config"]["neg_threshold"] = "low"  # str, not numeric

        with pytest.raises(ValueError, match="vad_config.neg_threshold"):
            _validate_preset(data, "test.json")

    def test_optional_threshold_absent_passes(self):
        """Preset without threshold (e.g. WebRTC) should pass validation."""
        from livecap_cli.vad.presets import _validate_preset

        data = self._make_valid_preset()
        del data["vad_config"]["threshold"]

        _validate_preset(data, "test.json")  # Should not raise


class TestBackendDefault:
    """Test backend key default behavior."""

    def test_backend_defaults_to_empty_for_silero(self):
        """Silero presets should not have backend key (no backend params)."""
        preset = get_optimized_preset("silero", "ja", "parakeet_ja")
        assert preset is not None
        # Silero has empty backend in JSON, so backend key should be absent
        # in the loaded entry (only included when non-empty)
        assert "backend" not in preset or preset.get("backend") == {}

    def test_backend_present_for_tenvad(self):
        """TenVAD presets should have backend with hop_size."""
        preset = get_optimized_preset("tenvad", "ja", "parakeet_ja")
        assert preset is not None
        assert "backend" in preset
        assert "hop_size" in preset["backend"]

    def test_backend_present_for_webrtc(self):
        """WebRTC presets should have backend with mode and frame_duration_ms."""
        preset = get_optimized_preset("webrtc", "ja", "parakeet_ja")
        assert preset is not None
        assert "backend" in preset
        assert "mode" in preset["backend"]
        assert "frame_duration_ms" in preset["backend"]
