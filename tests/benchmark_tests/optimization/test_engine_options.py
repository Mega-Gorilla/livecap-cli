"""Tests for VADOptimizer._build_engine_options().

Regression tests for #265: ensures language and engine-specific options
are correctly passed when loading engines for VAD optimization.
"""

from __future__ import annotations

import pytest

pytest.importorskip("optuna", reason="optuna not installed")

from benchmarks.optimization.vad_optimizer import VADOptimizer


class TestBuildEngineOptions:
    """Verify _build_engine_options returns correct options per engine."""

    def test_whispers2t_en(self):
        """whispers2t EN must set language='en' and use_vad=False."""
        opt = VADOptimizer(
            vad_type="silero", language="en",
            engine_id="whispers2t", device="cpu",
        )
        options = opt._build_engine_options()

        assert options["language"] == "en"
        assert options["use_vad"] is False

    def test_whispers2t_ja(self):
        """whispers2t JA must set language='ja' and use_vad=False."""
        opt = VADOptimizer(
            vad_type="silero", language="ja",
            engine_id="whispers2t", device="cpu",
        )
        options = opt._build_engine_options()

        assert options["language"] == "ja"
        assert options["use_vad"] is False

    @pytest.mark.parametrize("engine_id", ["canary", "voxtral"])
    def test_multilingual_engine_sets_language(self, engine_id: str):
        """canary/voxtral must set language, no use_vad override."""
        opt = VADOptimizer(
            vad_type="silero", language="en",
            engine_id=engine_id, device="cpu",
        )
        options = opt._build_engine_options()

        assert options["language"] == "en"
        assert "use_vad" not in options

    def test_parakeet_returns_empty(self):
        """parakeet (monolingual) should return empty options."""
        opt = VADOptimizer(
            vad_type="silero", language="en",
            engine_id="parakeet", device="cpu",
        )
        options = opt._build_engine_options()

        assert options == {}

    def test_parakeet_ja_returns_empty(self):
        """parakeet_ja (monolingual) should return empty options."""
        opt = VADOptimizer(
            vad_type="silero", language="ja",
            engine_id="parakeet_ja", device="cpu",
        )
        options = opt._build_engine_options()

        assert options == {}
