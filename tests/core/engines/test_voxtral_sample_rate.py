"""Regression test for voxtral WAV header sample rate bug (#265).

Verifies that sf.write() uses the required sample rate (16kHz) rather
than the original input sample rate after resampling.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def _mock_voxtral_deps():
    """Provide minimal mocks for voxtral engine dependencies."""
    mock_transformers = MagicMock()
    mock_mistral = MagicMock()

    with (
        patch.dict("sys.modules", {
            "transformers": mock_transformers,
            "mistral_common": mock_mistral,
        }),
        patch(
            "livecap_cli.engines.voxtral_engine.check_transformers_availability",
            return_value=True,
        ),
        patch(
            "livecap_cli.engines.voxtral_engine.LibraryPreloader",
        ),
    ):
        yield


class TestVoxtralSampleRateWrite:
    """Verify WAV temp file is written with the correct sample rate."""

    def test_sf_write_uses_required_sr_not_input_sr(self, _mock_voxtral_deps):
        """sf.write must use required_sr (16000) after resampling, not the original rate."""
        from livecap_cli.engines.voxtral_engine import VoxtralEngine

        engine = VoxtralEngine.__new__(VoxtralEngine)
        # Minimal state for _transcribe_single_chunk
        engine._initialized = True
        engine.model = MagicMock()
        engine.processor = MagicMock()
        engine.torch_device = "cpu"
        engine.language = "en"
        engine.model_name = "test-model"
        engine.do_sample = False
        engine.max_new_tokens = 448

        # 44.1kHz input — NOT the required 16kHz
        input_sr = 44100
        duration_s = 0.5
        audio = np.random.randn(int(input_sr * duration_s)).astype(np.float32)

        captured_sr = {}

        def fake_sf_write(path, data, sr):
            captured_sr["value"] = sr

        mock_predicted_ids = MagicMock()
        engine.model.generate.return_value = mock_predicted_ids
        mock_predicted_ids.__getitem__ = MagicMock(return_value=mock_predicted_ids)
        engine.processor.batch_decode.return_value = ["hello world"]

        with (
            patch("livecap_cli.engines.voxtral_engine.sf.write", side_effect=fake_sf_write),
            patch("livecap_cli.engines.voxtral_engine.get_temp_dir") as mock_temp_dir,
            patch("librosa.resample", return_value=np.random.randn(int(16000 * duration_s)).astype(np.float32)),
            patch("torch.no_grad"),
        ):
            mock_temp_dir.return_value = MagicMock()
            temp_path_mock = MagicMock()
            temp_path_mock.exists.return_value = True
            mock_temp_dir.return_value.__truediv__ = MagicMock(return_value=temp_path_mock)

            text, confidence = engine._transcribe_single_chunk(audio, input_sr)

        # The critical assertion: sf.write must use 16000, NOT 44100
        assert captured_sr["value"] == 16000, (
            f"sf.write was called with sr={captured_sr['value']}, "
            f"expected 16000 (required_sr), not {input_sr} (input_sr)"
        )
