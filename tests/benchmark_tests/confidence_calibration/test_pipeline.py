"""Tests for ``benchmarks.confidence_calibration.pipeline`` (Issue #338 PR-α)。

Corpus loader (manifest.jsonl + audio file load + resampling) を pin。
test fixture として超小 wav を生成 (soundfile / numpy で 0.1 sec mono)。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from benchmarks.confidence_calibration.pipeline import (
    load_calibration_corpus,
    resolve_corpus_dir,
)


def _write_silence_wav(path: Path, duration_sec: float = 0.1, sample_rate: int = 16000) -> None:
    """Test 用 0.1 sec silence wav を生成。"""
    import soundfile as sf

    n = int(duration_sec * sample_rate)
    audio = np.zeros(n, dtype=np.float32)
    sf.write(str(path), audio, sample_rate)


def _write_silence_wav_48k(path: Path, duration_sec: float = 0.1) -> None:
    """48 kHz silence wav (resampling test 用)。"""
    import soundfile as sf

    n = int(duration_sec * 48000)
    audio = np.zeros(n, dtype=np.float32)
    sf.write(str(path), audio, 48000)


# ----------------- load_calibration_corpus ----------------------------


class TestLoadCalibrationCorpus:
    def test_load_valid_manifest(self, tmp_path: Path):
        (tmp_path / "ja_clean").mkdir()
        _write_silence_wav(tmp_path / "ja_clean" / "a.wav")
        _write_silence_wav(tmp_path / "ja_clean" / "b.wav")

        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "path": "ja_clean/a.wav",
                            "label": "speech",
                            "language": "ja",
                            "noise": "clean",
                        }
                    ),
                    json.dumps(
                        {
                            "path": "ja_clean/b.wav",
                            "label": "non_speech",
                            "subtype": "silence",
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        items = load_calibration_corpus(tmp_path)
        assert len(items) == 2
        assert items[0].label == "speech"
        assert items[0].sample_rate == 16000
        assert items[0].audio.dtype == np.float32
        assert items[0].metadata["language"] == "ja"
        assert items[1].label == "non_speech"
        assert items[1].metadata["subtype"] == "silence"

    def test_48k_resampled_to_16k(self, tmp_path: Path):
        (tmp_path / "ja_clean").mkdir()
        _write_silence_wav_48k(tmp_path / "ja_clean" / "a.wav")
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(
            json.dumps({"path": "ja_clean/a.wav", "label": "speech"}) + "\n",
            encoding="utf-8",
        )
        items = load_calibration_corpus(tmp_path)
        assert items[0].sample_rate == 16000
        # 48k → 16k は 1/3、0.1 sec → ~1600 samples (with rounding)
        assert 1500 < len(items[0].audio) < 1700

    def test_missing_manifest_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="manifest.jsonl missing"):
            load_calibration_corpus(tmp_path)

    def test_missing_audio_raises(self, tmp_path: Path):
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(
            json.dumps({"path": "missing.wav", "label": "speech"}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(FileNotFoundError, match="audio file not found"):
            load_calibration_corpus(tmp_path)

    def test_invalid_label_raises(self, tmp_path: Path):
        (tmp_path / "ja_clean").mkdir()
        _write_silence_wav(tmp_path / "ja_clean" / "a.wav")
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(
            json.dumps({"path": "ja_clean/a.wav", "label": "unknown"}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="invalid label"):
            load_calibration_corpus(tmp_path)

    def test_missing_path_raises(self, tmp_path: Path):
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(
            json.dumps({"label": "speech"}) + "\n",  # path 欠落
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="missing 'path'"):
            load_calibration_corpus(tmp_path)

    def test_malformed_json_raises(self, tmp_path: Path):
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text("{malformed", encoding="utf-8")
        with pytest.raises(ValueError, match="malformed JSON"):
            load_calibration_corpus(tmp_path)

    def test_empty_lines_skipped(self, tmp_path: Path):
        (tmp_path / "ja_clean").mkdir()
        _write_silence_wav(tmp_path / "ja_clean" / "a.wav")
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(
            "\n"
            + json.dumps({"path": "ja_clean/a.wav", "label": "speech"})
            + "\n"
            + "\n",  # trailing blank line
            encoding="utf-8",
        )
        items = load_calibration_corpus(tmp_path)
        assert len(items) == 1

    def test_noisy_speech_label_accepted(self, tmp_path: Path):
        (tmp_path / "ja_noisy").mkdir()
        _write_silence_wav(tmp_path / "ja_noisy" / "a.wav")
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(
            json.dumps({"path": "ja_noisy/a.wav", "label": "noisy_speech"}) + "\n",
            encoding="utf-8",
        )
        items = load_calibration_corpus(tmp_path)
        assert items[0].label == "noisy_speech"


# ----------------- resolve_corpus_dir ----------------------------------


class TestResolveCorpusDir:
    def test_env_var_set(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LIVECAP_CALIBRATION_CORPUS_DIR", str(tmp_path))
        resolved = resolve_corpus_dir()
        assert resolved == tmp_path.resolve()

    def test_env_var_not_set(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LIVECAP_CALIBRATION_CORPUS_DIR", raising=False)
        assert resolve_corpus_dir() is None

    def test_env_var_empty_string_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("LIVECAP_CALIBRATION_CORPUS_DIR", "")
        assert resolve_corpus_dir() is None

    def test_env_var_with_tilde_expanded(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """``~/path`` が expand される。"""
        monkeypatch.setenv("LIVECAP_CALIBRATION_CORPUS_DIR", "~/my_corpus")
        resolved = resolve_corpus_dir()
        assert resolved is not None
        assert "~" not in str(resolved)
