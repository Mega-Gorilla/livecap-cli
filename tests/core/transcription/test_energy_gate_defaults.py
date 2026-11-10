"""EnergyGate の既定値は 1 か所 (``livecap_cli.audio``) にあり、CLI と ``StreamTranscriber`` が共有する (#470)。

VAD 最適化の目的関数 (``benchmarks/optimization/objective.py``) も同じ定数で本番の後段を
再現する。既定値をどこかで書き直すと、最適化が本番と違う後段を評価するようになる。
"""

from __future__ import annotations

import inspect

import pytest

from livecap_cli import cli
from livecap_cli.audio import (
    DEFAULT_ENGINE_ENERGY_FRAME_MS,
    DEFAULT_ENGINE_ENERGY_METRIC,
    DEFAULT_ENGINE_MIN_RMS_DBFS,
    ENERGY_METRICS,
)
from livecap_cli.transcription.stream import StreamTranscriber


def test_defaults_are_the_documented_values():
    assert DEFAULT_ENGINE_MIN_RMS_DBFS == -45.0
    assert DEFAULT_ENGINE_ENERGY_METRIC == "max_frame_rms"
    assert DEFAULT_ENGINE_ENERGY_FRAME_MS == 32.0
    assert DEFAULT_ENGINE_ENERGY_METRIC in ENERGY_METRICS


def test_stream_transcriber_uses_the_shared_defaults():
    params = inspect.signature(StreamTranscriber.__init__).parameters

    assert params["engine_min_rms_dbfs"].default == DEFAULT_ENGINE_MIN_RMS_DBFS
    assert params["engine_energy_metric"].default == DEFAULT_ENGINE_ENERGY_METRIC
    assert params["engine_energy_frame_ms"].default == DEFAULT_ENGINE_ENERGY_FRAME_MS


def test_cli_transcribe_uses_the_shared_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path):
    captured = {}

    def capture(args):
        captured.update(vars(args))
        return 0

    monkeypatch.setattr(cli, "cmd_transcribe", capture)
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF")

    assert cli.main(["transcribe", str(audio)]) == 0
    assert captured["engine_min_rms"] == DEFAULT_ENGINE_MIN_RMS_DBFS
    assert captured["engine_energy_metric"] == DEFAULT_ENGINE_ENERGY_METRIC
    assert captured["engine_energy_frame_ms"] == DEFAULT_ENGINE_ENERGY_FRAME_MS
