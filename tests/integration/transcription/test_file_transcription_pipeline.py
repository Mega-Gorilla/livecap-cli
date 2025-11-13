from __future__ import annotations

import os

import numpy as np
import pytest

from livecap_core.config.defaults import get_default_config
from livecap_core.transcription import (
    FileProcessingResult,
    FileTranscriptionPipeline,
    FileTranscriptionProgress,
    FileTranscriptionCancelled,
)

pytestmark = pytest.mark.skipif(
    os.getenv("LIVECAP_ENABLE_INTEGRATION") != "true",
    reason="Set LIVECAP_ENABLE_INTEGRATION=true to run integration tests (requires model downloads).",
)

sf = pytest.importorskip("soundfile")


def _write_test_wave(path):
    sample_rate = 16000
    duration_seconds = 1.0
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    data = 0.2 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, data, sample_rate)
    return sample_rate


def test_process_file_creates_srt(tmp_path):
    audio_path = tmp_path / "example.wav"
    sample_rate = _write_test_wave(audio_path)

    pipeline = FileTranscriptionPipeline(config=get_default_config())
    try:
        result = pipeline.process_file(
            audio_path,
            segment_transcriber=lambda audio, sr: f"len={len(audio)} sr={sr}",
        )
    finally:
        pipeline.close()

    assert result.success
    assert result.output_path == audio_path.with_suffix(".srt")
    assert result.output_path and result.output_path.exists()
    assert result.subtitles
    assert result.metadata["sample_rate"] == sample_rate

    srt_content = result.output_path.read_text(encoding="utf-8")
    assert "len=" in srt_content


def test_process_files_emits_callbacks(tmp_path):
    audio_path = tmp_path / "batch.wav"
    _write_test_wave(audio_path)

    pipeline = FileTranscriptionPipeline(config=get_default_config())
    progress_events: list[FileTranscriptionProgress] = []
    status_events: list[str] = []
    results: list[FileProcessingResult] = []
    try:
        pipeline.process_files(
            [audio_path],
            segment_transcriber=lambda audio, sr: "ok",
            progress_callback=progress_events.append,
            status_callback=status_events.append,
            result_callback=results.append,
        )
    finally:
        pipeline.close()

    assert status_events
    statuses = {event.status for event in progress_events}
    assert "processed" in statuses
    assert "segment" in statuses
    assert results and results[0].success


def test_process_files_cancel(monkeypatch, tmp_path):
    audio_path = tmp_path / "cancel.wav"
    _write_test_wave(audio_path)

    pipeline = FileTranscriptionPipeline(config=get_default_config())
    cancel_flag = {"value": False}

    def trigger_cancel(progress):
        cancel_flag["value"] = True

    def should_cancel():
        return cancel_flag["value"]

    with pytest.raises(FileTranscriptionCancelled):
        pipeline.process_files(
            [audio_path],
            segment_transcriber=lambda audio, sr: "ok",
            progress_callback=trigger_cancel,
            should_cancel=should_cancel,
        )

    pipeline.close()


def test_process_file_custom_segmenter(tmp_path):
    audio_path = tmp_path / "segment.wav"
    _write_test_wave(audio_path)

    segments = [(0.0, 0.5), (0.5, 1.0)]
    segment_progress: list[FileTranscriptionProgress] = []

    pipeline = FileTranscriptionPipeline(
        config=get_default_config(),
        segmenter=lambda *_args: segments,
    )
    try:
        result = pipeline.process_file(
            audio_path,
            segment_transcriber=lambda audio, sr: f"text-{len(audio)}",
            progress_callback=segment_progress.append,
        )
    finally:
        pipeline.close()

    assert len(result.subtitles) == len(segments)
    segment_statuses = [event.status for event in segment_progress if event.status == "segment"]
    assert len(segment_statuses) == len(segments)
