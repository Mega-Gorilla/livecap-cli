from __future__ import annotations

import os
import stat

import numpy as np
import pytest

from livecap_core.config.defaults import get_default_config
from livecap_core.transcription import (
    FileProcessingResult,
    FileTranscriptionCancelled,
    FileTranscriptionPipeline,
    FileTranscriptionProgress,
)

sf = pytest.importorskip("soundfile")


def _write_test_wave(path):
    sample_rate = 16000
    duration_seconds = 1.0
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    data = 0.2 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, data, sample_rate)
    return sample_rate


def _make_fake_binary(path):
    """
    Create a tiny executable placeholder that works on Unix and Windows.
    """
    if os.name == "nt":
        path.write_text("@echo off\nexit /b 0\n")
    else:
        path.write_text("#!/bin/sh\nexit 0\n")
    mode = path.stat().st_mode
    if hasattr(stat, "S_IEXEC"):
        mode |= stat.S_IEXEC
    path.chmod(mode)


@pytest.fixture
def ffmpeg_manager_stub(tmp_path):
    bin_dir = tmp_path / "ffmpeg-bin"
    bin_dir.mkdir()
    ffmpeg_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    ffprobe_name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    ffmpeg_path = bin_dir / ffmpeg_name
    ffprobe_path = bin_dir / ffprobe_name
    _make_fake_binary(ffmpeg_path)
    _make_fake_binary(ffprobe_path)

    class _StubFFmpegManager:
        def configure_environment(self):
            return ffmpeg_path

        def resolve_probe(self):
            return ffprobe_path

    return _StubFFmpegManager()


@pytest.fixture
def pipeline_factory(ffmpeg_manager_stub):
    pipelines: list[FileTranscriptionPipeline] = []

    def _factory(**kwargs):
        pipeline = FileTranscriptionPipeline(
            config=get_default_config(),
            ffmpeg_manager=ffmpeg_manager_stub,
            **kwargs,
        )
        pipelines.append(pipeline)
        return pipeline

    yield _factory

    for pipeline in pipelines:
        pipeline.close()


def test_process_file_creates_srt(tmp_path, pipeline_factory):
    audio_path = tmp_path / "example.wav"
    sample_rate = _write_test_wave(audio_path)

    pipeline = pipeline_factory()
    result = pipeline.process_file(
        audio_path,
        segment_transcriber=lambda audio, sr: f"len={len(audio)} sr={sr}",
    )

    assert result.success
    assert result.output_path == audio_path.with_suffix(".srt")
    assert result.output_path and result.output_path.exists()
    assert result.subtitles
    assert result.metadata["sample_rate"] == sample_rate

    srt_content = result.output_path.read_text(encoding="utf-8")
    assert "len=" in srt_content


def test_process_files_emits_callbacks(tmp_path, pipeline_factory):
    audio_path = tmp_path / "batch.wav"
    _write_test_wave(audio_path)

    pipeline = pipeline_factory()
    progress_events: list[FileTranscriptionProgress] = []
    status_events: list[str] = []
    results: list[FileProcessingResult] = []

    pipeline.process_files(
        [audio_path],
        segment_transcriber=lambda audio, sr: "ok",
        progress_callback=progress_events.append,
        status_callback=status_events.append,
        result_callback=results.append,
    )

    assert status_events
    statuses = {event.status for event in progress_events}
    assert "processed" in statuses
    assert "segment" in statuses
    assert results and results[0].success


def test_process_files_cancel(tmp_path, pipeline_factory):
    audio_path = tmp_path / "cancel.wav"
    _write_test_wave(audio_path)

    pipeline = pipeline_factory()
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


def test_process_file_custom_segmenter(tmp_path, pipeline_factory):
    audio_path = tmp_path / "segment.wav"
    _write_test_wave(audio_path)

    segments = [(0.0, 0.5), (0.5, 1.0)]
    segment_progress: list[FileTranscriptionProgress] = []

    pipeline = pipeline_factory(segmenter=lambda *_args: segments)
    result = pipeline.process_file(
        audio_path,
        segment_transcriber=lambda audio, sr: f"text-{len(audio)}",
        progress_callback=segment_progress.append,
    )

    assert len(result.subtitles) == len(segments)
    segment_statuses = [event.status for event in segment_progress if event.status == "segment"]
    assert len(segment_statuses) == len(segments)
