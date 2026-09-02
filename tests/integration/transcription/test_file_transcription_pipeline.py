from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

import numpy as np
import pytest

from livecap_cli.transcription import (
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
    ffmpeg_path = bin_dir / ffmpeg_name
    _make_fake_binary(ffmpeg_path)

    class _StubFFmpegManager:
        def configure_environment(self):
            return ffmpeg_path

        def resolve_probe(self):  # pragma: no cover - 呼ばれたら失敗させる
            # **pipeline は ffprobe を解決しない** (#387 PR C)。`FFPROBE_BINARY` は
            # 読み手が 1 つも無く、`_ffprobe_path` はそれへ流す以外に使われていな
            # かったので連鎖ごと削除した。再導入されたらここで気付く。
            raise AssertionError(
                "resolve_probe() が呼ばれた - ffprobe 解決が再導入されている (#387 PR C)"
            )

    return _StubFFmpegManager()


class TestFfmpegEnvIsNotExported:
    """**pipeline はプロセス env へ ffmpeg / ffprobe path を流さない** (#387 PR C)。

    削除した理由:

    - ``FFPROBE_BINARY`` は**読み手が 1 つも無い** (venv 全体で 0 件)。moviepy が
      読むのは ``FFMPEG_BINARY`` と ``FFPLAY_BINARY`` だけである
    - ``FFMPEG_BINARY`` は moviepy が **import 時にだけ**読む。本 package は
      moviepy を import せず、抽出は ``ffmpeg.run(..., cmd=self._ffmpeg_path)`` で
      **path を直接渡している**

    再導入されると **process-wide な副作用が黙って戻る**ので、ここで固定する。
    """

    def test_env_stays_unset(self, pipeline_factory, monkeypatch):
        monkeypatch.delenv("FFMPEG_BINARY", raising=False)
        monkeypatch.delenv("FFPROBE_BINARY", raising=False)

        pipeline_factory()

        assert "FFMPEG_BINARY" not in os.environ, (
            "pipeline の構築で FFMPEG_BINARY が設定された - env export が戻っている"
        )
        assert "FFPROBE_BINARY" not in os.environ, (
            "pipeline の構築で FFPROBE_BINARY が設定された - 読み手が 1 つも無い変数である"
        )

    def test_host_value_is_not_overwritten(self, pipeline_factory, monkeypatch):
        """**host が設定した値を触らない。** setdefault ですら書かない。"""
        monkeypatch.setenv("FFMPEG_BINARY", "host-provided")

        pipeline_factory()

        assert os.environ["FFMPEG_BINARY"] == "host-provided"

    def test_resolved_ffmpeg_is_still_used_for_extraction(self, pipeline_factory):
        """**解決した path は引き続き使う。** 削除したのは env への複製だけである。"""
        pipeline = pipeline_factory()

        assert pipeline._ffmpeg_path is not None
        assert Path(pipeline._ffmpeg_path).name.startswith("ffmpeg")
        # ffprobe 側は属性ごと消えている (読み手が無かったため)。
        assert not hasattr(pipeline, "_ffprobe_path")


@pytest.fixture
def pipeline_factory(ffmpeg_manager_stub):
    pipelines: list[FileTranscriptionPipeline] = []

    def _factory(**kwargs):
        pipeline = FileTranscriptionPipeline(
            ffmpeg_manager=ffmpeg_manager_stub,
            **kwargs,
        )
        pipelines.append(pipeline)
        return pipeline

    yield _factory

    for pipeline in pipelines:
        pipeline.close()


@pytest.fixture
def real_ffmpeg_pipeline_factory():
    """
    Factory that wires the real FFmpeg manager, used when the extraction
    path must be exercised (e.g., MKV regression tests).
    """
    pipelines: list[FileTranscriptionPipeline] = []

    def _factory(**kwargs):
        pipeline = FileTranscriptionPipeline(
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


def test_mkv_input_triggers_ffmpeg_extraction(tmp_path, real_ffmpeg_pipeline_factory):
    tests_root = Path(__file__).resolve().parents[2]
    mkv_source = tests_root / "assets" / "audio" / "common" / "test_tone_1s.mkv"
    assert mkv_source.exists(), f"MKV fixture not found: {mkv_source}"

    working_mkv = tmp_path / "input.mkv"
    shutil.copy2(mkv_source, working_mkv)

    pipeline = real_ffmpeg_pipeline_factory()
    result = pipeline.process_file(
        working_mkv,
        segment_transcriber=lambda audio, sr: f"len={len(audio)} sr={sr}",
    )

    assert result.success
    assert result.output_path == working_mkv.with_suffix(".srt")
    assert result.output_path and result.output_path.exists()
    assert result.subtitles
    assert result.metadata["sample_rate"] == 16000

    srt_content = result.output_path.read_text(encoding="utf-8")
    assert "len=" in srt_content
