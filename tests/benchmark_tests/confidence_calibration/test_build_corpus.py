"""Tests for ``benchmarks.confidence_calibration.build_corpus`` (Issue #338 PR-β)。

yt-dlp / ffmpeg / engine は実 invoke 不要、subprocess mock / Engine mock で
pure logic と integration boundary を test。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from benchmarks.confidence_calibration.build_corpus import (
    BuildResult,
    _load_existing_paths,
    append_manifest,
    compute_alignment_score,
    download_audio,
    fetch_reference_text,
    ffmpeg_trim_and_resample,
    is_url,
    load_wav_16k_mono,
    write_wav,
)


# ----------------- is_url ----------------------------------------------


class TestIsUrl:
    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("https://www.youtube.com/watch?v=abc", True),
            ("http://example.com", True),
            ("./local.wav", False),
            ("/abs/path.wav", False),
            ("C:/Windows/path.wav", False),
            ("", False),
        ],
    )
    def test_is_url(self, input_str, expected):
        assert is_url(input_str) is expected


# ----------------- compute_alignment_score -----------------------------


class TestAlignmentScore:
    def test_exact_substring_match(self):
        transcribed = "Once upon a time"
        reference = "Long ago. Once upon a time there was a prince. The end."
        score, matched = compute_alignment_score(transcribed, reference)
        assert score > 0.0
        assert "Once upon a time" in (matched or "")

    def test_no_match(self):
        score, matched = compute_alignment_score("xyz qrst", "completely unrelated text")
        # 一部 char は一致するため 0 ではないが、低いはず
        assert score < 0.5

    def test_empty_transcribed(self):
        score, matched = compute_alignment_score("", "some reference")
        assert score == 0.0
        assert matched is None

    def test_whitespace_only_transcribed(self):
        score, matched = compute_alignment_score("   ", "some reference")
        assert score == 0.0


# ----------------- _load_existing_paths --------------------------------


class TestLoadExistingPaths:
    def test_empty_when_missing(self, tmp_path: Path):
        assert _load_existing_paths(tmp_path / "missing.jsonl") == set()

    def test_loads_paths(self, tmp_path: Path):
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(
            "\n".join(
                [
                    json.dumps({"path": "ja_clean/a.wav", "label": "speech"}),
                    json.dumps({"path": "ja_clean/b.wav", "label": "non_speech"}),
                ]
            ),
            encoding="utf-8",
        )
        paths = _load_existing_paths(manifest)
        assert paths == {"ja_clean/a.wav", "ja_clean/b.wav"}

    def test_skips_malformed_lines(self, tmp_path: Path):
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(
            '{"path": "a.wav", "label": "speech"}\n'
            "garbage line\n"
            '{"path": "b.wav", "label": "speech"}\n',
            encoding="utf-8",
        )
        paths = _load_existing_paths(manifest)
        assert paths == {"a.wav", "b.wav"}


# ----------------- append_manifest -------------------------------------


class TestAppendManifest:
    def test_appends_jsonl(self, tmp_path: Path):
        manifest = tmp_path / "manifest.jsonl"
        append_manifest(manifest, {"path": "a.wav", "label": "speech"})
        append_manifest(manifest, {"path": "b.wav", "label": "non_speech"})
        lines = manifest.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        e0 = json.loads(lines[0])
        e1 = json.loads(lines[1])
        assert e0["path"] == "a.wav"
        assert e1["path"] == "b.wav"


# ----------------- write_wav + load_wav_16k_mono -----------------------


class TestWavIO:
    def test_write_and_load_16k_mono(self, tmp_path: Path):
        audio = np.random.uniform(-0.5, 0.5, 16000).astype(np.float32)  # 1 sec
        path = tmp_path / "test.wav"
        write_wav(path, audio)
        loaded = load_wav_16k_mono(path)
        assert loaded.dtype == np.float32
        assert len(loaded) == 16000
        # Quantization / floating-point error 許容
        assert np.allclose(loaded, audio, atol=1e-3)


# ----------------- fetch_reference_text (local file) -------------------


class TestFetchReferenceTextLocal:
    def test_plain_text_file(self, tmp_path: Path):
        text_file = tmp_path / "ref.txt"
        text_file.write_text("Hello world.\nSecond line.", encoding="utf-8")
        text = fetch_reference_text(str(text_file))
        # 連続空白は 1 つに圧縮される
        assert "Hello world" in text
        assert "Second line" in text

    def test_html_strip(self, tmp_path: Path):
        html_file = tmp_path / "ref.html"
        html_file.write_text(
            "<html><body>"
            "<script>var x = 1;</script>"
            "<style>p { color: red; }</style>"
            "<p>Once upon a time.</p>"
            "<p>The prince said &quot;hello&quot;.</p>"
            "</body></html>",
            encoding="utf-8",
        )
        text = fetch_reference_text(str(html_file))
        assert "Once upon a time" in text
        assert 'hello' in text
        assert "var x = 1" not in text  # script 内除去
        assert "color: red" not in text  # style 内除去
        assert "<p>" not in text  # tag 除去


# ----------------- download_audio (mock subprocess) --------------------


class TestDownloadAudio:
    def test_local_file_copy(self, tmp_path: Path):
        src = tmp_path / "source.wav"
        src.write_bytes(b"fake wav content")
        dst = tmp_path / "dest" / "out.wav"
        result = download_audio(str(src), dst)
        assert result == dst
        assert dst.read_bytes() == b"fake wav content"

    def test_local_file_skip_when_exists(self, tmp_path: Path):
        src = tmp_path / "source.wav"
        src.write_bytes(b"new content")
        dst = tmp_path / "out.wav"
        dst.write_bytes(b"existing content")
        download_audio(str(src), dst, force=False)
        # 既存 file が上書きされない
        assert dst.read_bytes() == b"existing content"

    def test_local_file_force_overwrite(self, tmp_path: Path):
        src = tmp_path / "source.wav"
        src.write_bytes(b"new content")
        dst = tmp_path / "out.wav"
        dst.write_bytes(b"old content")
        download_audio(str(src), dst, force=True)
        assert dst.read_bytes() == b"new content"

    def test_local_file_missing_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            download_audio(str(tmp_path / "nonexistent.wav"), tmp_path / "out.wav")

    @patch("benchmarks.confidence_calibration.build_corpus.subprocess.run")
    def test_url_invokes_yt_dlp(self, mock_run: MagicMock, tmp_path: Path):
        """URL の場合 yt-dlp が subprocess.run で呼ばれる。"""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["yt-dlp"], returncode=0, stdout="", stderr=""
        )
        dst = tmp_path / "out.wav"
        download_audio("https://www.youtube.com/watch?v=abc", dst, force=True)
        # subprocess.run が呼ばれた、cmd に "yt-dlp" が含まれる
        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        assert "yt-dlp" in cmd
        assert "https://www.youtube.com/watch?v=abc" in cmd

    @patch("benchmarks.confidence_calibration.build_corpus.subprocess.run")
    def test_url_skip_when_exists(self, mock_run: MagicMock, tmp_path: Path):
        dst = tmp_path / "out.wav"
        dst.write_bytes(b"cached content")
        download_audio("https://example.com/audio", dst, force=False)
        # yt-dlp は呼ばれない (skip)
        assert not mock_run.called

    @patch("benchmarks.confidence_calibration.build_corpus.subprocess.run")
    def test_url_failure_raises(self, mock_run: MagicMock, tmp_path: Path):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["yt-dlp"], returncode=1, stdout="", stderr="network error"
        )
        dst = tmp_path / "out.wav"
        with pytest.raises(RuntimeError, match="yt-dlp failed"):
            download_audio("https://example.com/audio", dst, force=True)


# ----------------- ffmpeg_trim_and_resample (mock subprocess) ----------


class TestFfmpegTrimResample:
    @patch("benchmarks.confidence_calibration.build_corpus.subprocess.run")
    def test_invokes_ffmpeg_with_trim(self, mock_run: MagicMock, tmp_path: Path):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        src = tmp_path / "src.wav"
        src.write_bytes(b"fake")
        dst = tmp_path / "dst.wav"
        ffmpeg_trim_and_resample(
            src,
            dst,
            start_offset_sec=6.0,
            max_duration_sec=900.0,
            force=True,
        )
        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        # -ss 6.000 (trim offset)
        assert "-ss" in cmd
        ss_idx = cmd.index("-ss")
        assert cmd[ss_idx + 1] == "6.000"
        # -t 900.000 (max duration)
        assert "-t" in cmd
        # -ar 16000 (sample rate)
        assert "-ar" in cmd
        assert "16000" in cmd
        # -ac 1 (mono)
        assert "-ac" in cmd
        assert "1" in cmd

    @patch("benchmarks.confidence_calibration.build_corpus.subprocess.run")
    def test_skip_when_exists(self, mock_run: MagicMock, tmp_path: Path):
        src = tmp_path / "src.wav"
        src.write_bytes(b"fake")
        dst = tmp_path / "dst.wav"
        dst.write_bytes(b"cached")
        ffmpeg_trim_and_resample(src, dst, force=False)
        assert not mock_run.called

    @patch("benchmarks.confidence_calibration.build_corpus.subprocess.run")
    def test_no_trim_when_offset_zero(self, mock_run: MagicMock, tmp_path: Path):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        src = tmp_path / "src.wav"
        src.write_bytes(b"fake")
        dst = tmp_path / "dst.wav"
        ffmpeg_trim_and_resample(src, dst, start_offset_sec=0.0, force=True)
        cmd = mock_run.call_args[0][0]
        # start_offset_sec=0 のとき -ss flag は付かない
        assert "-ss" not in cmd
