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
    _load_manifest_entries,
    _write_manifest,
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
    """coverage-based alignment score の動作 pin (PR #340 review 指摘 2 fix)。

    旧実装 ``SequenceMatcher.ratio()`` は長文 reference で完全一致 substring
    でも score ≈ 0.006 と極小だった。新実装 ``match.size / len(transcribed)``
    (= coverage) では同 case で score = 1.0。
    """

    def test_exact_substring_match(self):
        transcribed = "Once upon a time"
        reference = "Long ago. Once upon a time there was a prince. The end."
        score, matched = compute_alignment_score(transcribed, reference)
        # 完全一致 substring → coverage = 1.0
        assert score == pytest.approx(1.0, abs=1e-6)
        assert "Once upon a time" in (matched or "")

    def test_substring_in_long_reference(self):
        """長文 reference 内に完全一致 substring → score = 1.0 (review 指摘 2)。"""
        transcribed = "Once upon a time"
        # 旧 ratio() なら ~0.006 だった case
        reference = "x" * 5000 + " Once upon a time " + "y" * 5000
        score, matched = compute_alignment_score(transcribed, reference)
        assert score == pytest.approx(1.0, abs=1e-6)
        assert matched == "Once upon a time"

    def test_partial_match_returns_partial_coverage(self):
        """transcribed の前半だけ reference に match → coverage = (match 部分の比率)。"""
        transcribed = "Once upon a time XYZNEVERMATCH"  # 30 chars 中 16 chars が match
        reference = "Long ago. Once upon a time there was a prince."
        score, matched = compute_alignment_score(transcribed, reference)
        # 30 chars 中 16 chars = "Once upon a time" が match → 16/30 ≈ 0.533
        assert 0.4 < score < 0.7
        assert "Once upon a time" in (matched or "")

    def test_no_match(self):
        score, matched = compute_alignment_score("xyz qrst", "completely unrelated text")
        # 一部 char (e.g. "t" / 空白) は match するが、< 0.5 (LCS が短い)
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


# ----------------- _load_manifest_entries (upsert 用、PR #340 review 1 fix) ----


class TestLoadManifestEntries:
    def test_empty_when_missing(self, tmp_path: Path):
        assert _load_manifest_entries(tmp_path / "missing.jsonl") == {}

    def test_loads_entries_as_path_map(self, tmp_path: Path):
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(
            "\n".join(
                [
                    json.dumps({"path": "a.wav", "label": "speech", "score": 0.9}),
                    json.dumps({"path": "b.wav", "label": "non_speech"}),
                ]
            ),
            encoding="utf-8",
        )
        entries = _load_manifest_entries(manifest)
        assert set(entries.keys()) == {"a.wav", "b.wav"}
        assert entries["a.wav"]["score"] == 0.9

    def test_last_wins_on_duplicate_path(self, tmp_path: Path):
        """重複 path がある manifest を読むと last entry が勝つ (legacy data
        recovery 用、PR #340 review 1 fix で重複は新規発生しなくなる)。"""
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(
            "\n".join(
                [
                    json.dumps({"path": "a.wav", "label": "speech", "v": 1}),
                    json.dumps({"path": "a.wav", "label": "speech", "v": 2}),
                ]
            ),
            encoding="utf-8",
        )
        entries = _load_manifest_entries(manifest)
        assert len(entries) == 1
        assert entries["a.wav"]["v"] == 2

    def test_load_existing_paths_uses_same_logic(self, tmp_path: Path):
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(
            json.dumps({"path": "x.wav", "label": "speech"}), encoding="utf-8"
        )
        assert _load_existing_paths(manifest) == {"x.wav"}


class TestWriteManifest:
    def test_rewrite_entries_overwrites(self, tmp_path: Path):
        """_write_manifest は既存 file を上書きする (append しない)。"""
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text("old garbage\n", encoding="utf-8")
        _write_manifest(
            manifest,
            [
                {"path": "a.wav", "label": "speech"},
                {"path": "b.wav", "label": "non_speech"},
            ],
        )
        text = manifest.read_text(encoding="utf-8")
        assert "old garbage" not in text
        lines = text.strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["path"] == "a.wav"
        assert json.loads(lines[1])["path"] == "b.wav"

    def test_round_trip_load_after_write(self, tmp_path: Path):
        manifest = tmp_path / "manifest.jsonl"
        entries_in = [
            {"path": "a.wav", "label": "speech", "alignment_score": 0.95},
            {"path": "b.wav", "label": "non_speech"},
        ]
        _write_manifest(manifest, entries_in)
        loaded = _load_manifest_entries(manifest)
        assert set(loaded.keys()) == {"a.wav", "b.wav"}
        assert loaded["a.wav"]["alignment_score"] == 0.95


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


# ----------------- build_corpus() force/upsert (PR #340 review 1 fix) ---


class TestBuildCorpusForceUpsert:
    """``--force`` 再実行で manifest.jsonl の同 path が重複しないことを pin
    (PR #340 review 指摘 1 fix)。"""

    @patch("benchmarks.confidence_calibration.build_corpus.load_wav_16k_mono")
    @patch("benchmarks.confidence_calibration.build_corpus.fetch_reference_text")
    @patch("benchmarks.confidence_calibration.build_corpus.chunk_audio_by_vad")
    @patch("benchmarks.confidence_calibration.build_corpus.ffmpeg_trim_and_resample")
    @patch("benchmarks.confidence_calibration.build_corpus.download_audio")
    @patch("livecap_cli.engines.engine_factory.EngineFactory.create_engine")
    def test_force_rerun_does_not_duplicate_manifest_paths(
        self,
        mock_create_engine: MagicMock,
        mock_download: MagicMock,
        mock_ffmpeg: MagicMock,
        mock_chunk_vad: MagicMock,
        mock_fetch_text: MagicMock,
        mock_load_wav: MagicMock,
        tmp_path: Path,
    ):
        # 全 dependencies を mock (実 yt-dlp / ffmpeg / engine 不要)
        mock_download.return_value = tmp_path / "raw_audio.wav"
        mock_ffmpeg.return_value = tmp_path / "normalized.wav"
        mock_load_wav.return_value = np.zeros(16000 * 30, dtype=np.float32)  # 30 sec
        mock_chunk_vad.return_value = [
            (0.0, 1.0),
            (1.5, 2.5),
            (3.0, 4.0),
        ]  # 3 segments
        mock_fetch_text.return_value = "Reference Once upon a time text long enough."

        mock_result = MagicMock()
        mock_result.text = "Once upon a time"
        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = mock_result
        mock_engine.load_model = MagicMock()
        mock_create_engine.return_value = mock_engine

        # 1 回目 build (force=False、新規)
        from benchmarks.confidence_calibration.build_corpus import build_corpus

        output_dir = tmp_path / "corpus" / "ja_clean"
        manifest_path = tmp_path / "corpus" / "manifest.jsonl"

        result1 = build_corpus(
            source="https://example.com/audio",
            reference_text_source="https://example.com/text",
            output_dir=output_dir,
            language="ja",
            manifest_path=manifest_path,
        )
        assert result1.segments_created == 3

        # manifest を読み、3 entry、path は unique であることを確認
        entries_after_first = _load_manifest_entries(manifest_path)
        assert len(entries_after_first) == 3

        with manifest_path.open("r", encoding="utf-8") as f:
            lines_first = [line for line in f.read().splitlines() if line.strip()]
        assert len(lines_first) == 3

        # 2 回目 build (force=True、再生成)
        result2 = build_corpus(
            source="https://example.com/audio",
            reference_text_source="https://example.com/text",
            output_dir=output_dir,
            language="ja",
            manifest_path=manifest_path,
            force=True,
        )
        assert result2.segments_created == 3
        assert result2.segments_skipped == 0

        # 重要 assertion: manifest の line 数も path 数も 3 のまま
        # (旧実装では 6 行になっていた、重複 append のため)
        with manifest_path.open("r", encoding="utf-8") as f:
            lines_second = [line for line in f.read().splitlines() if line.strip()]
        assert len(lines_second) == 3, (
            f"manifest.jsonl should have 3 entries after force re-run, "
            f"got {len(lines_second)}"
        )
        entries_after_second = _load_manifest_entries(manifest_path)
        assert len(entries_after_second) == 3
        assert set(entries_after_second.keys()) == set(entries_after_first.keys())

    @patch("benchmarks.confidence_calibration.build_corpus.load_wav_16k_mono")
    @patch("benchmarks.confidence_calibration.build_corpus.fetch_reference_text")
    @patch("benchmarks.confidence_calibration.build_corpus.chunk_audio_by_vad")
    @patch("benchmarks.confidence_calibration.build_corpus.ffmpeg_trim_and_resample")
    @patch("benchmarks.confidence_calibration.build_corpus.download_audio")
    @patch("livecap_cli.engines.engine_factory.EngineFactory.create_engine")
    def test_build_preserves_other_source_entries(
        self,
        mock_create_engine: MagicMock,
        mock_download: MagicMock,
        mock_ffmpeg: MagicMock,
        mock_chunk_vad: MagicMock,
        mock_fetch_text: MagicMock,
        mock_load_wav: MagicMock,
        tmp_path: Path,
    ):
        """ja_clean を rebuild しても en_clean / non_speech の既存 entry は保持される。

        upsert 実装が source 単位の partial rewrite として正しく動作することを pin。
        """
        manifest_path = tmp_path / "corpus" / "manifest.jsonl"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        # 既存 entry: 別 source (en_clean / ja_non_speech)
        manifest_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {"path": "en_clean/seg_001.wav", "label": "speech", "language": "en"}
                    ),
                    json.dumps(
                        {
                            "path": "ja_non_speech/applause.wav",
                            "label": "non_speech",
                            "language": "ja",
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        # ja_clean を新規 build
        mock_download.return_value = tmp_path / "raw_audio.wav"
        mock_ffmpeg.return_value = tmp_path / "normalized.wav"
        mock_load_wav.return_value = np.zeros(16000 * 10, dtype=np.float32)
        mock_chunk_vad.return_value = [(0.0, 1.0)]  # 1 segment
        mock_fetch_text.return_value = "Reference text Once upon a time."

        mock_result = MagicMock()
        mock_result.text = "Once upon a time"
        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = mock_result
        mock_engine.load_model = MagicMock()
        mock_create_engine.return_value = mock_engine

        from benchmarks.confidence_calibration.build_corpus import build_corpus

        build_corpus(
            source="https://example.com/audio",
            reference_text_source="https://example.com/text",
            output_dir=tmp_path / "corpus" / "ja_clean",
            language="ja",
            manifest_path=manifest_path,
            force=True,
        )

        entries = _load_manifest_entries(manifest_path)
        # 既存 2 + 新規 1 = 3 entries
        assert len(entries) == 3
        assert "en_clean/seg_001.wav" in entries  # 既存保持
        assert "ja_non_speech/applause.wav" in entries  # 既存保持
        assert "ja_clean/segment_0000.wav" in entries  # 新規


# ----------------- CLI --engine-kwargs (PR #340 review 3 fix) ----------


class TestBuildCorpusCli:
    @patch("benchmarks.confidence_calibration.build_corpus.build_corpus")
    def test_cli_passes_engine_kwargs(
        self, mock_build: MagicMock, tmp_path: Path
    ):
        from benchmarks.confidence_calibration.build_corpus import main

        mock_build.return_value = BuildResult(
            segments_created=0,
            segments_skipped=0,
            low_alignment_warnings=0,
            total_duration_sec=0.0,
            manifest_path=tmp_path / "manifest.jsonl",
        )
        rc = main(
            [
                "--source",
                "https://example.com/audio",
                "--reference-text",
                "https://example.com/text",
                "--output-dir",
                str(tmp_path / "ja_clean"),
                "--language",
                "ja",
                "--engine",
                "whispers2t",
                "--engine-kwargs",
                "model_size=base",
                "compute_type=int8",
            ]
        )
        assert rc == 0
        call = mock_build.call_args
        assert call.kwargs["engine_kwargs"] == {
            "model_size": "base",
            "compute_type": "int8",
        }

    @patch("benchmarks.confidence_calibration.build_corpus.build_corpus")
    def test_cli_default_engine_kwargs_is_none(
        self, mock_build: MagicMock, tmp_path: Path
    ):
        from benchmarks.confidence_calibration.build_corpus import main

        mock_build.return_value = BuildResult(
            segments_created=0,
            segments_skipped=0,
            low_alignment_warnings=0,
            total_duration_sec=0.0,
            manifest_path=tmp_path / "manifest.jsonl",
        )
        rc = main(
            [
                "--source",
                "https://example.com/audio",
                "--reference-text",
                "https://example.com/text",
                "--output-dir",
                str(tmp_path / "ja_clean"),
                "--language",
                "ja",
            ]
        )
        assert rc == 0
        # --engine-kwargs なしなら None (空 dict から build_corpus() default の {})
        assert mock_build.call_args.kwargs["engine_kwargs"] is None
