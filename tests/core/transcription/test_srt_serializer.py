"""Public SRT serializer (`build_srt` / `write_srt`) のテスト (Issue #363)。

`FileTranscriptionPipeline` の private serializer から抽出した公開関数の
出力形式と、pipeline 側委譲 (`_write_srt`) の出力一致を固定する。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from livecap_cli.transcription import build_srt, write_srt
from livecap_cli.transcription.file_pipeline import (
    FileSubtitleSegment,
    FileTranscriptionPipeline,
)


def _segment(
    index: int,
    start: float,
    end: float,
    text: str,
    translated_text: str | None = None,
) -> FileSubtitleSegment:
    return FileSubtitleSegment(
        index=index,
        start=start,
        end=end,
        text=text,
        translated_text=translated_text,
        target_language="en" if translated_text else None,
    )


class TestBuildSrt:
    def test_basic_format(self):
        content = build_srt(
            [
                _segment(1, 0.0, 1.5, "こんにちは"),
                _segment(2, 1.5, 3.0, "世界"),
            ]
        )

        assert content == (
            "1\n"
            "00:00:00,000 --> 00:00:01,500\n"
            "こんにちは\n"
            "\n"
            "2\n"
            "00:00:01,500 --> 00:00:03,000\n"
            "世界\n"
            ""
        )

    def test_timestamp_hours_and_comma_decimal(self):
        content = build_srt([_segment(1, 3661.25, 3662.0, "x")])

        assert "01:01:01,250 --> 01:01:02,000" in content

    def test_empty_list_returns_empty_string(self):
        assert build_srt([]) == ""

    def test_translated_filters_and_keeps_original_index(self):
        """translated=True: 翻訳失敗 segment を skip、index は renumber しない"""
        content = build_srt(
            [
                _segment(1, 0.0, 1.0, "一", translated_text="one"),
                _segment(2, 1.0, 2.0, "二", translated_text=None),  # 翻訳失敗
                _segment(3, 2.0, 3.0, "三", translated_text="three"),
            ],
            translated=True,
        )

        assert "one" in content
        assert "three" in content
        assert "二" not in content  # 原文 fallback しない
        blocks = [b for b in content.split("\n\n") if b.strip()]
        assert len(blocks) == 2
        assert blocks[0].startswith("1\n")
        assert blocks[1].startswith("3\n")  # 元 index 維持

    def test_translated_empty_when_no_translations(self):
        content = build_srt(
            [_segment(1, 0.0, 1.0, "一", translated_text=None)],
            translated=True,
        )
        assert content == ""


class TestWriteSrt:
    def test_writes_utf8_and_returns_path(self, tmp_path: Path):
        out = tmp_path / "out.srt"
        subtitles = [_segment(1, 0.0, 1.0, "こんにちは")]

        returned = write_srt(out, subtitles)

        assert returned == out
        assert out.read_text(encoding="utf-8") == build_srt(subtitles)

    def test_missing_parent_dir_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            write_srt(
                tmp_path / "no_such_dir" / "out.srt",
                [_segment(1, 0.0, 1.0, "x")],
            )

    def test_translated_mode(self, tmp_path: Path):
        out = tmp_path / "out_en.srt"
        subtitles = [_segment(1, 0.0, 1.0, "一", translated_text="one")]

        write_srt(out, subtitles, translated=True)

        content = out.read_text(encoding="utf-8")
        assert "one" in content
        assert "一" not in content


class TestPipelineDelegation:
    """pipeline の SRT 書き出しが serializer と同一出力であること (委譲 regression)"""

    def test_write_srt_matches_build_srt(self, tmp_path: Path):
        source = tmp_path / "audio.wav"
        source.touch()
        subtitles = [
            _segment(1, 0.0, 1.5, "こんにちは"),
            _segment(2, 1.5, 3.0, "世界"),
        ]

        pipeline = FileTranscriptionPipeline()
        try:
            output_path = pipeline._write_srt(source, subtitles)
        finally:
            pipeline.close()

        assert output_path == source.with_suffix(".srt")
        assert output_path.read_text(encoding="utf-8") == build_srt(subtitles)


class TestCloseDefense:
    """close() の getattr 防御 (#363 二次エラー): _temp_root 未初期化でも例外を出さない"""

    def test_close_on_uninitialised_instance(self):
        pipeline = FileTranscriptionPipeline.__new__(FileTranscriptionPipeline)
        # __init__ 未実行 (構築時 TypeError 相当) — AttributeError を出さないこと
        pipeline.close()
        del pipeline
