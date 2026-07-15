"""Public SRT serializer for file transcription subtitles (Issue #363).

`FileTranscriptionPipeline` の private serializer (`_build_srt` /
`_build_translated_srt` / `_format_timestamp`) を公開関数として抽出したもの。
CLI など pipeline の外側が `process_file(write_subtitles=False)` の
`result.subtitles` を任意の出力先へ serialize するために使う。

出力形式は pipeline が従来書き出していた SRT とバイト単位で同一:
segment index (renumber しない) / ``HH:MM:SS,mmm --> HH:MM:SS,mmm`` /
本文 / 空行、を ``\\n`` で連結。
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover - 型注釈のみ (循環 import 回避)
    from .file_pipeline import FileSubtitleSegment

__all__ = ["build_srt", "write_srt"]


def _format_timestamp(position: float) -> str:
    td = timedelta(seconds=position)
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    seconds = td.total_seconds() % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")


def build_srt(
    subtitles: Sequence["FileSubtitleSegment"],
    *,
    translated: bool = False,
) -> str:
    """Build SRT content from subtitle segments.

    Args:
        subtitles: `FileProcessingResult.subtitles` の segment 列。
        translated: True の場合、`translated_text` が truthy の segment のみ
            を対象に `translated_text` を本文として出力する (翻訳失敗 segment
            は skip され、index は元の値のまま維持される)。

    Returns:
        SRT 形式の文字列。対象 segment が無ければ空文字列。
    """
    lines: list[str] = []
    for segment in subtitles:
        if translated:
            if not segment.translated_text:
                continue
            text = segment.translated_text
        else:
            text = segment.text
        lines.append(str(segment.index))
        lines.append(
            f"{_format_timestamp(segment.start)} --> {_format_timestamp(segment.end)}"
        )
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def write_srt(
    path: str | Path,
    subtitles: Sequence["FileSubtitleSegment"],
    *,
    translated: bool = False,
) -> Path:
    """Write SRT content to ``path`` (UTF-8) and return the written path.

    親 directory は作成しない (存在しない場合は ``FileNotFoundError``)。
    """
    output_path = Path(path)
    content = build_srt(subtitles, translated=translated)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return output_path
