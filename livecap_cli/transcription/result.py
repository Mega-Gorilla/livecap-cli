"""統一された文字起こし結果型

リアルタイム・ファイル文字起こし両方で使用する統一型を定義。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """
    文字起こし結果を表すイミュータブルなデータクラス

    リアルタイム・ファイル文字起こし両方で使用する統一型。

    Attributes:
        text: 文字起こしテキスト
        start_time: セグメント開始時刻（秒）
        end_time: セグメント終了時刻（秒）
        is_final: 確定結果かどうか（リアルタイム用）
        confidence: 信頼度スコア（0.0-1.0）
        language: 検出された言語コード（= 翻訳元言語）
        source_id: 音声ソースID（マルチソース対応用）
        translated_text: 翻訳結果テキスト（翻訳なしの場合は None）
        target_language: 翻訳先言語コード（翻訳なしの場合は None）
    """

    text: str
    start_time: float
    end_time: float
    is_final: bool = True
    confidence: float = 1.0
    language: str = ""
    source_id: str = "default"
    # Phase 5: 翻訳フィールド
    translated_text: Optional[str] = None
    target_language: Optional[str] = None

    @property
    def duration(self) -> float:
        """セグメントの長さ（秒）"""
        return self.end_time - self.start_time

    def to_srt_entry(self, index: int) -> str:
        """
        SRT形式のエントリに変換

        Args:
            index: SRTエントリの番号（1から開始）

        Returns:
            SRT形式の文字列
        """
        return (
            f"{index}\n"
            f"{_format_srt_time(self.start_time)} --> {_format_srt_time(self.end_time)}\n"
            f"{self.text}\n"
        )


@dataclass(frozen=True, slots=True)
class InterimResult:
    """
    中間結果（確定前の途中経過）

    TranscriptionResult とは別の型として明示的に区別。
    リアルタイム文字起こしで、発話中の途中経過を表示するために使用。

    Attributes:
        text: 中間テキスト
        accumulated_time: 発話開始からの累積時間（秒）
        source_id: 音声ソースID
    """

    text: str
    accumulated_time: float
    source_id: str = "default"


def _format_srt_time(seconds: float) -> str:
    """
    秒数をSRT形式のタイムスタンプに変換

    Args:
        seconds: 秒数

    Returns:
        "HH:MM:SS,mmm" 形式の文字列
    """
    if seconds < 0:
        seconds = 0.0

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
