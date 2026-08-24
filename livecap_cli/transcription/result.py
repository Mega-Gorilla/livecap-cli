"""統一された文字起こし結果型

リアルタイム・ファイル文字起こし両方で使用する統一型を定義。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

#: なぜ ``translated_text`` が入っていないのか (Issue #402 D10)。
#:
#: 原文がそのまま字幕になる状態は 1 つではない。表示側が「翻訳が壊れている」のか
#: 「翻訳を頼んでいない」のか「輻輳で今回だけ飛ばした」のかを区別できないと、
#: 障害と正常な方針の区別がつかない。
#:
#: * ``not_requested``  -- 翻訳を指定していない
#: * ``translated``     -- 正常に翻訳された
#: * ``failed``         -- 翻訳が失敗した
#: * ``skipped_busy``   -- 前の翻訳が終わっておらず今回は飛ばした (輻輳時の方針)
#: * ``empty``          -- 翻訳は成功したが空文字だった
TranslationState = Literal[
    "not_requested", "translated", "failed", "skipped_busy", "empty"
]


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
        translation_state: ``translated_text`` が無い理由。詳細は
            :data:`TranslationState`。既定は ``"not_requested"`` なので、翻訳を
            使っていない caller は無改修で動く。
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
    # Issue #402 D10: 「原文が出ている理由」を segment 単位で運ぶ。
    # 独立イベントにすると (source_id, start_time, end_time) での突き合わせが要り、
    # float の一致比較と配信順序の保証に依存してしまう。結果そのものの属性にすれば
    # 表示側は手元の result を見るだけで済む。
    translation_state: TranslationState = "not_requested"

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
