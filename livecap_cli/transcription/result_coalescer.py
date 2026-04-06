"""短い ASR 結果を後続と結合し、字幕表示の自然さを改善する。

VAD・ASR のパイプラインには一切干渉しない出力整形レイヤー。
判定は外部ライブラリに依存せず、多段ヒューリスティクスで行う。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from .result import TranscriptionResult

# スペース区切りが必要な言語セット
_SPACE_DELIMITED_LANGS = frozenset(
    {
        "en",
        "fr",
        "de",
        "es",
        "it",
        "pt",
        "nl",
        "ru",
        "ko",
        "pl",
        "sv",
        "da",
        "no",
        "fi",
        "cs",
        "hu",
        "ro",
        "tr",
        "vi",
        "id",
        "ms",
    }
)

# 文末句読点（全角・半角）
_SENTENCE_ENDINGS = frozenset("。！？.!?")


class ResultCoalescer:
    """短い ASR 結果を後続と結合し、字幕表示の自然さを改善する。

    VAD・ASR のパイプラインには一切干渉しない出力整形レイヤー。
    判定は外部ライブラリに依存せず、多段ヒューリスティクスで行う。
    将来的に ICU BreakIterator 等へのバックエンド差し替えが可能な設計。
    """

    def __init__(
        self,
        max_words: int = 1,
        max_chars_single_token: int = 4,
        merge_window_s: float = 2.0,
    ) -> None:
        self._max_words = max_words
        self._max_chars_single_token = max_chars_single_token
        self._merge_window_s = merge_window_s
        self._pending: Optional[TranscriptionResult] = None

    def push(
        self, result: TranscriptionResult, now: float
    ) -> list[TranscriptionResult]:
        """結果を受け取り、0〜2 件の確定結果を返す。

        Args:
            result: ASR 結果
            now: 現在の音声タイムライン時刻（秒）
        """
        outputs: list[TranscriptionResult] = []

        if self._pending is not None:
            gap = result.start_time - self._pending.end_time
            if gap <= self._merge_window_s:
                # 窓内 → pending と result を結合
                merged = self._merge(self._pending, result)
                self._pending = None
                # 結合後もまだ短い場合は再度保留
                if self._is_short(merged.text):
                    self._pending = merged
                else:
                    outputs.append(merged)
                return outputs
            else:
                # 窓外 → pending を単独 flush
                outputs.append(self._pending)
                self._pending = None

        # 新しい result の判定
        if self._is_short(result.text):
            self._pending = result
        else:
            outputs.append(result)

        return outputs

    def flush(
        self, now: float, *, force: bool = False
    ) -> Optional[TranscriptionResult]:
        """保留中の結果をタイムアウト判定して返す。

        期限内なら None。finalize 時は force=True で強制 flush。
        feed_audio() の末尾で毎回呼ぶことで、タイムアウト flush を実現する。
        """
        if self._pending is None:
            return None

        if force or now >= self._pending.end_time + self._merge_window_s:
            result = self._pending
            self._pending = None
            return result

        return None

    def _is_short(self, text: str) -> bool:
        """多段ヒューリスティクスで「短い発話」を判定する。

        判定ロジック:
        1. 句読点終端 → 完結した文としてマージしない
        2. スペース区切り（2語以上）→ word_count で判定
        3. 単一トークン（CJK / 単語）→ char_count で判定
        """
        stripped = text.strip()
        if not stripped:
            return False

        # 1. 句読点終端 → 完結した文としてマージしない
        if stripped[-1] in _SENTENCE_ENDINGS:
            return False

        words = stripped.split()

        # 2. スペース区切り（2語以上）→ word_count で判定
        if len(words) >= 2:
            return len(words) <= self._max_words

        # 3. 単一トークン（CJK / 単語）→ char_count で判定
        return len(stripped) <= self._max_chars_single_token

    def _join_text(self, a: str, b: str, language: Optional[str]) -> str:
        """言語に応じたテキスト結合。"""
        if language and language[:2].lower() in _SPACE_DELIMITED_LANGS:
            return f"{a} {b}"
        return f"{a}{b}"

    def _merge(
        self, a: TranscriptionResult, b: TranscriptionResult
    ) -> TranscriptionResult:
        """2 つの結果を結合する。"""
        language = a.language or b.language
        joined = self._join_text(a.text, b.text, language)
        return replace(
            a,
            text=joined,
            end_time=b.end_time,
            confidence=min(a.confidence, b.confidence),
            language=language,
            # 翻訳は coalescer 出力後に実行するため、ここでは None
            translated_text=None,
            target_language=None,
        )

    def reset(self) -> None:
        """状態リセット。"""
        self._pending = None
