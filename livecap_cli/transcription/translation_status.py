"""翻訳エンジンの稼働状態イベント (Issue #402 D1)。

なぜ結果型と分けるのか
--------------------
翻訳が失敗したとき、以前は ``logger.warning`` を出して原文をそのまま字幕にして
いた。ユーザから見ると「日本語→英語が日本語→日本語になった」だけで、何が起きたのか
分からない。実際その状態で「モデルを変えても再起動しても直らない」という報告が来た
(原因は Google 側の User-Agent 制限で、こちら側では何も変わっていなかった)。

そこで失敗を表に出すが、**診断メッセージを ``TranscriptionResult`` へ混ぜない**。
字幕テキストを運ぶ型にエラー文字列を載せると、表示側がそれを字幕として出してしまう。
セッションの状態として別に扱う。

個々の字幕が「なぜ原文のままなのか」は別の問いで、そちらは
``TranscriptionResult.translation_state`` が答える。本イベントは
**エンジンが壊れた / 直った**という状態遷移だけを運ぶ。

前例は ``UtteranceSettledEvent`` (``transcription/utterance.py``)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

__all__ = ["TranslationStatusEvent", "TranslationErrorType", "TranslationStatus"]

TranslationStatus = Literal["failed", "recovered"]

#: 呼び出し側が対応を変えられる粒度だけを公開する。
#:
#: * ``network``  -- 一時的。待てば直る可能性がある
#: * ``timeout``  -- 予算内に返らなかった
#: * ``fatal``    -- 設定やレイアウト変更など、待っても直らない
TranslationErrorType = Literal["network", "timeout", "fatal"]


@dataclass(frozen=True, slots=True)
class TranslationStatusEvent:
    """翻訳エンジンが壊れた / 直ったことの通知。

    **segment ごとには発火しない。** 状態が変わったときだけ 1 回出る:

    ==================  ================
    遷移                通知
    ==================  ================
    healthy -> failed   ``failed``
    failed  -> failed   なし
    failed  -> healthy  ``recovered``
    ==================  ================

    Attributes:
        translator: 翻訳エンジン識別子 ("google" 等)。
        status: ``"failed"`` または ``"recovered"``。
        error_type: 失敗の種別。``recovered`` では ``None``。
        message: **sanitize 済みの**メッセージ。``recovered`` では ``None``。
        recoverable: 待てば直る見込みがあるか。``recovered`` では ``None``。

    Note:
        ``message`` に**翻訳対象テキストを含めてはならない**。テキストは Google への
        GET query に入るため、通信ライブラリの例外文字列には発話内容が URL ごと
        含まれる。adapter 側で ``from None`` と構造化フィールドにより除去済みだが、
        イベントは GUI まで届くので、ここでも同じ制約が要る (Issue #402 D8)。
    """

    translator: str
    status: TranslationStatus
    error_type: Optional[TranslationErrorType] = None
    message: Optional[str] = None
    recoverable: Optional[bool] = None

    def __post_init__(self) -> None:
        """不変条件を constructor で強制する。

        :meth:`failed` / :meth:`recovered` を用意しても dataclass の constructor は
        公開されたままなので、``status="recovered"`` なのに ``error_type`` が入って
        いる、といった状態が作れてしまう。読む側がそれを想定しなくて済むように弾く。
        """
        if self.status not in ("failed", "recovered"):
            raise ValueError(f"status must be 'failed' or 'recovered' (got {self.status!r})")

        if self.status == "failed":
            if self.error_type is None:
                raise ValueError("a failed event must carry an error_type")
            if self.error_type not in ("network", "timeout", "fatal"):
                raise ValueError(f"unknown error_type {self.error_type!r}")
        else:
            for field, value in (
                ("error_type", self.error_type),
                ("message", self.message),
                ("recoverable", self.recoverable),
            ):
                if value is not None:
                    raise ValueError(f"a recovered event must not carry {field}")

        if not self.translator:
            raise ValueError("translator must be a non-empty identifier")

    @classmethod
    def failed(
        cls,
        translator: str,
        error_type: TranslationErrorType,
        message: str,
        *,
        recoverable: bool,
    ) -> "TranslationStatusEvent":
        """失敗イベント。``message`` は sanitize 済みのものだけを渡すこと。"""
        return cls(
            translator=translator,
            status="failed",
            error_type=error_type,
            message=message,
            recoverable=recoverable,
        )

    @classmethod
    def recovered(cls, translator: str) -> "TranslationStatusEvent":
        """復旧イベント。回復を知らせないと、いつまで壊れているのか分からない。"""
        return cls(translator=translator, status="recovered")
