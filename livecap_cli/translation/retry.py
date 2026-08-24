"""
リトライ

``RetryPolicy`` を提供する。``TranslationNetworkError`` のみを再試行する —
恒久的な失敗 (4xx、解析不能、長すぎる入力) を繰り返し投げても結果は変わらないため。

なぜ policy が呼び出し側にあるか (Issue #402 D10)
------------------------------------------------
以前は retry デコレータを Google adapter に直接付けていたが、**adapter は自分が
リアルタイム字幕に使われているのかファイル処理に使われているのか判断できない**。
両者は要求が正反対で、リアルタイムでは 3 秒遅れた翻訳は無価値なのに対し、ファイル
処理では時間をかけてでも成功させたい。

そこで **分類は adapter、方針は呼び出し側**に分けた。adapter は 503 と 404 の違いを
型で表し、何回・何秒まで試すかは用途を知っている側が決める
(:data:`FILE_RETRY_POLICY`)。

**リアルタイム経路は retry しない。** 失敗したら次の発話へ進む方が、遅れて出すより
良いため。待つ時間の上限は ``StreamTranscriber`` が持っており
(``LIVECAP_TRANSLATION_TIMEOUT``)、この module は関与しない — policy を置いても
``max_attempts=1`` では direct call と同じで、env の parse だけが二重になる。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import Callable, TypeVar

from .exceptions import TranslationNetworkError

logger = logging.getLogger(__name__)

T = TypeVar("T")

@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """リトライの方針。

    Attributes:
        max_attempts: 最大試行回数。1 は「リトライしない」。
        total_timeout_seconds: **新しい試行を開始してよい期限**。``None`` で無制限。
        base_delay: 指数バックオフの初回待機。
        estimated_attempt_seconds: 1 試行にかかる時間の見積。次の試行を始めてよいかの
            判断に使う。**上限の保証ではない** (下記 Note)。

    Note:
        deadline は **試行回数より優先される**。残り時間が次の待機に足りなければ、
        あるいは既に期限を過ぎていれば、試行回数が残っていても打ち切る — リアルタイム
        字幕では「何回試したか」ではなく「いつまでに出るか」が品質だから。

        **これは soft deadline である。** 実行中の 1 試行を途中で止める手段はここには
        無く、それは呼び出される側の責務である。したがって最悪の総時間は
        ``deadline + 実行中の 1 試行`` になる。

        ``estimated_attempt_seconds`` は「残り予算で次の試行が終わりそうか」の判断に
        使う (admission control)。**上限の保証としては扱わない** — HTTP client の
        read timeout はバイト間の待ち時間であって総 wall-clock ではないため、
        見積を超える試行は原理的にあり得る。

        **任意の translator に対して一律の値を宣言してはならない。** ローカルモデル
        (opus_mt / riva) は推論時間が入力に依存し、Google adapter も constructor で
        timeout を変更できるうえ注入 transport の所要時間は保証できない。見積は
        **呼び出し先から取得する** こと (:meth:`BaseTranslator.estimated_attempt_seconds`)。
    """

    max_attempts: int = 1
    total_timeout_seconds: float | None = None
    base_delay: float = 0.5
    estimated_attempt_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1 (got {self.max_attempts})")
        if self.total_timeout_seconds is not None and self.total_timeout_seconds <= 0:
            raise ValueError(
                f"total_timeout_seconds must be positive (got {self.total_timeout_seconds})"
            )
        if self.estimated_attempt_seconds is not None and self.estimated_attempt_seconds <= 0:
            raise ValueError(
                f"estimated_attempt_seconds must be positive "
                f"(got {self.estimated_attempt_seconds})"
            )

    def call(self, func: Callable[[], T], *, sleep: Callable[[float], None] = time.sleep,
             monotonic: Callable[[], float] = time.monotonic) -> T:
        """``func`` を方針に従って実行する。

        ``TranslationNetworkError`` のみ再試行し、他は素通しする。
        """
        started = monotonic()
        last_error: TranslationNetworkError | None = None

        for attempt in range(1, self.max_attempts + 1):
            # 開始時点でも期限を見る。sleep 前だけを見ていると、前の試行自体が
            # 長引いて期限を過ぎていても次の試行を始めてしまう。
            if attempt > 1 and not self._can_start(started, monotonic):
                logger.debug(
                    "Translation retry budget exhausted before attempt %d", attempt
                )
                break
            try:
                return func()
            except TranslationNetworkError as exc:
                last_error = exc
                if attempt == self.max_attempts:
                    break
                delay = self.base_delay * (2 ** (attempt - 1))
                if self.total_timeout_seconds is not None:
                    remaining = self.total_timeout_seconds - (monotonic() - started)
                    if remaining <= delay:
                        logger.debug(
                            "Translation retry budget exhausted after %d attempt(s)", attempt
                        )
                        break
                logger.warning(
                    "Translation failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt,
                    self.max_attempts,
                    delay,
                    exc,
                )
                sleep(delay)

        assert last_error is not None
        raise last_error

    def _can_start(self, started: float, monotonic: Callable[[], float]) -> bool:
        """次の試行を始めてよいか (admission control)。

        見積が与えられていれば、**それが収まるだけの残りが無い限り始めない** —
        始めても deadline を大きく超えたところまで走るだけで益が無いため。見積は
        保証ではないので、これは超過を減らす措置であって無くす措置ではない。
        """
        if self.total_timeout_seconds is None:
            return True
        remaining = self.total_timeout_seconds - (monotonic() - started)
        return remaining >= (self.estimated_attempt_seconds or 0)


#: ファイル処理: 時間をかけてでも成功させる。
#:
#: 1 試行あたりの見積は**ここでは宣言しない** — この policy は任意の
#: ``BaseTranslator`` に適用され、所要時間は translator ごとに違うため。
#: 呼び出し側が :func:`for_translator` で translator から取得して構成する。
FILE_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    total_timeout_seconds=10.0,
    base_delay=1.0,
)


def for_translator(policy: RetryPolicy, translator: object) -> RetryPolicy:
    """translator の見積を policy へ取り込む (admission control 用)。

    見積が無ければ policy をそのまま返す。``BaseTranslator`` の既定は ``None`` なので、
    所要時間を見積もれない実装 (ローカルモデル等) は自動的にそのまま通る。
    """
    estimated = getattr(translator, "estimated_attempt_seconds", None)
    if estimated is None:
        return policy
    return replace(policy, estimated_attempt_seconds=estimated)
