"""
リトライ

``RetryPolicy`` (推奨) と ``with_retry`` デコレータを提供する。
いずれも ``TranslationNetworkError`` のみを再試行する — 恒久的な失敗
(4xx、解析不能、長すぎる入力) を繰り返し投げても結果は変わらないため。

なぜ policy が呼び出し側にあるか (Issue #402 D10)
------------------------------------------------
以前は ``@with_retry`` を Google adapter に直接付けていたが、**adapter は自分が
リアルタイム字幕に使われているのかファイル処理に使われているのか判断できない**。
両者は要求が正反対で、リアルタイムでは 3 秒遅れた翻訳は無価値なのに対し、ファイル
処理では時間をかけてでも成功させたい。

そこで **分類は adapter、方針は呼び出し側**に分けた。adapter は 503 と 404 の違いを
型で表し、何回・何秒まで試すかは用途を知っている側が :data:`REALTIME_RETRY_POLICY` /
:data:`FILE_RETRY_POLICY` で決める。
"""

from __future__ import annotations

import functools
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

from .exceptions import TranslationNetworkError

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)
T = TypeVar("T")

#: リアルタイム字幕の deadline 既定値 (秒)。実測では Session 再利用時の中央値が
#: 166-191ms、観測した最悪が 1331ms だったため、正常な遅い応答を切らずに被害を
#: 2 秒で止められる。環境変数で上書き可能 (回線・地域差で一律失敗させないため)。
DEFAULT_REALTIME_DEADLINE_SECONDS = 2.0

ENV_REALTIME_DEADLINE = "LIVECAP_TRANSLATION_REALTIME_DEADLINE"


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """リトライの方針。

    Attributes:
        max_attempts: 最大試行回数。1 は「リトライしない」。
        total_timeout_seconds: **新しい試行を開始してよい期限**。``None`` で無制限。
        base_delay: 指数バックオフの初回待機。

    Note:
        deadline は **試行回数より優先される**。残り時間が次の待機に足りなければ、
        あるいは既に期限を過ぎていれば、試行回数が残っていても打ち切る — リアルタイム
        字幕では「何回試したか」ではなく「いつまでに出るか」が品質だから。

        **実行中の 1 試行を途中で止める手段はここには無い** — それは呼び出される側の
        HTTP timeout の役目である。そこで ``attempt_timeout_seconds`` に「1 試行の
        最悪所要時間」を宣言してもらい、**残り予算がそれを下回ったら次を始めない**
        ことで総時間を実際に縛る。宣言が正しい限り、総時間は deadline を超えない。

        ``attempt_timeout_seconds`` が ``None`` の場合、deadline が縛るのは「開始」
        だけになり、最悪の総時間は ``deadline + 実行中の 1 試行`` になる。
    """

    max_attempts: int = 1
    total_timeout_seconds: float | None = None
    base_delay: float = 0.5
    attempt_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1 (got {self.max_attempts})")
        if self.total_timeout_seconds is not None and self.total_timeout_seconds <= 0:
            raise ValueError(
                f"total_timeout_seconds must be positive (got {self.total_timeout_seconds})"
            )
        if self.attempt_timeout_seconds is not None and self.attempt_timeout_seconds <= 0:
            raise ValueError(
                f"attempt_timeout_seconds must be positive (got {self.attempt_timeout_seconds})"
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
        """次の試行を始めてよいか。

        ``attempt_timeout_seconds`` が宣言されていれば、**それが収まるだけの残りが
        無い限り始めない** — 始めてしまうと deadline を超えるところまで走り切って
        しまうため。
        """
        if self.total_timeout_seconds is None:
            return True
        remaining = self.total_timeout_seconds - (monotonic() - started)
        return remaining >= (self.attempt_timeout_seconds or 0)


def resolve_realtime_deadline() -> float:
    """リアルタイム deadline を環境変数から解決する (不正値は既定へ)。"""
    raw = os.environ.get(ENV_REALTIME_DEADLINE)
    if raw is None:
        return DEFAULT_REALTIME_DEADLINE_SECONDS
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Invalid %s value %r, using default %.1fs",
            ENV_REALTIME_DEADLINE,
            raw,
            DEFAULT_REALTIME_DEADLINE_SECONDS,
        )
        return DEFAULT_REALTIME_DEADLINE_SECONDS
    if value <= 0:
        logger.warning(
            "%s must be positive (got %.1f), using default %.1fs",
            ENV_REALTIME_DEADLINE,
            value,
            DEFAULT_REALTIME_DEADLINE_SECONDS,
        )
        return DEFAULT_REALTIME_DEADLINE_SECONDS
    return value


#: リアルタイム字幕: 失敗したら次の発話へ進む。遅れて出すより落とす方がよい。
#:
#: ``max_attempts=1`` なので **この policy 自体は時間を縛らない** — 縛れるのは
#: 「次の試行を始めるか」だけで、リトライしないなら判断する場面が無いため。
#: リアルタイムの実効的な上限は **adapter の HTTP timeout** であり、同じ
#: :func:`resolve_realtime_deadline` の値から構成する (配線は PR 2)。
REALTIME_RETRY_POLICY = RetryPolicy(
    max_attempts=1, total_timeout_seconds=resolve_realtime_deadline()
)

#: ファイル処理: 時間をかけてでも成功させる。バッチ処理なのでレイテンシは重要でない。
#: ``attempt_timeout_seconds`` は Google adapter の既定 ``(connect 3s, read 5s)`` の
#: 最悪値。3 試行 + バックオフが 30 秒に収まるよう deadline を取ってある
#: (8 + 1 + 8 + 2 + 8 = 27 秒)。
FILE_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    total_timeout_seconds=30.0,
    base_delay=1.0,
    attempt_timeout_seconds=8.0,
)


def with_retry(max_retries: int = 3, base_delay: float = 1.0) -> Callable[[F], F]:
    """
    指数バックオフリトライデコレータ

    TranslationNetworkError が発生した場合にリトライを行う。
    リトライ間隔は指数的に増加（base_delay * 2^attempt）。

    Args:
        max_retries: 最大リトライ回数（デフォルト: 3）
        base_delay: 初回リトライまでの待機時間（秒、デフォルト: 1.0）

    Returns:
        デコレータ関数

    Examples:
        >>> @with_retry(max_retries=3, base_delay=1.0)
        ... def translate_text(text):
        ...     # ネットワーク API 呼び出し
        ...     pass
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except TranslationNetworkError as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)
                        logger.warning(
                            "Translation failed (attempt %d/%d), retrying in %.1fs: %s",
                            attempt + 1,
                            max_retries,
                            delay,
                            e,
                        )
                        time.sleep(delay)
            raise last_error

        return wrapper  # type: ignore[return-value]

    return decorator
