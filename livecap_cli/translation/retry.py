"""
リトライデコレータ

指数バックオフによるリトライ機能を提供。
主に Google Translate などのネットワーク API 呼び出しで使用。
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Callable, TypeVar

from .exceptions import TranslationNetworkError

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)


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
