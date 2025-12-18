"""
リトライデコレータのテスト
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from livecap_cli.translation.exceptions import (
    TranslationError,
    TranslationNetworkError,
)
from livecap_cli.translation.retry import with_retry


class TestWithRetry:
    """with_retry デコレータのテスト"""

    def test_success_first_attempt(self):
        """初回成功時はリトライしない"""
        mock_func = MagicMock(return_value="success")
        decorated = with_retry(max_retries=3)(mock_func)
        assert decorated() == "success"
        assert mock_func.call_count == 1

    def test_success_after_failure(self):
        """失敗後にリトライして成功"""
        mock_func = MagicMock(
            side_effect=[
                TranslationNetworkError("fail1"),
                TranslationNetworkError("fail2"),
                "success",
            ]
        )
        decorated = with_retry(max_retries=3, base_delay=0.01)(mock_func)
        assert decorated() == "success"
        assert mock_func.call_count == 3

    def test_retry_exhausted(self):
        """リトライ回数を使い切った場合"""
        mock_func = MagicMock(side_effect=TranslationNetworkError("always fail"))
        decorated = with_retry(max_retries=2, base_delay=0.01)(mock_func)
        with pytest.raises(TranslationNetworkError, match="always fail"):
            decorated()
        assert mock_func.call_count == 2

    def test_non_network_error_not_retried(self):
        """TranslationNetworkError 以外はリトライしない"""
        mock_func = MagicMock(side_effect=TranslationError("non-network error"))
        decorated = with_retry(max_retries=3, base_delay=0.01)(mock_func)
        with pytest.raises(TranslationError, match="non-network error"):
            decorated()
        # リトライしないので1回のみ呼ばれる
        assert mock_func.call_count == 1

    def test_preserves_function_metadata(self):
        """functools.wraps でメタデータが保持される"""

        @with_retry(max_retries=3)
        def my_function():
            """My docstring"""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring"

    def test_arguments_passed_through(self):
        """引数が正しく渡される"""
        mock_func = MagicMock(return_value="result")
        decorated = with_retry(max_retries=3)(mock_func)

        decorated("arg1", "arg2", kwarg1="value1")

        mock_func.assert_called_once_with("arg1", "arg2", kwarg1="value1")
