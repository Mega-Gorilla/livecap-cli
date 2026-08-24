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
from livecap_cli.translation.retry import (
    DEFAULT_REALTIME_DEADLINE_SECONDS,
    ENV_REALTIME_DEADLINE,
    FILE_RETRY_POLICY,
    REALTIME_RETRY_POLICY,
    RetryPolicy,
    resolve_realtime_deadline,
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


# ---------------------------------------------------------------------------
# RetryPolicy (Issue #402 D10)
# ---------------------------------------------------------------------------


class FakeClock:
    """Advances only when the policy sleeps, so tests never actually wait."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def monotonic(self) -> float:
        return self.now


def _failing(times: int, result: str = "ok"):
    """A callable that raises a network error `times` times, then succeeds."""
    calls: list[int] = []

    def call():
        calls.append(1)
        if len(calls) <= times:
            raise TranslationNetworkError("temporary", provider="google")
        return result

    call.calls = calls  # type: ignore[attr-defined]
    return call


class TestRetryPolicyValidation:
    def test_rejects_zero_attempts(self):
        with pytest.raises(ValueError, match="max_attempts"):
            RetryPolicy(max_attempts=0)

    def test_rejects_non_positive_deadline(self):
        with pytest.raises(ValueError, match="total_timeout_seconds"):
            RetryPolicy(max_attempts=3, total_timeout_seconds=0)


class TestRetryPolicyBehaviour:
    def test_success_on_first_attempt_does_not_sleep(self):
        clock = FakeClock()
        func = _failing(0)
        assert RetryPolicy(max_attempts=3).call(
            func, sleep=clock.sleep, monotonic=clock.monotonic
        ) == "ok"
        assert len(func.calls) == 1
        assert clock.slept == []

    def test_retries_then_succeeds(self):
        clock = FakeClock()
        func = _failing(2)
        policy = RetryPolicy(max_attempts=3, total_timeout_seconds=10.0, base_delay=1.0)
        assert policy.call(func, sleep=clock.sleep, monotonic=clock.monotonic) == "ok"
        assert len(func.calls) == 3
        assert clock.slept == [1.0, 2.0]

    def test_permanent_error_is_not_retried(self):
        """Only TranslationNetworkError is transient; the rest is an answer."""
        clock = FakeClock()
        calls: list[int] = []

        def call():
            calls.append(1)
            raise TranslationError("permanent", provider="google")

        with pytest.raises(TranslationError):
            RetryPolicy(max_attempts=3).call(
                call, sleep=clock.sleep, monotonic=clock.monotonic
            )
        assert len(calls) == 1

    def test_raises_the_last_network_error(self):
        clock = FakeClock()
        func = _failing(99)
        with pytest.raises(TranslationNetworkError):
            RetryPolicy(max_attempts=2, base_delay=1.0).call(
                func, sleep=clock.sleep, monotonic=clock.monotonic
            )
        assert len(func.calls) == 2


class TestRetryPolicyDeadline:
    def test_deadline_stops_retrying_before_the_attempt_budget(self):
        """A subtitle that arrives late is worthless, so time wins over count."""
        clock = FakeClock()
        func = _failing(99)
        policy = RetryPolicy(max_attempts=5, total_timeout_seconds=2.0, base_delay=1.0)
        with pytest.raises(TranslationNetworkError):
            policy.call(func, sleep=clock.sleep, monotonic=clock.monotonic)

        assert sum(clock.slept) < 2.0
        assert len(func.calls) < 5

    def test_total_sleep_never_exceeds_the_deadline(self):
        for deadline in (1.0, 2.0, 5.0, 10.0):
            clock = FakeClock()
            policy = RetryPolicy(
                max_attempts=10, total_timeout_seconds=deadline, base_delay=1.0
            )
            with pytest.raises(TranslationNetworkError):
                policy.call(_failing(99), sleep=clock.sleep, monotonic=clock.monotonic)
            assert sum(clock.slept) < deadline, deadline


class TestShippedPolicies:
    def test_realtime_does_not_retry(self):
        clock = FakeClock()
        func = _failing(99)
        with pytest.raises(TranslationNetworkError):
            REALTIME_RETRY_POLICY.call(
                func, sleep=clock.sleep, monotonic=clock.monotonic
            )
        assert len(func.calls) == 1
        assert clock.slept == []

    def test_realtime_deadline_default(self):
        assert REALTIME_RETRY_POLICY.total_timeout_seconds == 2.0

    def test_file_retries_up_to_three_times(self):
        clock = FakeClock()
        func = _failing(2)
        assert FILE_RETRY_POLICY.call(
            func, sleep=clock.sleep, monotonic=clock.monotonic
        ) == "ok"
        assert len(func.calls) == 3

    def test_file_deadline_leaves_room_for_every_attempt(self):
        """Raised from 10s to 30s: with an 8s worst-case attempt declared, a 10s
        budget only ever fit one attempt, so retry never actually happened."""
        assert FILE_RETRY_POLICY.total_timeout_seconds == 30.0


class TestRealtimeDeadlineConfiguration:
    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv(ENV_REALTIME_DEADLINE, raising=False)
        assert resolve_realtime_deadline() == DEFAULT_REALTIME_DEADLINE_SECONDS

    def test_environment_override(self, monkeypatch):
        monkeypatch.setenv(ENV_REALTIME_DEADLINE, "4.5")
        assert resolve_realtime_deadline() == 4.5

    @pytest.mark.parametrize("raw", ["abc", "", "-1", "0"])
    def test_invalid_values_fall_back_to_the_default(self, monkeypatch, raw):
        """A bad setting must not make translation unusable."""
        monkeypatch.setenv(ENV_REALTIME_DEADLINE, raw)
        assert resolve_realtime_deadline() == DEFAULT_REALTIME_DEADLINE_SECONDS


class TestDeadlineIsARealBound:
    """`total_timeout_seconds` must bound the wall clock, not just the sleeps.

    Checking the budget only before sleeping let a slow attempt start with 0.1s
    left and run for seconds: a 10s policy measured 10.8s across two calls.
    """

    def test_slow_attempt_does_not_overrun_the_deadline(self):
        clock = FakeClock()
        calls: list[int] = []

        def slow_fail():
            calls.append(1)
            clock.now += 4.9  # the attempt itself consumes budget
            raise TranslationNetworkError("503", provider="google")

        policy = RetryPolicy(
            max_attempts=3,
            total_timeout_seconds=10.0,
            base_delay=1.0,
            attempt_timeout_seconds=5.0,
        )
        with pytest.raises(TranslationNetworkError):
            policy.call(slow_fail, sleep=clock.sleep, monotonic=clock.monotonic)

        assert clock.now <= 10.0
        assert len(calls) == 1  # a second attempt could not have finished in time

    def test_fast_attempts_still_use_the_full_attempt_budget(self):
        clock = FakeClock()
        calls: list[int] = []

        def quick_fail():
            calls.append(1)
            clock.now += 0.2
            raise TranslationNetworkError("503", provider="google")

        policy = RetryPolicy(
            max_attempts=3,
            total_timeout_seconds=10.0,
            base_delay=1.0,
            attempt_timeout_seconds=5.0,
        )
        with pytest.raises(TranslationNetworkError):
            policy.call(quick_fail, sleep=clock.sleep, monotonic=clock.monotonic)

        assert len(calls) == 3
        assert clock.now <= 10.0

    def test_declared_attempt_cost_gates_the_next_start(self):
        """With no room for another attempt we stop, even with attempts left."""
        clock = FakeClock()
        calls: list[int] = []

        def fail():
            calls.append(1)
            clock.now += 1.0
            raise TranslationNetworkError("503", provider="google")

        policy = RetryPolicy(
            max_attempts=5,
            total_timeout_seconds=4.0,
            base_delay=0.5,
            attempt_timeout_seconds=3.0,
        )
        with pytest.raises(TranslationNetworkError):
            policy.call(fail, sleep=clock.sleep, monotonic=clock.monotonic)

        assert clock.now <= 4.0
        assert len(calls) < 5

    def test_rejects_non_positive_attempt_timeout(self):
        with pytest.raises(ValueError, match="attempt_timeout_seconds"):
            RetryPolicy(max_attempts=3, attempt_timeout_seconds=0)


class TestShippedPolicyBounds:
    def test_file_policy_fits_three_attempts_in_its_deadline(self):
        p = FILE_RETRY_POLICY
        worst = p.attempt_timeout_seconds * p.max_attempts + sum(
            p.base_delay * (2**i) for i in range(p.max_attempts - 1)
        )
        assert worst <= p.total_timeout_seconds, worst

    def test_file_policy_declares_its_attempt_cost(self):
        """Without the declaration the deadline is only a start bound."""
        assert FILE_RETRY_POLICY.attempt_timeout_seconds is not None
