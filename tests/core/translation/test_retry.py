"""RetryPolicy のテスト (Issue #402)。

リトライは adapter ではなく呼び出し側が持つ。分類は adapter、方針はここ。
"""

from __future__ import annotations

import pytest

from livecap_cli.translation.exceptions import (
    TranslationError,
    TranslationNetworkError,
)
from livecap_cli.translation.retry import (
    FILE_RETRY_POLICY,
    RetryPolicy,
    for_translator,
)


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
    def test_file_retries_up_to_three_times(self):
        clock = FakeClock()
        func = _failing(2)
        assert FILE_RETRY_POLICY.call(
            func, sleep=clock.sleep, monotonic=clock.monotonic
        ) == "ok"
        assert len(func.calls) == 3

    def test_file_deadline_is_ten_seconds(self):
        """Matches Issue #402 D10 and the CHANGELOG."""
        assert FILE_RETRY_POLICY.total_timeout_seconds == 10.0
