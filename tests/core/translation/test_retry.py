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


class TestDeadlineAdmissionControl:
    """`total_timeout_seconds` must gate when work *starts*, not just sleeps.

    Checking the budget only before sleeping let a slow attempt start with 0.1s
    left and run for seconds: a 10s policy measured 10.8s across two calls.

    The estimate is not a guarantee - an in-flight attempt cannot be stopped from
    here - so these assert on admission behaviour, not on a hard ceiling.
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
            estimated_attempt_seconds=5.0,
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
            estimated_attempt_seconds=5.0,
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
            estimated_attempt_seconds=3.0,
        )
        with pytest.raises(TranslationNetworkError):
            policy.call(fail, sleep=clock.sleep, monotonic=clock.monotonic)

        assert clock.now <= 4.0
        assert len(calls) < 5

    def test_rejects_non_positive_attempt_timeout(self):
        with pytest.raises(ValueError, match="estimated_attempt_seconds"):
            RetryPolicy(max_attempts=3, estimated_attempt_seconds=0)


class TestPerTranslatorBudget:
    """The per-attempt budget must come from the translator, not a constant.

    ``FILE_RETRY_POLICY`` is applied to any ``BaseTranslator``; a local model
    cannot promise a bound at all, and even the Google adapter's timeout is
    configurable. Declaring one fixed number would be a claim we cannot keep.
    """

    def test_base_translator_estimates_nothing(self):
        """The default is "cannot estimate", so a local model opts out."""
        from livecap_cli.translation.base import BaseTranslator

        assert BaseTranslator.estimated_attempt_seconds.fget(object()) is None

    def test_file_policy_declares_nothing_by_itself(self):
        assert FILE_RETRY_POLICY.estimated_attempt_seconds is None

    def test_translator_without_a_declaration_stays_soft(self):
        assert for_translator(FILE_RETRY_POLICY, object()) is FILE_RETRY_POLICY

    def test_estimate_is_taken_from_the_translator(self):
        class Declaring:
            estimated_attempt_seconds = 4.0

        policy = for_translator(FILE_RETRY_POLICY, Declaring())
        assert policy.estimated_attempt_seconds == 4.0
        assert policy.total_timeout_seconds == FILE_RETRY_POLICY.total_timeout_seconds

    def test_google_estimates_from_its_http_timeout(self):
        from livecap_cli.translation.impl.google import GoogleTranslator

        translator = GoogleTranslator(timeout=(1.5, 2.5))
        try:
            assert translator.estimated_attempt_seconds == 4.0
        finally:
            translator.cleanup()

    def test_google_policy_fits_retries_inside_the_deadline(self):
        """The deadline stays at the documented 10s and retry still happens."""
        from livecap_cli.translation.impl.google import GoogleTranslator

        translator = GoogleTranslator()
        try:
            policy = for_translator(FILE_RETRY_POLICY, translator)
        finally:
            translator.cleanup()

        clock = FakeClock()
        calls: list[int] = []

        def fail():
            calls.append(1)
            clock.now += policy.estimated_attempt_seconds
            raise TranslationNetworkError("503", provider="google")

        with pytest.raises(TranslationNetworkError):
            policy.call(fail, sleep=clock.sleep, monotonic=clock.monotonic)

        assert clock.now <= policy.total_timeout_seconds
        assert len(calls) >= 2, "a single attempt would mean retry never happens"
