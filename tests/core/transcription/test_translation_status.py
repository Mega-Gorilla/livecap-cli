"""`TranslationStatusEvent` の不変条件 (Issue #402 D1)。

この型は GUI まで届く公開 API なので、読む側が「recovered なのに error_type が
入っている」ような状態を想定しなくて済むよう、constructor で弾く。
"""

from __future__ import annotations

import pytest

from livecap_cli.transcription.translation_status import TranslationStatusEvent


class TestFactories:
    def test_failed_carries_the_diagnosis(self):
        event = TranslationStatusEvent.failed(
            "google", "network", "HTTP 503", recoverable=True
        )
        assert event.status == "failed"
        assert event.translator == "google"
        assert event.error_type == "network"
        assert event.message == "HTTP 503"
        assert event.recoverable is True

    def test_recovered_carries_nothing_else(self):
        event = TranslationStatusEvent.recovered("google")
        assert event.status == "recovered"
        assert event.error_type is None
        assert event.message is None
        assert event.recoverable is None

    def test_is_immutable(self):
        event = TranslationStatusEvent.recovered("google")
        with pytest.raises(Exception):
            event.status = "failed"  # type: ignore[misc]


class TestInvariants:
    """The factories are not enough: the dataclass constructor stays public."""

    def test_failed_requires_an_error_type(self):
        with pytest.raises(ValueError, match="error_type"):
            TranslationStatusEvent(translator="google", status="failed")

    def test_failed_rejects_an_unknown_error_type(self):
        with pytest.raises(ValueError, match="error_type"):
            TranslationStatusEvent(
                translator="google", status="failed", error_type="weird"  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        ("field", "value"),
        [("error_type", "network"), ("message", "x"), ("recoverable", True)],
    )
    def test_recovered_rejects_failure_fields(self, field, value):
        with pytest.raises(ValueError, match=field):
            TranslationStatusEvent(
                translator="google", status="recovered", **{field: value}
            )

    def test_rejects_unknown_status(self):
        with pytest.raises(ValueError, match="status"):
            TranslationStatusEvent(translator="google", status="broken")  # type: ignore[arg-type]

    def test_rejects_empty_translator(self):
        with pytest.raises(ValueError, match="translator"):
            TranslationStatusEvent.recovered("")
