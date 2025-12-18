import time

import pytest

from livecap_cli.transcription_types import normalize_to_event_dict


def test_rehydrate_transcription_with_status_only():
    event = {
        "event_type": "transcription",
        "status": "ステータス更新",
        "source_id": "source1",
    }

    normalized = normalize_to_event_dict(event)
    assert normalized is not None
    assert normalized["text"] == "ステータス更新"
    assert normalized["is_final"] is False
    assert normalized["phase"] == "interim"
    assert normalized["metadata"]["status"] == "ステータス更新"


def test_rehydrate_transcription_with_metadata_text_and_timestamp(monkeypatch):
    fake_time = 1234.56
    event = {
        "event_type": "transcription",
        "source_id": "sourceA",
        "metadata": {"text": "メタテキスト", "timestamp": fake_time, "is_final": True},
    }

    normalized = normalize_to_event_dict(event)
    assert normalized is not None
    assert normalized["text"] == "メタテキスト"
    assert normalized["timestamp"] == fake_time
    assert normalized["is_final"] is True
    assert normalized["metadata"]["text"] == "メタテキスト"


def test_rehydrate_transcription_without_any_text_returns_none():
    event = {
        "event_type": "transcription",
        "source_id": "sourceB",
        "metadata": {"status": ""},
    }

    assert normalize_to_event_dict(event) is None


def test_normalize_existing_valid_event_pass_through():
    event = {
        "event_type": "transcription",
        "text": "既存テキスト",
        "source_id": "sourceC",
        "is_final": True,
        "timestamp": 10.0,
    }

    normalized = normalize_to_event_dict(event)
    assert normalized is event
