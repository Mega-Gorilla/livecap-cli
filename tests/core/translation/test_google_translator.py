"""Google Translate adapter (Issue #402).

Why this file is shaped the way it is: the adapter scrapes a web page, so almost
every interesting failure is "the HTTP layer said something unexpected". The
tests therefore drive a fake transport rather than mocking a library, which is
also what makes the User-Agent — the actual root cause of #402 — assertable.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
import requests

from livecap_cli.translation.exceptions import (
    TranslationError,
    TranslationNetworkError,
    UnsupportedLanguagePairError,
)
from livecap_cli.translation.impl.google import (
    BROWSER_UA,
    MAX_ENCODED_URL_BYTES,
    GoogleTranslator,
    _extract_result,
)

#: Text that must never appear in a log line, an exception, or a traceback.
SECRET = "来期の人員削減について田中部長と話しました"


def _page(inner: str) -> str:
    """A minimal reply shaped like the real one, links block included.

    The trailing ``links-container`` is not decoration: an earlier version of the
    parser stopped tracking depth at zero instead of latching, so the footer got
    appended to every translation.
    """
    return (
        "<!DOCTYPE html><html><head><style>.result-container { color: #fff; }"
        "</style></head><body>"
        '<div class="header">Translate</div>'
        f'<div class="result-container">{inner}</div>'
        '<div class="links-container"><ul><li><a href="#">Google home</a></li>'
        "<li><a href=\"#\">Switch to full site</a></li></ul></div>"
        "</body></html>"
    )


class FakeTransport:
    """Stands in for ``requests.Session``. Records what was actually sent."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.requests: list[tuple[str, dict]] = []
        self.sent_headers: list[dict] = []
        self.headers: dict[str, str] = {}
        self.closed = False

    def get(self, url, params=None, timeout=None, headers=None):
        self.requests.append((url, dict(params or {})))
        self.sent_headers.append(dict(headers or {}))
        reply = self._responses.pop(0) if self._responses else self._responses_exhausted()
        if isinstance(reply, Exception):
            raise reply
        return reply

    def _responses_exhausted(self):
        raise AssertionError("transport called more times than the test allowed")

    def close(self):
        self.closed = True


def _response(status=200, text="", url="https://translate.google.com/m"):
    return SimpleNamespace(status_code=status, text=text, url=url)


def _translator(*responses, **kwargs) -> tuple[GoogleTranslator, FakeTransport]:
    transport = FakeTransport(*responses)
    return GoogleTranslator(transport=transport, **kwargs), transport


# ---------------------------------------------------------------------------
# The root cause of #402
# ---------------------------------------------------------------------------


class TestUserAgent:
    """Assert on what is *sent*, not on session attributes.

    Setting the header on the session only covered the session the adapter built
    itself; an injected one kept ``python-requests/2.x`` and walked straight back
    into #402.
    """

    def test_user_agent_is_sent_with_the_request(self):
        translator, transport = _translator(_response(text=_page("Hello")))
        translator.translate("こんにちは", "ja", "en")
        assert transport.sent_headers[0]["User-Agent"] == BROWSER_UA

    def test_injected_session_also_gets_the_browser_user_agent(self):
        """The regression this class exists for."""
        translator, transport = _translator(_response(text=_page("Hello")))
        translator.translate("こんにちは", "ja", "en")
        sent = transport.sent_headers[0]["User-Agent"]
        assert "python-requests" not in sent
        assert sent == BROWSER_UA

    def test_injected_session_headers_are_not_mutated(self):
        """We do not own it, so we must not change it permanently."""
        session = requests.Session()
        original = session.headers.get("User-Agent")
        translator = GoogleTranslator(transport=session)
        translator.cleanup()
        assert session.headers.get("User-Agent") == original

    def test_user_agent_looks_like_a_real_browser(self):
        assert BROWSER_UA.startswith("Mozilla/5.0")
        assert "Chrome/" in BROWSER_UA
        assert "python-requests" not in BROWSER_UA


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestExtraction:
    def test_plain_text(self):
        assert _extract_result(_page("Hello")) == "Hello"

    def test_does_not_swallow_the_trailing_links_block(self):
        assert _extract_result(_page("Hello")) == "Hello"

    def test_entities_are_unescaped(self):
        html = _page("&quot;Test&quot; of A and B&lt;C&gt;")
        assert _extract_result(html) == '"Test" of A and B<C>'

    def test_newlines_are_preserved(self):
        assert _extract_result(_page("Hello\nGood morning")) == "Hello\nGood morning"

    def test_br_becomes_a_newline(self):
        assert _extract_result(_page("Hello<br>Good morning")) == "Hello\nGood morning"
        assert _extract_result(_page("Hello<br/>Good morning")) == "Hello\nGood morning"

    def test_nested_elements_are_flattened(self):
        html = _page("<span>Hello</span> <b>there</b>")
        assert _extract_result(html) == "Hello there"

    def test_nested_div_does_not_end_capture_early(self):
        html = _page("<div><span>Hello</span></div> there")
        assert _extract_result(html) == "Hello there"

    def test_css_mention_alone_is_not_a_result(self):
        html = "<html><head><style>.result-container { color: red; }</style></head><body></body></html>"
        assert _extract_result(html) is None

    def test_missing_element_returns_none(self):
        assert _extract_result("<html><body><div>nope</div></body></html>") is None


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


class TestRequest:
    def test_sends_expected_query(self):
        translator, transport = _translator(_response(text=_page("Hello")))
        translator.translate("こんにちは", "ja", "en")

        url, params = transport.requests[0]
        assert url == "https://translate.google.com/m"
        assert params["sl"] == "ja"
        assert params["tl"] == "en"
        assert params["q"] == "こんにちは"

    def test_makes_exactly_one_attempt(self):
        """Retry belongs to the caller now (#402 D10)."""
        translator, transport = _translator(_response(status=503, text=""))
        with pytest.raises(TranslationNetworkError):
            translator.translate("こんにちは", "ja", "en")
        assert len(transport.requests) == 1

    def test_empty_input_short_circuits(self):
        translator, transport = _translator()
        result = translator.translate("   ", "ja", "en")
        assert result.text == ""
        assert transport.requests == []

    def test_same_language_is_rejected(self):
        translator, transport = _translator()
        with pytest.raises(UnsupportedLanguagePairError):
            translator.translate("こんにちは", "ja", "ja")
        assert transport.requests == []


# ---------------------------------------------------------------------------
# Context is deliberately ignored (#402 D4)
# ---------------------------------------------------------------------------


class TestContextIsIgnored:
    def test_explicit_context_is_not_sent(self):
        """Joining context with newlines makes Google translate line by line,
        which breaks a sentence that VAD split across segments."""
        translator, transport = _translator(_response(text=_page("this is a test")))
        translator.translate(
            "これはテストです", "ja", "en", context=["過去の文1", "過去の文2"]
        )

        _, params = transport.requests[0]
        assert params["q"] == "これはテストです"
        assert "過去の文1" not in params["q"]
        assert "\n" not in params["q"]

    def test_default_context_sentences_is_zero(self):
        translator, _ = _translator()
        assert translator.default_context_sentences == 0

    def test_context_does_not_leak_through_the_zero_slice(self):
        """``context[-0:]`` is ``context[:]`` — the whole history. Setting the
        count to zero without ignoring context would have made things worse."""
        translator, transport = _translator(
            _response(text=_page("ok")), default_context_sentences=0
        )
        translator.translate("今", "ja", "en", context=["古い1", "古い2", "古い3"])
        _, params = transport.requests[0]
        assert params["q"] == "今"


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestClassification:
    @pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
    def test_transient_statuses_are_network_errors(self, status):
        translator, _ = _translator(_response(status=status))
        with pytest.raises(TranslationNetworkError) as excinfo:
            translator.translate("こんにちは", "ja", "en")
        assert excinfo.value.status_code == status
        assert excinfo.value.reason == "http_status"

    @pytest.mark.parametrize("status", [400, 403, 404, 451])
    def test_permanent_statuses_are_fatal(self, status):
        translator, _ = _translator(_response(status=status))
        with pytest.raises(TranslationError) as excinfo:
            translator.translate("こんにちは", "ja", "en")
        assert not isinstance(excinfo.value, TranslationNetworkError)
        assert excinfo.value.status_code == status

    def test_embedded_error_page_with_http_200_is_retryable(self):
        """The exact shape of #402: status says 200, the body says 500."""
        body = "<html><body><p>Error 500 (Server Error)!!1</p></body></html>"
        translator, _ = _translator(_response(status=200, text=body))
        with pytest.raises(TranslationNetworkError) as excinfo:
            translator.translate("こんにちは", "ja", "en")
        assert excinfo.value.reason == "embedded_error_page"

    def test_error_text_inside_a_real_result_is_not_an_error(self):
        """A translation may legitimately contain the words we scan for, so the
        marker is only consulted when no result element was found."""
        translator, _ = _translator(_response(text=_page("Error 500 (Server Error)")))
        result = translator.translate("エラー500", "ja", "en")
        assert result.text == "Error 500 (Server Error)"

    def test_missing_result_element_is_fatal(self):
        translator, _ = _translator(_response(text="<html><body>changed</body></html>"))
        with pytest.raises(TranslationError) as excinfo:
            translator.translate("こんにちは", "ja", "en")
        assert not isinstance(excinfo.value, TranslationNetworkError)
        assert excinfo.value.reason == "layout_changed"

    def test_empty_result_is_fatal(self):
        translator, _ = _translator(_response(text=_page("   ")))
        with pytest.raises(TranslationError) as excinfo:
            translator.translate("こんにちは", "ja", "en")
        assert excinfo.value.reason == "empty_result"

    def test_timeout_is_a_network_error(self):
        translator, _ = _translator(requests.Timeout("timed out"))
        with pytest.raises(TranslationNetworkError) as excinfo:
            translator.translate("こんにちは", "ja", "en")
        assert excinfo.value.reason == "timeout"

    def test_transport_failure_is_a_network_error(self):
        translator, _ = _translator(requests.ConnectionError("dns failure"))
        with pytest.raises(TranslationNetworkError) as excinfo:
            translator.translate("こんにちは", "ja", "en")
        assert excinfo.value.reason == "transport"


# ---------------------------------------------------------------------------
# The user's speech must not escape (#402 D8)
# ---------------------------------------------------------------------------


class TestNoSpeechLeak:
    """The text being translated travels in the GET query, so a naive
    ``raise ... from error`` puts the user's words into every traceback."""

    def _failures(self):
        long_url = f"https://translate.google.com/m?q={SECRET}"
        return [
            requests.ConnectionError(f"Max retries exceeded with url: /m?q={SECRET}"),
            requests.Timeout(f"timed out for url: {long_url}"),
            _response(status=503, text="", url=long_url),
            _response(status=404, text="", url=long_url),
            _response(status=200, text="<html>Error 500 (Server Error)!!1</html>"),
            _response(status=200, text="<html>changed</html>"),
        ]

    @pytest.mark.parametrize("index", range(6))
    def test_exception_message_has_no_speech(self, index):
        translator, _ = _translator(self._failures()[index])
        with pytest.raises(TranslationError) as excinfo:
            translator.translate(SECRET, "ja", "en")
        assert SECRET not in str(excinfo.value)

    @pytest.mark.parametrize("index", range(6))
    def test_cause_chain_is_severed(self, index):
        """``from None``. With ``from error`` the URL survives in ``__cause__``
        and reappears whenever anyone logs with ``exc_info=True``."""
        translator, _ = _translator(self._failures()[index])
        with pytest.raises(TranslationError) as excinfo:
            translator.translate(SECRET, "ja", "en")
        assert excinfo.value.__cause__ is None

    @pytest.mark.parametrize("index", range(6))
    def test_nothing_leaks_through_exc_info_logging(self, index, caplog):
        """The end-to-end property: a caller may log however they like."""
        translator, _ = _translator(self._failures()[index])
        with caplog.at_level(logging.WARNING):
            try:
                translator.translate(SECRET, "ja", "en")
            except TranslationError as exc:
                logging.getLogger("test").warning(
                    "Translation failed: %s", exc, exc_info=True
                )
        assert SECRET not in caplog.text

    def test_too_long_error_reports_size_not_content(self):
        translator, transport = _translator()
        with pytest.raises(TranslationError) as excinfo:
            translator.translate(SECRET * 400, "ja", "en")
        message = str(excinfo.value)
        assert SECRET not in message
        assert "bytes" in message
        assert transport.requests == []


# ---------------------------------------------------------------------------
# URL length (#402 D10)
# ---------------------------------------------------------------------------


class TestUrlLength:
    @pytest.mark.parametrize(
        ("label", "char"),
        [("ascii", "a"), ("japanese", "あ"), ("emoji", "😀")],
    )
    def test_over_limit_is_rejected_before_sending(self, label, char):
        """Character count is a bad proxy: one character costs 1.0-12.0 bytes
        once percent-encoded, so the check has to measure the encoded URL."""
        translator, transport = _translator()
        with pytest.raises(TranslationError) as excinfo:
            translator.translate(char * MAX_ENCODED_URL_BYTES, "ja", "en")
        assert excinfo.value.reason == "request_too_long"
        assert transport.requests == []

    def test_normal_length_is_sent(self):
        translator, transport = _translator(_response(text=_page("ok")))
        translator.translate("あ" * 100, "ja", "en")
        assert len(transport.requests) == 1


# ---------------------------------------------------------------------------
# Session ownership (#402 D9)
# ---------------------------------------------------------------------------


class TestSessionOwnership:
    def test_own_session_is_closed_by_cleanup(self):
        """Assert on the connection pools, not on ``session.adapters``.

        ``Session.close()`` leaves ``adapters`` populated - it closes each
        adapter instead - so checking that dict proves nothing. What actually
        changes is the pool manager, which is what leaks if cleanup is skipped.
        """
        translator = GoogleTranslator()
        session = translator._session
        pool_manager = session.adapters["https://"].poolmanager
        # Populate a pool without touching the network.
        pool_manager.connection_from_url("https://translate.google.com")
        assert len(pool_manager.pools.keys()) == 1

        translator.cleanup()

        assert len(pool_manager.pools.keys()) == 0

    def test_injected_transport_is_not_closed(self):
        """Whoever passed it in owns it."""
        translator, transport = _translator()
        translator.cleanup()
        assert transport.closed is False

    def test_cleanup_is_idempotent(self):
        translator = GoogleTranslator()
        translator.cleanup()
        translator.cleanup()


# ---------------------------------------------------------------------------
# Opt-in: the real endpoint
# ---------------------------------------------------------------------------


@pytest.mark.network
class TestGoogleTranslatorNetwork:
    """Guards the #402 failure mode itself. Assertions stay loose because
    Google's wording changes; what matters is that *something translated*
    came back."""

    def test_translate_ja_to_en(self):
        translator = GoogleTranslator()
        try:
            result = translator.translate("こんにちは", "ja", "en")
        finally:
            translator.cleanup()

        assert result.text.strip()
        assert result.text != "こんにちは"
        assert any(char.isascii() and char.isalpha() for char in result.text)
        assert result.original_text == "こんにちは"

    def test_translate_en_to_ja(self):
        translator = GoogleTranslator()
        try:
            result = translator.translate("Hello", "en", "ja")
        finally:
            translator.cleanup()

        assert result.text.strip()
        assert result.text != "Hello"

    def test_consecutive_translations_reuse_the_session(self):
        """Realtime subtitles are a stream, not one call: a fresh TLS handshake
        per line doubled the latency (403ms -> 191ms measured)."""
        translator = GoogleTranslator()
        try:
            for text in ["こんにちは", "今日はいい天気ですね", "配信を始めます"]:
                assert translator.translate(text, "ja", "en").text.strip()
        finally:
            translator.cleanup()
