"""Google Translate adapter (Issue #402 / #442).

Why this file is shaped the way it is: the adapter talks to an unofficial HTTP
endpoint, so almost every interesting failure is "the HTTP layer said something
unexpected". The tests therefore drive a fake transport rather than mocking a
library, which is also what makes the User-Agent — the actual root cause of
#402 — and the bot-challenge redirect — the root cause of #442 — assertable.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest
import requests

from livecap_cli.translation.exceptions import (
    TranslationError,
    TranslationNetworkError,
    UnsupportedLanguagePairError,
)
from livecap_cli.translation.impl.google import (
    BROWSER_UA,
    ENDPOINT,
    MAX_ENCODED_URL_BYTES,
    GoogleTranslator,
    _extract_translation,
)
from livecap_cli.translation.retry import FILE_RETRY_POLICY, RetryPolicy

#: Text that must never appear in a log line, an exception, or a traceback.
SECRET = "来期の人員削減について田中部長と話しました"

#: What Google's abuse gate actually returned on 2026-09-15 (#442). The
#: ``continue=`` parameter carries the whole original query, speech included.
SORRY_URL = f"https://www.google.com/sorry/index?continue=https://translate.google.com/m%3Fq%3D{SECRET}&q=EgR"
SORRY_BODY = (
    '<form id="captcha-form" action="index" method="post">'
    '<script src="https://www.google.com/recaptcha/enterprise.js"></script>'
    "Our systems have detected unusual traffic from your computer network."
)

#: The *other* shape, measured on the JSON host the same day: no redirect, no
#: CAPTCHA, a bare 429 whose body is Google's "Sorry..." page. Without the body
#: text there is nothing to tell it apart from an ordinary rate limit.
GTX_SORRY_BODY = (
    '<html><head><meta http-equiv="content-type" content="text/html; charset=utf-8"/>'
    "<title>Sorry...</title></head><body><div>We're sorry... but your computer or "
    "network may be sending automated queries. To protect our users, we can't "
    "process your request right now.</div></body></html>"
)


def _json(*sentences: str, extra: dict | None = None) -> str:
    """A reply shaped like the real ``dj=1`` one: one object per sentence,
    ``trans`` next to ``orig``, plus the debug keys Google tacks on."""
    payload = {
        "sentences": [
            {"trans": s, "orig": "x", "backend": 3, "model_specification": [{}]}
            for s in sentences
        ],
        "src": "ja",
        "spell": {},
    }
    if extra:
        payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


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


def _response(status=200, text="", url=ENDPOINT):
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
        translator, transport = _translator(_response(text=_json("Hello")))
        translator.translate("こんにちは", "ja", "en")
        assert transport.sent_headers[0]["User-Agent"] == BROWSER_UA

    def test_injected_session_also_gets_the_browser_user_agent(self):
        """The regression this class exists for."""
        translator, transport = _translator(_response(text=_json("Hello")))
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
# Parsing the gtx JSON (#442)
# ---------------------------------------------------------------------------


class TestExtraction:
    def test_single_sentence(self):
        assert _extract_translation(_json("Hello")) == "Hello"

    def test_multiple_sentences_are_concatenated(self):
        """Google splits on sentence boundaries; the caller wants one string.
        Measured shape: ``"Hello. "`` / ``"The weather is nice today.\\n"`` / ..."""
        body = _json("Hello. ", "The weather is nice today.\n", "It will rain tomorrow.")
        assert _extract_translation(body) == (
            "Hello. The weather is nice today.\nIt will rain tomorrow."
        )

    def test_newlines_inside_trans_are_preserved(self):
        assert _extract_translation(_json("Hello\nGood morning")) == "Hello\nGood morning"

    def test_entries_without_trans_are_ignored(self):
        """Asking for more ``dt`` values appends transliteration-only objects."""
        body = json.dumps(
            {"sentences": [{"trans": "Hello"}, {"translit": "konnichiwa"}], "src": "ja"}
        )
        assert _extract_translation(body) == "Hello"

    def test_non_string_trans_is_ignored(self):
        body = json.dumps({"sentences": [{"trans": None}, {"trans": "Hello"}]})
        assert _extract_translation(body) == "Hello"

    def test_no_sentences_key_returns_none(self):
        assert _extract_translation(json.dumps({"src": "ja"})) is None

    def test_empty_sentences_returns_none(self):
        assert _extract_translation(json.dumps({"sentences": []})) is None

    def test_array_form_returns_none(self):
        """``dt=t`` without ``dj=1`` yields positional arrays. We do not parse
        those on purpose: their layout shifts silently when elements are added."""
        assert _extract_translation('[[["Hello","こんにちは",null,null,10]],null,"ja"]') is None

    def test_non_json_returns_none(self):
        assert _extract_translation("<html><body>changed</body></html>") is None

    def test_extra_debug_keys_are_ignored(self):
        body = _json("Hello", extra={"confidence": 0.98, "ld_result": {"srclangs": ["ja"]}})
        assert _extract_translation(body) == "Hello"


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


class TestRequest:
    def test_sends_expected_query(self):
        translator, transport = _translator(_response(text=_json("Hello")))
        translator.translate("こんにちは", "ja", "en")

        url, params = transport.requests[0]
        assert url == "https://translate.googleapis.com/translate_a/single"
        assert params["client"] == "at"
        assert params["dt"] == "t"
        assert params["dj"] == "1"
        assert params["sl"] == "ja"
        assert params["tl"] == "en"
        assert params["q"] == "こんにちは"
        assert "hl" not in params

    def test_client_is_not_gtx(self):
        """``client=gtx`` has been answered with a 429 "Sorry" page since
        2026-09-14 (reproduced from several ISPs by eeeXun/gtt#43 and locally
        via ``requests``). Reverting to it silently re-breaks translation."""
        translator, transport = _translator(_response(text=_json("Hello")))
        translator.translate("こんにちは", "ja", "en")
        _, params = transport.requests[0]
        assert params["client"] != "gtx"

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

    @pytest.mark.parametrize("bad", ["xx", "jp", "zz", "english", ""])
    def test_unknown_language_is_rejected_before_sending(self, bad):
        """gtx answers ``tl=xx`` with HTTP 200 and the *source text* (measured
        2026-09-15). That is a silent failure, so the only place to catch a bad
        code is before the request leaves."""
        translator, transport = _translator()
        with pytest.raises(UnsupportedLanguagePairError):
            translator.translate("こんにちは", "ja", bad)
        with pytest.raises(UnsupportedLanguagePairError):
            translator.translate("Hello", bad, "ja")
        assert transport.requests == []

    @pytest.mark.parametrize("code", ["fr", "zh-TW", "zh-CN", "pt-BR", "ko"])
    def test_known_languages_are_sent(self, code):
        translator, transport = _translator(_response(text=_json("ok")))
        translator.translate("Hello", "en", code)
        assert len(transport.requests) == 1


# ---------------------------------------------------------------------------
# Context is deliberately ignored (#402 D4)
# ---------------------------------------------------------------------------


class TestContextIsIgnored:
    def test_explicit_context_is_not_sent(self):
        """Joining context with newlines makes Google translate line by line,
        which breaks a sentence that VAD split across segments."""
        translator, transport = _translator(_response(text=_json("this is a test")))
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
            _response(text=_json("ok")), default_context_sentences=0
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
        """The exact shape of #402: status says 200, the body says 500. With the
        JSON endpoint that body simply fails to parse, and the markers decide."""
        body = "<html><body><p>Error 500 (Server Error)!!1</p></body></html>"
        translator, _ = _translator(_response(status=200, text=body))
        with pytest.raises(TranslationNetworkError) as excinfo:
            translator.translate("こんにちは", "ja", "en")
        assert excinfo.value.reason == "embedded_error_page"

    def test_error_text_inside_a_real_result_is_not_an_error(self):
        """A translation may legitimately contain the words we scan for, so the
        marker is only consulted when the body was not usable JSON."""
        translator, _ = _translator(_response(text=_json("Error 500 (Server Error)")))
        result = translator.translate("エラー500", "ja", "en")
        assert result.text == "Error 500 (Server Error)"

    def test_unexpected_json_shape_is_fatal(self):
        translator, _ = _translator(_response(text=json.dumps({"foo": 1})))
        with pytest.raises(TranslationError) as excinfo:
            translator.translate("こんにちは", "ja", "en")
        assert not isinstance(excinfo.value, TranslationNetworkError)
        assert excinfo.value.reason == "layout_changed"

    def test_non_json_body_is_fatal(self):
        translator, _ = _translator(_response(text="<html><body>changed</body></html>"))
        with pytest.raises(TranslationError) as excinfo:
            translator.translate("こんにちは", "ja", "en")
        assert excinfo.value.reason == "layout_changed"

    def test_empty_result_is_fatal(self):
        translator, _ = _translator(_response(text=_json("   ")))
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
# Bot challenge is not a transient failure (#442)
# ---------------------------------------------------------------------------


class TestBotChallenge:
    """Google's abuse gate answers with a reCAPTCHA page. Retrying it does not
    help and feeds the "unusual traffic" score, so it must not be classified as
    a ``TranslationNetworkError`` — that is the type ``RetryPolicy`` re-tries."""

    def test_redirect_to_sorry_is_fatal(self):
        """The shape measured on 2026-09-15: 302 to /sorry/, final status 429."""
        translator, _ = _translator(_response(status=429, text=SORRY_BODY, url=SORRY_URL))
        with pytest.raises(TranslationError) as excinfo:
            translator.translate(SECRET, "ja", "en")
        assert not isinstance(excinfo.value, TranslationNetworkError)
        assert excinfo.value.reason == "bot_challenge"
        assert excinfo.value.status_code == 429

    def test_sorry_redirect_is_recognised_by_url_alone(self):
        """A sorry page with an empty body (or a 200) is still a sorry page."""
        translator, _ = _translator(_response(status=200, text="", url=SORRY_URL))
        with pytest.raises(TranslationError) as excinfo:
            translator.translate("こんにちは", "ja", "en")
        assert excinfo.value.reason == "bot_challenge"

    def test_direct_429_with_captcha_body_is_fatal(self):
        """If the JSON host ever serves the CAPTCHA page without redirecting."""
        translator, _ = _translator(_response(status=429, text=SORRY_BODY))
        with pytest.raises(TranslationError) as excinfo:
            translator.translate("こんにちは", "ja", "en")
        assert not isinstance(excinfo.value, TranslationNetworkError)
        assert excinfo.value.reason == "bot_challenge"

    def test_gtx_inline_sorry_page_is_fatal(self):
        """What the JSON host actually did on 2026-09-15: a bare 429 on its own
        host, no redirect, no CAPTCHA — just the "automated queries" page. The
        first marker set missed this and would have retried it three times."""
        translator, _ = _translator(_response(status=429, text=GTX_SORRY_BODY, url=ENDPOINT))
        with pytest.raises(TranslationError) as excinfo:
            translator.translate("こんにちは", "ja", "en")
        assert not isinstance(excinfo.value, TranslationNetworkError)
        assert excinfo.value.reason == "bot_challenge"

    def test_gtx_inline_sorry_page_is_not_retried(self):
        translator, transport = _translator(
            _response(status=429, text=GTX_SORRY_BODY, url=ENDPOINT),
            _response(status=429, text=GTX_SORRY_BODY, url=ENDPOINT),
            _response(status=429, text=GTX_SORRY_BODY, url=ENDPOINT),
        )
        policy = RetryPolicy(max_attempts=FILE_RETRY_POLICY.max_attempts, base_delay=0.0)
        with pytest.raises(TranslationError):
            policy.call(lambda: translator.translate("こんにちは", "ja", "en"))
        assert len(transport.requests) == 1

    def test_bare_429_stays_retryable(self):
        """A plain rate limit is still worth a retry."""
        translator, _ = _translator(_response(status=429, text="Too Many Requests"))
        with pytest.raises(TranslationNetworkError) as excinfo:
            translator.translate("こんにちは", "ja", "en")
        assert excinfo.value.reason == "http_status"

    def test_file_retry_policy_does_not_retry_a_bot_challenge(self):
        """The amplification #442 found: three hits on the sorry page per
        subtitle. Drive the real policy and count the HTTP calls."""
        translator, transport = _translator(
            _response(status=429, text=SORRY_BODY, url=SORRY_URL),
            _response(status=429, text=SORRY_BODY, url=SORRY_URL),
            _response(status=429, text=SORRY_BODY, url=SORRY_URL),
        )
        policy = RetryPolicy(max_attempts=FILE_RETRY_POLICY.max_attempts, base_delay=0.0)
        with pytest.raises(TranslationError) as excinfo:
            policy.call(lambda: translator.translate("こんにちは", "ja", "en"))
        assert excinfo.value.reason == "bot_challenge"
        assert len(transport.requests) == 1

    def test_bare_429_is_still_retried_by_the_same_policy(self):
        """Control for the test above: the policy itself still retries."""
        translator, transport = _translator(
            _response(status=429, text=""),
            _response(status=429, text=""),
            _response(text=_json("Hello")),
        )
        policy = RetryPolicy(max_attempts=3, base_delay=0.0)
        result = policy.call(lambda: translator.translate("こんにちは", "ja", "en"))
        assert result.text == "Hello"
        assert len(transport.requests) == 3

    def test_message_names_the_cause_and_the_way_out(self):
        translator, _ = _translator(_response(status=429, text=SORRY_BODY, url=SORRY_URL))
        with pytest.raises(TranslationError) as excinfo:
            translator.translate("こんにちは", "ja", "en")
        message = str(excinfo.value)
        assert "bot challenge" in message
        assert "opus_mt" in message

    def test_message_does_not_carry_the_sorry_url(self):
        """The sorry URL's ``continue=`` embeds the whole query, speech included."""
        translator, _ = _translator(_response(status=429, text=SORRY_BODY, url=SORRY_URL))
        with pytest.raises(TranslationError) as excinfo:
            translator.translate(SECRET, "ja", "en")
        message = str(excinfo.value)
        assert SECRET not in message
        assert "continue=" not in message
        assert "sorry" not in message
        assert excinfo.value.__cause__ is None


# ---------------------------------------------------------------------------
# The user's speech must not escape (#402 D8)
# ---------------------------------------------------------------------------


class TestNoSpeechLeak:
    """The text being translated travels in the GET query, so a naive
    ``raise ... from error`` puts the user's words into every traceback."""

    def _failures(self):
        long_url = f"{ENDPOINT}?client=gtx&q={SECRET}"
        return [
            requests.ConnectionError(f"Max retries exceeded with url: /translate_a/single?q={SECRET}"),
            requests.Timeout(f"timed out for url: {long_url}"),
            _response(status=503, text="", url=long_url),
            _response(status=404, text="", url=long_url),
            _response(status=200, text="<html>Error 500 (Server Error)!!1</html>"),
            _response(status=200, text="<html>changed</html>"),
            _response(status=429, text=SORRY_BODY, url=SORRY_URL),
            _response(status=429, text=GTX_SORRY_BODY, url=f"{ENDPOINT}?client=gtx&q={SECRET}"),
        ]

    @pytest.mark.parametrize("index", range(8))
    def test_exception_message_has_no_speech(self, index):
        translator, _ = _translator(self._failures()[index])
        with pytest.raises(TranslationError) as excinfo:
            translator.translate(SECRET, "ja", "en")
        assert SECRET not in str(excinfo.value)

    @pytest.mark.parametrize("index", range(8))
    def test_cause_chain_is_severed(self, index):
        """``from None``. With ``from error`` the URL survives in ``__cause__``
        and reappears whenever anyone logs with ``exc_info=True``."""
        translator, _ = _translator(self._failures()[index])
        with pytest.raises(TranslationError) as excinfo:
            translator.translate(SECRET, "ja", "en")
        assert excinfo.value.__cause__ is None

    @pytest.mark.parametrize("index", range(8))
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
        translator, transport = _translator(_response(text=_json("ok")))
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
        pool_manager.connection_from_url("https://translate.googleapis.com")
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
    """Guards the #402 / #442 failure modes themselves. Assertions stay loose
    because Google's wording changes; what matters is that *something
    translated* came back and that we were not bounced to a CAPTCHA."""

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

    def test_multiple_sentences_come_back_as_one_string(self):
        """The #442 endpoint splits sentences into separate objects."""
        translator = GoogleTranslator()
        try:
            result = translator.translate("こんにちは。今日はいい天気ですね。", "ja", "en")
        finally:
            translator.cleanup()

        assert result.text.strip()
        assert "." in result.text
        assert result.text != "こんにちは。今日はいい天気ですね。"

    def test_consecutive_translations_reuse_the_session(self):
        """Realtime subtitles are a stream, not one call: a fresh TLS handshake
        per line doubled the latency (403ms -> 191ms measured)."""
        translator = GoogleTranslator()
        try:
            for text in ["こんにちは", "今日はいい天気ですね", "配信を始めます"]:
                assert translator.translate(text, "ja", "en").text.strip()
        finally:
            translator.cleanup()
