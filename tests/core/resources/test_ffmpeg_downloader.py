"""Retry policy for pinned downloads (Issue #398).

The table under test is the one settled in #395 D1. It exists because a single
``curl`` invocation cannot express it: ``--retry`` skips DNS failures,
``--retry-all-errors`` retries permanent 4xx, and ``--retry-delay`` replaces the
exponential backoff with a fixed one.
"""

from __future__ import annotations

import socket
import urllib.error
from pathlib import Path

import pytest

from livecap_cli.resources import downloader


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.invalid/x.zip", code, "", {}, None)


class TestClassification:
    @pytest.mark.parametrize("code", sorted(downloader.RETRYABLE_STATUS))
    def test_transient_statuses_retry(self, code: int) -> None:
        assert downloader.classify(_http_error(code)) == downloader.RETRY

    @pytest.mark.parametrize("code", [400, 401, 403, 404, 410, 451])
    def test_permanent_4xx_is_fatal(self, code: int) -> None:
        assert downloader.classify(_http_error(code)) == downloader.FATAL

    @pytest.mark.parametrize(
        "exc",
        [
            TimeoutError("timed out"),
            ConnectionResetError("reset"),
            socket.timeout("timed out"),
            urllib.error.URLError("Name or service not known"),
        ],
    )
    def test_transport_and_dns_retry(self, exc: BaseException) -> None:
        """DNS resolution failure is curl exit 6, which curl's own --retry skips."""
        assert downloader.classify(exc) == downloader.RETRY

    @pytest.mark.parametrize("exc", [ValueError("bad"), RuntimeError("boom")])
    def test_everything_else_is_fatal(self, exc: BaseException) -> None:
        assert downloader.classify(exc) == downloader.FATAL


class TestBackoff:
    def test_is_exponential_and_bounded(self) -> None:
        assert downloader.backoff_delays(5, base=1.0, cap=8.0) == [1.0, 2.0, 4.0, 8.0]

    def test_respects_the_cap(self) -> None:
        assert downloader.backoff_delays(6, base=1.0, cap=4.0) == [1.0, 2.0, 4.0, 4.0, 4.0]

    def test_single_attempt_never_sleeps(self) -> None:
        assert downloader.backoff_delays(1) == []


class TestDownloadWithRetry:
    def test_retries_transient_then_succeeds(self, tmp_path: Path) -> None:
        slept: list[float] = []
        calls: list[int] = []

        def fetch(url: str, destination: Path) -> None:
            calls.append(len(calls) + 1)
            if len(calls) < 3:
                raise _http_error(503)
            destination.write_bytes(b"payload")

        result = downloader.download_with_retry(
            "https://example.invalid/x.zip",
            tmp_path / "x.zip",
            fetch=fetch,
            sleep=slept.append,
        )

        assert result.read_bytes() == b"payload"
        assert len(calls) == 3
        assert slept == [1.0, 2.0]

    def test_permanent_error_is_not_retried(self, tmp_path: Path) -> None:
        calls: list[int] = []

        def fetch(url: str, destination: Path) -> None:
            calls.append(1)
            raise _http_error(404)

        with pytest.raises(urllib.error.HTTPError):
            downloader.download_with_retry(
                "https://example.invalid/x.zip",
                tmp_path / "x.zip",
                fetch=fetch,
                sleep=lambda _: None,
            )

        assert calls == [1]

    def test_partial_file_is_removed_before_the_next_attempt(self, tmp_path: Path) -> None:
        seen: list[bool] = []

        def fetch(url: str, destination: Path) -> None:
            seen.append(destination.exists())
            destination.write_bytes(b"half")
            raise _http_error(503)

        with pytest.raises(downloader.DownloadFailed):
            downloader.download_with_retry(
                "https://example.invalid/x.zip",
                tmp_path / "x.zip",
                attempts=3,
                fetch=fetch,
                sleep=lambda _: None,
            )

        assert seen == [False, False, False]
        assert not (tmp_path / "x.zip").exists()

    def test_final_error_says_what_to_do(self, tmp_path: Path) -> None:
        def fetch(url: str, destination: Path) -> None:
            raise _http_error(503)

        with pytest.raises(downloader.DownloadFailed) as excinfo:
            downloader.download_with_retry(
                "https://example.invalid/ffmpeg.zip",
                tmp_path / "x.zip",
                attempts=2,
                fetch=fetch,
                sleep=lambda _: None,
            )

        message = str(excinfo.value)
        assert "https://example.invalid/ffmpeg.zip" in message
        assert "2 attempt" in message
        assert "503" in message
        assert "LIVECAP_FFMPEG_BIN" in message

    def test_logs_each_retry(self, tmp_path: Path) -> None:
        lines: list[str] = []
        calls: list[int] = []

        def fetch(url: str, destination: Path) -> None:
            calls.append(1)
            if len(calls) < 2:
                raise _http_error(500)
            destination.write_bytes(b"ok")

        downloader.download_with_retry(
            "https://example.invalid/x.zip",
            tmp_path / "x.zip",
            fetch=fetch,
            sleep=lambda _: None,
            log=lines.append,
        )

        assert len(lines) == 1
        assert "retrying in 1s" in lines[0]
