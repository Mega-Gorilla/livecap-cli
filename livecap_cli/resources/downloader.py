"""Retrying HTTP download for pinned resources (Issue #398).

Why this is not in :mod:`livecap_cli.resources.model_manager`
-------------------------------------------------------------
``ModelManager.download_file()`` is used by every model fetch. Adding retries and
timeouts there would change the behaviour of all of them at once, which is a
separate decision from fixing the FFmpeg downloader (#398 D5). This module is
therefore deliberately narrow: one function, no cache-layout knowledge, no
awareness of what is being downloaded.

The classification table is the one settled in #395 D1 and shared, in intent,
with ``.github/actions/setup-livecap-ffmpeg/setup_ffmpeg.py``. It exists because
``curl --retry`` alone does not cover DNS failures, ``--retry-all-errors``
retries permanent 4xx, and ``--retry-delay`` disables exponential backoff.
"""

from __future__ import annotations

import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

__all__ = [
    "DownloadFailed",
    "RETRYABLE_STATUS",
    "backoff_delays",
    "classify",
    "download_with_retry",
]

#: HTTP statuses worth another attempt. Every other 4xx is permanent.
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})

MAX_ATTEMPTS = 5
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_CAP_SECONDS = 8.0
REQUEST_TIMEOUT_SECONDS = 120

USER_AGENT = "livecap-cli/1"

RETRY = "retry"
FATAL = "fatal"


class DownloadFailed(RuntimeError):
    """Every attempt failed. Carries what a user needs to act on.

    ``transient`` separates "the network or the host was unavailable" from a
    permanent answer such as 404. Callers that are willing to degrade rather
    than fail need that distinction: waiting out an outage is reasonable,
    silently accepting a supply-chain problem is not.
    """

    def __init__(
        self,
        url: str,
        attempts: int,
        last_error: BaseException,
        *,
        transient: bool = True,
    ) -> None:
        super().__init__(
            f"Failed to download after {attempts} attempt(s).\n"
            f"  url:        {url}\n"
            f"  attempts:   {attempts}\n"
            f"  last error: {last_error!r}\n"
            "Workaround: install FFmpeg yourself and point LIVECAP_FFMPEG_BIN at "
            "the directory holding ffmpeg/ffprobe."
        )
        self.url = url
        self.attempts = attempts
        self.last_error = last_error
        self.transient = transient


def classify(exc: BaseException) -> str:
    """Decide whether ``exc`` is worth another attempt."""
    # HTTPError subclasses URLError, so it has to be tested first.
    if isinstance(exc, urllib.error.HTTPError):
        return RETRY if exc.code in RETRYABLE_STATUS else FATAL
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return RETRY
    if isinstance(exc, urllib.error.URLError):
        # DNS resolution and transport failures land here.
        return RETRY
    return FATAL


def backoff_delays(
    attempts: int = MAX_ATTEMPTS,
    base: float = BACKOFF_BASE_SECONDS,
    cap: float = BACKOFF_CAP_SECONDS,
) -> list[float]:
    """Exponential and bounded: 1, 2, 4, 8 seconds for five attempts."""
    return [min(base * (2**index), cap) for index in range(max(attempts - 1, 0))]


def _fetch(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        with open(destination, "wb") as handle:
            shutil.copyfileobj(response, handle, 1 << 20)


def download_with_retry(
    url: str,
    destination: Path,
    *,
    attempts: int = MAX_ATTEMPTS,
    fetch: Callable[[str, Path], None] = _fetch,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] | None = None,
) -> Path:
    """Download ``url`` to ``destination``, retrying only transient failures.

    One attempt performs exactly one request. The loop lives here rather than in
    the transport so the retry policy is stated in a single place.
    """
    delays = backoff_delays(attempts)
    last_error: BaseException | None = None

    for attempt in range(1, attempts + 1):
        try:
            fetch(url, destination)
            return destination
        except BaseException as exc:  # noqa: BLE001 - re-raised or wrapped below
            last_error = exc
            # Never leave a partial file for the next attempt to mistake for a
            # complete one.
            destination.unlink(missing_ok=True)

            if classify(exc) == FATAL:
                raise
            if attempt == attempts:
                break
            delay = delays[attempt - 1]
            if log is not None:
                log(f"attempt {attempt}/{attempts} failed ({exc!r}); retrying in {delay:g}s")
            sleep(delay)

    assert last_error is not None
    raise DownloadFailed(url, attempts, last_error)
