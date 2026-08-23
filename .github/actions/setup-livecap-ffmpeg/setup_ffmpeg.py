#!/usr/bin/env python3
"""Pinned, checksum-verified FFmpeg setup for CI (Issue #395).

Why this is Python and not two shell scripts
--------------------------------------------
The action used to carry a bash implementation and a PowerShell implementation
side by side. Issue #395 requires that **both OSes retry on the same
conditions**; with two implementations that is a hope, not a guarantee. Keeping
the classification table, the backoff and the verification in one place makes it
structurally true — and testable (``tests/ci/test_setup_ffmpeg.py``).

The direct motivation is #398: the runtime downloader built asset URLs that
404 on *every* platform and nobody noticed, because that code path had no test.

Constraints
-----------
* **Standard library only.** This runs before ``uv sync``, so no third-party
  package is importable.
* **No network access at import time.** The module must be importable by pytest.

Verification is an invariant, not a step
----------------------------------------
Bytes reach ``--bin-dir`` from three different places: an ``actions/cache``
restore, a self-hosted persistent directory, or a fresh download. All three go
through :func:`verify_binaries`. That is what stops a stale binary from living
forever in ``C:\\LiveCap\\Cache\\ffmpeg-bin``, which no cache key can reach.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

MANIFEST_NAME = "ffmpeg-manifest.json"

#: HTTP statuses worth retrying. Everything else in 4xx is permanent.
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})

MAX_ATTEMPTS = 5
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_CAP_SECONDS = 8.0
REQUEST_TIMEOUT_SECONDS = 120

USER_AGENT = "livecap-cli-ci-setup-ffmpeg/1"

RETRY = "retry"
FATAL = "fatal"

_VERSION_LINE = re.compile(r"^\w+ version (\S+)")


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class SetupError(RuntimeError):
    """Fail-loud condition. Never retried."""


class ChecksumMismatch(SetupError):
    def __init__(self, path: Path, expected: str, actual: str) -> None:
        super().__init__(
            f"SHA-256 mismatch for {path.name}\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            "The archive is corrupt or has been replaced. Not retrying."
        )
        self.expected = expected
        self.actual = actual


class ArchiveContentError(SetupError):
    """The archive downloaded fine but does not contain what we expected."""


class DownloadFailed(SetupError):
    def __init__(self, url: str, attempts: int, last_error: BaseException) -> None:
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


# --------------------------------------------------------------------------
# Manifest / platform
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> tuple[dict, str]:
    """Return ``(manifest, sha256_of_manifest_file)``.

    The digest is computed here rather than with the workflow's ``hashFiles()``
    so the cache key does not depend on how expressions behave inside a
    composite action.
    """
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return manifest, sha256_file(path)


def resolve_platform_key(manifest: dict, override: str | None = None) -> str:
    """Map the runner to a manifest platform key, or fail loud."""
    supported = sorted(manifest["platforms"])
    if override:
        if override not in manifest["platforms"]:
            raise SetupError(
                f"Unknown platform override {override!r}. Supported: {', '.join(supported)}"
            )
        return override

    system = os.environ.get("RUNNER_OS") or platform.system()
    arch = os.environ.get("RUNNER_ARCH")
    if not arch:
        machine = platform.machine().lower()
        if machine in ("amd64", "x86_64", "x64"):
            arch = "X64"
        elif machine in ("arm64", "aarch64"):
            arch = "ARM64"
        else:
            arch = machine.upper()

    key = f"{system}-{arch}"
    if key not in manifest["platforms"]:
        raise SetupError(
            f"No pinned FFmpeg for {key!r}. Supported: {', '.join(supported)}.\n"
            "Add the platform to ffmpeg-manifest.json (see its SHA-256 refresh notes)."
        )
    return key


def compute_cache_key(manifest: dict, manifest_sha: str, platform_key: str) -> str:
    """Exact key only — the action never uses ``restore-keys``.

    A broad ``restore-keys`` would resurrect a cache built for an older pinned
    version, silently defeating the pinning (Issue #395, finding 6).
    """
    return (
        f"ffmpeg-{manifest['version']}-{platform_key}"
        f"-{manifest_sha[:16]}-g{manifest['cache_generation']}"
    )


# --------------------------------------------------------------------------
# Download: classification, backoff, retry
# --------------------------------------------------------------------------


def classify(exc: BaseException) -> str:
    """Decide whether ``exc`` is worth another attempt.

    This is the single source of truth referenced by Issue #395 D1. It replaces
    ``curl --retry-all-errors`` (which retries permanent 4xx) and
    ``--retry-delay`` (which disables exponential backoff).
    """
    # HTTPError is a subclass of URLError, so it has to be checked first.
    if isinstance(exc, urllib.error.HTTPError):
        return RETRY if exc.code in RETRYABLE_STATUS else FATAL
    if isinstance(exc, SetupError):
        # Checksum / archive-content failures are supply-chain problems.
        return FATAL
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return RETRY
    if isinstance(exc, urllib.error.URLError):
        # DNS resolution and transport failures land here. curl's built-in
        # --retry does *not* cover these, which is one reason we do our own loop.
        return RETRY
    return FATAL


def backoff_delays(
    attempts: int = MAX_ATTEMPTS,
    base: float = BACKOFF_BASE_SECONDS,
    cap: float = BACKOFF_CAP_SECONDS,
) -> list[float]:
    """Exponential and bounded: 1, 2, 4, 8 seconds for 5 attempts."""
    return [min(base * (2**index), cap) for index in range(max(attempts - 1, 0))]


def _fetch(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        with open(dest, "wb") as handle:
            shutil.copyfileobj(response, handle, 1 << 20)


def download_with_retry(
    url: str,
    dest: Path,
    *,
    attempts: int = MAX_ATTEMPTS,
    fetch=_fetch,
    sleep=time.sleep,
    log=print,
) -> Path:
    """Download ``url`` to ``dest``, retrying only transient failures.

    One attempt performs exactly one request; the loop lives here so retry
    policy is not split between this function and the transport.
    """
    delays = backoff_delays(attempts)
    last_error: BaseException | None = None

    for attempt in range(1, attempts + 1):
        try:
            fetch(url, dest)
            return dest
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            last_error = exc
            # Never leave a partial archive behind for the next attempt to
            # mistake for a complete one.
            dest.unlink(missing_ok=True)

            if classify(exc) == FATAL:
                raise
            if attempt == attempts:
                break
            delay = delays[attempt - 1]
            log(f"  attempt {attempt}/{attempts} failed ({exc!r}); retrying in {delay:g}s")
            sleep(delay)

    assert last_error is not None
    raise DownloadFailed(url, attempts, last_error)


def verify_archive(path: Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ChecksumMismatch(path, expected_sha256, actual)


def extract_binary(archive: Path, member_name: str, destination: Path) -> None:
    """Pull ``member_name`` out of ``archive``, naming what is missing on failure."""
    with zipfile.ZipFile(archive) as bundle:
        match = next(
            (
                info
                for info in bundle.infolist()
                if not info.is_dir() and Path(info.filename).name == member_name
            ),
            None,
        )
        if match is None:
            listing = ", ".join(sorted(info.filename for info in bundle.infolist())[:20])
            raise ArchiveContentError(
                f"{archive.name} does not contain {member_name!r}.\n"
                f"  archive members: {listing or '(empty)'}\n"
                "The upstream layout changed; update ffmpeg-manifest.json."
            )
        with bundle.open(match) as source, open(destination, "wb") as target:
            shutil.copyfileobj(source, target, 1 << 20)

    if os.name != "nt":
        destination.chmod(destination.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


# --------------------------------------------------------------------------
# Verification (independent of where the bytes came from)
# --------------------------------------------------------------------------


def probe_version(executable: Path) -> tuple[int, str, str]:
    """Return ``(returncode, first_line, configuration_line)``."""
    try:
        completed = subprocess.run(
            [str(executable), "-version"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except OSError as exc:
        return 126, f"(could not execute: {exc})", ""

    lines = (completed.stdout or completed.stderr or "").splitlines()
    first = lines[0].strip() if lines else ""
    configuration = next(
        (line.split(":", 1)[1].strip() for line in lines if line.startswith("configuration:")),
        "",
    )
    return completed.returncode, first, configuration


def version_matches(first_line: str, expected: str) -> bool:
    """``ffmpeg version 6.1-full_build-...`` matches ``6.1``; ``6.1.1`` does not."""
    match = _VERSION_LINE.match(first_line)
    if not match:
        return False
    token = match.group(1)
    return token == expected or token.startswith(f"{expected}-")


def verify_binaries(bin_dir: Path, spec: dict, version: str) -> tuple[bool, list[str], dict]:
    """Check every pinned binary. Returns ``(ok, reasons, details)``.

    Reasons name the specific failure so a red job says *what* was wrong rather
    than only that something was.
    """
    reasons: list[str] = []
    details: dict[str, dict] = {}

    for role, entry in spec["binaries"].items():
        path = bin_dir / entry["name"]
        info: dict[str, str] = {"path": str(path)}
        details[role] = info

        if not path.is_file():
            reasons.append(f"{role}: missing at {path}")
            continue

        actual = sha256_file(path)
        info["sha256"] = actual
        if actual != entry["sha256"]:
            reasons.append(
                f"{role}: SHA-256 mismatch (expected {entry['sha256'][:16]}..., got {actual[:16]}...)"
            )
            continue

        returncode, first_line, configuration = probe_version(path)
        info["version_line"] = first_line
        info["configuration"] = configuration
        if returncode != 0:
            reasons.append(f"{role}: not runnable (exit {returncode}): {first_line}")
            continue
        if not version_matches(first_line, version):
            reasons.append(f"{role}: expected version {version}, reported {first_line!r}")

    return not reasons, reasons, details


# --------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------


def install(manifest: dict, spec: dict, bin_dir: Path, *, log=print) -> dict:
    """Download, verify and extract both binaries. Returns archive digests."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}

    with tempfile.TemporaryDirectory(prefix="ffmpeg-install-") as work:
        work_dir = Path(work)
        for role, entry in spec["archives"].items():
            url = f"{manifest['base_url']}/{entry['asset']}"
            archive = work_dir / entry["asset"]

            log(f"Downloading {entry['asset']}")
            download_with_retry(url, archive, log=log)

            verify_archive(archive, entry["sha256"])
            digests[role] = entry["sha256"]
            log(f"  sha256 verified: {entry['sha256']}")

            target = bin_dir / spec["binaries"][role]["name"]
            extract_binary(archive, spec["binaries"][role]["name"], target)
            log(f"  installed: {target}")

    return digests


# --------------------------------------------------------------------------
# GitHub Actions plumbing
# --------------------------------------------------------------------------


def pinned_provenance(manifest: dict, spec: dict) -> list[tuple[str, str]]:
    """Where the bytes are supposed to come from, and what they must hash to.

    Derived from the manifest alone, so it is available on every run. The
    downloaded digests are not: on a cache hit nothing is fetched, yet Issue
    #395 still wants the origin and the expected checksums recorded.
    """
    rows: list[tuple[str, str]] = []
    for role, entry in spec["archives"].items():
        rows.append((f"{role} archive url", f"{manifest['base_url']}/{entry['asset']}"))
        rows.append((f"{role} archive sha256 (expected)", entry["sha256"]))
        rows.append((f"{role} binary sha256 (expected)", spec["binaries"][role]["sha256"]))
    return rows


def emit_outputs(values: dict[str, str]) -> None:
    target = os.environ.get("GITHUB_OUTPUT")
    lines = [f"{key}={value}" for key, value in values.items()]
    if target:
        with open(target, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    else:
        print("\n".join(lines))


def write_summary(rows: Iterable[tuple[str, str]], title: str) -> None:
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if not target:
        return
    body = [f"### {title}", "", "| item | value |", "| --- | --- |"]
    for name, value in rows:
        # Escaped outside the f-string: backslashes in f-string expressions are
        # a syntax error before Python 3.12, and this runs on 3.10 runners.
        cell = str(value).replace("|", "\\|")
        body.append(f"| {name} | {cell} |")
    with open(target, "a", encoding="utf-8") as handle:
        handle.write("\n".join(body) + "\n\n")


def warn(message: str) -> None:
    print(f"::warning::{message}")


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def _manifest_path(args: argparse.Namespace) -> Path:
    return Path(args.manifest) if args.manifest else Path(__file__).with_name(MANIFEST_NAME)


def cmd_plan(args: argparse.Namespace) -> int:
    manifest, manifest_sha = load_manifest(_manifest_path(args))
    platform_key = resolve_platform_key(manifest, args.platform)
    bin_dir = Path(args.bin_dir).resolve()

    emit_outputs(
        {
            "platform": platform_key,
            "version": manifest["version"],
            "cache-key": compute_cache_key(manifest, manifest_sha, platform_key),
            "cache-generation": str(manifest["cache_generation"]),
            "manifest-sha256": manifest_sha,
            "bin-dir": str(bin_dir),
        }
    )
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    manifest, _ = load_manifest(_manifest_path(args))
    platform_key = resolve_platform_key(manifest, args.platform)
    spec = manifest["platforms"][platform_key]
    version = manifest["version"]
    bin_dir = Path(args.bin_dir).resolve()
    bin_dir.mkdir(parents=True, exist_ok=True)
    cache_hit = args.cache_hit == "true"
    # "miss" and "not asked for" are different diagnoses. Self-hosted runners
    # pass cache: 'false' because their bin dir lives outside the workspace.
    cache_state = "hit" if cache_hit else "miss"
    if args.cache != "true":
        cache_state = "disabled"

    # Printed whether or not anything is downloaded this run.
    provenance = pinned_provenance(manifest, spec)
    for name, value in provenance:
        print(f"  {name}: {value}")

    ok, reasons, details = verify_binaries(bin_dir, spec, version)
    poisoned = False
    digests: dict[str, str] = {}

    if ok:
        source = "cache" if cache_hit else "existing directory"
        print(f"FFmpeg {version} already present and verified ({source}). Skipping download.")
    else:
        for reason in reasons:
            print(f"  needs install - {reason}")

        if cache_hit:
            # actions/cache entries are immutable: deleting this one locally
            # would not replace the remote entry, so every later run would
            # restore the same bad bytes. Re-fetch, keep the job green, and
            # tell a human to bump the generation.
            poisoned = True
            warn(
                "Poisoned FFmpeg cache: an exact cache hit failed verification "
                f"({'; '.join(reasons)}). Re-downloading and skipping cache save. "
                "Bump 'cache_generation' in "
                ".github/actions/setup-livecap-ffmpeg/ffmpeg-manifest.json."
            )

        digests = install(manifest, spec, bin_dir)
        source = "download"

        ok, reasons, details = verify_binaries(bin_dir, spec, version)
        if not ok:
            raise SetupError(
                "FFmpeg verification failed after a fresh install:\n  "
                + "\n  ".join(reasons)
            )

    ffmpeg = details["ffmpeg"]
    ffprobe = details["ffprobe"]

    print(f"ffmpeg:  {ffmpeg['path']}")
    print(f"         {ffmpeg.get('version_line', '')}")
    print(f"ffprobe: {ffprobe['path']}")
    print(f"         {ffprobe.get('version_line', '')}")
    print(f"configuration: {ffmpeg.get('configuration', '')}")

    write_summary(
        [
            ("platform", platform_key),
            ("version", version),
            ("source", source),
            ("cache", cache_state),
            ("poisoned cache", "yes (save skipped)" if poisoned else "no"),
            ("ffmpeg", ffmpeg.get("version_line", "")),
            ("ffprobe", ffprobe.get("version_line", "")),
            ("ffmpeg path", ffmpeg["path"]),
            ("ffprobe path", ffprobe["path"]),
            ("build configuration", ffmpeg.get("configuration", "")),
            *provenance,
            # Pinning makes runs comparable. It does not make the Linux and
            # Windows builds behave identically - their configurations differ.
            ("note", "same version != same behaviour; build configuration still differs"),
        ],
        "FFmpeg setup",
    )

    emit_outputs(
        {
            "ffmpeg-path": ffmpeg["path"],
            "ffprobe-path": ffprobe["path"],
            "version": version,
            "source": source,
            "cache-state": cache_state,
            "poisoned": "true" if poisoned else "false",
            "downloaded": "true" if digests else "false",
        }
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", help="override manifest path (tests)")
    parser.add_argument("--platform", help="override platform key, e.g. Windows-X64")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="emit cache key and platform outputs")
    plan.add_argument("--bin-dir", required=True)
    plan.set_defaults(func=cmd_plan)

    setup = sub.add_parser("setup", help="verify, install if needed, verify again")
    setup.add_argument("--bin-dir", required=True)
    setup.add_argument("--cache-hit", default="false", choices=["true", "false"])
    setup.add_argument(
        "--cache",
        default="true",
        choices=["true", "false"],
        help="whether actions/cache is in use; only affects how the run is reported",
    )
    setup.set_defaults(func=cmd_setup)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SetupError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
