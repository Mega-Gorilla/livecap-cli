"""Tests for the CI FFmpeg setup script (Issue #395).

Why this file exists: #398 found that the *runtime* FFmpeg downloader built
asset URLs that 404 on every platform, and nobody noticed for a long time
because that code path had no test at all. The CI downloader is now written in
Python specifically so the same class of bug is catchable here.

The network-backed URL check is opt-in (``-m network``, already excluded by the
default ``addopts``) so the default suite stays hermetic.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import urllib.error
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTION_DIR = REPO_ROOT / ".github" / "actions" / "setup-livecap-ffmpeg"
#: One manifest serves both the action and the runtime downloader (#398 D3).
MANIFEST_PATH = REPO_ROOT / "livecap_cli" / "resources" / "ffmpeg_manifest.json"


def _load_module():
    """Import by path: ``.github`` is not an importable package."""
    spec = importlib.util.spec_from_file_location("_setup_ffmpeg", ACTION_DIR / "setup_ffmpeg.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup_ffmpeg = _load_module()


@pytest.fixture(scope="module")
def manifest() -> dict:
    data, _ = setup_ffmpeg.load_manifest(MANIFEST_PATH)
    return data


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.invalid/x.zip", code, "", {}, None)


# ---------------------------------------------------------------------------
# Retry classification -- the single table both OSes share (#395 D1)
# ---------------------------------------------------------------------------


class TestClassification:
    @pytest.mark.parametrize("code", sorted(setup_ffmpeg.RETRYABLE_STATUS))
    def test_transient_statuses_retry(self, code: int) -> None:
        assert setup_ffmpeg.classify(_http_error(code)) == setup_ffmpeg.RETRY

    @pytest.mark.parametrize("code", [400, 401, 403, 404, 410, 418, 451])
    def test_permanent_4xx_is_fatal(self, code: int) -> None:
        """``curl --retry-all-errors`` would retry these. That is why we do not use it."""
        assert setup_ffmpeg.classify(_http_error(code)) == setup_ffmpeg.FATAL

    @pytest.mark.parametrize(
        "exc",
        [
            TimeoutError("timed out"),
            ConnectionResetError("reset"),
            ConnectionRefusedError("refused"),
            # DNS failure. curl's built-in --retry does NOT cover this, which is
            # one of the reasons the retry loop lives in our own code.
            urllib.error.URLError("[Errno -2] Name or service not known"),
        ],
    )
    def test_transport_failures_retry(self, exc: BaseException) -> None:
        assert setup_ffmpeg.classify(exc) == setup_ffmpeg.RETRY

    @pytest.mark.parametrize(
        "exc",
        [
            setup_ffmpeg.ChecksumMismatch(Path("a.zip"), "a" * 64, "b" * 64),
            setup_ffmpeg.ArchiveContentError("missing member"),
            ValueError("something else entirely"),
        ],
    )
    def test_supply_chain_and_unknown_failures_are_fatal(self, exc: BaseException) -> None:
        assert setup_ffmpeg.classify(exc) == setup_ffmpeg.FATAL


class TestBackoff:
    def test_is_exponential_and_bounded(self) -> None:
        assert setup_ffmpeg.backoff_delays(5, base=1.0, cap=8.0) == [1.0, 2.0, 4.0, 8.0]

    def test_cap_applies(self) -> None:
        assert setup_ffmpeg.backoff_delays(6, base=1.0, cap=4.0) == [1.0, 2.0, 4.0, 4.0, 4.0]

    def test_single_attempt_never_sleeps(self) -> None:
        assert setup_ffmpeg.backoff_delays(1) == []


# ---------------------------------------------------------------------------
# Retry loop
# ---------------------------------------------------------------------------


class TestDownloadWithRetry:
    def test_retries_transient_then_succeeds(self, tmp_path: Path) -> None:
        calls: list[int] = []

        def fetch(url: str, dest: Path) -> None:
            calls.append(1)
            if len(calls) < 3:
                raise _http_error(503)
            dest.write_bytes(b"payload")

        slept: list[float] = []
        setup_ffmpeg.download_with_retry(
            "https://example.invalid/a.zip",
            tmp_path / "a.zip",
            fetch=fetch,
            sleep=slept.append,
            log=lambda *_: None,
        )

        assert len(calls) == 3
        assert slept == [1.0, 2.0]
        assert (tmp_path / "a.zip").read_bytes() == b"payload"

    def test_permanent_4xx_does_not_retry(self, tmp_path: Path) -> None:
        calls: list[int] = []

        def fetch(url: str, dest: Path) -> None:
            calls.append(1)
            raise _http_error(404)

        with pytest.raises(urllib.error.HTTPError):
            setup_ffmpeg.download_with_retry(
                "https://example.invalid/a.zip",
                tmp_path / "a.zip",
                fetch=fetch,
                sleep=lambda _: None,
                log=lambda *_: None,
            )

        assert calls == [1], "a permanent 404 must not be retried"

    def test_partial_file_is_removed_before_the_next_attempt(self, tmp_path: Path) -> None:
        seen: list[bool] = []

        def fetch(url: str, dest: Path) -> None:
            # Record whether the previous attempt's leftovers are still around.
            seen.append(dest.exists())
            dest.write_bytes(b"half")  # truncated download
            raise _http_error(500)

        with pytest.raises(setup_ffmpeg.DownloadFailed):
            setup_ffmpeg.download_with_retry(
                "https://example.invalid/a.zip",
                tmp_path / "a.zip",
                attempts=3,
                fetch=fetch,
                sleep=lambda _: None,
                log=lambda *_: None,
            )

        assert seen == [False, False, False], "a partial archive survived into the next attempt"
        assert not (tmp_path / "a.zip").exists()

    def test_final_failure_reports_url_attempts_and_last_error(self, tmp_path: Path) -> None:
        with pytest.raises(setup_ffmpeg.DownloadFailed) as excinfo:
            setup_ffmpeg.download_with_retry(
                "https://example.invalid/pinned.zip",
                tmp_path / "pinned.zip",
                attempts=2,
                fetch=lambda url, dest: (_ for _ in ()).throw(_http_error(503)),
                sleep=lambda _: None,
                log=lambda *_: None,
            )

        message = str(excinfo.value)
        assert "https://example.invalid/pinned.zip" in message
        assert "2 attempt" in message
        assert "503" in message
        assert "LIVECAP_FFMPEG_BIN" in message, "the message must name the manual workaround"


# ---------------------------------------------------------------------------
# Archive handling
# ---------------------------------------------------------------------------


def _zip_with(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, payload in members.items():
            bundle.writestr(name, payload)
    return buffer.getvalue()


class TestArchiveHandling:
    def test_checksum_mismatch_is_fatal_and_does_not_redownload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The download loop never sees a checksum failure, so it cannot retry it."""
        downloads: list[str] = []

        def fake_download(url: str, dest: Path, **kwargs):
            downloads.append(url)
            dest.write_bytes(b"not the pinned archive")
            return dest

        monkeypatch.setattr(setup_ffmpeg, "download_with_retry", fake_download)

        manifest = {"base_url": "https://example.invalid", "version": "6.1"}
        spec = {
            "archives": {"ffmpeg": {"asset": "ffmpeg.zip", "sha256": "a" * 64}},
            "binaries": {"ffmpeg": {"name": "ffmpeg", "sha256": "b" * 64}},
        }

        with pytest.raises(setup_ffmpeg.ChecksumMismatch):
            setup_ffmpeg.install(manifest, spec, tmp_path / "bin", log=lambda *_: None)

        assert len(downloads) == 1, "a corrupt archive must not be fetched again"

    def test_missing_member_names_what_was_expected(self, tmp_path: Path) -> None:
        archive = tmp_path / "ffmpeg-6.1-win-64.zip"
        archive.write_bytes(_zip_with({"bin/something-else.exe": b"x"}))

        with pytest.raises(setup_ffmpeg.ArchiveContentError) as excinfo:
            setup_ffmpeg.extract_binary(archive, "ffmpeg.exe", tmp_path / "ffmpeg.exe")

        message = str(excinfo.value)
        assert "ffmpeg.exe" in message
        assert "bin/something-else.exe" in message, "the failure must list what the archive holds"

    def test_extracts_a_nested_member(self, tmp_path: Path) -> None:
        archive = tmp_path / "a.zip"
        archive.write_bytes(_zip_with({"ffmpeg-6.1/bin/ffmpeg.exe": b"binary bytes"}))

        destination = tmp_path / "ffmpeg.exe"
        setup_ffmpeg.extract_binary(archive, "ffmpeg.exe", destination)

        assert destination.read_bytes() == b"binary bytes"


# ---------------------------------------------------------------------------
# Verification -- the invariant that does not care where the bytes came from
# ---------------------------------------------------------------------------


class TestVerifyBinaries:
    @staticmethod
    def _install_fake(bin_dir: Path, payload: bytes = b"pretend ffmpeg") -> dict:
        bin_dir.mkdir(parents=True, exist_ok=True)
        spec: dict = {"binaries": {}}
        for role in ("ffmpeg", "ffprobe"):
            blob = payload + role.encode()
            (bin_dir / role).write_bytes(blob)
            spec["binaries"][role] = {"name": role, "sha256": hashlib.sha256(blob).hexdigest()}
        return spec

    def test_healthy_install_passes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        spec = self._install_fake(tmp_path)
        monkeypatch.setattr(
            setup_ffmpeg, "probe_version", lambda p: (0, "ffmpeg version 6.1-static", "--enable-gpl")
        )

        ok, reasons, details = setup_ffmpeg.verify_binaries(tmp_path, spec, "6.1")

        assert ok and reasons == []
        assert details["ffmpeg"]["configuration"] == "--enable-gpl"

    def test_missing_binary_is_named(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        spec = self._install_fake(tmp_path)
        (tmp_path / "ffprobe").unlink()
        monkeypatch.setattr(setup_ffmpeg, "probe_version", lambda p: (0, "ffmpeg version 6.1", ""))

        ok, reasons, _ = setup_ffmpeg.verify_binaries(tmp_path, spec, "6.1")

        assert not ok
        assert any("ffprobe" in r and "missing" in r for r in reasons)

    def test_tampered_binary_is_named(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """This is what catches a stale binary in a self-hosted persistent directory."""
        spec = self._install_fake(tmp_path)
        (tmp_path / "ffmpeg").write_bytes(b"a different build entirely")
        monkeypatch.setattr(setup_ffmpeg, "probe_version", lambda p: (0, "ffmpeg version 6.1", ""))

        ok, reasons, _ = setup_ffmpeg.verify_binaries(tmp_path, spec, "6.1")

        assert not ok
        assert any("ffmpeg" in r and "SHA-256 mismatch" in r for r in reasons)

    def test_unrunnable_binary_is_named(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        spec = self._install_fake(tmp_path)
        monkeypatch.setattr(setup_ffmpeg, "probe_version", lambda p: (126, "(could not execute)", ""))

        ok, reasons, _ = setup_ffmpeg.verify_binaries(tmp_path, spec, "6.1")

        assert not ok
        assert all("not runnable" in r for r in reasons)

    def test_wrong_version_is_named(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        spec = self._install_fake(tmp_path)
        monkeypatch.setattr(setup_ffmpeg, "probe_version", lambda p: (0, "ffmpeg version 7.0", ""))

        ok, reasons, _ = setup_ffmpeg.verify_binaries(tmp_path, spec, "6.1")

        assert not ok
        assert all("expected version 6.1" in r for r in reasons)

    @pytest.mark.parametrize(
        "line, expected, matches",
        [
            ("ffmpeg version 6.1-full_build-www.gyan.dev Copyright (c)", "6.1", True),
            ("ffprobe version 6.1-static https://johnvansickle.com/ffmpeg/", "6.1", True),
            ("ffmpeg version 6.1 Copyright (c)", "6.1", True),
            # A point release is a different build, so it must not pass as 6.1.
            ("ffmpeg version 6.1.1 Copyright (c)", "6.1", False),
            ("ffmpeg version 7.0-static", "6.1", False),
            ("", "6.1", False),
        ],
    )
    def test_version_matching(self, line: str, expected: str, matches: bool) -> None:
        assert setup_ffmpeg.version_matches(line, expected) is matches


# ---------------------------------------------------------------------------
# Manifest / cache key
# ---------------------------------------------------------------------------


class TestManifest:
    def test_every_platform_pins_two_archives_and_two_binaries(self, manifest: dict) -> None:
        assert manifest["platforms"], "manifest pins no platforms"
        for key, spec in manifest["platforms"].items():
            assert set(spec["archives"]) == {"ffmpeg", "ffprobe"}, key
            assert set(spec["binaries"]) == {"ffmpeg", "ffprobe"}, key

    def test_all_hashes_are_full_sha256(self, manifest: dict) -> None:
        for key, spec in manifest["platforms"].items():
            for group in ("archives", "binaries"):
                for role, entry in spec[group].items():
                    digest = entry["sha256"]
                    assert len(digest) == 64, f"{key}/{group}/{role}"
                    assert set(digest) <= set("0123456789abcdef"), f"{key}/{group}/{role}"

    def test_asset_names_embed_the_version(self, manifest: dict) -> None:
        """ffbinaries names assets ``<tool>-<version>-<platform>.zip``.

        Dropping the version is exactly the bug #398 found in the runtime
        downloader, where every generated URL 404s.
        """
        version = manifest["version"]
        for key, spec in manifest["platforms"].items():
            for role, entry in spec["archives"].items():
                assert entry["asset"].startswith(f"{role}-{version}-"), f"{key}/{role}"
                assert entry["asset"].endswith(".zip"), f"{key}/{role}"

    def test_platform_tokens_match_upstream_naming(self, manifest: dict) -> None:
        """``windows-64`` / ``osx-64`` do not exist upstream; ``win-64`` / ``macos-64`` do."""
        for spec in manifest["platforms"].values():
            for entry in spec["archives"].values():
                assert "-windows-" not in entry["asset"]
                assert "-osx-" not in entry["asset"]

    def test_windows_binaries_carry_the_exe_suffix(self, manifest: dict) -> None:
        for role, entry in manifest["platforms"]["win-64"]["binaries"].items():
            assert entry["name"] == f"{role}.exe"
        for role, entry in manifest["platforms"]["linux-64"]["binaries"].items():
            assert entry["name"] == role

    def test_base_url_is_pinned_not_rolling(self, manifest: dict) -> None:
        """``releases/latest`` makes the installed version drift silently (#395, #398)."""
        assert "/releases/latest" not in manifest["base_url"]
        assert manifest["base_url"].endswith(f"/{manifest['release_tag']}")


class TestPlatformAndCacheKey:
    def test_unknown_platform_fails_loud(self, manifest: dict) -> None:
        with pytest.raises(setup_ffmpeg.SetupError, match="Supported"):
            setup_ffmpeg.resolve_platform_key(manifest, "plan9-64")

    def test_runner_environment_selects_the_platform(
        self, manifest: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RUNNER_OS", "Windows")
        monkeypatch.setenv("RUNNER_ARCH", "X64")
        assert setup_ffmpeg.resolve_platform_key(manifest) == "win-64"

    def test_key_changes_with_generation(self, manifest: dict) -> None:
        digest = "0" * 64
        first = setup_ffmpeg.compute_cache_key(manifest, digest, "linux-64")
        bumped = setup_ffmpeg.compute_cache_key(
            {**manifest, "cache_generation": manifest["cache_generation"] + 1}, digest, "linux-64"
        )
        assert first != bumped

    def test_key_changes_with_manifest_content(self, manifest: dict) -> None:
        first = setup_ffmpeg.compute_cache_key(manifest, "0" * 64, "linux-64")
        second = setup_ffmpeg.compute_cache_key(manifest, "1" * 64, "linux-64")
        assert first != second

    def test_platforms_do_not_share_a_key(self, manifest: dict) -> None:
        digest = "0" * 64
        keys = {
            setup_ffmpeg.compute_cache_key(manifest, digest, key) for key in manifest["platforms"]
        }
        assert len(keys) == len(manifest["platforms"])


class TestProvenance:
    """What a run reports about where its bytes came from.

    The digests of an actual download are only available when a download
    happens; a cache hit has none. Provenance is derived from the manifest so
    the origin and the expected checksums are reported either way.
    """

    def test_reports_url_and_both_expected_digests(self, manifest: dict) -> None:
        spec = manifest["platforms"]["win-64"]
        rows = dict(setup_ffmpeg.pinned_provenance(manifest, spec))

        for role in ("ffmpeg", "ffprobe"):
            url = rows[f"{role} archive url"]
            assert url.startswith(manifest["base_url"] + "/")
            assert url.endswith(spec["archives"][role]["asset"])
            assert rows[f"{role} archive sha256 (expected)"] == spec["archives"][role]["sha256"]
            assert rows[f"{role} binary sha256 (expected)"] == spec["binaries"][role]["sha256"]

    def test_covers_every_platform(self, manifest: dict) -> None:
        for key, spec in manifest["platforms"].items():
            rows = setup_ffmpeg.pinned_provenance(manifest, spec)
            assert len(rows) == 3 * len(spec["archives"]), key

    def test_digests_are_not_truncated(self, manifest: dict) -> None:
        """A prefix cannot be compared against a build by hand."""
        rows = setup_ffmpeg.pinned_provenance(manifest, manifest["platforms"]["linux-64"])
        for name, value in rows:
            if name.endswith("sha256 (expected)"):
                assert len(value) == 64 and "..." not in value


# ---------------------------------------------------------------------------
# Opt-in: the pinned URLs actually resolve
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_every_pinned_asset_url_resolves(manifest: dict) -> None:
    """Guards the #398 failure mode: a plausible-looking URL that 404s everywhere.

    Run with ``uv run pytest tests/ci -m network``.
    """
    import urllib.request

    failures: list[str] = []
    for platform_key, spec in manifest["platforms"].items():
        for role, entry in spec["archives"].items():
            url = f"{manifest['base_url']}/{entry['asset']}"
            request = urllib.request.Request(
                url, method="HEAD", headers={"User-Agent": setup_ffmpeg.USER_AGENT}
            )
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 200:
                        failures.append(f"{platform_key}/{role}: HTTP {response.status} {url}")
            except urllib.error.HTTPError as exc:
                failures.append(f"{platform_key}/{role}: HTTP {exc.code} {url}")

    assert not failures, "pinned assets do not exist:\n  " + "\n  ".join(failures)
