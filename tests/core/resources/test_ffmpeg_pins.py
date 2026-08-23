"""Pinned FFmpeg manifest and platform selection (Issue #398).

The bug this guards against: the old selector built asset names that 404 on every
platform, and picked builds by asking whether ``"64"`` appeared anywhere in
``platform.machine()`` - which is true of ``aarch64``. Neither had a test, so
both survived from the first commit.
"""

from __future__ import annotations

import urllib.request

import pytest

from livecap_cli.resources import ffmpeg_pins


@pytest.fixture(scope="module")
def manifest() -> dict:
    return ffmpeg_pins.load_manifest()


class TestManifest:
    def test_every_platform_pins_two_archives_and_two_binaries(self, manifest: dict) -> None:
        assert manifest["platforms"], "manifest has no platforms"
        for token, entry in manifest["platforms"].items():
            assert set(entry["archives"]) == {"ffmpeg", "ffprobe"}, token
            assert set(entry["binaries"]) == {"ffmpeg", "ffprobe"}, token

    def test_digests_are_full_sha256(self, manifest: dict) -> None:
        for token, entry in manifest["platforms"].items():
            for section in ("archives", "binaries"):
                for role, spec in entry[section].items():
                    digest = spec["sha256"]
                    assert len(digest) == 64, (token, section, role)
                    assert set(digest) <= set("0123456789abcdef"), (token, section, role)

    def test_asset_names_carry_the_version(self, manifest: dict) -> None:
        """``releases/latest`` cannot supply it: the version is part of the name."""
        version = manifest["version"]
        for token, entry in manifest["platforms"].items():
            for role, spec in entry["archives"].items():
                assert spec["asset"].startswith(f"{role}-{version}-"), (token, role)

    def test_no_asset_uses_a_token_that_does_not_exist_upstream(self, manifest: dict) -> None:
        """``windows-64`` / ``osx-64`` are what produced the 404s."""
        for entry in manifest["platforms"].values():
            for spec in entry["archives"].values():
                assert "-windows-" not in spec["asset"]
                assert "-osx-" not in spec["asset"]

    def test_ffprobe_asset_is_not_derived_from_the_ffmpeg_asset(self, manifest: dict) -> None:
        """Upstream naming is not symmetric (ffmpeg armel-32 vs ffprobe armel-64)."""
        for token, entry in manifest["platforms"].items():
            ffmpeg_asset = entry["archives"]["ffmpeg"]["asset"]
            ffprobe_asset = entry["archives"]["ffprobe"]["asset"]
            assert ffmpeg_asset != ffprobe_asset, token

    def test_base_url_is_pinned_not_rolling(self, manifest: dict) -> None:
        assert "/releases/latest" not in manifest["base_url"]
        assert manifest["base_url"].endswith("/" + manifest["release_tag"])

    def test_license_is_recorded(self, manifest: dict) -> None:
        """Pinning means owning the update decision; the terms have to be written down."""
        assert manifest["license"]["effective"]
        assert manifest["license"]["evidence"]


class TestPlatformSelection:
    @pytest.mark.parametrize(
        ("system", "machine", "expected"),
        [
            ("Windows", "AMD64", "win-64"),
            ("Windows", "x86_64", "win-64"),
            ("windows", "x64", "win-64"),
            ("Linux", "x86_64", "linux-64"),
            ("Linux", "AMD64", "linux-64"),
            ("Darwin", "x86_64", "macos-64"),
        ],
    )
    def test_supported(self, system: str, machine: str, expected: str) -> None:
        assert ffmpeg_pins.resolve_platform_token(system, machine) == expected

    @pytest.mark.parametrize(
        ("system", "machine"),
        [
            ("Linux", "aarch64"),
            ("Linux", "arm64"),
            ("Linux", "armv7l"),
            ("Linux", "i686"),
            ("Darwin", "arm64"),
            ("Windows", "ARM64"),
            ("Windows", "x86"),
            ("Haiku", "x86_64"),
        ],
    )
    def test_unsupported_fails_loud(self, system: str, machine: str) -> None:
        with pytest.raises(ffmpeg_pins.UnsupportedPlatformError):
            ffmpeg_pins.resolve_platform_token(system, machine)

    def test_aarch64_does_not_get_the_x86_64_build(self) -> None:
        """``"64" in "aarch64"`` was true, so ARM machines were served x86-64."""
        with pytest.raises(ffmpeg_pins.UnsupportedPlatformError):
            ffmpeg_pins.resolve_platform_token("Linux", "aarch64")

    def test_armv7l_does_not_get_the_32bit_x86_build(self) -> None:
        with pytest.raises(ffmpeg_pins.UnsupportedPlatformError):
            ffmpeg_pins.resolve_platform_token("Linux", "armv7l")

    @pytest.mark.parametrize(
        ("system", "machine", "hint"),
        [
            ("Darwin", "arm64", "brew install ffmpeg"),
            ("Linux", "aarch64", "package manager"),
            ("Windows", "x86", "winget"),
        ],
    )
    def test_error_names_a_way_out(self, system: str, machine: str, hint: str) -> None:
        with pytest.raises(ffmpeg_pins.UnsupportedPlatformError) as excinfo:
            ffmpeg_pins.resolve_platform_token(system, machine)
        message = str(excinfo.value)
        assert hint in message
        assert "LIVECAP_FFMPEG_BIN" in message

    def test_macos_arm64_refuses_rather_than_using_rosetta(self) -> None:
        with pytest.raises(ffmpeg_pins.UnsupportedPlatformError) as excinfo:
            ffmpeg_pins.resolve_platform_token("Darwin", "arm64")
        assert "Rosetta" in str(excinfo.value)


class TestPlatformSpec:
    def test_urls_are_built_from_base_url_and_asset(self, manifest: dict) -> None:
        spec = ffmpeg_pins.resolve_platform_spec("Windows", "AMD64")
        for role, archive in spec.archives.items():
            assert archive.url == f"{manifest['base_url']}/{archive.asset}"

    def test_windows_binaries_keep_the_exe_suffix(self) -> None:
        spec = ffmpeg_pins.resolve_platform_spec("Windows", "AMD64")
        assert spec.binaries["ffmpeg"].name == "ffmpeg.exe"
        assert spec.binaries["ffprobe"].name == "ffprobe.exe"

    def test_posix_binaries_have_no_suffix(self) -> None:
        spec = ffmpeg_pins.resolve_platform_spec("Linux", "x86_64")
        assert spec.binaries["ffmpeg"].name == "ffmpeg"

    def test_version_matches_the_manifest(self, manifest: dict) -> None:
        spec = ffmpeg_pins.resolve_platform_spec("Darwin", "x86_64")
        assert spec.version == manifest["version"] == ffmpeg_pins.pinned_version()


# ---------------------------------------------------------------------------
# Opt-in: the pinned URLs actually resolve
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_every_pinned_asset_url_resolves(manifest: dict) -> None:
    """The #398 failure mode itself: a plausible-looking URL that 404s."""
    failures = []
    for token, entry in manifest["platforms"].items():
        for role, spec in entry["archives"].items():
            url = f"{manifest['base_url']}/{spec['asset']}"
            request = urllib.request.Request(url, method="HEAD")
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 200:
                        failures.append(f"{token}/{role}: HTTP {response.status} {url}")
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                failures.append(f"{token}/{role}: {exc!r} {url}")

    assert not failures, "\n".join(failures)
