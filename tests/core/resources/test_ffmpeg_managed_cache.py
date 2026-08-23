"""Managed FFmpeg cache: pair contract, stamp, atomic install (Issue #398).

These exercise the install path end to end with a fake upstream, because the
bugs being fixed were all in the seams: an archive that was never fetched, a
binary that was silently skipped, and a cache that could never be repaired.
"""

from __future__ import annotations

import hashlib
import io
import os
import threading
import urllib.error
import zipfile
from pathlib import Path
from types import MethodType

import pytest

from livecap_cli.resources import ffmpeg_manager as fm
from livecap_cli.resources import ffmpeg_pins
from livecap_cli.resources.ffmpeg_pins import ArchiveSpec, BinarySpec, PlatformSpec

FFMPEG_PAYLOAD = b"fake ffmpeg binary" * 64
FFPROBE_PAYLOAD = b"fake ffprobe binary" * 64


def _zip_bytes(member: str, payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr(member, payload)
    return buffer.getvalue()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeUpstream:
    """Serves two archives and counts what was fetched."""

    def __init__(self, windows: bool) -> None:
        self.ffmpeg_name = "ffmpeg.exe" if windows else "ffmpeg"
        self.ffprobe_name = "ffprobe.exe" if windows else "ffprobe"
        self.blobs = {
            "https://test.invalid/ffmpeg.zip": _zip_bytes(self.ffmpeg_name, FFMPEG_PAYLOAD),
            "https://test.invalid/ffprobe.zip": _zip_bytes(self.ffprobe_name, FFPROBE_PAYLOAD),
        }
        self.fetched: list[str] = []
        self.fail_for: set[str] = set()

    @property
    def spec(self) -> PlatformSpec:
        return PlatformSpec(
            token="test-64",
            version="6.1",
            archives={
                "ffmpeg": ArchiveSpec(
                    asset="ffmpeg-6.1-test-64.zip",
                    sha256=_sha(self.blobs["https://test.invalid/ffmpeg.zip"]),
                    url="https://test.invalid/ffmpeg.zip",
                ),
                "ffprobe": ArchiveSpec(
                    asset="ffprobe-6.1-test-64.zip",
                    sha256=_sha(self.blobs["https://test.invalid/ffprobe.zip"]),
                    url="https://test.invalid/ffprobe.zip",
                ),
            },
            binaries={
                "ffmpeg": BinarySpec(name=self.ffmpeg_name, sha256=_sha(FFMPEG_PAYLOAD)),
                "ffprobe": BinarySpec(name=self.ffprobe_name, sha256=_sha(FFPROBE_PAYLOAD)),
            },
        )

    def download(self, url: str, destination: Path, **kwargs) -> Path:
        self.fetched.append(url)
        if url in self.fail_for:
            raise fm.DownloadFailed(url, 5, RuntimeError("upstream is down"))
        destination.write_bytes(self.blobs[url])
        return destination


@pytest.fixture()
def upstream(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeUpstream:
    monkeypatch.setenv("LIVECAP_CORE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LIVECAP_CORE_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.delenv("LIVECAP_FFMPEG_BIN", raising=False)

    fake = FakeUpstream(windows=os.name == "nt")
    monkeypatch.setattr(ffmpeg_pins, "resolve_platform_spec", lambda *a, **k: fake.spec)
    monkeypatch.setattr(fm.ffmpeg_pins, "resolve_platform_spec", lambda *a, **k: fake.spec)
    monkeypatch.setattr(fm, "download_with_retry", fake.download)
    # The fake payloads are not runnable; the post-install probe is covered by
    # its own test below, which puts this method back.
    fake.real_probe = fm.FFmpegManager._probe_version
    monkeypatch.setattr(fm.FFmpegManager, "_probe_version", lambda self, *a: "fake 6.1")
    return fake


@pytest.fixture()
def manager(upstream: FakeUpstream, monkeypatch: pytest.MonkeyPatch) -> fm.FFmpegManager:
    instance = fm.FFmpegManager()
    # Isolate from a real ffmpeg on this machine: only the managed cache counts.
    monkeypatch.setattr(instance, "_candidate_from_bundled", lambda name: None)
    monkeypatch.setattr(instance, "_candidate_from_system", lambda name: None)
    return instance


class TestInstall:
    def test_installs_both_binaries_from_two_archives(
        self, manager: fm.FFmpegManager, upstream: FakeUpstream
    ) -> None:
        """The old code fetched one archive and quietly went without ffprobe."""
        executable = manager.ensure_executable()

        assert executable.read_bytes() == FFMPEG_PAYLOAD
        probe = manager.resolve_probe()
        assert probe is not None and probe.read_bytes() == FFPROBE_PAYLOAD
        assert len(upstream.fetched) == 2

    def test_writes_a_stamp(self, manager: fm.FFmpegManager) -> None:
        manager.ensure_executable()
        stamp = manager._cache_dir / fm.STAMP_NAME
        assert stamp.is_file()
        assert set(manager._read_stamp()) == {"ffmpeg", "ffprobe"}

    def test_unsupported_platform_fails_with_instructions(
        self, manager: fm.FFmpegManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unsupported(*args, **kwargs):
            raise ffmpeg_pins.UnsupportedPlatformError("no build for darwin/arm64: reasons")

        monkeypatch.setattr(fm.ffmpeg_pins, "resolve_platform_spec", unsupported)

        with pytest.raises(fm.FFmpegNotFoundError) as excinfo:
            manager.ensure_executable()
        assert "darwin/arm64" in str(excinfo.value)


class TestPairContract:
    def test_missing_ffprobe_invalidates_the_whole_pair(
        self, manager: fm.FFmpegManager, upstream: FakeUpstream
    ) -> None:
        """ffmpeg alone used to be enough, so a half cache lived forever."""
        manager.ensure_executable()
        probe = manager._cache_dir / upstream.ffprobe_name
        probe.unlink()

        fresh = _reopen(manager)
        assert fresh._candidate_from_managed_cache(upstream.ffmpeg_name) is None

        fresh.ensure_executable()
        assert probe.read_bytes() == FFPROBE_PAYLOAD
        assert len(upstream.fetched) == 4  # both archives fetched again

    def test_tampered_ffmpeg_invalidates_the_whole_pair(
        self, manager: fm.FFmpegManager, upstream: FakeUpstream
    ) -> None:
        manager.ensure_executable()
        binary = manager._cache_dir / upstream.ffmpeg_name
        binary.write_bytes(FFMPEG_PAYLOAD + b"x")

        fresh = _reopen(manager)
        fresh.ensure_executable()

        assert binary.read_bytes() == FFMPEG_PAYLOAD
        assert len(upstream.fetched) == 4

    def test_invalid_pair_is_repaired_even_when_path_has_ffmpeg(
        self, upstream: FakeUpstream, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A corrupted cache must not silently demote us to the host's FFmpeg.

        The managed cache outranks PATH, so falling through on corruption would
        change which FFmpeg the application runs - invisibly, and to an unknown
        version.
        """
        instance = fm.FFmpegManager()
        monkeypatch.setattr(instance, "_candidate_from_bundled", lambda name: None)
        monkeypatch.setattr(instance, "_candidate_from_system", lambda name: None)
        instance.ensure_executable()
        (instance._cache_dir / upstream.ffprobe_name).unlink()

        # This time a perfectly good ffmpeg exists on PATH.
        system_dir = tmp_path / "system"
        system_dir.mkdir()
        system_ffmpeg = system_dir / upstream.ffmpeg_name
        system_ffmpeg.write_bytes(b"the host's own ffmpeg")

        fresh = fm.FFmpegManager()
        monkeypatch.setattr(fresh, "_candidate_from_bundled", lambda name: None)
        monkeypatch.setattr(
            fresh, "_candidate_from_system", lambda name: system_ffmpeg if "ffmpeg" in name else None
        )

        resolved = fresh.ensure_executable()

        assert resolved == fresh._cache_dir / upstream.ffmpeg_name
        assert resolved != system_ffmpeg
        assert (fresh._cache_dir / upstream.ffprobe_name).read_bytes() == FFPROBE_PAYLOAD

    def test_absent_pair_defers_to_path_without_downloading(
        self, upstream: FakeUpstream, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Absent is not broken: nothing of ours needs fixing."""
        system_dir = tmp_path / "system"
        system_dir.mkdir()
        system_ffmpeg = system_dir / upstream.ffmpeg_name
        system_ffmpeg.write_bytes(b"the host's own ffmpeg")

        instance = fm.FFmpegManager()
        monkeypatch.setattr(instance, "_candidate_from_bundled", lambda name: None)
        monkeypatch.setattr(
            instance,
            "_candidate_from_system",
            lambda name: system_ffmpeg if "ffmpeg" in name else None,
        )

        assert instance.ensure_executable() == system_ffmpeg
        assert upstream.fetched == []

    def test_repair_failure_falls_back_instead_of_failing_outright(
        self, upstream: FakeUpstream, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Being offline with a corrupt cache should not be worse than having none."""
        instance = fm.FFmpegManager()
        monkeypatch.setattr(instance, "_candidate_from_bundled", lambda name: None)
        monkeypatch.setattr(instance, "_candidate_from_system", lambda name: None)
        instance.ensure_executable()
        (instance._cache_dir / upstream.ffprobe_name).unlink()

        upstream.fail_for = set(upstream.blobs)
        system_dir = tmp_path / "system"
        system_dir.mkdir()
        system_ffmpeg = system_dir / upstream.ffmpeg_name
        system_ffmpeg.write_bytes(b"the host's own ffmpeg")

        fresh = fm.FFmpegManager()
        monkeypatch.setattr(fresh, "_candidate_from_bundled", lambda name: None)
        monkeypatch.setattr(
            fresh, "_candidate_from_system", lambda name: system_ffmpeg if "ffmpeg" in name else None
        )

        assert fresh.ensure_executable() == system_ffmpeg

    def test_host_managed_binaries_are_never_replaced(
        self, upstream: FakeUpstream, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A binary the user pointed us at is theirs, digest or no digest."""
        host_dir = tmp_path / "host-ffmpeg"
        host_dir.mkdir()
        host_ffmpeg = host_dir / upstream.ffmpeg_name
        host_ffmpeg.write_bytes(b"a completely different build")
        monkeypatch.setenv("LIVECAP_FFMPEG_BIN", str(host_dir))

        instance = fm.FFmpegManager()
        resolved = instance.ensure_executable()

        assert resolved == host_ffmpeg
        assert host_ffmpeg.read_bytes() == b"a completely different build"
        assert upstream.fetched == []


class TestStamp:
    def test_matching_stamp_skips_hashing(
        self, manager: fm.FFmpegManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hashing 268 MB on every start is the cost the stamp exists to avoid."""
        manager.ensure_executable()

        hashed: list[Path] = []
        original = fm._sha256_file
        monkeypatch.setattr(fm, "_sha256_file", lambda p: (hashed.append(p), original(p))[1])

        fresh = _reopen(manager)
        assert fresh.resolve_executable().name == manager._cached_ffmpeg.name
        assert hashed == []

    def test_missing_stamp_falls_back_to_hashing_and_rewrites_it(
        self, manager: fm.FFmpegManager, monkeypatch: pytest.MonkeyPatch, upstream: FakeUpstream
    ) -> None:
        manager.ensure_executable()
        (manager._cache_dir / fm.STAMP_NAME).unlink()

        hashed: list[Path] = []
        original = fm._sha256_file
        monkeypatch.setattr(fm, "_sha256_file", lambda p: (hashed.append(p), original(p))[1])

        fresh = _reopen(manager)
        fresh.resolve_executable()

        assert len(hashed) == 2  # both binaries, no download
        assert len(upstream.fetched) == 2
        assert (manager._cache_dir / fm.STAMP_NAME).is_file()

    def test_stale_stamp_does_not_bless_wrong_bytes(
        self, manager: fm.FFmpegManager, upstream: FakeUpstream
    ) -> None:
        """Touching a file changes mtime, so the stamp stops being trusted."""
        manager.ensure_executable()
        binary = manager._cache_dir / upstream.ffmpeg_name
        binary.write_bytes(b"replaced after the stamp was written")

        fresh = _reopen(manager)
        assert fresh._candidate_from_managed_cache(upstream.ffmpeg_name) is None


class TestAtomicity:
    def test_failed_second_download_leaves_the_cache_untouched(
        self, manager: fm.FFmpegManager, upstream: FakeUpstream
    ) -> None:
        upstream.fail_for = {"https://test.invalid/ffprobe.zip"}

        with pytest.raises(fm.FFmpegNotFoundError) as excinfo:
            manager.ensure_executable()

        assert "upstream is down" in str(excinfo.value)
        assert not (manager._cache_dir / upstream.ffmpeg_name).exists()
        assert not (manager._cache_dir / upstream.ffprobe_name).exists()
        assert not (manager._cache_dir / fm.STAMP_NAME).exists()

    def test_permanent_error_still_names_the_url_and_the_way_out(
        self, manager: fm.FFmpegManager, upstream: FakeUpstream, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 404 is not retried, so it never becomes a DownloadFailed on its own."""

        def refuse(url: str, destination: Path, **kwargs) -> Path:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

        monkeypatch.setattr(fm, "download_with_retry", refuse)

        with pytest.raises(fm.FFmpegNotFoundError) as excinfo:
            manager.ensure_executable()

        message = str(excinfo.value)
        assert "https://test.invalid/ffmpeg.zip" in message
        assert "404" in message
        assert "LIVECAP_FFMPEG_BIN" in message

    def test_second_publish_failure_is_detected_on_the_next_run(
        self, manager: fm.FFmpegManager, upstream: FakeUpstream, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Publishing is two renames, so it is not atomic - but it is detectable.

        The stamp is written after both renames. A failure between them leaves
        the stamp describing the old state, so the next verification hashes the
        files and reinstalls the pair.
        """
        real_replace = os.replace
        calls: list[int] = []
        armed = [True]

        def replace_once(src, dst):
            calls.append(1)
            if armed[0] and len(calls) == 2:
                raise PermissionError("second rename failed")
            return real_replace(src, dst)

        monkeypatch.setattr(fm.os, "replace", replace_once)

        with pytest.raises(fm.FFmpegNotFoundError):
            manager.ensure_executable()

        # Half-published: one binary landed, the stamp did not.
        assert not (manager._cache_dir / fm.STAMP_NAME).exists()

        armed[0] = False
        fresh = _reopen(manager)
        assert fresh._managed_pair_state() == fm._STATE_INVALID
        fresh.ensure_executable()
        assert (fresh._cache_dir / upstream.ffprobe_name).read_bytes() == FFPROBE_PAYLOAD

    def test_failed_reinstall_leaves_the_previous_pair_intact(
        self, manager: fm.FFmpegManager, upstream: FakeUpstream
    ) -> None:
        manager.ensure_executable()
        (manager._cache_dir / upstream.ffprobe_name).unlink()
        upstream.fail_for = {"https://test.invalid/ffprobe.zip"}

        fresh = _reopen(manager)
        with pytest.raises(fm.FFmpegNotFoundError):
            fresh.ensure_executable()

        # ffmpeg was already correct and must not have been rolled back.
        assert (manager._cache_dir / upstream.ffmpeg_name).read_bytes() == FFMPEG_PAYLOAD

    def test_archive_without_the_expected_member_names_what_it_holds(
        self, manager: fm.FFmpegManager, upstream: FakeUpstream
    ) -> None:
        upstream.blobs["https://test.invalid/ffprobe.zip"] = _zip_bytes("README", b"nope")

        with pytest.raises(fm.FFmpegNotFoundError) as excinfo:
            manager.ensure_executable()

        message = str(excinfo.value)
        assert "does not contain" in message
        assert "README" in message  # says what the archive actually held
        assert not (manager._cache_dir / upstream.ffmpeg_name).exists()


class TestConcurrency:
    def test_parallel_ensure_installs_once(
        self, manager: fm.FFmpegManager, upstream: FakeUpstream
    ) -> None:
        """Sync and async paths share one lock, so they cannot race the cache."""
        barrier = threading.Barrier(4)
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                barrier.wait(timeout=10)
                manager.ensure_executable()
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert not errors
        assert len(upstream.fetched) == 2
        assert (manager._cache_dir / upstream.ffprobe_name).read_bytes() == FFPROBE_PAYLOAD

    def test_separate_manager_instances_share_the_lock(
        self, upstream: FakeUpstream, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """They share one cache directory, so a per-instance lock would not guard it."""
        managers = []
        for _ in range(3):
            instance = fm.FFmpegManager()
            monkeypatch.setattr(instance, "_candidate_from_bundled", lambda name: None)
            monkeypatch.setattr(instance, "_candidate_from_system", lambda name: None)
            managers.append(instance)

        barrier = threading.Barrier(len(managers), timeout=10)
        errors: list[BaseException] = []

        def worker(instance: fm.FFmpegManager) -> None:
            try:
                barrier.wait()
                instance.ensure_executable()
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(m,)) for m in managers]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert not errors
        assert len(upstream.fetched) == 2
        # Exactly one stamp, and no abandoned temporary files next to it.
        leftovers = sorted(
            p.name for p in managers[0]._cache_dir.iterdir() if p.name.startswith(fm.STAMP_NAME)
        )
        assert leftovers == [fm.STAMP_NAME]


class TestRunnabilityCheck:
    @staticmethod
    def _with_real_probe(
        monkeypatch: pytest.MonkeyPatch, upstream: FakeUpstream
    ) -> fm.FFmpegManager:
        instance = fm.FFmpegManager()
        monkeypatch.setattr(instance, "_candidate_from_bundled", lambda name: None)
        monkeypatch.setattr(instance, "_candidate_from_system", lambda name: None)
        # Restore the real probe that the fixture stubbed out.
        monkeypatch.setattr(instance, "_probe_version", MethodType(upstream.real_probe, instance))
        return instance

    def test_unrunnable_binary_is_reported(
        self, upstream: FakeUpstream, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """"Ensure" promises a usable binary, not merely a correctly hashed one."""
        instance = self._with_real_probe(monkeypatch, upstream)

        with pytest.raises(fm.FFmpegNotFoundError) as excinfo:
            instance.ensure_executable()
        assert "could not be executed" in str(excinfo.value) or "exited" in str(excinfo.value)

    def test_unrunnable_binary_is_never_published(
        self, upstream: FakeUpstream, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The check runs while staged: a binary that fails it must not survive.

        Publishing first would leave the failed binary - and a stamp blessing
        it - for every later run to trust.
        """
        instance = self._with_real_probe(monkeypatch, upstream)

        with pytest.raises(fm.FFmpegNotFoundError):
            instance.ensure_executable()

        assert not (instance._cache_dir / upstream.ffmpeg_name).exists()
        assert not (instance._cache_dir / upstream.ffprobe_name).exists()
        assert not (instance._cache_dir / fm.STAMP_NAME).exists()

        # Neither the same manager nor a new one may serve the failed binary.
        with pytest.raises(fm.FFmpegNotFoundError):
            instance.ensure_executable()
        fresh = self._with_real_probe(monkeypatch, upstream)
        with pytest.raises(fm.FFmpegNotFoundError):
            fresh.ensure_executable()

    def test_both_binaries_are_probed(
        self, upstream: FakeUpstream, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pair contract covers usability, not just presence."""
        probed: list[str] = []

        def record(self, executable: Path, role: str, spec) -> str:
            probed.append(role)
            return "fake 6.1"

        instance = fm.FFmpegManager()
        monkeypatch.setattr(instance, "_candidate_from_bundled", lambda name: None)
        monkeypatch.setattr(instance, "_candidate_from_system", lambda name: None)
        monkeypatch.setattr(instance, "_probe_version", MethodType(record, instance))

        instance.ensure_executable()
        assert sorted(probed) == ["ffmpeg", "ffprobe"]


def _reopen(manager: fm.FFmpegManager) -> fm.FFmpegManager:
    """A new manager over the same cache: no in-process memo carries over."""
    fresh = fm.FFmpegManager()
    fresh._candidate_from_bundled = lambda name: None  # type: ignore[method-assign]
    fresh._candidate_from_system = lambda name: None  # type: ignore[method-assign]
    assert fresh._cache_dir == manager._cache_dir
    return fresh
