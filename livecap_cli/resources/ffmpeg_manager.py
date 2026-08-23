"""FFmpeg resolution helpers.

Resolution order (unchanged): ``LIVECAP_FFMPEG_BIN`` -> managed cache -> bundled
``ffmpeg-bin`` -> system PATH. Only when all of those miss does anything get
downloaded.

Two ideas from Issue #398 shape the rest of this module:

**Managed vs host-managed.** The managed cache (``<cache_root>/ffmpeg``) is ours:
we put pinned builds there, so we verify them and replace them when they do not
match. Everything else - ``LIVECAP_FFMPEG_BIN``, the bundled ``ffmpeg-bin``, and
PATH - belongs to whoever set it up, and is used as-is. We never overwrite a
binary a user chose.

**The pair is indivisible.** ``ffmpeg`` and ``ffprobe`` ship as separate archives
upstream. The old code downloaded one archive, looked for both binaries in it,
and ``continue``\\ d when ffprobe was missing - so a half-installed cache was
invisible and permanent, because ``ensure_executable()`` returns as soon as
ffmpeg is found. The managed cache is now valid only when *both* binaries match
their pinned digests; if either fails, neither is offered and the pair is
reinstalled together. A managed cache that is *invalid* is repaired rather than
skipped: it outranks PATH, so falling through would change which FFmpeg runs
because a file got corrupted. One that is merely *absent* is not a problem, and
the ordinary search continues.

Verifying two ~134 MB binaries costs ~190 ms warm and far more cold, which is too
much to pay on every start, so a stamp file records the ``(size, mtime_ns)`` the
digests were computed for. Matching stamps skip hashing. This detects staleness
and corruption; it is not a defence against someone with write access to the
cache directory.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import platform
import shutil
import stat
import subprocess
import tempfile
import threading
import zipfile
from pathlib import Path
from typing import Optional

from . import ffmpeg_pins
from .downloader import DownloadFailed, download_with_retry
from .ffmpeg_pins import PlatformSpec, UnsupportedPlatformError
from .model_manager import ModelManager
from .resource_locator import ResourceLocator

__all__ = ["FFmpegManager", "FFmpegNotFoundError", "FFmpegUpstreamUnavailable"]

logger = logging.getLogger(__name__)

#: Written next to the managed binaries. See the module docstring.
STAMP_NAME = ".livecap-ffmpeg.json"
STAMP_SCHEMA = 1

_STATE_OK = "ok"
#: Nothing of ours is installed. The ordinary search (bundled, PATH) applies.
_STATE_ABSENT = "absent"
#: Something of ours is installed and does not match the pins. Distinct from
#: absent on purpose: falling through to PATH here would silently change which
#: FFmpeg the application runs because a file got corrupted.
_STATE_INVALID = "invalid"
#: No pinned build exists for this machine, so the cache directory is not ours
#: to manage and nothing in it can be verified.
_STATE_UNMANAGED = "unmanaged"


class FFmpegNotFoundError(FileNotFoundError):
    """Raised when FFmpeg cannot be located."""


class FFmpegUpstreamUnavailable(FFmpegNotFoundError):
    """The pinned build could not be fetched: the network or host was down.

    Separate from its parent because it is the *only* install failure a caller
    may reasonably degrade around. A checksum mismatch, an unexpected archive,
    a binary that will not run, or a local write error all mean something is
    wrong with what we would install - never a reason to quietly use something
    else instead.
    """


class ChecksumMismatch(RuntimeError):
    """A downloaded file does not have its pinned digest. Never retried."""


class ArchiveContentError(RuntimeError):
    """An archive downloaded fine but does not contain what was expected."""


class ExecutableCheckFailed(RuntimeError):
    """The bytes are right but the binary will not run on this machine."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_digest(path: Path, expected: str, *, what: str) -> None:
    actual = _sha256_file(path)
    if actual != expected:
        raise ChecksumMismatch(
            f"SHA-256 mismatch for {what} ({path.name})\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            "The file is corrupt or has been replaced. Not retrying."
        )


def _extract_member(archive: Path, member_name: str, destination: Path) -> None:
    """Extract exactly ``member_name``, naming what is missing on failure.

    Deliberately not ``extractall``: only the one file we pinned is written, so
    an archive whose layout changed fails loud instead of scattering files.
    """
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
                "The upstream layout changed; update livecap_cli/resources/ffmpeg_manifest.json."
            )
        with bundle.open(match) as source, open(destination, "wb") as target:
            shutil.copyfileobj(source, target, 1 << 20)

    if os.name != "nt":
        mode = destination.stat().st_mode
        destination.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class FFmpegManager:
    """Resolve FFmpeg executables across platforms."""

    ENV_FFMPEG_PATH = "LIVECAP_FFMPEG_BIN"

    #: Guards the managed cache against concurrent installs. Class-level, not
    #: per-instance: every FFmpegManager in a process shares one cache
    #: directory, so an instance lock would not actually guard it. The async
    #: entry points funnel into the same synchronous install, so one lock covers
    #: both. Across processes there is no lock; the install is non-destructive
    #: until the final renames, which are atomic per file.
    _install_lock = threading.Lock()

    def __init__(self, locator: Optional[ResourceLocator] = None) -> None:
        self._locator = locator or ResourceLocator()
        self._model_manager = ModelManager()
        self._cache_dir = self._model_manager.cache_root / "ffmpeg"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cached_ffmpeg: Optional[Path] = None
        self._cached_ffprobe: Optional[Path] = None
        self._managed_state: Optional[str] = None

    @property
    def _is_windows(self) -> bool:
        return platform.system().lower().startswith("win")

    # ------------------------------------------------------------------
    # Candidate sources
    # ------------------------------------------------------------------

    def _candidate_from_env(self, binary_name: str) -> Optional[Path]:
        env_value = os.getenv(self.ENV_FFMPEG_PATH)
        if not env_value:
            return None

        candidate = Path(env_value).expanduser()
        if candidate.is_dir():
            candidate = candidate / binary_name

        if candidate.exists():
            return candidate
        return None

    def _candidate_from_managed_cache(self, binary_name: str) -> Optional[Path]:
        """Offer the managed binary only when the whole pinned pair is intact."""
        if self._managed_pair_state() != _STATE_OK:
            return None
        candidate = self._cache_dir / binary_name
        return candidate if candidate.is_file() else None

    def _candidate_from_bundled(self, binary_name: str) -> Optional[Path]:
        """Binaries shipped alongside the package: host-managed, used as-is."""
        try:
            bin_dir = self._locator.resolve("ffmpeg-bin")
        except FileNotFoundError:
            return None

        candidate = bin_dir / binary_name
        return candidate if candidate.exists() else None

    @staticmethod
    def _candidate_from_system(binary_name: str) -> Optional[Path]:
        which = shutil.which(binary_name)
        return Path(which) if which else None

    def _resolve_binary(self, binary_name: str, cache_attr: str) -> Path:
        cached = getattr(self, cache_attr)
        if cached and cached.exists():
            return cached

        finders = (
            lambda: self._candidate_from_env(binary_name),
            lambda: self._candidate_from_managed_cache(binary_name),
            lambda: self._candidate_from_bundled(binary_name),
            lambda: self._candidate_from_system(binary_name),
        )

        for finder in finders:
            path = finder()
            if path:
                setattr(self, cache_attr, path)
                return path

        raise FFmpegNotFoundError(
            f"{binary_name} not found. Install ffmpeg or set {self.ENV_FFMPEG_PATH}."
        )

    def resolve_executable(self) -> Path:
        """Locate an FFmpeg executable, caching the result."""
        binary = "ffmpeg.exe" if self._is_windows else "ffmpeg"
        return self._resolve_binary(binary, "_cached_ffmpeg")

    def resolve_probe(self) -> Optional[Path]:
        """Locate ffprobe if available.

        Stays optional on purpose: a host-managed FFmpeg install may legitimately
        lack ffprobe, and callers handle ``None``. The strictness lives in the
        managed install, which refuses to leave a half-installed pair behind.
        """
        binary = "ffprobe.exe" if self._is_windows else "ffprobe"
        try:
            return self._resolve_binary(binary, "_cached_ffprobe")
        except FFmpegNotFoundError:
            return None

    def _should_repair_managed_cache(self) -> bool:
        """Whether a broken managed cache must be fixed before searching on.

        A managed cache that is merely *absent* is not a problem: the ordinary
        search continues to the bundled directory and PATH. One that is
        *invalid* is different. It outranks PATH, so quietly skipping it would
        change which FFmpeg the application runs because a file got corrupted -
        an invisible switch to an unknown version. Repair what we manage.

        ``LIVECAP_FFMPEG_BIN`` is exempt: it outranks the managed cache anyway,
        so a download would be pure waste.
        """
        if self._cached_ffmpeg is not None and self._cached_ffmpeg.exists():
            # Already resolved; the source was decided on an earlier call.
            return False
        if self._managed_pair_state() != _STATE_INVALID:
            return False
        binary = "ffmpeg.exe" if self._is_windows else "ffmpeg"
        return self._candidate_from_env(binary) is None

    def _repair_managed_cache(self) -> Optional[FFmpegNotFoundError]:
        """Reinstall the pair. Returns the failure if degrading is allowed.

        Only an unreachable upstream is degraded around: a usable FFmpeg
        elsewhere beats failing outright while the network is down. Anything
        else - a bad checksum, an unexpected archive, a binary that will not
        run - is a statement about what we would have installed, and answering
        it by quietly running some other build is exactly the silent
        degradation this issue set out to remove.
        """
        try:
            self._install_pinned_pair()
        except FFmpegUpstreamUnavailable as exc:
            logger.warning(
                "Could not repair the managed FFmpeg cache (%s). "
                "Falling back to the bundled directory or PATH.",
                exc,
            )
            return exc
        return None

    def ensure_executable(self) -> Path:
        """
        Ensure an FFmpeg executable exists, attempting a download if necessary.
        """
        repair_failure = None
        if self._should_repair_managed_cache():
            repair_failure = self._repair_managed_cache()
        try:
            return self.resolve_executable()
        except FFmpegNotFoundError:
            if repair_failure is not None:
                # The install below is the one that just failed; repeating it
                # only doubles the wait before the same answer.
                raise repair_failure
            self._install_pinned_pair()
            return self.resolve_executable()

    async def ensure_executable_async(self) -> Path:
        """
        Asynchronous counterpart to :meth:`ensure_executable`.

        The heavy download/extraction step is awaited so UI event loops remain
        responsive.
        """
        repair_failure = None
        if self._should_repair_managed_cache():
            repair_failure = await asyncio.to_thread(self._repair_managed_cache)
        try:
            return self.resolve_executable()
        except FFmpegNotFoundError:
            if repair_failure is not None:
                raise repair_failure
            await asyncio.to_thread(self._install_pinned_pair)
            return self.resolve_executable()

    def configure_environment(self) -> Path:
        """
        Ensure PATH or related environment settings include the resolved FFmpeg.

        Returns:
            Path to the resolved executable.
        """
        executable = self.ensure_executable()
        return self._finalise_environment(executable)

    async def configure_environment_async(self) -> Path:
        """
        Asynchronous variant of :meth:`configure_environment`.
        """
        executable = await self.ensure_executable_async()
        return self._finalise_environment(executable)

    # ------------------------------------------------------------------
    # Managed cache: verification
    # ------------------------------------------------------------------

    def _stamp_path(self) -> Path:
        return self._cache_dir / STAMP_NAME

    def _read_stamp(self) -> dict:
        try:
            payload = json.loads(self._stamp_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if payload.get("schema") != STAMP_SCHEMA:
            return {}
        entries = payload.get("binaries")
        return entries if isinstance(entries, dict) else {}

    def _write_stamp(self, spec: PlatformSpec) -> None:
        payload = {
            "schema": STAMP_SCHEMA,
            "version": spec.version,
            "platform": spec.token,
            "binaries": {},
        }
        for role, binary in spec.binaries.items():
            info = (self._cache_dir / binary.name).stat()
            payload["binaries"][role] = {
                "name": binary.name,
                "sha256": binary.sha256,
                "size": info.st_size,
                "mtime_ns": info.st_mtime_ns,
            }

        # A fixed temporary name would collide with another process writing the
        # same cache: one would replace the other's file out from under it.
        handle, name = tempfile.mkstemp(
            prefix=STAMP_NAME + ".", suffix=".tmp", dir=self._cache_dir
        )
        temporary = Path(name)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2)
            os.replace(temporary, self._stamp_path())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def _managed_pair_state(self) -> str:
        if self._managed_state is None:
            self._managed_state = self._compute_managed_pair_state()
        return self._managed_state

    def _compute_managed_pair_state(self) -> str:
        try:
            spec = ffmpeg_pins.resolve_platform_spec()
        except UnsupportedPlatformError:
            # Nothing pinned for this machine, so nothing here can be checked.
            return _STATE_UNMANAGED

        stamp = self._read_stamp()
        stamp_is_current = True
        present = [
            binary.name
            for binary in spec.binaries.values()
            if (self._cache_dir / binary.name).is_file()
        ]
        if not present:
            return _STATE_ABSENT

        for role, binary in spec.binaries.items():
            path = self._cache_dir / binary.name
            if not path.is_file():
                # A half-installed pair is broken, not missing: the other half
                # is ours and has to be replaced along with it.
                logger.info(
                    "Managed FFmpeg cache is incomplete: %s is missing; the pair "
                    "will be reinstalled.",
                    binary.name,
                )
                return _STATE_INVALID

            info = path.stat()
            entry = stamp.get(role)
            if (
                isinstance(entry, dict)
                and entry.get("sha256") == binary.sha256
                and entry.get("size") == info.st_size
                and entry.get("mtime_ns") == info.st_mtime_ns
            ):
                continue

            stamp_is_current = False
            if _sha256_file(path) != binary.sha256:
                logger.info(
                    "Managed FFmpeg cache is stale: %s does not match the pinned "
                    "%s build; it will be reinstalled.",
                    binary.name,
                    spec.version,
                )
                return _STATE_INVALID

        if not stamp_is_current:
            # Contents are right but the stamp was missing or outdated; refresh
            # it so the next process does not have to hash again.
            try:
                self._write_stamp(spec)
            except OSError:
                logger.debug("Could not refresh the FFmpeg stamp file", exc_info=True)

        return _STATE_OK

    # ------------------------------------------------------------------
    # Managed cache: install
    # ------------------------------------------------------------------

    def _install_pinned_pair(self) -> None:
        """Download, verify and install the pinned ffmpeg/ffprobe pair.

        Both binaries are staged, verified and run before either is moved into
        place, so a failure up to that point leaves the managed cache untouched.
        Publishing itself is two renames and the stamp is written afterwards, so
        a failure between them is detected and repaired by the next verification
        rather than rolled back.

        Raises :class:`FFmpegUpstreamUnavailable` when the pinned archives could
        not be fetched and plain :class:`FFmpegNotFoundError` for everything
        else. That distinction decides whether a caller may degrade.
        """
        try:
            spec = ffmpeg_pins.resolve_platform_spec()
        except UnsupportedPlatformError as exc:
            raise FFmpegNotFoundError(str(exc)) from exc

        with self._install_lock:
            # Another thread may have installed while this one waited.
            self._managed_state = None
            if self._managed_pair_state() == _STATE_OK:
                return

            try:
                self._install_verified_pair(spec)
            except (
                DownloadFailed,
                ChecksumMismatch,
                ArchiveContentError,
                ExecutableCheckFailed,
                OSError,
            ) as exc:
                self._managed_state = None
                message = f"Could not install the pinned FFmpeg {spec.version} build.\n{exc}"
                if isinstance(exc, DownloadFailed) and exc.transient:
                    raise FFmpegUpstreamUnavailable(message) from exc
                raise FFmpegNotFoundError(message) from exc

    def _install_verified_pair(self, spec: PlatformSpec) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Staged inside the cache root so os.replace() stays on one volume, and
        # in a private directory so concurrent or abandoned installs cannot see
        # each other's files (#386 / #398 D2).
        with tempfile.TemporaryDirectory(
            prefix="ffmpeg-install-", dir=self._model_manager.cache_root
        ) as workspace:
            work_dir = Path(workspace)
            staged: dict[str, Path] = {}

            for role, archive in spec.archives.items():
                binary = spec.binaries[role]
                archive_path = work_dir / archive.asset

                logger.info("Downloading %s", archive.url)
                try:
                    download_with_retry(archive.url, archive_path, log=logger.warning)
                except DownloadFailed:
                    raise
                except OSError as exc:
                    # A permanent status (404, 403, ...) is not retried, so it
                    # arrives raw. Give it the same context a retried failure
                    # gets - which URL, and what the user can do instead - but
                    # mark it permanent so nobody waits it out.
                    raise DownloadFailed(archive.url, 1, exc, transient=False) from exc
                _verify_digest(archive_path, archive.sha256, what=f"{role} archive")

                target = work_dir / binary.name
                _extract_member(archive_path, binary.name, target)
                _verify_digest(target, binary.sha256, what=role)
                archive_path.unlink(missing_ok=True)
                staged[role] = target

            # Runnability is checked while everything is still staged. Doing it
            # after publishing would leave an unusable binary - and a stamp
            # blessing it - behind for every later run to trust.
            versions = {
                role: self._probe_version(path, role, spec) for role, path in staged.items()
            }

            # Everything verified: only now is the managed cache touched. This
            # is two renames, not one atomic step; see _write_stamp below.
            for role, source in staged.items():
                os.replace(source, self._cache_dir / spec.binaries[role].name)

        # Written last on purpose. If a rename above fails, the stamp still
        # describes the previous state, so the next verification hashes the
        # files, sees the mismatch, and reinstalls the pair.
        self._write_stamp(spec)
        self._managed_state = _STATE_OK
        self._cached_ffmpeg = self._cache_dir / spec.binaries["ffmpeg"].name
        self._cached_ffprobe = self._cache_dir / spec.binaries["ffprobe"].name
        logger.info(
            "Installed FFmpeg %s (%s): %s", spec.version, spec.token, versions.get("ffmpeg", "")
        )

    def _probe_version(self, executable: Path, role: str, spec: PlatformSpec) -> str:
        """Run the binary once. A correct digest is not proof it can execute."""
        try:
            completed = subprocess.run(
                [str(executable), "-version"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ExecutableCheckFailed(
                f"The pinned {role} {spec.version} ({spec.token}) has the expected "
                f"SHA-256 but could not be executed: {exc}"
            ) from exc

        output = completed.stdout or completed.stderr or ""
        first_line = output.splitlines()[0].strip() if output else ""
        if completed.returncode != 0:
            raise ExecutableCheckFailed(
                f"The pinned {role} {spec.version} ({spec.token}) has the expected "
                f"SHA-256 but '{role} -version' exited {completed.returncode}: {first_line}"
            )
        return first_line

    def _finalise_environment(self, executable: Path) -> Path:
        bin_dir = executable.parent

        if self._is_windows:
            current_path = os.environ.get("PATH", "")
            parts = current_path.split(os.pathsep) if current_path else []
            if str(bin_dir) not in parts:
                parts.insert(0, str(bin_dir))
                os.environ["PATH"] = os.pathsep.join(parts)

        return executable
