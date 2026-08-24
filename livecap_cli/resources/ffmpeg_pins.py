"""Pinned FFmpeg builds: which one this machine needs, and what it must hash to.

Issue #398. The manifest (``ffmpeg_manifest.json``, packaged next to this module)
is the single source of truth shared with the CI setup action; see its
``_comment`` block.

Platform selection used to be ``"64" in platform.machine()``, which sent an
x86-64 build to every ``aarch64`` machine, and a 32-bit x86 build to ``armv7l``.
Substring matching cannot express "instruction set", so this module uses an
explicit table and fails loud for anything not in it.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Optional

__all__ = [
    "ArchiveSpec",
    "BinarySpec",
    "PlatformSpec",
    "UnsupportedPlatformError",
    "load_manifest",
    "pinned_version",
    "resolve_platform_spec",
    "resolve_platform_token",
]

MANIFEST_NAME = "ffmpeg_manifest.json"

#: ``(system, machine)`` -> upstream ffbinaries token, both lower-cased.
#:
#: Absent on purpose (#398 D4): ``win-32`` has no upstream asset at all; macOS on
#: arm64 has none either and we will not run the Intel build under Rosetta 2
#: without saying so; the Linux arm assets exist but are left unpinned because we
#: have no way to verify them.
_PLATFORM_TABLE = {
    ("windows", "x86_64"): "win-64",
    ("linux", "x86_64"): "linux-64",
    ("darwin", "x86_64"): "macos-64",
}

#: Machines we recognise but deliberately do not serve, with the reason. Keeping
#: these separate from "unknown" is what makes the error message actionable.
_KNOWN_UNSUPPORTED = {
    ("windows", "arm64"): "no upstream Windows arm64 build exists",
    ("windows", "x86"): "no upstream 32-bit Windows build exists",
    ("windows", "i686"): "no upstream 32-bit Windows build exists",
    ("darwin", "arm64"): (
        "no upstream macOS arm64 build exists, and the Intel build would run "
        "under Rosetta 2 without saying so"
    ),
    ("linux", "aarch64"): "the Linux arm64 build is not pinned yet",
    ("linux", "arm64"): "the Linux arm64 build is not pinned yet",
    ("linux", "armv7l"): "the 32-bit Linux arm build is not pinned yet",
    ("linux", "i686"): "the 32-bit Linux x86 build is not pinned yet",
}

_INSTALL_HINT = {
    "darwin": "brew install ffmpeg",
    "linux": "your distribution's package manager, e.g. apt install ffmpeg",
    "windows": "winget install Gyan.FFmpeg",
}


class UnsupportedPlatformError(RuntimeError):
    """This machine has no pinned build, and we will not guess one."""


@dataclass(frozen=True)
class ArchiveSpec:
    """One downloadable archive and the digest it must have."""

    asset: str
    sha256: str
    url: str


@dataclass(frozen=True)
class BinarySpec:
    """One executable inside an archive and the digest it must have."""

    name: str
    sha256: str


@dataclass(frozen=True)
class PlatformSpec:
    """Everything needed to install and verify the pair for one platform."""

    token: str
    version: str
    archives: dict[str, ArchiveSpec]
    binaries: dict[str, BinarySpec]

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(self.binaries)


@lru_cache(maxsize=1)
def load_manifest() -> dict:
    """Read the packaged manifest. Cached: it never changes at runtime."""
    resource = files(__package__).joinpath(MANIFEST_NAME)
    return json.loads(resource.read_text(encoding="utf-8"))


def pinned_version() -> str:
    return load_manifest()["version"]


def _normalise(system: Optional[str], machine: Optional[str]) -> tuple[str, str]:
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()
    # AMD64/x64/x86_64 are the same instruction set. i386..i686 are NOT - that
    # conflation is exactly what the old ``"64" in machine`` check got wrong.
    if machine in ("x64", "amd64", "x86_64"):
        machine = "x86_64"
    elif machine in ("i386", "i486", "i586", "i686", "x86"):
        machine = "x86" if system == "windows" else "i686"
    elif machine in ("aarch64", "arm64") and system in ("windows", "darwin"):
        machine = "arm64"
    return system, machine


def resolve_platform_token(
    system: Optional[str] = None, machine: Optional[str] = None
) -> str:
    """Return the ffbinaries token for this machine, or fail loud."""
    key = _normalise(system, machine)
    token = _PLATFORM_TABLE.get(key)
    if token is not None:
        return token

    reason = _KNOWN_UNSUPPORTED.get(key, "no pinned build is available")
    hint = _INSTALL_HINT.get(key[0], "your platform's package manager")
    raise UnsupportedPlatformError(
        f"Automatic FFmpeg download is not supported on {key[0]}/{key[1]}: {reason}.\n"
        f"  Install FFmpeg with {hint}, or set LIVECAP_FFMPEG_BIN to a directory "
        "holding ffmpeg/ffprobe.\n"
        "  A binary already on PATH is picked up automatically."
    )


def resolve_platform_spec(
    system: Optional[str] = None, machine: Optional[str] = None
) -> PlatformSpec:
    """Resolve this machine to its pinned archives and binaries."""
    manifest = load_manifest()
    token = resolve_platform_token(system, machine)
    entry = manifest["platforms"][token]
    base_url = manifest["base_url"]

    return PlatformSpec(
        token=token,
        version=manifest["version"],
        archives={
            role: ArchiveSpec(
                asset=spec["asset"],
                sha256=spec["sha256"],
                url=f"{base_url}/{spec['asset']}",
            )
            for role, spec in entry["archives"].items()
        },
        binaries={
            role: BinarySpec(name=spec["name"], sha256=spec["sha256"])
            for role, spec in entry["binaries"].items()
        },
    )
