"""FFmpeg resolution helpers."""
from __future__ import annotations

import asyncio
import os
import platform
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Tuple

from .model_manager import ModelManager
from .resource_locator import ResourceLocator

__all__ = ["FFmpegManager", "FFmpegNotFoundError"]


class FFmpegNotFoundError(FileNotFoundError):
    """Raised when FFmpeg cannot be located."""


class FFmpegManager:
    """Resolve FFmpeg executables across platforms."""

    ENV_FFMPEG_PATH = "LIVECAP_FFMPEG_BIN"

    def __init__(self, locator: Optional[ResourceLocator] = None) -> None:
        self._locator = locator or ResourceLocator()
        self._model_manager = ModelManager()
        self._cache_dir = self._model_manager.cache_root / "ffmpeg"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cached_ffmpeg: Optional[Path] = None
        self._cached_ffprobe: Optional[Path] = None

    @property
    def _is_windows(self) -> bool:
        return platform.system().lower().startswith("win")

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

    def _candidate_from_packaged(self, binary_name: str) -> Optional[Path]:
        # First check cached downloads
        cached_candidate = self._cache_dir / binary_name
        if cached_candidate.exists():
            if (
                cached_candidate.is_file()
                and os.access(cached_candidate, os.X_OK)
                and cached_candidate.stat().st_size > 1024
            ):
                return cached_candidate
            # Detected a stub or invalid cached binary; ignore and allow fallback.
            cached_candidate.unlink(missing_ok=True)

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
            lambda: self._candidate_from_packaged(binary_name),
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
        """Locate ffprobe if available."""
        binary = "ffprobe.exe" if self._is_windows else "ffprobe"
        try:
            return self._resolve_binary(binary, "_cached_ffprobe")
        except FFmpegNotFoundError:
            return None

    def ensure_executable(self) -> Path:
        """
        Ensure an FFmpeg executable exists, attempting a download if necessary.
        """
        try:
            return self.resolve_executable()
        except FFmpegNotFoundError:
            self._download_static_build()
            return self.resolve_executable()

    async def ensure_executable_async(self) -> Path:
        """
        Asynchronous counterpart to :meth:`ensure_executable`.

        The heavy download/extraction step is awaited so UI event loops remain
        responsive.
        """
        try:
            return self.resolve_executable()
        except FFmpegNotFoundError:
            await self._download_static_build_async()
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

    def _download_static_build(self) -> None:
        archive_path, archive_ext = self._download_static_build_archive()
        self._extract_static_build(archive_path, archive_ext)

    async def _download_static_build_async(self) -> None:
        archive_path, archive_ext = await self._download_static_build_archive_async()
        await asyncio.to_thread(self._extract_static_build, archive_path, archive_ext)

    def _select_variant(self) -> Tuple[str, str]:
        system = platform.system()
        arch = platform.machine().lower()

        if system == "Windows":
            variant = "windows-64" if "64" in arch else "windows-32"
            return variant, ".zip"
        if system == "Linux":
            variant = "linux-64" if "64" in arch else "linux-32"
            return variant, ".zip"
        if system == "Darwin":
            variant = "osx-64"
            return variant, ".zip"

        raise FFmpegNotFoundError(f"Unsupported platform for automatic FFmpeg download: {system}")

    def _place_binaries(self, extracted_root: Path) -> None:
        binary_names = ["ffmpeg.exe", "ffprobe.exe"] if self._is_windows else ["ffmpeg", "ffprobe"]

        for binary_name in binary_names:
            source = next(extracted_root.rglob(binary_name), None)
            if not source:
                continue
            destination = self._cache_dir / binary_name
            shutil.copy2(source, destination)
            destination.chmod(destination.stat().st_mode | 0o111)
            if "ffmpeg" in binary_name:
                self._cached_ffmpeg = destination
            if "ffprobe" in binary_name:
                self._cached_ffprobe = destination

    def _download_static_build_archive(self) -> tuple[Path, str]:
        url, archive_name, archive_ext = self._static_build_spec()
        download_path = self._model_manager.download_file(url, filename=archive_name)
        return download_path, archive_ext

    async def _download_static_build_archive_async(self) -> tuple[Path, str]:
        url, archive_name, archive_ext = self._static_build_spec()
        download_path = await self._model_manager.download_file_async(url, filename=archive_name)
        return download_path, archive_ext

    def _static_build_spec(self) -> tuple[str, str, str]:
        variant, archive_ext = self._select_variant()
        archive_name = f"ffmpeg-{variant}{archive_ext}"
        url = f"https://github.com/ffbinaries/ffbinaries-prebuilt/releases/latest/download/{archive_name}"
        return url, archive_name, archive_ext

    def _extract_static_build(self, archive_path: Path, archive_ext: str) -> None:
        try:
            with self._model_manager.temporary_directory("ffmpeg-extract") as temp_dir:
                if archive_ext == ".zip":
                    with zipfile.ZipFile(archive_path, "r") as zf:
                        zf.extractall(temp_dir)
                else:
                    raise RuntimeError(f"Unsupported archive format: {archive_ext}")

                self._place_binaries(temp_dir)
        finally:
            archive_path.unlink(missing_ok=True)

    def _finalise_environment(self, executable: Path) -> Path:
        bin_dir = executable.parent

        if self._is_windows:
            current_path = os.environ.get("PATH", "")
            parts = current_path.split(os.pathsep) if current_path else []
            if str(bin_dir) not in parts:
                parts.insert(0, str(bin_dir))
                os.environ["PATH"] = os.pathsep.join(parts)

        return executable
