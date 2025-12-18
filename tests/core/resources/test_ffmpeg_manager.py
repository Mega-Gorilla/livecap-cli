import asyncio
import os
import stat

from livecap_cli.resources import get_ffmpeg_manager, reset_resource_managers
from livecap_cli.resources.ffmpeg_manager import FFmpegManager, FFmpegNotFoundError


def _make_fake_binary(path):
    path.write_text("#!/bin/sh\n")
    mode = path.stat().st_mode | stat.S_IEXEC
    path.chmod(mode)


def test_ffmpeg_manager_prefers_env(monkeypatch, tmp_path):
    reset_resource_managers()
    binary_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    probe_name = "ffprobe.exe" if os.name == "nt" else "ffprobe"

    custom_dir = tmp_path / "ffmpeg-custom"
    custom_dir.mkdir()
    ffmpeg_path = custom_dir / binary_name
    ffprobe_path = custom_dir / probe_name
    _make_fake_binary(ffmpeg_path)
    _make_fake_binary(ffprobe_path)

    monkeypatch.setenv("LIVECAP_FFMPEG_BIN", str(custom_dir))

    manager = get_ffmpeg_manager(force_reset=True)
    executable = manager.resolve_executable()
    probe = manager.resolve_probe()

    assert executable == ffmpeg_path
    assert probe == ffprobe_path

    reset_resource_managers()


def test_ffmpeg_manager_download_fallback(monkeypatch):
    reset_resource_managers()
    manager = get_ffmpeg_manager(force_reset=True)

    binary_name = "ffmpeg.exe" if manager._is_windows else "ffmpeg"
    probe_name = "ffprobe.exe" if manager._is_windows else "ffprobe"

    monkeypatch.setattr(manager, "_candidate_from_env", lambda name: None)
    monkeypatch.setattr(manager, "_candidate_from_packaged", lambda name: None)
    monkeypatch.setattr(manager, "_candidate_from_system", lambda name: None)

    def fake_download(self):
        ffmpeg_path = self._cache_dir / binary_name
        ffprobe_path = self._cache_dir / probe_name
        _make_fake_binary(ffmpeg_path)
        _make_fake_binary(ffprobe_path)
        self._cached_ffmpeg = ffmpeg_path
        self._cached_ffprobe = ffprobe_path

    monkeypatch.setattr(type(manager), "_download_static_build", fake_download, raising=False)

    executable = manager.ensure_executable()
    assert executable == manager._cache_dir / binary_name
    assert manager.resolve_probe() == manager._cache_dir / probe_name

    reset_resource_managers()


def test_configure_environment(monkeypatch, tmp_path):
    reset_resource_managers()
    manager = get_ffmpeg_manager(force_reset=True)

    binary_name = "ffmpeg.exe" if manager._is_windows else "ffmpeg"
    fake_dir = tmp_path / "ffmpeg-bin"
    fake_dir.mkdir()
    fake_bin = fake_dir / binary_name
    _make_fake_binary(fake_bin)

    monkeypatch.setattr(manager, "ensure_executable", lambda: fake_bin)

    original_path = os.environ.get("PATH", "")
    if manager._is_windows:
        monkeypatch.setenv("PATH", "")

    resolved = manager.configure_environment()
    assert resolved == fake_bin

    if manager._is_windows:
        assert os.environ["PATH"].split(os.pathsep)[0] == str(fake_dir)
    else:
        assert os.environ.get("PATH", "") == original_path

    reset_resource_managers()


def test_ffmpeg_manager_async_uses_cached_executable(tmp_path):
    manager = FFmpegManager()
    cached = tmp_path / ("ffmpeg.exe" if manager._is_windows else "ffmpeg")  # pylint: disable=protected-access
    _make_fake_binary(cached)
    manager._cached_ffmpeg = cached  # pylint: disable=protected-access

    async def run():
        return await manager.ensure_executable_async()

    resolved = asyncio.run(run())
    assert resolved == cached


def test_ffmpeg_manager_async_triggers_download(monkeypatch, tmp_path):
    manager = FFmpegManager()
    calls: list[str] = []

    async def fake_download() -> None:
        calls.append("download")
        cached = tmp_path / ("ffmpeg.exe" if manager._is_windows else "ffmpeg")  # pylint: disable=protected-access
        _make_fake_binary(cached)
        manager._cached_ffmpeg = cached  # pylint: disable=protected-access

    def fake_resolve():
        calls.append("resolve")
        if len(calls) == 1:
            raise FFmpegNotFoundError("missing")
        return manager._cached_ffmpeg  # type: ignore[return-value]

    monkeypatch.setattr(manager, "resolve_executable", fake_resolve)
    monkeypatch.setattr(manager, "_download_static_build_async", fake_download)

    async def run():
        return await manager.ensure_executable_async()

    resolved = asyncio.run(run())
    assert resolved == manager._cached_ffmpeg
    assert calls == ["resolve", "download", "resolve"]
