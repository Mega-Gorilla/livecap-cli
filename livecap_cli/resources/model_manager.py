"""Model storage utilities."""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import os
import tempfile
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional
from urllib.parse import urlparse

__all__ = ["ModelManager"]


class ModelManager:
    """Handle model and cache directory access.

    root の**解決**は :mod:`livecap_cli.resources.configuration` が行い、ここは
    解決済みの値を受け取って使うだけである (Issue #375)。

    Phase 1 公開仕様で保証するプロパティ / メソッド:
    - `models_root` / `cache_root`
    - `get_models_dir`
    - `get_temp_dir`
    - `temporary_directory`
    - `download_file`
    - `huggingface_cache`
    """

    def __init__(self, *, models_root: Path, cache_root: Path) -> None:
        """解決済みの root を受け取る。

        **env は読まない。** 優先順位の解決は
        :mod:`livecap_cli.resources.configuration` の責務で、ここが env を読むと
        freeze した configuration の外側で root が決まってしまう (Issue #375)。

        構築は :func:`livecap_cli.resources.graph.build_resource_graph` のみが
        行う。root の作成もここで起きる — ``configure_resources()`` は明示指定
        root しか検証せず、preview は filesystem を触らないため。
        """
        self._models_root = models_root
        self._cache_root = cache_root

        self._models_root.mkdir(parents=True, exist_ok=True)
        self._cache_root.mkdir(parents=True, exist_ok=True)

    @property
    def models_root(self) -> Path:
        """Return the root directory where models are stored."""
        return self._models_root

    @property
    def cache_root(self) -> Path:
        """Return the cache directory used for temporary data."""
        return self._cache_root

    def get_models_dir(self, engine_name: Optional[str] = None) -> Path:
        """
        Return a directory path for models.

        Args:
            engine_name: Optional engine identifier to scope the directory.
        """
        if engine_name:
            path = self._models_root / engine_name
        else:
            path = self._models_root
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_temp_dir(self, purpose: str = "runtime") -> Path:
        """Return a temp directory path for the given purpose."""
        path = self._cache_root / purpose
        path.mkdir(parents=True, exist_ok=True)
        return path

    def download_file(
        self,
        url: str,
        *,
        filename: Optional[str] = None,
        expected_sha256: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        """
        Download a file into the cache-managed download directory.

        Args:
            url: Source URL.
            filename: Optional filename override.
            expected_sha256: Optional checksum for verification.
            progress_callback: Callable receiving (downloaded_bytes, total_bytes).
        """
        download_dir = self.get_temp_dir("downloads")
        parsed = urlparse(url)
        name_from_url = Path(parsed.path).name or "download"
        target_name = filename or name_from_url
        destination = download_dir / target_name

        def _report(block_num: int, block_size: int, total_size: int):
            if progress_callback:
                downloaded = block_num * block_size
                progress_callback(min(downloaded, total_size if total_size > 0 else downloaded), total_size)

        urllib.request.urlretrieve(url, destination, reporthook=_report)

        if expected_sha256:
            self._verify_sha256(destination, expected_sha256)

        return destination

    async def download_file_async(
        self,
        url: str,
        *,
        filename: Optional[str] = None,
        expected_sha256: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        """
        Asynchronous wrapper around :meth:`download_file`.

        The download itself is executed in a worker thread so that event loops
        (e.g. Qt / asyncio) remain responsive. Progress callbacks that return an
        awaitable are scheduled back onto the calling event loop; synchronous
        callbacks are invoked directly from the worker thread.
        """

        loop = asyncio.get_running_loop()

        if progress_callback is None:
            callback_for_thread = None
        else:

            def callback_for_thread(downloaded: int, total: int) -> None:
                result = progress_callback(downloaded, total)
                if inspect.isawaitable(result):
                    asyncio.run_coroutine_threadsafe(result, loop)

        return await loop.run_in_executor(
            None,
            lambda: self.download_file(
                url,
                filename=filename,
                expected_sha256=expected_sha256,
                progress_callback=callback_for_thread,
            ),
        )

    def _verify_sha256(self, path: Path, expected: str) -> None:
        hasher = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if digest.lower() != expected.lower():
            raise ValueError(f"SHA256 mismatch for {path.name}: expected {expected}, got {digest}")

    @contextmanager
    def huggingface_cache(self) -> Iterator[Path]:
        """
        Context manager that points HF_HOME to a cache directory managed by the model manager.
        """
        cache_dir = self.cache_root / "huggingface"
        cache_dir.mkdir(parents=True, exist_ok=True)

        old_cache = os.environ.get("HF_HOME")
        os.environ["HF_HOME"] = str(cache_dir)
        try:
            yield cache_dir
        finally:
            if old_cache is None:
                os.environ.pop("HF_HOME", None)
            else:
                os.environ["HF_HOME"] = old_cache

    @contextmanager
    def temporary_directory(self, purpose: str = "downloads") -> Iterator[Path]:
        """
        Provide a temporary directory within the cache tree.
        """
        base = self.get_temp_dir(purpose)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            yield Path(tmp)
