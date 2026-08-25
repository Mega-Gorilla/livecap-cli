"""Resource path resolution utilities."""
from __future__ import annotations

from contextlib import ExitStack
from importlib import resources
from pathlib import Path
from typing import Dict, Sequence


class ResourceLocator:
    """Resolve static resource paths with optional package fallbacks.

    検索順の**決定**は :mod:`livecap_cli.resources.configuration` の責務で、ここは
    与えられた順序で探すだけである (Issue #375)。
    """

    def __init__(self, *, search_roots: Sequence[Path]) -> None:
        """解決済みの検索順を受け取る。

        **env は読まず、root の組み立てもしない。** 検索順は
        :mod:`livecap_cli.resources.configuration` が決める — API 指定があれば
        ``LIVECAP_RESOURCE_ROOT`` を**検索順から除外する**という契約 (Issue #375)
        は、env をここで読んでいる限り実装できない。

        構築は :func:`livecap_cli.resources.graph.build_resource_graph` のみが行う。
        """
        self._stack = ExitStack()
        self._search_roots = list(search_roots)

        self._package_map: Dict[str, str] = {
            "src": "src",
            "config": "src.config",
            "languages": "languages",
            "html": "html",
            "fonts": "fonts",
        }

    def __del__(self) -> None:  # pragma: no cover - destructor safety
        self._stack.close()

    def resolve(self, relative_path: str) -> Path:
        """
        Resolve a resource path.

        Args:
            relative_path: resource path relative to project root or package.

        Raises:
            FileNotFoundError: when the resource cannot be located.
        """
        normalized = relative_path.strip("/").replace("\\", "/")

        for root in self._search_roots:
            candidate = root / normalized
            if candidate.exists():
                return candidate

        parts = normalized.split("/", 1)
        package_key = parts[0]
        remainder = parts[1] if len(parts) > 1 else ""

        package = self._package_map.get(package_key)
        if package:
            resource = resources.files(package)
            if remainder:
                for part in remainder.split("/"):
                    if part:
                        resource = resource.joinpath(part)
            path = self._stack.enter_context(resources.as_file(resource))
            resolved = Path(path)
            if resolved.exists():
                return resolved

        raise FileNotFoundError(f"Resource '{relative_path}' not found")
