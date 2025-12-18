"""Resource path resolution utilities."""
from __future__ import annotations

import os
from contextlib import ExitStack
from importlib import resources
from pathlib import Path
from typing import Dict, Iterable, Optional


class ResourceLocator:
    """Resolve static resource paths with optional package fallbacks."""

    ENV_RESOURCE_ROOT = "LIVECAP_RESOURCE_ROOT"

    def __init__(self, extra_roots: Optional[Iterable[str | Path]] = None) -> None:
        self._stack = ExitStack()
        self._source_root = Path(__file__).resolve().parents[2]
        self._project_root = self._source_root.parent
        env_root = os.getenv(self.ENV_RESOURCE_ROOT)

        self._search_roots = [
            Path(root).expanduser()
            for root in ([env_root] if env_root else [])
        ]
        self._search_roots.extend([self._project_root, self._source_root])

        if extra_roots:
            for root in extra_roots:
                self._search_roots.append(Path(root).expanduser())

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
