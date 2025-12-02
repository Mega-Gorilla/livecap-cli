"""Minimal CLI utilities for validating a LiveCap Core installation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict

from .i18n import I18nDiagnostics, diagnose as diagnose_i18n
from .resources import (
    get_ffmpeg_manager,
    get_model_manager,
    get_resource_locator,
)

__all__ = ["DiagnosticReport", "diagnose", "main"]


@dataclass
class DiagnosticReport:
    """Simple diagnostic payload returned by the CLI."""

    models_root: str
    cache_root: str
    ffmpeg_path: str | None
    resource_root: str | None
    i18n: I18nDiagnostics
    available_engines: list[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def _ensure_ffmpeg(ensure: bool) -> str | None:
    manager = get_ffmpeg_manager()
    if ensure:
        return str(manager.ensure_executable())
    try:
        return str(manager.resolve_executable())
    except Exception:
        return None


def _get_available_engines() -> list[str]:
    """Get list of available engine IDs."""
    try:
        from engines.metadata import EngineMetadata
        return list(EngineMetadata.get_all().keys())
    except ImportError:
        return []


def diagnose(
    *,
    ensure_ffmpeg: bool = False,
) -> DiagnosticReport:
    """Programmatic entry point mirroring the CLI behaviour."""
    model_manager = get_model_manager()
    resource_locator = get_resource_locator()

    try:
        resolved_root = str(resource_locator.resolve("."))
    except FileNotFoundError:
        resolved_root = None

    return DiagnosticReport(
        models_root=str(model_manager.models_root),
        cache_root=str(model_manager.cache_root),
        ffmpeg_path=_ensure_ffmpeg(ensure_ffmpeg),
        resource_root=resolved_root,
        i18n=diagnose_i18n(),
        available_engines=_get_available_engines(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="livecap-core",
        description="Inspect and validate a LiveCap Core installation.",
    )
    parser.add_argument(
        "--list-engines",
        action="store_true",
        help="List available ASR engines.",
    )
    parser.add_argument(
        "--ensure-ffmpeg",
        action="store_true",
        help="Attempt to download or locate an FFmpeg binary when not already available.",
    )
    parser.add_argument(
        "--as-json",
        action="store_true",
        help="Emit the diagnostic report as JSON for automation.",
    )
    args = parser.parse_args(argv)

    report = diagnose(ensure_ffmpeg=args.ensure_ffmpeg)

    if args.list_engines:
        try:
            from engines.metadata import EngineMetadata
            print("Available ASR Engines:")
            for engine_id, info in EngineMetadata.get_all().items():
                print(f"  {engine_id}:")
                print(f"    Name: {info.display_name}")
                print(f"    Languages: {', '.join(info.supported_languages)}")
                if info.default_params:
                    print(f"    Default params: {info.default_params}")
                print()
        except ImportError:
            print("Error: Could not import engines module")
            return 1

    if args.as_json:
        print(report.to_json())
    elif not args.list_engines:
        print("LiveCap Core diagnostics:")
        print(f"  Models root: {report.models_root}")
        print(f"  Cache root: {report.cache_root}")
        print(f"  FFmpeg path: {report.ffmpeg_path or 'not detected'}")
        print(f"  Available engines: {len(report.available_engines)}")
        translator = report.i18n.translator
        if translator.registered:
            extras = f" extras={','.join(translator.extras)}" if translator.extras else ""
            name = translator.name or "translator"
            print(f"  Translator: {name}{extras}")
        else:
            print("  Translator: not registered (fallback only)")
        if report.i18n.fallback_count:
            print(f"  i18n fallback keys: {report.i18n.fallback_count} registered")
        else:
            print("  i18n fallback keys: none registered")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
