"""Minimal CLI utilities for validating a LiveCap Core installation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

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
    cuda_available: bool
    cuda_device: str | None
    vad_backends: list[str]
    available_engines: list[str]
    i18n: I18nDiagnostics

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


def _get_cuda_info() -> tuple[bool, str | None]:
    """Get CUDA availability and device name."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            return True, device_name
        return False, None
    except ImportError:
        return False, None
    except Exception:
        return False, None


def _get_vad_backends() -> list[str]:
    """Get list of available VAD backend types."""
    try:
        from .vad.presets import get_available_presets
        # Extract unique VAD types from presets
        presets = get_available_presets()
        vad_types = sorted(set(vad_type for vad_type, _ in presets))
        return vad_types
    except ImportError:
        return []
    except Exception:
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

    cuda_available, cuda_device = _get_cuda_info()

    return DiagnosticReport(
        models_root=str(model_manager.models_root),
        cache_root=str(model_manager.cache_root),
        ffmpeg_path=_ensure_ffmpeg(ensure_ffmpeg),
        resource_root=resolved_root,
        cuda_available=cuda_available,
        cuda_device=cuda_device,
        vad_backends=_get_vad_backends(),
        available_engines=_get_available_engines(),
        i18n=diagnose_i18n(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="livecap-core",
        description="LiveCap Core installation diagnostics.",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show installation info (FFmpeg, CUDA, VAD backends, ASR engines, etc.)",
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

    if args.as_json:
        print(report.to_json())
        return 0

    # --info shows full diagnostics (same as default but may be extended)
    # Default output also shows diagnostics per Phase 2 plan
    print("LiveCap Core diagnostics:")
    print(f"  FFmpeg: {report.ffmpeg_path or 'not detected'}")
    print(f"  Models root: {report.models_root}")
    print(f"  Cache root: {report.cache_root}")

    # CUDA info
    if report.cuda_available:
        cuda_info = f"yes ({report.cuda_device})" if report.cuda_device else "yes"
        print(f"  CUDA available: {cuda_info}")
    else:
        print("  CUDA available: no")

    # VAD backends
    if report.vad_backends:
        print(f"  VAD backends: {', '.join(report.vad_backends)}")
    else:
        print("  VAD backends: none detected")

    # ASR engines
    if report.available_engines:
        print(f"  ASR engines: {', '.join(report.available_engines)}")
    else:
        print("  ASR engines: none detected")

    # Translator
    translator = report.i18n.translator
    if translator.registered:
        extras = f" extras={','.join(translator.extras)}" if translator.extras else ""
        name = translator.name or "translator"
        print(f"  Translator: {name}{extras}")
    else:
        print("  Translator: not registered (fallback only)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
