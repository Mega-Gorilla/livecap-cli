"""CLI for livecap-cli - High-performance speech transcription."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from typing import Any

from .i18n import I18nDiagnostics, diagnose as diagnose_i18n
from .resources import (
    get_ffmpeg_manager,
    get_model_manager,
    get_resource_locator,
)

__all__ = ["DiagnosticReport", "diagnose", "main"]


@dataclass
class DiagnosticReport:
    """Diagnostic payload for the info command."""

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
        from livecap_cli.engines.metadata import EngineMetadata
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
        presets = get_available_presets()
        vad_types = sorted(set(vad_type for vad_type, _ in presets))
        return vad_types
    except ImportError:
        return []
    except Exception:
        return []


def diagnose(*, ensure_ffmpeg: bool = False) -> DiagnosticReport:
    """Programmatic entry point for diagnostics."""
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


# =============================================================================
# Subcommand: info
# =============================================================================

def cmd_info(args: argparse.Namespace) -> int:
    """Show installation diagnostics."""
    report = diagnose(ensure_ffmpeg=args.ensure_ffmpeg)

    if args.as_json:
        print(report.to_json())
        return 0

    print("livecap-cli diagnostics:")
    print(f"  FFmpeg: {report.ffmpeg_path or 'not detected'}")
    print(f"  Models root: {report.models_root}")
    print(f"  Cache root: {report.cache_root}")

    if report.cuda_available:
        cuda_info = f"yes ({report.cuda_device})" if report.cuda_device else "yes"
        print(f"  CUDA available: {cuda_info}")
    else:
        print("  CUDA available: no")

    if report.vad_backends:
        print(f"  VAD backends: {', '.join(report.vad_backends)}")
    else:
        print("  VAD backends: none detected")

    if report.available_engines:
        print(f"  ASR engines: {', '.join(report.available_engines)}")
    else:
        print("  ASR engines: none detected")

    translator = report.i18n.translator
    if translator.registered:
        extras = f" extras={','.join(translator.extras)}" if translator.extras else ""
        name = translator.name or "translator"
        print(f"  Translator: {name}{extras}")
    else:
        print("  Translator: not registered (fallback only)")

    return 0


# =============================================================================
# Subcommand: devices
# =============================================================================

def cmd_devices(args: argparse.Namespace) -> int:
    """List available audio input devices."""
    try:
        from livecap_cli import MicrophoneSource
        devices = MicrophoneSource.list_devices()

        if not devices:
            print("No audio input devices found.")
            return 0

        for dev in devices:
            default = " (default)" if dev.is_default else ""
            print(f"[{dev.index}] {dev.name}{default}")

        return 0
    except ImportError as e:
        print(f"Error: Could not import MicrophoneSource: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error listing devices: {e}", file=sys.stderr)
        return 1


# =============================================================================
# Subcommand: engines
# =============================================================================

def cmd_engines(args: argparse.Namespace) -> int:
    """List available ASR engines."""
    try:
        from livecap_cli.engines.metadata import EngineMetadata

        engines = EngineMetadata.get_all()
        if not engines:
            print("No ASR engines found.")
            return 0

        for engine_id, meta in engines.items():
            device_info = ", ".join(meta.device_support) if meta.device_support else "unknown"
            print(f"{engine_id}: {meta.display_name} [{device_info}]")

        return 0
    except ImportError as e:
        print(f"Error: Could not import EngineMetadata: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error listing engines: {e}", file=sys.stderr)
        return 1


# =============================================================================
# Subcommand: translators
# =============================================================================

def cmd_translators(args: argparse.Namespace) -> int:
    """List available translators."""
    try:
        from livecap_cli.translation.metadata import TranslatorMetadata

        translators = TranslatorMetadata.get_all()
        if not translators:
            print("No translators found.")
            return 0

        for tid, info in translators.items():
            gpu = " (GPU)" if info.requires_gpu else ""
            print(f"{tid}: {info.display_name}{gpu}")

        return 0
    except ImportError as e:
        print(f"Error: Could not import TranslatorMetadata: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error listing translators: {e}", file=sys.stderr)
        return 1


# =============================================================================
# Subcommand: transcribe
# =============================================================================

def _map_device(device: str) -> str:
    """Map CLI device names to internal names."""
    if device == "gpu":
        return "cuda"
    return device


def cmd_transcribe(args: argparse.Namespace) -> int:
    """Transcribe audio from microphone or file."""
    # Check for required arguments
    if args.realtime:
        if args.mic is None:
            print("Error: --mic is required for realtime transcription", file=sys.stderr)
            return 1
        return _transcribe_realtime(args)
    elif args.input_file:
        return _transcribe_file(args)
    else:
        print("Error: Either --realtime --mic <id> or <input_file> is required", file=sys.stderr)
        return 1


def _get_vad_processor(language: str, vad_backend: str):
    """Create VAD processor based on --vad option."""
    from livecap_cli.vad import VADProcessor

    if vad_backend == "auto":
        try:
            return VADProcessor.from_language(language)
        except ValueError as e:
            # Fallback to Silero for unsupported languages
            print(f"Warning: {e}. Using Silero VAD.", file=sys.stderr)
            return VADProcessor()
    elif vad_backend == "silero":
        return VADProcessor()
    elif vad_backend == "tenvad":
        from livecap_cli.vad.backends import TenVAD
        return VADProcessor(backend=TenVAD())
    elif vad_backend == "webrtc":
        from livecap_cli.vad.backends import WebRTCVAD
        return VADProcessor(backend=WebRTCVAD())
    else:
        print(f"Warning: Unknown VAD backend '{vad_backend}'. Using Silero.", file=sys.stderr)
        return VADProcessor()


def _transcribe_realtime(args: argparse.Namespace) -> int:
    """Realtime transcription from microphone."""
    try:
        from livecap_cli import StreamTranscriber, MicrophoneSource
        from livecap_cli.engines import EngineFactory

        device = _map_device(args.device)

        # Create engine
        engine_kwargs: dict[str, Any] = {}
        # model_size is only applicable to whispers2t
        if args.engine == "whispers2t" and args.model_size:
            engine_kwargs["model_size"] = args.model_size

        print(f"Loading engine: {args.engine} (device={device})...", file=sys.stderr)
        engine = EngineFactory.create_engine(args.engine, device=device, **engine_kwargs)
        engine.load_model()

        # Create VAD processor
        vad_processor = _get_vad_processor(args.language, args.vad)

        # Start transcription
        print(f"Starting realtime transcription (mic={args.mic}, language={args.language})...", file=sys.stderr)
        print("Press Ctrl+C to stop.\n", file=sys.stderr)

        with StreamTranscriber(engine=engine, vad_processor=vad_processor) as transcriber:
            with MicrophoneSource(device_index=args.mic) as mic:
                try:
                    for result in transcriber.transcribe_sync(mic):
                        print(f"[{result.start_time:.2f}s] {result.text}")
                except KeyboardInterrupt:
                    print("\nStopping...", file=sys.stderr)

        return 0
    except ImportError as e:
        print(f"Error: Missing dependency: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error during transcription: {e}", file=sys.stderr)
        return 1


def _transcribe_file(args: argparse.Namespace) -> int:
    """Transcribe from file."""
    try:
        from livecap_cli.transcription import FileTranscriptionPipeline
        from livecap_cli.engines import EngineFactory

        device = _map_device(args.device)

        # Create engine
        engine_kwargs: dict[str, Any] = {}
        # model_size is only applicable to whispers2t
        if args.engine == "whispers2t" and args.model_size:
            engine_kwargs["model_size"] = args.model_size

        print(f"Loading engine: {args.engine} (device={device})...", file=sys.stderr)
        engine = EngineFactory.create_engine(args.engine, device=device, **engine_kwargs)
        engine.load_model()

        # Create translator if specified
        translator = None
        if args.translate:
            from livecap_cli.translation import TranslatorFactory

            print(f"Loading translator: {args.translate}...", file=sys.stderr)
            translator = TranslatorFactory.create_translator(args.translate)
            translator.initialize()

        # Create pipeline
        pipeline = FileTranscriptionPipeline(engine=engine)

        # Transcribe
        print(f"Transcribing: {args.input_file}...", file=sys.stderr)
        result = pipeline.transcribe(
            args.input_file,
            language=args.language,
            translator=translator,
            source_lang=args.language if translator else None,
            target_lang=args.target_lang if translator else None,
        )

        # Output
        if args.output:
            # Write to file (SRT format)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(result.to_srt())
            print(f"Output written to: {args.output}", file=sys.stderr)
        else:
            # Print to stdout
            for segment in result.segments:
                print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")

        return 0
    except ImportError as e:
        print(f"Error: Missing dependency: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError:
        print(f"Error: File not found: {args.input_file}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error during transcription: {e}", file=sys.stderr)
        return 1


# =============================================================================
# Main entry point
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="livecap-cli",
        description="High-performance speech transcription CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # info command
    info_parser = subparsers.add_parser("info", help="Show installation diagnostics")
    info_parser.add_argument(
        "--ensure-ffmpeg",
        action="store_true",
        help="Attempt to download or locate an FFmpeg binary",
    )
    info_parser.add_argument(
        "--as-json",
        action="store_true",
        help="Output as JSON",
    )
    info_parser.set_defaults(func=cmd_info)

    # devices command
    devices_parser = subparsers.add_parser("devices", help="List audio input devices")
    devices_parser.set_defaults(func=cmd_devices)

    # engines command
    engines_parser = subparsers.add_parser("engines", help="List available ASR engines")
    engines_parser.set_defaults(func=cmd_engines)

    # translators command
    translators_parser = subparsers.add_parser("translators", help="List available translators")
    translators_parser.set_defaults(func=cmd_translators)

    # transcribe command
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe audio")
    transcribe_parser.add_argument(
        "input_file",
        nargs="?",
        help="Input audio/video file",
    )
    transcribe_parser.add_argument(
        "-o", "--output",
        help="Output file (SRT format)",
    )
    transcribe_parser.add_argument(
        "--realtime",
        action="store_true",
        help="Enable realtime transcription mode",
    )
    transcribe_parser.add_argument(
        "--mic",
        type=int,
        help="Microphone device index (use 'devices' command to list)",
    )
    transcribe_parser.add_argument(
        "--engine",
        default="whispers2t",
        help="ASR engine ID (default: whispers2t)",
    )
    transcribe_parser.add_argument(
        "--device",
        choices=["auto", "gpu", "cpu"],
        default="auto",
        help="Device to use (default: auto)",
    )
    transcribe_parser.add_argument(
        "--language",
        default="ja",
        help="Input language code (default: ja)",
    )
    transcribe_parser.add_argument(
        "--model-size",
        default="base",
        help="Model size for WhisperS2T (default: base)",
    )
    transcribe_parser.add_argument(
        "--vad",
        choices=["auto", "silero", "tenvad", "webrtc"],
        default="auto",
        help="VAD backend (default: auto)",
    )
    transcribe_parser.add_argument(
        "--translate",
        help="Translator ID (e.g., google, opus_mt, riva_instruct)",
    )
    transcribe_parser.add_argument(
        "--target-lang",
        default="en",
        help="Target language for translation (default: en)",
    )
    transcribe_parser.set_defaults(func=cmd_transcribe)

    args = parser.parse_args(argv)

    # No command specified - show help
    if args.command is None:
        parser.print_help()
        return 0

    # Execute the command
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
