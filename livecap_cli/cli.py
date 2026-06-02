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
        vad_types = sorted(set(vad_type for vad_type, _, _ in presets))
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

        # Windows では WASAPI デバイスのみ表示（重複削減・低レイテンシ）
        devices = MicrophoneSource.list_devices(prefer_wasapi=True)

        if not devices:
            print("No audio input devices found.")
            return 0

        for dev in devices:
            default = " (default)" if dev.is_default else ""
            host_api = f" [{dev.host_api}]" if dev.host_api else ""
            print(f"[{dev.index}] {dev.name}{default}{host_api}")

        return 0
    except ImportError as e:
        print(f"Error: Could not import MicrophoneSource: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error listing devices: {e}", file=sys.stderr)
        return 1


# =============================================================================
# Subcommand: levels
# =============================================================================


def cmd_levels(args: argparse.Namespace) -> int:
    """Monitor microphone input levels in real time."""
    try:
        import time

        import numpy as np

        from livecap_cli import MicrophoneSource
        from livecap_cli.audio import (
            ENGINE_MIN_RMS_SAFETY_MARGIN_DB,
            PEAK_SAFETY_MARGIN_DB,
            analyze_noise_samples,
        )

        # Windows cp932 等で Unicode バー文字が encode できない環境向け fallback
        stream_encoding = getattr(sys.stderr, "encoding", None) or "utf-8"
        try:
            "█░—".encode(stream_encoding)
            bar_full, bar_empty, dash = "█", "░", "—"
        except (UnicodeEncodeError, LookupError):
            bar_full, bar_empty, dash = "#", "-", "-"

        with MicrophoneSource(device=args.mic) as mic:
            mic.start()

            if args.json:
                if args.duration is not None:
                    intro = f"Sampling mic {args.mic} for {args.duration:.1f}s..."
                else:
                    intro = f"Sampling mic {args.mic} until Ctrl+C..."
                print(intro, file=sys.stderr)
            else:
                print(
                    f"Monitoring mic {args.mic}... Press Ctrl+C to stop.\n",
                    file=sys.stderr,
                )
                print(
                    "  -60dB       -40dB       -20dB        0dB",
                    file=sys.stderr,
                )
                print(
                    "    |           |           |           |",
                    file=sys.stderr,
                )

            all_rms_levels: list[float] = []
            all_peak_levels: list[float] = []
            start_time = time.monotonic()
            try:
                while True:
                    if args.duration is not None:
                        if time.monotonic() - start_time >= args.duration:
                            break
                    chunk = mic.read(timeout=0.2)
                    if chunk is None:
                        continue
                    rms = float(np.sqrt(np.mean(chunk**2)))
                    peak = float(np.max(np.abs(chunk)))
                    rms_db = 20 * np.log10(max(rms, 1e-10))
                    peak_db = 20 * np.log10(max(peak, 1e-10))
                    all_rms_levels.append(rms_db)
                    all_peak_levels.append(peak_db)

                    if not args.json:
                        bar_width = 40
                        pos = int(
                            max(0, min(bar_width, (rms_db + 60) / 60 * bar_width))
                        )
                        bar = bar_full * pos + bar_empty * (bar_width - pos)
                        print(
                            f"\r    {bar}  {rms_db:6.1f} dB",
                            end="",
                            flush=True,
                            file=sys.stderr,
                        )
            except KeyboardInterrupt:
                print("", file=sys.stderr)

            if not all_rms_levels:
                print("Error: No samples collected.", file=sys.stderr)
                return 1

            elapsed = time.monotonic() - start_time
            sample_rate_hz = len(all_rms_levels) / max(elapsed, 1e-6)
            engine_margin = (
                args.engine_min_rms_margin
                if getattr(args, "engine_min_rms_margin", None) is not None
                else ENGINE_MIN_RMS_SAFETY_MARGIN_DB
            )
            analysis = analyze_noise_samples(
                all_rms_levels,
                all_peak_levels,
                sample_rate_hz=sample_rate_hz,
                engine_min_rms_margin_db=engine_margin,
            )

            if args.json:
                print(json.dumps(asdict(analysis), indent=2))
            else:
                print(
                    f"Noise floor:    ~{analysis.noise_floor_db:.1f} dB "
                    f"(RMS 25%ile)",
                    file=sys.stderr,
                )
                print(
                    f"Noise RMS p95:  ~{analysis.noise_rms_p95_db:.1f} dB "
                    f"(RMS 95%ile)",
                    file=sys.stderr,
                )
                print(
                    f"Peak p95:       ~{analysis.peak_p95_db:.1f} dB "
                    f"(|x|.max() 95%ile, threshold の基準)",
                    file=sys.stderr,
                )
                print(
                    f"Suggested --noise-gate-threshold: "
                    f"{analysis.suggested_threshold_db:.0f} dB "
                    f"(= peak_p95 + {PEAK_SAFETY_MARGIN_DB:g}; "
                    f"per-sample peak unit)",
                    file=sys.stderr,
                )
                print(
                    f"Suggested --engine-min-rms:       "
                    f"{analysis.suggested_engine_min_rms_dbfs:.0f} dB "
                    f"(= noise_rms_p95 + {engine_margin:g}; "
                    f"per-frame RMS unit, #292 EnergyGate)",
                    file=sys.stderr,
                )
                print(
                    f"  (Danger zone: {analysis.danger_zone[0]:.0f} ~ "
                    f"{analysis.danger_zone[1]:.0f} dB {dash} "
                    "RMS-unit; avoid manually setting thresholds here)",
                    file=sys.stderr,
                )
                print(
                    "",
                    file=sys.stderr,
                )
                print(
                    "Note: The suggested value is a calibrated starting "
                    "point for the current NoiseGate "
                    "(auto hysteresis + hard-mute). Very quiet speech or "
                    "extreme low-SNR conditions may still need manual "
                    "tuning.",
                    file=sys.stderr,
                )

        return 0
    except ImportError as e:
        print(f"Error: Missing dependency: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error monitoring levels: {e}", file=sys.stderr)
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


def _parse_engine_min_rms(value: str) -> float:
    """argparse type for --engine-min-rms.

    Accepts numeric dBFS values or the strings ``off`` / ``disabled`` /
    ``none`` (case-insensitive) which map to ``float("-inf")``.

    Note:
        argparse rejects bare ``-inf`` as a value because leading-``-`` is
        parsed as another option. Use ``--engine-min-rms=-inf`` (equals form)
        or ``--engine-min-rms off`` instead.
    """
    if value.lower() in ("off", "disabled", "none"):
        return float("-inf")
    try:
        return float(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"invalid value for --engine-min-rms: {value!r} "
            f"(expected number, 'off', 'disabled', or 'none')"
        ) from e


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


def _get_vad_processor(language: str, vad_backend: str, engine: str | None = None):
    """Create VAD processor based on --vad option."""
    from livecap_cli.vad import VADProcessor

    if vad_backend == "auto":
        try:
            return VADProcessor.from_language(language, engine=engine)
        except ValueError as e:
            # Fallback to Silero for unsupported languages
            print(f"Warning: {e}. Using Silero VAD.", file=sys.stderr)
            return VADProcessor()
    elif vad_backend in ("silero", "tenvad", "webrtc"):
        return VADProcessor.from_preset(vad_backend, language, engine=engine)
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
        vad_processor = _get_vad_processor(args.language, args.vad, engine=args.engine)

        # Create noise gate (if enabled)
        noise_gate = None
        if args.noise_gate:
            from livecap_cli.audio.noise_gate import NoiseGate

            noise_gate = NoiseGate(
                threshold_db=args.noise_gate_threshold,
                close_threshold_db=args.noise_gate_close_threshold,
                attack_ms=args.noise_gate_attack,
                release_ms=args.noise_gate_release,
                noise_floor_db=(
                    args.noise_gate_floor
                    if args.noise_gate_floor is not None
                    else float("-inf")
                ),
            )

        # Start transcription
        print(f"Starting realtime transcription (mic={args.mic}, language={args.language})...", file=sys.stderr)
        print("Press Ctrl+C to stop.\n", file=sys.stderr)

        with StreamTranscriber(
            engine=engine,
            vad_processor=vad_processor,
            noise_gate=noise_gate,
            engine_min_rms_dbfs=args.engine_min_rms,
            engine_energy_metric=args.engine_energy_metric,
            engine_energy_frame_ms=args.engine_energy_frame_ms,
        ) as transcriber:
            with MicrophoneSource(device=args.mic) as mic:
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

    # levels command
    levels_parser = subparsers.add_parser(
        "levels", help="Monitor microphone input levels"
    )
    levels_parser.add_argument(
        "--mic",
        type=int,
        default=0,
        help="Microphone device index (default: 0)",
    )
    levels_parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Auto-stop after N seconds (default: run until Ctrl+C)",
    )
    levels_parser.add_argument(
        "--json",
        action="store_true",
        help="Output analysis as JSON to stdout (suppresses bar chart)",
    )
    levels_parser.add_argument(
        "--engine-min-rms-margin",
        type=float,
        default=None,
        help=(
            "Safety margin (dB) for suggested_engine_min_rms_dbfs "
            "(#292 EnergyGate). "
            "Default: 6.0. Larger value = more aggressive engine-input gate."
        ),
    )
    levels_parser.set_defaults(func=cmd_levels)

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
    transcribe_parser.add_argument(
        "--noise-gate",
        action="store_true",
        help="Enable noise gate (reduces environmental noise before VAD)",
    )
    transcribe_parser.add_argument(
        "--noise-gate-threshold",
        type=float,
        default=-35,
        help="Noise gate threshold in dB (default: -35)",
    )
    transcribe_parser.add_argument(
        "--noise-gate-attack",
        type=float,
        default=0.5,
        help="Noise gate attack time in ms (default: 0.5)",
    )
    transcribe_parser.add_argument(
        "--noise-gate-release",
        type=float,
        default=100,
        help="Noise gate release time in ms (default: 100)",
    )
    transcribe_parser.add_argument(
        "--noise-gate-close-threshold",
        type=float,
        default=None,
        help=(
            "Noise gate close threshold in dB for hysteresis "
            "(default: open threshold - 6 dB; "
            "pass the same value as --noise-gate-threshold to disable hysteresis)"
        ),
    )
    transcribe_parser.add_argument(
        "--noise-gate-floor",
        type=float,
        default=None,
        help=(
            "Noise floor in dB when gate is closed "
            "(default: hard-mute / -inf; "
            "pass e.g. -60 for legacy soft-mute behavior)"
        ),
    )
    # === #292 EnergyGate (engine-input low-energy guard) ===
    transcribe_parser.add_argument(
        "--engine-min-rms",
        type=_parse_engine_min_rms,
        default=-45.0,
        help=(
            "Engine-input low-energy gate threshold in dBFS "
            "(default: -45.0). Use 'off' or '=-inf' to disable. "
            "NOTE: argparse cannot accept '-inf' with a space; "
            "use '=-inf' or 'off' instead. "
            "This threshold is per-segment RMS-unit; different physical "
            "quantity from --noise-gate-threshold (per-sample peak). "
            "Do not share values across the two gates."
        ),
    )
    transcribe_parser.add_argument(
        "--engine-energy-metric",
        choices=("max_frame_rms", "whole_rms", "p95_frame_rms", "top3_frame_rms"),
        default="max_frame_rms",
        help=(
            "Per-segment energy metric for EnergyGate "
            "(default: max_frame_rms). "
            "max_frame_rms: robust to VAD padding dilution (recommended). "
            "whole_rms: aggressive, may false-drop padded short utterances. "
            "p95_frame_rms: balanced. "
            "top3_frame_rms: resistant to single-frame transient false-pass."
        ),
    )
    transcribe_parser.add_argument(
        "--engine-energy-frame-ms",
        type=float,
        default=32.0,
        help=(
            "Frame size (ms) for frame-based energy metrics "
            "(default: 32, typical range: 10-200). "
            "Ignored when --engine-energy-metric=whole_rms."
        ),
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
