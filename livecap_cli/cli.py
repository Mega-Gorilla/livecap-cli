"""CLI for livecap-cli - High-performance speech transcription."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
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
            peak_margin = (
                args.noise_gate_margin
                if getattr(args, "noise_gate_margin", None) is not None
                else PEAK_SAFETY_MARGIN_DB
            )
            analysis = analyze_noise_samples(
                all_rms_levels,
                all_peak_levels,
                sample_rate_hz=sample_rate_hz,
                engine_min_rms_margin_db=engine_margin,
                peak_safety_margin_db=peak_margin,
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
                    f"(= peak_p95 + {peak_margin:g}; "
                    f"per-sample peak unit)",
                    file=sys.stderr,
                )
                print(
                    f"Suggested --engine-min-rms:       "
                    f"{analysis.suggested_engine_min_rms_dbfs:.0f} dB "
                    f"(= noise_rms_p95 + {engine_margin:g}; "
                    f"RMS-unit, calibrated from chunk RMS p95; #292 EnergyGate)",
                    file=sys.stderr,
                )
                print(
                    f"  (CLI default is -45 dB; pass the suggested value "
                    f"above with --engine-min-rms for env-specific tuning.)",
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

    ``nan`` and ``+inf`` are rejected:
      - ``nan`` would silently disable the gate (every ``energy_dbfs < nan``
        is False), which is the worst kind of misbehavior -appears to work
        but does nothing.
      - ``+inf`` would drop every segment unconditionally.

    Note:
        argparse rejects bare ``-inf`` as a value because leading-``-`` is
        parsed as another option. Use ``--engine-min-rms=-inf`` (equals form)
        or ``--engine-min-rms off`` instead.
    """
    import math

    if value.lower() in ("off", "disabled", "none"):
        return float("-inf")
    try:
        result = float(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"invalid value for --engine-min-rms: {value!r} "
            f"(expected number, 'off', 'disabled', or 'none')"
        ) from e
    if math.isnan(result):
        raise argparse.ArgumentTypeError(
            f"--engine-min-rms cannot be NaN (got {value!r}). "
            "Use a finite number, '-inf', 'off', 'disabled', or 'none'."
        )
    if result == float("inf"):
        raise argparse.ArgumentTypeError(
            f"--engine-min-rms cannot be +inf (got {value!r}). "
            "Use a finite number or -inf to opt out."
        )
    return result


def _parse_engine_energy_frame_ms(value: str) -> float:
    """argparse type for --engine-energy-frame-ms.

    Accepts finite positive values only. ``nan`` / ``inf`` / non-positive
    values are rejected at parse time (otherwise they would either silently
    bypass the ``frame_ms <= 0`` check in ``StreamTranscriber.__init__`` or
    crash later in ``int(sample_rate * frame_ms / 1000.0)``).
    """
    import math

    try:
        result = float(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"invalid value for --engine-energy-frame-ms: {value!r} "
            "(expected a positive finite number in milliseconds)"
        ) from e
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError(
            f"--engine-energy-frame-ms must be finite (got {value!r})."
        )
    if result <= 0:
        raise argparse.ArgumentTypeError(
            f"--engine-energy-frame-ms must be positive (got {result})."
        )
    return result


def cmd_transcribe(args: argparse.Namespace) -> int:
    """Transcribe audio from microphone or file."""
    # Check for required arguments
    if args.realtime:
        if args.mic is None:
            print("Error: --mic is required for realtime transcription", file=sys.stderr)
            return 1
        if args.vad == "off":
            # #366: VAD なし realtime は StreamTranscriber の segment 生成方式を
            # 変える別機能 — モデルロード前に fail-fast する
            print(
                "Error: --vad off is file mode only "
                "(realtime transcription requires VAD).",
                file=sys.stderr,
            )
            return 1
    elif not args.input_file:
        print("Error: Either --realtime --mic <id> or <input_file> is required", file=sys.stderr)
        return 1

    # Issue #365: --language を engine 別に単一解決 (モデルロード前の fail-fast)。
    # 以降の全 consumer (engine kwargs / VAD preset / translator / ログ) は
    # resolved 値を読む。
    from livecap_cli.engines import EngineMetadata

    try:
        resolved_language = EngineMetadata.resolve_language(args.engine, args.language)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(
        f"Language: requested={args.language or '(engine default)'}, "
        f"resolved={resolved_language}",
        file=sys.stderr,
    )
    args.language = resolved_language

    if args.realtime:
        return _transcribe_realtime(args)
    return _transcribe_file(args)


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


def _build_segment_transcriber(args: argparse.Namespace, engine, input_path: Path):
    """file mode 用の segment_transcriber closure を構築 (#366 Phase 2)。

    処理順は realtime と同一: EnergyGate (ASR 前) → engine.transcribe →
    confidence filter (ASR 後)。判定式は共通 module (should_drop_low_energy /
    apply_filter) を realtime と共有し、結果は `SegmentOutcome` で pipeline へ
    運ぶ (drop は reason 別に metadata `drop_counts` へ集計される)。
    """
    from livecap_cli.audio import should_drop_low_energy
    from livecap_cli.transcription import (
        REASON_ENERGY_GATE,
        REASON_FILTER_REJECT,
        SegmentOutcome,
    )
    from livecap_cli.transcription.confidence_filter import apply_filter

    # closure 構築時に一度だけ解決 (segment 毎に解決しない)
    filter_config = _create_filter_config(args)  # env > flag の precedence を共有
    engine_name = engine.get_engine_name()
    source_id = str(input_path)  # 複数ファイルの observe log を区別可能に

    def segment_transcriber(audio, sample_rate):
        should_drop, _energy = should_drop_low_energy(
            audio,
            sample_rate,
            threshold_dbfs=args.engine_min_rms,
            metric=args.engine_energy_metric,
            frame_ms=args.engine_energy_frame_ms,
        )
        if should_drop:
            # ASR 前に弾く gate — engine.transcribe() は呼ばない
            return SegmentOutcome.dropped(REASON_ENERGY_GATE, asr_called=False)

        result = engine.transcribe(audio, sample_rate)
        filtered = apply_filter(
            result,
            filter_config,
            source_id=source_id,
            engine_name=engine_name,
            is_interim=False,
        )
        if filtered is None:  # mode "on" の reject (observe は素通り + log)
            return SegmentOutcome.dropped(REASON_FILTER_REJECT)
        return SegmentOutcome.success(filtered.text)

    return segment_transcriber


def _build_audio_preprocessor(args: argparse.Namespace):
    """file mode 用の NoiseGate preprocessor を構築 (#366 Phase 3)。opt-in。

    per-file factory: 呼び出し (=ファイル) ごとに新しい NoiseGate を生成し、
    `process_files()` のファイル間・例外後の状態非共有を構造的に保証する
    (pipeline は `reset()` のような暗黙契約を持たない)。args mapping は
    realtime (`_transcribe_realtime`) の NoiseGate 構築と同一。
    """
    if not args.noise_gate:
        return None
    from livecap_cli.audio.noise_gate import NoiseGate

    gate_kwargs = dict(
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

    # resolved 値ログ (AGENTS.md: log resolved values)。
    # NoiseGate は不正値を raise せず fallback / clamp するため、args をそのまま
    # 表示すると実設定と乖離する (例: --noise-gate-threshold 5 は内部で -35 に
    # fallback)。実インスタンスの解決済み公開属性から表示する。
    resolved = NoiseGate(**gate_kwargs)
    print(
        f"Noise gate enabled: open={resolved.threshold_db:.1f}dB, "
        f"close={resolved.close_threshold_db:.1f}dB, "
        f"floor={resolved.describe_noise_floor()}, "
        f"attack={resolved.attack_ms:.1f}ms, "
        f"release={resolved.release_ms:.1f}ms",
        file=sys.stderr,
    )

    # per-file gate には**解決済みの値**を渡す。raw args を渡すと各 file の
    # 構築時に fallback / clamp が再判定され、不正設定時に同じ warning が
    # ファイル数だけ重複出力される (PR #372 レビュー)。
    resolved_kwargs = dict(
        threshold_db=resolved.threshold_db,
        close_threshold_db=resolved.close_threshold_db,
        attack_ms=resolved.attack_ms,
        release_ms=resolved.release_ms,
        noise_floor_db=(
            float("-inf")
            if resolved.noise_floor_db is None
            else resolved.noise_floor_db
        ),
    )

    def preprocessor(audio, sample_rate):
        gate = NoiseGate(**resolved_kwargs, sample_rate=sample_rate)  # per-file 生成
        return gate.process(audio)

    return preprocessor


def _build_file_segmenter(args: argparse.Namespace):
    """file mode 用の VAD segmenter を構築する (#366 Phase 1)。

    ``--vad off`` は None を返し、pipeline は従来どおり音声全体を
    1 segment として処理する (`segmenter=None` の fallback)。
    それ以外は resolved language (#365) の preset で VADProcessor を作り
    `VADFileSegmenter` adapter に包む。
    """
    if args.vad == "off":
        return None
    from livecap_cli.vad import VADFileSegmenter

    return VADFileSegmenter(
        _get_vad_processor(args.language, args.vad, engine=args.engine)
    )


def _create_filter_config(args: argparse.Namespace):
    """``FilterConfig`` を CLI args + env var から構築 (PR-A.1 / Issue #308 v3.1)。

    Precedence (highest to lowest):

    1. ``LIVECAP_CONFIDENCE_FILTER`` env var (whitelist 検証: off/observe/on)
    2. ``--confidence-filter`` CLI flag
    3. ``FilterConfig`` default (``"on"``)

    invalid env var 値は warning log を出して無視 (CLI flag を採用)。
    """
    from livecap_cli.transcription.confidence_filter import FilterConfig

    mode_from_env = os.environ.get("LIVECAP_CONFIDENCE_FILTER", "").strip().lower()
    valid_modes = ("off", "observe", "on")

    if mode_from_env:
        if mode_from_env in valid_modes:
            return FilterConfig(mode=mode_from_env)
        else:
            print(
                f"Warning: LIVECAP_CONFIDENCE_FILTER={mode_from_env!r} is invalid "
                f"(expected one of {valid_modes}); falling back to CLI flag.",
                file=sys.stderr,
            )

    return FilterConfig(mode=args.confidence_filter)


# language constructor 引数を持つ multilingual engine (Issue #365)。
# 単一言語 engine (reazonspeech/parakeet/parakeet_ja) は language 引数を
# 持たないため渡さない — 不一致は resolve_language() が事前に拒否する。
_LANGUAGE_ROUTED_ENGINES = ("whispers2t", "canary", "voxtral", "qwen3asr")


def _build_engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Construct engine_kwargs for ``EngineFactory.create_engine``.

    Centralizes per-engine option routing so realtime / file paths stay in
    sync. `args.language` は `cmd_transcribe` で `resolve_language()` 済みの
    値である前提 (Issue #365 — 従来は qwen3asr のみ routing され、他 engine
    は constructor default で silent 動作していた)。
    """
    engine_kwargs: dict[str, Any] = {}
    if args.engine == "whispers2t" and args.model_size:
        engine_kwargs["model_size"] = args.model_size
    if args.engine in _LANGUAGE_ROUTED_ENGINES and args.language:
        engine_kwargs["language"] = args.language
    return engine_kwargs


def _load_engine(args: argparse.Namespace):
    """Create and load the ASR engine (realtime / file 両経路で共有).

    engine 生成 boilerplate を一元化し、経路間の contract drift (#363 の原因
    パターン) を防ぐ。option routing は `_build_engine_kwargs` に集約済み。
    """
    from livecap_cli.engines import EngineFactory

    device = _map_device(args.device)
    engine_kwargs = _build_engine_kwargs(args)

    print(f"Loading engine: {args.engine} (device={device})...", file=sys.stderr)
    engine = EngineFactory.create_engine(args.engine, device=device, **engine_kwargs)
    try:
        engine.load_model()
    except Exception:
        # caller へ返る前の失敗は caller の finally が拾えない — ここで cleanup
        with contextlib.suppress(Exception):
            engine.cleanup()
        raise
    return engine


def _transcribe_realtime(args: argparse.Namespace) -> int:
    """Realtime transcription from microphone."""
    try:
        from livecap_cli import StreamTranscriber, MicrophoneSource

        engine = _load_engine(args)

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

        # === Layer 1: DSP transient detector (#295 PR-B) ==================
        # Status: EXPERIMENTAL. The 2026-06-07 calibration sweep showed
        # 0 pp improvement on the real-corpus AC target cell (WebRTC x
        # parakeet_ja x desk_tap hallucination), so this layer is not a
        # production hallucination mitigation candidate. Phase 2 SED is
        # the planned successor. See docs/audio-filter-reference.md.
        transient_detector = None
        if args.transient_filter != "off":
            from livecap_cli.audio import (
                TransientDetector,
                TransientDetectorConfig,
            )

            print(
                "WARNING: --transient-filter is EXPERIMENTAL. Calibration "
                "showed no improvement on the real-corpus target cell "
                "(webrtc x parakeet_ja x desk_tap). Keep --transient-filter=off "
                "for production hallucination mitigation unless you are "
                "explicitly testing burst-applause scenarios. "
                "See docs/audio-filter-reference.md.",
                file=sys.stderr,
            )

            transient_detector = TransientDetector(
                TransientDetectorConfig(
                    mode=args.transient_filter,
                    flatness_min=args.transient_flatness_min,
                    centroid_min_hz=args.transient_centroid_min_hz,
                    zcr_min=args.transient_zcr_min,
                    onset_ratio=args.transient_onset_ratio,
                    voiced_max=args.transient_voiced_max,
                    rms_min_db=args.transient_rms_min_db,
                ),
                sample_rate=16000,
            )

        # Start transcription
        print(f"Starting realtime transcription (mic={args.mic}, language={args.language})...", file=sys.stderr)
        print("Press Ctrl+C to stop.\n", file=sys.stderr)

        # === Layer 3: Confidence filter (PR-A.1 / Issue #308) ==============
        # PR-A.0 で expose した engine_confidence を見て非音声判定を弾く。
        # default `on` (Issue #308 v3.1)、`LIVECAP_CONFIDENCE_FILTER=off` で
        # 完全に旧挙動に戻せる。
        filter_config = _create_filter_config(args)

        with StreamTranscriber(
            engine=engine,
            vad_processor=vad_processor,
            noise_gate=noise_gate,
            transient_detector=transient_detector,
            engine_min_rms_dbfs=args.engine_min_rms,
            engine_energy_metric=args.engine_energy_metric,
            engine_energy_frame_ms=args.engine_energy_frame_ms,
            filter_config=filter_config,
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


# file mode では適用されない realtime (StreamTranscriber) 経路専用オプション。
# (option 表示名, args attribute 名, parser default) — parser 定義と同期必須。
# 同期は tests/core/cli/test_transcribe_file.py の defaults 同期テストで CI 固定。
# 「既定値と同値の明示指定」の検出は #382 (argparse.SUPPRESS + resolution 層) の scope。
_REALTIME_ONLY_OPTIONS: tuple[tuple[str, str, Any], ...] = (
    ("--mic", "mic", None),
    ("--transient-filter", "transient_filter", "off"),
    ("--transient-flatness-min", "transient_flatness_min", 0.30),
    ("--transient-centroid-min-hz", "transient_centroid_min_hz", 2500.0),
    ("--transient-zcr-min", "transient_zcr_min", 0.12),
    ("--transient-onset-ratio", "transient_onset_ratio", 3.0),
    ("--transient-voiced-max", "transient_voiced_max", 0.25),
    ("--transient-rms-min-db", "transient_rms_min_db", -35.0),
)


def _warn_realtime_only_options(args: argparse.Namespace) -> None:
    """file mode で無視される realtime 専用オプションの warning (#363).

    parser 既定値から変更されたもののみ対象 (silent no-op の解消)。
    """
    changed = [
        option
        for option, attr, default in _REALTIME_ONLY_OPTIONS
        if getattr(args, attr, default) != default
    ]
    if changed:
        print(
            "Warning: the following options are realtime-only and ignored in "
            f"file mode: {', '.join(changed)}. See docs/reference/cli.md.",
            file=sys.stderr,
        )


def _transcribe_file(args: argparse.Namespace) -> int:
    """Transcribe from file via FileTranscriptionPipeline (Issue #363).

    出力マトリクス (issue #363 v3):
    - ``-o`` あり: 指定パスへ SRT (翻訳指定時は翻訳 SRT)
    - ``-o`` なし: SRT content を stdout へ (進捗/警告は stderr)
    - ASR 全滅 (``success=False``) / 翻訳全件失敗: exit 1、出力ファイル非生成
    - 翻訳一部失敗: 翻訳成功 segment のみ出力 + 件数付き warning
    """
    # モデルロード前に入力を検証する
    input_path = Path(args.input_file)
    if not input_path.is_file():
        print(f"Error: File not found: {args.input_file}", file=sys.stderr)
        return 1

    _warn_realtime_only_options(args)

    # Issue #365: 翻訳併用は resolved concrete 言語が必須 (モデルロード前に拒否)。
    # engine 結果型に検出言語が無く translator へ渡せない + OPUS-MT は
    # source_lang="auto" で実在しない model 名を生成するため。
    if args.translate and args.language == "auto":
        print(
            f"Error: --translate requires a concrete source language, but "
            f"engine '{args.engine}' resolved --language to 'auto'. "
            f"Specify one explicitly (e.g. --language en).",
            file=sys.stderr,
        )
        return 1

    engine = None
    translator = None
    pipeline = None
    try:
        from livecap_cli.transcription import (
            FileTranscriptionPipeline,
            build_srt,
            write_srt,
        )

        # VAD segmenter を先に構築 (#366 Phase 1) — 構築失敗を重い engine
        # ロードより前に検出する。--vad off は None (全音声 1 segment)。
        segmenter = _build_file_segmenter(args)
        # NoiseGate preprocessor (#366 Phase 3、opt-in — 未指定は None)
        audio_preprocessor = _build_audio_preprocessor(args)

        engine = _load_engine(args)

        # EnergyGate + confidence filter を含む closure (#366 Phase 2)
        segment_transcriber = _build_segment_transcriber(args, engine, input_path)

        if args.translate:
            from livecap_cli.translation import TranslatorFactory

            print(f"Loading translator: {args.translate}...", file=sys.stderr)
            # OPUS-MT は言語ペアを constructor で受け取る (Google は無視して良い)
            translator = TranslatorFactory.create_translator(
                args.translate,
                source_lang=args.language,
                target_lang=args.target_lang,
            )
            translator.load_model()

        pipeline = FileTranscriptionPipeline(
            segmenter=segmenter, audio_preprocessor=audio_preprocessor
        )

        print(f"Transcribing: {args.input_file}...", file=sys.stderr)
        result = pipeline.process_file(
            input_path,
            segment_transcriber=segment_transcriber,
            translator=translator,
            source_lang=args.language if translator else None,
            target_lang=args.target_lang if translator else None,
            # 出力先は CLI が制御する (入力横への sidecar 生成を抑止)
            write_subtitles=False,
            write_translated_subtitles=False,
        )

        if not result.success:
            print(f"Error: {result.error}", file=sys.stderr)
            return 1

        if result.metadata.get("segmentation_empty"):
            # #366: VAD (等の注入 segmenter) がセグメントなしと判定。
            # 仕様: exit 0 / 出力は空 (SRT も空 file) / 情報は stderr のみ
            print("No speech segments detected.", file=sys.stderr)

        drop_counts = result.metadata.get("drop_counts") or {}
        if drop_counts:
            # #366 Phase 2: gate / filter による drop の可視化 (stderr のみ)
            summary = ", ".join(f"{k}={v}" for k, v in sorted(drop_counts.items()))
            print(f"Dropped segments: {summary}", file=sys.stderr)

        subtitles = result.subtitles
        use_translated = translator is not None
        if use_translated and subtitles:
            translated = [s for s in subtitles if s.translated_text]
            if not translated:
                # --translate 明示時に原文へ silent fallback しない (#363)
                print(
                    f"Error: translation failed for all {len(subtitles)} "
                    "segments; no output generated.",
                    file=sys.stderr,
                )
                return 1
            if len(translated) < len(subtitles):
                print(
                    f"Warning: translation failed for "
                    f"{len(subtitles) - len(translated)}/{len(subtitles)} "
                    "segments; only translated segments are output.",
                    file=sys.stderr,
                )

        if args.output:
            write_srt(Path(args.output), subtitles, translated=use_translated)
            print(f"Output written to: {args.output}", file=sys.stderr)
        else:
            sys.stdout.write(build_srt(subtitles, translated=use_translated))

        return 0
    except ImportError as e:
        print(f"Error: Missing dependency: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error during transcription: {e}", file=sys.stderr)
        return 1
    finally:
        for closer in (
            getattr(translator, "cleanup", None),
            getattr(engine, "cleanup", None),
            getattr(pipeline, "close", None),
        ):
            if closer is not None:
                with contextlib.suppress(Exception):
                    closer()


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
    levels_parser.add_argument(
        "--noise-gate-margin",
        type=float,
        default=None,
        help=(
            "Safety margin (dB) for suggested_threshold_db = peak_p95 + margin "
            "(NoiseGate, peak unit; Issue #327). "
            "Default: 6.0 (calibrated for USB mics / general environments). "
            "Smaller value (e.g., 2) for high-SNR mics (AT4040, SM7B). "
            "Negative values are allowed (e.g., -5 for studio condenser mics "
            "where peak_p95 is already conservative, ~-60 dB)."
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
        help="Output SRT file (file mode; default: write SRT to stdout)",
    )
    transcribe_parser.add_argument(
        "--realtime",
        action="store_true",
        help="Enable realtime transcription mode",
    )
    transcribe_parser.add_argument(
        "--mic",
        type=int,
        help="[realtime only] Microphone device index (use 'devices' command to list)",
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
        default=None,
        help=(
            "Input language code (e.g. ja, en; BCP-47 like ja-JP accepted). "
            "Default depends on engine: whispers2t/qwen3asr/reazonspeech/"
            "parakeet_ja -> ja, canary/parakeet -> en, voxtral -> auto. "
            "'auto' is only valid for engines with native auto-detect "
            "(voxtral, qwen3asr). Unsupported or malformed codes fail before "
            "model load. See docs/reference/cli.md."
        ),
    )
    transcribe_parser.add_argument(
        "--model-size",
        default="base",
        help="Model size for WhisperS2T (default: base)",
    )
    transcribe_parser.add_argument(
        "--vad",
        choices=["auto", "silero", "tenvad", "webrtc", "off"],
        default="auto",
        help=(
            "VAD backend for speech segmentation (default: auto). "
            "In file mode, audio is segmented by VAD before ASR (#366); "
            "'off' disables VAD segmentation and processes the whole audio "
            "as one segment. 'off' is file mode only - realtime "
            "transcription requires VAD."
        ),
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
            "(default: -45.0; conservative, chosen to preserve whisper-quiet "
            "speech in any environment). "
            "**RECOMMENDED**: if hallucinations on silence persist, run "
            "`livecap-cli levels --mic <id> --duration 5` to get an "
            "environment-specific suggested value (empirically 2-3x more "
            "effective in noisy-silence environments; #292). "
            "Use 'off' or '=-inf' to disable. "
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
        type=_parse_engine_energy_frame_ms,
        default=32.0,
        help=(
            "Frame size (ms) for frame-based energy metrics "
            "(default: 32, typical range: 10-200, must be finite positive). "
            "Ignored when --engine-energy-metric=whole_rms."
        ),
    )

    # === Layer 1: DSP Transient/Applause Detector (#295 PR-B) =============
    transcribe_parser.add_argument(
        "--transient-filter",
        choices=("off", "observe", "on"),
        default="off",
        help=(
            "[realtime only] Layer 1 DSP transient detector mode (default: off). "
            "EXPERIMENTAL: the 2026-06-07 calibration sweep showed no "
            "improvement on the real-corpus target cell, so this layer "
            "is not a production hallucination mitigation candidate. "
            "Keep off for production. "
            "'observe' computes features + telemetry only (audio "
            "unchanged); 'on' zeros out frames classified as applause-"
            "like. Use observe/on only for DSP calibration experiments. "
            "See docs/audio-filter-reference.md for the full status and "
            "docs/benchmarks/calibration-results-2026-06-07.md for the "
            "empirical evidence."
        ),
    )
    transcribe_parser.add_argument(
        "--transient-flatness-min",
        type=float,
        default=0.30,
        help="[realtime only] Spectral flatness lower bound for applause-like (default: 0.30).",
    )
    transcribe_parser.add_argument(
        "--transient-centroid-min-hz",
        type=float,
        default=2500.0,
        help="[realtime only] Spectral centroid lower bound in Hz (default: 2500).",
    )
    transcribe_parser.add_argument(
        "--transient-zcr-min",
        type=float,
        default=0.12,
        help="[realtime only] Zero-crossing rate lower bound (default: 0.12).",
    )
    transcribe_parser.add_argument(
        "--transient-onset-ratio",
        type=float,
        default=3.0,
        help=(
            "[realtime only] Onset-strength must exceed rolling baseline by this multiple "
            "(default: 3.0)."
        ),
    )
    transcribe_parser.add_argument(
        "--transient-voiced-max",
        type=float,
        default=0.25,
        help=(
            "[realtime only] Voiced confidence upper bound (autocorrelation peak ratio; "
            "default: 0.25 -speech is usually well above this)."
        ),
    )
    transcribe_parser.add_argument(
        "--transient-rms-min-db",
        type=float,
        default=-35.0,
        help=(
            "[realtime only] RMS dBFS lower bound to even consider a frame "
            "(default: -35; suppresses background-noise false positives)."
        ),
    )
    # === Layer 3: Engine confidence filter (PR-A.1 / Issue #308) =========
    transcribe_parser.add_argument(
        "--confidence-filter",
        choices=("off", "observe", "on"),
        default="on",
        help=(
            "Engine confidence filter mode (default: on). "
            "Filters ASR output that the engine itself judged as "
            "low-confidence / non-speech (WhisperS2T no_speech_prob > 0.71, "
            "Parakeet (ja/en) / Canary token_confidence_mean < 0.001 "
            "[ja: PR-A.0 CTC, en: PR-A.4.3 TDT+preserve_alignments, Canary: "
            "PR-A.4.2 via greedy decoding], Voxtral avg_logprob < -1.0 "
            "[PR-A.4.1, strict-gated: only when other signals are absent], "
            "ReazonSpeech avg_logprob < -0.40 [PR-A.5.1 -> [#334] PR-4, "
            "engine-specific, Phase 2 report Pareto relaxed_B], "
            "qwen3-asr avg_logprob < -0.42 [PR-A.5.2 -> [#334] PR-4, "
            "wrapper bypass + repetition_penalty=1.1, Phase 2 report Pareto "
            "relaxed_C; auto-detect mode (--language=auto) is fail-open]). "
            "'off' disables the post-ASR reject only - engine-side generation "
            "parameters (Canary greedy, qwen3asr repetition_penalty) are fixed "
            "and remain applied. "
            "'observe' logs reject decisions but does not drop anything "
            "(use for PR-A.3 calibration data collection). "
            "'on' silently drops rejected outputs. "
            "Override via LIVECAP_CONFIDENCE_FILTER env var. "
            "See docs/audio-filter-reference.md."
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
