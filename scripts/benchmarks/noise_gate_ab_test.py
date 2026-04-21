"""NoiseGate A/B benchmark harness.

livecap-cli の ``NoiseGate`` の挙動を閾値ごとに測定し、ASR エンジン別の
ハルシネーション指標 (entries / total_chars / max_char_run) を JSON 出力する。

本番コードパス ``StreamTranscriber.feed_audio() → NoiseGate.process() → VAD → engine``
を完全に再現するため、ハーネスとしての信頼性は高い。

Usage
-----
前提: `livecap-gui` の reference audio (``neko_reference.wav`` /
``neko_reference_noisy.wav``) を持っていること。

  https://github.com/Mega-Gorilla/livecap-gui/tree/main/experiments/noise_filter_comparison/test_data

Example::

    uv run python scripts/benchmarks/noise_gate_ab_test.py \\
        --test-data-dir /path/to/livecap-gui/experiments/noise_filter_comparison/test_data \\
        --engine whispers2t \\
        --files neko_reference_noisy.wav \\
        --output /tmp/noise_gate_ab_result.json

結果 JSON のスキーマ
-------------------
::

    {
      "engine": "whispers2t",
      "files": [
        {
          "file": "neko_reference_noisy.wav",
          "duration_s": 15.94,
          "sample_rate": 16000,
          "reference_text": "吾輩は猫である...",
          "results": [
            {
              "config": "baseline (no gate)",
              "rtf": 0.31,
              "n_entries": 6,
              "total_chars": 99,
              "max_char_run": 0,
              "entries": [
                {"start": 0.0, "end": 1.2, "text": "..."},
                ...
              ]
            },
            ...
          ]
        }
      ]
    }

評価メトリクス
-------------
- ``n_entries``: 転記エントリ数 (多すぎる = 過剰な断片化、少なすぎる = 発話欠落)
- ``total_chars``: 全エントリの文字数合計 (baseline 比で大幅増加 = ハルシネーション暴走)
- ``max_char_run``: 同一文字連続の最大長 (5 以上 = 暴走 / loop hallucination のサイン)

Related
-------
- Issue #280, PR #281, PR #282, Issue #283
- docs/benchmarks/noise-gate-ab.md (過去の実測結果の永続記録)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from livecap_cli.audio import NoiseGate
from livecap_cli.engines import EngineFactory
from livecap_cli.transcription import StreamTranscriber
from livecap_cli.vad import VADProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("noise_gate_ab")

CHUNK_MS = 100


def transcribe_file(
    audio: np.ndarray,
    sample_rate: int,
    engine: Any,
    vad: Any,
    gate: NoiseGate | None,
) -> tuple[list[Any], float]:
    """StreamTranscriber に ``gate`` を適用して転記を収集する。"""
    if gate is not None:
        gate.reset()
    transcriber = StreamTranscriber(
        engine=engine,
        vad_processor=vad,
        noise_gate=gate,
    )
    collected: list[Any] = []
    transcriber.set_callbacks(on_result=lambda r: collected.append(r))

    chunk_size = sample_rate * CHUNK_MS // 1000
    t0 = time.perf_counter()
    for i in range(0, len(audio), chunk_size):
        transcriber.feed_audio(audio[i : i + chunk_size], sample_rate)
    try:
        finals = transcriber.finalize()
        collected.extend(finals)
    except Exception as e:
        logger.warning("finalize failed: %s", e)
    elapsed = time.perf_counter() - t0
    return collected, elapsed


def summarize(entries: list[Any]) -> tuple[str, dict]:
    """転記エントリを集計してハルシネーション指標を返す。"""
    texts = [str(getattr(e, "text", "")) for e in entries]
    joined = " / ".join(texts)

    max_char_run = 0
    for t in texts:
        if not t:
            continue
        prev: str | None = None
        run = 0
        for c in t:
            if c == prev:
                run += 1
                max_char_run = max(max_char_run, run)
            else:
                run = 1
                prev = c

    metrics = {
        "n_entries": len(entries),
        "total_chars": sum(len(t) for t in texts),
        "max_char_run": max_char_run,
    }
    return joined, metrics


def run_all(
    audio_path: Path,
    reference_text: str,
    engine: Any,
    vad: Any,
    configs: list[tuple[str, NoiseGate | None]],
) -> dict:
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    duration = len(audio) / sr

    results = []
    for name, gate in configs:
        logger.info("Running: %s", name)
        entries, elapsed = transcribe_file(audio, sr, engine, vad, gate)
        _, metrics = summarize(entries)
        results.append(
            {
                "config": name,
                "rtf": round(elapsed / duration, 3),
                "n_entries": metrics["n_entries"],
                "total_chars": metrics["total_chars"],
                "max_char_run": metrics["max_char_run"],
                "entries": [
                    {
                        "start": round(float(getattr(e, "start_time", 0.0)), 2),
                        "end": round(float(getattr(e, "end_time", 0.0)), 2),
                        "text": str(getattr(e, "text", "")),
                    }
                    for e in entries
                ],
            }
        )

    return {
        "file": audio_path.name,
        "duration_s": round(duration, 2),
        "sample_rate": sr,
        "reference_text": reference_text,
        "results": results,
    }


def build_default_configs() -> list[tuple[str, NoiseGate | None]]:
    """標準的な A/B 比較用の config (baseline + 4 thresholds)。"""
    return [
        ("baseline (no gate)", None),
        ("gate threshold=-35 dB (default)", NoiseGate(threshold_db=-35)),
        ("gate threshold=-25 dB", NoiseGate(threshold_db=-25)),
        ("gate threshold=-20 dB", NoiseGate(threshold_db=-20)),
        ("gate threshold=-17 dB (user mic test)", NoiseGate(threshold_db=-17)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0] if __doc__ else "",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--test-data-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing reference WAVs + neko_reference.txt. "
            "Typical source: "
            "livecap-gui/experiments/noise_filter_comparison/test_data/"
        ),
    )
    parser.add_argument(
        "--engine",
        default="whispers2t",
        help="ASR engine ID (default: whispers2t)",
    )
    parser.add_argument(
        "--model-size",
        default="base",
        help="Model size for whispers2t (default: base)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device (default: cpu)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=["neko_reference.wav", "neko_reference_noisy.wav"],
        help="WAV file names (relative to --test-data-dir)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file path",
    )
    args = parser.parse_args()

    test_dir: Path = args.test_data_dir
    if not test_dir.is_dir():
        print(
            f"ERROR: --test-data-dir not found or not a directory: {test_dir}",
            file=sys.stderr,
        )
        return 1

    reference_txt = test_dir / "neko_reference.txt"
    if not reference_txt.exists():
        print(
            f"ERROR: neko_reference.txt not found in --test-data-dir: {test_dir}",
            file=sys.stderr,
        )
        return 1
    reference_text = reference_txt.read_text(encoding="utf-8").strip()

    print(f"Loading engine: {args.engine} (device={args.device})...")
    engine_kwargs: dict[str, Any] = {"device": args.device, "language": "ja"}
    if args.engine == "whispers2t":
        engine_kwargs["model_size"] = args.model_size
    engine = EngineFactory.create_engine(args.engine, **engine_kwargs)
    engine.load_model()
    print(f"Engine loaded: {args.engine}")

    try:
        vad = VADProcessor.from_language("ja", engine=args.engine)
        print(f"VAD: {vad.backend_name}")
    except Exception as e:
        print(f"VAD fallback: {e}")
        vad = None

    configs = build_default_configs()

    output: dict[str, Any] = {"engine": args.engine, "files": []}
    for fname in args.files:
        audio_path = test_dir / fname
        if not audio_path.exists():
            print(f"ERROR: audio file not found: {audio_path}", file=sys.stderr)
            return 1
        output["files"].append(
            run_all(audio_path, reference_text, engine, vad, configs)
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Results written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
