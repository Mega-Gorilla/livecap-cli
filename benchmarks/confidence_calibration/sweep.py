"""Stage 2 CLI: active calibration sweep (Issue #338 PR-β)。

`build_corpus.py` で生成した ``manifest.jsonl`` corpus に対し、ASR engine の
``engine.transcribe()`` を実行 → ``engine_confidence`` の signal field を抽出 →
``_core.sweep_threshold()`` で threshold sweep + recommended threshold 提示。

Stage 1 (``parse_observe.py``) との違い:
- Stage 1: observe mode log (既存 livecap-cli 運用の judgment 記録) を input
- Stage 2 (本 CLI): user 提供 audio corpus を input、engine を直接 invoke

CLI usage:

    python -m benchmarks.confidence_calibration.sweep \\
        --engine reazonspeech \\
        --signal avg_logprob \\
        --filter-by-language ja \\
        --quantization float32 \\
        --output report.json

Design (Plan D7, D8, D9):
- Engine.transcribe() は 1 sample 1 回のみ呼ぶ (重い処理、結果 cache)
- Sweep は cached value で計算 (engine 呼出回数は sample 数だけ、threshold 数ではない)
- argparse は ``benchmarks/non_speech_filter/sweep.py`` の canonical pattern を踏襲
- 5 engine 対応 (ReazonSpeech / Qwen3-ASR / Parakeet (ja/en) / Canary / WhisperS2T / Voxtral) — engine_factory.py 経由で dynamic load
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

from ._core import (
    Criterion,
    Direction,
    LabeledSample,
    report_to_dict,
    sweep_threshold,
)
from .parse_observe import SIGNAL_DIRECTION, normalize_engine_id
from .pipeline import load_calibration_corpus, resolve_corpus_dir

logger = logging.getLogger(__name__)


def _parse_engine_kwargs(raw: list[str]) -> dict[str, Any]:
    """argparse ``--engine-kwargs key=value`` を dict に変換。

    値は bool / int / float / str を自動推論 (Python literal eval は避けて simple parse)。
    例: ``["use_int8=true", "model_size=base"]`` → ``{"use_int8": True, "model_size": "base"}``
    """
    parsed: dict[str, Any] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"--engine-kwargs entry must be key=value, got: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        # 型推論: true/false → bool、int → int、float → float、その他 → str
        if value.lower() in ("true", "false"):
            parsed[key] = value.lower() == "true"
        else:
            try:
                parsed[key] = int(value)
            except ValueError:
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
    return parsed


def measure_signals(
    samples: list[Any],  # CalibrationCorpusItem from pipeline
    engine: Any,
    signal_field: str,
) -> list[LabeledSample]:
    """Each corpus sample で ``engine.transcribe()`` を 1 回呼び、signal 値を抽出。

    Args:
        samples: ``CalibrationCorpusItem`` list (path / label / audio)。
        engine: ASR engine instance、``transcribe(audio, sample_rate)`` 呼出可能。
        signal_field: ``avg_logprob`` / ``no_speech_prob`` / ``token_confidence_mean``。

    Returns:
        ``LabeledSample`` list、各 sample に signal_value (None なら sweep で除外)。
    """
    results: list[LabeledSample] = []
    skipped_no_signal = 0
    for idx, item in enumerate(samples):
        try:
            result = engine.transcribe(item.audio, item.sample_rate)
        except Exception as exc:
            logger.warning(
                "engine.transcribe() failed for sample %s (%d/%d): %s",
                item.path,
                idx + 1,
                len(samples),
                exc,
            )
            # Phase 6a: manifest 由来の metadata (snr_db / subtype 等) も pass-through
            error_metadata = dict(item.metadata)
            error_metadata["transcribe_error"] = str(exc)
            results.append(
                LabeledSample(
                    signal_value=None,
                    label=item.label,
                    path=str(item.path),
                    metadata=error_metadata,
                )
            )
            continue

        ec = result.engine_confidence
        signal_value = getattr(ec, signal_field, None)
        if signal_value is None:
            skipped_no_signal += 1
        # Phase 6a: manifest 由来の全 metadata (snr_db / subtype / noise_source_dataset 等)
        # を pass-through、 engine/result 由来の 3 key で override (--breakdown-by で
        # 参照可能にするため)。
        sample_metadata = dict(item.metadata)
        sample_metadata["text"] = result.text
        sample_metadata["is_available"] = ec.is_available
        # language は既に item.metadata から継承 (backward compat)
        results.append(
            LabeledSample(
                signal_value=signal_value,
                label=item.label,
                path=str(item.path),
                metadata=sample_metadata,
            )
        )
        if (idx + 1) % 25 == 0:
            logger.info("Measured %d/%d samples", idx + 1, len(samples))

    if skipped_no_signal > 0:
        logger.info(
            "Skipped %d samples with %s = None (fail-open / not populated)",
            skipped_no_signal,
            signal_field,
        )
    return results


def breakdown_list(value: str) -> list[str]:
    """argparse type: comma-separated metadata key list、 空文字 / duplicate reject。

    ``--breakdown-by snr_db,subtype`` → ``["snr_db", "subtype"]``。
    ``positive_int`` / ``snr_list`` の pattern 踏襲 (Phase 2 / Layer 3)。
    """
    if not value or not value.strip():
        raise argparse.ArgumentTypeError("--breakdown-by must not be empty")
    parts = [p.strip() for p in value.split(",")]
    result: list[str] = []
    for p in parts:
        if not p:
            raise argparse.ArgumentTypeError(
                f"--breakdown-by has empty item in {value!r}"
            )
        result.append(p)
    if len(result) != len(set(result)):
        seen: set[str] = set()
        dup = next(v for v in result if v in seen or seen.add(v))
        raise argparse.ArgumentTypeError(
            f"--breakdown-by has duplicate key {dup!r} in {value!r}"
        )
    return result


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmarks.confidence_calibration.sweep",
        description=(
            "Active threshold calibration: load corpus → engine.transcribe() → "
            "threshold sweep → report.json. Stage 2 of Issue #338."
        ),
    )
    parser.add_argument(
        "--engine",
        required=True,
        help=(
            "engine ID from livecap_cli/engines/metadata.py:_ENGINES "
            "(reazonspeech, qwen3asr, parakeet, parakeet_ja, canary, voxtral, whispers2t)"
        ),
    )
    parser.add_argument(
        "--signal",
        required=True,
        choices=list(SIGNAL_DIRECTION.keys()),
        help="signal field to sweep",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help=(
            "Corpus directory (manifest.jsonl + audio files). "
            "Default: $LIVECAP_CALIBRATION_CORPUS_DIR"
        ),
    )
    parser.add_argument(
        "--manifest-name",
        default="manifest.jsonl",
        help="manifest filename inside corpus-dir (default manifest.jsonl)",
    )
    parser.add_argument(
        "--filter-by-language",
        default=None,
        help="Restrict to corpus samples with this language (e.g. ja, en)",
    )
    parser.add_argument(
        "--breakdown-by",
        type=breakdown_list,
        default=[],
        help=(
            "Comma-separated metadata keys for per-value confusion-matrix breakdown "
            "(e.g. 'snr_db,subtype,noise_source_dataset'). Each key produces a "
            "per-value threshold sweep in report.breakdown[<key>]. Missing values "
            "grouped as '__none__' bucket. Default empty (no breakdown, backward compat "
            "with Phase 1 report schema)."
        ),
    )
    parser.add_argument("--threshold-min", type=float, default=None)
    parser.add_argument("--threshold-max", type=float, default=None)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument(
        "--criterion",
        choices=["f1", "youden_j", "precision", "recall"],
        default="f1",
    )
    parser.add_argument(
        "--quantization",
        default=None,
        help="Quantization metadata (e.g. int8 / float32) for ReazonSpeech",
    )
    parser.add_argument(
        "--engine-kwargs",
        nargs="*",
        default=[],
        help="Extra engine kwargs as key=value (e.g. use_int8=true model_size=base)",
    )
    parser.add_argument("--output", type=Path, default=Path("report.json"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # 1. Corpus directory 解決
    corpus_dir = args.corpus_dir
    if corpus_dir is None:
        corpus_dir = resolve_corpus_dir()
    if corpus_dir is None:
        logger.error(
            "corpus directory not set: pass --corpus-dir or set "
            "LIVECAP_CALIBRATION_CORPUS_DIR env var"
        )
        return 1

    # 2. Corpus load
    logger.info("Loading corpus from %s ...", corpus_dir)
    items = load_calibration_corpus(corpus_dir, manifest_name=args.manifest_name)
    logger.info("Loaded %d corpus items", len(items))

    # 3. Language filter
    if args.filter_by_language:
        before = len(items)
        items = [
            it for it in items if it.metadata.get("language") == args.filter_by_language
        ]
        logger.info(
            "Filtered to language=%s: %d → %d items",
            args.filter_by_language,
            before,
            len(items),
        )

    if not items:
        logger.error("No corpus items after filtering")
        return 1

    # 4. Engine 準備
    from livecap_cli.engines.engine_factory import EngineFactory

    engine_kwargs = _parse_engine_kwargs(args.engine_kwargs)
    logger.info("Creating engine %s with kwargs=%s", args.engine, engine_kwargs)
    engine = EngineFactory.create_engine(args.engine, **engine_kwargs)
    engine.load_model()

    # 5. Signal 値の measurement (engine.transcribe() を sample ごとに 1 回)
    samples = measure_signals(items, engine, args.signal)

    # 6. Threshold range (signal 種別で default 推定)
    direction = SIGNAL_DIRECTION[args.signal]
    if args.threshold_min is None or args.threshold_max is None:
        if args.signal == "avg_logprob":
            default_min, default_max = -1.0, -0.05
        elif args.signal == "token_confidence_mean":
            default_min, default_max = 0.001, 0.5
        elif args.signal == "no_speech_prob":
            default_min, default_max = 0.1, 0.95
        else:
            default_min, default_max = -1.0, 1.0
        threshold_min = args.threshold_min if args.threshold_min is not None else default_min
        threshold_max = args.threshold_max if args.threshold_max is not None else default_max
    else:
        threshold_min = args.threshold_min
        threshold_max = args.threshold_max

    # 7. Sweep
    metadata: dict[str, Any] = {
        "engine_normalized": normalize_engine_id(engine.get_engine_name()),
        "engine_display": engine.get_engine_name(),
        "corpus_dir": str(corpus_dir),
        "corpus_size_loaded": len(items),
        "samples_with_signal": sum(1 for s in samples if s.signal_value is not None),
    }
    if args.quantization:
        metadata["quantization"] = args.quantization
    if args.filter_by_language:
        metadata["language"] = args.filter_by_language
    if engine_kwargs:
        metadata["engine_kwargs"] = engine_kwargs
    if args.breakdown_by:
        metadata["breakdown_by"] = args.breakdown_by  # 実施した breakdown key の記録

    # Phase 6a: warn on breakdown keys not present in any sample (typo detection)
    if args.breakdown_by:
        for key in args.breakdown_by:
            if not any(key in s.metadata for s in samples):
                logger.warning(
                    "breakdown_by key %r not found in any sample metadata, "
                    "all samples will fall into '__none__' bucket (typo?)",
                    key,
                )

    report = sweep_threshold(
        samples,
        engine=args.engine,
        signal_field=args.signal,
        direction=direction,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        step=args.step,
        criterion=args.criterion,
        metadata=metadata,
        breakdown_by=args.breakdown_by or None,
    )

    # 8. Output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report_to_dict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "Wrote report to %s: recommended %s = %.4f (criterion=%s, F1=%.3f, samples=%s)",
        args.output,
        args.signal,
        report.recommended_threshold,
        args.criterion,
        report.recommended_metrics.f1,
        report.sample_count,
    )

    # Cleanup engine resources
    if hasattr(engine, "cleanup"):
        try:
            engine.cleanup()
        except Exception:
            pass

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
