"""Stage 1 CLI: parse ``confidence_filter[observe]`` JSON log → sweep report (Issue #338 PR-α)。

``LIVECAP_CONFIDENCE_FILTER=observe`` で蓄積した JSON log (``confidence_filter.py``
の ``_decision_to_dict()`` schema) を input、user 提供 label と join、
``_core.sweep_threshold()`` で sweep。

CLI usage:

    python -m benchmarks.confidence_calibration.parse_observe \\
        --log path/to/observe.jsonl \\
        --labels path/to/labels.jsonl \\
        --engine reazonspeech \\
        --signal avg_logprob \\
        --threshold-min -1.0 --threshold-max -0.05 --step 0.01 \\
        --output report.json

Schemas:

* observe log line format (``confidence_filter.py:_decision_to_dict()``):

  ::

      confidence_filter[observe]: {"source_id": "...", "engine": "reazonspeech",
                                    "text": "...", "decision": "pass" or "reject",
                                    "reason": null, "engine_confidence": {...}}

  ``"confidence_filter[<mode>]: "`` の prefix を strip して JSON parse。

* labels.jsonl schema (user 提供):

  ::

      {"source_id": "mic_001_chunk_00042", "text": "...", "label": "speech"}
      {"source_id": "mic_001_chunk_00043", "label": "non_speech", "subtype": "applause"}

  ``source_id`` + (optional) ``text`` で log entry と join。``text`` match は
  fuzzy (lower + strip)、source_id match は exact。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ._core import (
    Criterion,
    Direction,
    LabeledSample,
    report_to_dict,
    sweep_threshold,
)

logger = logging.getLogger(__name__)

# Signal direction map (engine が出す signal による、_core.py docstring 参照)
SIGNAL_DIRECTION: dict[str, Direction] = {
    "avg_logprob": "reject_if_less",
    "token_confidence_mean": "reject_if_less",
    "no_speech_prob": "reject_if_greater",
}

LOG_PREFIX = "confidence_filter["


@dataclass(frozen=True)
class LogEntry:
    source_id: str
    engine: str
    text: str
    decision: str  # "pass" / "reject"
    signal_value: Optional[float]


def parse_log_line(line: str, signal_field: str) -> Optional[LogEntry]:
    """1 行を parse、unmatched / malformed は ``None`` を返す。

    Expected format::

        <timestamp/level prefix>... confidence_filter[<mode>]: <JSON>

    Python logging の標準 format は前置 prefix を含む可能性があるため、
    ``confidence_filter[`` 以降を抽出して parse する。
    """
    idx = line.find(LOG_PREFIX)
    if idx < 0:
        return None
    # confidence_filter[<mode>]: <JSON> の部分を抽出
    rest = line[idx:]
    # ": " の後を JSON とみなす
    colon = rest.find("]: ")
    if colon < 0:
        return None
    json_part = rest[colon + 3 :].rstrip("\n\r")
    try:
        data = json.loads(json_part)
    except json.JSONDecodeError as exc:
        logger.warning("Malformed JSON line skipped: %s", exc)
        return None
    ec = data.get("engine_confidence") or {}
    signal_value = ec.get(signal_field)
    return LogEntry(
        source_id=data.get("source_id", ""),
        engine=data.get("engine", ""),
        text=data.get("text", ""),
        decision=data.get("decision", "pass"),
        signal_value=float(signal_value) if signal_value is not None else None,
    )


def load_labels(labels_path: Path) -> dict[str, dict[str, str]]:
    """``labels.jsonl`` を ``source_id`` で index 化。

    Returns:
        ``{source_id: {"label": ..., "text": ..., "subtype": ...}}``
    """
    if not labels_path.exists():
        raise FileNotFoundError(f"labels file not found: {labels_path}")
    index: dict[str, dict[str, str]] = {}
    for line_no, line in enumerate(
        labels_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("labels.jsonl line %d malformed, skipped: %s", line_no, exc)
            continue
        source_id = entry.get("source_id")
        if not source_id:
            logger.warning("labels.jsonl line %d missing source_id, skipped", line_no)
            continue
        index[source_id] = entry
    return index


def parse_observe_log(
    log_path: Path,
    labels_path: Path,
    engine: str,
    signal_field: str,
) -> list[LabeledSample]:
    """Log を parse して ``LabeledSample`` list を返す。"""
    if not log_path.exists():
        raise FileNotFoundError(f"log file not found: {log_path}")
    labels_index = load_labels(labels_path)
    samples: list[LabeledSample] = []
    unmatched = 0
    skipped_engine = 0
    for line in log_path.read_text(encoding="utf-8").splitlines():
        entry = parse_log_line(line, signal_field)
        if entry is None:
            continue
        if entry.engine != engine:
            skipped_engine += 1
            continue
        label_entry = labels_index.get(entry.source_id)
        if label_entry is None:
            unmatched += 1
            continue
        label = label_entry.get("label", "")
        if label not in ("speech", "non_speech", "noisy_speech"):
            logger.warning(
                "source_id=%s has invalid label=%r, skipped", entry.source_id, label
            )
            continue
        samples.append(
            LabeledSample(
                signal_value=entry.signal_value,
                label=label,  # type: ignore[arg-type]
                path=entry.source_id,
                metadata={"text": entry.text},
            )
        )
    if unmatched:
        logger.info("Unmatched log entries (no label): %d", unmatched)
    if skipped_engine:
        logger.info("Skipped log entries (other engine): %d", skipped_engine)
    return samples


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmarks.confidence_calibration.parse_observe",
        description="Parse confidence_filter observe-mode log + labels, run threshold sweep.",
    )
    parser.add_argument("--log", type=Path, required=True, help="observe log file (jsonl)")
    parser.add_argument(
        "--labels", type=Path, required=True, help="labels.jsonl file (user-provided)"
    )
    parser.add_argument(
        "--engine", required=True, help="target engine ID (e.g. reazonspeech, qwen3-asr)"
    )
    parser.add_argument(
        "--signal",
        required=True,
        choices=list(SIGNAL_DIRECTION.keys()),
        help="signal field to sweep",
    )
    parser.add_argument("--threshold-min", type=float, default=None)
    parser.add_argument("--threshold-max", type=float, default=None)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument(
        "--criterion",
        choices=["f1", "youden_j", "precision", "recall"],
        default="f1",
    )
    parser.add_argument("--output", type=Path, default=Path("report.json"))
    parser.add_argument(
        "--quantization", default=None, help="metadata: e.g. int8 / float32"
    )
    parser.add_argument("--language", default=None, help="metadata: ja / en etc.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    direction = SIGNAL_DIRECTION[args.signal]

    # Default threshold range は signal 種別に応じて推定
    if args.threshold_min is None or args.threshold_max is None:
        if args.signal in ("avg_logprob",):
            default_min, default_max = -1.0, -0.05
        elif args.signal in ("token_confidence_mean",):
            default_min, default_max = 0.001, 0.5
        elif args.signal in ("no_speech_prob",):
            default_min, default_max = 0.1, 0.95
        else:
            default_min, default_max = -1.0, 1.0
        threshold_min = args.threshold_min if args.threshold_min is not None else default_min
        threshold_max = args.threshold_max if args.threshold_max is not None else default_max
    else:
        threshold_min = args.threshold_min
        threshold_max = args.threshold_max

    samples = parse_observe_log(
        log_path=args.log,
        labels_path=args.labels,
        engine=args.engine,
        signal_field=args.signal,
    )
    if not samples:
        logger.error("No matched samples after log+labels join")
        return 1

    metadata: dict[str, str] = {}
    if args.quantization:
        metadata["quantization"] = args.quantization
    if args.language:
        metadata["language"] = args.language

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
    )

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
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
