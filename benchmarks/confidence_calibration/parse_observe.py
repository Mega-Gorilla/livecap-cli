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
from dataclasses import dataclass, field
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


# Engine display string prefix → metadata.py id の mapping (PR #339
# codex-review 3rd round fix)。display string の **先頭** が provider 名
# (NVIDIA / MistralAI) で始まる engine は first-word fallback で provider
# 名 ("nvidia" / "mistralai") を返してしまうため、明示的に prefix → id
# を pre-resolve する必要がある。
#
# 各 entry は (display string lower、metadata.py id) のタプル。
# Order は **長い prefix 優先** (parakeet_ja の "nvidia parakeet tdt ctc"
# が parakeet の "nvidia parakeet" より先に評価されること)。
#
# Source of truth: livecap_cli/engines/metadata.py:_ENGINES
#   - reazonspeech    "ReazonSpeech K2 v2"               → first-word で OK
#   - parakeet        "NVIDIA Parakeet TDT 0.6B v2"       ← prefix map 必須
#   - parakeet_ja     "NVIDIA Parakeet TDT CTC 0.6B JA"   ← prefix map 必須
#   - canary          "NVIDIA Canary 1B Flash"            ← prefix map 必須
#   - voxtral         "MistralAI Voxtral Mini 3B"          ← prefix map 必須
#   - whispers2t      "WhisperS2T base/large/..."          → first-word で OK
#   - qwen3asr        "Qwen3-ASR 0.6B" / "Qwen3-ASR 1.7B"  → alias で OK
_DISPLAY_PREFIX_MAP: list[tuple[str, str]] = [
    # 長い prefix 優先 (sort by length DESC)、parakeet_ja を parakeet より先
    ("nvidia parakeet tdt ctc", "parakeet_ja"),
    ("mistralai voxtral", "voxtral"),
    ("nvidia parakeet", "parakeet"),
    ("nvidia canary", "canary"),
]

# Engine ID aliases — `_engine_id_from_name()` 相当の normalize 結果と
# `metadata.py:_ENGINES` の `id` field が異なる engine の bridge
# (PR #339 codex-review 2nd round fix)。
#
# Qwen3-ASR の例:
#   metadata.py:159        id="qwen3asr"           ← CLI に渡す ID
#   display name           "Qwen3-ASR 0.6B"        ← log の engine field
#   normalize 1st pass     "qwen3-asr"             ← lower+first word
#   ↑ ここで hyphen 付きと無しの不一致が生じるため、alias で吸収。
#
# confidence_filter.py:162 で threshold dict key としては "qwen3-asr"
# (hyphen 付き) が使われているが、本 normalize の output は metadata.py
# CLI ID と統一する方針 (= "qwen3asr"、no hyphen)。これにより CLI から
# 渡される `--engine qwen3asr` と log の "Qwen3-ASR 0.6B" を bridge できる。
_ENGINE_ID_ALIASES: dict[str, str] = {
    "qwen3-asr": "qwen3asr",  # display "Qwen3-ASR 0.6B" → metadata.py id
}


def normalize_engine_id(name: str) -> str:
    """display string / engine ID を正規化 (PR #339 codex-review fix、3rd round 拡張)。

    Normalize の段階:

    1. ``strip + lower``
    2. **Multi-word prefix match** (``_DISPLAY_PREFIX_MAP``、長い prefix 優先) —
       display string が provider 名 (NVIDIA / MistralAI) で始まる engine
       (parakeet / parakeet_ja / canary / voxtral) を CLI ID に解決。
    3. **First-word fallback** + **alias** (``_ENGINE_ID_ALIASES``) —
       上記 prefix match で hit しない engine (reazonspeech / whispers2t /
       qwen3asr) を first-word で抽出、hyphen 付き → no-hyphen 等の alias で
       metadata.py id と一致させる。

    実例 (`livecap_cli/engines/metadata.py:_ENGINES` 全 7 engine):

    - ``"ReazonSpeech K2 (CPU, Int8)"`` → ``"reazonspeech"`` (first-word)
    - ``"ReazonSpeech K2 v2"`` → ``"reazonspeech"`` (first-word)
    - ``"NVIDIA Parakeet TDT 0.6B v2"`` → ``"parakeet"`` (prefix map)
    - ``"NVIDIA Parakeet TDT CTC 0.6B JA"`` → ``"parakeet_ja"`` (prefix map、長い側優先)
    - ``"NVIDIA Canary 1B Flash"`` → ``"canary"`` (prefix map)
    - ``"MistralAI Voxtral Mini 3B"`` → ``"voxtral"`` (prefix map)
    - ``"WhisperS2T base"`` / ``"WhisperS2T large-v3"`` → ``"whispers2t"`` (first-word)
    - ``"Qwen3-ASR 0.6B"`` / ``"Qwen3-ASR 1.7B"`` → ``"qwen3asr"`` (alias 経由)
    - ``"qwen3-asr"`` / ``"qwen3asr"`` → ``"qwen3asr"`` (alias / identity)
    - 既知 ID (e.g. ``"reazonspeech"``) → そのまま identity
    - ``""`` / 空白のみ → ``""``

    observe log の ``engine`` field は ``engine.get_engine_name()``
    (display 名) が入るため、CLI の ``--engine <metadata.py id>`` と
    match させるには本 normalize が必須。
    """
    if not name:
        return ""
    stripped = name.strip()
    if not stripped:
        return ""
    lowered = stripped.lower()

    # Stage 2: multi-word prefix match (長い側優先、list 順序が保証)
    for prefix, engine_id in _DISPLAY_PREFIX_MAP:
        if lowered.startswith(prefix):
            return engine_id

    # Stage 3: first-word fallback + alias
    primary = lowered.split()[0]
    return _ENGINE_ID_ALIASES.get(primary, primary)


@dataclass(frozen=True)
class LabelEntry:
    """label.jsonl の 1 行。"""

    source_id: str
    label: str  # speech / non_speech / noisy_speech
    text: Optional[str] = None
    occurrence_index: Optional[int] = None
    metadata: dict = field(default_factory=dict)


def load_labels(
    labels_path: Path,
) -> tuple[
    dict[tuple[str, int], LabelEntry],  # by (source_id, occurrence_index)
    dict[tuple[str, str], LabelEntry],  # by (source_id, text)
    dict[str, LabelEntry],  # by source_id alone (legacy / fallback)
]:
    """``labels.jsonl`` を 3 つの index に展開 (PR #339 codex-review fix)。

    実 observe log の ``source_id`` は ``StreamTranscriber.source_id``
    (default ``"default"``) で、複数 utterance が同 source_id を共有する
    ため、source_id 単独では multi-utterance log で衝突。本 PR では 3
    strategy で match:

    1. ``(source_id, occurrence_index)`` — user が label に
       ``"occurrence_index"`` を含めた時の primary match
    2. ``(source_id, text)`` — user が label に ``"text"`` を含めた時の
       secondary match (exact match、case sensitive、空白 trim なし)
    3. ``(source_id,)`` — legacy / fallback。**PR #339 codex-review 2nd
       round fix**: 同じ source で ``occurrence_index`` / ``text`` が
       一度でも使われている場合、その source の source-only fallback は
       **無効化** する (silent label corruption 回避)。これにより
       「partial occurrence labels では未ラベル occurrence が unmatched
       skip される」 動作になる。

    Returns:
        3 つの index dict tuple。Match strategy は呼出側
        (``parse_observe_log``) で順に try。
    """
    if not labels_path.exists():
        raise FileNotFoundError(f"labels file not found: {labels_path}")
    by_composite: dict[tuple[str, int], LabelEntry] = {}
    by_text: dict[tuple[str, str], LabelEntry] = {}
    by_source_all: dict[str, LabelEntry] = {}
    sources_with_specific_keys: set[str] = set()
    seen_source_alone: set[str] = set()
    for line_no, line in enumerate(
        labels_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("labels.jsonl line %d malformed, skipped: %s", line_no, exc)
            continue
        source_id = data.get("source_id")
        if not source_id:
            logger.warning("labels.jsonl line %d missing source_id, skipped", line_no)
            continue
        label = data.get("label", "")
        entry = LabelEntry(
            source_id=source_id,
            label=label,
            text=data.get("text"),
            occurrence_index=data.get("occurrence_index"),
            metadata={
                k: v
                for k, v in data.items()
                if k not in ("source_id", "label", "text", "occurrence_index")
            },
        )
        if entry.occurrence_index is not None:
            by_composite[(source_id, int(entry.occurrence_index))] = entry
            sources_with_specific_keys.add(source_id)
        if entry.text is not None:
            by_text[(source_id, entry.text)] = entry
            sources_with_specific_keys.add(source_id)
        if source_id in seen_source_alone:
            logger.warning(
                "labels.jsonl line %d: source_id=%r duplicated for source-only match; "
                "last entry wins (use occurrence_index or text for multi-utterance log)",
                line_no,
                source_id,
            )
        seen_source_alone.add(source_id)
        by_source_all[source_id] = entry

    # PR #339 codex-review 2nd round fix: source-only fallback は
    # specific key (occurrence_index / text) が一度も使われていない
    # source のみ有効。これにより partial occurrence labels の未ラベル
    # sample が source-only label に silently 当たる corruption を回避。
    # 例: source="default" で occurrence 0/1 label 済、occurrence 2 未 label の
    # 場合、occurrence 2 は本来 unmatched skip されるべき。
    by_source: dict[str, LabelEntry] = {
        sid: entry
        for sid, entry in by_source_all.items()
        if sid not in sources_with_specific_keys
    }
    return by_composite, by_text, by_source


def parse_observe_log(
    log_path: Path,
    labels_path: Path,
    engine: str,
    signal_field: str,
) -> list[LabeledSample]:
    """Log を parse して ``LabeledSample`` list を返す。"""
    if not log_path.exists():
        raise FileNotFoundError(f"log file not found: {log_path}")
    by_composite, by_text, by_source = load_labels(labels_path)

    # PR #339 codex-review fix: engine name は display string で log に入る
    # ため normalize して比較 (CLI --engine reazonspeech が
    # "ReazonSpeech K2 (CPU, Int8)" log と match する)。
    target_engine_id = normalize_engine_id(engine)

    samples: list[LabeledSample] = []
    unmatched = 0
    skipped_engine = 0
    # source_id ごとの出現順 counter (composite key match の occurrence_index 用)
    occurrence_by_source: dict[str, int] = {}

    for line in log_path.read_text(encoding="utf-8").splitlines():
        entry = parse_log_line(line, signal_field)
        if entry is None:
            continue
        entry_engine_id = normalize_engine_id(entry.engine)
        if entry_engine_id != target_engine_id:
            skipped_engine += 1
            continue

        # Composite key 候補 (順に try):
        occurrence = occurrence_by_source.get(entry.source_id, 0)
        occurrence_by_source[entry.source_id] = occurrence + 1

        label_entry: Optional[LabelEntry] = (
            by_composite.get((entry.source_id, occurrence))
            or (by_text.get((entry.source_id, entry.text)) if entry.text else None)
            or by_source.get(entry.source_id)
        )
        if label_entry is None:
            unmatched += 1
            continue
        if label_entry.label not in ("speech", "non_speech", "noisy_speech"):
            logger.warning(
                "source_id=%s occurrence=%d has invalid label=%r, skipped",
                entry.source_id,
                occurrence,
                label_entry.label,
            )
            continue
        samples.append(
            LabeledSample(
                signal_value=entry.signal_value,
                label=label_entry.label,  # type: ignore[arg-type]
                path=f"{entry.source_id}#{occurrence}",
                metadata={
                    "text": entry.text,
                    "engine_display": entry.engine,
                    "occurrence_index": occurrence,
                },
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
        "--engine",
        required=True,
        help=(
            "target engine ID from livecap_cli/engines/metadata.py:_ENGINES "
            "(e.g. reazonspeech, qwen3asr, parakeet, parakeet_ja, canary, "
            "voxtral, whispers2t). Display strings in observe log "
            "(e.g. 'NVIDIA Parakeet TDT 0.6B v2', 'Qwen3-ASR 0.6B') are "
            "auto-normalized to these IDs."
        ),
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
