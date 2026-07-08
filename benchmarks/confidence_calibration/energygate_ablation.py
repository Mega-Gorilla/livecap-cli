"""EnergyGate(#292) 限界寄与 ablation harness (Issue #357)。

realtime path の無音/非音声 drop は3層が担う: **VAD → EnergyGate(RMS gate,
既定 ON -45dBFS) → engine 空text guard(常時 ON) → ConfidenceFilter(#334,
既定 ON)**。ConfidenceFilter 単体の有効性は Phase 2 report で実測済み。本 harness
は **EnergyGate が ConfidenceFilter / 空text guard がある上で追加で必要か
(marginal necessity)** を data で判定する。

手法: corpus 全 sample に engine を1回実行し、3 guard の判定を **独立に記録**
してから 4 config (baseline / +energy / +confidence / both) を simulate する。
EnergyGate は本来 pre-engine だが、marginal を測るため engine は全件走らせる
(EnergyGate が落とす sample を「もし engine に通したら ConfidenceFilter /
空text guard が捕捉したか」を知るため)。

Confusion の positive class は ``non_speech`` (= 落とすべき)。EnergyGate の
**ユニーク寄与** = non_speech で ``energy_drop and not empty_text and not
conf_reject`` の件数 (= EnergyGate だけが救う真の付加価値)。EnergyGate の
**害** = speech で同条件の件数 (recall 毀損)。

CLI usage:

    python -m benchmarks.confidence_calibration.energygate_ablation \\
        --engine reazonspeech --corpus-dir .tmp/calibration_corpus_full \\
        --filter-by-language ja --output .tmp/energygate_ablation_reazonspeech.json

再利用: ``pipeline.load_calibration_corpus`` / ``_core._normalize_label`` /
``parse_observe.normalize_engine_id`` / ``livecap_cli.audio.analysis.
_segment_energy_dbfs`` / ``livecap_cli.transcription.confidence_filter.
should_reject``。engine 実行 loop は ``sweep.measure_signals`` を踏襲。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ._core import _normalize_label
from .parse_observe import normalize_engine_id
from .pipeline import load_calibration_corpus, resolve_corpus_dir

logger = logging.getLogger(__name__)

# EnergyGate (StreamTranscriber) の既定と一致させる
DEFAULT_ENERGY_THRESHOLD_DBFS = -45.0
ENERGY_METRIC = "max_frame_rms"
ENERGY_FRAME_MS = 32.0

# report 用: engine → 主 signal field (ConfidenceFilter の判定自体は
# should_reject が全 signal を見るため本 map には依存しない。表示専用)
_ENGINE_SIGNAL: dict[str, str] = {
    "reazonspeech": "avg_logprob",
    "qwen3asr": "avg_logprob",
    "voxtral": "avg_logprob",
    "whispers2t": "no_speech_prob",
    "parakeet": "token_confidence_mean",
    "parakeet_ja": "token_confidence_mean",
    "canary": "token_confidence_mean",
}


@dataclass
class GuardRecord:
    """1 sample に対する 3 guard の独立判定 + label。

    simulate ロジックは本 record のみに依存し engine を必要としない
    (単体 test 可能)。
    """

    path: str
    label: str  # raw manifest label (speech / noisy_speech / non_speech)
    energy_dbfs: float
    energy_drop: bool  # energy_dbfs < energy_threshold
    empty_text: bool  # engine 出力が空 (空text guard が drop)
    conf_reject: bool  # ConfidenceFilter が reject
    is_available: bool = True  # engine_confidence.is_available (fail-open 追跡)
    signal_value: Optional[float] = None
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def norm_label(self) -> str:
        """noisy_speech → speech (落とすと false reject)。"""
        return _normalize_label(self.label)


# --- 4 config の drop 判定 (pure、engine 非依存) ---
# drop = suppression。non_speech なら正解、speech なら有害。
# 空text guard (empty_text) は EnergyGate/ConfidenceFilter とは別の常時 ON guard。
CONFIG_DROP: dict[str, Callable[[GuardRecord], bool]] = {
    "baseline": lambda r: r.empty_text,
    "energy": lambda r: r.energy_drop or r.empty_text,
    "confidence": lambda r: r.empty_text or r.conf_reject,
    "both": lambda r: r.energy_drop or r.empty_text or r.conf_reject,
}
CONFIG_ORDER = ["baseline", "energy", "confidence", "both"]


def summarize(records: list[GuardRecord]) -> dict[str, Any]:
    """records から 4-config confusion + EnergyGate marginal 分解を算出。

    Returns:
        dict: ``configs`` (config→{non_speech_suppressed/total, speech_dropped/total,
        suppression_rate, false_drop_rate}) + ``energy_gate_marginal`` (ユニーク寄与
        / overlap / 害) + ``silence_hallucination`` (無音入力で engine 非空・高信頼)。
    """
    n_speech = sum(1 for r in records if r.norm_label == "speech")
    n_nonsp = sum(1 for r in records if r.norm_label == "non_speech")

    configs: dict[str, Any] = {}
    for name in CONFIG_ORDER:
        drop = CONFIG_DROP[name]
        ns_sup = sum(1 for r in records if r.norm_label == "non_speech" and drop(r))
        sp_drop = sum(1 for r in records if r.norm_label == "speech" and drop(r))
        configs[name] = {
            "non_speech_suppressed": ns_sup,
            "non_speech_total": n_nonsp,
            "suppression_rate": ns_sup / n_nonsp if n_nonsp else 0.0,
            "speech_dropped": sp_drop,
            "speech_total": n_speech,
            "false_drop_rate": sp_drop / n_speech if n_speech else 0.0,
        }

    # EnergyGate marginal: energy が落とすが他 guard は捕捉しない sample
    def energy_unique(r: GuardRecord) -> bool:
        return r.energy_drop and not r.empty_text and not r.conf_reject

    def energy_overlap(r: GuardRecord) -> bool:
        return r.energy_drop and (r.empty_text or r.conf_reject)

    ns = [r for r in records if r.norm_label == "non_speech"]
    sp = [r for r in records if r.norm_label == "speech"]
    marginal = {
        # non_speech: EnergyGate だけが救う真の付加価値
        "non_speech_energy_unique": sum(1 for r in ns if energy_unique(r)),
        "non_speech_energy_overlap": sum(1 for r in ns if energy_overlap(r)),
        "non_speech_energy_total_drop": sum(1 for r in ns if r.energy_drop),
        # speech: EnergyGate が新たに害する件数 (他 guard が落とさないのに energy が落とす)
        "speech_energy_unique_harm": sum(1 for r in sp if energy_unique(r)),
        "speech_energy_total_drop": sum(1 for r in sp if r.energy_drop),
    }

    # 無音入力 (all-zero / 極低 RMS) で engine が非空・ConfidenceFilter も pass →
    # EnergyGate 相補性の核心証拠。energy_drop な non_speech のうち engine が幻聴した件数。
    silent_ns = [r for r in ns if r.energy_drop]
    silence_hallucination = {
        "silent_non_speech_total": len(silent_ns),
        "engine_nonempty": sum(1 for r in silent_ns if not r.empty_text),
        "engine_nonempty_conf_pass": sum(
            1 for r in silent_ns if not r.empty_text and not r.conf_reject
        ),
        "examples": [
            {"path": r.path, "energy_dbfs": round(r.energy_dbfs, 1), "text": r.text[:40]}
            for r in silent_ns
            if not r.empty_text and not r.conf_reject
        ][:10],
    }

    return {
        "sample_count": {"speech": n_speech, "non_speech": n_nonsp, "total": len(records)},
        "configs": configs,
        "energy_gate_marginal": marginal,
        "silence_hallucination": silence_hallucination,
    }


def _print_table(summary: dict[str, Any], engine_display: str, threshold: float) -> None:
    sc = summary["sample_count"]
    print(f"\n=== EnergyGate ablation: {engine_display} (energy thr={threshold} dBFS) ===")
    print(f"samples: speech(+noisy)={sc['speech']}  non_speech={sc['non_speech']}\n")
    print(f"{'config':<12}{'ns_suppress':>13}{'ns_supp%':>10}{'sp_drop':>9}{'sp_FRR%':>9}")
    for name in CONFIG_ORDER:
        c = summary["configs"][name]
        print(f"{name:<12}{c['non_speech_suppressed']:>7}/{c['non_speech_total']:<5}"
              f"{100*c['suppression_rate']:>9.1f}%{c['speech_dropped']:>9}"
              f"{100*c['false_drop_rate']:>8.1f}%")
    m = summary["energy_gate_marginal"]
    print("\n-- EnergyGate marginal --")
    print(f"  non_speech: energy drops {m['non_speech_energy_total_drop']} "
          f"(unique={m['non_speech_energy_unique']}, overlap w/ other guards="
          f"{m['non_speech_energy_overlap']})")
    print(f"  speech harm: energy drops {m['speech_energy_total_drop']} "
          f"(net-new harm={m['speech_energy_unique_harm']})")
    h = summary["silence_hallucination"]
    print(f"\n-- silence hallucination (energy-dropped non_speech) --")
    print(f"  {h['silent_non_speech_total']} silent non_speech; engine non-empty="
          f"{h['engine_nonempty']}, of which ConfidenceFilter PASSES="
          f"{h['engine_nonempty_conf_pass']}  <- EnergyGate-only saves")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="EnergyGate(#292) marginal ablation over calibration corpus"
    )
    p.add_argument("--engine", required=True, help="engine id (e.g. reazonspeech, whispers2t)")
    p.add_argument("--corpus-dir", type=Path, default=None,
                   help="corpus dir (manifest.jsonl + audio). default: env / OS data dir")
    p.add_argument("--manifest-name", default="manifest.jsonl")
    p.add_argument("--filter-by-language", default=None, help="e.g. ja, en")
    p.add_argument("--energy-threshold", type=float, default=DEFAULT_ENERGY_THRESHOLD_DBFS,
                   help=f"EnergyGate dBFS threshold (default {DEFAULT_ENERGY_THRESHOLD_DBFS})")
    p.add_argument("--engine-kwargs", nargs="*", default=[],
                   help="extra engine kwargs key=value (e.g. use_int8=true model_size=base)")
    p.add_argument("--limit", type=int, default=None, help="cap sample count (smoke)")
    p.add_argument("--output", type=Path, default=Path("energygate_ablation.json"))
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _parse_engine_kwargs(raw: list[str]) -> dict[str, Any]:
    """``key=value`` list → dict (bool/int/float/str 自動推論)。sweep.py と同挙動。"""
    parsed: dict[str, Any] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"--engine-kwargs entry must be key=value, got: {item!r}")
        key, value = item.split("=", 1)
        key, value = key.strip(), value.strip()
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


def measure_guards(
    items: list[Any],
    engine: Any,
    energy_threshold: float,
    signal_field: Optional[str],
) -> list[GuardRecord]:
    """全 sample に engine を1回実行し 3 guard を独立判定 (measure_signals 踏襲)。"""
    # 遅延 import (torch を top-level に持ち込まない、単体 test を軽く保つ)
    from livecap_cli.audio.analysis import _segment_energy_dbfs
    from livecap_cli.transcription.confidence_filter import FilterConfig, should_reject

    config = FilterConfig()  # 現行 main の production 既定閾値
    engine_display = engine.get_engine_name()
    records: list[GuardRecord] = []
    for idx, item in enumerate(items):
        energy_dbfs = _segment_energy_dbfs(
            item.audio, item.sample_rate, ENERGY_METRIC, ENERGY_FRAME_MS
        )
        energy_drop = energy_dbfs < energy_threshold
        try:
            result = engine.transcribe(item.audio, item.sample_rate)
        except Exception as exc:  # measure_signals と同じく catch (無音で crash 等)
            logger.warning("transcribe failed for %s (%d/%d): %s",
                           item.path, idx + 1, len(items), exc)
            records.append(GuardRecord(
                path=str(item.path), label=item.label, energy_dbfs=energy_dbfs,
                energy_drop=energy_drop, empty_text=True, conf_reject=False,
                is_available=False, signal_value=None, text="",
                metadata={**item.metadata, "transcribe_error": str(exc)},
            ))
            continue
        ec = result.engine_confidence
        rejected, _reason = should_reject(result, config, engine_name=engine_display)
        records.append(GuardRecord(
            path=str(item.path), label=item.label, energy_dbfs=energy_dbfs,
            energy_drop=energy_drop, empty_text=(result.text.strip() == ""),
            conf_reject=rejected, is_available=ec.is_available,
            signal_value=getattr(ec, signal_field, None) if signal_field else None,
            text=result.text, metadata=dict(item.metadata),
        ))
        if (idx + 1) % 50 == 0:
            logger.info("measured %d/%d", idx + 1, len(items))
    return records


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    corpus_dir = args.corpus_dir or resolve_corpus_dir()
    if not corpus_dir.exists():
        logger.error("corpus directory does not exist: %s", corpus_dir)
        return 1

    logger.info("Loading corpus from %s ...", corpus_dir)
    items = load_calibration_corpus(corpus_dir, manifest_name=args.manifest_name)
    if args.filter_by_language:
        items = [it for it in items
                 if it.metadata.get("language") == args.filter_by_language]
    if args.limit:
        items = items[: args.limit]
    if not items:
        logger.error("No corpus items after filtering")
        return 1
    logger.info("Loaded %d corpus items", len(items))

    from livecap_cli.engines.engine_factory import EngineFactory

    engine_kwargs = _parse_engine_kwargs(args.engine_kwargs)
    logger.info("Creating engine %s kwargs=%s", args.engine, engine_kwargs)
    engine = EngineFactory.create_engine(args.engine, **engine_kwargs)
    engine.load_model()

    engine_id = normalize_engine_id(engine.get_engine_name())
    signal_field = _ENGINE_SIGNAL.get(engine_id)

    records = measure_guards(items, engine, args.energy_threshold, signal_field)
    summary = summarize(records)
    summary["metadata"] = {
        "engine": args.engine,
        "engine_id": engine_id,
        "engine_display": engine.get_engine_name(),
        "signal_field": signal_field,
        "energy_threshold_dbfs": args.energy_threshold,
        "energy_metric": ENERGY_METRIC,
        "energy_frame_ms": ENERGY_FRAME_MS,
        "corpus_dir": str(corpus_dir),
        "language": args.filter_by_language,
        "engine_kwargs": engine_kwargs,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # per-sample records も dump (再解析用)
    out = dict(summary)
    out["records"] = [
        {
            "path": r.path, "label": r.label, "energy_dbfs": round(r.energy_dbfs, 2),
            "energy_drop": r.energy_drop, "empty_text": r.empty_text,
            "conf_reject": r.conf_reject, "is_available": r.is_available,
            "signal_value": r.signal_value, "text": r.text,
            "snr_db": r.metadata.get("snr_db"),
            "source_dataset": r.metadata.get("source_dataset"),
            "subtype": r.metadata.get("subtype"),
        }
        for r in records
    ]
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_table(summary, engine.get_engine_name(), args.energy_threshold)
    logger.info("Wrote %s (%d records)", args.output, len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
