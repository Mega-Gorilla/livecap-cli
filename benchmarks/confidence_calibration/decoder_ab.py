"""Parakeet TDT/CTC decoder A/B 比較 harness (Issue #373)。

PR #309 は parakeet_ja (TDT-CTC hybrid) を CTC decoder へ無条件切替した
(confidence signal 取得 + greedy_batch で 1.83x 高速化)。切替時の text 品質
評価は 6 clip のみで、フィールドでは かな挿入 (「正い反対」型) / 文字種混在
(「vチュber」型) / 末尾脱落の報告がある (Issue #373)。本 harness は同一
corpus に対し CTC / TDT 両 decoder で ``engine.transcribe()`` を実行し、
TDT 復帰判断に必要な 3 軸を実測する:

1. **latency**: batch=1 (production stream 経路と同じセグメント単位) の
   per-clip 処理時間と RTF (= latency / clip 長) の percentile。
2. **threshold 分離**: ``token_confidence_mean`` の speech / non_speech 分布
   と、ConfidenceFilter 現行閾値での false reject 率。
3. **幻覚 + filter 捕捉**: non_speech で「非空 text を出し、かつ
   ConfidenceFilter をすり抜ける」leak 率。

quality 面は proxy 指標のみ集計する: 正規化 text の decoder 間一致率 /
文字種サンドイッチ混在 (保守的 regex 下限) / 末尾切り捨て非対称。
corpus の参照テキスト (``reference_text_matched``) はアラインメント部分
文字列で CER のノイズが大きいため、CER は本 harness の対象外
(Issue #373 の A/B コメント参照)。

CLI usage:

    python -m benchmarks.confidence_calibration.decoder_ab \\
        --corpus-dir .tmp/calibration_corpus_full \\
        --filter-by-language ja \\
        --output .tmp/decoder_ab_parakeet_ja.json

再利用: ``pipeline.load_calibration_corpus`` / ``pipeline.resolve_corpus_dir``
/ ``parse_observe.normalize_engine_id`` / ``_core._normalize_label`` /
``livecap_cli.transcription.confidence_filter.should_reject``。
decoding cfg は ``livecap_cli/engines/parakeet_engine.py``
``_configure_decoding_with_confidence()`` の Path 1 (CTC) / Path 1.5
(RNNT/TDT) と同一内容をミラーする (drift したら本 harness の測定が
production と乖離するため、変更時は両方を更新すること)。
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ._core import _normalize_label
from .parse_observe import normalize_engine_id
from .pipeline import load_calibration_corpus, resolve_corpus_dir

logger = logging.getLogger(__name__)

# ConfidenceFilter で使う confidence_cfg (Path 1 / Path 1.5 共通部)
_CONFIDENCE_CFG = {
    "preserve_frame_confidence": True,
    "preserve_token_confidence": True,
    "preserve_word_confidence": False,
    "exclude_blank": True,
    "aggregation": "mean",
}

#: 対応 decoder → (``change_decoding_strategy`` cfg, ``decoder_type``)
DECODERS = ("ctc", "tdt")


def build_decoding_cfg(decoder: str, strategy: str = "greedy_batch") -> tuple[dict, str]:
    """decoder 名から ``change_decoding_strategy`` の (cfg, decoder_type) を返す。

    ``parakeet_engine._configure_decoding_with_confidence()`` の Path 1 (CTC)
    / Path 1.5 (RNNT/TDT) と同一構成。confidence_cfg を両 decoder に付ける
    ことで「TDT でも filter signal を維持できるか」を production 相当の
    設定で比較する。

    Raises:
        ValueError: 未知の decoder 名。
    """
    if decoder not in DECODERS:
        raise ValueError(f"decoder must be one of {DECODERS}, got {decoder!r}")
    cfg = {
        "strategy": strategy,
        "preserve_alignments": True,
        "greedy": {
            "preserve_alignments": True,
            "preserve_frame_confidence": True,
        },
        "confidence_cfg": dict(_CONFIDENCE_CFG),
    }
    decoder_type = "ctc" if decoder == "ctc" else "rnnt"
    return cfg, decoder_type


@dataclass
class ClipMeasurement:
    """1 clip x 1 decoder の測定結果。

    集計関数 (``summarize_latency`` / ``filter_confusion`` /
    ``pairwise_quality``) は本 record のみに依存し engine を必要としない
    (単体 test 可能)。
    """

    path: str
    label: str  # raw manifest label (speech / noisy_speech / non_speech)
    duration_sec: float
    text: str = ""
    signal_value: Optional[float] = None  # token_confidence_mean
    is_available: bool = True
    latency_sec: float = 0.0
    rejected: bool = False
    reject_reason: Optional[str] = None
    error: bool = False
    error_reason: Optional[str] = None

    @property
    def norm_label(self) -> str:
        """noisy_speech → speech (reject されたら false reject)。"""
        return _normalize_label(self.label)


def percentile(values: list[float], q: float) -> float:
    """線形補間 percentile (numpy 非依存、q は 0-100)。

    Raises:
        ValueError: values が空。
    """
    if not values:
        raise ValueError("percentile() requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def summarize_latency(measurements: list[ClipMeasurement]) -> dict[str, Any]:
    """error を除く全 clip の latency / RTF percentile を集計。

    RTF (real-time factor) = latency / clip 長。ストリーム経路がリアルタイム
    流入に追従するには RTF < 1 が必要条件。
    """
    ok = [m for m in measurements if not m.error]
    if not ok:
        return {"n": 0}
    lat = [m.latency_sec for m in ok]
    rtf = [m.latency_sec / m.duration_sec for m in ok if m.duration_sec > 0]
    out: dict[str, Any] = {
        "n": len(ok),
        "latency_sec": {
            "mean": sum(lat) / len(lat),
            "p50": percentile(lat, 50),
            "p90": percentile(lat, 90),
            "p95": percentile(lat, 95),
            "max": max(lat),
        },
    }
    if rtf:
        out["rtf"] = {
            "mean": sum(rtf) / len(rtf),
            "p50": percentile(rtf, 50),
            "p95": percentile(rtf, 95),
            "max": max(rtf),
        }
    return out


def signal_distribution(measurements: list[ClipMeasurement]) -> dict[str, Any]:
    """norm label ごとの signal_value (token_confidence_mean) percentile。"""
    out: dict[str, Any] = {}
    for label in ("speech", "non_speech"):
        vals = [
            m.signal_value
            for m in measurements
            if not m.error and m.norm_label == label and m.signal_value is not None
        ]
        if not vals:
            out[label] = {"n": 0}
            continue
        out[label] = {
            "n": len(vals),
            "min": min(vals),
            "p5": percentile(vals, 5),
            "p50": percentile(vals, 50),
            "p95": percentile(vals, 95),
            "max": max(vals),
        }
    return out


def filter_confusion(measurements: list[ClipMeasurement]) -> dict[str, Any]:
    """ConfidenceFilter 判定の confusion を norm label 別に集計。

    - speech (noisy_speech 含む): ``false_reject_rate`` = reject された割合
      (recall 毀損)。
    - non_speech: ``leak_rate`` = 非空 text を出し、かつ reject されなかった
      割合 (= ユーザーに見える幻覚)。空 text は engine 空text guard が落とす
      ため leak に数えない。
    """
    out: dict[str, Any] = {}
    speech = [m for m in measurements if not m.error and m.norm_label == "speech"]
    if speech:
        rejected = sum(1 for m in speech if m.rejected)
        out["speech"] = {
            "n": len(speech),
            "rejected": rejected,
            "false_reject_rate": rejected / len(speech),
        }
    non_speech = [m for m in measurements if not m.error and m.norm_label == "non_speech"]
    if non_speech:
        non_empty = [m for m in non_speech if m.text.strip()]
        leaks = [m for m in non_empty if not m.rejected]
        out["non_speech"] = {
            "n": len(non_speech),
            "non_empty_text": len(non_empty),
            "rejected": sum(1 for m in non_speech if m.rejected),
            "leak": len(leaks),
            "leak_rate": len(leaks) / len(non_speech),
            "leak_examples": [
                {"path": m.path, "text": m.text, "signal": m.signal_value}
                for m in leaks[:10]
            ],
        }
    errors = sum(1 for m in measurements if m.error)
    if errors:
        out["errors_excluded"] = errors
    return out


def normalize_text(text: str) -> str:
    """比較用正規化: NFKC + 空白 / 主要句読点を除去。"""
    text = unicodedata.normalize("NFKC", text or "")
    return "".join(
        ch for ch in text if not ch.isspace() and ch not in "、。,.!?！？…・「」『』()（）"
    )


# 片仮名連続に挟まれた平仮名 1 文字 (「ヒツじ」) / 平仮名連続に挟まれた
# 片仮名 1 文字。単語境界を持たないため「エンジンのトラブル」のような正当な
# 表記も hit する保守的下限カウント (両 decoder に同条件で適用するため
# 相対比較には使える)。漢字語中の混在 (「長がい」) は検出対象外。
SANDWICH_RE = re.compile(r"(?:[ァ-ヴー][ぁ-ゖ][ァ-ヴー])|(?:[ぁ-ゖ][ァ-ヴー][ぁ-ゖ])")


def count_script_sandwich(measurements: list[ClipMeasurement]) -> dict[str, Any]:
    """speech clip の text から文字種サンドイッチ混在を検出 (保守的下限)。"""
    hits = []
    for m in measurements:
        if m.error or m.norm_label != "speech":
            continue
        found = SANDWICH_RE.findall(m.text)
        if found:
            hits.append({"path": m.path, "text": m.text, "matched": found})
    return {"clips": len(hits), "examples": hits[:20]}


def detect_truncation(text_a: str, text_b: str, min_gap: int = 2) -> Optional[str]:
    """末尾切り捨ての片側検出。

    句読点を除いた比較で、一方が他方の真の接頭辞かつ ``min_gap`` 文字以上
    短い場合に、短い側 (``"a"`` / ``"b"``) を返す。それ以外は None。
    """
    a = normalize_text(text_a)
    b = normalize_text(text_b)
    if a == b:
        return None
    if b.startswith(a) and len(b) - len(a) >= min_gap:
        return "a"
    if a.startswith(b) and len(a) - len(b) >= min_gap:
        return "b"
    return None


def pairwise_quality(
    ctc: list[ClipMeasurement], tdt: list[ClipMeasurement]
) -> dict[str, Any]:
    """speech clip の decoder 間 text 比較 (一致率 / 末尾切り捨て非対称)。"""
    tdt_by_path = {m.path: m for m in tdt}
    pairs = [
        (c, tdt_by_path[c.path])
        for c in ctc
        if c.path in tdt_by_path
        and not c.error
        and not tdt_by_path[c.path].error
        and c.norm_label == "speech"
    ]
    agree = sum(1 for c, t in pairs if normalize_text(c.text) == normalize_text(t.text))
    trunc_ctc: list[dict[str, str]] = []
    trunc_tdt: list[dict[str, str]] = []
    for c, t in pairs:
        side = detect_truncation(c.text, t.text)
        if side == "a":
            trunc_ctc.append({"path": c.path, "ctc": c.text, "tdt": t.text})
        elif side == "b":
            trunc_tdt.append({"path": c.path, "ctc": c.text, "tdt": t.text})
    return {
        "speech_pairs": len(pairs),
        "text_agreement": agree,
        "text_agreement_rate": (agree / len(pairs)) if pairs else None,
        "truncation_ctc": len(trunc_ctc),
        "truncation_tdt": len(trunc_tdt),
        "truncation_ctc_examples": trunc_ctc[:10],
        "truncation_tdt_examples": trunc_tdt[:10],
    }


def run_condition(
    engine: Any,
    engine_name: str,
    items: list[Any],  # CalibrationCorpusItem
    decoder: str,
    filter_config: Any,
    warmup: int = 3,
) -> list[ClipMeasurement]:
    """1 decoder 条件で全 clip を batch=1 実行し測定する。

    decoder 切替 → warmup (最初の clip を ``warmup`` 回、記録なし) →
    全 clip を 1 つずつ transcribe。timing は ``time.perf_counter``
    (transcribe は text を返すため GPU 完了と同期済み)。
    """
    from livecap_cli.transcription.confidence_filter import should_reject

    cfg, decoder_type = build_decoding_cfg(decoder)
    engine.model.change_decoding_strategy(cfg, decoder_type=decoder_type)
    logger.info("Decoder switched: %s (decoder_type=%s)", decoder, decoder_type)

    if items and warmup > 0:
        for _ in range(warmup):
            try:
                engine.transcribe(items[0].audio, items[0].sample_rate)
            except Exception as exc:  # pragma: no cover - warmup failure は続行
                logger.warning("Warmup transcribe failed: %s", exc)
                break

    results: list[ClipMeasurement] = []
    for idx, item in enumerate(items):
        duration = len(item.audio) / float(item.sample_rate)
        base = ClipMeasurement(
            path=str(item.path), label=item.label, duration_sec=duration
        )
        try:
            t0 = time.perf_counter()
            result = engine.transcribe(item.audio, item.sample_rate)
            base.latency_sec = time.perf_counter() - t0
        except Exception as exc:
            logger.warning(
                "engine.transcribe() failed for %s (%d/%d): %s",
                item.path,
                idx + 1,
                len(items),
                exc,
            )
            base.error = True
            base.error_reason = str(exc)
            results.append(base)
            continue

        ec = result.engine_confidence
        base.text = result.text
        base.signal_value = getattr(ec, "token_confidence_mean", None)
        base.is_available = bool(getattr(ec, "is_available", True))
        rejected, reason = should_reject(result, filter_config, engine_name=engine_name)
        base.rejected = bool(rejected)
        base.reject_reason = reason
        results.append(base)
        if (idx + 1) % 50 == 0:
            logger.info("[%s] measured %d/%d clips", decoder, idx + 1, len(items))
    return results


def _measurement_to_dict(m: ClipMeasurement) -> dict[str, Any]:
    return {
        "path": m.path,
        "label": m.label,
        "duration_sec": round(m.duration_sec, 3),
        "text": m.text,
        "signal_value": m.signal_value,
        "is_available": m.is_available,
        "latency_sec": round(m.latency_sec, 4),
        "rejected": m.rejected,
        "reject_reason": m.reject_reason,
        "error": m.error,
        "error_reason": m.error_reason,
    }


def _env_metadata() -> dict[str, Any]:
    """再現性のための実行環境記録 (GPU / torch / cuda-python 有無)。

    RNNT decoder は ``cuda-python`` 不在だと NeMo の CUDA graphs 高速化が
    無効になり decode が遅くなるため、latency 解釈に必須の記録。
    """
    meta: dict[str, Any] = {}
    try:
        import torch

        meta["torch"] = torch.__version__
        meta["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            meta["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover - torch 無し環境
        meta["cuda_available"] = False
    import importlib.util

    meta["has_cuda_python"] = importlib.util.find_spec("cuda") is not None
    return meta


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmarks.confidence_calibration.decoder_ab",
        description=(
            "Parakeet TDT/CTC decoder A/B: load corpus → 両 decoder で "
            "engine.transcribe() (batch=1) → latency / threshold 分離 / "
            "leak 率 / text 品質 proxy を report.json に出力 (Issue #373)。"
        ),
    )
    parser.add_argument(
        "--engine",
        default="parakeet_ja",
        help="EngineFactory の engine 名 (default: parakeet_ja。TDT-CTC hybrid のみ対応)",
    )
    parser.add_argument(
        "--corpus-dir",
        default=None,
        help="corpus root (default: LIVECAP_CALIBRATION_CORPUS_DIR → OS data dir)",
    )
    parser.add_argument("--manifest-name", default="manifest.jsonl")
    parser.add_argument(
        "--filter-by-language", default=None, help="manifest language filter (例: ja)"
    )
    parser.add_argument(
        "--labels",
        default="speech,noisy_speech,non_speech",
        help="対象 label (comma 区切り、default: 全 3 label)",
    )
    parser.add_argument(
        "--limit-per-label",
        type=int,
        default=None,
        help="label ごとの clip 数上限 (smoke 用)",
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="decoder 切替後の warmup 回数 (default: 3)"
    )
    parser.add_argument(
        "--output", default=None, help="report JSON の出力先 (default: stdout summary のみ)"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    corpus_dir = (
        Path(args.corpus_dir).expanduser().resolve()
        if args.corpus_dir
        else resolve_corpus_dir()
    )
    logger.info("Corpus dir: %s", corpus_dir)
    items = load_calibration_corpus(corpus_dir, manifest_name=args.manifest_name)
    if args.filter_by_language:
        items = [
            it for it in items if it.metadata.get("language") == args.filter_by_language
        ]
    wanted = {p.strip() for p in args.labels.split(",") if p.strip()}
    items = [it for it in items if it.label in wanted]
    if args.limit_per_label is not None:
        limited: list[Any] = []
        counts: dict[str, int] = {}
        for it in items:
            if counts.get(it.label, 0) < args.limit_per_label:
                limited.append(it)
                counts[it.label] = counts.get(it.label, 0) + 1
        items = limited
    if not items:
        logger.error("No corpus items after filtering")
        return 1
    label_counts: dict[str, int] = {}
    for it in items:
        label_counts[it.label] = label_counts.get(it.label, 0) + 1
    logger.info("Corpus items: %d %s", len(items), label_counts)

    from livecap_cli.engines.engine_factory import EngineFactory
    from livecap_cli.transcription.confidence_filter import FilterConfig

    engine = EngineFactory.create_engine(args.engine)
    engine.load_model()
    if not hasattr(engine, "model") or not hasattr(engine.model, "cur_decoder"):
        logger.error(
            "engine %s は TDT-CTC hybrid ではない (cur_decoder なし)。"
            "本 harness は hybrid model 専用。",
            args.engine,
        )
        return 1
    engine_name = normalize_engine_id(engine.get_engine_name())
    filter_config = FilterConfig(mode="on")

    conditions: dict[str, list[ClipMeasurement]] = {}
    for decoder in DECODERS:
        logger.info("=== condition: %s ===", decoder)
        conditions[decoder] = run_condition(
            engine, engine_name, items, decoder, filter_config, warmup=args.warmup
        )

    report: dict[str, Any] = {
        "metadata": {
            "engine": args.engine,
            "engine_display": engine.get_engine_name(),
            "engine_normalized": engine_name,
            "corpus_dir": str(corpus_dir),
            "labels": sorted(wanted),
            "label_counts": label_counts,
            "warmup": args.warmup,
            "token_conf_threshold": filter_config.token_conf_threshold,
            **_env_metadata(),
        },
        "conditions": {
            name: {
                "latency": summarize_latency(ms),
                "signal_distribution": signal_distribution(ms),
                "filter_confusion": filter_confusion(ms),
                "script_sandwich": count_script_sandwich(ms),
            }
            for name, ms in conditions.items()
        },
        "pairwise_quality": pairwise_quality(conditions["ctc"], conditions["tdt"]),
        "rows": {
            name: [_measurement_to_dict(m) for m in ms]
            for name, ms in conditions.items()
        },
    }

    summary = {k: v for k, v in report.items() if k != "rows"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        logger.info("Report written: %s", out_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
