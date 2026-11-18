"""長尺の評価の結果をまとめる (#488)。``summary.json`` と ``summary.md`` を書く。"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Sequence

from benchmarks.common.yodas2 import Yodas2Video

from .runner import load_hypotheses
from .scoring import CONDITIONS, MAIN_CONDITION, score_engine

__all__ = ["summarize"]


def _speed(run_dir: Path, videos: Sequence[Yodas2Video]) -> dict:
    elapsed = duration = 0.0
    drops: dict[str, int] = {}
    seg_secs: list[float] = []
    dropped_sec: dict[str, float] = {}
    vad_secs: list[float] = []
    for video in videos:
        record = json.loads((run_dir / f"{video.audio_id}.json").read_text(encoding="utf-8"))
        elapsed += record["elapsed_sec"]
        duration += record["duration_sec"]
        seg_secs += [e - s for s, e, _ in record["segments"]]
        for s, e, outcome in record.get("vad_segments", []):
            vad_secs.append(e - s)
            if outcome not in ("asr", "empty"):
                dropped_sec[outcome] = dropped_sec.get(outcome, 0.0) + (e - s)
        for reason, n in record["drop_counts"].items():
            drops[reason] = drops.get(reason, 0) + n
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    return {
        "rtf": elapsed / duration,
        "elapsed_sec": elapsed,
        "load_sec": run.get("load_sec"),
        "peak_gpu_reserved_mb": run.get("peak_gpu_reserved_mb"),
        "drop_counts": drops,
        # VAD の区間 (落ちたものを含む) の記録があればそれを、無ければ出力の残った区間を使う
        "segment_sec_median": statistics.median(vad_secs or seg_secs) if (vad_secs or seg_secs) else 0.0,
        "segment_sec_max": max(vad_secs or seg_secs, default=0.0),
        "segment_basis": "vad" if vad_secs else "output",
        "dropped_sec": {k: round(v, 1) for k, v in sorted(dropped_sec.items())},
        "args": run.get("args"),
        "engine_name": run.get("engine_name"),
        "versions": run.get("versions"),
    }


def _pct(x: float) -> str:
    return f"{x * 100:.2f}"


def summarize(out_dir: Path, labels: Sequence[str], videos: Sequence[Yodas2Video], *, split: str) -> dict:
    """``labels`` の ASR を採点し、``out_dir/summary.json`` と ``out_dir/summary.md`` を書く。"""
    results: dict[str, dict] = {}
    for label in labels:
        run_dir = out_dir / label
        hyps = load_hypotheses(run_dir, videos)
        scores = {c.key: score_engine(videos, hyps, c) for c in CONDITIONS}
        results[label] = {"speed": _speed(run_dir, videos), "scores": {k: s.to_dict() for k, s in scores.items()}}

    hours = sum(v.duration_sec for v in videos) / 3600
    speakers = sorted({v.speaker for v in videos})
    summary = {
        "split": split,
        "videos": len(videos),
        "speakers": len(speakers),
        "hours": round(hours, 2),
        "main_condition": MAIN_CONDITION.key,
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    ranked = sorted(labels, key=lambda lb: results[lb]["scores"][MAIN_CONDITION.key]["pooled"]["rate"])
    lines = [
        f"# 長尺の評価: YODAS2 日本語 `{split}` ({len(videos)} 本、{len(speakers)} 人、{hours:.1f} 時間)",
        "",
        f"主の値は `{MAIN_CONDITION.key}` (片方にしか無い口ぐせを数えない、表記だけ違う区間を除く) の CER (%)。"
        "全体は誤りの総数 ÷ 正解の総文字数、話者平均は話者ごとの CER の平均。",
        "",
        "| ASR | CER | 置換 | 脱落 | 挿入 | 話者平均 | そのまま | 数字をそろえる | 読みで比べる | RTF | 読み込み (秒) | GPU (torch, MB) | 区間の長さ (中央 / 最長, 秒) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ranked:
        sc = results[label]["scores"]
        main = sc[MAIN_CONDITION.key]
        p = main["pooled"]
        n = p["reference_length"]
        sp = results[label]["speed"]
        gpu = sp["peak_gpu_reserved_mb"]
        lines.append(
            f"| `{label}` | **{_pct(p['rate'])}** | {_pct(p['substitutions'] / n)} | {_pct(p['deletions'] / n)} | "
            f"{_pct(p['insertions'] / n)} | {_pct(main['speaker_mean_rate'])} | "
            f"{_pct(sc['plain']['pooled']['rate'])} | {_pct(sc['digits']['pooled']['rate'])} | "
            f"{_pct(sc['kana']['pooled']['rate'])} | {sp['rtf']:.3f} | {sp['load_sec']} | "
            f"{'' if gpu is None else round(gpu)} | {sp['segment_sec_median']:.1f} / {sp['segment_sec_max']:.0f} |"
        )
    lines += [
        "",
        "「そのまま」「数字をそろえる」「読みで比べる」は、表記だけ違う区間も含めた値。"
        "GPU は torch が確保した量の最大で、CTranslate2 (whispers2t) や ONNX Runtime の分は数えない。"
        "区間の長さは、VAD で区切って ASR に渡した区間 (出力の残ったもの) の長さ。",
        "",
        f"## 話者ごとの CER (`{MAIN_CONDITION.key}`、%)",
        "",
        "| ASR | " + " | ".join(speakers) + " |",
        "|---|" + "---:|" * len(speakers),
    ]
    for label in ranked:
        by = results[label]["scores"][MAIN_CONDITION.key]["by_speaker"]
        lines.append(f"| `{label}` | " + " | ".join(_pct(by[s]["rate"]) for s in speakers) + " |")
    lines += [
        "", "## drop の内訳 (EnergyGate / confidence filter で落ちた区間の数と、その長さの合計)", "",
        "| ASR | 区間の数 | 長さの合計 (秒) |", "|---|---|---|",
    ]
    for label in ranked:
        sp = results[label]["speed"]
        drops = sp["drop_counts"]
        secs = sp.get("dropped_sec") or {}
        lines.append(
            f"| `{label}` | {', '.join(f'{k}={v}' for k, v in sorted(drops.items())) or 'なし'} | "
            f"{', '.join(f'{k}={v}' for k, v in secs.items()) or ('記録なし' if sp.get('segment_basis') != 'vad' else 'なし')} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
