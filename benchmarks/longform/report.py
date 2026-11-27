"""長尺の評価の結果をまとめる (#488)。``summary.json`` と ``summary.md`` を書く。"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Sequence

from benchmarks.common.yodas2 import Yodas2Video

from .runner import _read_record, load_hypotheses
from .scoring import CONDITIONS, MAIN_CONDITION, score_engine

__all__ = ["summarize"]


def _speed(run_dir: Path, videos: Sequence[Yodas2Video]) -> dict:
    elapsed = duration = 0.0
    drops: dict[str, int] = {}
    seg_secs: list[float] = []
    dropped_sec: dict[str, float] = {}
    vad_secs: list[float] = []
    # VAD の区間ごとの記録: ある (0 区間も含む) / 対応が取れなかった / 無い (記録を足す前の結果)
    coverage = {"recorded": 0, "mismatch": 0, "absent": 0}
    for video in videos:
        record = _read_record(run_dir / f"{video.audio_id}.json")  # load_hypotheses と同じ検査
        elapsed += record["elapsed_sec"]
        duration += record["duration_sec"]
        seg_secs += [e - s for s, e, _ in record["segments"]]
        if "vad_segments" in record:
            coverage["recorded"] += 1
            for s, e, outcome in record["vad_segments"]:
                vad_secs.append(e - s)
                if outcome not in ("asr", "empty"):
                    dropped_sec[outcome] = dropped_sec.get(outcome, 0.0) + (e - s)
        else:
            coverage["mismatch" if "vad_segments_mismatch" in record else "absent"] += 1
        for reason, n in record["drop_counts"].items():
            drops[reason] = drops.get(reason, 0) + n
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    # 全部の動画に VAD の記録があるときだけ、VAD の値を使う。一部だけの値を全体の値に見せない
    full = coverage["recorded"] == len(videos)
    secs = vad_secs if full else seg_secs
    return {
        "rtf": elapsed / duration,
        "elapsed_sec": elapsed,
        "load_sec": run.get("load_sec"),
        "peak_gpu_reserved_mb": run.get("peak_gpu_reserved_mb"),
        "drop_counts": drops,
        # 全部に記録があれば VAD の区間 (落ちたものを含む)、無ければ ASR の出力の残った区間
        "segment_sec_median": statistics.median(secs) if secs else 0.0,
        "segment_sec_max": max(secs, default=0.0),
        "segment_basis": "vad" if full else "output",
        "vad_coverage": coverage,
        # 落ちた区間の長さの合計。全部に記録が無ければ出さない (None)
        "dropped_sec": {k: round(v, 1) for k, v in sorted(dropped_sec.items())} if full else None,
        "args": run.get("args"),
        "engine_name": run.get("engine_name"),
        # engine の書き起こし方の世代 (#518)。世代を記録する前の run.json は 1
        "engine_revision": run.get("engine_revision", 1),
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
            f"{'' if gpu is None else round(gpu)} | {sp['segment_sec_median']:.1f} / {sp['segment_sec_max']:.0f} "
            f"({'VAD' if sp['segment_basis'] == 'vad' else '出力'}) |"
        )
    lines += [
        "",
        "「そのまま」「数字をそろえる」「読みで比べる」は、表記だけ違う区間も含めた値。"
        "GPU は torch が確保した量の最大で、CTranslate2 (whispers2t) や ONNX Runtime の分は数えない。"
        "区間の長さは、全部の動画に VAD の区間ごとの記録 (`vad_segments`) があれば落ちた区間も含めた VAD の区間 (VAD)、"
        "1 本でも無ければ ASR の出力の残った区間 (出力) の長さ。",
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
        secs = sp["dropped_sec"]
        cov = sp["vad_coverage"]
        if secs is None:  # 一部の動画の値を全体の値に見せない
            seconds = f"未集計 (VAD の記録 {cov['recorded']}/{len(videos)} 本"
            seconds += f"、対応が取れなかった {cov['mismatch']} 本)" if cov["mismatch"] else ")"
        else:
            seconds = ", ".join(f"{k}={v}" for k, v in secs.items()) or "なし"
        lines.append(f"| `{label}` | {', '.join(f'{k}={v}' for k, v in sorted(drops.items())) or 'なし'} | {seconds} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
