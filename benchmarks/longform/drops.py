"""EnergyGate と confidence filter が捨てる音声を、区間ごとの記録から計算する (#488 段階 2)。

EnergyGate を外して書き起こした記録 (``transcribe --engine-min-rms off``) には、全部の VAD の区間について、
EnergyGate が見る音量と、filter にかける前の ASR の文と confidence が残る (:func:`~.runner.load_segment_details`)。
ここから、EnergyGate と filter の閾値を変えたときの出力を **CLI と同じ判定** (EnergyGate は ``音量 < 閾値`` で
捨てる、filter は :func:`~livecap_cli.transcription.confidence_filter.should_reject`) で計算し、採点する。
区間ごとの認識は前の区間に左右されないので (pipeline は区間をまたぐ文脈を ASR に渡さない)、捨てる区間を
変えても、残る区間の文は変わらない。

出すもの (``<out_dir>/drops.json`` と ``drops.md``。書き起こしの本文は入れない):

- 自己照合: 記録を書き起こしたときの設定で計算すると、記録の結果 (区間ごとの判定と出力) と同じになるか
- 照合 (``verify_dir``): CLI の既定の設定で実際に書き起こした結果と、計算した既定の結果が、区間ごとの判定と
  文まで同じか (CER が合うだけでは、違う誤りが打ち消し合っているかもしれないため)
- 閾値ごとの CER と、置換・脱落・挿入、捨てた区間の数と長さ (filter の閾値と、EnergyGate の閾値)
- 既定で filter が捨てた区間の分類: 字幕と重なるか、ASR の文が字幕と合うか、ほかの ASR (``reference_label``)
  がその時間に文を出したか (照らさなければ「不明」)。判定が割れた区間と不明の区間は、聞いて確かめる候補として
  ``drops-candidates/candidates.json`` に書く (文を含むので commit しない。フォルダに .gitignore を置く)
- VAD がどの区間にも入れなかった字幕の文字数 (VAD の取りこぼしの目安)
- 字幕の無い時間に出た文の量 (filter が防ぐはずのもの。字幕が付いていないだけの発話も入る)

字幕の時刻は人が確かめたものではないので、字幕との重なりで数える値は目安。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from benchmarks.common.yodas2 import Yodas2Video

from .runner import SegmentDetail, _json_value, load_hypotheses, load_segment_details
from .scoring import MAIN_CONDITION, HypSegment, score_engine

if TYPE_CHECKING:
    from livecap_cli.transcription.confidence_filter import FilterConfig

__all__ = [
    "Setting",
    "analyze",
    "analyze_label",
    "default_setting",
    "filter_signal",
    "gate_off_labels",
    "simulate",
]

#: 区間ごとの判定 (``vad_segments`` の結果と同じ語)
ASR, EMPTY, GATE, FILTER = "asr", "empty", "energy_gate:low_rms", "confidence_filter:reject"
#: 字幕と重なるかを見るときの、字幕の時刻のゆとり (秒)。字幕の時刻は人が確かめたものではない
CAPTION_TOLERANCE_SEC = 0.5
#: ASR の文が字幕 (またはほかの ASR の文) と「合う」とみなす、一致した文字の割合
MATCH_PRECISION = 0.5
#: 分析が計算で変える設定。既定の実測との照合で、分析元とそろわなくてよい
_ANALYZED_ARGS = ("engine_min_rms", "confidence_filter")
#: 聞いて確かめる候補にする分類 (判定が割れる 2 つと、ほかの ASR と照らしていない 1 つ)
_CANDIDATE_CLASSES = ("caption_mismatch", "no_caption_other_spoke", "no_caption_unknown")
#: 聞いて確かめる候補の、ASR ごと・分類ごとの上限
MAX_CANDIDATES_PER_CLASS = 10
#: 聞いて確かめる候補の置き場所 (出力先の下)。ASR の文を含むので、このフォルダに ``*`` だけの .gitignore を置く
CANDIDATES_DIR = "drops-candidates"


@dataclass(frozen=True)
class Setting:
    """計算する設定。``filter`` が ``None`` なら filter を外す。``min_rms_dbfs`` が ``-inf`` なら EnergyGate を外す。"""

    name: str
    min_rms_dbfs: float
    filter: FilterConfig | None


def default_setting() -> Setting:
    """CLI の既定 (EnergyGate -45 dBFS、filter ``on`` の既定の閾値)。"""
    from livecap_cli.audio import DEFAULT_ENGINE_MIN_RMS_DBFS
    from livecap_cli.transcription.confidence_filter import FilterConfig

    return Setting("既定", DEFAULT_ENGINE_MIN_RMS_DBFS, FilterConfig(mode="on"))


def simulate(
    details: Sequence[SegmentDetail], setting: Setting, *, engine_name: str
) -> tuple[list[HypSegment], list[str]]:
    """``setting`` で書き起こしたときの出力と、区間ごとの判定を計算する。

    CLI と同じ順で判定する: EnergyGate (``音量 < 閾値`` で捨てる) → filter (``should_reject``) → 空の出力を除く。
    出力の文は、pipeline と同じく前後の空白を除く。

    Raises:
        ValueError: ASR を呼んでいない区間を、この設定では ASR に渡すことになる (記録の EnergyGate より低い
            閾値は計算できない。EnergyGate を外した記録を使う)
    """
    from livecap_cli.engines.base_engine import TranscriptionResult
    from livecap_cli.transcription.confidence_filter import should_reject

    hyps: list[HypSegment] = []
    outcomes: list[str] = []
    for d in details:
        if d.energy_dbfs < setting.min_rms_dbfs:
            outcomes.append(GATE)
            continue
        if d.text is None or d.confidence is None:
            raise ValueError(
                f"segment {d.start}-{d.end} was not sent to the ASR when it was transcribed "
                f"({d.outcome}); use a run made with the EnergyGate off"
            )
        if setting.filter is not None and setting.filter.mode == "on":
            result = TranscriptionResult(text=d.text, confidence=1.0, engine_confidence=d.confidence)
            if should_reject(result, setting.filter, engine_name=engine_name)[0]:
                outcomes.append(FILTER)
                continue
        text = d.text.strip()
        outcomes.append(ASR if text else EMPTY)
        if text:
            hyps.append((d.start, d.end, text))
    return hyps, outcomes


def filter_signal(details_by_video: Mapping[str, Sequence[SegmentDetail]]) -> str | None:
    """filter が判定に使う値 (``no_speech_prob`` / ``token_confidence_mean`` / ``avg_logprob``)。

    ``should_reject`` と同じ優先の順で、値のある区間が最も多いものを返す。値がどれにも無ければ ``None``。
    """
    counts = {"no_speech_prob": 0, "token_confidence_mean": 0, "avg_logprob": 0}
    for details in details_by_video.values():
        for d in details:
            c = d.confidence
            if c is None or not c.is_available:
                continue
            if c.no_speech_prob is not None:
                counts["no_speech_prob"] += 1
            elif c.token_confidence_mean is not None:
                counts["token_confidence_mean"] += 1
            elif c.avg_logprob is not None:
                counts["avg_logprob"] += 1
    best = max(counts, key=lambda k: counts[k])
    return best if counts[best] else None


def _filter_grid(signal: str | None, engine_name: str) -> list[tuple[float | None, Setting]]:
    """filter の閾値を振る設定 (EnergyGate は既定のまま)。先頭は filter を外した設定。"""
    from livecap_cli.transcription.confidence_filter import _engine_id_from_name

    base = default_setting()
    config = base.filter
    rows: list[tuple[float | None, Setting]] = [(None, Setting("filter なし", base.min_rms_dbfs, None))]
    if signal == "no_speech_prob":
        current = config.no_speech_threshold
        values = [0.5, 0.6, current, 0.8, 0.9, 0.95, 0.99]
        make = lambda v: replace(config, no_speech_threshold=v)  # noqa: E731
    elif signal == "token_confidence_mean":
        current = config.token_conf_threshold
        values = [1e-5, 1e-4, 5e-4, current, 2e-3, 5e-3, 1e-2]
        make = lambda v: replace(config, token_conf_threshold=v)  # noqa: E731
    elif signal == "avg_logprob":
        key = _engine_id_from_name(engine_name)
        if key in config.avg_logprob_thresholds:
            current = config.avg_logprob_thresholds[key]
            make = lambda v: replace(config, avg_logprob_thresholds={**config.avg_logprob_thresholds, key: v})  # noqa: E731
        else:
            current = config.avg_logprob_threshold
            make = lambda v: replace(config, avg_logprob_threshold=v)  # noqa: E731
        values = [round(current + d, 3) for d in (0.1, 0.05, 0.0, -0.05, -0.1, -0.2, -0.3, -0.5, -0.8)]
    else:
        return rows
    for v in sorted(set(values)):
        rows.append((v, Setting(f"{signal} {v:g}" + (" (既定)" if v == current else ""), base.min_rms_dbfs, make(v))))
    return rows


def _gate_grid() -> list[tuple[float, Setting]]:
    """EnergyGate の閾値を振る設定 (filter は既定のまま)。"""
    base = default_setting()
    values = [-math.inf, -60.0, -55.0, -50.0, base.min_rms_dbfs, -40.0, -35.0]
    return [
        (v, Setting("EnergyGate なし" if math.isinf(v) else f"EnergyGate {v:g} dBFS" + (" (既定)" if v == base.min_rms_dbfs else ""), v, base.filter))
        for v in values
    ]


def _run_setting(run: Mapping[str, Any]) -> Setting:
    """run.json の設定 (書き起こしたときの EnergyGate と filter)。"""
    from livecap_cli.transcription.confidence_filter import FilterConfig

    args = run["args"]
    rms = args["engine_min_rms"]
    rms = float(rms) if isinstance(rms, str) else rms  # "-inf" (runner._json_value)
    mode = args["confidence_filter"]
    return Setting("記録の設定", rms, FilterConfig(mode="on") if mode == "on" else None)


def _chars(text: str) -> int:
    """採点と同じ正規化のあとの文字数。"""
    from benchmarks.common.text_normalization import normalize_text

    return len(normalize_text(text, lang="ja"))


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _cues(video: Yodas2Video, *, scored_only: bool) -> list[tuple[float, float, str]]:
    """字幕の区間。``scored_only`` なら主の値で採点する区間だけ (VAD の取りこぼしの文字数に使う)。

    字幕があるかないか (filter が捨てた区間の分類、字幕の無い時間の出力) は、採点から外す区間
    (``excluded``・``notation``) も含めて見る。外す区間にも発話はあり、字幕が無いのとは違う (PR #515 F3)。
    """
    if not scored_only:
        return [(s.start, s.end, s.text) for s in video.segments]
    skip = {"excluded"} if MAIN_CONDITION.include_notation else {"excluded", "notation"}
    return [(s.start, s.end, s.text) for s in video.segments if s.status not in skip]


def _caption_chars(cues: Sequence[tuple[float, float, str]], start: float, end: float) -> float:
    """区間に重なる字幕の文字数 (字幕の区間のうち重なる長さの割合で按分する)。"""
    total = 0.0
    for c0, c1, text in cues:
        ov = _overlap(start, end, c0, c1)
        if ov > 0:
            total += _chars(text) * ov / max(c1 - c0, 1e-6)
    return total


def _precision(reference: str, hypothesis: str) -> float:
    """``hypothesis`` の文字のうち、``reference`` と一致した割合 ((出力の長さ - 置換 - 挿入) / 出力の長さ)。

    ``reference`` は区間より広く取ってよい (脱落は数えない)。
    """
    from benchmarks.common.error_breakdown import count_errors

    n = _chars(hypothesis)
    if n == 0:
        return 0.0
    counts = count_errors(reference, hypothesis, lang="ja", variant=MAIN_CONDITION.variant)
    return max(0.0, (n - counts.substitutions - counts.insertions) / n)


def _score_row(videos: Sequence[Yodas2Video], hyps: Mapping[str, list[HypSegment]], outcomes: Mapping[str, list[str]],
               details: Mapping[str, Sequence[SegmentDetail]]) -> dict[str, Any]:
    pooled = score_engine(videos, hyps, MAIN_CONDITION).pooled
    n = pooled.reference_length
    dropped = {GATE: [0, 0.0], FILTER: [0, 0.0]}
    for aid, outs in outcomes.items():
        for d, o in zip(details[aid], outs):
            if o in dropped:
                dropped[o][0] += 1
                dropped[o][1] += d.end - d.start
    return {
        "cer": pooled.rate,
        "substitutions": pooled.substitutions / n,
        "deletions": pooled.deletions / n,
        "insertions": pooled.insertions / n,
        "gate_segments": dropped[GATE][0],
        "gate_sec": round(dropped[GATE][1], 1),
        "filter_segments": dropped[FILTER][0],
        "filter_sec": round(dropped[FILTER][1], 1),
    }


def _simulate_all(details: Mapping[str, Sequence[SegmentDetail]], setting: Setting, engine_name: str):
    hyps, outcomes = {}, {}
    for aid, ds in details.items():
        hyps[aid], outcomes[aid] = simulate(ds, setting, engine_name=engine_name)
    return hyps, outcomes


def _self_check(run_dir: Path, videos: Sequence[Yodas2Video], details, engine_name: str) -> dict[str, Any]:
    """記録の設定で計算した結果が、記録の結果 (区間ごとの判定と出力) と同じか。"""
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    hyps, outcomes = _simulate_all(details, _run_setting(run), engine_name)
    recorded = load_hypotheses(run_dir, videos)
    bad_outcomes = sum(
        o != d.outcome for aid in details for d, o in zip(details[aid], outcomes[aid])
    )
    bad_videos = [aid for aid in details if [(round(s, 3), round(e, 3), t) for s, e, t in hyps[aid]] != recorded[aid]]
    return {"segments": sum(len(v) for v in details.values()), "outcome_mismatches": bad_outcomes, "output_mismatch_videos": bad_videos}


def _verify_conditions(verify_run: Mapping[str, Any], source_run: Mapping[str, Any], audio_ids: Sequence[str]) -> list[str]:
    """照合できる条件がそろっているか。そろわない理由を返す (空ならそろっている)。

    実測側は CLI の既定の EnergyGate と filter で書き起こしたもので、それ以外の設定 (engine・大きさ・言語・装置・
    VAD の backend と設定など、run.json に残る結果に効く設定) と engine、動画ごとの音声 (SHA-256) が分析元と
    同じでなければ、区間ごとに比べても意味がない (PR #515 F1)。分析元の EnergyGate と filter の mode は
    そろわなくてよい (分析は EnergyGate を外した、filter にかける前の記録から計算する)。
    """
    expected = default_setting()
    problems = []
    args, source_args = verify_run.get("args") or {}, source_run.get("args") or {}
    if args.get("engine_min_rms") != expected.min_rms_dbfs or args.get("confidence_filter") != "on":
        problems.append(
            f"not the CLI default EnergyGate / filter (engine_min_rms={args.get('engine_min_rms')!r}, "
            f"confidence_filter={args.get('confidence_filter')!r})"
        )
    differ = sorted(
        k for k in set(args) | set(source_args)
        if k not in _ANALYZED_ARGS and args.get(k) != source_args.get(k)
    )
    if differ:
        problems.append(f"settings differ from the analyzed run: {', '.join(differ)}")
    for key in ("engine_name", "vad_config", "vad_backend"):
        if verify_run.get(key) != source_run.get(key):
            problems.append(f"{key} differs from the analyzed run")
    audio, source_audio = verify_run.get("audio_sha256") or {}, source_run.get("audio_sha256") or {}
    other_audio = [aid for aid in audio_ids if audio.get(aid) is None or audio.get(aid) != source_audio.get(aid)]
    if other_audio:
        problems.append(f"audio differs from the analyzed run for {len(other_audio)} videos (first: {other_audio[0]})")
    return problems


def _verify(verify_dir: Path, videos: Sequence[Yodas2Video], details, engine_name: str,
            source_run: Mapping[str, Any]) -> dict[str, Any]:
    """CLI の既定で実際に書き起こした結果 (``verify_dir``、動画の一部でよい) と、計算した既定の結果を区間ごとに照らす。

    ``status``:

    - ``compared``: 1 区間以上を照らした。``compared_videos`` が照らした動画、``spans_differ`` が VAD の区間が
      違って照らさなかった動画
    - ``no_segments``: 条件はそろったが、照らせた区間が 1 つも無い (どの動画も VAD の区間が違う、など)
    - ``not_comparable``: 条件がそろわない (``problems`` に理由)
    - ``no_videos``: 分析元と同じ動画が実測側に 1 本も無い

    ``compared`` のときだけ違いの数を出す。照らせなかったときに「違い 0」に見せない (PR #515 F1)。
    """
    ids = sorted(p.stem for p in verify_dir.glob("*.json") if p.name != "run.json")
    subset = [v for v in videos if v.audio_id in ids]
    if not subset:
        return {"status": "no_videos", "videos": []}
    verify_run = json.loads((verify_dir / "run.json").read_text(encoding="utf-8"))
    problems = _verify_conditions(verify_run, source_run, [v.audio_id for v in subset])
    if problems:
        return {"status": "not_comparable", "videos": [v.audio_id for v in subset], "problems": problems}
    actual = load_segment_details(verify_dir, subset)
    setting = default_setting()
    compared: list[str] = []
    spans_differ: list[str] = []
    segments = outcome_mismatches = text_mismatches = 0
    for video in subset:
        sim = details[video.audio_id]
        act = actual[video.audio_id]
        if [(d.start, d.end) for d in sim] != [(d.start, d.end) for d in act]:
            spans_differ.append(video.audio_id)  # VAD の区間が違えば、区間ごとには比べない
            continue
        compared.append(video.audio_id)
        _, outcomes = simulate(sim, setting, engine_name=engine_name)
        segments += len(sim)
        for s, a, o in zip(sim, act, outcomes):
            outcome_mismatches += o != a.outcome
            if a.text is not None and s.text != a.text:
                text_mismatches += 1
    out: dict[str, Any] = {
        "status": "compared" if segments else "no_segments",
        "videos": [v.audio_id for v in subset],
        "compared_videos": compared,
        "spans_differ": spans_differ,
    }
    if segments:
        out |= {"segments": segments, "outcome_mismatches": outcome_mismatches, "text_mismatches": text_mismatches}
    return out


def _classify_drops(videos, details, outcomes, reference_hyps) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """既定で filter が捨てた区間を、字幕・ASR の文・ほかの ASR の文で分ける。

    字幕が無い区間は、ほかの ASR (``reference_hyps``) と照らしたときだけ「文を出す / 出さない」に分け、照らして
    いなければ「不明」にする。ほかの ASR が文を出さないことは、話し声が無いことと同じではない (PR #515 F3)。
    """
    classes = {
        "caption_match": "字幕あり・ASR の文が字幕と合う",
        "caption_mismatch": "字幕あり・ASR の文が字幕と合わない",
        "no_caption_other_silent": "字幕なし・ほかの ASR も文を出さない",
        "no_caption_other_spoke": "字幕なし・ほかの ASR は文を出す",
        "no_caption_unknown": "字幕なし・ほかの ASR と照らしていない",
    }
    stats = {k: {"label": v, "segments": 0, "sec": 0.0, "caption_chars": 0.0, "asr_chars": 0} for k, v in classes.items()}
    candidates: list[dict[str, Any]] = []
    for video in videos:
        cues = _cues(video, scored_only=False)
        for d, o in zip(details[video.audio_id], outcomes[video.audio_id]):
            if o != FILTER:
                continue
            cap = _caption_chars(cues, d.start, d.end)
            other = None
            if cap >= 1.0:
                window = " ".join(t for c0, c1, t in cues if _overlap(d.start - CAPTION_TOLERANCE_SEC, d.end + CAPTION_TOLERANCE_SEC, c0, c1) > 0)
                key = "caption_match" if _precision(window, d.text) >= MATCH_PRECISION else "caption_mismatch"
            elif reference_hyps is None:
                key = "no_caption_unknown"
            else:
                other = " ".join(
                    t for s, e, t in reference_hyps.get(video.audio_id, [])
                    if _overlap(d.start, d.end, s, e) >= 0.5 * min(e - s, d.end - d.start)
                )
                key = "no_caption_other_spoke" if _chars(other) >= 2 else "no_caption_other_silent"
            st = stats[key]
            st["segments"] += 1
            st["sec"] += d.end - d.start
            st["caption_chars"] += cap
            st["asr_chars"] += _chars(d.text)
            if key in _CANDIDATE_CLASSES:  # 判定が割れる区間と、判定できない区間は、聞いて確かめる候補
                candidates.append({
                    "audio_id": video.audio_id, "start": d.start, "end": d.end, "class": key,
                    "energy_dbfs": round(d.energy_dbfs, 1), "text": d.text, "other_asr_text": other,
                    "confidence": {
                        k: _json_value(getattr(d.confidence, k))
                        for k in ("no_speech_prob", "avg_logprob", "token_confidence_mean")
                    },
                })
    for st in stats.values():
        st["sec"] = round(st["sec"], 1)
        st["caption_chars"] = round(st["caption_chars"], 1)
    # 分類ごとに、長い区間から上限まで (聞く手間に見合うものを先に)
    picked: list[dict[str, Any]] = []
    for key in _CANDIDATE_CLASSES:
        group = sorted((c for c in candidates if c["class"] == key), key=lambda c: c["end"] - c["start"], reverse=True)
        picked += group[:MAX_CANDIDATES_PER_CLASS]
    return stats, picked


def _vad_miss(videos: Sequence[Yodas2Video], details) -> dict[str, float]:
    """どの VAD の区間にも入らない字幕の文字数 (字幕の区間のうち、VAD の区間の外の長さの割合で按分)。

    採点する字幕だけを数える (脱落として CER に効く文字の目安)。
    """
    missed = total = 0.0
    for video in videos:
        spans = [(d.start, d.end) for d in details[video.audio_id]]
        for c0, c1, text in _cues(video, scored_only=True):
            n = _chars(text)
            covered = sum(_overlap(c0, c1, s, e) for s, e in spans)
            missed += n * max(0.0, 1.0 - covered / max(c1 - c0, 1e-6))
            total += n
    return {"caption_chars": round(total, 1), "outside_vad_chars": round(missed, 1), "share": missed / total if total else 0.0}


def _uncaptioned(videos: Sequence[Yodas2Video], hyps: Mapping[str, list[HypSegment]]) -> dict[str, float]:
    """字幕の無い時間 (字幕の時刻の前後 0.5 秒も字幕ありとする) に出た文の文字数と、1 分あたりの数。

    採点から外す字幕も「字幕あり」とする (外す区間の出力は、字幕の無い時間の出力ではない)。
    """
    chars = 0
    for video in videos:
        cues = _cues(video, scored_only=False)
        for s, e, t in hyps[video.audio_id]:
            if not any(_overlap(s, e, c0 - CAPTION_TOLERANCE_SEC, c1 + CAPTION_TOLERANCE_SEC) > 0 for c0, c1, _ in cues):
                chars += _chars(t)
    minutes = sum(v.duration_sec for v in videos) / 60
    return {"chars": chars, "chars_per_min": chars / minutes if minutes else 0.0}


def analyze_label(run_dir: Path, videos: Sequence[Yodas2Video], *, verify_dir: Path | None = None,
                  reference_hyps: Mapping[str, list[HypSegment]] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """1 つの記録 (EnergyGate を外して書き起こしたもの) を分析する。

    Raises:
        ValueError: 記録の EnergyGate が外れていない (それより低い閾値を計算できない)
        RuntimeError, FileNotFoundError: :func:`~.runner.load_segment_details` と同じ
    """
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if not math.isinf(_run_setting(run).min_rms_dbfs):
        raise ValueError(f"{run_dir}: transcribed with the EnergyGate on; use 'transcribe --engine-min-rms off'")
    engine_name = run["engine_name"]
    details = load_segment_details(run_dir, videos)
    signal = filter_signal(details)
    result: dict[str, Any] = {
        "engine_name": engine_name,
        "filter_signal": signal,
        "self_check": _self_check(run_dir, videos, details, engine_name),
        "verify": None if verify_dir is None else _verify(verify_dir, videos, details, engine_name, run),
    }
    rows = []
    for kind, grid in (("filter", _filter_grid(signal, engine_name)), ("gate", _gate_grid())):
        for value, setting in grid:
            hyps, outcomes = _simulate_all(details, setting, engine_name)
            rows.append({
                "kind": kind, "setting": setting.name, "value": _json_value(value),
                **_score_row(videos, hyps, outcomes, details),
            })
    result["sweep"] = rows
    hyps, outcomes = _simulate_all(details, default_setting(), engine_name)
    result["drops"], candidates = _classify_drops(videos, details, outcomes, reference_hyps)
    off_hyps, _ = _simulate_all(details, replace(default_setting(), filter=None), engine_name)
    result["uncaptioned"] = {"default": _uncaptioned(videos, hyps), "filter_off": _uncaptioned(videos, off_hyps)}
    result["vad_miss"] = _vad_miss(videos, details)
    for c in candidates:
        c["label"] = run_dir.name
    return result, candidates


def analyze(out_dir: Path, labels: Sequence[str], videos: Sequence[Yodas2Video], *, split: str,
            verify_dir: Path | None = None, reference_label: str | None = None) -> dict[str, Any]:
    """``labels`` の記録を分析し、``out_dir/drops.json``・``drops.md``・``drops-candidates/candidates.json`` を書く。

    ``verify_dir`` は CLI の既定で書き起こした結果の出力先 (同じ label の下にある動画だけを照らす)。
    ``reference_label`` は、字幕の無い区間でほかの ASR が文を出したかを見るのに使う記録 (既定の設定で計算する)。
    """
    reference_hyps = None
    if reference_label is not None:
        ref_dir = out_dir / reference_label
        ref_run = json.loads((ref_dir / "run.json").read_text(encoding="utf-8"))
        reference_hyps, _ = _simulate_all(load_segment_details(ref_dir, videos), default_setting(), ref_run["engine_name"])
    results: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = []
    for label in labels:
        verify = None
        if verify_dir is not None:
            base = label.removesuffix("-gateoff")
            if (verify_dir / base / "run.json").exists():
                verify = verify_dir / base
        results[label], cands = analyze_label(
            out_dir / label, videos, verify_dir=verify,
            reference_hyps=None if label == reference_label else reference_hyps,
        )
        candidates += cands
    hours = sum(v.duration_sec for v in videos) / 3600
    summary = {"split": split, "videos": len(videos), "hours": round(hours, 2), "reference_label": reference_label, "results": results}
    (out_dir / "drops.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    # 聞いて確かめる候補 (ASR の文を含むので commit しない。out_dir の下は .gitignore で除く場所に置く)
    _write_candidates(out_dir, candidates)
    (out_dir / "drops.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def _write_candidates(out_dir: Path, candidates: list[dict[str, Any]]) -> Path:
    """聞いて確かめる候補を ``<out_dir>/drops-candidates/candidates.json`` に書く。

    ASR の文を含むので commit しない。出力先の直下のファイルは既存の .gitignore で除かれない (除くのは ASR ごとの
    フォルダだけ) ので、専用のフォルダに置き、そこに ``*`` だけの .gitignore を置く。出力先がどこでも、
    ``git add`` で入らない (PR #515 F2)。数の集計 (drops.json・drops.md) は出力先の直下に残す。
    """
    folder = out_dir / CANDIDATES_DIR
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ".gitignore").write_text("# ASR の文を含むので commit しない (benchmarks/longform/drops.py)\n*\n", encoding="utf-8")
    path = folder / "candidates.json"
    path.write_text(json.dumps(candidates, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    return path


def _pct(x: float) -> str:
    return f"{x * 100:.2f}"


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        f"# EnergyGate と confidence filter が捨てる音声: YODAS2 日本語 `{summary['split']}` ({summary['videos']} 本、{summary['hours']:.1f} 時間)",
        "",
        f"EnergyGate を外して書き起こした記録から計算した値。CER は `{MAIN_CONDITION.key}` (%)。"
        "字幕との重なりで数える値は、字幕の時刻が人の確かめたものではないので目安。",
    ]
    for label, r in summary["results"].items():
        sc, vf = r["self_check"], r["verify"]
        lines += ["", f"## `{label}` ({r['engine_name']}、filter の値: `{r['filter_signal']}`)", ""]
        lines.append(
            f"- 自己照合: {sc['segments']} 区間で、判定の違い {sc['outcome_mismatches']}、出力の違う動画 {len(sc['output_mismatch_videos'])}"
        )
        if vf is None:
            lines.append("- 既定で実際に書き起こした結果との照合: しなかった (照合用の記録が無い)")
        elif vf["status"] == "no_videos":
            lines.append("- 既定で実際に書き起こした結果との照合: **できなかった** (分析元と同じ動画が無い)")
        elif vf["status"] == "not_comparable":
            lines.append(f"- 既定で実際に書き起こした結果との照合: **比べられない** ({'; '.join(vf['problems'])})")
        elif vf["status"] == "no_segments":
            lines.append(
                f"- 既定で実際に書き起こした結果との照合: **照らせた区間が無い** (VAD の区間が違う動画 "
                f"{len(vf['spans_differ'])} / {len(vf['videos'])} 本)"
            )
        else:
            lines.append(
                f"- 既定で実際に書き起こした結果との照合: {len(vf['compared_videos'])} 本・{vf['segments']} 区間で、"
                f"判定の違い {vf['outcome_mismatches']}、文の違い {vf['text_mismatches']} "
                f"(VAD の区間が違って照らさなかった動画 {len(vf['spans_differ'])} 本)"
            )
        vm = r["vad_miss"]
        lines.append(f"- VAD のどの区間にも入らない字幕: {vm['outside_vad_chars']:.0f} 文字 / {vm['caption_chars']:.0f} 文字 ({_pct(vm['share'])}%)")
        un = r["uncaptioned"]
        lines.append(
            f"- 字幕の無い時間に出た文: 既定 {un['default']['chars_per_min']:.2f} 文字/分、filter なし {un['filter_off']['chars_per_min']:.2f} 文字/分"
        )
        lines += ["", "| 設定 | CER | 置換 | 脱落 | 挿入 | filter で捨てた (区間 / 秒) | EnergyGate で捨てた (区間 / 秒) |", "|---|---:|---:|---:|---:|---:|---:|"]
        for row in r["sweep"]:
            lines.append(
                f"| {row['setting']} | {_pct(row['cer'])} | {_pct(row['substitutions'])} | {_pct(row['deletions'])} | "
                f"{_pct(row['insertions'])} | {row['filter_segments']} / {row['filter_sec']} | {row['gate_segments']} / {row['gate_sec']} |"
            )
        lines += ["", "既定で filter が捨てた区間の分け方:", "", "| 分類 | 区間 | 秒 | 重なる字幕の文字 | ASR の文字 |", "|---|---:|---:|---:|---:|"]
        for st in r["drops"].values():
            lines.append(f"| {st['label']} | {st['segments']} | {st['sec']} | {st['caption_chars']:.0f} | {st['asr_chars']} |")
    return "\n".join(lines) + "\n"



def gate_off_labels(out_dir: Path) -> list[str]:
    """``out_dir`` にある、EnergyGate を外して書き起こした記録の label。"""
    labels = []
    for path in sorted(out_dir.glob("*/run.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if math.isinf(_run_setting(run).min_rms_dbfs):
            labels.append(path.parent.name)
    return labels
