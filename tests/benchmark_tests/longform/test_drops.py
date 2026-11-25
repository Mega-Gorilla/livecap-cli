"""EnergyGate と confidence filter の閾値を変えた結果の計算 (benchmarks/longform/drops.py、#488 段階 2)。

モデルは読み込まない。CLI の部品 (segmenter / engine の読み込み) を差し替えて書き起こした記録から計算し、
CLI の判定と同じ結果になること、照合と分類の数え方を確かめる。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
pytest.importorskip("jiwer")

from benchmarks.common.yodas2 import Yodas2Segment, Yodas2Video  # noqa: E402
from benchmarks.longform.drops import (  # noqa: E402
    ASR,
    EMPTY,
    FILTER,
    GATE,
    Setting,
    _filter_grid,
    _uncaptioned,
    analyze,
    analyze_label,
    default_setting,
    gate_off_labels,
    simulate,
)
from benchmarks.longform.runner import EngineSpec, SegmentDetail, transcribe_videos  # noqa: E402
from livecap_cli import cli  # noqa: E402
from livecap_cli.engines.base_engine import EngineConfidence, TranscriptionResult  # noqa: E402

RATE = 16000
#: 区間の長さ (秒) ごとの ASR の出力と avg_logprob。0.8 秒の区間は自信が無い (この engine には既定の -1.0 が効く)
OUTPUTS = {0.6: ("こんにちは", -0.1), 0.8: ("ありがとう", -2.0), 1.3: ("えー", -0.1)}


class FakeEngine:
    """区間の長さで決まる出力を返す (書き起こす区間が変わっても、同じ区間には同じ文を返す)。"""

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, audio, sample_rate):
        self.calls += 1
        text, logprob = OUTPUTS[round(len(audio) / sample_rate, 1)]
        return TranscriptionResult(text=text, confidence=1.0, engine_confidence=EngineConfidence(avg_logprob=logprob))

    def get_engine_name(self) -> str:
        return "fake"

    def get_required_sample_rate(self) -> int:
        return RATE

    def cleanup(self) -> None:
        pass


@pytest.fixture
def fake_cli(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr("livecap_cli.runtime.configure_pytorch_runtime", lambda: None)
    monkeypatch.setattr(cli, "_load_engine", lambda args: engine)
    # 音 (0.6 秒・0.8 秒) と、後半の無音 (1.3 秒、既定の EnergyGate で落ちる) の 3 区間
    monkeypatch.setattr(cli, "_build_file_segmenter", lambda args: (lambda audio, sr: [(0.0, 0.6), (0.6, 1.4), (1.6, 2.9)]))
    return engine


def _video(audio_id: str, captions: tuple[tuple[float, float, str], ...]) -> Yodas2Video:
    segments = tuple(Yodas2Segment(s, e, t) for s, e, t in captions)
    return Yodas2Video(audio_id, "dev", "1", 3.0, "other", (), -20.0, "ja000", "00000001", segments)


def _setup(tmp_path: Path, fake_cli) -> tuple[Path, Path, list[Yodas2Video]]:
    """EnergyGate を外した記録 (out/qwen3asr-gateoff) と、CLI の既定の記録 (verify/qwen3asr) を作る。"""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    for aid in ("AAAAAAAAAAA", "BBBBBBBBBBB"):
        n = 3 * RATE  # 前半は音、後半は無音
        audio = np.zeros(n, dtype=np.float32)
        audio[: n // 2] = 0.3 * np.sin(2 * np.pi * 300 * np.arange(n // 2) / RATE)
        sf.write(audio_dir / f"{aid}.wav", audio, RATE, subtype="PCM_16")
    videos = [
        # A は 2 つ目の区間にも字幕がある (filter は話し声を正しく認識した区間を捨てる)。B には無い
        _video("AAAAAAAAAAA", ((0.0, 0.6, "こんにちは"), (0.6, 1.4, "ありがとう"))),
        _video("BBBBBBBBBBB", ((0.0, 0.6, "こんにちは"),)),
    ]
    out_dir, verify_dir = tmp_path / "out", tmp_path / "verify"
    transcribe_videos(EngineSpec.parse("qwen3asr", engine_min_rms=-math.inf), videos, audio_dir=audio_dir, out_dir=out_dir, device="cpu")
    transcribe_videos(EngineSpec("qwen3asr"), videos[:1], audio_dir=audio_dir, out_dir=verify_dir, device="cpu")
    return out_dir, verify_dir, videos


def _detail(start, end, energy, text=None, logprob=None, outcome=ASR) -> SegmentDetail:
    conf = None if text is None else EngineConfidence(avg_logprob=logprob)
    return SegmentDetail(start, end, outcome, energy, text, conf)


def test_simulate_follows_the_cli_order() -> None:
    """EnergyGate (音量 < 閾値) → filter → 空の出力、の順で判定する。文は前後の空白を除く。"""
    details = [
        _detail(0.0, 1.0, -20.0, " はい ", -0.1),
        _detail(1.0, 2.0, -50.0, "えー", -0.1),  # EnergyGate の既定 (-45) で落ちる
        _detail(2.0, 3.0, -20.0, "ノイズ", -2.0),  # filter で落ちる
        _detail(3.0, 4.0, -20.0, "  ", -0.1),
        _detail(4.0, 5.0, -45.0, "境目", -0.1),  # 閾値と同じ音量は通す (CLI と同じく strict な <)
    ]
    hyps, outcomes = simulate(details, default_setting(), engine_name="fake")
    assert outcomes == [ASR, GATE, FILTER, EMPTY, ASR]
    assert hyps == [(0.0, 1.0, "はい"), (4.0, 5.0, "境目")]
    _, outcomes = simulate(details, Setting("filter なし", -45.0, None), engine_name="fake")
    assert outcomes == [ASR, GATE, ASR, EMPTY, ASR]
    _, outcomes = simulate(details, Setting("EnergyGate なし", -math.inf, default_setting().filter), engine_name="fake")
    assert outcomes == [ASR, ASR, FILTER, EMPTY, ASR]
    # ASR を呼んでいない区間を ASR に渡す設定は計算できない (EnergyGate を外した記録で計算する)
    with pytest.raises(ValueError, match="EnergyGate off"):
        simulate([_detail(0.0, 1.0, -50.0, outcome=GATE)], Setting("x", -math.inf, None), engine_name="fake")


def test_analysis_reproduces_the_recorded_and_the_default_runs(tmp_path: Path, fake_cli) -> None:
    """自己照合・既定の実測との照合が区間ごとに一致し、閾値ごとの値と捨てた区間の分け方を出す。"""
    out_dir, verify_dir, videos = _setup(tmp_path, fake_cli)
    summary = analyze(out_dir, ["qwen3asr-gateoff"], videos, split="dev", verify_dir=verify_dir)
    r = summary["results"]["qwen3asr-gateoff"]
    assert r["filter_signal"] == "avg_logprob"
    assert r["self_check"] == {"segments": 6, "outcome_mismatches": 0, "output_mismatch_videos": []}
    assert r["verify"] == {
        "status": "compared", "videos": ["AAAAAAAAAAA"], "compared_videos": ["AAAAAAAAAAA"], "spans_differ": [],
        "segments": 3, "outcome_mismatches": 0, "text_mismatches": 0,
    }

    rows = {row["setting"]: row for row in r["sweep"]}
    default = rows["EnergyGate -45 dBFS (既定)"]
    assert (default["gate_segments"], default["filter_segments"]) == (2, 2)
    assert rows["avg_logprob -1 (既定)"] == {**default, "kind": "filter", "setting": "avg_logprob -1 (既定)", "value": -1.0}
    # filter を外すと、A の脱落 (ありがとう) が消え、B に挿入が出る
    off = rows["filter なし"]
    assert off["filter_segments"] == 0 and off["deletions"] < default["deletions"] and off["insertions"] > default["insertions"]
    assert rows["EnergyGate なし"]["gate_segments"] == 0 and rows["EnergyGate なし"]["value"] == "-inf"

    # 既定で filter が捨てた 2 区間: A は字幕と合う (話し声を捨てた)。B は字幕が無く、ほかの ASR と照らしていないので
    # 「不明」(話し声でないとはみなさない、PR #515 F3)
    drops = r["drops"]
    assert drops["caption_match"]["segments"] == 1 and drops["caption_match"]["caption_chars"] == 5.0
    assert drops["no_caption_unknown"]["segments"] == 1 and drops["no_caption_other_silent"]["segments"] == 0
    assert r["vad_miss"]["outside_vad_chars"] == 0.0

    # drops.json は厳密な JSON。drops.md に ASR ごとの節がある。不明の区間は聞く候補になる
    def strict(constant):
        raise AssertionError(constant)

    json.loads((out_dir / "drops.json").read_text(encoding="utf-8"), parse_constant=strict)
    assert "## `qwen3asr-gateoff`" in (out_dir / "drops.md").read_text(encoding="utf-8")
    candidates = json.loads((out_dir / "drops-candidates" / "candidates.json").read_text(encoding="utf-8"))
    assert [(c["audio_id"], c["class"]) for c in candidates] == [("BBBBBBBBBBB", "no_caption_unknown")]


def test_other_asr_output_marks_uncaptioned_drops_for_listening(tmp_path: Path, fake_cli) -> None:
    """字幕の無い区間でほかの ASR が文を出していれば、判定が割れた区間として聞く候補に入れる。"""
    out_dir, _, videos = _setup(tmp_path, fake_cli)
    # ほかの ASR として、同じ記録の写しを使う。字幕の無い B の 2 つ目の区間を、自信のある出力にする (既定でも残る)
    import shutil

    shutil.copytree(out_dir / "qwen3asr-gateoff", out_dir / "other-gateoff")
    path = out_dir / "other-gateoff" / "BBBBBBBBBBB.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["segment_details"][1]["confidence"]["avg_logprob"] = -0.1
    path.write_text(json.dumps(record), encoding="utf-8")
    summary = analyze(out_dir, ["qwen3asr-gateoff"], videos, split="dev", reference_label="other-gateoff")
    drops = summary["results"]["qwen3asr-gateoff"]["drops"]
    assert drops["no_caption_other_spoke"]["segments"] == 1
    candidates = json.loads((out_dir / "drops-candidates" / "candidates.json").read_text(encoding="utf-8"))
    assert [(c["audio_id"], c["class"], c["other_asr_text"]) for c in candidates] == [
        ("BBBBBBBBBBB", "no_caption_other_spoke", "ありがとう")
    ]
    assert candidates[0]["label"] == "qwen3asr-gateoff" and candidates[0]["confidence"]["avg_logprob"] == -2.0


def test_verification_needs_the_same_audio_and_the_default_settings(tmp_path: Path, fake_cli) -> None:
    """照合は、実測側が既定の設定で、ほかの設定と音声が分析元と同じときだけ行う。そろわなければ「違い 0」と
    言わずに、比べられない理由を出す。同じ動画が無ければ照合できない旨を出す (PR #515 F1)。"""
    out_dir, verify_dir, videos = _setup(tmp_path, fake_cli)
    source = out_dir / "qwen3asr-gateoff"

    # 中身の違う音声 (同じ ID) から既定で書き起こした結果: 区間の時刻は同じでも比べない
    other_audio = tmp_path / "other-audio"
    other_audio.mkdir()
    sf.write(other_audio / "AAAAAAAAAAA.wav", 0.2 * np.ones(3 * RATE, dtype=np.float32), RATE, subtype="PCM_16")
    other = tmp_path / "verify-other-audio"
    transcribe_videos(EngineSpec("qwen3asr"), videos[:1], audio_dir=other_audio, out_dir=other, device="cpu")
    verify = analyze_label(source, videos, verify_dir=other / "qwen3asr")[0]["verify"]
    assert verify["status"] == "not_comparable" and "outcome_mismatches" not in verify
    assert any("audio differs" in p for p in verify["problems"])

    # EnergyGate が既定と違う (-50 dBFS)
    gate = tmp_path / "verify-gate"
    transcribe_videos(EngineSpec.parse("qwen3asr", engine_min_rms=-50.0), videos[:1], audio_dir=tmp_path / "audio", out_dir=gate, device="cpu")
    verify = analyze_label(source, videos, verify_dir=gate / "qwen3asr-gate-50db")[0]["verify"]
    assert verify["status"] == "not_comparable" and "not the CLI default" in verify["problems"][0]

    # 分析する動画が実測側に 1 本も無い
    verify = analyze_label(source, videos[1:], verify_dir=verify_dir / "qwen3asr")[0]["verify"]
    assert verify == {"status": "no_videos", "videos": []}
    md = analyze(out_dir, ["qwen3asr-gateoff"], videos, split="dev", verify_dir=gate)  # label が違えば照合しない
    assert md["results"]["qwen3asr-gateoff"]["verify"] is None
    assert "比べられない" not in (out_dir / "drops.md").read_text(encoding="utf-8")


def test_verification_without_comparable_segments_is_not_reported_as_compared(
    tmp_path: Path, fake_cli, monkeypatch
) -> None:
    """条件と音声がそろっても、どの動画も VAD の区間が違えば、照らせた区間が無い旨を出す (違いの数は出さない)。"""
    out_dir, _, videos = _setup(tmp_path, fake_cli)
    # 長さは同じ (ASR の出力も同じ) だが、時刻のずれた区間
    monkeypatch.setattr(cli, "_build_file_segmenter", lambda args: (lambda audio, sr: [(0.1, 0.7), (0.7, 1.5), (1.6, 2.9)]))
    shifted = tmp_path / "verify-shifted"
    transcribe_videos(EngineSpec("qwen3asr"), videos[:1], audio_dir=tmp_path / "audio", out_dir=shifted, device="cpu")
    verify = analyze_label(out_dir / "qwen3asr-gateoff", videos, verify_dir=shifted / "qwen3asr")[0]["verify"]
    assert verify == {
        "status": "no_segments", "videos": ["AAAAAAAAAAA"], "compared_videos": [], "spans_differ": ["AAAAAAAAAAA"],
    }


def test_analyzed_run_may_have_the_filter_off(tmp_path: Path, fake_cli, monkeypatch) -> None:
    """分析元は filter にかける前の記録を使うので、分析元の filter の mode は既定の実測とそろわなくてよい。"""
    _, verify_dir, videos = _setup(tmp_path, fake_cli)
    monkeypatch.setenv("LIVECAP_CONFIDENCE_FILTER", "off")
    source_dir = tmp_path / "filter-off"
    dest = transcribe_videos(EngineSpec.parse("qwen3asr", engine_min_rms=-math.inf), videos, audio_dir=tmp_path / "audio", out_dir=source_dir, device="cpu")
    monkeypatch.delenv("LIVECAP_CONFIDENCE_FILTER")
    result = analyze_label(dest, videos, verify_dir=verify_dir / "qwen3asr")[0]
    assert result["self_check"]["outcome_mismatches"] == 0  # 記録の設定 (filter off) では、自信の無い区間も残る
    verify = result["verify"]
    assert verify["status"] == "compared" and verify["segments"] == 3
    assert (verify["outcome_mismatches"], verify["text_mismatches"]) == (0, 0)  # 既定 (filter on) の計算と実測が一致


def test_candidates_are_kept_out_of_git(tmp_path: Path) -> None:
    """聞く候補 (ASR の文を含む) は、出力先がどこでも git add で入らない (PR #515 F2)。"""
    import shutil
    import subprocess

    from benchmarks.common.yodas2 import PROJECT_ROOT
    from benchmarks.longform.drops import _write_candidates

    if shutil.which("git") is None:
        pytest.skip("git is not available")
    # 既定の出力先 (benchmark_results/longform/<split>/) の下
    rel = "benchmark_results/longform/dev/drops-candidates/candidates.json"
    assert subprocess.run(["git", "check-ignore", "--no-index", "-q", rel], cwd=PROJECT_ROOT).returncode == 0
    # ほかの場所 (別の Git の作業ツリー) でも、フォルダの .gitignore で除かれる
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    path = _write_candidates(repo / "out", [{"text": "本文"}])
    assert path == repo / "out" / "drops-candidates" / "candidates.json" and path.exists()
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo, capture_output=True, text=True, check=True)
    assert status.stdout == ""


def test_runs_with_the_gate_on_are_not_analyzed(tmp_path: Path, fake_cli) -> None:
    """EnergyGate が効いた記録からは、それより低い閾値を計算できないので分析しない。"""
    out_dir, verify_dir, videos = _setup(tmp_path, fake_cli)
    with pytest.raises(ValueError, match="engine-min-rms off"):
        analyze_label(verify_dir / "qwen3asr", videos[:1])
    assert gate_off_labels(out_dir) == ["qwen3asr-gateoff"] and gate_off_labels(verify_dir) == []


@pytest.mark.parametrize(
    ("signal", "engine_name", "current"),
    [
        ("no_speech_prob", "WhisperS2T large-v3", 0.71),
        ("token_confidence_mean", "Parakeet ja", 0.001),
        ("avg_logprob", "Qwen3-ASR 0.6B", -0.42),  # engine ごとの閾値 (qwen3-asr)
        ("avg_logprob", "ReazonSpeech K2 (CPU, Int8)", -0.40),
        ("avg_logprob", "fake", -1.0),  # engine ごとの閾値が無ければ、全体の既定
    ],
)
def test_filter_grid_brackets_the_current_threshold(signal: str, engine_name: str, current: float) -> None:
    grid = _filter_grid(signal, engine_name)
    assert grid[0][0] is None and grid[0][1].filter is None  # filter を外した設定が先頭
    values = [v for v, _ in grid[1:]]
    assert current in values and min(values) < current < max(values)
    assert sum("(既定)" in s.name for _, s in grid) == 1
    assert _filter_grid(None, engine_name) == grid[:1]


def test_uncaptioned_output_counts_only_text_outside_the_captions() -> None:
    video = _video("CCCCCCCCCCC", ((10.0, 12.0, "はい"),))
    hyps = {"CCCCCCCCCCC": [(0.0, 1.0, "ノイズ"), (12.2, 13.0, "はいはい"), (20.0, 21.0, "あ")]}
    out = _uncaptioned([video], hyps)  # 字幕の前後 0.5 秒は字幕あり (12.2 秒からの出力は数えない)
    assert out["chars"] == 4 and out["chars_per_min"] == pytest.approx(4 / (3.0 / 60))


def test_unscored_captions_still_count_as_captions() -> None:
    """採点から外す字幕 (notation・excluded) の時間も字幕ありとする。字幕の無い時間とは違う (PR #515 F3)。"""
    segments = (Yodas2Segment(0.0, 2.0, "コード", "notation"), Yodas2Segment(5.0, 6.0, "うん", "excluded"))
    video = Yodas2Video("DDDDDDDDDDD", "dev", "1", 10.0, "other", (), -20.0, "ja000", "00000001", segments)
    hyps = {"DDDDDDDDDDD": [(0.0, 2.0, "コード"), (5.0, 6.0, "うん"), (8.0, 9.0, "ノイズ")]}
    assert _uncaptioned([video], hyps)["chars"] == 3  # 字幕の無い 8 秒からの「ノイズ」だけ
