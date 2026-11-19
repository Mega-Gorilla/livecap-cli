"""長尺の評価の書き起こし (benchmarks/longform/runner.py、#488)。

モデルは読み込まない。CLI の部品 (segmenter / engine の読み込み) を差し替えて、CLI と同じ既定値・
同じ後段 (EnergyGate、confidence filter) を通ること、出力と再開の動きを確かめる。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
pytest.importorskip("jiwer")

from benchmarks.common.yodas2 import Yodas2Video  # noqa: E402
from benchmarks.longform.report import summarize  # noqa: E402
from benchmarks.longform.runner import (  # noqa: E402
    EngineSpec,
    build_transcribe_args,
    load_hypotheses,
    transcribe_videos,
)
from livecap_cli import cli  # noqa: E402
from livecap_cli.engines.base_engine import TranscriptionResult  # noqa: E402

RATE = 16000


class FakeEngine:
    def __init__(self) -> None:
        self.calls = 0
        self.cleaned = False

    def transcribe(self, audio, sample_rate):
        self.calls += 1
        return TranscriptionResult(text=f"区間{self.calls}", confidence=1.0)

    def get_engine_name(self) -> str:
        return "fake"

    def get_required_sample_rate(self) -> int:
        return RATE

    def cleanup(self) -> None:
        self.cleaned = True


def _video(audio_id: str, seconds: float = 3.0) -> Yodas2Video:
    return Yodas2Video(audio_id, "test", "1", seconds, "other", (), -20.0, "ja000", "00000001", ())


def _write_audio(path: Path, seconds: float) -> None:
    """前半は音、後半は無音 (EnergyGate で落ちる)。"""
    n = int(seconds * RATE)
    audio = np.zeros(n, dtype=np.float32)
    audio[: n // 2] = 0.3 * np.sin(2 * np.pi * 300 * np.arange(n // 2) / RATE)
    sf.write(path, audio, RATE, subtype="PCM_16")


@pytest.fixture
def fake_cli(monkeypatch):
    engine = FakeEngine()
    # PyTorch の実行環境の初期化は、環境変数の決定を process 全体で持つ。モデルを読まないので差し替える
    monkeypatch.setattr("livecap_cli.runtime.configure_pytorch_runtime", lambda: None)
    monkeypatch.setattr(cli, "_load_engine", lambda args: engine)
    # 前半 (音) と後半 (無音) の 2 区間を返す segmenter
    monkeypatch.setattr(cli, "_build_file_segmenter", lambda args: (lambda audio, sr: [(0.0, 1.4), (1.6, 2.9)]))
    return engine


def test_engine_spec_parse() -> None:
    assert EngineSpec.parse("qwen3asr_large") == EngineSpec("qwen3asr_large")
    assert EngineSpec.parse("whispers2t:large-v3").label == "whispers2t-large-v3"
    with pytest.raises(ValueError):
        EngineSpec.parse(":base")


def test_args_have_the_cli_defaults() -> None:
    args = build_transcribe_args(EngineSpec("qwen3asr_large"))
    cli_args = cli.build_parser().parse_args(["transcribe", "-", "--engine", "qwen3asr_large", "--language", "ja"])
    for key in ("vad", "engine_min_rms", "engine_energy_metric", "engine_energy_frame_ms", "confidence_filter", "noise_gate"):
        assert getattr(args, key) == getattr(cli_args, key), key
    assert args.language == "ja" and args.vad == "auto" and args.confidence_filter == "on"
    assert build_transcribe_args(EngineSpec("whispers2t", "large-v3")).model_size == "large-v3"


def test_engine_without_japanese_is_rejected_before_loading() -> None:
    with pytest.raises(ValueError):
        build_transcribe_args(EngineSpec("parakeet"))


def test_transcribes_through_the_cli_postprocess_and_resumes(tmp_path: Path, fake_cli) -> None:
    audio_dir, out_dir = tmp_path / "audio", tmp_path / "out"
    audio_dir.mkdir()
    for aid in ("AAAAAAAAAAA", "BBBBBBBBBBB"):
        _write_audio(audio_dir / f"{aid}.wav", 3.0)
    videos = [_video("AAAAAAAAAAA"), _video("BBBBBBBBBBB")]

    dest = transcribe_videos(EngineSpec("qwen3asr_large"), videos, audio_dir=audio_dir, out_dir=out_dir, device="cpu")

    assert dest == out_dir / "qwen3asr_large"
    record = json.loads((dest / "AAAAAAAAAAA.json").read_text(encoding="utf-8"))
    # 無音の区間は EnergyGate で落ち、ASR は呼ばれない (CLI と同じ後段)
    assert [s[2] for s in record["segments"]] == ["区間1"]
    assert record["drop_counts"] == {"energy_gate:low_rms": 1}
    # VAD の区間ごとに、時刻と結果 (認識した / どこで落ちたか) が残る
    assert record["vad_segments"] == [[0.0, 1.4, "asr"], [1.6, 2.9, "energy_gate:low_rms"]]
    assert record["success"] is True and record["rtf"] >= 0
    run = json.loads((dest / "run.json").read_text(encoding="utf-8"))
    assert run["engine_name"] == "fake" and run["args"]["vad"] == "auto" and run["args"]["confidence_filter"] == "on"
    assert fake_cli.calls == 2 and fake_cli.cleaned

    # 2 回目は書き起こし済みなので、engine を読み込まない
    fake_cli.calls = 0
    transcribe_videos(EngineSpec("qwen3asr_large"), videos, audio_dir=audio_dir, out_dir=out_dir, device="cpu")
    assert fake_cli.calls == 0
    assert load_hypotheses(dest, videos)["BBBBBBBBBBB"] == [(0.0, 1.4, "区間2")]


def _two_videos(tmp_path: Path) -> tuple[Path, Path, list[Yodas2Video]]:
    audio_dir, out_dir = tmp_path / "audio", tmp_path / "out"
    audio_dir.mkdir()
    for aid in ("AAAAAAAAAAA", "BBBBBBBBBBB"):
        _write_audio(audio_dir / f"{aid}.wav", 3.0)
    return audio_dir, out_dir, [_video("AAAAAAAAAAA"), _video("BBBBBBBBBBB")]


def test_resume_with_different_settings_stops_before_loading(tmp_path: Path, fake_cli) -> None:
    """CPU で途中まで書き起こした結果に、GPU の結果を混ぜない (#502 F1)。"""
    audio_dir, out_dir, videos = _two_videos(tmp_path)
    spec = EngineSpec("qwen3asr_large")
    transcribe_videos(spec, videos[:1], audio_dir=audio_dir, out_dir=out_dir, device="cpu")
    fake_cli.calls = 0
    for todo in (videos, videos[:1]):  # 残りがあるときも、全部済んでいるときも止まる
        with pytest.raises(RuntimeError, match="different or unknown settings"):
            transcribe_videos(spec, todo, audio_dir=audio_dir, out_dir=out_dir, device="gpu")
    assert fake_cli.calls == 0
    # 同じ設定なら残りだけを書き起こす。--force なら設定が違っても全部やり直す
    transcribe_videos(spec, videos, audio_dir=audio_dir, out_dir=out_dir, device="cpu")
    assert fake_cli.calls == 1
    transcribe_videos(spec, videos, audio_dir=audio_dir, out_dir=out_dir, device="gpu", force=True)
    run = json.loads((out_dir / spec.label / "run.json").read_text(encoding="utf-8"))
    assert run["args"]["device"] == "gpu"


def test_failed_and_broken_records_are_retried(tmp_path: Path, fake_cli) -> None:
    """失敗した記録・読めない記録は、再開のときにやり直す (#502 F3)。"""
    audio_dir, out_dir, videos = _two_videos(tmp_path)
    spec = EngineSpec("qwen3asr_large")
    dest = transcribe_videos(spec, videos, audio_dir=audio_dir, out_dir=out_dir, device="cpu")
    failed = json.loads((dest / "AAAAAAAAAAA.json").read_text(encoding="utf-8")) | {"success": False, "error": "boom"}
    (dest / "AAAAAAAAAAA.json").write_text(json.dumps(failed), encoding="utf-8")
    (dest / "BBBBBBBBBBB.json").write_text("{", encoding="utf-8")
    fake_cli.calls = 0
    transcribe_videos(spec, videos, audio_dir=audio_dir, out_dir=out_dir, device="cpu")
    assert fake_cli.calls == 2
    assert all(json.loads((dest / f"{v.audio_id}.json").read_text(encoding="utf-8"))["success"] for v in videos)


def test_interrupted_force_run_does_not_reuse_or_score_the_old_settings(tmp_path: Path, fake_cli, monkeypatch) -> None:
    """設定を変えた --force を途中で止めても、前の設定の記録を新しい設定の結果として使わない (#502 F1)。"""
    audio_dir, out_dir, videos = _two_videos(tmp_path)
    spec = EngineSpec("qwen3asr_large")
    dest = transcribe_videos(spec, videos, audio_dir=audio_dir, out_dir=out_dir, device="cpu")
    original = fake_cli.transcribe
    calls = {"n": 0}

    def stop_at_the_second_video(audio, sample_rate):
        calls["n"] += 1  # 動画ごとに ASR は 1 回 (後半の無音は EnergyGate で落ちる)
        if calls["n"] > 1:
            raise KeyboardInterrupt
        return original(audio, sample_rate)

    monkeypatch.setattr(fake_cli, "transcribe", stop_at_the_second_video)
    with pytest.raises(KeyboardInterrupt):
        transcribe_videos(spec, videos, audio_dir=audio_dir, out_dir=out_dir, device="gpu", force=True)
    # B は CPU の記録のまま。run.json は GPU なので、混ざったまま採点しない
    with pytest.raises(RuntimeError, match="settings other than run.json"):
        load_hypotheses(dest, videos)
    # 同じ GPU の設定で続けると、CPU の記録だけをやり直す
    monkeypatch.setattr(fake_cli, "transcribe", original)
    fake_cli.calls = 0
    transcribe_videos(spec, videos, audio_dir=audio_dir, out_dir=out_dir, device="gpu")
    assert fake_cli.calls == 1
    assert set(load_hypotheses(dest, videos)) == {v.audio_id for v in videos}


def test_replaced_audio_is_transcribed_again(tmp_path: Path, fake_cli) -> None:
    """同じ ID でも音声の中身が違えば、前の記録を使わない (#502 F1)。"""
    audio_dir, out_dir, videos = _two_videos(tmp_path)
    spec = EngineSpec("qwen3asr_large")
    transcribe_videos(spec, videos, audio_dir=audio_dir, out_dir=out_dir, device="cpu")
    _write_audio(audio_dir / "AAAAAAAAAAA.wav", 2.0)
    fake_cli.calls = 0
    transcribe_videos(spec, videos, audio_dir=audio_dir, out_dir=out_dir, device="cpu")
    assert fake_cli.calls == 1


@pytest.mark.parametrize(
    ("broken", "problem"),
    [
        ({"success": True}, "elapsed_sec"),
        ({"segments": [[0.0, 1.0, 5]]}, "segments"),
        ({"drop_counts": None}, "drop_counts"),
        ({"vad_segments": [[0.0, "1.0", "asr"]]}, "vad_segments"),
    ],
)
def test_incomplete_records_are_retried_and_not_scored(tmp_path: Path, fake_cli, broken, problem) -> None:
    """JSON として読めても、採点に要る項目・型がそろわない記録はやり直し、採点しない (#502 F3)。"""
    audio_dir, out_dir, videos = _two_videos(tmp_path)
    spec = EngineSpec("qwen3asr_large")
    dest = transcribe_videos(spec, videos, audio_dir=audio_dir, out_dir=out_dir, device="cpu")
    path = dest / "AAAAAAAAAAA.json"
    record = broken if broken == {"success": True} else json.loads(path.read_text(encoding="utf-8")) | broken
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match=problem):
        load_hypotheses(dest, videos)
    fake_cli.calls = 0
    transcribe_videos(spec, videos, audio_dir=audio_dir, out_dir=out_dir, device="cpu")
    assert fake_cli.calls == 1
    load_hypotheses(dest, videos)


def test_failures_left_after_the_run_are_reported(tmp_path: Path, fake_cli, monkeypatch) -> None:
    """失敗が残ったら、成功として終えない。記録は残り、次の呼び出しでやり直す (#502 F3)。"""
    audio_dir, out_dir, videos = _two_videos(tmp_path)

    def broken(audio, sample_rate):
        raise RuntimeError("model crashed")

    monkeypatch.setattr(fake_cli, "transcribe", broken)
    with pytest.raises(RuntimeError, match="2 of 2 videos were not transcribed"):
        transcribe_videos(EngineSpec("qwen3asr"), videos, audio_dir=audio_dir, out_dir=out_dir, device="cpu")
    record = json.loads((out_dir / "qwen3asr" / "AAAAAAAAAAA.json").read_text(encoding="utf-8"))
    assert record["success"] is False and "model crashed" in record["error"]


def test_missing_audio_is_reported_before_loading(tmp_path: Path, fake_cli) -> None:
    with pytest.raises(FileNotFoundError, match="prepare_yodas2_audio"):
        transcribe_videos(EngineSpec("qwen3asr"), [_video("CCCCCCCCCCC")], audio_dir=tmp_path, out_dir=tmp_path / "o")
    assert fake_cli.calls == 0


def test_failed_transcription_is_not_scored_as_empty(tmp_path: Path) -> None:
    run_dir = tmp_path / "x"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({"settings_id": "s1"}), encoding="utf-8")
    record = {
        "settings_id": "s1", "success": False, "error": "boom", "elapsed_sec": 1.0, "duration_sec": 3.0,
        "drop_counts": {}, "segments": [],
    }
    (run_dir / "DDDDDDDDDDD.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(RuntimeError, match="boom"):
        load_hypotheses(run_dir, [_video("DDDDDDDDDDD")])


def test_summary_files(tmp_path: Path, fake_cli) -> None:
    from benchmarks.common.yodas2 import Yodas2Segment

    audio_dir, out_dir = tmp_path / "audio", tmp_path / "out"
    audio_dir.mkdir()
    _write_audio(audio_dir / "EEEEEEEEEEE.wav", 3.0)
    video = Yodas2Video(
        "EEEEEEEEEEE", "test", "7", 3.0, "other", (), -20.0, "ja000", "00000001",
        (Yodas2Segment(0.0, 1.4, "区間1"),),
    )
    transcribe_videos(EngineSpec("qwen3asr"), [video], audio_dir=audio_dir, out_dir=out_dir, device="cpu")
    summary = summarize(out_dir, ["qwen3asr"], [video], split="test")
    assert summary["results"]["qwen3asr"]["scores"]["fillers-notation_excluded"]["pooled"]["rate"] == 0.0
    md = (out_dir / "summary.md").read_text(encoding="utf-8")
    assert "`qwen3asr`" in md and "| 7 |" in md
    assert summary["results"]["qwen3asr"]["speed"]["segment_basis"] == "vad"


def test_device_takes_the_cli_choices_only() -> None:
    """CLI の --device は auto / gpu / cpu。'cuda' は CLI が受け付けないので、入口で弾く。"""
    from benchmarks.longform.__main__ import main

    with pytest.raises(SystemExit):
        main(["transcribe", "--engine", "qwen3asr", "--device", "cuda"])


def test_max_speech_option_changes_only_the_cap_and_the_label() -> None:
    from benchmarks.longform.runner import _with_max_speech
    from livecap_cli.vad import VADConfig, VADFileSegmenter

    spec = EngineSpec.parse("qwen3asr", max_speech_ms=20000)
    assert spec.label == "qwen3asr-max20000ms"
    assert EngineSpec.parse("whispers2t:base", max_speech_ms=10000).label == "whispers2t-base-max10000ms"
    with pytest.raises(ValueError):
        EngineSpec.parse("qwen3asr", max_speech_ms=0)

    class Backend:
        name = "fake"
        frame_size = 512

        def process(self, frame):
            return 1.0  # always speech

        def reset(self):
            pass

    from livecap_cli.vad import VADProcessor

    original = VADFileSegmenter(VADProcessor(config=VADConfig(threshold=0.3, min_silence_ms=250), backend=Backend()))
    capped = _with_max_speech(original, 2000)
    assert capped._vad.config.max_speech_ms == 2000
    assert capped._vad.config.threshold == 0.3 and capped._vad.config.min_silence_ms == 250
    assert capped._vad._backend is original._vad._backend
    # 上限があると、続けて話していても 2 秒ごとに区間が切れる
    segments = capped(np.zeros(RATE * 7, dtype=np.float32), RATE)
    assert len(segments) >= 3
    assert max(end - start for start, end in segments) <= 2.1


def test_vad_stats_are_not_shown_for_part_of_the_videos(tmp_path: Path) -> None:
    """VAD の記録が一部の動画にしか無いとき、その値を全体の値に見せない (#502 F4)。"""
    from benchmarks.common.yodas2 import Yodas2Segment

    run_dir = tmp_path / "qwen3asr"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({"load_sec": 1.0}), encoding="utf-8")
    videos = [
        Yodas2Video(aid, "test", "1", 10.0, "other", (), -20.0, "ja000", "00000001", (Yodas2Segment(0.0, 1.0, "はい"),))
        for aid in ("OLDOLDOLDOL", "NEWNEWNEWNE")
    ]
    base = {"success": True, "error": None, "elapsed_sec": 1.0, "duration_sec": 10.0, "segments": [[0.0, 1.0, "はい"]]}
    old = base | {"drop_counts": {"energy_gate:low_rms": 1}}  # 記録を足す前の結果
    new = base | {"drop_counts": {}, "vad_segments": [[0.0, 1.0, "asr"]]}
    for aid, record in zip(("OLDOLDOLDOL", "NEWNEWNEWNE"), (old, new)):
        (run_dir / f"{aid}.json").write_text(json.dumps(record), encoding="utf-8")

    speed = summarize(tmp_path, ["qwen3asr"], videos, split="test")["results"]["qwen3asr"]["speed"]
    assert speed["vad_coverage"] == {"recorded": 1, "mismatch": 0, "absent": 1}
    assert speed["segment_basis"] == "output" and speed["dropped_sec"] is None
    md = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "未集計 (VAD の記録 1/2 本)" in md and "1.0 / 1 (出力)" in md

    # VAD が 1 区間も切らなかった動画も「記録あり」。全部そろえば VAD の値を出す
    (run_dir / "OLDOLDOLDOL.json").write_text(json.dumps(base | {"drop_counts": {}, "vad_segments": []}), encoding="utf-8")
    speed = summarize(tmp_path, ["qwen3asr"], videos, split="test")["results"]["qwen3asr"]["speed"]
    assert speed["segment_basis"] == "vad" and speed["dropped_sec"] == {}
    assert "| `qwen3asr` | なし | なし |" in (tmp_path / "summary.md").read_text(encoding="utf-8")
