"""``scripts/prepare_yodas2_audio.py`` の取り出しと変換 (#488)。

YODAS2 の shard (tar.gz) から、一覧にある動画だけを取り出し、16 kHz・モノラル・16 bit にする。
**音量は変えない** (元の音量の差も評価の対象)。ネットワークは使わない。
"""

from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
pytest.importorskip("scipy")

from benchmarks.common.yodas2 import Yodas2Video  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "prepare_yodas2_audio.py"


def _load_module():
    """Import by path: ``scripts`` is not an importable package."""
    spec = importlib.util.spec_from_file_location("_prepare_yodas2_audio", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = _load_module()


def _wav_bytes(seconds: float, rate: int, amplitude: float, channels: int) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    tone = (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, np.stack([tone] * channels, axis=1), rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _shard(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def _video(audio_id: str, seconds: float) -> Yodas2Video:
    return Yodas2Video(audio_id, "test", "1", seconds, "other", (), -20.0, "ja000", "00000001", ())


def test_extracts_only_listed_videos_as_16k_mono_without_changing_the_level(tmp_path: Path) -> None:
    wanted_id, other_id = "AAAAAAAAAAA", "BBBBBBBBBBB"
    archive = _shard(
        tmp_path / "00000001.tar.gz",
        {
            f"ja000/{wanted_id}.wav": _wav_bytes(1.0, 24000, 0.25, channels=2),
            f"ja000/{other_id}.wav": _wav_bytes(1.0, 24000, 0.25, channels=1),
            "ja000/readme.txt": b"not audio",
        },
    )
    out = tmp_path / "out"
    out.mkdir()

    found = prepare.extract(archive, {wanted_id: _video(wanted_id, 1.0)}, out)

    assert found == {wanted_id}
    assert sorted(p.name for p in out.iterdir()) == [f"{wanted_id}.wav"]
    audio, rate = sf.read(out / f"{wanted_id}.wav", dtype="float32")
    info = sf.info(out / f"{wanted_id}.wav")
    assert (rate, info.channels, info.subtype) == (16000, 1, "PCM_16")
    assert len(audio) == 16000
    # 音量をそろえない: 振幅 0.25 のまま (-1 dBFS などへ持ち上げない)
    assert np.abs(audio[100:-100]).max() == pytest.approx(0.25, abs=0.01)


def test_output_name_comes_from_the_manifest_not_the_tar_path(tmp_path: Path) -> None:
    """tar の中のディレクトリは使わない。出力先は一覧の ID から作る。"""
    audio_id = "CCCCCCCCCCC"
    archive = _shard(tmp_path / "s.tar.gz", {f"../../outside/{audio_id}.wav": _wav_bytes(0.5, 16000, 0.1, channels=1)})
    out = tmp_path / "out"
    out.mkdir()

    assert prepare.extract(archive, {audio_id: _video(audio_id, 0.5)}, out) == {audio_id}
    assert (out / f"{audio_id}.wav").exists()
    assert not (tmp_path.parent / "outside").exists()


def test_missing_video_is_reported_as_not_found(tmp_path: Path) -> None:
    archive = _shard(tmp_path / "s.tar.gz", {"ja000/DDDDDDDDDDD.wav": _wav_bytes(0.5, 16000, 0.1, channels=1)})
    out = tmp_path / "out"
    out.mkdir()
    assert prepare.extract(archive, {"EEEEEEEEEEE": _video("EEEEEEEEEEE", 0.5)}, out) == set()


def test_dry_run_lists_the_shards_without_downloading(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "argv", ["prepare_yodas2_audio.py", "--split", "test", "--dry-run", "--output-dir", str(tmp_path)])
    assert prepare.main() == 0
    out = capsys.readouterr().out
    assert "31 本のうち 31 本" in out and "shard は 16 個" in out
    assert not any(tmp_path.iterdir())
