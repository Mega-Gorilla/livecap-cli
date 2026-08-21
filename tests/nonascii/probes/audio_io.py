"""音声 I/O と出力側の境界プローブ (Issue #378)。

観測 dict にパスを含めないこと (control との等値比較に使うため)。
"""

from __future__ import annotations

import math
import wave

from ..record import ProbeContext, ProbeSkipped
from . import probe

_SR = 16000
_SECONDS = 0.2


def _write_tone_wav(path, seconds: float = _SECONDS) -> None:
    """CPython の wave モジュールだけで生成する (soundfile 非依存)。

    プローブ対象のライブラリで入力を作ると、失敗の原因が入力生成側なのか
    測定対象側なのか分からなくなるため。
    """
    n = int(_SR * seconds)
    frames = bytearray()
    for i in range(n):
        v = int(12000 * math.sin(2 * math.pi * 440.0 * i / _SR))
        frames += int(v).to_bytes(2, "little", signed=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_SR)
        w.writeframes(bytes(frames))


def _digest(samples) -> dict:
    """パスに依存しない観測。"""
    import numpy as np

    arr = np.asarray(samples, dtype="float64").reshape(-1)
    return {
        "n_samples": int(arr.size),
        "dtype_kind": str(arr.dtype.kind),
        "rms_e6": int(round(float(np.sqrt(np.mean(arr**2))) * 1_000_000)),
    }


@probe("soundfile.read.path")
def soundfile_read_path(ctx: ProbeContext) -> dict:
    """``audio_sources/file.py`` の ``sf.read(self.file_path)`` 相当。

    soundfile は Windows で ``sf_wchar_open`` を使う (soundfile.py で実物確認済み)
    ので wide path 対応が期待されるが、実測で確定させる。
    """
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ProbeSkipped(f"soundfile 未導入: {exc}") from exc

    path = ctx.root / "input.wav"
    _write_tone_wav(path)
    ctx.stage("write_wav")

    # 実コードと同じく **Path オブジェクトのまま** 渡す (str 化しない)
    data, sr = sf.read(path, dtype="float32")
    ctx.stage("sf_read")
    return {"sample_rate": int(sr), **_digest(data)}


@probe("soundfile.write.path")
def soundfile_write_path(ctx: ProbeContext) -> dict:
    """発話ごとの一時 wav 書き込み (``sf.write``) 相当。"""
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise ProbeSkipped(f"soundfile/numpy 未導入: {exc}") from exc

    audio = np.sin(
        2 * np.pi * 440.0 * np.arange(int(_SR * _SECONDS), dtype=np.float64) / _SR
    ).astype(np.float32)
    path = ctx.root / "out.wav"
    sf.write(str(path), audio, _SR)
    ctx.stage("sf_write")

    back, sr = sf.read(str(path), dtype="float32")
    ctx.stage("sf_read_back")
    return {"sample_rate": int(sr), "bytes": path.stat().st_size, **_digest(back)}


@probe("librosa.load.path")
def librosa_load_path(ctx: ProbeContext) -> dict:
    """``file_pipeline._load_audio`` が使う librosa の内部リーダ経路。"""
    try:
        import librosa
    except ImportError as exc:
        raise ProbeSkipped(f"librosa 未導入: {exc}") from exc

    path = ctx.root / "input.wav"
    _write_tone_wav(path)
    ctx.stage("write_wav")

    data, sr = librosa.load(str(path), sr=_SR, mono=True)
    ctx.stage("librosa_load")
    return {"sample_rate": int(sr), **_digest(data)}


@probe("stdlib.open_read")
def stdlib_open_read(ctx: ProbeContext) -> dict:
    """``base_engine._verify_model_integrity`` の ``open(model_path, 'rb')`` 相当。

    CPython は ``*W`` API を使うので通るはず — 「②wide-path」の根拠を実測で固定する。
    """
    path = ctx.root / "model.bin"
    path.write_bytes(b"\x00\x01\x02\x03" * 64)
    ctx.stage("write")

    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(4096), b""):
            h.update(chunk)
    ctx.stage("read")
    return {"size": path.stat().st_size, "sha256": h.hexdigest()}


@probe("srt.write_srt")
def srt_write_srt(ctx: ProbeContext) -> dict:
    """``transcription/srt.py::write_srt`` の出力側境界。"""
    try:
        from livecap_cli.transcription.file_pipeline import FileSubtitleSegment
        from livecap_cli.transcription.srt import write_srt
    except ImportError as exc:
        raise ProbeSkipped(f"livecap_cli 未 import: {exc}") from exc

    segments = [
        FileSubtitleSegment(index=1, start=0.0, end=1.5, text="こんにちは"),
        FileSubtitleSegment(index=2, start=1.5, end=3.0, text="world"),
    ]
    out = ctx.root / "out.srt"
    write_srt(out, segments)
    ctx.stage("write_srt")

    text = out.read_text(encoding="utf-8")
    ctx.stage("read_back")
    return {"chars": len(text), "lines": text.count("\n"), "has_cjk": "こんにちは" in text}
