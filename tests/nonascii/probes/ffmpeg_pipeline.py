"""ffmpeg と file_pipeline の境界プローブ (Issue #378)。

ffmpeg-python は argv list を組んで ``subprocess.Popen`` する (シェル文字列では
ない) ので、Windows では ``CreateProcessW`` 経由になる。実測で確定させる。

``space_paren`` variant がここで効く — 空白と括弧は **エンコーディングではなく
argv quoting** の問題であり、ASCII staging では直らない別 family である。
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import wave
from pathlib import Path

from ..record import ProbeContext, ProbeSkipped
from . import probe

_SR = 16000
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_tone_wav(path: Path, seconds: float = 0.5) -> None:
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


def _ffmpeg_binary() -> str:
    """``ffmpeg-bin/`` → ``LIVECAP_FFMPEG_BIN`` → PATH の順で解決する。"""
    import os

    env_value = os.environ.get("LIVECAP_FFMPEG_BIN")
    if env_value:
        candidate = Path(env_value)
        if candidate.is_dir():
            candidate = candidate / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        if candidate.is_file():
            return str(candidate)

    packaged = _REPO_ROOT / "ffmpeg-bin" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if packaged.is_file():
        return str(packaged)

    which = shutil.which("ffmpeg")
    if which:
        return which
    raise ProbeSkipped("ffmpeg が見つからない (ffmpeg-bin / LIVECAP_FFMPEG_BIN / PATH)")


def _wav_observation(path: Path) -> dict:
    with wave.open(str(path), "rb") as w:
        return {
            "channels": w.getnchannels(),
            "sampwidth": w.getsampwidth(),
            "framerate": w.getframerate(),
            "nframes_nonzero": w.getnframes() > 0,
        }


@probe("ffmpeg.input_path")
def ffmpeg_input_path(ctx: ProbeContext) -> dict:
    """``ffmpeg.input(str(source))`` — 入力パスのみ非 ASCII。

    出力は ASCII 側に置き、入力パスだけを変数にする。
    """
    try:
        import ffmpeg
    except ImportError as exc:
        raise ProbeSkipped(f"ffmpeg-python 未導入: {exc}") from exc

    binary = _ffmpeg_binary()
    source = ctx.root / "input.wav"
    _write_tone_wav(source)
    ctx.stage("write_input")

    # 出力は常に ASCII の一時領域に置く (入力側だけを測るため)。
    # runner が payload["ascii_scratch"] を必ず渡す。
    scratch = Path(ctx.payload["ascii_scratch"]) / ("control" if ctx.is_control else "trial")
    scratch.mkdir(parents=True, exist_ok=True)
    ascii_out = scratch / "out_input_probe.wav"

    stream = ffmpeg.input(str(source))
    stream = ffmpeg.output(stream, str(ascii_out), ac=1, ar=_SR, acodec="pcm_s16le")
    stream = ffmpeg.overwrite_output(stream)
    ffmpeg.run(stream, cmd=binary, capture_stdout=True, capture_stderr=True)
    ctx.stage("ffmpeg_run")

    return {"bytes_nonzero": ascii_out.stat().st_size > 0, **_wav_observation(ascii_out)}


@probe("ffmpeg.output_path")
def ffmpeg_output_path(ctx: ProbeContext) -> dict:
    """``ffmpeg.output(..., str(destination))`` — 出力パスが非 ASCII。

    実コードでは destination が ``f"{source.stem}_audio.wav"`` なので、
    **ユーザーのファイル名**がそのまま出力パスに流れ込む。
    """
    try:
        import ffmpeg
    except ImportError as exc:
        raise ProbeSkipped(f"ffmpeg-python 未導入: {exc}") from exc

    binary = _ffmpeg_binary()
    source = ctx.root / "src.wav"
    _write_tone_wav(source)
    ctx.stage("write_input")

    destination = ctx.root / "converted_audio.wav"
    stream = ffmpeg.input(str(source))
    stream = ffmpeg.output(stream, str(destination), ac=1, ar=_SR, acodec="pcm_s16le")
    stream = ffmpeg.overwrite_output(stream)
    ffmpeg.run(stream, cmd=binary, capture_stdout=True, capture_stderr=True)
    ctx.stage("ffmpeg_run")

    return {"bytes_nonzero": destination.stat().st_size > 0, **_wav_observation(destination)}


@probe("ffmpeg.binary_path")
def ffmpeg_binary_path(ctx: ProbeContext) -> dict:
    """**ffmpeg 実行ファイル自体**が非 ASCII パスにある場合。

    ``FFmpegManager`` は cache_root 配下に binary を配置するため、
    cache_root が非 ASCII ならこの経路になる。
    """
    binary = Path(_ffmpeg_binary())
    staged_dir = ctx.root / "bin"
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged = staged_dir / binary.name
    if not staged.exists():
        shutil.copy2(binary, staged)
    ctx.stage("stage_binary")

    proc = subprocess.run(
        [str(staged), "-hide_banner", "-version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    ctx.stage("run_binary")
    if proc.returncode != 0:
        raise RuntimeError(
            f"staged ffmpeg が異常終了 (exit={proc.returncode}): {proc.stderr[:400]}"
        )
    return {
        "exit_code": proc.returncode,
        "reports_version": "ffmpeg version" in (proc.stdout or ""),
    }


@probe("pipeline.extract_audio.nonascii_stem")
def pipeline_extract_audio_nonascii_stem(ctx: ProbeContext) -> dict:
    """``FileTranscriptionPipeline`` の抽出経路を非 ASCII **ファイル名**で通す。

    ``file_pipeline`` は作業ディレクトリを ``tempfile.mkdtemp()`` (= システム
    ``%TEMP%``) に作り、そこへ ``f"{source.stem}_audio.wav"`` を書く。つまり
    cache root が ASCII でも**ユーザーのファイル名**が非 ASCII なら temp パスが
    非 ASCII になる。cache root の行とは独立した行である。
    """
    try:
        from livecap_cli.transcription.file_pipeline import FileTranscriptionPipeline
    except ImportError as exc:
        raise ProbeSkipped(f"livecap_cli 未 import: {exc}") from exc

    # ファイル名側にも variant を効かせる (親ディレクトリは既に variant 配下)。
    # **決め打ちの非 ASCII 名は使わない** — ASCII-only variant (space_paren) に
    # encoding の要因が混入し、variant ごとの機構分離が壊れるため。
    from ..paths import variant as _variant

    stem = _variant(ctx.variant).segment
    source = ctx.root / f"{stem}.wav"
    _write_tone_wav(source, seconds=1.0)
    ctx.stage("write_source")

    seen: dict = {}

    def segmenter(audio, sample_rate):
        seen["n_samples"] = int(audio.size)
        seen["sample_rate"] = int(sample_rate)
        return [(0.0, 0.5)]

    pipeline = FileTranscriptionPipeline(segmenter=segmenter)
    try:
        result = pipeline.process_file(
            source,
            segment_transcriber=lambda audio, sr: "text",
            write_subtitles=False,
        )
    finally:
        pipeline.close()
    ctx.stage("process_file")

    return {
        "success": bool(result.success),
        "error": result.error,
        "n_subtitles": len(result.subtitles or []),
        "segmenter_saw_samples": seen.get("n_samples") is not None,
        "sample_rate": seen.get("sample_rate"),
    }


@probe("ffmpeg.path_env")
def ffmpeg_path_env(ctx: ProbeContext) -> dict:
    """**PATH に挿した非 ASCII の bin ディレクトリから実行ファイルを解決できるか。**

    `FFmpegManager._finalise_environment()` は Windows で、解決した ffmpeg の
    **bin ディレクトリを ``PATH`` の先頭へ挿す**。その後この process (と子孫) が
    ``ffmpeg`` を **basename だけ**で起動すると、`CreateProcessW` が PATH を辿って
    解決する。**そこが本境界である。**

    `ffmpeg.binary_path` (別行) は **フルパスを渡して**起動するので、PATH からの
    探索は測っていない。**あちらの pass をもって本境界を確認済みとはできない。**

    **production の `_finalise_environment()` を直接呼ぶ。** 手書きで同じ mutation を
    再現すると、**production 側の挿入条件・順序・正規化が壊れても probe は pass し
    続ける** — 「OS が非 ASCII PATH を解決できる」ことしか示せず、「**livecap-cli が
    その PATH を正しく構成している**」ことを示せない。

    公開入口の `configure_environment()` は使わない — `ensure_executable()` 経由で
    **ダウンロードを起こし得る** (cheap tier の「ネットワークを使わない」契約に反する)。
    `_finalise_environment()` 自体は **PATH を触るだけで I/O が無い**ので直接呼べる。

    **`os.environ` を書き換えるが、probe は子プロセスで走る** ので親には波及
    しない (`runner._child_env` は「親の os.environ は絶対に触らない」設計)。
    """
    if os.name != "nt":
        raise ProbeSkipped(
            "本境界は Windows 限定である "
            "(_finalise_environment は self._is_windows のときだけ PATH を触る)"
        )

    binary = Path(_ffmpeg_binary())
    staged_dir = ctx.root / "bin"
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged = staged_dir / binary.name
    if not staged.exists():
        shutil.copy2(binary, staged)
    ctx.stage("stage_binary")

    # **production の関数をそのまま通す。** 複製すると、あちらが壊れてもここは
    # 気付けない。`configure_environment()` (ダウンロードを起こし得る) ではなく、
    # PATH を触るだけの `_finalise_environment()` を直接呼ぶ。
    try:
        from livecap_cli.resources import get_ffmpeg_manager
    except ImportError as exc:  # pragma: no cover - livecap 未 import の環境
        raise ProbeSkipped(f"livecap_cli.resources 未 import: {exc}") from exc

    get_ffmpeg_manager()._finalise_environment(staged)
    ctx.stage("prepend_path")

    if str(staged_dir) not in os.environ.get("PATH", "").split(os.pathsep):
        raise RuntimeError(
            f"_finalise_environment() が PATH へ bin ディレクトリを挿さなかった: "
            f"{ascii(str(staged_dir))} - production 側の挿入条件が変わった可能性がある"
        )

    # **basename だけで起動する。** ここで CreateProcessW が PATH を辿る。
    proc = subprocess.run(
        [binary.name, "-hide_banner", "-version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    ctx.stage("run_from_path")
    if proc.returncode != 0:
        raise RuntimeError(
            f"PATH 経由の ffmpeg 起動が失敗 (exit={proc.returncode}): "
            f"bin_dir={staged_dir} / {proc.stderr[:400]}"
        )

    # **解決されたのが staged の方であること。** システムの ffmpeg を拾っていたら
    # 非 ASCII ディレクトリを一度も通っていない。
    resolved = shutil.which(binary.name)
    if resolved is None or Path(resolved).resolve() != staged.resolve():
        raise RuntimeError(
            f"PATH が staged の bin を先頭で解決していない: {ascii(str(resolved))} "
            f"(期待 {ascii(str(staged))}) - 非 ASCII ディレクトリを通っていない"
        )
    return {
        "exit_code": proc.returncode,
        "reports_version": "ffmpeg version" in (proc.stdout or ""),
        "resolved_from_staged_dir": True,
    }
