"""Prepare local audio for the speaker-embedding benchmark (issue #287 spike).

Downloads (or accepts) a conversation recording, cuts a segment, and resamples
it to 16 kHz mono WAV under ``benchmarks/speaker/data/`` (gitignored).

The reference clip is git-unshareable, so output stays local only.

Examples
--------
    # Default: fetch the configured stream segment via yt-dlp + ffmpeg.
    uv run python scripts/prepare_speaker_benchmark.py

    # Use a local file you already have (any ffmpeg-readable format).
    uv run python scripts/prepare_speaker_benchmark.py --input D:/clips/talk.m4a

Requirements
------------
    - yt-dlp (download): ``uv pip install yt-dlp``
    - ffmpeg (cut/resample): on PATH, in ./ffmpeg-bin/, or via ``imageio-ffmpeg``.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger("prepare_speaker_benchmark")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmarks" / "speaker" / "data" / "source_10min.wav"

# Default reference: 2-person conversation stream, 1:12:06 for 10 minutes.
DEFAULT_URL = "https://www.youtube.com/watch?v=my6prODcDM4"
DEFAULT_START = "01:12:06"
DEFAULT_DURATION_S = 600

TARGET_SR = 16000


def _hms_to_seconds(value: str) -> int:
    """Parse 'HH:MM:SS' / 'MM:SS' / 'SS' into seconds."""
    parts = [int(p) for p in value.split(":")]
    seconds = 0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def _seconds_to_hms(total: int) -> str:
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def resolve_ffmpeg(explicit: str | None = None) -> str:
    """Locate an ffmpeg executable (explicit > PATH > ffmpeg-bin/ > imageio-ffmpeg)."""
    if explicit:
        return explicit

    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path

    for name in ("ffmpeg.exe", "ffmpeg"):
        candidate = PROJECT_ROOT / "ffmpeg-bin" / name
        if candidate.exists():
            return str(candidate)

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover
        pass

    raise RuntimeError(
        "ffmpeg not found. Install it (PATH / ./ffmpeg-bin/) or run "
        "`uv pip install imageio-ffmpeg`."
    )


def _run(cmd: list[str]) -> None:
    logger.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def download_section(url: str, start: str, duration_s: int, ffmpeg: str, dest: Path) -> Path:
    """Download just the requested section via yt-dlp into ``dest`` (audio)."""
    if shutil.which("yt-dlp") is None:
        # Fall back to module form if installed as a library.
        try:
            import yt_dlp  # noqa: F401

            ytdlp_cmd = [sys.executable, "-m", "yt_dlp"]
        except ImportError as e:
            raise RuntimeError(
                "yt-dlp not found. Install with: uv pip install yt-dlp"
            ) from e
    else:
        ytdlp_cmd = ["yt-dlp"]

    end = _seconds_to_hms(_hms_to_seconds(start) + duration_s)
    section = f"*{start}-{end}"
    out_template = str(dest.with_suffix(".%(ext)s"))

    _run(
        ytdlp_cmd
        + [
            "-f", "bestaudio",
            "--download-sections", section,
            "--force-keyframes-at-cuts",
            "--ffmpeg-location", ffmpeg,
            "-o", out_template,
            url,
        ]
    )

    candidates = sorted(dest.parent.glob(dest.stem + ".*"))
    audio = [c for c in candidates if c.suffix.lower() != ".wav" or c != dest]
    if not audio:
        raise RuntimeError(f"yt-dlp produced no file matching {dest.stem}.*")
    return audio[0]


def to_16k_mono_wav(
    ffmpeg: str, src: Path, output: Path, start_s: int | None, duration_s: int
) -> None:
    """Cut (if start_s given) and resample ``src`` to 16 kHz mono WAV."""
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y"]
    if start_s is not None:
        cmd += ["-ss", str(start_s)]
    cmd += ["-i", str(src), "-t", str(duration_s)]
    cmd += ["-ar", str(TARGET_SR), "-ac", "1", "-f", "wav", str(output)]
    _run(cmd)


def verify(output: Path) -> None:
    import soundfile as sf

    info = sf.info(str(output))
    logger.info(
        "Output: %s | %.1fs | %d Hz | %d ch",
        output, info.duration, info.samplerate, info.channels,
    )
    if info.samplerate != TARGET_SR or info.channels != 1:
        logger.warning("Expected 16 kHz mono; got %d Hz / %d ch", info.samplerate, info.channels)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=DEFAULT_URL, help="Source URL (yt-dlp).")
    parser.add_argument("--start", default=DEFAULT_START, help="Start time HH:MM:SS.")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION_S, help="Duration (seconds).")
    parser.add_argument("--input", type=Path, default=None, help="Local audio file (skips download).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output 16k mono WAV.")
    parser.add_argument("--ffmpeg-location", default=None, help="Explicit ffmpeg path.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        ffmpeg = resolve_ffmpeg(args.ffmpeg_location)
        logger.info("Using ffmpeg: %s", ffmpeg)

        if args.input is not None:
            # Local file: cut [start, start+duration] and resample.
            if not args.input.exists():
                raise FileNotFoundError(f"Input not found: {args.input}")
            to_16k_mono_wav(
                ffmpeg, args.input, args.output,
                start_s=_hms_to_seconds(args.start), duration_s=args.duration,
            )
        else:
            # Download just the section, then resample the whole clip.
            with tempfile.TemporaryDirectory() as tmp:
                tmp_base = Path(tmp) / "section"
                downloaded = download_section(
                    args.url, args.start, args.duration, ffmpeg, tmp_base
                )
                to_16k_mono_wav(
                    ffmpeg, downloaded, args.output, start_s=None, duration_s=args.duration
                )

        verify(args.output)
        logger.info("Done. (Local-only; do NOT commit %s)", args.output)
        return 0
    except subprocess.CalledProcessError as e:
        logger.error("External command failed (%s).", e)
        return 1
    except Exception as e:
        logger.error("%s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
