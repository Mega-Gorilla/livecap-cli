#!/usr/bin/env python3
"""Benchmark data preparation script.

Converts source corpora (JSUT, LibriSpeech) to unified format for benchmarking.

Output format:
- WAV, 16kHz, mono, 16bit
- Peak normalized to -1dBFS
- Transcripts as UTF-8 .txt files

Usage:
    # Standard mode (100 files per language from basic5000/test-clean)
    python scripts/prepare_benchmark_data.py --mode standard

    # Full mode (all files from all subsets)
    python scripts/prepare_benchmark_data.py --mode full

    # Custom limits
    python scripts/prepare_benchmark_data.py --ja-limit 500 --en-limit 200

    # Force overwrite existing files
    python scripts/prepare_benchmark_data.py --mode standard --force

Environment Variables:
    LIVECAP_JSUT_DIR: Path to JSUT corpus (default: tests/assets/source/jsut/jsut_ver1.1)
    LIVECAP_LIBRISPEECH_DIR: Path to LibriSpeech (default: tests/assets/source/librispeech/test-clean)
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

# Optional imports with graceful fallback
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    from scipy.signal import resample_poly
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ASSETS_DIR = PROJECT_ROOT / "tests" / "assets"

# Default source paths
DEFAULT_JSUT_DIR = ASSETS_DIR / "source" / "jsut" / "jsut_ver1.1"
DEFAULT_LIBRISPEECH_DIR = ASSETS_DIR / "source" / "librispeech" / "test-clean"

# Output paths
PREPARED_DIR = ASSETS_DIR / "prepared"
AUDIO_DIR = ASSETS_DIR / "audio"  # For checking existing files to skip

# Target audio format
TARGET_SAMPLE_RATE = 16000
TARGET_PEAK_DB = -1.0  # dBFS
TARGET_PEAK_LINEAR = 10 ** (TARGET_PEAK_DB / 20)  # ~0.891

# JSUT subsets
JSUT_SUBSETS_STANDARD = ["basic5000"]
JSUT_SUBSETS_FULL = [
    "basic5000",
    "countersuffix26",
    "loanword128",
    "onomatopee300",
    "repeat500",
    "travel1000",
    "utparaphrase512",
    "voiceactress100",
]

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Statistics for processing results."""
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.failed += 1

    def summary(self) -> str:
        return f"Processed: {self.processed}, Skipped: {self.skipped}, Failed: {self.failed}"


def natural_sort_key(s: str) -> list:
    """Sort key for natural sorting (e.g., file1, file2, file10)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def check_dependencies() -> bool:
    """Check if required dependencies are available."""
    missing = []
    if not SOUNDFILE_AVAILABLE:
        missing.append("soundfile")
    if not SCIPY_AVAILABLE:
        missing.append("scipy")

    if missing:
        logger.error(f"Missing required dependencies: {', '.join(missing)}")
        logger.error("Install with: pip install soundfile scipy")
        return False
    return True


def convert_audio(
    input_path: Path,
    output_path: Path,
    target_sr: int = TARGET_SAMPLE_RATE,
    target_peak: float = TARGET_PEAK_LINEAR,
) -> None:
    """Convert audio to unified format (16kHz mono WAV, peak normalized).

    Args:
        input_path: Source audio file (WAV, FLAC, etc.)
        output_path: Output WAV file path
        target_sr: Target sample rate (default: 16000)
        target_peak: Target peak level in linear scale (default: -1dBFS)
    """
    # Read audio
    audio, sr = sf.read(input_path)

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        # Use resample_poly for better quality
        gcd = np.gcd(target_sr, sr)
        up = target_sr // gcd
        down = sr // gcd
        audio = resample_poly(audio, up, down)

    # Peak normalize
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio * (target_peak / peak)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write as 16-bit PCM WAV
    sf.write(output_path, audio, target_sr, subtype="PCM_16")


def get_existing_audio_stems(lang: str) -> set[str]:
    """Get stems of existing audio files in audio/{lang}/ to avoid duplicates."""
    audio_lang_dir = AUDIO_DIR / lang
    if not audio_lang_dir.exists():
        return set()

    stems = set()
    for f in audio_lang_dir.glob("*.wav"):
        stems.add(f.stem)
    return stems


# =============================================================================
# JSUT Processing
# =============================================================================

@dataclass
class JSUTUtterance:
    """A single JSUT utterance."""
    id: str
    text: str
    audio_path: Path
    subset: str


def parse_jsut_transcript(transcript_path: Path, subset: str) -> dict[str, str]:
    """Parse JSUT transcript_utf8.txt file.

    Format: ID:テキスト (e.g., BASIC5000_0001:水をマレーシアから買わなくてはならない。)

    Returns:
        Dict mapping utterance ID to transcript text
    """
    transcripts = {}
    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            # Split on first colon only (text may contain colons)
            parts = line.split(":", 1)
            if len(parts) == 2:
                utt_id, text = parts
                transcripts[utt_id.strip()] = text.strip()
    return transcripts


def iter_jsut_utterances(
    jsut_dir: Path,
    subsets: list[str],
    limit: int | None = None,
) -> Iterator[JSUTUtterance]:
    """Iterate over JSUT utterances from specified subsets.

    Args:
        jsut_dir: Path to JSUT corpus root (jsut_ver1.1/)
        subsets: List of subset names to process
        limit: Maximum number of utterances to yield (None for all)

    Yields:
        JSUTUtterance objects
    """
    count = 0

    for subset in subsets:
        subset_dir = jsut_dir / subset
        if not subset_dir.exists():
            logger.warning(f"JSUT subset not found: {subset_dir}")
            continue

        # Find transcript file
        transcript_path = subset_dir / "transcript_utf8.txt"
        if not transcript_path.exists():
            logger.warning(f"Transcript not found: {transcript_path}")
            continue

        # Parse transcripts
        transcripts = parse_jsut_transcript(transcript_path, subset)

        # Find WAV files
        wav_dir = subset_dir / "wav"
        if not wav_dir.exists():
            logger.warning(f"WAV directory not found: {wav_dir}")
            continue

        # Sort files naturally
        wav_files = sorted(wav_dir.glob("*.wav"), key=lambda p: natural_sort_key(p.stem))

        for wav_path in wav_files:
            if limit is not None and count >= limit:
                return

            utt_id = wav_path.stem
            if utt_id not in transcripts:
                logger.debug(f"No transcript for: {utt_id}")
                continue

            yield JSUTUtterance(
                id=utt_id,
                text=transcripts[utt_id],
                audio_path=wav_path,
                subset=subset,
            )
            count += 1


def process_jsut(
    jsut_dir: Path,
    output_dir: Path,
    subsets: list[str],
    limit: int | None = None,
    force: bool = False,
    quiet: bool = False,
) -> ProcessingStats:
    """Process JSUT corpus and output to prepared/ja/.

    Args:
        jsut_dir: Path to JSUT corpus
        output_dir: Output directory (prepared/ja/)
        subsets: List of subsets to process
        limit: Maximum files to process (None for all)
        force: Overwrite existing files
        quiet: Suppress progress bar

    Returns:
        ProcessingStats with results
    """
    stats = ProcessingStats()

    if not jsut_dir.exists():
        logger.error(f"JSUT directory not found: {jsut_dir}")
        logger.error("Download JSUT from: https://sites.google.com/site/shinnosuketakamichi/publication/jsut")
        logger.error(f"Or set LIVECAP_JSUT_DIR environment variable")
        return stats

    # Get existing files to skip
    existing_stems = get_existing_audio_stems("ja")

    # Collect utterances
    utterances = list(iter_jsut_utterances(jsut_dir, subsets, limit))

    if not utterances:
        logger.warning("No JSUT utterances found")
        return stats

    logger.info(f"Processing {len(utterances)} JSUT utterances from subsets: {subsets}")

    # Setup progress bar
    if TQDM_AVAILABLE and not quiet:
        utterances = tqdm(utterances, desc="JSUT", unit="files")

    output_dir.mkdir(parents=True, exist_ok=True)

    for utt in utterances:
        # Output filename: jsut_{subset}_{id}.wav
        # e.g., jsut_basic5000_0001.wav
        subset_short = utt.subset.lower()
        # Extract numeric part from ID (BASIC5000_0001 -> 0001)
        id_match = re.search(r"_(\d+)$", utt.id)
        if id_match:
            id_num = id_match.group(1)
            out_stem = f"jsut_{subset_short}_{id_num}"
        else:
            out_stem = f"jsut_{subset_short}_{utt.id}"

        # Skip if exists in audio/ (quick mode fixtures)
        if out_stem in existing_stems:
            stats.skipped += 1
            continue

        out_wav = output_dir / f"{out_stem}.wav"
        out_txt = output_dir / f"{out_stem}.txt"

        # Skip if already exists (unless force)
        if out_wav.exists() and out_txt.exists() and not force:
            stats.skipped += 1
            continue

        try:
            # Convert audio
            convert_audio(utt.audio_path, out_wav)

            # Write transcript
            out_txt.write_text(utt.text + "\n", encoding="utf-8")

            stats.processed += 1
        except Exception as e:
            stats.add_error(f"JSUT {utt.id}: {e}")
            logger.debug(f"Failed to process {utt.id}: {e}")

    return stats


# =============================================================================
# LibriSpeech Processing
# =============================================================================

@dataclass
class LibriSpeechUtterance:
    """A single LibriSpeech utterance."""
    id: str  # e.g., 1089-134686-0001
    text: str
    audio_path: Path
    speaker_id: str
    chapter_id: str


def parse_librispeech_transcript(trans_path: Path) -> dict[str, str]:
    """Parse LibriSpeech *.trans.txt file.

    Format: ID TEXT (e.g., 1089-134686-0001 STUFF IT INTO YOU HIS BELLY COUNSELLED HIM)

    Returns:
        Dict mapping utterance ID to transcript text
    """
    transcripts = {}
    with open(trans_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split on first space
            parts = line.split(" ", 1)
            if len(parts) == 2:
                utt_id, text = parts
                transcripts[utt_id.strip()] = text.strip()
    return transcripts


def iter_librispeech_utterances(
    librispeech_dir: Path,
    limit: int | None = None,
) -> Iterator[LibriSpeechUtterance]:
    """Iterate over LibriSpeech utterances.

    Expected structure:
    test-clean/LibriSpeech/test-clean/
    ├── 1089/
    │   ├── 134686/
    │   │   ├── 1089-134686-0000.flac
    │   │   └── 1089-134686.trans.txt

    Args:
        librispeech_dir: Path to LibriSpeech corpus
        limit: Maximum number of utterances to yield

    Yields:
        LibriSpeechUtterance objects
    """
    count = 0

    # Handle nested LibriSpeech structure
    # Could be test-clean/ or test-clean/LibriSpeech/test-clean/
    search_dir = librispeech_dir
    nested = librispeech_dir / "LibriSpeech" / librispeech_dir.name
    if nested.exists():
        search_dir = nested

    # Find all speaker directories
    speaker_dirs = sorted(
        [d for d in search_dir.iterdir() if d.is_dir()],
        key=lambda p: natural_sort_key(p.name)
    )

    for speaker_dir in speaker_dirs:
        if limit is not None and count >= limit:
            return

        speaker_id = speaker_dir.name

        # Find chapter directories
        chapter_dirs = sorted(
            [d for d in speaker_dir.iterdir() if d.is_dir()],
            key=lambda p: natural_sort_key(p.name)
        )

        for chapter_dir in chapter_dirs:
            if limit is not None and count >= limit:
                return

            chapter_id = chapter_dir.name

            # Find transcript file
            trans_files = list(chapter_dir.glob("*.trans.txt"))
            if not trans_files:
                continue

            transcripts = parse_librispeech_transcript(trans_files[0])

            # Find FLAC files
            flac_files = sorted(
                chapter_dir.glob("*.flac"),
                key=lambda p: natural_sort_key(p.stem)
            )

            for flac_path in flac_files:
                if limit is not None and count >= limit:
                    return

                utt_id = flac_path.stem
                if utt_id not in transcripts:
                    continue

                yield LibriSpeechUtterance(
                    id=utt_id,
                    text=transcripts[utt_id],
                    audio_path=flac_path,
                    speaker_id=speaker_id,
                    chapter_id=chapter_id,
                )
                count += 1


def process_librispeech(
    librispeech_dir: Path,
    output_dir: Path,
    limit: int | None = None,
    force: bool = False,
    quiet: bool = False,
) -> ProcessingStats:
    """Process LibriSpeech corpus and output to prepared/en/.

    Args:
        librispeech_dir: Path to LibriSpeech corpus
        output_dir: Output directory (prepared/en/)
        limit: Maximum files to process (None for all)
        force: Overwrite existing files
        quiet: Suppress progress bar

    Returns:
        ProcessingStats with results
    """
    stats = ProcessingStats()

    if not librispeech_dir.exists():
        logger.error(f"LibriSpeech directory not found: {librispeech_dir}")
        logger.error("Download from: https://www.openslr.org/12")
        logger.error(f"Or set LIVECAP_LIBRISPEECH_DIR environment variable")
        return stats

    # Get existing files to skip
    existing_stems = get_existing_audio_stems("en")

    # Collect utterances
    utterances = list(iter_librispeech_utterances(librispeech_dir, limit))

    if not utterances:
        logger.warning("No LibriSpeech utterances found")
        return stats

    logger.info(f"Processing {len(utterances)} LibriSpeech utterances")

    # Setup progress bar
    if TQDM_AVAILABLE and not quiet:
        utterances = tqdm(utterances, desc="LibriSpeech", unit="files")

    output_dir.mkdir(parents=True, exist_ok=True)

    for utt in utterances:
        # Output filename: librispeech_{speaker}-{chapter}-{seq}.wav
        # e.g., librispeech_1089-134686-0001.wav
        out_stem = f"librispeech_{utt.id}"

        # Skip if exists in audio/ (quick mode fixtures)
        if out_stem in existing_stems:
            stats.skipped += 1
            continue

        out_wav = output_dir / f"{out_stem}.wav"
        out_txt = output_dir / f"{out_stem}.txt"

        # Skip if already exists (unless force)
        if out_wav.exists() and out_txt.exists() and not force:
            stats.skipped += 1
            continue

        try:
            # Convert audio (FLAC -> WAV)
            convert_audio(utt.audio_path, out_wav)

            # Write transcript
            out_txt.write_text(utt.text + "\n", encoding="utf-8")

            stats.processed += 1
        except Exception as e:
            stats.add_error(f"LibriSpeech {utt.id}: {e}")
            logger.debug(f"Failed to process {utt.id}: {e}")

    return stats


# =============================================================================
# Main CLI
# =============================================================================

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare benchmark data from source corpora",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--mode",
        choices=["standard", "full"],
        default="standard",
        help="Processing mode: standard (100 files/lang from basic5000/test-clean) or full (all files from all subsets)",
    )
    parser.add_argument(
        "--ja-limit",
        type=int,
        default=None,
        help="Override Japanese file limit (default: 100 for standard, unlimited for full)",
    )
    parser.add_argument(
        "--en-limit",
        type=int,
        default=None,
        help="Override English file limit (default: 100 for standard, unlimited for full)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bar",
    )
    parser.add_argument(
        "--ja-only",
        action="store_true",
        help="Process only Japanese (JSUT)",
    )
    parser.add_argument(
        "--en-only",
        action="store_true",
        help="Process only English (LibriSpeech)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check dependencies
    if not check_dependencies():
        return 1

    # Determine limits based on mode
    if args.mode == "standard":
        ja_limit = args.ja_limit if args.ja_limit is not None else 100
        en_limit = args.en_limit if args.en_limit is not None else 100
        jsut_subsets = JSUT_SUBSETS_STANDARD
    else:  # full mode
        ja_limit = args.ja_limit  # None means unlimited
        en_limit = args.en_limit
        jsut_subsets = JSUT_SUBSETS_FULL

    # Get source directories from environment or defaults
    jsut_dir = Path(os.getenv("LIVECAP_JSUT_DIR", str(DEFAULT_JSUT_DIR)))
    librispeech_dir = Path(os.getenv("LIVECAP_LIBRISPEECH_DIR", str(DEFAULT_LIBRISPEECH_DIR)))

    logger.info(f"Mode: {args.mode}")
    logger.info(f"JSUT source: {jsut_dir}")
    logger.info(f"LibriSpeech source: {librispeech_dir}")
    logger.info(f"Output: {PREPARED_DIR}")

    total_stats = ProcessingStats()

    # Process Japanese (JSUT)
    if not args.en_only:
        logger.info(f"\n=== Processing JSUT (limit: {ja_limit or 'unlimited'}) ===")
        ja_stats = process_jsut(
            jsut_dir=jsut_dir,
            output_dir=PREPARED_DIR / "ja",
            subsets=jsut_subsets,
            limit=ja_limit,
            force=args.force,
            quiet=args.quiet,
        )
        logger.info(f"JSUT: {ja_stats.summary()}")
        total_stats.processed += ja_stats.processed
        total_stats.skipped += ja_stats.skipped
        total_stats.failed += ja_stats.failed
        total_stats.errors.extend(ja_stats.errors)

    # Process English (LibriSpeech)
    if not args.ja_only:
        logger.info(f"\n=== Processing LibriSpeech (limit: {en_limit or 'unlimited'}) ===")
        en_stats = process_librispeech(
            librispeech_dir=librispeech_dir,
            output_dir=PREPARED_DIR / "en",
            limit=en_limit,
            force=args.force,
            quiet=args.quiet,
        )
        logger.info(f"LibriSpeech: {en_stats.summary()}")
        total_stats.processed += en_stats.processed
        total_stats.skipped += en_stats.skipped
        total_stats.failed += en_stats.failed
        total_stats.errors.extend(en_stats.errors)

    # Print summary
    logger.info(f"\n=== Summary ===")
    logger.info(f"Total: {total_stats.summary()}")

    if total_stats.errors:
        logger.warning(f"\nErrors ({len(total_stats.errors)}):")
        for err in total_stats.errors[:10]:  # Show first 10 errors
            logger.warning(f"  - {err}")
        if len(total_stats.errors) > 10:
            logger.warning(f"  ... and {len(total_stats.errors) - 10} more")

    if total_stats.processed > 0:
        logger.info(f"\nOutput written to: {PREPARED_DIR}")

    return 0 if total_stats.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
