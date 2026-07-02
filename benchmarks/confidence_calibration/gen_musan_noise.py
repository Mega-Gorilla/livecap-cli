"""Augment the calibration corpus with MUSAN noise samples (Issue #338 Phase 2).

Loads MUSAN's ``noise/`` subset (skipping ``music/`` and ``speech/``), chunks each
variable-length file into 1.5-sec sub-clips, and appends ``non_speech`` entries.

Dataset (dev-only, raw audio never committed to git):
    URL: https://www.openslr.org/resources/17/musan.tar.gz  (~11 GB total)
    License: CC BY 4.0
    Layout: musan/noise/free-sound/*.wav + musan/noise/sound-bible/*.wav

MUSAN's ``music/`` subset is intentionally excluded (BGM vs speech is a
different problem), and ``speech/`` is excluded because those would be
false positives for confidence calibration on non_speech samples.

CLI usage::

    uv run python -m benchmarks.confidence_calibration.gen_musan_noise \\
        --source-dir .tmp/musan_source/musan \\
        --output-dir .tmp/calibration_corpus_full \\
        --samples 50 \\
        --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from ._augment_common import (
    build_non_speech_manifest_entry,
    chunk_audio,
    download_dataset,
    load_audio_16k_mono,
    positive_int,
    safe_extract_tar,
    upsert_manifest_entries,
    write_chunk_wav,
)
from .pipeline import resolve_corpus_dir

logger = logging.getLogger(__name__)

MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"
MUSAN_LICENSE = "CC BY 4.0"
MUSAN_SOURCE_DATASET = "musan"

# Subdirectories under musan/noise/ (skip music/, speech/)
NOISE_SUBDIRS: tuple[str, ...] = ("free-sound", "sound-bible")

# Output subdirectory under corpus root
OUTPUT_SUBDIR = "ja_non_speech_musan"


def _collect_noise_files(source_dir: Path) -> list[Path]:
    """Return sorted list of .wav files under noise/free-sound/ and noise/sound-bible/.

    Accepts either ``musan/`` (full corpus root) or ``musan/noise/`` as source_dir.
    """
    noise_dir: Optional[Path] = None
    if (source_dir / "noise").is_dir():
        noise_dir = source_dir / "noise"
    elif source_dir.name == "noise" or (source_dir / "free-sound").is_dir():
        noise_dir = source_dir
    else:
        # Sometimes users point at extracted archive root
        inner = source_dir / "musan"
        if (inner / "noise").is_dir():
            noise_dir = inner / "noise"

    if noise_dir is None or not noise_dir.is_dir():
        raise FileNotFoundError(
            f"MUSAN noise/ directory not found under {source_dir}\n"
            f"Expected one of:\n"
            f"  {source_dir}/noise/free-sound/*.wav\n"
            f"  {source_dir}/free-sound/*.wav (source_dir already noise/)\n"
            f"  {source_dir}/musan/noise/free-sound/*.wav"
        )

    files: list[Path] = []
    for subdir in NOISE_SUBDIRS:
        subdir_path = noise_dir / subdir
        if not subdir_path.is_dir():
            logger.warning("MUSAN sub-dir missing (skip): %s", subdir_path)
            continue
        files.extend(sorted(subdir_path.glob("*.wav")))
    if not files:
        raise FileNotFoundError(
            f"No .wav files found under {noise_dir}/{{{','.join(NOISE_SUBDIRS)}}}/"
        )
    return files


def _select_files(files: list[Path], n_samples: int) -> list[Path]:
    """Deterministic selection: uniform stride sampling from sorted list."""
    if n_samples >= len(files):
        return files
    if n_samples <= 1:
        return files[:1]
    stride_positions = [
        int(round(i * (len(files) - 1) / (n_samples - 1)))
        for i in range(n_samples)
    ]
    # De-duplicate while preserving order (stride can hit same index for tiny lists)
    seen: set[int] = set()
    selected: list[Path] = []
    for pos in stride_positions:
        if pos not in seen:
            seen.add(pos)
            selected.append(files[pos])
    return selected


def _download_and_extract_musan(dest_root: Path) -> Path:
    """MUSAN tar.gz を download して展開 (WARNING: ~11 GB)、 musan/ ディレクトリを返す。"""
    tar_path = dest_root / "musan.tar.gz"
    dest_root.mkdir(parents=True, exist_ok=True)
    logger.warning(
        "MUSAN download start (~11 GB, dev-only, raw audio not committed to git). "
        "Consider using --source-dir if you already have a local copy."
    )
    download_dataset(MUSAN_URL, tar_path)
    extracted = dest_root / "musan"
    if not extracted.exists():
        logger.info("Extracting %s -> %s (path-traversal guarded) ...", tar_path, dest_root)
        safe_extract_tar(tar_path, dest_root)
    return extracted


def _subtype_from_source_path(path: Path) -> str:
    """MUSAN 内 sub-dir 名を subtype として使う (free-sound / sound-bible)。"""
    parent = path.parent.name
    return f"musan_{parent}"


def augment(
    source_dir: Path,
    output_dir: Path,
    n_samples: int = 50,
    max_chunks_per_file: int = 5,
    language: str = "ja",
    force: bool = False,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """MUSAN augment 本体。 return (added, updated, removed) manifest counts。"""
    files = _collect_noise_files(source_dir)
    logger.info("MUSAN noise files found: %d", len(files))

    selected = _select_files(files, n_samples)
    logger.info("Selected %d files (uniform stride from %d total)", len(selected), len(files))

    output_wav_dir = output_dir / OUTPUT_SUBDIR
    manifest_path = output_dir / "manifest.jsonl"

    new_entries: list[dict] = []
    for source_wav in selected:
        audio = load_audio_16k_mono(source_wav)
        chunks = chunk_audio(
            audio,
            chunk_duration_sec=1.5,
            max_chunks_per_file=max_chunks_per_file,
        )
        stem = source_wav.stem
        subtype = _subtype_from_source_path(source_wav)
        for chunk_idx, chunk in enumerate(chunks):
            duration = len(chunk) / 16000.0
            output_name = f"{stem}_chunk{chunk_idx}.wav"
            output_wav_path = output_wav_dir / output_name
            relative_path = f"{OUTPUT_SUBDIR}/{output_name}"

            if not dry_run:
                write_chunk_wav(chunk, output_wav_path)

            entry = build_non_speech_manifest_entry(
                relative_path=relative_path,
                duration_sec=duration,
                subtype=subtype,
                source_dataset=MUSAN_SOURCE_DATASET,
                source_file=source_wav.name,
                source_license=MUSAN_LICENSE,
                language=language,
            )
            new_entries.append(entry)

    if dry_run:
        logger.info(
            "DRY RUN: would upsert %d entries to %s (force=%s)",
            len(new_entries),
            manifest_path,
            force,
        )
        return (len(new_entries), 0, 0)

    added, updated, removed = upsert_manifest_entries(
        manifest_path,
        new_entries,
        force=force,
        source_dataset_filter=MUSAN_SOURCE_DATASET if force else None,
    )
    logger.info(
        "Manifest upsert complete: +%d added, %d updated, %d removed (force=%s)",
        added,
        updated,
        removed,
        force,
    )
    return added, updated, removed


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmarks.confidence_calibration.gen_musan_noise",
        description=(
            "Augment calibration corpus with MUSAN noise samples "
            "(Issue #338 Phase 2, CC BY 4.0 dev-only). music/ and speech/ excluded."
        ),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help=(
            "Path to already-downloaded MUSAN (either musan/, musan/noise/, or a parent). "
            "Required unless --download is set."
        ),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help=(
            "Auto-download musan.tar.gz (~11 GB) into --source-dir "
            "(default .tmp/musan_source/). Raw audio is NOT committed to git."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Calibration corpus root (containing manifest.jsonl). "
            "Augmented wavs go under {output_dir}/ja_non_speech_musan/. "
            "Default: $LIVECAP_CALIBRATION_CORPUS_DIR、 未 set なら OS 標準 data "
            "dir (`user_data_dir('LiveCap', 'PineLab') / calibration_corpus`)。"
        ),
    )
    parser.add_argument(
        "--samples",
        type=positive_int,
        default=50,
        help=(
            "Total noise files to include (deterministic uniform stride). "
            "Must be >= 1. Default 50."
        ),
    )
    parser.add_argument(
        "--max-chunks-per-file",
        type=positive_int,
        default=5,
        help="Max 1.5-sec chunks to extract per source file. Must be >= 1. Default 5.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ja",
        help="Language tag for the manifest entries. Default 'ja'.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove all existing musan entries from the manifest before upsert (safe re-augment).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview counts without writing wavs or manifest.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.source_dir and not args.download:
        parser.error("Must specify --source-dir or --download")

    source_dir = args.source_dir or Path(".tmp/musan_source")
    if args.download:
        _download_and_extract_musan(source_dir)

    # Resolve output_dir: --output-dir → env var → OS 標準 data dir default
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = resolve_corpus_dir()
        logger.info(
            "--output-dir not specified, using %s (set LIVECAP_CALIBRATION_CORPUS_DIR "
            "to override)",
            output_dir,
        )

    added, updated, removed = augment(
        source_dir=source_dir,
        output_dir=output_dir,
        n_samples=args.samples,
        max_chunks_per_file=args.max_chunks_per_file,
        language=args.language,
        force=args.force,
        dry_run=args.dry_run,
    )

    print(f"MUSAN augment done: added={added}, updated={updated}, removed={removed}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
