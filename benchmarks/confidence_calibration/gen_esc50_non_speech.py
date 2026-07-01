"""Augment the calibration corpus with ESC-50 environmental sounds (Issue #338 Phase 2).

Loads a subset of ESC-50 categories (default 15 production-realistic
non_speech sources), chunks each 5-sec clip into 1.5-sec sub-clips, and
appends ``non_speech`` entries to the manifest with attribution fields.

Dataset (dev-only, raw audio never committed to git):
    URL: https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip
    License: CC BY-NC 4.0 (Non-Commercial)
    Layout: ESC-50-master/audio/<filename>.wav + ESC-50-master/meta/esc50.csv

CLI usage::

    uv run python -m benchmarks.confidence_calibration.gen_esc50_non_speech \\
        --source-dir .tmp/esc50_source/ESC-50-master \\
        --output-dir .tmp/calibration_corpus_full \\
        --samples-per-category 10 \\
        --force
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import zipfile
from pathlib import Path
from typing import Optional

from ._augment_common import (
    build_non_speech_manifest_entry,
    chunk_audio,
    download_dataset,
    load_audio_16k_mono,
    upsert_manifest_entries,
    write_chunk_wav,
)

logger = logging.getLogger(__name__)

ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
ESC50_LICENSE = "CC BY-NC 4.0"
ESC50_SOURCE_DATASET = "esc50"

# Plan D2 (production-realistic non_speech sources for livecap-gui). ESC-50 の
# 50 category 中、mic に混入しうる 15 category に限定。 音楽 (BGM 等) は含めない
# (speech と混在した場合の判断が別問題)。
DEFAULT_CATEGORIES: tuple[str, ...] = (
    # Human non-speech (5)
    "laughing",
    "sneezing",
    "coughing",
    "breathing",
    "clapping",
    "footsteps",
    # Natural (1)
    "rain",
    # Interior (5)
    "door_wood_knock",
    "mouse_click",
    "keyboard_typing",
    "clock_tick",
    "glass_breaking",
    # Exterior (3)
    "engine",
    "car_horn",
    "siren",
)

# Output subdirectory under corpus root
OUTPUT_SUBDIR = "ja_non_speech_esc50"


def _parse_esc50_csv(meta_path: Path) -> list[dict[str, str]]:
    """meta/esc50.csv を読んで dict list に変換。

    csv columns: filename, fold, target, category, esc10, src_file, take
    """
    if not meta_path.exists():
        raise FileNotFoundError(
            f"ESC-50 meta CSV not found: {meta_path}\n"
            f"Expected inside --source-dir at meta/esc50.csv"
        )
    rows: list[dict[str, str]] = []
    with meta_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _select_files_by_category(
    rows: list[dict[str, str]],
    categories: list[str],
    samples_per_category: int,
) -> list[dict[str, str]]:
    """Category filter + deterministic (先頭 N 件) selection。"""
    valid_categories = {row["category"] for row in rows}
    unknown = [c for c in categories if c not in valid_categories]
    if unknown:
        raise ValueError(
            f"Unknown ESC-50 categories: {unknown}\n"
            f"Available: {sorted(valid_categories)}"
        )
    selected: list[dict[str, str]] = []
    for cat in categories:
        cat_rows = [r for r in rows if r["category"] == cat]
        # Sort by filename for determinism (rows may already be sorted, but be defensive)
        cat_rows.sort(key=lambda r: r["filename"])
        selected.extend(cat_rows[:samples_per_category])
    return selected


def _resolve_source_dir(source_dir: Path) -> Path:
    """--source-dir が ``ESC-50-master`` 直下でない場合の path 補正。

    ZIP を展開すると通常 ``.tmp/esc50_source/ESC-50-master/audio/...`` になる。
    ``--source-dir`` が親 dir を指した場合も対応。
    """
    # Case 1: source_dir 直下に audio/ と meta/ がある = 正しい path
    if (source_dir / "audio").is_dir() and (source_dir / "meta").is_dir():
        return source_dir
    # Case 2: source_dir/ESC-50-master/audio 等 (ZIP 展開直後の形)
    inner = source_dir / "ESC-50-master"
    if (inner / "audio").is_dir() and (inner / "meta").is_dir():
        return inner
    raise FileNotFoundError(
        f"ESC-50 audio/ and meta/ directories not found under {source_dir}\n"
        f"Expected either:\n"
        f"  {source_dir}/audio/ + {source_dir}/meta/\n"
        f"  {source_dir}/ESC-50-master/audio/ + {source_dir}/ESC-50-master/meta/"
    )


def _download_and_extract_esc50(dest_root: Path) -> Path:
    """ESC-50 ZIP を download して展開、 ESC-50-master ディレクトリを返す。"""
    zip_path = dest_root / "ESC-50-master.zip"
    dest_root.mkdir(parents=True, exist_ok=True)
    logger.info("ESC-50 download start (~600 MB, dev-only, raw audio not committed to git)")
    download_dataset(ESC50_URL, zip_path)
    extracted = dest_root / "ESC-50-master"
    if not extracted.exists():
        logger.info("Extracting %s -> %s", zip_path, dest_root)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_root)
    return extracted


def augment(
    source_dir: Path,
    output_dir: Path,
    categories: list[str],
    samples_per_category: int,
    language: str = "ja",
    force: bool = False,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """ESC-50 augment 本体。 return (added, updated, removed) manifest counts。"""
    resolved_source = _resolve_source_dir(source_dir)
    meta_csv = resolved_source / "meta" / "esc50.csv"
    audio_root = resolved_source / "audio"

    rows = _parse_esc50_csv(meta_csv)
    logger.info(
        "ESC-50 meta parsed: %d total files across %d categories",
        len(rows),
        len({r["category"] for r in rows}),
    )

    selected = _select_files_by_category(rows, categories, samples_per_category)
    logger.info(
        "Selected %d files across %d categories (samples_per_category=%d)",
        len(selected),
        len(categories),
        samples_per_category,
    )

    output_wav_dir = output_dir / OUTPUT_SUBDIR
    manifest_path = output_dir / "manifest.jsonl"

    new_entries: list[dict] = []
    for row in selected:
        original_filename = row["filename"]
        category = row["category"]
        source_wav = audio_root / original_filename
        if not source_wav.exists():
            logger.warning("Audio file missing (skip): %s", source_wav)
            continue

        audio = load_audio_16k_mono(source_wav)
        chunks = chunk_audio(audio, chunk_duration_sec=1.5, max_chunks_per_file=3)
        stem = Path(original_filename).stem  # e.g. "1-100032-A-22"
        for chunk_idx, chunk in enumerate(chunks):
            duration = len(chunk) / 16000.0
            output_name = f"{category}_{stem}_chunk{chunk_idx}.wav"
            output_wav_path = output_wav_dir / output_name
            relative_path = f"{OUTPUT_SUBDIR}/{output_name}"

            if not dry_run:
                write_chunk_wav(chunk, output_wav_path)

            entry = build_non_speech_manifest_entry(
                relative_path=relative_path,
                duration_sec=duration,
                subtype=category,
                source_dataset=ESC50_SOURCE_DATASET,
                source_file=original_filename,
                source_license=ESC50_LICENSE,
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
        source_dataset_filter=ESC50_SOURCE_DATASET if force else None,
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
        prog="benchmarks.confidence_calibration.gen_esc50_non_speech",
        description=(
            "Augment calibration corpus with ESC-50 environmental sounds "
            "(Issue #338 Phase 2, CC BY-NC 4.0 dev-only)."
        ),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help=(
            "Path to already-downloaded ESC-50 (either ESC-50-master/ or its parent). "
            "Required unless --download is set."
        ),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help=(
            "Auto-download ESC-50-master.zip (~600 MB) into --source-dir "
            "(default .tmp/esc50_source/). Raw audio is NOT committed to git."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Calibration corpus root (containing manifest.jsonl). "
            "Augmented wavs go under {output_dir}/ja_non_speech_esc50/."
        ),
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help=(
            "Comma-separated ESC-50 category names. Default = 15 production-realistic "
            f"categories: {','.join(DEFAULT_CATEGORIES)}"
        ),
    )
    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=10,
        help="How many files to sample per category (deterministic, sorted by filename). Default 10.",
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
        help="Remove all existing esc50 entries from the manifest before upsert (safe re-augment).",
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

    source_dir = args.source_dir or Path(".tmp/esc50_source")
    if args.download:
        _download_and_extract_esc50(source_dir)

    categories = (
        [c.strip() for c in args.categories.split(",")]
        if args.categories
        else list(DEFAULT_CATEGORIES)
    )

    added, updated, removed = augment(
        source_dir=source_dir,
        output_dir=args.output_dir,
        categories=categories,
        samples_per_category=args.samples_per_category,
        language=args.language,
        force=args.force,
        dry_run=args.dry_run,
    )

    print(f"ESC-50 augment done: added={added}, updated={updated}, removed={removed}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
