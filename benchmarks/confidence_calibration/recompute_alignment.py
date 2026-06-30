"""Re-compute kana-level alignment for existing manifest entries (Issue #338 PR-γ).

This CLI migrates existing ``manifest.jsonl`` produced by ``build_corpus`` to
include the kana-level alignment metric, **without re-transcribing audio**.

For each entry that has a non-empty ``transcribed_text`` and a ``language``
field matching one of the provided reference texts, this tool:

1. Loads the language-specific reference text (URL or local path),
2. Computes ``compute_alignment_score_kana(transcribed_text, reference_text)``,
3. Adds three new fields to the entry:

   - ``alignment_score_kana`` (float, 0.0–1.0)
   - ``reference_text_matched_kana`` (Optional[str])
   - ``transcribed_text_kana`` (str)

4. Writes the updated manifest back using the same upsert/rewrite pattern as
   ``build_corpus`` (no append duplication, other-source entries preserved).

By default, entries that already have ``alignment_score_kana`` are skipped.
Use ``--force`` to recompute regardless.

Usage::

    uv run python -m benchmarks.confidence_calibration.recompute_alignment \\
        --manifest "$LIVECAP_CALIBRATION_CORPUS_DIR/manifest.jsonl" \\
        --reference-text-ja "https://example.com/ja-chapter1" \\
        --reference-text-en "https://example.com/en-chapter1"

This is **dev/benchmark-only** — see ``_normalize_jp.py`` license caveat
(pykakasi is GPL-3.0-or-later, dev dependency).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .build_corpus import (
    _load_manifest_entries,
    _write_manifest,
    compute_alignment_score_kana,
    fetch_reference_text,
)

logger = logging.getLogger(__name__)


def recompute_kana_alignment(
    *,
    manifest_path: Path,
    references: dict[str, str],
    force: bool = False,
) -> dict[str, int]:
    """Re-compute kana alignment for matching entries; rewrite manifest in place.

    Args:
        manifest_path: Path to manifest.jsonl.
        references: ``{language: reference_text}``. Only entries whose
            ``language`` field is a key of this dict will be processed.
        force: If True, recompute even when ``alignment_score_kana`` is already
            present.

    Returns:
        Counters: ``{"updated": int, "skipped_already_done": int,
        "skipped_no_language_match": int, "skipped_no_transcribed_text": int,
        "total": int}``.
    """
    entries = _load_manifest_entries(manifest_path)
    counters = {
        "updated": 0,
        "skipped_already_done": 0,
        "skipped_no_language_match": 0,
        "skipped_no_transcribed_text": 0,
        "total": len(entries),
    }

    for path, entry in entries.items():
        lang = entry.get("language")
        if lang not in references:
            counters["skipped_no_language_match"] += 1
            continue
        transcribed = entry.get("transcribed_text", "") or ""
        if not transcribed.strip():
            counters["skipped_no_transcribed_text"] += 1
            continue
        if not force and "alignment_score_kana" in entry:
            counters["skipped_already_done"] += 1
            continue

        (
            score_kana,
            matched_kana,
            transcribed_kana,
            _matched_dup,
        ) = compute_alignment_score_kana(transcribed, references[lang])
        entry["alignment_score_kana"] = round(score_kana, 4)
        entry["reference_text_matched_kana"] = matched_kana
        entry["transcribed_text_kana"] = transcribed_kana
        counters["updated"] += 1

    _write_manifest(manifest_path, list(entries.values()))
    return counters


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="recompute_alignment",
        description=(
            "Re-compute kana-level alignment for existing manifest.jsonl "
            "entries (Issue #338 PR-gamma). Does not re-transcribe audio -- "
            "only re-runs alignment calculation on already-stored "
            "transcribed_text."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to manifest.jsonl produced by build_corpus.",
    )
    parser.add_argument(
        "--reference-text-ja",
        type=str,
        default=None,
        help=(
            "Reference text source for JA entries (URL or local file path). "
            "If omitted, JA entries are skipped (no_language_match)."
        ),
    )
    parser.add_argument(
        "--reference-text-en",
        type=str,
        default=None,
        help=(
            "Reference text source for EN entries (URL or local file path). "
            "If omitted, EN entries are skipped (no_language_match)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Recompute kana fields even when alignment_score_kana is already "
            "present (default: skip already-processed entries)."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.manifest.exists():
        logger.error("Manifest does not exist: %s", args.manifest)
        return 2

    references: dict[str, str] = {}
    if args.reference_text_ja:
        logger.info("Fetching JA reference: %s", args.reference_text_ja)
        references["ja"] = fetch_reference_text(args.reference_text_ja)
        logger.info("JA reference: %d chars", len(references["ja"]))
    if args.reference_text_en:
        logger.info("Fetching EN reference: %s", args.reference_text_en)
        references["en"] = fetch_reference_text(args.reference_text_en)
        logger.info("EN reference: %d chars", len(references["en"]))

    if not references:
        logger.error(
            "No reference text provided. Use --reference-text-ja and/or "
            "--reference-text-en."
        )
        return 2

    counters = recompute_kana_alignment(
        manifest_path=args.manifest,
        references=references,
        force=args.force,
    )

    print(f"Total entries:                {counters['total']}")
    print(f"Updated (kana added):         {counters['updated']}")
    print(f"Skipped (already had kana):   {counters['skipped_already_done']}")
    print(f"Skipped (no language match):  {counters['skipped_no_language_match']}")
    print(f"Skipped (no transcribed_text): {counters['skipped_no_transcribed_text']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
