"""Generate Layer 3 SNR-mixed noisy_speech corpus (Issue #338).

Loads clean speech entries and Layer 2 (ESC-50 / MUSAN) non_speech entries
from the calibration corpus manifest, mixes them at a grid of target SNR
values, and appends ``label=noisy_speech`` entries with additive
``snr_db`` / ``noise_source_dataset`` / ``noise_source_file`` /
``noise_source_path`` fields.

Used by Issue #334 PR-4's Pareto gate ``noisy_speech_frr by SNR ≤ 5%``:
per-SNR FRR characterization requires speech mixed with production-realistic
noise at controlled SNR levels.

Prerequisites (fail-fast per Plan D11):
  * ``{output_dir}/manifest.jsonl`` exists
  * At least ``--samples`` entries with ``label=speech`` and matching language
  * At least 1 entry with ``source_dataset in noise_datasets`` (Layer 2 output)

CLI usage::

    uv run python -m benchmarks.confidence_calibration.gen_mixed_noisy_speech \\
        --output-dir "$LIVECAP_CALIBRATION_CORPUS_DIR" \\
        --samples 50 \\
        --snr-db-list "-5,0,5,10,20" \\
        --noise-datasets esc50,musan

Design (Plan D1-D12):
  * SNR grid default ``[-5, 0, 5, 10, 20]`` dB (5 values)
  * uniform stride rotation via ``_uniform_stride_indices()`` (Plan D3 revised in
    Phase 2b per [Phase 2 report §5.7](../../docs/research/calibration-japan-engines-phase2-2026-07.md);
    pre-fix ``noise_pool[i % len(noise_pool)]`` combined with alphabetical path
    sort picked only the first N=n_samples entries, biasing selection to
    alphabetically-early ESC-50 categories (breathing + car_horn for the
    default 15-category set))
  * paired evaluation: same speech sample mixed at all SNR values (Plan D2)
  * ``source_dataset="layer3_mix"`` marker for safe re-augment via
    ``upsert_manifest_entries(..., source_dataset_filter="layer3_mix")``
  * ``--speech-language`` is the **single source of truth** for language:
    it selects clean speech, tags output manifest entries, and determines
    the output subdirectory (``{output_dir}/{speech_language}_noisy_speech/``).
    JA and EN can coexist in the same corpus without path/manifest
    collision.
  * No sweep.py changes (Plan D12 — Phase 6 concern)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .pipeline import resolve_corpus_dir
from ._augment_common import (
    SAMPLE_RATE,
    load_audio_16k_mono,
    positive_int,
    upsert_manifest_entries,
    write_chunk_wav,
)
from ._mix_snr import check_and_renorm, mix_at_snr

logger = logging.getLogger(__name__)

LAYER3_SOURCE_DATASET = "layer3_mix"
LAYER3_SOURCE_LICENSE_PREFIX = "derivative (clean speech + "

DEFAULT_SNR_DB_LIST: tuple[float, ...] = (-5.0, 0.0, 5.0, 10.0, 20.0)
DEFAULT_NOISE_DATASETS: tuple[str, ...] = ("esc50", "musan")


def output_subdir_for(speech_language: str) -> str:
    """Layer 3 output subdir を language 別に返す (multi-language path collision 防止)。

    ``--speech-language`` が manifest ``language`` field と output subdir の
    **単一 source of truth** となる設計。 JA と EN の両方を同一 corpus に
    augment しても path が衝突しない (``ja_noisy_speech/`` vs
    ``en_noisy_speech/``)。
    """
    return f"{speech_language}_noisy_speech"


def snr_list(value: str) -> list[float]:
    """argparse type: comma-separated float の list、 NaN/inf reject、 duplicate reject。

    Duplicate は 2 段階で reject:
    * raw value duplicate (例: ``"10,10"``): 同一 float 値が複数
    * formatted value duplicate (例: ``"3.54,3.5"``): ``format_snr_str`` 後
      の filename 部分が衝突 (round to 1 decimal 由来)

    これらを許容すると同一 filename が multiple 回生成され、 manifest upsert
    で last-wins になり user が期待した entry 数と実際に生成される entry 数
    がずれる (Plan D6 の filename encoding + upsert 挙動の相互作用)。
    """
    if not value or not value.strip():
        raise argparse.ArgumentTypeError("--snr-db-list must not be empty")
    parts = [p.strip() for p in value.split(",")]
    result: list[float] = []
    for p in parts:
        if not p:
            raise argparse.ArgumentTypeError(
                f"--snr-db-list has empty item in {value!r}"
            )
        try:
            f = float(p)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"--snr-db-list has non-numeric value {p!r} in {value!r}"
            )
        if not math.isfinite(f):
            raise argparse.ArgumentTypeError(
                f"--snr-db-list has non-finite value {p!r} (NaN/inf not allowed)"
            )
        result.append(f)

    # 1. raw value duplicate (e.g. "10,10")
    if len(result) != len(set(result)):
        seen: set[float] = set()
        dup = next(v for v in result if v in seen or seen.add(v))
        raise argparse.ArgumentTypeError(
            f"--snr-db-list has duplicate value {dup} in {value!r}"
        )

    # 2. formatted duplicate (e.g. "3.54,3.5" both round to "3.5")
    formatted = [format_snr_str(v) for v in result]
    if len(formatted) != len(set(formatted)):
        seen_fmt: set[str] = set()
        dup_fmt = next(f for f in formatted if f in seen_fmt or seen_fmt.add(f))
        # Find the original raw values that map to this formatted string
        collisions = [v for v, f in zip(result, formatted) if f == dup_fmt]
        raise argparse.ArgumentTypeError(
            f"--snr-db-list values {collisions} collide to same formatted "
            f"filename part 'snr{dup_fmt}dB' in {value!r} (filenames must be unique)"
        )

    return result


def dataset_list(value: str) -> list[str]:
    """argparse type: comma-separated dataset name list。"""
    if not value or not value.strip():
        raise argparse.ArgumentTypeError("--noise-datasets must not be empty")
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--noise-datasets must not be empty")
    return parts


def _load_manifest(manifest_path: Path) -> list[dict]:
    """Manifest.jsonl を list[dict] で読込 (順序保持、 filter 用)。

    ``build_corpus._load_manifest_entries`` は dict[path → entry] を返すが、
    こちらは filter + sort のため list 形式が便利。
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.jsonl not found: {manifest_path}\n"
            f"Prerequisite: run build_corpus.py to create speech entries and "
            f"gen_esc50_non_speech.py / gen_musan_noise.py for Layer 2 noise."
        )
    entries: list[dict] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def select_speech_samples(
    entries: list[dict],
    language: str,
    n_samples: int,
) -> list[dict]:
    """label=speech + language filter → filename sort → 先頭 N 件。"""
    filtered = [
        e for e in entries
        if e.get("label") == "speech" and e.get("language") == language
    ]
    filtered.sort(key=lambda e: e.get("path", ""))
    return filtered[:n_samples]


def select_noise_pool(
    entries: list[dict],
    datasets: list[str],
) -> list[dict]:
    """source_dataset in datasets の filter → filename sort。"""
    dataset_set = set(datasets)
    filtered = [
        e for e in entries
        if e.get("label") == "non_speech"
        and e.get("source_dataset") in dataset_set
    ]
    filtered.sort(key=lambda e: e.get("path", ""))
    return filtered


def check_prerequisites(
    entries: list[dict],
    language: str,
    n_samples: int,
    noise_datasets: list[str],
) -> tuple[list[dict], list[dict]]:
    """Prerequisite validation。 fail 時は explicit error message で loud fail。

    Returns:
        (speech_samples, noise_pool) tuple、 両方非空を保証。
    """
    speech = select_speech_samples(entries, language, n_samples)
    if len(speech) < n_samples:
        raise ValueError(
            f"Insufficient speech entries: got {len(speech)} with "
            f"label=speech + language={language!r}, need >= {n_samples}. "
            f"Prerequisite: run build_corpus.py to create more speech entries "
            f"or reduce --samples."
        )

    noise = select_noise_pool(entries, noise_datasets)
    if not noise:
        raise ValueError(
            f"No noise entries found with source_dataset in {noise_datasets!r}. "
            f"Prerequisite: run gen_esc50_non_speech.py / gen_musan_noise.py to "
            f"add Layer 2 non_speech entries before mixing Layer 3."
        )
    return speech, noise


def _uniform_stride_indices(pool_size: int, n_samples: int) -> list[int]:
    """Return N uniformly-strided indices from ``[0, pool_size - 1]``.

    Fixes Layer 3 noise diversity bug (Issue #338 Phase 2b、 Phase 2 report §5.4 /
    §5.7): ``noise_pool[i % len]`` combined with ``select_noise_pool``'s
    alphabetical path sort picked only the first ``n_samples`` entries,
    biasing selection to alphabetically-early subtypes. For ESC-50 default
    (``DEFAULT_CATEGORIES`` sorted → breathing → car_horn → ...), typical
    n_samples=50 selected 30 breathing + 20 car_horn from a 450+ pool,
    leaving 13 categories + MUSAN entries unused. Uniform stride via
    ``np.linspace`` spans the full pool evenly.

    Semantics:
      * ``n_samples <= pool_size`` (typical usage、 50 speech in 646 pool):
        indices span the full range ``[0, pool_size - 1]``、 approximately
        evenly spaced (spacing ≈ ``pool_size / n_samples``).
      * ``n_samples > pool_size`` (small-pool edge case): each pool index
        used floor(n/pool) or ceil(n/pool) times (max diff = 1)、 grouped
        consecutively (not interleaved).
      * ``n_samples == 1``: returns ``[0]``.
      * ``n_samples == 0``: returns ``[]``.

    Args:
        pool_size: number of noise entries in the pool (must be > 0)
        n_samples: number of speech samples to pair with noise (must be >= 0)

    Returns:
        List of ``n_samples`` int indices into ``[0, pool_size - 1]``.

    Raises:
        ValueError: if ``pool_size <= 0``.

    Examples:
        >>> _uniform_stride_indices(646, 50)[:5]   # Realistic Phase 2 case
        [0, 13, 26, 39, 53]
        >>> _uniform_stride_indices(646, 50)[-5:]
        [592, 606, 619, 632, 645]
        >>> _uniform_stride_indices(50, 50)[:5]    # identity when equal
        [0, 1, 2, 3, 4]
        >>> _uniform_stride_indices(2, 4)          # more samples than pool (divmod balanced)
        [0, 0, 1, 1]
        >>> _uniform_stride_indices(3, 8)          # counts {0:3, 1:3, 2:2} — max diff = 1
        [0, 0, 0, 1, 1, 1, 2, 2]
        >>> _uniform_stride_indices(4, 7)          # counts {0:2, 1:2, 2:2, 3:1} — max diff = 1
        [0, 0, 1, 1, 2, 2, 3]
        >>> _uniform_stride_indices(1, 3)
        [0, 0, 0]
    """
    if pool_size <= 0:
        raise ValueError(f"pool_size must be positive, got {pool_size}")
    if n_samples <= 0:
        return []
    if n_samples == 1:
        return [0]
    if n_samples > pool_size:
        # Balanced repeat via divmod (PR #350 codex-review fix): ``np.linspace``
        # + round does NOT guarantee max diff = 1 for n > pool (e.g. pool=3,
        # n=8 gives counts {0:2, 1:4, 2:2} with max diff = 2). Explicit divmod
        # ensures each index is used floor(n/pool) or ceil(n/pool) times,
        # with the first ``remainder`` indices getting the extra count.
        quotient, remainder = divmod(n_samples, pool_size)
        indices: list[int] = []
        for i in range(pool_size):
            count = quotient + (1 if i < remainder else 0)
            indices.extend([i] * count)
        return indices
    return np.linspace(0, pool_size - 1, n_samples).round().astype(int).tolist()


def format_snr_str(snr_db: float) -> str:
    """SNR を filename 用文字列に。 整数値は `10`、 非整数は `3.5` 等。"""
    if snr_db == int(snr_db):
        return str(int(snr_db))
    # non-integer: round to 1 decimal for filename cleanliness
    return f"{snr_db:.1f}"


def build_layer3_manifest_entry(
    *,
    relative_path: str,
    speech_entry: dict,
    noise_entry: dict,
    snr_db: float,
    duration_sec: float,
    language: str,
) -> dict:
    """Layer 3 manifest entry を Phase 1 + Phase 2 additive field 付きで構築。

    Note: alignment_score / _kana は placeholder (0.0)、 Phase 6 で
    recompute_alignment 相当の CLI (別 PR) で更新想定。 reference_text は
    speech 由来をそのまま継承 (mixed audio の "正解 text" として使う)。
    """
    noise_ds = noise_entry.get("source_dataset", "unknown")
    return {
        "path": relative_path,
        "label": "noisy_speech",
        "language": language,
        "noise": None,
        "subtype": noise_entry.get("subtype"),
        "reference_text_matched": speech_entry.get("reference_text_matched"),
        "transcribed_text": "",
        "alignment_score": 0.0,
        "alignment_score_kana": 0.0,
        "reference_text_matched_kana": speech_entry.get("reference_text_matched_kana"),
        "transcribed_text_kana": "",
        "engine_used": "n/a (mixed, will be filled at sweep time)",
        "start_sec": 0.0,
        "end_sec": round(duration_sec, 3),
        "duration_sec": round(duration_sec, 3),
        # Phase 2 attribution (source_dataset for filter, source_file for attribution)
        "source_dataset": LAYER3_SOURCE_DATASET,
        "source_file": Path(speech_entry["path"]).name,
        "source_license": f"{LAYER3_SOURCE_LICENSE_PREFIX}{noise_ds})",
        # Layer 3 additive
        "snr_db": snr_db,
        "noise_source_dataset": noise_ds,
        "noise_source_file": noise_entry.get("source_file", ""),
        "noise_source_path": noise_entry["path"],
    }


def augment(
    *,
    output_dir: Path,
    speech_language: str,
    noise_datasets: list[str],
    snr_db_list: list[float],
    n_samples: int,
    force: bool = False,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """Layer 3 augment 本体。 return ``(added, updated, removed)`` manifest counts。

    Output manifest entry の ``language`` field は ``speech_language`` を継承
    (codex-review 対応: mixed noisy_speech の language は clean speech と一致
    するのが自然、 別引数だと EN speech なのに ``language="ja"`` 等の不整合
    を招き ``sweep.py --filter-by-language`` を汚染する)。
    """
    manifest_path = output_dir / "manifest.jsonl"
    entries = _load_manifest(manifest_path)
    logger.info(
        "Loaded manifest %s: %d total entries", manifest_path, len(entries)
    )

    speech_samples, noise_pool = check_prerequisites(
        entries, speech_language, n_samples, noise_datasets
    )
    logger.info(
        "Prerequisites OK: %d speech samples, %d noise entries in pool",
        len(speech_samples),
        len(noise_pool),
    )

    output_subdir = output_subdir_for(speech_language)
    output_wav_dir = output_dir / output_subdir
    new_entries: list[dict] = []
    clip_count = 0

    # Uniform stride rotation (Phase 2b fix per Phase 2 report §5.7): spans
    # the full noise pool evenly instead of picking the first N sorted
    # entries. See ``_uniform_stride_indices`` docstring for semantics.
    noise_indices = _uniform_stride_indices(len(noise_pool), len(speech_samples))

    for i, speech_entry in enumerate(speech_samples):
        noise_entry = noise_pool[noise_indices[i]]
        speech_path = output_dir / speech_entry["path"]
        noise_path = output_dir / noise_entry["path"]
        if not speech_path.exists():
            logger.warning("Speech audio missing (skip): %s", speech_path)
            continue
        if not noise_path.exists():
            logger.warning("Noise audio missing (skip): %s", noise_path)
            continue

        speech_audio = load_audio_16k_mono(speech_path)
        noise_audio = load_audio_16k_mono(noise_path)

        for snr_db in snr_db_list:
            mixed = mix_at_snr(speech_audio, noise_audio, snr_db)
            mixed, was_clipped = check_and_renorm(mixed)
            if was_clipped:
                clip_count += 1

            speech_stem = Path(speech_entry["path"]).stem
            noise_subtype = noise_entry.get("subtype", "unknown")
            snr_str = format_snr_str(snr_db)
            output_name = f"{speech_stem}_snr{snr_str}dB_{noise_subtype}.wav"
            output_wav_path = output_wav_dir / output_name
            relative_path = f"{output_subdir}/{output_name}"

            if not dry_run:
                write_chunk_wav(mixed, output_wav_path, sample_rate=SAMPLE_RATE)

            entry = build_layer3_manifest_entry(
                relative_path=relative_path,
                speech_entry=speech_entry,
                noise_entry=noise_entry,
                snr_db=snr_db,
                duration_sec=len(mixed) / SAMPLE_RATE,
                language=speech_language,
            )
            new_entries.append(entry)

    if clip_count > 0:
        logger.warning(
            "%d/%d mixed samples were renormalized due to clipping "
            "(SNR ratio preserved, peak lowered to 0.95)",
            clip_count,
            len(new_entries),
        )

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
        source_dataset_filter=LAYER3_SOURCE_DATASET if force else None,
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
        prog="benchmarks.confidence_calibration.gen_mixed_noisy_speech",
        description=(
            "Generate Layer 3 SNR-mixed noisy_speech corpus for Issue #334 PR-4 "
            "Pareto gate `noisy_speech_frr by SNR ≤ 5%`."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Calibration corpus root (must contain manifest.jsonl with clean "
            "speech and Layer 2 non_speech entries). Mixed wavs go under "
            "{output_dir}/{speech_language}_noisy_speech/ (e.g. "
            "ja_noisy_speech/ for --speech-language ja). "
            "Default: $LIVECAP_CALIBRATION_CORPUS_DIR、 未 set なら OS 標準 data "
            "dir (`user_data_dir('LiveCap', 'PineLab') / calibration_corpus`)。"
        ),
    )
    parser.add_argument(
        "--speech-language",
        type=str,
        default="ja",
        help=(
            "Filter clean speech entries by language + tag output manifest "
            "entries with the same language + place output wavs under "
            "{output_dir}/{value}_noisy_speech/. Single source of truth "
            "for language (there is no separate --language flag). Default 'ja'."
        ),
    )
    parser.add_argument(
        "--noise-datasets",
        type=dataset_list,
        default=list(DEFAULT_NOISE_DATASETS),
        help=(
            "Comma-separated noise source_dataset filter. "
            f"Default '{','.join(DEFAULT_NOISE_DATASETS)}'."
        ),
    )
    parser.add_argument(
        "--snr-db-list",
        type=snr_list,
        default=list(DEFAULT_SNR_DB_LIST),
        help=(
            "Comma-separated SNR values in dB. "
            f"Default '{','.join(str(int(s)) if s == int(s) else str(s) for s in DEFAULT_SNR_DB_LIST)}'."
        ),
    )
    parser.add_argument(
        "--samples",
        type=positive_int,
        default=50,
        help=(
            "Number of clean speech samples to mix (each mixed at all SNR values). "
            "Must be >= 1. Default 50."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            f"Remove all existing '{LAYER3_SOURCE_DATASET}' entries from the "
            "manifest before upsert (safe re-augment)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview entry count without writing wavs or updating manifest.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

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
        output_dir=output_dir,
        speech_language=args.speech_language,
        noise_datasets=args.noise_datasets,
        snr_db_list=args.snr_db_list,
        n_samples=args.samples,
        force=args.force,
        dry_run=args.dry_run,
    )

    total = len(args.snr_db_list) * args.samples
    print(
        f"Layer 3 augment done: added={added}, updated={updated}, "
        f"removed={removed} (expected {total} = {args.samples} speech × "
        f"{len(args.snr_db_list)} SNR)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
