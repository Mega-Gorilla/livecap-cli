"""Shared helpers for ESC-50 / MUSAN augmentation CLIs (Issue #338 Phase 2).

Both ``gen_esc50_non_speech.py`` and ``gen_musan_noise.py`` produce augmented
``non_speech`` entries for the calibration corpus. This module factors out:

* 16 kHz mono resampling (reuses ``pipeline._resample_to_16k_mono``)
* Deterministic fixed-window chunking (1-2 sec sub-clips from longer source)
* Manifest entry construction with additive Phase 2 attribution fields
  (``source_dataset`` / ``source_file`` / ``source_license``)
* Upsert-based manifest write (reuses ``build_corpus._load_manifest_entries``
  and ``_write_manifest``) for idempotent re-runs
* Optional dataset download to ``.tmp/`` (raw audio never committed to git)

The manifest entries produced here follow the Phase 1 schema (see
``pipeline.py`` module docstring): ``label="non_speech"``,
``subtype=<category>``, and ``transcribed_text=""`` (transcribe happens at
sweep time, not at augment time — see Plan D6).
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class AugmentChunk:
    """1 chunked non_speech sample, ready for manifest write."""

    audio: np.ndarray  # 16 kHz mono float32
    duration_sec: float
    source_file: str  # basename of the original dataset file
    subtype: str  # e.g. "applause" (ESC-50 category) or "hvac" (MUSAN)


def load_audio_16k_mono(path: Path) -> np.ndarray:
    """soundfile で load し ``_resample_to_16k_mono`` で 16 kHz mono float32 に正規化。"""
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - environment dep
        raise ImportError("soundfile is required for augmentation") from exc

    from .pipeline import _resample_to_16k_mono

    audio, sr = sf.read(str(path))
    return _resample_to_16k_mono(audio, int(sr))


def chunk_audio(
    audio: np.ndarray,
    chunk_duration_sec: float = 1.5,
    max_chunks_per_file: int = 3,
    sample_rate: int = SAMPLE_RATE,
) -> list[np.ndarray]:
    """Fixed-window chunker with deterministic uniform-stride positions.

    Rationale (Plan D3): ESC-50 は 5 sec fixed → 3 × 1.5 sec が自然。
    MUSAN noise は 5-60 sec 可変 → 均等分割で最大 ``max_chunks_per_file`` 個。

    Determinism: 位置は audio 長さから計算、random seed なし。
    再現性重視 (Plan D2)。

    Args:
        audio: 1D float32 numpy array, 16 kHz mono.
        chunk_duration_sec: 各 chunk の長さ (default 1.5 sec, Phase 1 synthetic の中央値).
        max_chunks_per_file: 1 file から取れる chunk 数の上限。
        sample_rate: audio の sample rate (default 16 kHz)。

    Returns:
        chunk numpy array の list。 audio が chunk_duration 未満なら [audio] (原音そのまま,
        0-pad しない = signal 特性を保持)。 audio が空なら []。
    """
    total_samples = len(audio)
    chunk_samples = int(chunk_duration_sec * sample_rate)
    if total_samples == 0:
        return []
    if total_samples < chunk_samples:
        # Too short to fill even one chunk — return the whole thing (0-pad NOT applied)
        return [audio.astype(np.float32, copy=False)]

    max_start = total_samples - chunk_samples
    if max_start == 0:
        return [audio[:chunk_samples].astype(np.float32, copy=False)]

    # Maximum non-overlapping chunks that fit; capped by max_chunks_per_file
    n_chunks = min(max_chunks_per_file, max(1, total_samples // chunk_samples))

    if n_chunks == 1:
        starts = [max_start // 2]
    else:
        # Uniform stride from 0 to max_start (inclusive at both ends)
        starts = [int(round(i * max_start / (n_chunks - 1))) for i in range(n_chunks)]

    return [audio[s : s + chunk_samples].astype(np.float32, copy=False) for s in starts]


def write_chunk_wav(
    chunk: np.ndarray,
    output_path: Path,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    """Write 16 kHz mono chunk to WAV (soundfile default subtype, matches build_corpus.write_wav)."""
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - environment dep
        raise ImportError("soundfile is required for augmentation") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), chunk, sample_rate)


def build_non_speech_manifest_entry(
    relative_path: str,
    duration_sec: float,
    subtype: str,
    source_dataset: str,
    source_file: str,
    source_license: str,
    language: str = "ja",
) -> dict:
    """Build a manifest entry dict for a Phase 2 augmented non_speech sample.

    Fields follow Phase 1 ``build_corpus`` schema (14 field) + additive
    Phase 2 attribution fields (``source_dataset`` / ``source_file`` /
    ``source_license``). The kana field values are placeholder (``0.0`` /
    ``""``) since ``label==non_speech`` has no reference text to align against.

    Note: ``transcribed_text`` is empty; sweep.py transcribes fresh at run
    time from ``item.audio``, so this field is never consumed downstream
    for non_speech samples (Plan D6).
    """
    return {
        "path": relative_path,
        "label": "non_speech",
        "language": language,
        "noise": None,
        "subtype": subtype,
        "reference_text_matched": None,
        "transcribed_text": "",
        "alignment_score": 0.0,
        "alignment_score_kana": 0.0,
        "reference_text_matched_kana": None,
        "transcribed_text_kana": "",
        "engine_used": "n/a (non_speech sample)",
        "start_sec": 0.0,
        "end_sec": round(duration_sec, 3),
        "duration_sec": round(duration_sec, 3),
        "source_dataset": source_dataset,
        "source_file": source_file,
        "source_license": source_license,
    }


def upsert_manifest_entries(
    manifest_path: Path,
    new_entries: list[dict],
    *,
    force: bool = False,
    source_dataset_filter: Optional[str] = None,
) -> tuple[int, int, int]:
    """既存 manifest + new entries を upsert し書き戻す。

    Args:
        manifest_path: manifest.jsonl の path。
        new_entries: 追記/更新する entry 群。 各 dict は ``"path"`` field 必須。
        force: True なら ``source_dataset_filter`` にマッチする既存 entry を
            削除してから upsert (再 augment 用 safety)。
        source_dataset_filter: force=True 時の削除対象 filter。 例: ``"esc50"``。

    Returns:
        ``(added, updated, removed)`` の tuple。
    """
    from .build_corpus import _load_manifest_entries, _write_manifest

    entries = _load_manifest_entries(manifest_path)

    removed = 0
    if force and source_dataset_filter is not None:
        keep: dict[str, dict] = {}
        for p, e in entries.items():
            if e.get("source_dataset") == source_dataset_filter:
                removed += 1
            else:
                keep[p] = e
        entries = keep

    added = 0
    updated = 0
    for entry in new_entries:
        path = entry.get("path")
        if not path:
            raise ValueError(f"new entry missing 'path' field: {entry!r}")
        if path in entries:
            updated += 1
        else:
            added += 1
        entries[path] = entry

    _write_manifest(manifest_path, list(entries.values()))
    return added, updated, removed


def download_dataset(
    url: str,
    dest: Path,
    *,
    expected_sha256: Optional[str] = None,
    force: bool = False,
) -> None:
    """URL からファイル download。

    Args:
        url: source URL。
        dest: local path (拡張子含む)。
        expected_sha256: 期待 SHA-256 hex (小文字)。 与えられた場合は検証。
        force: True なら既存 file を上書き。 False (default) なら skip。

    Raises:
        ValueError: hash 不一致 (削除は呼出側の責務)。
    """
    if dest.exists() and not force:
        logger.info(
            "Dataset already exists at %s (%.1f MB), skipping download (use --force to re-download)",
            dest,
            dest.stat().st_size / 1e6,
        )
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s ...", url, dest)
    urllib.request.urlretrieve(url, str(dest))
    logger.info(
        "Download complete: %s (%.1f MB)", dest, dest.stat().st_size / 1e6
    )
    if expected_sha256 is not None:
        actual = hashlib.sha256(dest.read_bytes()).hexdigest()
        if actual != expected_sha256.lower():
            raise ValueError(
                f"Downloaded dataset hash mismatch: expected {expected_sha256}, got {actual}"
            )
