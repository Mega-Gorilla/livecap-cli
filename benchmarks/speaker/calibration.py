"""Speaker-verification threshold calibration (FAR/FRR/EER).

Computes a cosine-similarity operating threshold for a target-speaker gate from
labeled segments, using a leave-one-out (LOO) target centroid so a test segment
never contaminates its own enrollment model.

Label sources (provided by the caller):
- gold  : human-verified per-segment speaker labels  -> true FAR/FRR.
- silver: an external diarizer's labels               -> approximate FAR/FRR.
- self  : KMeans(2) cluster labels (same embeddings)  -> *optimistic* upper bound
          (internal separability margin, not real accuracy).

Definitions (target speaker T):
- target trial : a segment of T scored vs the LOO mean of T's *other* segments.
- impostor trial: a segment of the other speaker scored vs T's full mean.
- FAR(τ) = P(impostor score >= τ)   (false accept)
- FRR(τ) = P(target score   <  τ)   (false reject)
- EER    = operating point where FAR == FRR.
Trials are pooled over both speakers acting as target.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import l2_normalize

DEFAULT_THRESHOLDS = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def load_labels(path: str | Path) -> dict[int, int | None]:
    """Load gold/silver speaker labels from a CSV or JSON file.

    Accepted formats:
    - ``.csv`` with at least ``idx`` and ``speaker`` columns (extra columns like
      ``start``/``end``/``transcript`` are ignored). Blank ``speaker`` -> None.
    - ``.json`` of the form ``{"labels": {idx: speaker}}``.

    Distinct non-empty speaker values (e.g. "A"/"B", names, or ints) are mapped
    to stable integer ids (sorted by string). Blank/missing -> None (uncertain).

    Returns:
        Mapping of segment index -> integer speaker id (or None).
    """
    p = Path(path)
    raw: dict[int, str | None] = {}

    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        for k, v in data.get("labels", {}).items():
            raw[int(k)] = None if v is None or str(v).strip() == "" else str(v).strip()
    elif p.suffix.lower() == ".csv":
        with p.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("idx") in (None, ""):
                    continue
                spk = (row.get("speaker") or "").strip()
                raw[int(row["idx"])] = spk or None
    else:
        raise ValueError(f"Unsupported labels file type: {p.suffix} (use .csv or .json)")

    distinct = sorted({v for v in raw.values() if v is not None}, key=str)
    mapping = {s: i for i, s in enumerate(distinct)}
    return {idx: (mapping[v] if v is not None else None) for idx, v in raw.items()}


def compute_eer(
    target_scores: np.ndarray, impostor_scores: np.ndarray
) -> tuple[float | None, float | None]:
    """Return (EER, threshold-at-EER). None if a class is empty."""
    target_scores = np.asarray(target_scores, dtype=np.float64)
    impostor_scores = np.asarray(impostor_scores, dtype=np.float64)
    if target_scores.size == 0 or impostor_scores.size == 0:
        return None, None

    candidates = np.unique(
        np.concatenate([target_scores, impostor_scores])
    )
    best_eer = 1.0
    best_thr = float(candidates[0])
    best_gap = np.inf
    for thr in candidates:
        far = float(np.mean(impostor_scores >= thr))
        frr = float(np.mean(target_scores < thr))
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap = gap
            best_eer = (far + frr) / 2.0
            best_thr = float(thr)
    return best_eer, best_thr


def sweep_far_frr(
    target_scores: np.ndarray,
    impostor_scores: np.ndarray,
    thresholds: list[float] | None = None,
) -> list[dict[str, float]]:
    """FAR/FRR at each threshold."""
    target_scores = np.asarray(target_scores, dtype=np.float64)
    impostor_scores = np.asarray(impostor_scores, dtype=np.float64)
    thresholds = thresholds or DEFAULT_THRESHOLDS
    rows: list[dict[str, float]] = []
    for thr in thresholds:
        far = (
            float(np.mean(impostor_scores >= thr)) if impostor_scores.size else None
        )
        frr = float(np.mean(target_scores < thr)) if target_scores.size else None
        rows.append({"threshold": float(thr), "far": far, "frr": frr})
    return rows


def _cosine_to(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of each row of (already L2-normalized) matrix to vec."""
    v = vec / (np.linalg.norm(vec) or 1.0)
    return matrix @ v


def calibrate_from_labels(
    embeddings: np.ndarray,
    labels: list[int | None],
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    """Compute EER + FAR/FRR sweep from per-segment embeddings and speaker labels.

    Args:
        embeddings: (n_segments, dim).
        labels: speaker id per segment (None = ignore). Exactly the two most
            frequent labels are treated as the two speakers.
        thresholds: cosine thresholds to report FAR/FRR at.

    Returns:
        dict with eer, eer_threshold, far_frr (list), n_target, n_impostor,
        n_uncertain, n_speakers. eer is None if not computable.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    normed = l2_normalize(embeddings)
    labels_arr = np.array(
        [l if l is not None else -1 for l in labels], dtype=int
    )

    valid = labels_arr[labels_arr >= 0]
    uniq, counts = np.unique(valid, return_counts=True)
    n_uncertain = int(np.sum(labels_arr < 0))

    if uniq.size < 2:
        return {
            "eer": None,
            "eer_threshold": None,
            "far_frr": [],
            "n_target": 0,
            "n_impostor": 0,
            "n_uncertain": n_uncertain,
            "n_speakers": int(uniq.size),
        }

    # Two most frequent speakers.
    top2 = uniq[np.argsort(counts)[::-1][:2]]
    target_scores: list[float] = []
    impostor_scores: list[float] = []

    for tgt in top2:
        tgt_idx = np.where(labels_arr == tgt)[0]
        imp_idx = np.where((labels_arr >= 0) & (labels_arr != tgt))[0]
        if tgt_idx.size < 2 or imp_idx.size == 0:
            continue

        tgt_emb = normed[tgt_idx]
        tgt_sum = tgt_emb.sum(axis=0)
        # Leave-one-out target centroid for each target segment.
        for j in range(tgt_idx.size):
            loo_centroid = (tgt_sum - tgt_emb[j]) / (tgt_idx.size - 1)
            target_scores.append(float(_cosine_to(loo_centroid, tgt_emb[j : j + 1])[0]))

        # Impostors scored vs the full target centroid.
        full_centroid = tgt_sum / tgt_idx.size
        imp_sims = _cosine_to(full_centroid, normed[imp_idx])
        impostor_scores.extend(float(s) for s in imp_sims)

    t = np.array(target_scores)
    i = np.array(impostor_scores)
    eer, eer_thr = compute_eer(t, i)
    return {
        "eer": eer,
        "eer_threshold": eer_thr,
        "far_frr": sweep_far_frr(t, i, thresholds),
        "n_target": int(t.size),
        "n_impostor": int(i.size),
        "n_uncertain": n_uncertain,
        "n_speakers": int(uniq.size),
    }


__all__ = [
    "compute_eer",
    "sweep_far_frr",
    "calibrate_from_labels",
    "load_labels",
    "DEFAULT_THRESHOLDS",
]
