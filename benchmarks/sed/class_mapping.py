"""AudioSet class taxonomy + livecap reject-signal aggregation.

Maps EfficientAT's 527-dim multi-label output to two named groups:

- ``TARGET_CLASSES`` — non-speech transients whose presence in a 1-second
  window means we want to suppress ASR (the ``desk_tap`` / applause family).
- ``SPEECH_LIKE_CLASSES`` — speech-adjacent classes whose presence MUST veto a
  suppression decision; included in policy 3 below as a safety subtractor.

Three threshold policies are provided so the decision document can compare
them per Issue #305 v3:

- :func:`max_policy` — ``max(probs[target_indices])``
- :func:`sum_policy` — ``sum(probs[target_indices])``
- :func:`target_minus_speech_policy` —
  ``max(probs[target]) - max(probs[speech_like])``

Indices are pinned against AudioSet's canonical ordering
(``class_labels_indices.csv`` in the EfficientAT repository).
:func:`load_class_names` lets metric-writing code resolve index → display name
when the CSV is available; the integrity test in
``tests/integration/sed/test_class_mapping.py`` cross-checks the hardcoded
indices against the CSV when EfficientAT is cloned.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Pinned AudioSet indices
#
# Indices are taken from EfficientAT/metadata/class_labels_indices.csv (which
# is the canonical AudioSet 527-class ordering used by the model output). If
# upstream renames or re-orders classes, the integrity test will catch the
# drift before any decision-document numbers are trusted.
# ---------------------------------------------------------------------------

# Sharp non-speech transients we want the SED layer to flag as
# "reject this 1-second window from speech recognition".
_TARGET_ENTRIES: tuple[tuple[int, str], ...] = (
    (61, "Hands"),
    (62, "Finger snapping"),
    (63, "Clapping"),
    (67, "Applause"),
    (354, "Door"),
    (357, "Sliding door"),
    (358, "Slam"),
    (359, "Knock"),
    (360, "Tap"),
    (460, "Thump, thud"),
)

# Speech-adjacent classes whose probability subtracts from the target score
# in the ``target_minus_speech`` policy (the conservative variant of
# Issue #305 v3's "target − speech_like").
_SPEECH_LIKE_ENTRIES: tuple[tuple[int, str], ...] = (
    (0, "Speech"),
    (1, "Male speech, man speaking"),
    (2, "Female speech, woman speaking"),
    (3, "Child speech, kid speaking"),
    (4, "Conversation"),
    (5, "Narration, monologue"),
    (27, "Singing"),
)

#: Frozenset of AudioSet display names that count as reject signals.
TARGET_CLASSES: frozenset[str] = frozenset(name for _, name in _TARGET_ENTRIES)

#: Frozenset of AudioSet display names that count as speech-like (suppression
#: subtractor for policy 3).
SPEECH_LIKE_CLASSES: frozenset[str] = frozenset(name for _, name in _SPEECH_LIKE_ENTRIES)

#: Sorted tuple of pinned AudioSet indices for ``TARGET_CLASSES``.
TARGET_INDICES: tuple[int, ...] = tuple(sorted(idx for idx, _ in _TARGET_ENTRIES))

#: Sorted tuple of pinned AudioSet indices for ``SPEECH_LIKE_CLASSES``.
SPEECH_LIKE_INDICES: tuple[int, ...] = tuple(
    sorted(idx for idx, _ in _SPEECH_LIKE_ENTRIES)
)

#: Total AudioSet class count exposed by EfficientAT's classifier head.
NUM_AUDIOSET_CLASSES: int = 527


def target_indices() -> tuple[int, ...]:
    """Return the sorted tuple of AudioSet indices in ``TARGET_CLASSES``."""

    return TARGET_INDICES


def speech_like_indices() -> tuple[int, ...]:
    """Return the sorted tuple of AudioSet indices in ``SPEECH_LIKE_CLASSES``."""

    return SPEECH_LIKE_INDICES


# ---------------------------------------------------------------------------
# Threshold policies
#
# Each takes a 527-vector of per-class probabilities (output of one
# sigmoid'd inference pass) and returns a single scalar reject score. The
# orchestrator sweeps thresholds against these scores to draw P-R curves.
# ---------------------------------------------------------------------------


def _validate_probs(probs: np.ndarray) -> None:
    if probs.ndim != 1 or probs.shape[0] != NUM_AUDIOSET_CLASSES:
        raise ValueError(
            f"probs must be a 1-D array of length {NUM_AUDIOSET_CLASSES}, "
            f"got shape {probs.shape}"
        )


def max_policy(probs: np.ndarray) -> float:
    """Reject score = max probability over the target class set.

    Simple and self-explanatory; tends to over-fire when several non-target
    classes are co-active.
    """

    _validate_probs(probs)
    return float(np.max(probs[list(TARGET_INDICES)]))


def sum_policy(probs: np.ndarray) -> float:
    """Reject score = sum of probabilities over the target class set.

    Sensitive to class co-activation but unbounded above 1.0 — the threshold
    sweep absorbs the rescaling.
    """

    _validate_probs(probs)
    return float(np.sum(probs[list(TARGET_INDICES)]))


def target_minus_speech_policy(probs: np.ndarray) -> float:
    """Reject score = max(target) − max(speech_like).

    The Issue #305 v3 design rationale: if the model is simultaneously
    confident a speech-like class is present, prefer to *not* reject the
    window. Returns a value in roughly ``[-1, 1]``.
    """

    _validate_probs(probs)
    target_max = float(np.max(probs[list(TARGET_INDICES)]))
    speech_max = float(np.max(probs[list(SPEECH_LIKE_INDICES)]))
    return target_max - speech_max


#: All three policies addressable by name from CLI / decision doc.
POLICIES: dict[str, "PolicyFn"] = {
    "max": max_policy,
    "sum": sum_policy,
    "target_minus_speech": target_minus_speech_policy,
}


# Type alias to avoid forward-reference noise in callers.
from collections.abc import Callable  # noqa: E402  (after POLICIES declaration)

PolicyFn = Callable[[np.ndarray], float]


# ---------------------------------------------------------------------------
# Optional CSV verifier — used by tests and the decision doc generator
# ---------------------------------------------------------------------------


def load_class_names(csv_path: Path) -> tuple[str, ...]:
    """Load AudioSet display names in their canonical index order.

    The CSV format matches EfficientAT's
    ``metadata/class_labels_indices.csv``: ``index,mid,display_name``.

    Raises :class:`FileNotFoundError` when the CSV is absent so callers can
    decide to skip (tests) or surface a helpful error (decision doc).
    """

    rows: list[tuple[int, str]] = []
    with csv_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append((int(row["index"]), row["display_name"]))

    if len(rows) != NUM_AUDIOSET_CLASSES:
        raise ValueError(
            f"Expected {NUM_AUDIOSET_CLASSES} class rows, got {len(rows)} from {csv_path}"
        )

    rows.sort(key=lambda entry: entry[0])
    return tuple(name for _, name in rows)


def verify_indices(class_names: Iterable[str]) -> None:
    """Cross-check pinned indices against a canonical name table.

    Raises :class:`AssertionError` with a precise diff if any
    ``TARGET_INDICES`` / ``SPEECH_LIKE_INDICES`` entry does not match the
    corresponding display name in the supplied table.
    """

    names = tuple(class_names)

    for idx, expected in _TARGET_ENTRIES:
        actual = names[idx]
        if actual != expected:
            raise AssertionError(
                f"TARGET index {idx} drifted: pinned={expected!r}, csv={actual!r}"
            )

    for idx, expected in _SPEECH_LIKE_ENTRIES:
        actual = names[idx]
        if actual != expected:
            raise AssertionError(
                f"SPEECH_LIKE index {idx} drifted: pinned={expected!r}, csv={actual!r}"
            )


__all__ = [
    "NUM_AUDIOSET_CLASSES",
    "TARGET_CLASSES",
    "SPEECH_LIKE_CLASSES",
    "TARGET_INDICES",
    "SPEECH_LIKE_INDICES",
    "target_indices",
    "speech_like_indices",
    "max_policy",
    "sum_policy",
    "target_minus_speech_policy",
    "POLICIES",
    "PolicyFn",
    "load_class_names",
    "verify_indices",
]
