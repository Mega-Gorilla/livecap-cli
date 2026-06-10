"""Phase 2 Sound-Event Detection (SED) off-line evaluation harness (Issue #305 PR-D0).

This package provides the research-only evaluation pipeline for assessing
whether a learned sound-event-detection model can replace the DSP transient
detector that PR-B calibration (#304) empirically confirmed cannot solve the
``parakeet_ja x WebRTC x real desk_tap`` hallucination case.

Scope discipline: PR-D0 does **not** integrate any SED code into
``livecap_cli/``. This package only produces:

1. Per-clip per-class probability CSV (``benchmark_results/sed/<date>/``)
2. Per-axis latency / memory measurements
3. A decision document (``docs/research/phase2-sed-evaluation-<date>.md``)

Integration is deferred to PR-D1; production default decision to PR-D2.
"""

from __future__ import annotations

__all__ = [
    "class_mapping",
    "inference",
    "metrics",
    "latency",
]
