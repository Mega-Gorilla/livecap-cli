"""Confidence threshold calibration harness (Issue #338).

新規 ASR engine の confidence_filter threshold を audio corpus から自動最適化
する CLI tooling。Issue #334 の PR-2 / PR-3 / PR-4 (observe mode 1-2 月運用に
依存) を ~1-2 週に短縮する。

Sub-modules:
  - ``_core``: signal-agnostic な sweep logic (PR-α / PR-β 共通)
  - ``parse_observe``: Stage 1 CLI、observe mode JSON log → sweep report
  - ``pipeline``: corpus loader、``manifest.jsonl`` schema、audio resampling
  - ``sweep``: (PR-β) Stage 2 CLI、user 提供 audio corpus → sweep report
  - ``build_corpus``: (PR-β) yt-dlp + VAD chunking + 原稿 fuzzy match

CLI usage:
  - ``python -m benchmarks.confidence_calibration.parse_observe ...``
  - ``python -m benchmarks.confidence_calibration.sweep ...`` (PR-β)
  - ``python -m benchmarks.confidence_calibration.build_corpus ...`` (PR-β)

See ``benchmarks/confidence_calibration/README.md`` for details.
"""

from __future__ import annotations

__all__ = []  # CLI 経由で使う、re-export なし
