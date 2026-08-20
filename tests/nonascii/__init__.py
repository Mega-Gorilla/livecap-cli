"""非 ASCII パス境界の検証ハーネス (Issue #378 / epic #380)。

実装 PR (#375 / #379 / #377) から回帰テストとして再利用できるよう、
`run_probe()` と registry を公開する。

使い方は ``tests/nonascii/README.md`` を参照。
"""

from __future__ import annotations

from .record import ProbeResult, RunMetadata, Verdict
from .registry import BOUNDARIES, BoundarySpec, Method
from .runner import run_probe

__all__ = [
    "BOUNDARIES",
    "BoundarySpec",
    "Method",
    "ProbeResult",
    "RunMetadata",
    "Verdict",
    "run_probe",
]
