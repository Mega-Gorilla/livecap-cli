"""Entry point: ``python -m benchmarks.sed``.

Orchestrates the full PR-D0 off-line evaluation pipeline:

1. Loads the EfficientAT model (variant via ``--model``)
2. Iterates over the real non-speech corpus (manifest.json)
3. Computes per-window AudioSet 527-class probabilities
4. Records per-clip per-window CSV + run metadata
5. Runs latency / memory measurement
6. Writes outputs to ``--output-dir``

The decision document is written separately by the human author (this script
only emits raw measurements).
"""

from __future__ import annotations

import sys

from benchmarks.sed.cli import main


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
