"""Run a pytest suite and require that the named tests PASSED (Integration Tests GPU jobs).

A gate that only checks the exit code goes green when its tests are skipped,
renamed, deselected or never collected (the first real-model CI run skipped
silently because the model was missing). So the suites below name the tests
that must appear as ``PASSED``.

Usage::

    python .github/scripts/run_required_tests.py <suite>

Used by the jobs that run a suite on more than one runner, so the required list
lives in one place.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Sequence

#: suite -> (gate label, pytest arguments, tests that must PASS)
SUITES: dict[str, tuple[str, list[str], list[str]]] = {
    # #409: the ReazonSpeech recognizer cache identity. engine_smoke + slow, so the
    # smoke steps do not collect it; only a runner with the real models can run it.
    # The default addopts is `-m 'not network and not slow'`, hence `-m slow`.
    "reazonspeech-cache-identity": (
        "#409 の cache identity check",
        ["tests/integration/engines/test_reazonspeech_cache_hit.py", "-m", "slow", "-v", "-rs"],
        [
            "test_second_load_returns_the_cached_recognizer[int8]",
            "test_second_load_returns_the_cached_recognizer[float32]",
            "test_changing_num_threads_rebuilds[int8]",
            "test_changing_num_threads_rebuilds[float32]",
        ],
    ),
}


def missing_passes(output: str, required: Sequence[str]) -> list[str]:
    """The required tests whose ``<id> PASSED`` line is not in the pytest output."""
    return [test for test in required if f"{test} PASSED" not in output]


def main(argv: Sequence[str]) -> int:
    if len(argv) != 2 or argv[1] not in SUITES:
        print(f"usage: run_required_tests.py <suite>  (suites: {', '.join(SUITES)})", file=sys.stderr)
        return 2
    label, args, required = SUITES[argv[1]]
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *args], capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    output = result.stdout + result.stderr
    print(output, flush=True)
    if result.returncode != 0:
        return result.returncode
    missing = missing_passes(output, required)
    if missing:
        print(
            f"error: {label} が PASSED していない - skip / 改名 / 未収集のいずれか。ゲートが失効している。"
            f"PASSED が無い test: {', '.join(missing)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
