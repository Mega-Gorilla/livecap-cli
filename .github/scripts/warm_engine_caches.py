"""Load each engine once so the persistent models root holds its model (Integration Tests GPU jobs).

Usage::

    python .github/scripts/warm_engine_caches.py <target> [<target> ...]
    python .github/scripts/warm_engine_caches.py all

Targets are the keys of ``TARGETS``. A failed load is printed as ``SKIP`` and does
not fail the step - the tests that need the model decide (most of them require
``PASSED``). Each engine is cleaned up before the next one so VRAM does not pile up.

Both GPU runners use this (Windows: every target; Linux: everything but Voxtral,
which needs 16 GB). Notes on targets:

* ``reazonspeech-int8``: the #409 cache identity check needs both the int8 and the
  float32 model; without int8 it skips and the gate stops working.
* ``qwen3asr``: fills the HF hub cache. Without it the non-ASCII real-model probe
  falls back to a download, breaking the "no network" contract of that tier.
"""

from __future__ import annotations

import gc
import os
import sys
from typing import Any, Sequence

#: target -> EngineFactory.create_engine() keyword arguments (device is always cuda)
TARGETS: dict[str, dict[str, Any]] = {
    "reazonspeech": {"engine_type": "reazonspeech"},
    "reazonspeech-int8": {"engine_type": "reazonspeech", "use_int8": True},
    "whispers2t": {"engine_type": "whispers2t", "language": "en", "model_size": "base"},
    "parakeet": {"engine_type": "parakeet"},
    "parakeet_ja": {"engine_type": "parakeet_ja"},
    "canary": {"engine_type": "canary", "language": "en"},
    "voxtral": {"engine_type": "voxtral", "language": "en"},
    "qwen3asr": {"engine_type": "qwen3asr"},
}


def _patch_uname_for_lhotse() -> None:
    """lhotse calls ``os.uname()``, which Windows lacks. Keep in sync with ``tests/conftest.py``."""
    if sys.platform != "win32" or hasattr(os, "uname"):
        return
    from collections import namedtuple

    UnameResult = namedtuple("UnameResult", ["sysname", "nodename", "release", "version", "machine"])

    def uname():
        # Avoid platform.* calls that might recurse back to os.uname
        return UnameResult(
            sysname="Windows",
            nodename=os.environ.get("COMPUTERNAME", "unknown"),
            release=os.environ.get("OS", "unknown"),
            version="10.0.xxxx",
            machine=os.environ.get("PROCESSOR_ARCHITECTURE", "AMD64"),
        )

    os.uname = uname


def resolve_targets(names: Sequence[str]) -> list[str]:
    """``all`` -> every target in order; otherwise the given names, validated."""
    if list(names) == ["all"]:
        return list(TARGETS)
    unknown = [name for name in names if name not in TARGETS]
    if unknown or not names:
        raise ValueError(f"unknown or missing targets {unknown or names!r}; choose from {list(TARGETS)} or 'all'")
    return list(names)


def warm(name: str) -> bool:
    import torch

    from livecap_cli import EngineFactory

    options = dict(TARGETS[name])
    engine_type = options.pop("engine_type")
    engine = None
    try:
        engine = EngineFactory.create_engine(engine_type=engine_type, device="cuda", **options)
        engine.load_model()
        print(f"[{name}] cached at: {engine.model_manager.models_root}", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001 - a failed warm-up is reported, the tests decide
        print(f"[{name}] SKIP (warm-up failed): {exc}", flush=True)
        return False
    finally:
        if engine is not None and hasattr(engine, "cleanup"):
            engine.cleanup()
        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(argv: Sequence[str]) -> int:
    try:
        names = resolve_targets(argv[1:])
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _patch_uname_for_lhotse()
    for name in names:
        warm(name)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
