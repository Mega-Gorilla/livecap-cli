"""Print the installed torch family as `name==version` pins, one per line (#470).

Run with the venv's interpreter after `uv sync --locked`. The action reinstalls
exactly these versions from the CUDA wheel index, so the CUDA build matches the
lock without writing version numbers anywhere else.

* The local version suffix is dropped (`2.9.1+cpu` -> `2.9.1`): the CUDA index
  has no `+cpu` wheels.
* A family member that the chosen extras did not install is skipped, not an
  error - only torch itself is required.

Exit codes: 0 = pins printed, 3 = torch is not installed.
"""

from __future__ import annotations

import importlib.metadata
import sys
from typing import Callable, Optional

#: Order matters only for readable logs; torch must be present.
TORCH_FAMILY = ("torch", "torchaudio", "torchvision")

EXIT_NO_TORCH = 3


def installed_pins(version_of: Callable[[str], str] = importlib.metadata.version) -> list[str]:
    """`name==public_version` for each installed torch family member (missing ones skipped)."""
    pins = []
    for name in TORCH_FAMILY:
        try:
            version = version_of(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        pins.append(f"{name}=={version.split('+', 1)[0]}")
    return pins


def main(argv: Optional[list[str]] = None) -> int:
    pins = installed_pins()
    if not any(pin.startswith("torch==") for pin in pins):
        print("torch is not installed by the given extras", file=sys.stderr)
        return EXIT_NO_TORCH
    print("\n".join(pins))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
