"""Generate a gold-label template CSV from a speaker-benchmark run.

Reads a run's ``transcripts.json`` (segment idx/start/end/text produced by the
speaker benchmark) and writes a CSV with an empty ``speaker`` column for a human
to fill in. The filled CSV can be fed straight back via:

    python -m benchmarks.speaker --backend titanet ecapa pyannote --device cuda \
        --calibrate --label-source gold --labels-file <filled.csv>

Fill ``speaker`` with two distinct values (e.g. A / B); leave blank for
overlapping / unclear segments (treated as uncertain and excluded).

IMPORTANT: segment indices must match the run that produced the transcripts,
so calibrate with the SAME --min-segment-s (default 0.3). The transcript text is
git-unshareable conversation content, so the template stays local (gitignored).

Output defaults to ``benchmarks/speaker/data/labels_local.csv`` (gitignored) so
filling it in does not dirty the tracked reference ``labels_template.csv``.

Examples
--------
    # From a specific results dir (auto-finds transcripts.json):
    uv run python scripts/make_label_template.py benchmark_results/speaker_<ts>

    # From an explicit transcripts.json, custom output:
    uv run python scripts/make_label_template.py path/to/transcripts.json -o labels.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Default output is gitignored (data/* except labels_template.csv) so that filling
# in labels does not dirty the tracked reference template (labels_template.csv).
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmarks" / "speaker" / "data" / "labels_local.csv"


def _resolve_transcripts(src: Path) -> Path:
    if src.is_dir():
        cand = src / "transcripts.json"
        if not cand.exists():
            raise FileNotFoundError(f"transcripts.json not found in {src}")
        return cand
    if src.suffix.lower() == ".json":
        return src
    raise ValueError(f"Expected a results dir or transcripts.json, got: {src}")


def _latest_results_dir() -> Path | None:
    base = PROJECT_ROOT / "benchmark_results"
    if not base.exists():
        return None
    dirs = sorted(
        (d for d in base.glob("speaker_*") if (d / "transcripts.json").exists()),
        key=lambda d: d.stat().st_mtime,
    )
    return dirs[-1] if dirs else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help="Results dir or transcripts.json. Default: latest benchmark_results/speaker_*",
    )
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    src = Path(args.source) if args.source else _latest_results_dir()
    if src is None:
        print(
            "No source given and no benchmark_results/speaker_* with transcripts found.\n"
            "Run the benchmark first (it transcribes with parakeet_ja by default).",
            file=sys.stderr,
        )
        return 1

    transcripts_path = _resolve_transcripts(src)
    data = json.loads(transcripts_path.read_text(encoding="utf-8"))
    segments = data.get("segments", [])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "start", "end", "speaker", "transcript"])
        for s in segments:
            writer.writerow([s["idx"], s.get("start"), s.get("end"), "", s.get("text", "")])

    print(f"Wrote {len(segments)} rows -> {args.output}")
    print(
        "Fill the 'speaker' column (e.g. A / B); leave blank for unclear segments, "
        "then pass it to --labels-file with --label-source gold."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
