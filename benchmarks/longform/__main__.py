"""長尺の評価 (YODAS2 の日本語データ、#488)。

    # 1. 音声を用意する (最終評価用だけなら --split test)
    python scripts/prepare_yodas2_audio.py --split test

    # 2. 書き起こす (CLI の transcribe と同じ部品・既定値。ASR ごとに別の process で回すとよい)
    python -m benchmarks.longform transcribe --split test --engine qwen3asr_large
    python -m benchmarks.longform transcribe --split test --engine whispers2t:large-v3

    # 3. 採点して summary.json / summary.md を書く
    python -m benchmarks.longform score --split test
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from benchmarks.common.yodas2 import AUDIO_DIR, SPLITS, load_videos

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "benchmark_results" / "longform"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.longform", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("transcribe", help="CLI と同じ部品で書き起こす")
    t.add_argument("--engine", required=True, help="engine id。whispers2t は 'whispers2t:large-v3' のようにサイズを付けられる")
    t.add_argument("--split", choices=SPLITS, default="test")
    t.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto", help="CLI の --device と同じ (既定: auto)")
    t.add_argument("--audio-dir", type=Path, default=AUDIO_DIR)
    t.add_argument("--out-dir", type=Path, default=None, help=f"既定: {DEFAULT_OUT}/<split>")
    t.add_argument("--force", action="store_true", help="書き起こし済みの動画もやり直す")

    s = sub.add_parser("score", help="採点して summary.json / summary.md を書く")
    s.add_argument("--split", choices=SPLITS, default="test")
    s.add_argument("--out-dir", type=Path, default=None, help=f"既定: {DEFAULT_OUT}/<split>")
    s.add_argument("--labels", default=None, help="採点する ASR (カンマ区切り。既定: out-dir にあるもの全部)")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = args.out_dir or DEFAULT_OUT / args.split
    videos = load_videos(args.split)

    if args.command == "transcribe":
        from .runner import EngineSpec, transcribe_videos

        dest = transcribe_videos(
            EngineSpec.parse(args.engine), videos, audio_dir=args.audio_dir, out_dir=out_dir,
            device=args.device, force=args.force,
        )
        print(f"done: {dest}")
        return 0

    from .report import summarize

    labels = args.labels.split(",") if args.labels else sorted(p.name for p in out_dir.iterdir() if (p / "run.json").exists())
    if not labels:
        print(f"no transcriptions in {out_dir}", file=sys.stderr)
        return 1
    summarize(out_dir, labels, videos, split=args.split)
    print((out_dir / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
