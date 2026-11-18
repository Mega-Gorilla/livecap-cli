#!/usr/bin/env python3
"""YODAS2 の長尺評価データ (日本語) の音声を取る (#488)。

リポジトリにあるのは字幕と一覧だけで、音声は無い。この script が、一覧
(``benchmarks/data/yodas2/ja/manifest.tsv``) にある動画の音声を YODAS2 (``espnet/yodas2``、CC BY 3.0)
から取り、``tests/assets/prepared/yodas2/ja/<audio_id>.wav`` に置く。

- 出力は 16 kHz・モノラル・16 bit の WAV。**音量は変えない** (元の音量の差も評価の対象。話している部分の
  音量は一覧の ``speech_level_db`` にある)
- YODAS2 は動画を shard (tar.gz、1 つ 1〜2 GB) にまとめて配っている。必要な shard を 1 つずつ取り、
  要る動画だけ取り出して、shard は消す (``--keep-shards`` で残す)
- 取った音声は Git に入れない (``tests/assets/prepared/`` は .gitignore に入っている)

Usage:
    python scripts/prepare_yodas2_audio.py                 # 全部 (77 本、15.6 時間)
    python scripts/prepare_yodas2_audio.py --split test    # 最終評価用だけ
    python scripts/prepare_yodas2_audio.py --dry-run       # 取る shard と本数を出すだけ

必要なもの: ``huggingface_hub``、``soundfile``、``scipy`` (どれも本体の依存)。
"""

from __future__ import annotations

import argparse
import collections
import io
import re
import sys
import tarfile
from math import gcd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.common.yodas2 import AUDIO_DIR, SPLITS, Yodas2Video, load_videos  # noqa: E402

REPO = "espnet/yodas2"
TARGET_RATE = 16000
#: tar の中の名前は使わず、一覧の ID と照らす。出力先は ID から作る (tar の中のパスに左右されない)
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{6,20}$")


def convert(raw: bytes, dest: Path) -> float:
    """WAV を 16 kHz・モノラル・16 bit にして書く。音量は変えない。長さ (秒) を返す。"""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if rate != TARGET_RATE:
        g = gcd(TARGET_RATE, rate)
        mono = resample_poly(mono, TARGET_RATE // g, rate // g)
    tmp = dest.with_suffix(".tmp.wav")
    sf.write(tmp, np.clip(mono, -1.0, 1.0), TARGET_RATE, subtype="PCM_16")
    tmp.replace(dest)
    return len(mono) / TARGET_RATE


def extract(archive: Path, wanted: dict[str, Yodas2Video], out_dir: Path) -> set[str]:
    """shard から、一覧にある動画だけを取り出す。取り出せた ID を返す。"""
    found: set[str] = set()
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".wav"):
                continue
            audio_id = Path(member.name).name[:-4]
            if audio_id not in wanted:
                continue
            src = tar.extractfile(member)
            if src is None:
                continue
            seconds = convert(src.read(), out_dir / f"{audio_id}.wav")
            expected = wanted[audio_id].duration_sec
            note = "" if abs(seconds - expected) < 1.0 else f"  (一覧の長さ {expected:.1f} 秒と違う)"
            print(f"  {audio_id}: {seconds:.1f} 秒{note}", flush=True)
            found.add(audio_id)
            if found == set(wanted):
                break
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=SPLITS, action="append", help="取る split (複数可。既定は全部)")
    parser.add_argument("--output-dir", type=Path, default=AUDIO_DIR, help=f"出力先 (既定: {AUDIO_DIR})")
    parser.add_argument("--cache-dir", type=Path, help="shard を置く場所 (既定: <出力先>/_shards)")
    parser.add_argument("--keep-shards", action="store_true", help="取り出したあとも shard を消さない")
    parser.add_argument("--force", action="store_true", help="すでにある WAV も取り直す")
    parser.add_argument("--dry-run", action="store_true", help="取る shard と本数を出すだけ")
    args = parser.parse_args()

    videos = [v for v in load_videos() if not args.split or v.split in args.split]
    bad = [v.audio_id for v in videos if not SAFE_ID.match(v.audio_id)]
    if bad:
        print(f"一覧に想定外の ID がある: {bad[:3]}", file=sys.stderr)
        return 2
    out_dir: Path = args.output_dir
    todo = [v for v in videos if args.force or not (out_dir / f"{v.audio_id}.wav").exists()]
    by_shard: dict[tuple[str, str], dict[str, Yodas2Video]] = collections.defaultdict(dict)
    for v in todo:
        by_shard[(v.yodas2_subset, v.yodas2_shard)][v.audio_id] = v
    hours = sum(v.duration_sec for v in todo) / 3600
    print(f"{len(videos)} 本のうち {len(todo)} 本 ({hours:.1f} 時間) を取る。shard は {len(by_shard)} 個")
    if args.dry_run or not todo:
        for (subset, shard), wanted in sorted(by_shard.items()):
            print(f"  {subset}/{shard}: {len(wanted)} 本")
        return 0

    from huggingface_hub import hf_hub_download

    out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir or out_dir / "_shards"
    missing: list[str] = []
    for n, ((subset, shard), wanted) in enumerate(sorted(by_shard.items()), 1):
        print(f"[{n}/{len(by_shard)}] {subset}/{shard} ({len(wanted)} 本)", flush=True)
        archive = Path(
            hf_hub_download(REPO, f"data/{subset}/audio/{shard}.tar.gz", repo_type="dataset", local_dir=str(cache))
        )
        found = extract(archive, wanted, out_dir)
        missing += sorted(set(wanted) - found)
        if not args.keep_shards:
            archive.unlink()
    if missing:
        print(f"shard に無かった動画: {missing}", file=sys.stderr)
        return 1
    print(f"完了: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
