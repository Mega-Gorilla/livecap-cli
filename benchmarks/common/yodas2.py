"""YODAS2 から作った日本語の長尺評価データを読む (#488)。

リポジトリにあるのは字幕 (``captions/<audio_id>.srt``)、一覧 (``manifest.tsv``)、区間の印 (``segments.tsv``) で、
音声は無い。音声は ``scripts/prepare_yodas2_audio.py`` で YODAS2 から取り、``tests/assets/prepared/yodas2/ja/``
に置く。データの作り方と限界は ``docs/guides/benchmark/yodas2-longform-dataset.md``、出典は
``benchmarks/data/yodas2/ATTRIBUTION.md``。

split は 3 つ。

==================  ============================================================================
``test``            最終評価用。話者ごとに字幕を聞いて確かめた。設定の調整には使わない
``dev``             調整用。話者ごとに代表を聞いて「使える」とした話者
``dev_unverified``  調整用の追加分。字幕は機械の決まりで選び、声 (人か合成音声か) だけ人が聞いた
==================  ============================================================================

区間の印 (:attr:`Yodas2Segment.status`) は ``test`` にだけ付く。

==============  ===============================================================================
``ok``          印なし (聞いて「合っている」とした区間と、聞いていない区間の両方)
``edited``      字幕の言葉を直した
``notation``    言葉は合っているが、書き方が発話と違う (コードや URL の読み上げ、英語とカタカナ)
``disfluency``  字幕が口ぐせ・言い直しを省いている
``excluded``    聞き取れない。採点から外す
==============  ===============================================================================
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "AUDIO_DIR",
    "DATA_DIR",
    "SPLITS",
    "STATUSES",
    "Yodas2Segment",
    "Yodas2Video",
    "load_videos",
    "parse_srt",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
#: 字幕・一覧・区間の印 (リポジトリに入っている)
DATA_DIR = PROJECT_ROOT / "benchmarks" / "data" / "yodas2" / "ja"
#: 音声の置き場所 (リポジトリには入れない。``scripts/prepare_yodas2_audio.py`` が作る)
AUDIO_DIR = PROJECT_ROOT / "tests" / "assets" / "prepared" / "yodas2" / "ja"
SPLITS = ("test", "dev", "dev_unverified")
STATUSES = ("ok", "edited", "notation", "disfluency", "excluded")

_TIME = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{3})")


@dataclass(frozen=True)
class Yodas2Segment:
    """字幕の 1 区間。時刻は元の字幕のもので、人が確かめたものではない。"""

    start: float
    end: float
    text: str
    status: str = "ok"


@dataclass(frozen=True)
class Yodas2Video:
    """1 本の動画。``audio_id`` は YODAS2 が付けた仮の ID で、YouTube の動画の ID ではない。"""

    audio_id: str
    split: str
    speaker: str
    duration_sec: float
    kind: str
    tags: tuple[str, ...]
    speech_level_db: float
    yodas2_subset: str
    yodas2_shard: str
    segments: tuple[Yodas2Segment, ...]

    def reference(self, *, include_notation: bool = True) -> str:
        """採点の正解。``excluded`` の区間は入れない。``include_notation=False`` なら ``notation`` も外す。"""
        skip = _unscored_statuses(include_notation)
        return " ".join(s.text for s in self.segments if s.status not in skip)

    def unscored_spans(self, *, include_notation: bool = True) -> list[tuple[float, float]]:
        """採点から外す区間の時刻。ASR の出力のうち、ここに重なる部分も外してから比べる。"""
        skip = _unscored_statuses(include_notation)
        return [(s.start, s.end) for s in self.segments if s.status in skip]


def _unscored_statuses(include_notation: bool) -> frozenset[str]:
    """採点から外す区間の印。``reference()`` と ``unscored_spans()`` で同じものを使う。"""
    return frozenset({"excluded"} if include_notation else {"excluded", "notation"})


def _seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_srt(path: Path) -> list[tuple[float, float, str]]:
    """SRT を (開始, 終了, 文) の並びにする。文が複数行のときは空白でつなぐ。"""
    out: list[tuple[float, float, str]] = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip()):
        lines = [line for line in block.splitlines() if line.strip()]
        for i, line in enumerate(lines):
            m = _TIME.search(line)
            if m:
                text = " ".join(part.strip() for part in lines[i + 1:])
                if text:
                    out.append((_seconds(*m.groups()[:4]), _seconds(*m.groups()[4:]), text))
                break
    return out


def _marks(data_dir: Path) -> dict[tuple[str, float], str]:
    path = data_dir / "segments.tsv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", newline="") as fh:
        return {(r["audio_id"], round(float(r["start"]), 2)): r["status"] for r in csv.DictReader(fh, delimiter="\t")}


def load_videos(split: str | None = None, *, data_dir: Path = DATA_DIR) -> list[Yodas2Video]:
    """一覧と字幕を読む。

    Args:
        split: :data:`SPLITS` のどれか。``None`` なら全部
        data_dir: 字幕と一覧の場所

    Raises:
        ValueError: ``split`` が :data:`SPLITS` に無い。区間の印が、字幕に無い区間を指している
        FileNotFoundError: 一覧か字幕が無い
    """
    if split is not None and split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    marks = _marks(data_dir)
    used: set[tuple[str, float]] = set()
    videos = []
    with open(data_dir / "manifest.tsv", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            aid = row["audio_id"]
            segments = []
            for start, end, text in parse_srt(data_dir / "captions" / f"{aid}.srt"):
                key = (aid, round(start, 2))
                if key in marks:
                    used.add(key)
                segments.append(Yodas2Segment(start, end, text, marks.get(key, "ok")))
            if split is None or row["split"] == split:
                videos.append(
                    Yodas2Video(
                        audio_id=aid,
                        split=row["split"],
                        speaker=row["speaker"],
                        duration_sec=float(row["duration_sec"]),
                        kind=row["kind"],
                        tags=tuple(t for t in row["tags"].split(",") if t),
                        speech_level_db=float(row["speech_level_db"]),
                        yodas2_subset=row["yodas2_subset"],
                        yodas2_shard=row["yodas2_shard"],
                        segments=tuple(segments),
                    )
                )
    stray = sorted(set(marks) - used)
    if stray:
        raise ValueError(f"segments.tsv marks segments that are not in the captions: {stray[:5]}")
    return videos
