"""YODAS2 の長尺評価データ (benchmarks/data/yodas2/ja、#488) と、その読み込みの test。

データそのものが、文書に書いた決まり (話者を分ける、音声を置かない、印が字幕と合う) を満たしているかも確かめる。
"""

from __future__ import annotations

import collections
import csv
import re
from pathlib import Path

import pytest

from benchmarks.common.yodas2 import DATA_DIR, SPLITS, STATUSES, load_videos, parse_srt


@pytest.fixture(scope="module")
def videos():
    return load_videos()


class TestParseSrt:
    def test_blocks_times_and_multiline_text(self, tmp_path: Path) -> None:
        srt = tmp_path / "a.srt"
        srt.write_text(
            "1\n00:00:01,070 --> 00:00:06,120\nこんにちは\n\n2\n01:02:03,500 --> 01:02:04,000\n二行の\n字幕\n\n3\n00:00:07,000 --> 00:00:08,000\n\n",
            encoding="utf-8",
        )
        assert parse_srt(srt) == [(1.07, 6.12, "こんにちは"), (3723.5, 3724.0, "二行の 字幕")]


class TestDataset:
    def test_sizes_match_the_documentation(self, videos) -> None:
        by_split = collections.defaultdict(list)
        for v in videos:
            by_split[v.split].append(v)
        assert set(by_split) == set(SPLITS)
        sizes = {s: (len({v.speaker for v in vs}), len(vs), round(sum(v.duration_sec for v in vs) / 3600, 1)) for s, vs in by_split.items()}
        assert sizes == {"test": (8, 31, 5.7), "dev": (4, 25, 5.9), "dev_unverified": (19, 21, 4.0)}

    def test_no_speaker_is_in_both_test_and_dev(self, videos) -> None:
        test = {v.speaker for v in videos if v.split == "test"}
        dev = {v.speaker for v in videos if v.split != "test"}
        assert not test & dev

    def test_test_split_rules(self, videos) -> None:
        """最終評価用は 5 人以上、1 人の時間は目標 5.5 時間の 3 割 (1.65 時間) まで。"""
        hours = collections.Counter()
        for v in videos:
            if v.split == "test":
                hours[v.speaker] += v.duration_sec / 3600
        assert len(hours) >= 5
        assert max(hours.values()) <= 1.65 + 1e-6

    def test_every_video_has_captions_within_its_length(self, videos) -> None:
        assert len({v.audio_id for v in videos}) == len(videos)
        for v in videos:
            assert v.segments, v.audio_id
            assert 300 <= v.duration_sec <= 1800, v.audio_id  # 5〜30 分
            assert all(s.text.strip() and s.end >= s.start for s in v.segments), v.audio_id
            assert [s.start for s in v.segments] == sorted(s.start for s in v.segments), v.audio_id

    def test_marks_are_only_on_the_test_split_and_known(self, videos) -> None:
        counts = collections.Counter()
        for v in videos:
            for s in v.segments:
                assert s.status in STATUSES
                if s.status != "ok":
                    assert v.split == "test", v.audio_id
                    counts[s.status] += 1
        assert counts == {"notation": 19, "edited": 17, "disfluency": 7, "excluded": 3}

    def test_captions_have_no_markup_left(self, videos) -> None:
        """字幕から消すと決めたもの (`[音楽]`、ふりがな、`×3` の略記) が残っていない。"""
        text = "\n".join(s.text for v in videos for s in v.segments)
        for token in ("[音楽]", "[拍手]", "《", "&amp;", "♪"):
            assert token not in text
        test_text = "\n".join(s.text for v in videos if v.split == "test" for s in v.segments)
        assert "×3" not in test_text and "米国" not in test_text

    def test_change_ledger_records_how_each_change_was_decided(self) -> None:
        """PR #497 レビュー F1: 人が聞いた直しと、ASR の一致で決めた直しを区別できる。"""
        with open(DATA_DIR / "caption_changes.tsv", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        methods = collections.Counter(r["method"] for r in rows)
        assert methods == {"text_rule": 586, "asr_agreement": 36, "listened": 13}
        for r in rows:
            rules = set(r["rules"].split(","))
            if r["method"] == "text_rule":
                assert not rules & {"edited", "bracket_kept"}, r
            else:
                assert rules & {"edited", "bracket_kept"}, r
        # 「米国」を「アメリカ」に置き換えた区間は、ASR の一致 22 と、人が聞いた 5 に分かれる
        beikoku = collections.Counter(r["method"] for r in rows if "米国" in r["before"] and "アメリカ" in r["after"])
        assert beikoku == {"asr_agreement": 22, "listened": 5}

    def test_removed_speaker_labels_are_not_republished_in_the_ledger(self) -> None:
        """字幕から消した話者の印 (``Y:`` など) は話した言葉ではないので、「直す前」の列でも ``<話者>:`` に伏せる。"""
        label = re.compile(r"(?:^|[\s　])[A-Za-z](?:&amp;|[A-Za-z&]){0,9}[:：](?!//)(?=\D|$)")
        with open(DATA_DIR / "caption_changes.tsv", encoding="utf-8", newline="") as fh:
            rows = [r for r in csv.DictReader(fh, delimiter="\t") if "speaker_label" in r["rules"].split(",")]
        assert len(rows) == 285
        assert all("<話者>:" in r["before"] for r in rows)
        assert not [r["before"] for r in rows if label.search(r["before"].replace("<話者>:", " "))]

    def test_no_audio_is_committed(self) -> None:
        suffixes = {p.suffix.lower() for p in DATA_DIR.parent.rglob("*") if p.is_file()}
        assert suffixes <= {".srt", ".tsv", ".md"}

    def test_every_caption_file_is_listed(self, videos) -> None:
        assert {p.stem for p in (DATA_DIR / "captions").glob("*.srt")} == {v.audio_id for v in videos}


class TestReference:
    def test_excluded_segments_are_left_out(self, videos) -> None:
        video = next(v for v in videos if any(s.status == "excluded" for s in v.segments))
        excluded = [s for s in video.segments if s.status == "excluded"]
        assert len(video.reference()) < len(" ".join(s.text for s in video.segments))
        assert video.unscored_spans() == [(s.start, s.end) for s in excluded]

    def test_notation_can_be_left_out_too(self, videos) -> None:
        video = next(v for v in videos if any(s.status == "notation" for s in v.segments))
        assert len(video.reference(include_notation=False)) < len(video.reference())
        assert len(video.unscored_spans(include_notation=False)) > len(video.unscored_spans())


def test_split_filter_and_unknown_split() -> None:
    assert {v.split for v in load_videos("dev")} == {"dev"}
    with pytest.raises(ValueError, match="unknown split"):
        load_videos("train")


def test_mark_on_a_missing_segment_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "captions").mkdir()
    (tmp_path / "captions" / "abc.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nあ\n", encoding="utf-8")
    (tmp_path / "manifest.tsv").write_text(
        "audio_id\tyodas2_subset\tyodas2_shard\tsplit\tspeaker\tduration_sec\tkind\ttags\tspeech_level_db\tsegments\tsegments_changed\n"
        "abc\tja000\t00000001\ttest\t1\t600\tother\t\t-20.0\t1\t0\n",
        encoding="utf-8",
    )
    (tmp_path / "segments.tsv").write_text("audio_id\tstart\tend\tstatus\nabc\t9.0\t10.0\texcluded\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not in the captions"):
        load_videos(data_dir=tmp_path)
