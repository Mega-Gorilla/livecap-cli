"""Tests for ``benchmarks.confidence_calibration.recompute_alignment`` (PR-γ).

The recompute CLI adds kana fields to existing manifest entries without
re-transcribing audio. Verified invariants:

* idempotent: re-running on a manifest with kana fields is a no-op (unless
  ``--force``).
* additive: existing text-level fields (``alignment_score``,
  ``transcribed_text``, ``reference_text_matched``) are NOT modified.
* language filter: entries whose ``language`` does not match a provided
  reference are skipped.
* manifest integrity: total entry count unchanged; non-matching entries
  preserved verbatim.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.confidence_calibration.build_corpus import (
    _load_manifest_entries,
    _write_manifest,
)
from benchmarks.confidence_calibration.recompute_alignment import (
    recompute_kana_alignment,
)


def _make_manifest(tmp_path: Path, entries: list[dict]) -> Path:
    """Write entries to a manifest.jsonl in tmp_path."""
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, entries)
    return manifest


@pytest.fixture
def ja_reference() -> str:
    return (
        "声劇・朗読用台本『星の王子さま』 前編。"
        "そうしてくれたら、真っ先に僕に知らせてもらえたはずなのだが。"
        "ぼくはそのとき、1人でエンジンを修理しなければならなかった。"
    )


@pytest.fixture
def en_reference() -> str:
    return (
        "Once I saw a magnificent picture in a book. "
        "It was a picture of a boa constrictor. "
        "And I thought about it for some time."
    )


class TestRecomputeBasic:
    def test_adds_kana_fields_to_existing_entries(
        self, tmp_path: Path, ja_reference: str
    ) -> None:
        entries = [
            {
                "path": "ja_clean/segment_0000.wav",
                "label": "speech",
                "language": "ja",
                "transcribed_text": "一人でエンジンを修理しなければならなかった",
                "reference_text_matched": "...",
                "alignment_score": 0.95,
            }
        ]
        manifest = _make_manifest(tmp_path, entries)

        result = recompute_kana_alignment(
            manifest_path=manifest,
            references={"ja": ja_reference},
        )

        assert result["updated"] == 1
        assert result["skipped_already_done"] == 0
        assert result["total"] == 1

        loaded = _load_manifest_entries(manifest)
        entry = loaded["ja_clean/segment_0000.wav"]
        assert "alignment_score_kana" in entry
        assert entry["alignment_score_kana"] >= 0.9, (
            "1人で vs 一人で の表記差は kana 化で吸収されるべき"
        )
        assert "reference_text_matched_kana" in entry
        assert "transcribed_text_kana" in entry

    def test_preserves_text_level_fields(
        self, tmp_path: Path, ja_reference: str
    ) -> None:
        """Recompute MUST NOT modify text-level fields (forensic safety)."""
        original_entry = {
            "path": "ja_clean/segment_0001.wav",
            "label": "speech",
            "language": "ja",
            "transcribed_text": "一人でエンジン",
            "reference_text_matched": "一人でエンジン",
            "alignment_score": 0.7777,
            "engine_used": "whispers2t",
        }
        manifest = _make_manifest(tmp_path, [original_entry.copy()])

        recompute_kana_alignment(
            manifest_path=manifest,
            references={"ja": ja_reference},
        )

        loaded = _load_manifest_entries(manifest)
        entry = loaded["ja_clean/segment_0001.wav"]
        # text-level fields must be byte-identical
        for key in (
            "label",
            "language",
            "transcribed_text",
            "reference_text_matched",
            "alignment_score",
            "engine_used",
        ):
            assert entry[key] == original_entry[key], (
                f"text-level field {key!r} should not be modified by recompute"
            )


class TestRecomputeIdempotent:
    def test_second_call_skips_when_already_done(
        self, tmp_path: Path, ja_reference: str
    ) -> None:
        entries = [
            {
                "path": "ja_clean/segment_0000.wav",
                "label": "speech",
                "language": "ja",
                "transcribed_text": "一人で",
                "alignment_score": 0.5,
            }
        ]
        manifest = _make_manifest(tmp_path, entries)

        # 1st call
        r1 = recompute_kana_alignment(
            manifest_path=manifest, references={"ja": ja_reference}
        )
        assert r1["updated"] == 1

        # 2nd call (no force)
        r2 = recompute_kana_alignment(
            manifest_path=manifest, references={"ja": ja_reference}
        )
        assert r2["updated"] == 0
        assert r2["skipped_already_done"] == 1

    def test_force_recomputes_regardless(
        self, tmp_path: Path, ja_reference: str
    ) -> None:
        entries = [
            {
                "path": "ja_clean/segment_0000.wav",
                "label": "speech",
                "language": "ja",
                "transcribed_text": "一人で",
                "alignment_score": 0.5,
                "alignment_score_kana": 0.0,  # stale stub
                "reference_text_matched_kana": "x",
                "transcribed_text_kana": "y",
            }
        ]
        manifest = _make_manifest(tmp_path, entries)

        result = recompute_kana_alignment(
            manifest_path=manifest,
            references={"ja": ja_reference},
            force=True,
        )
        assert result["updated"] == 1
        loaded = _load_manifest_entries(manifest)
        entry = loaded["ja_clean/segment_0000.wav"]
        # Stub kana_score=0.0 should have been overwritten with real value > 0
        assert entry["alignment_score_kana"] > 0.0


class TestRecomputeLanguageFilter:
    def test_skips_entries_without_matching_reference(
        self, tmp_path: Path, ja_reference: str
    ) -> None:
        entries = [
            {
                "path": "ja_clean/segment_0000.wav",
                "label": "speech",
                "language": "ja",
                "transcribed_text": "一人で",
            },
            {
                "path": "en_clean/segment_0000.wav",
                "label": "speech",
                "language": "en",
                "transcribed_text": "It was a picture.",
            },
            {
                "path": "zh_clean/segment_0000.wav",
                "label": "speech",
                "language": "zh",
                "transcribed_text": "中文",
            },
        ]
        manifest = _make_manifest(tmp_path, entries)

        # Only JA reference provided
        result = recompute_kana_alignment(
            manifest_path=manifest, references={"ja": ja_reference}
        )
        assert result["updated"] == 1  # only the ja entry
        assert result["skipped_no_language_match"] == 2

        loaded = _load_manifest_entries(manifest)
        # JA entry got kana fields
        assert "alignment_score_kana" in loaded["ja_clean/segment_0000.wav"]
        # EN / zh entries did NOT
        assert "alignment_score_kana" not in loaded["en_clean/segment_0000.wav"]
        assert "alignment_score_kana" not in loaded["zh_clean/segment_0000.wav"]

    def test_both_languages_processed(
        self,
        tmp_path: Path,
        ja_reference: str,
        en_reference: str,
    ) -> None:
        entries = [
            {
                "path": "ja_clean/s0.wav",
                "label": "speech",
                "language": "ja",
                "transcribed_text": "一人で",
            },
            {
                "path": "en_clean/s0.wav",
                "label": "speech",
                "language": "en",
                "transcribed_text": "It was a picture",
            },
        ]
        manifest = _make_manifest(tmp_path, entries)

        result = recompute_kana_alignment(
            manifest_path=manifest,
            references={"ja": ja_reference, "en": en_reference},
        )
        assert result["updated"] == 2
        loaded = _load_manifest_entries(manifest)
        assert "alignment_score_kana" in loaded["ja_clean/s0.wav"]
        assert "alignment_score_kana" in loaded["en_clean/s0.wav"]


class TestRecomputeEdgeCases:
    def test_skips_empty_transcribed_text(
        self, tmp_path: Path, ja_reference: str
    ) -> None:
        entries = [
            {
                "path": "ja_clean/s0.wav",
                "label": "speech",
                "language": "ja",
                "transcribed_text": "",
            },
            {
                "path": "ja_clean/s1.wav",
                "label": "speech",
                "language": "ja",
                "transcribed_text": "  ",
            },
            {
                "path": "ja_clean/s2.wav",
                "label": "speech",
                "language": "ja",
                "transcribed_text": "一人で",
            },
        ]
        manifest = _make_manifest(tmp_path, entries)

        result = recompute_kana_alignment(
            manifest_path=manifest, references={"ja": ja_reference}
        )
        assert result["updated"] == 1
        assert result["skipped_no_transcribed_text"] == 2

    def test_manifest_line_count_preserved(
        self, tmp_path: Path, ja_reference: str
    ) -> None:
        """Recompute must not add or drop manifest lines (upsert pattern)."""
        entries = [
            {
                "path": f"ja_clean/s{i}.wav",
                "label": "speech",
                "language": "ja",
                "transcribed_text": f"transcribed {i}",
            }
            for i in range(5)
        ]
        manifest = _make_manifest(tmp_path, entries)

        recompute_kana_alignment(
            manifest_path=manifest, references={"ja": ja_reference}
        )

        with manifest.open("r", encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if line.strip()]
        assert len(lines) == 5, (
            f"manifest line count should be preserved, got {len(lines)}"
        )

    def test_empty_manifest_safe(self, tmp_path: Path, ja_reference: str) -> None:
        manifest = _make_manifest(tmp_path, [])
        result = recompute_kana_alignment(
            manifest_path=manifest, references={"ja": ja_reference}
        )
        assert result["updated"] == 0
        assert result["total"] == 0

    def test_non_matching_entries_preserved_byte_identical(
        self, tmp_path: Path, ja_reference: str
    ) -> None:
        """Entries with unsupported language must be retained verbatim."""
        zh_entry = {
            "path": "zh_clean/s0.wav",
            "label": "speech",
            "language": "zh",
            "transcribed_text": "你好",
            "custom_field": 42,
        }
        manifest = _make_manifest(tmp_path, [zh_entry.copy()])

        recompute_kana_alignment(
            manifest_path=manifest, references={"ja": ja_reference}
        )

        loaded = _load_manifest_entries(manifest)
        assert loaded["zh_clean/s0.wav"] == zh_entry


class TestRecomputeCli:
    def test_cli_invocation_basic(
        self, tmp_path: Path, ja_reference: str, capsys: pytest.CaptureFixture
    ) -> None:
        """CLI integration: --manifest + --reference-text-ja + local file path."""
        entries = [
            {
                "path": "ja_clean/s0.wav",
                "label": "speech",
                "language": "ja",
                "transcribed_text": "一人で",
            }
        ]
        manifest = _make_manifest(tmp_path, entries)
        ja_ref_file = tmp_path / "ja_ref.txt"
        ja_ref_file.write_text(ja_reference, encoding="utf-8")

        from benchmarks.confidence_calibration.recompute_alignment import main

        rc = main(
            [
                "--manifest",
                str(manifest),
                "--reference-text-ja",
                str(ja_ref_file),
            ]
        )
        assert rc == 0
        captured = capsys.readouterr()
        assert "Updated (kana added):" in captured.out

    def test_cli_no_reference_returns_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """No --reference-text-* args → exit 2 with helpful error."""
        manifest = _make_manifest(
            tmp_path,
            [
                {
                    "path": "ja_clean/s0.wav",
                    "language": "ja",
                    "transcribed_text": "x",
                }
            ],
        )

        from benchmarks.confidence_calibration.recompute_alignment import main

        rc = main(["--manifest", str(manifest)])
        assert rc == 2

    def test_cli_missing_manifest_returns_error(
        self, tmp_path: Path
    ) -> None:
        from benchmarks.confidence_calibration.recompute_alignment import main

        rc = main(
            [
                "--manifest",
                str(tmp_path / "does_not_exist.jsonl"),
                "--reference-text-ja",
                "ignored",
            ]
        )
        assert rc == 2
