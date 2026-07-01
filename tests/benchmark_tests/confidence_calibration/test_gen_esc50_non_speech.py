"""Tests for ``benchmarks.confidence_calibration.gen_esc50_non_speech`` (Issue #338 Phase 2).

Real ESC-50 dataset download is NOT exercised (that's a Phase 4 user-env step).
These tests build a synthetic mini-ESC-50 layout with a few fake wav files.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.confidence_calibration.gen_esc50_non_speech import (
    DEFAULT_CATEGORIES,
    ESC50_LICENSE,
    ESC50_SOURCE_DATASET,
    _parse_esc50_csv,
    _resolve_source_dir,
    _select_files_by_category,
    augment,
    main,
)


def _write_fake_esc50(root: Path, categories: list[str], files_per_category: int = 3) -> None:
    """Create a mini ESC-50 layout: root/audio/*.wav + root/meta/esc50.csv."""
    import soundfile as sf

    audio_dir = root / "audio"
    meta_dir = root / "meta"
    audio_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    target_id = 0
    for category in categories:
        for i in range(files_per_category):
            filename = f"1-{100000 + target_id * 100 + i}-A-{target_id}.wav"
            # 5 sec of quiet noise at 16 kHz (real ESC-50 is 44.1 kHz but our resampler handles it)
            audio = np.random.RandomState(target_id * 100 + i).randn(5 * 16000).astype(np.float32) * 0.01
            sf.write(str(audio_dir / filename), audio, 16000)
            rows.append({
                "filename": filename,
                "fold": "1",
                "target": str(target_id),
                "category": category,
                "esc10": "False",
                "src_file": "0",
                "take": "A",
            })
        target_id += 1

    with (meta_dir / "esc50.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "fold", "target", "category", "esc10", "src_file", "take"],
        )
        writer.writeheader()
        writer.writerows(rows)


# ------------------- _parse_esc50_csv --------------------------------------


class TestParseEsc50Csv:
    def test_parses_valid_csv(self, tmp_path: Path):
        _write_fake_esc50(tmp_path, ["clapping", "rain"], files_per_category=2)
        rows = _parse_esc50_csv(tmp_path / "meta" / "esc50.csv")
        assert len(rows) == 4
        assert rows[0]["category"] in {"clapping", "rain"}

    def test_missing_csv_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="ESC-50 meta CSV"):
            _parse_esc50_csv(tmp_path / "nonexistent.csv")


# ------------------- _select_files_by_category -----------------------------


class TestSelectFiles:
    def test_selects_top_n_per_category(self, tmp_path: Path):
        _write_fake_esc50(tmp_path, ["clapping", "rain", "dog"], files_per_category=5)
        rows = _parse_esc50_csv(tmp_path / "meta" / "esc50.csv")
        selected = _select_files_by_category(rows, ["clapping", "rain"], samples_per_category=3)
        assert len(selected) == 6  # 3 clapping + 3 rain
        clapping_rows = [r for r in selected if r["category"] == "clapping"]
        assert len(clapping_rows) == 3

    def test_deterministic_selection(self, tmp_path: Path):
        _write_fake_esc50(tmp_path, ["clapping"], files_per_category=5)
        rows = _parse_esc50_csv(tmp_path / "meta" / "esc50.csv")
        sel1 = _select_files_by_category(rows, ["clapping"], samples_per_category=2)
        sel2 = _select_files_by_category(rows, ["clapping"], samples_per_category=2)
        assert [r["filename"] for r in sel1] == [r["filename"] for r in sel2]

    def test_unknown_category_raises(self, tmp_path: Path):
        _write_fake_esc50(tmp_path, ["clapping"], files_per_category=2)
        rows = _parse_esc50_csv(tmp_path / "meta" / "esc50.csv")
        with pytest.raises(ValueError, match="Unknown ESC-50 categories"):
            _select_files_by_category(rows, ["clapping", "not_a_real_category"], samples_per_category=1)


# ------------------- _resolve_source_dir -----------------------------------


class TestResolveSourceDir:
    def test_direct_dir(self, tmp_path: Path):
        (tmp_path / "audio").mkdir()
        (tmp_path / "meta").mkdir()
        resolved = _resolve_source_dir(tmp_path)
        assert resolved == tmp_path

    def test_wrapper_dir_ESC50_master(self, tmp_path: Path):
        inner = tmp_path / "ESC-50-master"
        (inner / "audio").mkdir(parents=True)
        (inner / "meta").mkdir(parents=True)
        resolved = _resolve_source_dir(tmp_path)
        assert resolved == inner

    def test_missing_dirs_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="audio.*meta"):
            _resolve_source_dir(tmp_path)


# ------------------- augment (end-to-end sandbox) --------------------------


class TestAugment:
    def test_end_to_end_writes_manifest_and_wavs(self, tmp_path: Path):
        source = tmp_path / "esc50"
        _write_fake_esc50(source, ["clapping", "rain"], files_per_category=3)
        output = tmp_path / "corpus"
        added, updated, removed = augment(
            source_dir=source,
            output_dir=output,
            categories=["clapping", "rain"],
            samples_per_category=2,
            language="ja",
        )
        # 2 categories × 2 files × 3 chunks = 12 entries
        assert added == 12
        assert updated == 0
        assert removed == 0

        manifest = output / "manifest.jsonl"
        assert manifest.exists()
        lines = manifest.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 12

        first = json.loads(lines[0])
        assert first["label"] == "non_speech"
        assert first["source_dataset"] == ESC50_SOURCE_DATASET
        assert first["source_license"] == ESC50_LICENSE
        assert first["subtype"] in {"clapping", "rain"}
        assert first["path"].startswith("ja_non_speech_esc50/")
        # Verify wav actually written
        wav_path = output / first["path"]
        assert wav_path.exists()

    def test_dry_run_writes_nothing(self, tmp_path: Path):
        source = tmp_path / "esc50"
        _write_fake_esc50(source, ["clapping"], files_per_category=2)
        output = tmp_path / "corpus"
        augment(
            source_dir=source,
            output_dir=output,
            categories=["clapping"],
            samples_per_category=1,
            dry_run=True,
        )
        assert not (output / "manifest.jsonl").exists()
        assert not (output / "ja_non_speech_esc50").exists()

    def test_force_reruns_cleanly(self, tmp_path: Path):
        source = tmp_path / "esc50"
        _write_fake_esc50(source, ["clapping"], files_per_category=2)
        output = tmp_path / "corpus"
        # First run
        augment(
            source_dir=source, output_dir=output,
            categories=["clapping"], samples_per_category=1,
        )
        first_count = len((output / "manifest.jsonl").read_text(encoding="utf-8").strip().split("\n"))
        # Second run with --force removes old and re-adds
        added, updated, removed = augment(
            source_dir=source, output_dir=output,
            categories=["clapping"], samples_per_category=1,
            force=True,
        )
        assert removed == first_count
        assert added == first_count
        assert updated == 0

    def test_preserves_non_esc50_entries(self, tmp_path: Path):
        source = tmp_path / "esc50"
        _write_fake_esc50(source, ["clapping"], files_per_category=1)
        output = tmp_path / "corpus"
        # Pre-seed manifest with a non-esc50 entry
        output.mkdir()
        (output / "manifest.jsonl").write_text(
            json.dumps({
                "path": "ja_clean/existing.wav",
                "label": "speech",
                "language": "ja",
            }) + "\n",
            encoding="utf-8",
        )
        augment(
            source_dir=source, output_dir=output,
            categories=["clapping"], samples_per_category=1,
            force=True,
        )
        lines = (output / "manifest.jsonl").read_text(encoding="utf-8").strip().split("\n")
        paths = {json.loads(l)["path"] for l in lines}
        assert "ja_clean/existing.wav" in paths


# ------------------- main (CLI) --------------------------------------------


class TestMain:
    def test_error_when_no_source(self, tmp_path: Path):
        with pytest.raises(SystemExit):
            main(["--output-dir", str(tmp_path)])

    def test_returns_zero_on_success(self, tmp_path: Path, capsys):
        source = tmp_path / "esc50"
        _write_fake_esc50(source, ["clapping"], files_per_category=2)
        output = tmp_path / "corpus"
        rc = main([
            "--source-dir", str(source),
            "--output-dir", str(output),
            "--categories", "clapping",
            "--samples-per-category", "1",
        ])
        assert rc == 0
        captured = capsys.readouterr()
        assert "ESC-50 augment done" in captured.out

    def test_rejects_zero_samples_per_category(self, tmp_path: Path):
        source = tmp_path / "esc50"
        _write_fake_esc50(source, ["clapping"], files_per_category=1)
        output = tmp_path / "corpus"
        with pytest.raises(SystemExit):
            main([
                "--source-dir", str(source),
                "--output-dir", str(output),
                "--samples-per-category", "0",
            ])

    def test_rejects_negative_samples_per_category(self, tmp_path: Path):
        source = tmp_path / "esc50"
        _write_fake_esc50(source, ["clapping"], files_per_category=1)
        output = tmp_path / "corpus"
        with pytest.raises(SystemExit):
            main([
                "--source-dir", str(source),
                "--output-dir", str(output),
                "--samples-per-category", "-3",
            ])


# ------------------- DEFAULT_CATEGORIES invariant --------------------------


class TestDefaultCategoriesInvariant:
    def test_default_count_matches_plan(self):
        # Plan D2 committed to 15 categories
        assert len(DEFAULT_CATEGORIES) == 15

    def test_default_categories_are_unique(self):
        assert len(DEFAULT_CATEGORIES) == len(set(DEFAULT_CATEGORIES))
