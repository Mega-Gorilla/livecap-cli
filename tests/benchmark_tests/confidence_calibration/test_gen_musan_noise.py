"""Tests for ``benchmarks.confidence_calibration.gen_musan_noise`` (Issue #338 Phase 2).

Real MUSAN dataset download is NOT exercised (~11 GB, user-env only).
Tests use a synthetic mini-MUSAN layout with a few fake wav files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.confidence_calibration.gen_musan_noise import (
    MUSAN_LICENSE,
    MUSAN_SOURCE_DATASET,
    _collect_noise_files,
    _select_files,
    _subtype_from_source_path,
    augment,
    main,
)


def _write_fake_musan_noise(root: Path, n_free_sound: int = 3, n_sound_bible: int = 2) -> None:
    """Create a mini musan/noise/ layout with fake wav files."""
    import soundfile as sf

    fs_dir = root / "noise" / "free-sound"
    sb_dir = root / "noise" / "sound-bible"
    fs_dir.mkdir(parents=True, exist_ok=True)
    sb_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_free_sound):
        # 10-sec random noise
        audio = np.random.RandomState(i).randn(10 * 16000).astype(np.float32) * 0.01
        sf.write(str(fs_dir / f"noise-free-sound-{i:04d}.wav"), audio, 16000)
    for i in range(n_sound_bible):
        audio = np.random.RandomState(i + 100).randn(8 * 16000).astype(np.float32) * 0.01
        sf.write(str(sb_dir / f"noise-sound-bible-{i:04d}.wav"), audio, 16000)


# ------------------- _collect_noise_files ----------------------------------


class TestCollectNoiseFiles:
    def test_musan_root(self, tmp_path: Path):
        _write_fake_musan_noise(tmp_path, 3, 2)
        files = _collect_noise_files(tmp_path)
        assert len(files) == 5
        # Sorted, free-sound comes first
        assert "free-sound" in str(files[0])

    def test_musan_noise_root(self, tmp_path: Path):
        _write_fake_musan_noise(tmp_path, 2, 2)
        files = _collect_noise_files(tmp_path / "noise")
        assert len(files) == 4

    def test_wrapper_dir_musan(self, tmp_path: Path):
        # source_dir/musan/noise/ layout
        _write_fake_musan_noise(tmp_path / "musan", 2, 2)
        files = _collect_noise_files(tmp_path)
        assert len(files) == 4

    def test_missing_noise_dir_raises(self, tmp_path: Path):
        (tmp_path / "music").mkdir()  # only music, no noise
        with pytest.raises(FileNotFoundError, match="noise"):
            _collect_noise_files(tmp_path)

    def test_missing_subdirs_returns_empty(self, tmp_path: Path):
        (tmp_path / "noise").mkdir()  # noise/ but no free-sound/ or sound-bible/
        with pytest.raises(FileNotFoundError, match="No .wav files"):
            _collect_noise_files(tmp_path)


# ------------------- _select_files -----------------------------------------


class TestSelectFiles:
    def test_selects_uniform_stride(self, tmp_path: Path):
        files = [Path(f"noise-{i:03d}.wav") for i in range(20)]
        selected = _select_files(files, n_samples=5)
        assert len(selected) == 5
        # First and last are always included
        assert selected[0] == files[0]
        assert selected[-1] == files[-1]

    def test_deterministic(self, tmp_path: Path):
        files = [Path(f"n-{i}.wav") for i in range(10)]
        assert _select_files(files, 3) == _select_files(files, 3)

    def test_n_gte_len_returns_all(self):
        files = [Path(f"n-{i}.wav") for i in range(5)]
        assert _select_files(files, 10) == files

    def test_dedup_when_stride_collides(self):
        files = [Path("a.wav"), Path("b.wav")]
        # Requesting 3 samples from 2 files: stride hits [0, 0-1, 1] → dedup → 2 unique
        selected = _select_files(files, 3)
        assert len(set(selected)) == len(selected)


# ------------------- _subtype_from_source_path -----------------------------


class TestSubtype:
    def test_free_sound_subtype(self):
        p = Path("/tmp/musan/noise/free-sound/noise-free-sound-0001.wav")
        assert _subtype_from_source_path(p) == "musan_free-sound"

    def test_sound_bible_subtype(self):
        p = Path("/tmp/musan/noise/sound-bible/noise-sound-bible-0001.wav")
        assert _subtype_from_source_path(p) == "musan_sound-bible"


# ------------------- augment (end-to-end) ----------------------------------


class TestAugment:
    def test_end_to_end(self, tmp_path: Path):
        source = tmp_path / "musan"
        _write_fake_musan_noise(source, 3, 2)
        output = tmp_path / "corpus"
        added, updated, removed = augment(
            source_dir=source,
            output_dir=output,
            n_samples=5,
            max_chunks_per_file=5,
        )
        # 5 files × up to 5 chunks each; 10-sec files chunk into up to 5 chunks (1.5 sec × 5 = 7.5 sec fits)
        # 8-sec files chunk into 5 chunks (1.5 × 5 = 7.5 fits)
        assert added > 0
        assert updated == 0
        assert removed == 0

        manifest = output / "manifest.jsonl"
        assert manifest.exists()
        entries = [json.loads(l) for l in manifest.read_text(encoding="utf-8").strip().split("\n")]
        assert all(e["source_dataset"] == MUSAN_SOURCE_DATASET for e in entries)
        assert all(e["source_license"] == MUSAN_LICENSE for e in entries)
        assert all(e["label"] == "non_speech" for e in entries)
        assert all(e["subtype"].startswith("musan_") for e in entries)

    def test_dry_run_writes_nothing(self, tmp_path: Path):
        source = tmp_path / "musan"
        _write_fake_musan_noise(source, 2, 2)
        output = tmp_path / "corpus"
        augment(
            source_dir=source, output_dir=output,
            n_samples=2, dry_run=True,
        )
        assert not (output / "manifest.jsonl").exists()

    def test_force_removes_existing_musan(self, tmp_path: Path):
        source = tmp_path / "musan"
        _write_fake_musan_noise(source, 2, 1)
        output = tmp_path / "corpus"
        augment(source_dir=source, output_dir=output, n_samples=3)
        first_count = len(
            (output / "manifest.jsonl").read_text(encoding="utf-8").strip().split("\n")
        )
        added, updated, removed = augment(
            source_dir=source, output_dir=output, n_samples=3, force=True,
        )
        assert removed == first_count
        assert added == first_count


# ------------------- main (CLI) --------------------------------------------


class TestMain:
    def test_error_without_source_or_download(self, tmp_path: Path):
        with pytest.raises(SystemExit):
            main(["--output-dir", str(tmp_path)])

    def test_returns_zero_on_success(self, tmp_path: Path, capsys):
        source = tmp_path / "musan"
        _write_fake_musan_noise(source, 2, 1)
        output = tmp_path / "corpus"
        rc = main([
            "--source-dir", str(source),
            "--output-dir", str(output),
            "--samples", "2",
        ])
        assert rc == 0
        assert "MUSAN augment done" in capsys.readouterr().out

    def test_rejects_zero_samples(self, tmp_path: Path):
        source = tmp_path / "musan"
        _write_fake_musan_noise(source, 2, 1)
        output = tmp_path / "corpus"
        with pytest.raises(SystemExit):
            main([
                "--source-dir", str(source),
                "--output-dir", str(output),
                "--samples", "0",
            ])

    def test_rejects_negative_max_chunks(self, tmp_path: Path):
        source = tmp_path / "musan"
        _write_fake_musan_noise(source, 2, 1)
        output = tmp_path / "corpus"
        with pytest.raises(SystemExit):
            main([
                "--source-dir", str(source),
                "--output-dir", str(output),
                "--samples", "2",
                "--max-chunks-per-file", "-1",
            ])
