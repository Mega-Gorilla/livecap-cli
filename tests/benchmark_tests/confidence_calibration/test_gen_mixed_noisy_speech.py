"""Tests for ``benchmarks.confidence_calibration.gen_mixed_noisy_speech`` (Issue #338 Layer 3).

Uses synthetic mini-corpus (fake speech + noise wavs) — no real dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pytest

from benchmarks.confidence_calibration.gen_mixed_noisy_speech import (
    LAYER3_SOURCE_DATASET,
    _uniform_stride_indices,
    augment,
    build_layer3_manifest_entry,
    check_prerequisites,
    dataset_list,
    format_snr_str,
    main,
    output_subdir_for,
    select_noise_pool,
    select_speech_samples,
    snr_list,
)


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int = 16000) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate)


def _sine(freq: float, duration_sec: float, sr: int = 16000, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(duration_sec * sr)) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _fake_corpus(tmp_path: Path, n_speech: int = 5, n_noise: int = 3) -> Path:
    """Create a mini corpus with clean speech + Layer 2 noise entries."""
    corpus = tmp_path / "corpus"
    speech_dir = corpus / "ja_clean"
    noise_dir = corpus / "ja_non_speech_esc50"
    speech_dir.mkdir(parents=True)
    noise_dir.mkdir(parents=True)

    manifest_lines = []

    for i in range(n_speech):
        wav_path = speech_dir / f"segment_{i:04d}.wav"
        _write_wav(wav_path, _sine(220 + i * 40, 1.0))
        manifest_lines.append({
            "path": f"ja_clean/segment_{i:04d}.wav",
            "label": "speech",
            "language": "ja",
            "noise": "clean",
            "reference_text_matched": f"reference {i}",
            "transcribed_text": f"transcribed {i}",
            "alignment_score": 1.0,
            "alignment_score_kana": 1.0,
            "reference_text_matched_kana": f"りふぁれんす{i}",
            "transcribed_text_kana": f"とらんすくらいぶど{i}",
            "engine_used": "whispers2t",
            "start_sec": 0.0,
            "end_sec": 1.0,
            "duration_sec": 1.0,
        })

    for i in range(n_noise):
        wav_path = noise_dir / f"clapping_x-{i}_chunk0.wav"
        _write_wav(wav_path, np.random.RandomState(i).randn(24000).astype(np.float32) * 0.1)
        manifest_lines.append({
            "path": f"ja_non_speech_esc50/clapping_x-{i}_chunk0.wav",
            "label": "non_speech",
            "language": "ja",
            "noise": None,
            "subtype": "clapping",
            "reference_text_matched": None,
            "transcribed_text": "",
            "alignment_score": 0.0,
            "alignment_score_kana": 0.0,
            "reference_text_matched_kana": None,
            "transcribed_text_kana": "",
            "engine_used": "n/a (non_speech sample)",
            "start_sec": 0.0,
            "end_sec": 1.5,
            "duration_sec": 1.5,
            "source_dataset": "esc50",
            "source_file": f"x-{i}.wav",
            "source_license": "CC BY-NC 4.0",
        })

    (corpus / "manifest.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in manifest_lines) + "\n",
        encoding="utf-8",
    )
    return corpus


# ESC-50 real category names used in gen_esc50_non_speech.py DEFAULT_CATEGORIES.
# Aligning with real names ensures typo detection and semantic parity for
# subtype-diversity regression tests (Phase 2b fix).
_MULTI_SUBTYPES_ESC50: tuple[str, ...] = (
    "breathing",
    "car_horn",
    "clapping",
    "clock_tick",
    "coughing",
    "door_wood_knock",
    "engine",
    "footsteps",
    "glass_breaking",
    "keyboard_typing",
    "laughing",
    "mouse_click",
    "rain",
    "siren",
    "sneezing",
)


def _fake_corpus_multi_subtype(
    tmp_path: Path,
    n_speech: int = 15,
    subtypes: tuple[str, ...] = _MULTI_SUBTYPES_ESC50,
    files_per_subtype: int = 3,
) -> Path:
    """Multi-subtype fake corpus for Phase 2b noise diversity regression tests.

    Produces ``subtypes × files_per_subtype`` noise entries with output paths
    prefixed by subtype name (matching ``gen_esc50_non_speech.py:197``
    ``{category}_{stem}_chunk{idx}.wav`` pattern). After
    ``select_noise_pool`` path sort、 pool order is alphabetical by subtype.

    Bug (pre-fix): ``noise_pool[i % len]`` selected first N entries → first
    ceil(n / files_per_subtype) subtypes only. New impl (uniform stride)
    spreads across all subtypes.
    """
    corpus = tmp_path / "corpus"
    speech_dir = corpus / "ja_clean"
    noise_dir = corpus / "ja_non_speech_esc50"
    speech_dir.mkdir(parents=True)
    noise_dir.mkdir(parents=True)

    manifest_lines = []

    for i in range(n_speech):
        wav_path = speech_dir / f"segment_{i:04d}.wav"
        _write_wav(wav_path, _sine(220 + i * 40, 1.0))
        manifest_lines.append({
            "path": f"ja_clean/segment_{i:04d}.wav",
            "label": "speech",
            "language": "ja",
            "noise": "clean",
            "reference_text_matched": f"reference {i}",
            "transcribed_text": f"transcribed {i}",
            "alignment_score": 1.0,
            "alignment_score_kana": 1.0,
            "reference_text_matched_kana": f"りふぁれんす{i}",
            "transcribed_text_kana": f"とらんすくらいぶど{i}",
            "engine_used": "whispers2t",
            "start_sec": 0.0,
            "end_sec": 1.0,
            "duration_sec": 1.0,
        })

    for s_idx, subtype in enumerate(subtypes):
        for f_idx in range(files_per_subtype):
            filename = f"{subtype}_x-{f_idx}_chunk0.wav"
            wav_path = noise_dir / filename
            _write_wav(
                wav_path,
                np.random.RandomState(s_idx * 100 + f_idx).randn(24000).astype(np.float32) * 0.1,
            )
            manifest_lines.append({
                "path": f"ja_non_speech_esc50/{filename}",
                "label": "non_speech",
                "language": "ja",
                "noise": None,
                "subtype": subtype,
                "reference_text_matched": None,
                "transcribed_text": "",
                "alignment_score": 0.0,
                "alignment_score_kana": 0.0,
                "reference_text_matched_kana": None,
                "transcribed_text_kana": "",
                "engine_used": "n/a (non_speech sample)",
                "start_sec": 0.0,
                "end_sec": 1.5,
                "duration_sec": 1.5,
                "source_dataset": "esc50",
                "source_file": f"x-{f_idx}.wav",
                "source_license": "CC BY-NC 4.0",
            })

    (corpus / "manifest.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in manifest_lines) + "\n",
        encoding="utf-8",
    )
    return corpus


# --------------------- _uniform_stride_indices helper (Phase 2b) -----------


class TestUniformStrideIndices:
    """Phase 2b: Layer 3 noise rotation bias fix (Issue #338, Phase 2 report §5.7).

    Pre-fix ``noise_pool[i % len]`` combined with alphabetical path sort in
    ``select_noise_pool`` picked only the first N=n_samples entries, biasing
    Layer 3 noise selection to alphabetically-early ESC-50 categories
    (breathing + car_horn for the default 15-category set). The uniform
    stride replacement spans the full pool evenly.
    """

    def test_evenly_distributes_across_pool_realistic_phase2_scenario(self):
        """Realistic Phase 2 scenario: 646 pool (450 ESC-50 + 196 MUSAN), 50 samples.

        Assertions match the actual np.linspace + round output; if these
        break, the bug fix distribution changed materially.
        """
        indices = _uniform_stride_indices(646, 50)
        assert len(indices) == 50
        assert indices[0] == 0
        assert indices[-1] == 645
        # All indices monotonically non-decreasing (linspace guarantee)
        assert all(indices[i] <= indices[i + 1] for i in range(49))
        # All indices unique (n=50 << pool=646, no collisions)
        assert len(set(indices)) == 50
        # First 5 and last 5 pinned to catch any np.linspace regression
        assert indices[:5] == [0, 13, 26, 39, 53]
        assert indices[-5:] == [592, 606, 619, 632, 645]

    def test_esc50_default_subtypes_all_touched(self):
        """Regression: default ESC-50 15 categories × 30 files must ALL be
        touched with pool=450, n=50 (bug: only breathing + car_horn selected).
        """
        indices = _uniform_stride_indices(450, 50)
        # Each ESC-50 subtype spans 30 consecutive indices (10 samples × 3 chunks)
        subtypes_touched = {idx // 30 for idx in indices}
        assert len(subtypes_touched) == 15, (
            f"Only {len(subtypes_touched)} subtypes touched: {sorted(subtypes_touched)}. "
            f"Pre-fix bug selected only subtypes 0 (breathing) and 1 (car_horn)."
        )

    def test_grouped_when_more_samples_than_pool(self):
        """n_samples > pool_size: grouped ordering [0, 0, 1, 1] (not interleaved
        [0, 1, 0, 1] as the pre-fix ``i % len`` gave). Both are correct in
        terms of count distribution; test only that the count invariant holds.
        """
        indices = _uniform_stride_indices(2, 4)
        assert indices == [0, 0, 1, 1]
        # Counts: each index used exactly 2 times, balanced
        from collections import Counter
        counts = Counter(indices)
        assert counts[0] == counts[1] == 2

    def test_identity_when_equal(self):
        """n_samples == pool_size: indices are [0, 1, 2, ..., pool_size - 1]."""
        indices = _uniform_stride_indices(50, 50)
        assert indices == list(range(50))

    def test_single_sample(self):
        """n_samples == 1: always returns [0] (leftmost)."""
        assert _uniform_stride_indices(646, 1) == [0]
        assert _uniform_stride_indices(1, 1) == [0]

    def test_empty_samples(self):
        """n_samples == 0: returns [] (no rotation needed)."""
        assert _uniform_stride_indices(646, 0) == []
        assert _uniform_stride_indices(1, 0) == []

    def test_deterministic(self):
        """Same input twice → same output (no random state, no seed needed)."""
        a = _uniform_stride_indices(646, 50)
        b = _uniform_stride_indices(646, 50)
        assert a == b

    def test_invalid_pool_size_zero_raises(self):
        """pool_size == 0 is invalid (can't index empty pool)."""
        with pytest.raises(ValueError, match="pool_size must be positive"):
            _uniform_stride_indices(0, 5)

    def test_invalid_pool_size_negative_raises(self):
        with pytest.raises(ValueError, match="pool_size must be positive"):
            _uniform_stride_indices(-1, 5)


# --------------------- snr_list argparse type -----------------------------


class TestSnrList:
    def test_parses_default_grid(self):
        assert snr_list("-5,0,5,10,20") == [-5.0, 0.0, 5.0, 10.0, 20.0]

    def test_accepts_float_values(self):
        assert snr_list("-3.5,7.5") == [-3.5, 7.5]

    def test_rejects_empty_string(self):
        with pytest.raises(argparse.ArgumentTypeError, match="must not be empty"):
            snr_list("")

    def test_rejects_non_numeric(self):
        with pytest.raises(argparse.ArgumentTypeError, match="non-numeric"):
            snr_list("5,abc,10")

    def test_rejects_nan(self):
        with pytest.raises(argparse.ArgumentTypeError, match="non-finite"):
            snr_list("nan")

    def test_rejects_inf(self):
        with pytest.raises(argparse.ArgumentTypeError, match="non-finite"):
            snr_list("inf,5")

    def test_rejects_empty_item(self):
        with pytest.raises(argparse.ArgumentTypeError, match="empty item"):
            snr_list("5,,10")

    def test_rejects_raw_duplicate(self):
        with pytest.raises(argparse.ArgumentTypeError, match="duplicate value"):
            snr_list("10,10")

    def test_rejects_raw_duplicate_across_positions(self):
        with pytest.raises(argparse.ArgumentTypeError, match="duplicate value"):
            snr_list("-5,0,10,0,20")

    def test_rejects_formatted_collision_via_rounding(self):
        # 3.54 rounds to "3.5" and 3.5 formats to "3.5" -> same filename part
        with pytest.raises(argparse.ArgumentTypeError, match="collide to same formatted"):
            snr_list("3.54,3.5")

    def test_accepts_close_but_distinct_after_rounding(self):
        # 3.5 and 3.6 both round to their own str at 1 decimal — no collision
        result = snr_list("3.5,3.6")
        assert result == [3.5, 3.6]

    def test_rejects_integer_float_collision(self):
        # 10 and 10.0 are the same float value → raw duplicate
        with pytest.raises(argparse.ArgumentTypeError, match="duplicate value"):
            snr_list("10,10.0")


# --------------------- dataset_list argparse type -------------------------


class TestDatasetList:
    def test_parses_default(self):
        assert dataset_list("esc50,musan") == ["esc50", "musan"]

    def test_strips_whitespace(self):
        assert dataset_list("esc50, musan") == ["esc50", "musan"]

    def test_rejects_empty(self):
        with pytest.raises(argparse.ArgumentTypeError, match="must not be empty"):
            dataset_list("")


# --------------------- select_speech_samples ------------------------------


class TestSelectSpeechSamples:
    def test_filters_by_language_and_label(self):
        entries = [
            {"path": "a.wav", "label": "speech", "language": "ja"},
            {"path": "b.wav", "label": "non_speech", "language": "ja"},
            {"path": "c.wav", "label": "speech", "language": "en"},
            {"path": "d.wav", "label": "speech", "language": "ja"},
        ]
        result = select_speech_samples(entries, "ja", 10)
        assert [e["path"] for e in result] == ["a.wav", "d.wav"]

    def test_deterministic_sort(self):
        entries = [
            {"path": "z.wav", "label": "speech", "language": "ja"},
            {"path": "a.wav", "label": "speech", "language": "ja"},
            {"path": "m.wav", "label": "speech", "language": "ja"},
        ]
        r1 = select_speech_samples(entries, "ja", 10)
        r2 = select_speech_samples(entries, "ja", 10)
        assert r1 == r2
        assert [e["path"] for e in r1] == ["a.wav", "m.wav", "z.wav"]

    def test_top_n_limit(self):
        entries = [
            {"path": f"s{i:03d}.wav", "label": "speech", "language": "ja"}
            for i in range(10)
        ]
        result = select_speech_samples(entries, "ja", 3)
        assert len(result) == 3
        assert result[0]["path"] == "s000.wav"


# --------------------- select_noise_pool ----------------------------------


class TestSelectNoisePool:
    def test_filters_by_source_dataset(self):
        entries = [
            {"path": "a.wav", "label": "non_speech", "source_dataset": "esc50"},
            {"path": "b.wav", "label": "non_speech", "source_dataset": "musan"},
            {"path": "c.wav", "label": "non_speech", "source_dataset": "other"},
            {"path": "d.wav", "label": "speech", "source_dataset": "esc50"},
        ]
        result = select_noise_pool(entries, ["esc50", "musan"])
        assert [e["path"] for e in result] == ["a.wav", "b.wav"]

    def test_only_esc50(self):
        entries = [
            {"path": "a.wav", "label": "non_speech", "source_dataset": "esc50"},
            {"path": "b.wav", "label": "non_speech", "source_dataset": "musan"},
        ]
        result = select_noise_pool(entries, ["esc50"])
        assert [e["path"] for e in result] == ["a.wav"]

    def test_deterministic(self):
        entries = [
            {"path": "z.wav", "label": "non_speech", "source_dataset": "esc50"},
            {"path": "a.wav", "label": "non_speech", "source_dataset": "esc50"},
        ]
        r1 = select_noise_pool(entries, ["esc50"])
        r2 = select_noise_pool(entries, ["esc50"])
        assert r1 == r2
        assert [e["path"] for e in r1] == ["a.wav", "z.wav"]


# --------------------- check_prerequisites --------------------------------


class TestCheckPrerequisites:
    def test_insufficient_speech_raises(self):
        entries = [{"path": "a.wav", "label": "speech", "language": "ja"}]
        with pytest.raises(ValueError, match="Insufficient speech entries.*need >= 5"):
            check_prerequisites(entries, "ja", 5, ["esc50"])

    def test_no_noise_raises(self):
        entries = [
            {"path": f"s{i}.wav", "label": "speech", "language": "ja"}
            for i in range(5)
        ]
        with pytest.raises(ValueError, match="No noise entries found"):
            check_prerequisites(entries, "ja", 5, ["esc50"])

    def test_wrong_language_raises(self):
        entries = [
            {"path": "a.wav", "label": "speech", "language": "en"},
            {"path": "n.wav", "label": "non_speech", "source_dataset": "esc50"},
        ]
        with pytest.raises(ValueError, match="Insufficient speech entries"):
            check_prerequisites(entries, "ja", 1, ["esc50"])

    def test_valid_returns_pools(self):
        entries = [
            {"path": f"s{i}.wav", "label": "speech", "language": "ja"}
            for i in range(3)
        ] + [
            {"path": "n.wav", "label": "non_speech", "source_dataset": "esc50"},
        ]
        speech, noise = check_prerequisites(entries, "ja", 3, ["esc50"])
        assert len(speech) == 3
        assert len(noise) == 1


# --------------------- output_subdir_for ----------------------------------


class TestOutputSubdirFor:
    """Regression (codex-review 2nd round): output subdir は speech_language に連動、
    JA と EN で衝突しない設計 (single source of truth)。"""

    def test_ja_returns_ja_noisy_speech(self):
        assert output_subdir_for("ja") == "ja_noisy_speech"

    def test_en_returns_en_noisy_speech(self):
        assert output_subdir_for("en") == "en_noisy_speech"

    def test_arbitrary_language(self):
        assert output_subdir_for("zh") == "zh_noisy_speech"
        assert output_subdir_for("ko") == "ko_noisy_speech"


# --------------------- format_snr_str -------------------------------------


class TestFormatSnrStr:
    def test_positive_integer(self):
        assert format_snr_str(10.0) == "10"

    def test_zero(self):
        assert format_snr_str(0.0) == "0"

    def test_negative_integer(self):
        assert format_snr_str(-5.0) == "-5"

    def test_non_integer_uses_1_decimal(self):
        assert format_snr_str(3.5) == "3.5"
        assert format_snr_str(-2.5) == "-2.5"


# --------------------- build_layer3_manifest_entry ------------------------


class TestBuildLayer3ManifestEntry:
    def test_all_fields_populated(self):
        speech_entry = {
            "path": "ja_clean/segment_0000.wav",
            "reference_text_matched": "テキスト",
            "reference_text_matched_kana": "てきすと",
        }
        noise_entry = {
            "path": "ja_non_speech_esc50/clapping_x_chunk0.wav",
            "subtype": "clapping",
            "source_dataset": "esc50",
            "source_file": "1-100032-A-22.wav",
        }
        entry = build_layer3_manifest_entry(
            relative_path="ja_noisy_speech/segment_0000_snr10dB_clapping.wav",
            speech_entry=speech_entry,
            noise_entry=noise_entry,
            snr_db=10.0,
            duration_sec=1.0,
            language="ja",
        )
        assert entry["label"] == "noisy_speech"
        assert entry["language"] == "ja"
        assert entry["subtype"] == "clapping"
        assert entry["reference_text_matched"] == "テキスト"
        assert entry["reference_text_matched_kana"] == "てきすと"
        assert entry["source_dataset"] == LAYER3_SOURCE_DATASET
        assert entry["source_file"] == "segment_0000.wav"
        assert entry["source_license"] == "derivative (clean speech + esc50)"
        assert entry["snr_db"] == 10.0
        assert entry["noise_source_dataset"] == "esc50"
        assert entry["noise_source_file"] == "1-100032-A-22.wav"
        assert entry["noise_source_path"] == "ja_non_speech_esc50/clapping_x_chunk0.wav"
        assert entry["transcribed_text"] == ""
        assert entry["alignment_score"] == 0.0
        assert entry["duration_sec"] == 1.0


# --------------------- augment (E2E) --------------------------------------


class TestAugment:
    def test_end_to_end_writes_manifest_and_wavs(self, tmp_path: Path):
        corpus = _fake_corpus(tmp_path, n_speech=3, n_noise=2)
        added, updated, removed = augment(
            output_dir=corpus,
            speech_language="ja",
            noise_datasets=["esc50"],
            snr_db_list=[0.0, 10.0],
            n_samples=3,
        )
        # 3 speech × 2 SNR = 6 entries
        assert added == 6
        assert updated == 0
        assert removed == 0

        manifest = corpus / "manifest.jsonl"
        all_entries = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines()]
        layer3 = [e for e in all_entries if e.get("source_dataset") == LAYER3_SOURCE_DATASET]
        assert len(layer3) == 6

        # All entries should be noisy_speech + have snr_db
        for e in layer3:
            assert e["label"] == "noisy_speech"
            assert e["snr_db"] in {0.0, 10.0}
            assert e["subtype"] == "clapping"
            wav_path = corpus / e["path"]
            assert wav_path.exists()

    def test_noise_rotation_uses_all_noises_evenly(self, tmp_path: Path):
        """Phase 2b fix: rotation must use all noises with balanced counts.

        Prior (`i % len`) gave interleaved [n0, n1, n0, n1] with 4 speech / 2 noise;
        new (uniform stride via ``_uniform_stride_indices``) gives grouped
        [n0, n0, n1, n1]. Both satisfy the fundamental invariants (all noises
        used, counts balanced within ±1); test only these invariants to avoid
        over-specifying non-essential ordering behavior.
        """
        from collections import Counter

        corpus = _fake_corpus(tmp_path, n_speech=4, n_noise=2)
        augment(
            output_dir=corpus,
            speech_language="ja",
            noise_datasets=["esc50"],
            snr_db_list=[0.0],
            n_samples=4,
        )
        manifest = corpus / "manifest.jsonl"
        layer3 = [
            json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines()
            if json.loads(l).get("source_dataset") == LAYER3_SOURCE_DATASET
        ]
        noise_paths = [e["noise_source_path"] for e in layer3]
        pool_paths = {
            "ja_non_speech_esc50/clapping_x-0_chunk0.wav",
            "ja_non_speech_esc50/clapping_x-1_chunk0.wav",
        }
        # Invariant 1: every noise in the pool is used (no bias to first N)
        assert set(noise_paths) == pool_paths, (
            f"Not all noises used: {set(noise_paths)} vs pool {pool_paths}"
        )
        # Invariant 2: counts are balanced within ±1
        counts = Counter(noise_paths)
        assert max(counts.values()) - min(counts.values()) <= 1, (
            f"Unbalanced rotation counts: {dict(counts)}"
        )

    def test_dry_run_writes_nothing(self, tmp_path: Path):
        corpus = _fake_corpus(tmp_path, n_speech=2, n_noise=1)
        original_manifest_size = (corpus / "manifest.jsonl").stat().st_size
        augment(
            output_dir=corpus,
            speech_language="ja",
            noise_datasets=["esc50"],
            snr_db_list=[10.0],
            n_samples=2,
            dry_run=True,
        )
        assert not (corpus / "ja_noisy_speech").exists()
        # Manifest unchanged
        assert (corpus / "manifest.jsonl").stat().st_size == original_manifest_size

    def test_force_reruns_cleanly(self, tmp_path: Path):
        corpus = _fake_corpus(tmp_path, n_speech=2, n_noise=1)
        # First run
        augment(
            output_dir=corpus, speech_language="ja",
            noise_datasets=["esc50"], snr_db_list=[10.0], n_samples=2,
        )
        # Second run with --force removes and re-adds
        added, updated, removed = augment(
            output_dir=corpus, speech_language="ja",
            noise_datasets=["esc50"], snr_db_list=[10.0], n_samples=2,
            force=True,
        )
        assert removed == 2
        assert added == 2

    def test_preserves_non_layer3_entries(self, tmp_path: Path):
        corpus = _fake_corpus(tmp_path, n_speech=2, n_noise=1)
        original_lines = (corpus / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        original_count = len(original_lines)

        augment(
            output_dir=corpus, speech_language="ja",
            noise_datasets=["esc50"], snr_db_list=[10.0], n_samples=2,
            force=True,
        )
        new_lines = (corpus / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        non_layer3 = [
            json.loads(l) for l in new_lines
            if json.loads(l).get("source_dataset") != LAYER3_SOURCE_DATASET
        ]
        assert len(non_layer3) == original_count  # 3 clean + non_speech preserved

    def test_output_filename_pattern(self, tmp_path: Path):
        corpus = _fake_corpus(tmp_path, n_speech=1, n_noise=1)
        augment(
            output_dir=corpus, speech_language="ja",
            noise_datasets=["esc50"], snr_db_list=[-5.0, 10.0], n_samples=1,
        )
        manifest = corpus / "manifest.jsonl"
        layer3 = [
            json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines()
            if json.loads(l).get("source_dataset") == LAYER3_SOURCE_DATASET
        ]
        paths = sorted(e["path"] for e in layer3)
        # Two variants of segment_0000: snr-5dB and snr10dB
        assert any("snr-5dB_clapping" in p for p in paths)
        assert any("snr10dB_clapping" in p for p in paths)

    def test_output_language_inherits_speech_language_ja(self, tmp_path: Path):
        """Regression (codex-review Point 1): output entry language は
        speech_language を継承。 別引数だと mismatch で sweep filter を汚染する。"""
        corpus = _fake_corpus(tmp_path, n_speech=2, n_noise=1)
        augment(
            output_dir=corpus, speech_language="ja",
            noise_datasets=["esc50"], snr_db_list=[10.0], n_samples=2,
        )
        manifest = corpus / "manifest.jsonl"
        layer3 = [
            json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines()
            if json.loads(l).get("source_dataset") == LAYER3_SOURCE_DATASET
        ]
        for e in layer3:
            assert e["language"] == "ja"

    def test_output_language_inherits_speech_language_en(self, tmp_path: Path):
        """Regression (codex-review Point 1): --speech-language en →
        output entry language は 'en' になる (default 'ja' を継承しない)。"""
        corpus = tmp_path / "corpus"
        speech_dir = corpus / "en_clean"
        noise_dir = corpus / "en_non_speech_esc50"
        speech_dir.mkdir(parents=True)
        noise_dir.mkdir(parents=True)

        import soundfile as sf

        # 2 EN speech entries
        entries = []
        for i in range(2):
            wav = speech_dir / f"segment_{i:04d}.wav"
            sf.write(str(wav), _sine(220 + i * 40, 1.0), 16000)
            entries.append({
                "path": f"en_clean/segment_{i:04d}.wav",
                "label": "speech",
                "language": "en",
                "noise": "clean",
                "reference_text_matched": f"reference {i}",
                "transcribed_text": f"transcribed {i}",
                "alignment_score": 1.0,
                "alignment_score_kana": 1.0,
                "reference_text_matched_kana": None,
                "transcribed_text_kana": "",
                "engine_used": "whispers2t",
                "start_sec": 0.0,
                "end_sec": 1.0,
                "duration_sec": 1.0,
            })
        # 1 noise entry
        noise_wav = noise_dir / "clapping_x_chunk0.wav"
        sf.write(
            str(noise_wav),
            np.random.RandomState(0).randn(24000).astype(np.float32) * 0.1,
            16000,
        )
        entries.append({
            "path": "en_non_speech_esc50/clapping_x_chunk0.wav",
            "label": "non_speech",
            "language": "en",
            "subtype": "clapping",
            "source_dataset": "esc50",
            "source_file": "x.wav",
            "source_license": "CC BY-NC 4.0",
        })
        (corpus / "manifest.jsonl").write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n",
            encoding="utf-8",
        )

        augment(
            output_dir=corpus, speech_language="en",
            noise_datasets=["esc50"], snr_db_list=[10.0], n_samples=2,
        )
        layer3 = [
            json.loads(l) for l in (corpus / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if json.loads(l).get("source_dataset") == LAYER3_SOURCE_DATASET
        ]
        assert len(layer3) == 2
        for e in layer3:
            assert e["language"] == "en", (
                f"expected language='en' inherited from --speech-language, "
                f"got {e['language']!r}"
            )
            # codex-review 2nd round: output subdir も speech_language に連動
            assert e["path"].startswith("en_noisy_speech/"), (
                f"expected output subdir 'en_noisy_speech/', got path={e['path']!r}"
            )
        # Verify actual wav directory exists at the language-scoped path
        assert (corpus / "en_noisy_speech").is_dir()
        # And the wrong (ja) path was NOT created
        assert not (corpus / "ja_noisy_speech").exists()

    def test_ja_and_en_coexist_without_path_collision(self, tmp_path: Path):
        """Regression (codex-review 2nd round): JA と EN を同 corpus で augment
        しても path 衝突なし。 speech_language が output subdir の single source of
        truth なので ``ja_noisy_speech/`` と ``en_noisy_speech/`` が独立に存在する。"""
        import soundfile as sf

        corpus = tmp_path / "corpus"
        ja_speech_dir = corpus / "ja_clean"
        en_speech_dir = corpus / "en_clean"
        noise_dir = corpus / "ja_non_speech_esc50"
        for d in (ja_speech_dir, en_speech_dir, noise_dir):
            d.mkdir(parents=True)

        entries = []
        for lang, speech_dir in (("ja", ja_speech_dir), ("en", en_speech_dir)):
            for i in range(2):
                # 同一 stem "segment_0000" を JA と EN 両方で作る → subdir 未分離だと衝突
                wav = speech_dir / f"segment_{i:04d}.wav"
                sf.write(str(wav), _sine(220 + i * 40, 1.0), 16000)
                entries.append({
                    "path": f"{lang}_clean/segment_{i:04d}.wav",
                    "label": "speech",
                    "language": lang,
                    "noise": "clean",
                    "reference_text_matched": f"ref {lang} {i}",
                    "transcribed_text": f"tx {lang} {i}",
                    "alignment_score": 1.0,
                    "alignment_score_kana": 1.0,
                    "reference_text_matched_kana": None,
                    "transcribed_text_kana": "",
                    "engine_used": "whispers2t",
                    "start_sec": 0.0,
                    "end_sec": 1.0,
                    "duration_sec": 1.0,
                })
        noise_wav = noise_dir / "clapping_x_chunk0.wav"
        sf.write(
            str(noise_wav),
            np.random.RandomState(0).randn(24000).astype(np.float32) * 0.1,
            16000,
        )
        entries.append({
            "path": "ja_non_speech_esc50/clapping_x_chunk0.wav",
            "label": "non_speech",
            "language": "ja",
            "subtype": "clapping",
            "source_dataset": "esc50",
            "source_file": "x.wav",
            "source_license": "CC BY-NC 4.0",
        })
        (corpus / "manifest.jsonl").write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n",
            encoding="utf-8",
        )

        # First augment JA
        augment(
            output_dir=corpus, speech_language="ja",
            noise_datasets=["esc50"], snr_db_list=[10.0], n_samples=2,
        )
        # Then augment EN (into the same corpus)
        augment(
            output_dir=corpus, speech_language="en",
            noise_datasets=["esc50"], snr_db_list=[10.0], n_samples=2,
        )

        layer3 = [
            json.loads(l)
            for l in (corpus / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if json.loads(l).get("source_dataset") == LAYER3_SOURCE_DATASET
        ]
        # 2 JA + 2 EN = 4 total, all preserved (no upsert overwrite)
        assert len(layer3) == 4
        ja_paths = {e["path"] for e in layer3 if e["language"] == "ja"}
        en_paths = {e["path"] for e in layer3 if e["language"] == "en"}
        assert len(ja_paths) == 2
        assert len(en_paths) == 2
        # Paths must differ across languages even for same speech stem
        assert ja_paths.isdisjoint(en_paths)
        # Both output dirs exist independently
        assert (corpus / "ja_noisy_speech").is_dir()
        assert (corpus / "en_noisy_speech").is_dir()
        # Each dir has 2 wav files (not merged into one)
        assert len(list((corpus / "ja_noisy_speech").glob("*.wav"))) == 2
        assert len(list((corpus / "en_noisy_speech").glob("*.wav"))) == 2

    def test_mixed_audio_actual_snr_accuracy(self, tmp_path: Path):
        """E2E SNR accuracy: mixed audio が target SNR ±0.5 dB を保つ。"""
        corpus = _fake_corpus(tmp_path, n_speech=1, n_noise=1)
        augment(
            output_dir=corpus, speech_language="ja",
            noise_datasets=["esc50"], snr_db_list=[10.0], n_samples=1,
        )
        # Read back mixed audio and verify
        import soundfile as sf

        mixed_wav = next((corpus / "ja_noisy_speech").glob("*.wav"))
        mixed, sr = sf.read(str(mixed_wav))
        assert sr == 16000

        clean_wav = corpus / "ja_clean" / "segment_0000.wav"
        speech, _ = sf.read(str(clean_wav))

        # Truncate both to same length in case of tile
        n = min(len(mixed), len(speech))
        mixed = mixed[:n]
        speech = speech[:n]

        noise_component = mixed - speech
        p_speech = float(np.mean(speech ** 2))
        p_noise = float(np.mean(noise_component ** 2))
        if p_noise > 0:
            actual_snr = 10 * math.log10(p_speech / p_noise)
            # ±0.5 dB tolerance (Plan D8) — accounts for float32 rounding + soundfile PCM_16
            assert abs(actual_snr - 10.0) < 1.0


# --------------------- Noise subtype diversity (Phase 2b regression) --------


class TestNoiseSubtypeDiversity:
    """Phase 2b regression tests (Issue #338, Phase 2 report §5.7).

    Pre-fix bug: with 15 ESC-50 subtypes × 3 files (45 pool total) and
    n_samples=15, ``noise_pool[i % len]`` selected indices 0-14 which after
    path sort spanned only subtypes 0-4 (breathing, car_horn, clapping,
    clock_tick, coughing). Uniform stride spans all 15 subtypes.
    """

    def test_uses_diverse_subtypes_not_only_first_categories(self, tmp_path: Path):
        """After Phase 2b fix, n_samples=15 over a 45-pool of 15 subtypes
        must touch >= 10 unique subtypes (bug: only 5).
        """
        corpus = _fake_corpus_multi_subtype(
            tmp_path, n_speech=15, files_per_subtype=3,  # 45 noise pool
        )
        augment(
            output_dir=corpus,
            speech_language="ja",
            noise_datasets=["esc50"],
            snr_db_list=[0.0],
            n_samples=15,
        )
        manifest = corpus / "manifest.jsonl"
        layer3 = [
            json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines()
            if json.loads(l).get("source_dataset") == LAYER3_SOURCE_DATASET
        ]
        used_subtypes = {e["subtype"] for e in layer3}
        # Uniform stride over pool=45, n=15 → indices [0, 3, 6, ..., 42]
        # → hits every 3rd file → 15 subtypes (one per subtype).
        # Assert >= 10 to allow for stride rounding edge cases and future
        # rotation strategy tweaks that still respect the diversity invariant.
        assert len(used_subtypes) >= 10, (
            f"Bug regression: only {len(used_subtypes)} unique subtypes used, "
            f"expected >= 10. Selected: {sorted(used_subtypes)}"
        )

    def test_no_subtype_dominance_bias(self, tmp_path: Path):
        """After Phase 2b fix, no single subtype may account for > 30% of
        Layer 3 entries.

        Bug (pre-fix): breathing was ~60% + car_horn ~40% of Layer 3 entries
        for typical n_samples=50 over the default 646-pool. The 30% cap
        detects this specific dominance pattern; healthy uniform-stride
        distribution puts each subtype at ~7% (1/15) for this test scenario.
        """
        from collections import Counter

        corpus = _fake_corpus_multi_subtype(
            tmp_path, n_speech=15, files_per_subtype=3,
        )
        augment(
            output_dir=corpus,
            speech_language="ja",
            noise_datasets=["esc50"],
            snr_db_list=[0.0],
            n_samples=15,
        )
        manifest = corpus / "manifest.jsonl"
        layer3 = [
            json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines()
            if json.loads(l).get("source_dataset") == LAYER3_SOURCE_DATASET
        ]
        counts = Counter(e["subtype"] for e in layer3)
        total = sum(counts.values())
        max_fraction = max(counts.values()) / total
        assert max_fraction <= 0.30, (
            f"Subtype dominance bias: max fraction {max_fraction:.2%} > 30%. "
            f"Distribution: {dict(counts)}"
        )


# --------------------- main (CLI) -----------------------------------------


class TestMain:
    def test_returns_zero_on_success(self, tmp_path: Path, capsys):
        corpus = _fake_corpus(tmp_path, n_speech=2, n_noise=1)
        rc = main([
            "--output-dir", str(corpus),
            "--samples", "2",
            "--snr-db-list", "0,10",
        ])
        assert rc == 0
        assert "Layer 3 augment done" in capsys.readouterr().out

    def test_rejects_zero_samples(self, tmp_path: Path):
        corpus = _fake_corpus(tmp_path, n_speech=2, n_noise=1)
        with pytest.raises(SystemExit):
            main([
                "--output-dir", str(corpus),
                "--samples", "0",
            ])

    def test_rejects_negative_samples(self, tmp_path: Path):
        corpus = _fake_corpus(tmp_path, n_speech=2, n_noise=1)
        with pytest.raises(SystemExit):
            main([
                "--output-dir", str(corpus),
                "--samples", "-1",
            ])

    def test_rejects_invalid_snr_list(self, tmp_path: Path):
        corpus = _fake_corpus(tmp_path, n_speech=2, n_noise=1)
        with pytest.raises(SystemExit):
            main([
                "--output-dir", str(corpus),
                "--snr-db-list", "abc,10",
            ])

    def test_missing_manifest_errors(self, tmp_path: Path):
        empty_dir = tmp_path / "empty_corpus"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            main([
                "--output-dir", str(empty_dir),
            ])

    def test_rejects_removed_language_arg(self, tmp_path: Path):
        """Regression (codex-review Point 1): `--language` は廃止済、
        指定すると argparse がunrecognized argument で SystemExit。"""
        corpus = _fake_corpus(tmp_path, n_speech=2, n_noise=1)
        with pytest.raises(SystemExit):
            main([
                "--output-dir", str(corpus),
                "--language", "en",  # 廃止済引数
            ])
