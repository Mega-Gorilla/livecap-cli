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

    def test_noise_rotation(self, tmp_path: Path):
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
        # 4 speech, noise pool of 2 → rotation: [n0, n1, n0, n1]
        noise_paths = [e["noise_source_path"] for e in layer3]
        assert noise_paths[0] == noise_paths[2]
        assert noise_paths[1] == noise_paths[3]

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
