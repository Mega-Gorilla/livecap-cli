"""Tests for ``benchmarks.confidence_calibration._augment_common`` (Issue #338 Phase 2).

Fixture wav via soundfile + numpy, no real dataset download.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.confidence_calibration._augment_common import (
    SAMPLE_RATE,
    build_non_speech_manifest_entry,
    chunk_audio,
    download_dataset,
    load_audio_16k_mono,
    upsert_manifest_entries,
    write_chunk_wav,
)


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate)


# ------------------- chunk_audio -------------------------------------------


class TestChunkAudio:
    def test_empty_input_returns_empty_list(self):
        assert chunk_audio(np.array([], dtype=np.float32)) == []

    def test_shorter_than_chunk_returns_original(self):
        # 0.5 sec < 1.5 sec chunk_duration
        audio = np.ones(SAMPLE_RATE // 2, dtype=np.float32)
        chunks = chunk_audio(audio, chunk_duration_sec=1.5)
        assert len(chunks) == 1
        assert len(chunks[0]) == SAMPLE_RATE // 2
        assert chunks[0].dtype == np.float32

    def test_exactly_one_chunk_length(self):
        audio = np.ones(int(1.5 * SAMPLE_RATE), dtype=np.float32)
        chunks = chunk_audio(audio, chunk_duration_sec=1.5, max_chunks_per_file=3)
        assert len(chunks) == 1
        assert len(chunks[0]) == int(1.5 * SAMPLE_RATE)

    def test_esc50_5sec_produces_3_chunks(self):
        # ESC-50 pattern: 5 sec fixed at 16 kHz
        audio = np.arange(5 * SAMPLE_RATE, dtype=np.float32)
        chunks = chunk_audio(audio, chunk_duration_sec=1.5, max_chunks_per_file=3)
        assert len(chunks) == 3
        # All same length
        for c in chunks:
            assert len(c) == int(1.5 * SAMPLE_RATE)
        # Chunks are deterministic — from start, middle, and end
        max_start = 5 * SAMPLE_RATE - int(1.5 * SAMPLE_RATE)
        assert chunks[0][0] == 0.0
        assert chunks[2][0] == max_start
        # Middle chunk should be at ~max_start/2
        assert chunks[1][0] == pytest.approx(max_start / 2, abs=1.0)

    def test_deterministic_same_input_same_output(self):
        audio = np.arange(5 * SAMPLE_RATE, dtype=np.float32)
        chunks1 = chunk_audio(audio, chunk_duration_sec=1.5, max_chunks_per_file=3)
        chunks2 = chunk_audio(audio, chunk_duration_sec=1.5, max_chunks_per_file=3)
        for c1, c2 in zip(chunks1, chunks2):
            assert np.array_equal(c1, c2)

    def test_max_chunks_cap_applied(self):
        # 60 sec (MUSAN-like) — many chunks possible, cap at 5
        audio = np.arange(60 * SAMPLE_RATE, dtype=np.float32)
        chunks = chunk_audio(audio, chunk_duration_sec=1.5, max_chunks_per_file=5)
        assert len(chunks) == 5

    def test_returns_float32(self):
        audio = np.ones(3 * SAMPLE_RATE, dtype=np.float64)
        chunks = chunk_audio(audio)
        assert all(c.dtype == np.float32 for c in chunks)

    def test_chunks_do_not_overlap_boundary_start_end(self):
        # 5 sec input, 3 chunks of 1.5 sec each: verify chunks span [0, len]
        audio = np.arange(5 * SAMPLE_RATE, dtype=np.float32)
        chunks = chunk_audio(audio, chunk_duration_sec=1.5, max_chunks_per_file=3)
        # First chunk must start at 0
        assert chunks[0][0] == 0.0
        # Last chunk must end at total length
        assert chunks[-1][-1] == audio[-1]


# ------------------- load_audio_16k_mono -----------------------------------


class TestLoadAudio16kMono:
    def test_load_16k_mono_no_resample(self, tmp_path: Path):
        audio = np.random.randn(SAMPLE_RATE).astype(np.float32)
        wav = tmp_path / "a.wav"
        _write_wav(wav, audio, SAMPLE_RATE)
        loaded = load_audio_16k_mono(wav)
        assert loaded.dtype == np.float32
        assert loaded.ndim == 1
        assert len(loaded) == SAMPLE_RATE

    def test_load_44k_resamples_to_16k(self, tmp_path: Path):
        # 1 sec @ 44.1 kHz (ESC-50 pattern)
        audio = np.random.randn(44100).astype(np.float32)
        wav = tmp_path / "esc50_style.wav"
        _write_wav(wav, audio, 44100)
        loaded = load_audio_16k_mono(wav)
        assert loaded.dtype == np.float32
        # 44100 * (16000/44100) ~ 16000 samples (allow ±small drift due to resample_poly)
        assert abs(len(loaded) - SAMPLE_RATE) <= 10

    def test_load_stereo_converts_to_mono(self, tmp_path: Path):
        # 1 sec stereo at 16 kHz
        stereo = np.random.randn(SAMPLE_RATE, 2).astype(np.float32)
        wav = tmp_path / "stereo.wav"
        _write_wav(wav, stereo, SAMPLE_RATE)
        loaded = load_audio_16k_mono(wav)
        assert loaded.ndim == 1
        assert len(loaded) == SAMPLE_RATE


# ------------------- write_chunk_wav ---------------------------------------


class TestWriteChunkWav:
    def test_write_and_read_roundtrip(self, tmp_path: Path):
        audio = np.linspace(-0.5, 0.5, SAMPLE_RATE, dtype=np.float32)
        out = tmp_path / "chunk.wav"
        write_chunk_wav(audio, out)
        assert out.exists()
        import soundfile as sf

        loaded, sr = sf.read(str(out))
        assert sr == SAMPLE_RATE
        assert len(loaded) == SAMPLE_RATE

    def test_creates_parent_dir(self, tmp_path: Path):
        audio = np.zeros(SAMPLE_RATE // 10, dtype=np.float32)
        out = tmp_path / "nested" / "dir" / "chunk.wav"
        write_chunk_wav(audio, out)
        assert out.exists()


# ------------------- build_non_speech_manifest_entry -----------------------


class TestBuildNonSpeechManifestEntry:
    def test_all_required_fields_present(self):
        entry = build_non_speech_manifest_entry(
            relative_path="ja_non_speech_esc50/clapping_1_chunk0.wav",
            duration_sec=1.5,
            subtype="clapping",
            source_dataset="esc50",
            source_file="1-100032-A-22.wav",
            source_license="CC BY-NC 4.0",
            language="ja",
        )
        assert entry["path"] == "ja_non_speech_esc50/clapping_1_chunk0.wav"
        assert entry["label"] == "non_speech"
        assert entry["language"] == "ja"
        assert entry["subtype"] == "clapping"
        assert entry["source_dataset"] == "esc50"
        assert entry["source_file"] == "1-100032-A-22.wav"
        assert entry["source_license"] == "CC BY-NC 4.0"
        assert entry["duration_sec"] == 1.5
        assert entry["end_sec"] == 1.5
        assert entry["start_sec"] == 0.0

    def test_transcribed_text_empty(self):
        entry = build_non_speech_manifest_entry(
            relative_path="p", duration_sec=1.0,
            subtype="hvac", source_dataset="musan",
            source_file="noise-free-sound-0100.wav",
            source_license="CC BY 4.0",
        )
        assert entry["transcribed_text"] == ""
        assert entry["transcribed_text_kana"] == ""
        assert entry["alignment_score"] == 0.0
        assert entry["alignment_score_kana"] == 0.0

    def test_language_defaults_to_ja(self):
        entry = build_non_speech_manifest_entry(
            relative_path="p", duration_sec=1.0,
            subtype="engine", source_dataset="esc50",
            source_file="x.wav", source_license="CC BY-NC 4.0",
        )
        assert entry["language"] == "ja"


# ------------------- upsert_manifest_entries -------------------------------


class TestUpsertManifestEntries:
    def _write_manifest(self, path: Path, entries: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    def test_add_to_empty_manifest(self, tmp_path: Path):
        manifest = tmp_path / "manifest.jsonl"
        new = [build_non_speech_manifest_entry(
            "esc50/a.wav", 1.5, "applause", "esc50", "1.wav", "CC BY-NC 4.0"
        )]
        added, updated, removed = upsert_manifest_entries(manifest, new)
        assert (added, updated, removed) == (1, 0, 0)
        lines = manifest.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0])["path"] == "esc50/a.wav"

    def test_upsert_updates_existing_path(self, tmp_path: Path):
        manifest = tmp_path / "manifest.jsonl"
        self._write_manifest(manifest, [
            {"path": "esc50/a.wav", "label": "non_speech", "subtype": "old"},
        ])
        new = [build_non_speech_manifest_entry(
            "esc50/a.wav", 2.0, "new", "esc50", "1.wav", "CC BY-NC 4.0"
        )]
        added, updated, removed = upsert_manifest_entries(manifest, new)
        assert (added, updated, removed) == (0, 1, 0)
        lines = manifest.read_text(encoding="utf-8").strip().split("\n")
        assert json.loads(lines[0])["subtype"] == "new"

    def test_add_alongside_existing_different_path(self, tmp_path: Path):
        manifest = tmp_path / "manifest.jsonl"
        self._write_manifest(manifest, [
            {"path": "ja_clean/x.wav", "label": "speech", "language": "ja"},
        ])
        new = [build_non_speech_manifest_entry(
            "esc50/y.wav", 1.5, "dog", "esc50", "d.wav", "CC BY-NC 4.0"
        )]
        added, updated, removed = upsert_manifest_entries(manifest, new)
        assert (added, updated, removed) == (1, 0, 0)
        lines = manifest.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

    def test_force_removes_matching_source_dataset(self, tmp_path: Path):
        manifest = tmp_path / "manifest.jsonl"
        self._write_manifest(manifest, [
            {"path": "ja_clean/x.wav", "label": "speech"},
            build_non_speech_manifest_entry(
                "esc50/old1.wav", 1.5, "rain", "esc50", "r1.wav", "CC BY-NC 4.0"
            ),
            build_non_speech_manifest_entry(
                "esc50/old2.wav", 1.5, "dog", "esc50", "d1.wav", "CC BY-NC 4.0"
            ),
            build_non_speech_manifest_entry(
                "musan/n1.wav", 1.5, "hvac", "musan", "n1.wav", "CC BY 4.0"
            ),
        ])
        new = [build_non_speech_manifest_entry(
            "esc50/fresh.wav", 1.5, "engine", "esc50", "e1.wav", "CC BY-NC 4.0"
        )]
        added, updated, removed = upsert_manifest_entries(
            manifest, new, force=True, source_dataset_filter="esc50"
        )
        assert (added, updated, removed) == (1, 0, 2)
        lines = manifest.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3  # 1 speech + 1 musan + 1 new esc50
        paths = {json.loads(l)["path"] for l in lines}
        assert paths == {"ja_clean/x.wav", "musan/n1.wav", "esc50/fresh.wav"}

    def test_missing_path_field_raises(self, tmp_path: Path):
        manifest = tmp_path / "manifest.jsonl"
        with pytest.raises(ValueError, match="missing 'path'"):
            upsert_manifest_entries(manifest, [{"label": "non_speech"}])

    def test_forward_compat_preserves_existing_extra_fields(self, tmp_path: Path):
        manifest = tmp_path / "manifest.jsonl"
        self._write_manifest(manifest, [
            {
                "path": "ja_clean/x.wav",
                "label": "speech",
                "language": "ja",
                "alignment_score_kana": 0.85,  # PR-γ field
                "custom_future_field": "value",
            },
        ])
        new = [build_non_speech_manifest_entry(
            "esc50/y.wav", 1.5, "dog", "esc50", "d.wav", "CC BY-NC 4.0"
        )]
        upsert_manifest_entries(manifest, new)
        entries = {
            json.loads(l)["path"]: json.loads(l)
            for l in manifest.read_text(encoding="utf-8").strip().split("\n")
        }
        # Existing entry preserved with all its fields
        assert entries["ja_clean/x.wav"]["alignment_score_kana"] == 0.85
        assert entries["ja_clean/x.wav"]["custom_future_field"] == "value"


# ------------------- download_dataset (dev-only path) ----------------------


class TestDownloadDataset:
    def test_skip_when_dest_exists(self, tmp_path: Path):
        dest = tmp_path / "already.zip"
        dest.write_bytes(b"pre-existing content")
        download_dataset("https://example.invalid/never-fetched", dest)
        # No exception, no re-download (URL is invalid but we skip)
        assert dest.read_bytes() == b"pre-existing content"

    def test_hash_mismatch_raises(self, tmp_path: Path, monkeypatch):
        dest = tmp_path / "fake.zip"

        def fake_urlretrieve(url, path):
            Path(path).write_bytes(b"fake content")

        monkeypatch.setattr(
            "benchmarks.confidence_calibration._augment_common.urllib.request.urlretrieve",
            fake_urlretrieve,
        )
        wrong_hash = "0" * 64
        with pytest.raises(ValueError, match="hash mismatch"):
            download_dataset(
                "https://example.invalid/x.zip", dest,
                expected_sha256=wrong_hash,
            )
