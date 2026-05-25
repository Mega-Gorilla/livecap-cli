"""Speaker-embedding benchmark runner.

Pipeline:
1. Load the (16 kHz mono) source audio and VAD-segment it once (shared across
   all backends) using the project's Silero VAD.
2. For each backend: load model (measure load time + GPU model memory), warm
   up, then extract an embedding per segment while measuring per-segment
   latency, RTF, GPU peak and RAM peak.
3. Compute label-free separability (KMeans(2) + silhouette) and target cosine
   similarity distribution.
4. Optionally measure ASR co-residency (Parakeet + backend GPU footprint).
5. Write JSON + Markdown + console reports.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from benchmarks.common.metrics import GPUMemoryTracker, calculate_rtf

from .factory import create_embedding_backend, get_backend_info
from .metrics import percentile, separability, target_similarity_stats
from .reports import SpeakerBenchmarkReporter, SpeakerBenchmarkResult

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


@dataclass
class SpeakerBenchmarkConfig:
    """Configuration for the speaker benchmark."""

    backends: list[str] = field(default_factory=lambda: ["titanet"])
    device: str = "cuda"
    input_path: Path | None = None
    target_enroll_path: Path | None = None
    output_dir: Path | None = None
    min_segment_s: float = 0.3
    max_segments: int | None = None
    coresidency: bool = False
    coresidency_engine: str = "parakeet"
    # Run each backend in its own subprocess. Prevents global-state collisions
    # between ML toolkits (e.g. SpeechBrain ECAPA breaking pyannote's
    # SpeechBrain-backed wespeaker) and gives each backend a clean CUDA context.
    isolate: bool = True
    # Per-segment ASR transcript export (for manual verification of clustering).
    # asr_engine=None disables it. The audio is Japanese; Parakeet-JA (NeMo,
    # already installed) is the default. ReazonSpeech is selectable.
    asr_engine: str | None = "parakeet_ja"
    language: str = "ja"


class SpeakerBenchmarkRunner:
    """Runs the speaker-embedding benchmark across backends."""

    def __init__(self, config: SpeakerBenchmarkConfig) -> None:
        self.config = config
        self.gpu = GPUMemoryTracker()
        project_root = Path(__file__).resolve().parents[2]
        self.output_dir = config.output_dir or (project_root / "benchmark_results")
        self.reporter = SpeakerBenchmarkReporter(
            device=config.device,
            audio_source=str(config.input_path) if config.input_path else "",
        )
        self._segments: list[np.ndarray] = []
        self._spans: list[tuple[float, float]] = []
        self._audio_duration: float = 0.0
        # Per-segment detail of the most recent backend: idx/start/end/cluster/target_sim.
        self._detail: list[dict] = []

    # --- audio + segmentation -----------------------------------------

    def _load_audio(self) -> np.ndarray:
        import soundfile as sf

        path = self.config.input_path
        if path is None or not Path(path).exists():
            raise FileNotFoundError(
                f"Source audio not found: {path}. Run "
                "scripts/prepare_speaker_benchmark.py first, or pass --input."
            )
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype(np.float32)
        if sr != SAMPLE_RATE:
            from math import gcd

            from scipy import signal

            g = gcd(sr, SAMPLE_RATE)
            audio = signal.resample_poly(audio, SAMPLE_RATE // g, sr // g).astype(
                np.float32
            )
        return audio

    def _segment_audio(self, audio: np.ndarray) -> list[np.ndarray]:
        from benchmarks.vad.factory import create_vad

        vad = create_vad("silero")
        spans = vad.process_audio(audio, SAMPLE_RATE)

        segments: list[np.ndarray] = []
        kept_spans: list[tuple[float, float]] = []
        for start_s, end_s in spans:
            if end_s - start_s < self.config.min_segment_s:
                continue
            start = max(0, int(start_s * SAMPLE_RATE))
            end = min(len(audio), int(end_s * SAMPLE_RATE))
            seg = audio[start:end]
            if seg.size > 0:
                segments.append(seg)
                kept_spans.append((start_s, end_s))

        if self.config.max_segments is not None:
            segments = segments[: self.config.max_segments]
            kept_spans = kept_spans[: self.config.max_segments]

        # Spans aligned with segments (for transcript/cluster export).
        self._spans = kept_spans
        return segments

    # --- ASR transcripts (backend-independent) ------------------------

    def _transcribe_segments(self, segments: list[np.ndarray]) -> list[str] | None:
        """Transcribe each segment once (Parakeet-JA / ReazonSpeech) for export.

        Returns one transcript per segment, or None if ASR is disabled/unavailable.
        """
        if not self.config.asr_engine:
            return None
        try:
            from benchmarks.common.engines import BenchmarkEngineManager

            mgr = BenchmarkEngineManager()
            logger.info(
                "Loading ASR engine for transcripts: %s (%s)",
                self.config.asr_engine, self.config.language,
            )
            engine = mgr.get_engine(
                self.config.asr_engine,
                device=self.config.device,
                language=self.config.language,
            )
        except Exception as e:
            logger.warning(
                "ASR engine '%s' unavailable (%s); skipping transcripts.",
                self.config.asr_engine, e,
            )
            return None

        texts: list[str] = []
        for seg in segments:
            try:
                txt, _ = engine.transcribe(seg, SAMPLE_RATE)
                texts.append((txt or "").strip())
            except Exception as e:  # one bad segment shouldn't abort all
                logger.debug("ASR failed on a segment: %s", e)
                texts.append("")
        try:
            mgr.clear_cache()
        except Exception:
            pass
        return texts

    # --- per-segment export -------------------------------------------

    def _write_transcripts(
        self, result_dir: Path, transcripts: list[str]
    ) -> None:
        """Write the shared per-segment transcript (timestamps + text)."""
        import json

        rows = [
            {
                "idx": i,
                "start": round(self._spans[i][0], 2) if i < len(self._spans) else None,
                "end": round(self._spans[i][1], 2) if i < len(self._spans) else None,
                "text": transcripts[i] if i < len(transcripts) else "",
            }
            for i in range(len(transcripts))
        ]
        (result_dir / "transcripts.json").write_text(
            json.dumps(
                {"engine": self.config.asr_engine, "language": self.config.language, "segments": rows},
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        lines = [
            f"# Segment transcripts ({self.config.asr_engine}, {self.config.language})",
            "",
            "| # | start | end | transcript |",
            "|---|------|-----|------------|",
        ]
        for r in rows:
            lines.append(
                f"| {r['idx']} | {r['start']} | {r['end']} | {_md_cell(r['text'])} |"
            )
        (result_dir / "transcripts.md").write_text("\n".join(lines), encoding="utf-8")

    def _write_segment_report(
        self,
        result_dir: Path,
        backend_id: str,
        detail: list[dict],
        transcripts: list[str] | None,
    ) -> None:
        """Write per-backend segment report: timestamp + cluster + sim + transcript."""
        import json

        rows = []
        for d in detail:
            idx = d.get("idx")
            text = (
                transcripts[idx]
                if transcripts is not None and idx is not None and idx < len(transcripts)
                else ""
            )
            rows.append({**d, "text": text})

        (result_dir / f"segments_{backend_id}.json").write_text(
            json.dumps({"backend": backend_id, "segments": rows}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        lines = [
            f"# Segments — {backend_id}",
            "",
            "cluster = KMeans(2) assignment; sim = cosine vs target (larger-cluster centroid).",
            "",
            "| # | start | end | cluster | sim | transcript |",
            "|---|------|-----|---------|-----|------------|",
        ]
        for r in rows:
            lines.append(
                f"| {r['idx']} | {r.get('start')} | {r.get('end')} | "
                f"{r.get('cluster')} | {r.get('target_sim')} | {_md_cell(r['text'])} |"
            )
        (result_dir / f"segments_{backend_id}.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    # --- per-backend measurement --------------------------------------

    def _benchmark_backend(self, backend_id: str) -> SpeakerBenchmarkResult:
        info = get_backend_info(backend_id)
        result = SpeakerBenchmarkResult(
            backend=backend_id,
            device=self.config.device,
            num_segments=len(self._segments),
            audio_duration_s=self._audio_duration,
        )

        # Instantiate + load (graceful skip on missing deps / gated access).
        try:
            backend = create_embedding_backend(backend_id)
        except Exception as e:
            result.status = "failed"
            result.detail = f"create failed: {e}"
            return result

        if self.gpu.available:
            gpu_before = self.gpu.get_allocated() or 0.0
        else:
            gpu_before = 0.0

        load_start = time.perf_counter()
        try:
            backend.load(self.config.device)
        except (ImportError, RuntimeError) as e:
            result.status = "skipped"
            result.detail = str(e).splitlines()[0]
            logger.warning("Skipping %s: %s", backend_id, result.detail)
            return result
        except Exception as e:  # pragma: no cover - unexpected
            result.status = "failed"
            result.detail = f"load failed: {e}"
            logger.warning("Backend %s failed to load: %s", backend_id, e)
            return result
        result.load_s = time.perf_counter() - load_start

        if self.gpu.available:
            result.gpu_model_mb = max(0.0, (self.gpu.get_allocated() or 0.0) - gpu_before)

        # Warm-up.
        try:
            backend.extract_embedding(self._segments[0], SAMPLE_RATE)
        except Exception as e:
            result.status = "failed"
            result.detail = f"warm-up failed: {e}"
            return result

        if self.gpu.available:
            self.gpu.reset_peak()

        # Measure extraction over all segments.
        import tracemalloc

        embeddings: list[np.ndarray] = []
        latencies_ms: list[float] = []
        tracemalloc.start()
        total_start = time.perf_counter()
        try:
            for seg in self._segments:
                t0 = time.perf_counter()
                emb = backend.extract_embedding(seg, SAMPLE_RATE)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                embeddings.append(np.asarray(emb, dtype=np.float64))
        except Exception as e:
            tracemalloc.stop()
            result.status = "failed"
            result.detail = f"extraction failed: {e}"
            return result
        total_elapsed = time.perf_counter() - total_start
        _, ram_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result.ram_peak_mb = ram_peak / (1024 * 1024)
        result.embedding_dim = int(embeddings[0].shape[0]) if embeddings else None
        result.embed_latency_ms_mean = float(np.mean(latencies_ms)) if latencies_ms else None
        result.embed_latency_ms_p50 = percentile(latencies_ms, 50)
        result.embed_latency_ms_p95 = percentile(latencies_ms, 95)
        result.rtf = calculate_rtf(self._audio_duration, total_elapsed)
        if self.gpu.available:
            result.gpu_peak_mb = self.gpu.get_peak()

        # Label-free separability + target similarity.
        emb_matrix = np.vstack(embeddings)
        sep = separability(emb_matrix, n_clusters=2)
        result.silhouette = sep["silhouette"]
        result.cluster_sizes = sep["cluster_sizes"]

        labels = sep["labels"]
        target = self._resolve_target(backend, emb_matrix, labels)
        per_seg_sim: list[float | None] = [None] * len(embeddings)
        if target is not None:
            stats = target_similarity_stats(emb_matrix, target, labels or None)
            result.target_sim_mean = stats["mean"]
            result.target_sim_std = stats["std"]
            result.target_cluster_mean_gap = stats.get("cluster_mean_gap")
            from .metrics import cosine_similarity

            per_seg_sim = [cosine_similarity(e, target) for e in emb_matrix]

        # Per-segment detail (idx/start/end/cluster/target_sim) for export.
        self._detail = []
        for i in range(len(embeddings)):
            start, end = self._spans[i] if i < len(self._spans) else (None, None)
            self._detail.append(
                {
                    "idx": i,
                    "start": round(start, 2) if start is not None else None,
                    "end": round(end, 2) if end is not None else None,
                    "cluster": int(labels[i]) if i < len(labels) else None,
                    "target_sim": (
                        round(per_seg_sim[i], 4) if per_seg_sim[i] is not None else None
                    ),
                }
            )

        # Optional ASR co-residency.
        if self.config.coresidency and self.gpu.available:
            self._measure_coresidency(result)

        # Cleanup backend / GPU before next backend.
        del backend
        gc.collect()
        if self.gpu.available:
            import torch

            torch.cuda.empty_cache()

        return result

    def _resolve_target(
        self, backend, emb_matrix: np.ndarray, labels: list[int]
    ) -> np.ndarray | None:
        """Target embedding: explicit enrollment clip, else larger-cluster centroid."""
        if self.config.target_enroll_path and Path(self.config.target_enroll_path).exists():
            import soundfile as sf

            audio, sr = sf.read(str(self.config.target_enroll_path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1).astype(np.float32)
            try:
                return np.asarray(
                    backend.extract_embedding(audio, sr), dtype=np.float64
                )
            except Exception as e:  # pragma: no cover
                logger.warning("Target enroll embedding failed: %s", e)

        if labels:
            labels_arr = np.asarray(labels)
            # Larger cluster as the pseudo-target speaker.
            counts = np.bincount(labels_arr)
            target_cluster = int(np.argmax(counts))
            return emb_matrix[labels_arr == target_cluster].mean(axis=0)
        return None

    def _measure_coresidency(self, result: SpeakerBenchmarkResult) -> None:
        """Load the ASR engine alongside and record combined GPU footprint."""
        try:
            from benchmarks.common.engines import BenchmarkEngineManager

            mgr = BenchmarkEngineManager()
            self.gpu.reset_peak()
            mgr.get_engine(
                self.config.coresidency_engine,
                device=self.config.device,
                language=self.config.language,
            )
            result.coresidency_combined_gpu_mb = self.gpu.get_allocated()
            result.coresidency_oom = False
            mgr.clear_cache()
        except Exception as e:  # pragma: no cover - hardware dependent
            msg = str(e).lower()
            result.coresidency_oom = "out of memory" in msg or "oom" in msg
            logger.warning("Co-residency measurement failed: %s", e)

    # --- orchestration -------------------------------------------------

    def run(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = self.output_dir / f"speaker_{timestamp}"
        result_dir.mkdir(parents=True, exist_ok=True)

        if self.config.isolate:
            # Parent transcribes once (backend-independent); workers extract
            # embeddings in isolation and return per-segment cluster/sim.
            transcripts = self._prepare_transcripts(result_dir)
            for backend_id in self.config.backends:
                logger.info("Benchmarking backend (isolated): %s", backend_id)
                result, detail = self._run_isolated(backend_id)
                self.reporter.add_result(result)
                if detail:
                    self._write_segment_report(
                        result_dir, backend_id, detail, transcripts
                    )
        else:
            audio = self._load_audio()
            self._audio_duration = len(audio) / SAMPLE_RATE
            logger.info("Loaded audio: %.1fs", self._audio_duration)
            self._segments = self._segment_audio(audio)
            logger.info("VAD produced %d segments", len(self._segments))
            if not self._segments:
                raise RuntimeError("No speech segments detected in source audio")
            transcripts = self._transcribe_segments(self._segments)
            if transcripts is not None:
                self._write_transcripts(result_dir, transcripts)
            for backend_id in self.config.backends:
                logger.info("Benchmarking backend: %s", backend_id)
                self.reporter.add_result(self._benchmark_backend(backend_id))
                if transcripts is not None and self._detail:
                    self._write_segment_report(
                        result_dir, backend_id, self._detail, transcripts
                    )

        self.reporter.save_json(result_dir / "results.json")
        self.reporter.save_markdown(result_dir / "summary.md")
        self.reporter.to_console()
        logger.info("Results saved to: %s", result_dir)
        return result_dir

    def _prepare_transcripts(self, result_dir: Path) -> list[str] | None:
        """Parent-side: segment once + transcribe once (for isolated mode)."""
        if not self.config.asr_engine:
            return None
        audio = self._load_audio()
        self._audio_duration = len(audio) / SAMPLE_RATE
        self._segments = self._segment_audio(audio)
        if not self._segments:
            raise RuntimeError("No speech segments detected in source audio")
        transcripts = self._transcribe_segments(self._segments)
        if transcripts is not None:
            self._write_transcripts(result_dir, transcripts)
        return transcripts

    def _run_isolated(
        self, backend_id: str
    ) -> tuple[SpeakerBenchmarkResult, list[dict]]:
        """Benchmark one backend in a child process; return (result, per-seg detail)."""
        import json
        import subprocess
        import sys
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "result.json"
            cmd = [
                sys.executable, "-m", "benchmarks.speaker",
                "--backend", backend_id,
                "--device", self.config.device,
                "--min-segment-s", str(self.config.min_segment_s),
                "--worker-out", str(out),
                "--output-dir", td,
                "--no-asr",  # parent already transcribed; workers skip ASR
            ]
            if self.config.input_path:
                cmd += ["--input", str(self.config.input_path)]
            if self.config.target_enroll_path:
                cmd += ["--target-enroll", str(self.config.target_enroll_path)]
            if self.config.max_segments is not None:
                cmd += ["--max-segments", str(self.config.max_segments)]
            if self.config.coresidency:
                cmd += ["--coresidency", "--coresidency-engine", self.config.coresidency_engine]

            proc = subprocess.run(cmd, capture_output=True, text=True)
            if out.exists():
                payload = json.loads(out.read_text(encoding="utf-8"))
                results = payload.get("results", [])
                detail = payload.get("detail", [])
                if results:
                    return SpeakerBenchmarkResult(**results[0]), detail

            tail = (proc.stderr or "").strip().splitlines()
            reason = tail[-1][:280] if tail else f"subprocess exit {proc.returncode}"
            logger.warning("Isolated backend %s failed: %s", backend_id, reason)
            return (
                SpeakerBenchmarkResult(
                    backend=backend_id,
                    device=self.config.device,
                    status="failed",
                    detail=f"subprocess: {reason}",
                ),
                [],
            )


def _md_cell(text: str) -> str:
    """Sanitize text for a Markdown table cell (escape pipes / newlines)."""
    return (text or "").replace("|", "\\|").replace("\n", " ").strip()


__all__ = ["SpeakerBenchmarkConfig", "SpeakerBenchmarkRunner"]
