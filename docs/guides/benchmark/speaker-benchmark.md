# Speaker Embedding Benchmark

Pre-implementation spike for the **SpeakerGate** feature (issue #287). Measures
GPU / memory / latency / label-free separability of speaker-embedding backends so
the production backend can be chosen on evidence rather than guesswork.

> This is a benchmark spike, **not** the SpeakerGate implementation. The gate
> itself (`livecap_cli/diarization/`) is tracked in issue #287.

## What it measures

For each backend, over the speech segments of a conversation recording:

| Metric | Meaning |
|---|---|
| `load_s` | Model load time |
| `embed_latency_ms_p50/p95/mean` | Per-segment embedding latency |
| `rtf` | Embedding-only real-time factor (total embed time / audio duration) |
| `gpu_model_mb` / `gpu_peak_mb` | GPU memory after load / inference peak |
| `ram_peak_mb` | Python RAM peak during extraction |
| `silhouette` | KMeans(2) cosine silhouette — **label-free** 2-speaker separability |
| `target_sim_*` | Cosine similarity distribution vs a target embedding |

Accuracy here is intentionally **label-free** (no manual speaker labels): a high
silhouette means the embedding space cleanly separates the two speakers, which is
the property a SpeakerGate relies on.

### Per-segment transcripts (manual verification)

To let you eyeball whether the clustering actually matches speaker turns, the
runner also transcribes each VAD segment **once** (backend-independent) and emits,
per backend, a `segments_<backend>.md` / `.json` with `start–end | cluster | sim |
transcript`, plus a shared `transcripts.md`. ASR defaults to **`parakeet_ja`**
(NeMo, already installed); **`reazonspeech`** is selectable. Disable with `--no-asr`.

> Note: speaker-embedding models here are English/VoxCeleb-trained applied to
> **Japanese** audio. Embeddings are largely language-independent, so the
> comparison is valid, but absolute separability and any thresholds must be
> re-calibrated on Japanese data for the production gate.

## Backends

| id | model | install | license |
|---|---|---|---|
| `titanet` | NeMo TitaNet-L | `engines-nemo` (already required by Parakeet) | CC-BY-4.0 (attribution) |
| `ecapa` | SpeechBrain ECAPA | `uv pip install -e ".[speaker-speechbrain]"` | Apache-2.0 toolkit |
| `pyannote` | pyannote/embedding | `uv pip install -e ".[speaker-pyannote]"` + HF token | MIT, **gated** |
| `mock` | deterministic FFT features | none (tests only) | n/a |

`pyannote/embedding` is gated: accept the terms at
<https://huggingface.co/pyannote/embedding> and set `HF_TOKEN` (or run
`huggingface-cli login`). If the token is missing, the backend is **skipped**
gracefully and the other backends still run.

## Data

The reference recording is **git-unshareable** (a 2-person conversation stream
clip) and lives under `benchmarks/speaker/data/` which is **gitignored**.

```bash
# Fetch + cut + resample the 10-minute segment into 16 kHz mono wav (local only).
uv run python scripts/prepare_speaker_benchmark.py

# Or point at any local wav you already have:
uv run python scripts/prepare_speaker_benchmark.py --input /path/to/clip.wav
```

The audio is for **local evaluation only** — never commit or redistribute it.

## Running

```bash
uv run python -m benchmarks.speaker --list-backends
uv run python -m benchmarks.speaker --backend titanet ecapa pyannote --device cuda
uv run python -m benchmarks.speaker --backend titanet --device cpu --max-segments 50

# Choose ASR engine for transcripts, or disable it:
uv run python -m benchmarks.speaker --backend titanet --asr-engine reazonspeech
uv run python -m benchmarks.speaker --backend titanet --no-asr

# Also report ASR (Parakeet) + backend combined GPU footprint:
uv run python -m benchmarks.speaker --backend titanet --coresidency
```

Each backend runs in its **own subprocess** by default (`--no-isolate` to disable)
to avoid ML-toolkit global-state collisions (e.g. SpeechBrain ECAPA vs pyannote's
SpeechBrain-backed model) and to give each a clean CUDA context.

Results are written to `benchmark_results/speaker_<timestamp>/`:
`results.json`, `summary.md`, `transcripts.md/json`, and per-backend
`segments_<backend>.md/json` — all printed/summarized to the console.

## Tests

```bash
uv run pytest tests/benchmark_tests/speaker -v
```

Tests use synthetic 2-speaker audio + the `mock` backend and require no heavy
models or the gitignored data, so they run in CI.
