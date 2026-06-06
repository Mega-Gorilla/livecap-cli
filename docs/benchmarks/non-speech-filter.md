# Non-Speech Filter Evaluation Harness (Issue #295 PR-0)

The non-speech filter evaluation harness measures how the production pipeline
(NoiseGate + VAD + EnergyGate, plus future Phase 1 layers) responds to
applause and other non-speech audio — the failure mode tracked in Issue
[#295] and livecap-gui#331.

It has two faces:

- **CI baseline tests** under `tests/integration/non_speech_filter/`
  that persist per-backend metric snapshots, and
- An **ad-hoc benchmark runner** under `benchmarks/non_speech_filter/`
  that drives synthetic and real-audio corpora through one or more
  ASR engines for richer analysis (e.g., engine-driven hallucination
  measurement).

---

## What it measures

| Metric | Definition |
|---|---|
| `false_asr_trigger_rate` | Fraction of **negative** items (applause, music, ...) that reached `engine.transcribe()`. Lower is better. |
| `speech_recall` | Fraction of **positive** items (speech-like) that reached `engine.transcribe()`. Higher is better. |
| `short_utterance_recall` | Recall restricted to short responses (はい / OK / うん). The single most fragile metric — Phase 1 PRs must not regress it. |
| `non_empty_hallucination_rate` | (engine runs only) Fraction of negative items where the engine produced non-empty text. Reflects how the engine reacts when the gates fail. |
| `added_latency_p50_ms` / `_p95_ms` | Pipeline wall-clock per corpus item. Tracks Phase 1 layer cost. |

All metrics are computed per `(backend, corpus)` and persisted to JSON.

---

## Corpora

### Synthetic (always available)

`tests/integration/non_speech_filter/corpus.py` constructs 13 deterministic
items at 16 kHz mono float32:

- **negative (8)**: applause_single, applause_burst, applause_distant,
  keyboard_taps, door_close, cough, music_chord, silence_amplified.
- **positive (5)**: normal_speech, short_utterance_hai,
  short_utterance_ok, post_applause_speech, overlapping_applause_speech.

The synthesisers are intentionally simple (numpy + scipy only) and seeded.
They are good enough to differentiate the three VAD backends but **cannot
fully model real human speech** — see "Known limitations" below.

### Real (optional)

Real audio fixtures can be supplied by setting

```
LIVECAP_NON_SPEECH_CORPUS_DIR=/path/to/corpus
```

The directory must contain:

- `manifest.json` — a list of entries with `file`, `label`, `kind`
  (`"negative"` or `"positive"`), and optional `is_short_utterance`.
- The referenced audio files (any sample rate; stereo gets mixed to mono
  and the result is resampled to 16 kHz).

Example manifest:

```json
[
  {"file": "esc50/applause_01.wav", "label": "applause_01", "kind": "negative"},
  {"file": "common_voice/short_hai.wav", "label": "short_hai", "kind": "positive", "is_short_utterance": true}
]
```

Recommended sources:

- ESC-50 (`Applause`, `Clapping`, `Door`, `Cough`, `Keyboard typing`, …) —
  CC-BY-4.0.
- Common Voice short utterances or FSD50K for non-speech events.

Real audio is **not** vendored in git; users must point the env var at a
locally prepared directory.

---

## Running the CI baseline tests

```bash
# All three backends × synthetic corpus
uv run pytest tests/integration/non_speech_filter/ -v -m evaluation_harness

# Only one backend
uv run pytest tests/integration/non_speech_filter/ -v -m evaluation_harness -k silero
```

Each `test_baseline_synthetic_corpus[<backend>]` run writes
`tests/integration/non_speech_filter/baselines/<backend>.json`. Subsequent
Phase 1 PRs (B/C/A) read these snapshots to detect regression.

`test_baseline_real_corpus` is skipped unless `LIVECAP_NON_SPEECH_CORPUS_DIR`
is set; on success it writes `<backend>.real.json` alongside the synthetic
snapshot.

`test_baseline_hallucination_marker_present` is gated by both
`@pytest.mark.engine_smoke` and `LIVECAP_ENABLE_HALLUCINATION_EVAL=1`; it is
a sanity probe for the engine-run code path, not a real engine measurement
(that lives in the benchmark runner).

---

## Running the ad-hoc benchmark runner

```bash
# Synthetic only, default backend (silero)
uv run python -m benchmarks.non_speech_filter --mode quick

# All three backends, multiple runs (recommended for noise averaging)
uv run python -m benchmarks.non_speech_filter \
    --mode standard --backend silero,tenvad,webrtc

# Real audio + real engine (e.g., whispers2t) on GPU
uv run python -m benchmarks.non_speech_filter \
    --backend silero,tenvad \
    --engine whispers2t \
    --corpus-dir ~/data/non_speech_corpus \
    --device cuda --runs 3
```

Reports land in `benchmark_results/non_speech_filter/`:

- `non_speech_filter_<timestamp>.json` — full per-record snapshot.
- `non_speech_filter_<timestamp>.md` — summary table per
  `(backend, engine, corpus)`.

Hallucination measurement is automatic whenever a non-mock engine is
selected: the runner wraps the real engine in `InstrumentedEngine`, which
delegates `transcribe()` to the underlying model and records call counts +
output text so the metric layer sees the same surface it sees for
`MockEngine`. Without this wrapper, real engines would silently report
`non_empty_hallucination_rate = 0` even when they hallucinate.

---

## Stable regression invariants

`tests/integration/non_speech_filter/test_baseline.py::test_baseline_regression_thresholds`
asserts per-backend tolerance bands so Phase 1 PR-B/C/A cannot silently
regress metrics:

| Backend | `false_asr_trigger_rate ≤` | `speech_recall ≥` | `short_utterance_recall ≥` |
|---|---|---|---|
| silero | 0.20 | — (synthetic limitation) | — |
| tenvad | 0.50 | 0.80 | 0.80 |
| webrtc | 0.90 | 0.80 | 0.80 |

The bands are deliberately loose so platform-specific noise (e.g. CPU
versus GPU, Windows versus Linux) does not cause CI flakiness. Phase 1 PRs
that improve the metrics should tighten the bands alongside their feature
work and document the new floor in the CHANGELOG.

---

## Pipeline error handling

`evaluate_pipeline()` accepts `fail_fast` (default `True`):

- `True` — surface any pipeline exception immediately. Used by the pytest
  baseline tests so a real bug fails CI loudly.
- `False` — capture the exception in `per_label[label]['error']`, treat the
  item as not triggering ASR, and continue with the rest of the corpus.
  Used by the benchmark runner so a single environmental glitch (transient
  CUDA error, sherpa-onnx warm-up issue) does not bin the whole report.

---

## Baseline schema

`tests/integration/non_speech_filter/baselines/<backend>.json`:

```json
{
  "schema_version": "1",
  "backend_name": "silero",
  "totals": {"negative": 8, "positive": 5, "short_utterance": 2},
  "metrics": {
    "false_asr_trigger_rate": 0.0,
    "speech_recall": 0.0,
    "short_utterance_recall": 0.0,
    "non_empty_hallucination_rate": null,
    "added_latency_p50_ms": 5.9,
    "added_latency_p95_ms": 68.2
  },
  "per_label": { "applause_single": {"kind": "negative", "engine_calls": 0, ...}, ... }
}
```

Bumping `schema_version` must be matched by an update to
`REQUIRED_BASELINE_KEYS` in `metrics.py` and a CHANGELOG note.

---

## Known limitations

1. **Synthetic speech is not real speech.** Silero VAD (which is trained on
   real human speech) does not classify our synthetic speech proxy as
   speech, so the baseline shows `speech_recall = 0` for Silero. TenVAD
   and WebRTC are more permissive and recognise the proxy. For accurate
   Silero baselines, supply real audio via `LIVECAP_NON_SPEECH_CORPUS_DIR`.
2. **WebRTC outputs binary probabilities (0.0 / 1.0).** Hysteresis-based
   improvements in Phase 1 PR-C will be a no-op on WebRTC; only the
   duration-based cooldown applies there. Reports should be interpreted
   per backend, not pooled blindly.
3. **TenVAD has a permissive but limited license** (see warning emitted on
   import). The harness skips it gracefully when unavailable; the project
   LICENSE may need a follow-up addendum for redistribution scenarios.
4. **MockEngine measures only filter behaviour.** It cannot observe
   hallucination text — that requires a real engine via
   `--engine whispers2t` (or `parakeet_ja` etc.).
5. **Baselines are point estimates from one run by default.** Use
   `--runs 3` (or higher) in the benchmark runner for less noisy reporting.

---

## How Phase 1 PRs interact with this harness

| PR | What it must achieve against the baseline |
|---|---|
| PR-B (Layer 1: DSP transient detector) | Initially `observe`-only; baseline metrics unchanged. With `--transient-filter=on`, `false_asr_trigger_rate` should drop for TenVAD/WebRTC on `applause_*`, `keyboard_taps`, `door_close`, `cough`, while `short_utterance_recall` stays ≥ baseline. |
| PR-C (Layer 2: VADStateMachine cooldown extension) | Backend-aware: hysteresis only affects Silero / TenVAD; duration-based cooldown affects all three. Must reduce post-applause false triggers without dropping `post_applause_speech`. |
| PR-A (Layer 3 + 4: Confidence filter + Prompt reset) | Only measurable with a real engine. With `--engine whispers2t`, `non_empty_hallucination_rate` should approach 0 on the negative set; engine call counts for negatives may stay non-zero (Layer 3 is post-engine). |

See Issue [#295] for the full Phase 1 plan.

[#295]: https://github.com/Mega-Gorilla/livecap-cli/issues/295
