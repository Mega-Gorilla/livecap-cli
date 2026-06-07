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

#### Reference corpus used for Issue #295 baselines

The numbers quoted later in this document and in the
[Issue #295](https://github.com/Mega-Gorilla/livecap-cli/issues/295)
comments were measured against a **private 6-item Japanese real corpus**.
The audio cannot be redistributed, but the manifest schema, per-clip
metadata and per-backend behaviour are recorded here so that the figures
are interpretable even without access to the original WAVs.

```json
[
  {"file": "applause_5_claps.wav",            "label": "applause_5_claps",            "kind": "negative", "is_short_utterance": false},
  {"file": "desk_tap.wav",                    "label": "desk_tap",                    "kind": "negative", "is_short_utterance": false},
  {"file": "short_utterances_mixed.wav",      "label": "short_utterances_mixed",      "kind": "positive", "is_short_utterance": true},
  {"file": "normal_speech_neko.wav",          "label": "normal_speech_neko",          "kind": "positive", "is_short_utterance": false},
  {"file": "applause_then_speech.wav",        "label": "applause_then_speech",        "kind": "positive", "is_short_utterance": false},
  {"file": "overlapping_applause_speech.wav", "label": "overlapping_applause_speech", "kind": "positive", "is_short_utterance": false}
]
```

| Clip | Duration | Peak dBFS | RMS dBFS | Content |
|---|---|---|---|---|
| `applause_5_claps` | 8.55 s | -6.9 | -43.6 | 5 isolated claps spread across 8.5 s |
| `desk_tap` | 5.91 s | -18.2 | -46.6 | Knuckle tapping on a wooden desk |
| `short_utterances_mixed` | 8.13 s | -21.3 | -43.6 | Japanese「はい」「OK」「うん」「はいOKうん」 |
| `normal_speech_neko` | 15.59 s | -22.5 | -41.2 | One sentence from *I Am a Cat* (Sōseki) |
| `applause_then_speech` | 7.25 s | -4.4 | -42.7 | Applause burst followed by the sentence above |
| `overlapping_applause_speech` | 14.89 s | -7.4 | -42.0 | The full opening of *I Am a Cat*, applauding throughout |

##### Per-clip per-backend baseline (MockEngine, current pipeline)

`triggered` = at least one `engine.transcribe` call was made for the clip.

| Clip | kind | Silero triggered | TenVAD triggered | WebRTC triggered |
|---|---|---|---|---|
| applause_5_claps | negative | ✅ rejected | ✅ rejected | ✅ rejected |
| desk_tap | negative | ✅ rejected | ✅ rejected | ❌ **triggered** (1 call) |
| short_utterances_mixed | positive | ✅ triggered (4) | ✅ triggered (5) | ✅ triggered (4) |
| normal_speech_neko | positive | ✅ triggered (8) | ✅ triggered (10) | ✅ triggered (10) |
| applause_then_speech | positive | ✅ triggered (2) | ✅ triggered (4) | ✅ triggered (2) |
| overlapping_applause_speech | positive | ✅ triggered (6) | ✅ triggered (11) | ✅ triggered (12) |

##### Per-clip per-engine hallucination (real engine, fail_fast=False)

`hallucination` = engine returned a non-empty transcription for a negative
clip. Empty cells mean the gate rejected the clip first (no engine call).

| Clip | kind | whispers2t (large-v3) | parakeet_ja | reazonspeech |
|---|---|---|---|---|
| applause_5_claps (all backends) | negative | — | — | — |
| desk_tap (Silero / TenVAD) | negative | — | — | — |
| desk_tap (WebRTC) | negative | empty (no hallucination) | **hallucination** | **hallucination** |
| positive clips | positive | transcribed correctly across all backends | transcribed correctly across all backends | transcribed correctly across all backends |

This is why the "Reference baselines from the private corpus" table later
in the document shows `webrtc / parakeet_ja / real` and `webrtc /
reazonspeech / real` at 50 % `non_empty_hallucination_rate` while
whispers2t stays at 0 % — the engine's internal `no_speech` defence
absorbs the WebRTC gate leak for whispers2t but not for the other two.

#### Synthetic vs private real corpus — what each measures

| Corpus | Strengths | Limitations |
|---|---|---|
| Synthetic (always available) | Deterministic, redistributable, generated at test time, exposes the rapid-burst / overlap cases that the private corpus does not cover. | Silero VAD ignores the speech proxy by design (`speech_recall = 0`). False-trigger rates for non-Silero backends are **higher** than on natural audio because the synthetic burst is denser. |
| Private real corpus (6 clips above) | Reflects the actual `livecap-gui#331` failure mode. WebRTC × desk_tap is the single most informative cell. Real speech that Silero recognises. | Not redistributable, single speaker / one room, no rapid-burst applause (each clap is well-separated). |

Both are run in CI by `python -m benchmarks.non_speech_filter` whenever
`LIVECAP_NON_SPEECH_CORPUS_DIR` is set; the synthetic numbers are also
asserted by `test_baseline_regression_thresholds` to guarantee no silent
drift.

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

## Layer 1: DSP transient/applause detector (PR-B)

[Issue #295] PR-B adds an optional Layer 1 detector that consumes audio
*before* the VAD. Six per-frame DSP features
(`spectral_flatness`, `spectral_centroid_hz`, `zero_crossing_rate`,
`onset_strength`, `voiced_ratio`, `rms_db`) are AND-combined to flag
frames whose acoustic shape resembles rapid-burst applause. Three modes
are exposed:

| Mode | Audio effect | Telemetry |
|---|---|---|
| `off` (default) | — | Detector not constructed |
| `observe` | Audio passes through unchanged | Frame counters + per-feature pass tallies |
| `on` | Applause-flagged frames are zeroed out | Same telemetry plus an applause-frame count |

Reject default ON is intentionally out of scope for PR-B; calibration
follow-up uses the sweep harness below.

#### Streaming semantics (causal, no lookahead)

The detector processes audio frame-by-frame with a residual buffer so
feature computation stays continuous across chunked feeds. Telemetry
counters (`frames_processed`, `applause_frames`) therefore match between
a single full feed and a chunked feed of the same audio.

In `on` mode the masking is **causal**: it can only zero out the part of
a flagged frame that falls inside the current chunk. Frames that start
in the residual (i.e. inside audio already returned to the caller in a
previous call) cannot be retroactively muted. Chunked output is a
best-effort upper bound on the single-chunk result; the two outputs are
not bit-identical. A 1-frame lookahead delay would close the gap at the
cost of +32 ms latency; that enhancement is tracked separately and was
intentionally deferred to keep PR-B small.

### CLI surface

```bash
livecap-cli transcribe \
    --transient-filter observe \
    --transient-flatness-min 0.30 \
    --transient-centroid-min-hz 2500 \
    --transient-zcr-min 0.12 \
    --transient-onset-ratio 3.0 \
    --transient-voiced-max 0.25 \
    --transient-rms-min-db -35 \
    <input>
```

The benchmark CLI exposes the same flags so sweep data and production
runs share a single configuration model.

### Per-clip feature pass rates on the private real corpus

Run with `--transient-filter observe` and read the `pass_*` counters
from the detector's telemetry. Numbers are the fraction of frames that
crossed each feature's threshold in isolation (the AND result is
``applause%``).

| Clip | Kind | Frames | Applause % | Flatness % | Centroid % | ZCR % | Onset % | Voiced % | RMS % |
|---|---|---|---|---|---|---|---|---|---|
| `applause_5_claps` | neg | 533 | 0 | 1 | 7 | 12 | 12 | 38 | 2 |
| `desk_tap` | neg | 368 | 0 | 0 | 0 | 0 | 20 | 12 | 1 |
| `short_utterances_mixed` | pos (short) | 507 | 0 | 0 | 1 | 6 | 14 | 37 | 5 |
| `normal_speech_neko` | pos | 973 | 0 | 0 | 4 | 10 | 15 | 18 | 5 |
| `applause_then_speech` | pos | 452 | 0 | 0 | 3 | 8 | 12 | 31 | 3 |
| `overlapping_applause_speech` | pos | 929 | 0 | 2 | 6 | 16 | 17 | 19 | 2 |

Two findings drive the calibration follow-up:

1. **Real-clip RMS is too low for the default `--transient-rms-min-db -35`.**
   Only 1-5 % of frames in any clip pass the floor because the source
   audio sits at -41 to -46 dBFS RMS — the default was tuned for the
   synthetic rapid-burst case.
2. **`desk_tap` shows the opposite spectral profile to applause** — its
   spectral centroid is below 2500 Hz for 100 % of frames, so the centroid
   condition rejects it independently of the rest. A targeted "thump"
   preset must drop or relax the centroid floor.

### Threshold sweep harness

```bash
python -m benchmarks.non_speech_filter.sweep \
    --backend silero,tenvad,webrtc \
    --corpus-dir "$LIVECAP_NON_SPEECH_CORPUS_DIR"
```

The default sweep walks five labelled presets — `baseline_off`,
`observe_defaults`, `on_conservative`, `on_moderate`, `on_aggressive` —
and writes CSV + Markdown into
`benchmark_results/non_speech_filter/sweep/`. The included MockEngine
column already surfaces gate-level deltas; pass `--engine
whispers2t,parakeet_ja,reazonspeech --device cuda` for the hallucination
column.

#### Observed deltas (mock engine, private real + synthetic corpus)

| Backend × Corpus | `baseline_off` | `on_moderate` | `on_aggressive` |
|---|---|---|---|
| WebRTC × synthetic | 75 % false_trigger | **62.5 %** | **62.5 %** |
| WebRTC × real | 50 % | 50 % | 50 % |
| TenVAD × synthetic | 25 % | 25 % | 25 % |
| TenVAD × real | 0 % | 0 % | 0 % |
| Silero × {synth, real} | unchanged | unchanged | unchanged |

WebRTC × synthetic applause burst is the only cell that moves on default
presets. The persistent 50 % on WebRTC × real desk_tap matches the
per-clip table above: `desk_tap` does not satisfy the AND combination
under any default preset.

#### Calibration follow-up (2026-06-07)

Three hypothesis-driven candidate presets were added by the PR-B
calibration follow-up — `on_relaxed_rms`, `on_low_freq_aware`, and
`on_speech_safe` — and the full 144-cell matrix (8 presets × 3 backends
× 3 engines × 2 corpora) was run on a single RTX 4090. The findings,
including per-engine hallucination deltas and the Pareto summary across
all presets, are recorded permanently in
[`docs/benchmarks/calibration-results-2026-06-07.md`](calibration-results-2026-06-07.md).

Top-line numbers:

| Engine × Backend × Corpus | `baseline_off` hallucination | best on-mode hallucination | Δ |
|---|---|---|---|
| `parakeet_ja` × WebRTC × real (the AC target cell) | 50.0 % | **50.0 %** | **0.0 pp** |
| `reazonspeech` × WebRTC × real | 50.0 % | **50.0 %** | **0.0 pp** |
| `whispers2t` × WebRTC × real | 0.0 % | 0.0 % | 0.0 pp (already at floor) |
| `parakeet_ja` × WebRTC × synthetic | 75.0 % | **62.5 %** | **-12.5 pp** |
| `reazonspeech` × WebRTC × synthetic | 62.5 % | 62.5 % | 0.0 pp |
| `whispers2t` × WebRTC × synthetic | 25.0 % | 25.0 % | 0.0 pp (already low) |

Conclusion: **no candidate preset hit the ≥30 % hallucination-drop
threshold on the AC target cell**. The DSP detector with the current
6-feature AND combination is structurally unable to reject
`desk_tap`-style low-frequency thumps: the clip's centroid sits below
2500 Hz on every frame and its flatness sits at 0 %, so widening any
single threshold leaves the others as independent blockers. **All 8
presets remained recall-safe** — no `speech_recall` or
`short_utterance_recall` regression in any cell.

The verdict therefore stays:

- `--transient-filter=off` remains the CLI default.
- The PR-B Acceptance Criteria target `50 % → 0 %` is reframed in
  Issue #295 v6 to the empirical achievable value.
- A Phase 2 SED epic (sound-event detection model — YAMNet / EfficientAT
  / equivalent) is the right place to handle non-broadband transients;
  filing it is tracked as a separate follow-up.
- `on_moderate` is documented as the **recommended on-mode preset**
  for users who explicitly want best-effort burst-applause filtering
  on rapid-clap material — the synthetic-burst improvement is real,
  just too small to justify changing the default for everyone.

---

## How Phase 1 PRs interact with this harness

| PR | What it must achieve against the baseline |
|---|---|
| PR-B (Layer 1: DSP transient detector) | Default `--transient-filter=off` so baseline runtime behaviour is unchanged; calibration uses an explicit `--transient-filter=observe` opt-in. With `--transient-filter=on`, `false_asr_trigger_rate` should drop for TenVAD/WebRTC on `applause_*`, `keyboard_taps`, `door_close`, `cough`, while `short_utterance_recall` stays ≥ baseline. |
| PR-C (Layer 2: VADStateMachine cooldown extension) | Backend-aware: hysteresis only affects Silero / TenVAD; duration-based cooldown affects all three. Must reduce post-applause false triggers without dropping `post_applause_speech`. |
| PR-A (Layer 3 + 4: Confidence filter + Prompt reset) | Only measurable with a real engine. With `--engine whispers2t`, `non_empty_hallucination_rate` should approach 0 on the negative set; engine call counts for negatives may stay non-zero (Layer 3 is post-engine). |

See Issue [#295] for the full Phase 1 plan.

[#295]: https://github.com/Mega-Gorilla/livecap-cli/issues/295
