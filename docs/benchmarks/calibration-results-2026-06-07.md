# Calibration Results — Transient Detector (Issue #295 PR-B follow-up)

Permanent record of the **2026-06-07 calibration sweep**. The full
machine-readable CSV/Markdown lives under
`benchmark_results/non_speech_filter/sweep/calibration-2026-06-07/`
(gitignored); this document is the human summary that ships with the
repo so future work has the raw numbers to compare against.

## Setup

| Item | Value |
|---|---|
| Date | 2026-06-07 |
| Branch | `feat/issue-295-pr-b-calibration` |
| Sweep CLI | `python -m benchmarks.non_speech_filter.sweep --backend silero,tenvad,webrtc --engine whispers2t,parakeet_ja,reazonspeech --corpus-dir .tmp/non_speech_corpus --device cuda` |
| Presets | 8 (5 from PR-B + 3 hypothesis-driven additions) |
| Backends | 3 (`silero`, `tenvad`, `webrtc`) |
| Engines | 3 (`whispers2t` large-v3, `parakeet_ja`, `reazonspeech`) |
| Corpora | 2 (13 synthetic items, 6 private real items) |
| Matrix | 144 cells |
| Wall-clock | ~16.5 min (engine-load amortised across presets) |
| GPU | NVIDIA RTX 4090, CUDA 12.8 |

## Hypotheses tested

The 3 new candidate presets added by this PR are designed to probe
specific failures observed in the PR-B sweep against the private real
corpus (`docs/benchmarks/non-speech-filter.md` → Layer 1 per-clip
feature pass-rate table).

| Preset | Hypothesis | Threshold delta vs `on_moderate` |
|---|---|---|
| `on_relaxed_rms` | Real corpus sits at -41 to -46 dBFS RMS; default `-35` floor blocks > 95 % of frames before the AND combination ever fires | `rms_min_db = -45` |
| `on_low_freq_aware` | `desk_tap` has centroid < 1 kHz on every frame; default `centroid_min_hz=2500` is a hard blocker | `centroid_min_hz = 500`, `voiced_max = 0.15` (compensate to keep low-pitched speech protected) |
| `on_speech_safe` | Maximum-safety ceiling — only fire on textbook rapid-burst applause | `flatness_min = 0.45`, `centroid_min_hz = 3000`, `onset_ratio = 5.0` |

## Findings

### Finding 1: Real-corpus `desk_tap` hallucination is unchanged across every preset

| Engine | Backend | Corpus | `baseline_off` | best on-mode preset | Δ |
|---|---|---|---|---|---|
| `whispers2t` | `webrtc` | `real` | 0.0 % | 0.0 % | 0.0 pp |
| `parakeet_ja` | `webrtc` | `real` | 50.0 % | **50.0 %** (every on-mode) | **0.0 pp** |
| `reazonspeech` | `webrtc` | `real` | 50.0 % | **50.0 %** (every on-mode) | **0.0 pp** |

The PR-B v4 AC target — `WebRTC × desk_tap (real) hallucination 50 % →
0 %` for `parakeet_ja` / `reazonspeech` — **is not achieved by any of
the 8 presets**, including the 3 hypothesis-driven additions. The 6-
feature AND combination cannot fire on a clip whose spectral centroid
sits below 2500 Hz on 100 % of frames *and* whose flatness sits at 0 %
of frames *and* whose RMS sits at 1 % of frames simultaneously; widening
any single axis (e.g. `on_low_freq_aware` dropping `centroid_min_hz` to
500) still leaves flatness / ZCR / RMS as independent blockers.

### Finding 2: Synthetic-burst hallucination drops modestly, but only for `parakeet_ja`

| Engine | Backend | Corpus | `baseline_off` | best on-mode preset | Δ |
|---|---|---|---|---|---|
| `parakeet_ja` | `webrtc` | `synthetic` | 75.0 % | **62.5 %** (`on_moderate`, `on_aggressive`, `on_relaxed_rms`, `on_low_freq_aware`) | **-12.5 pp** |
| `reazonspeech` | `webrtc` | `synthetic` | 62.5 % | 62.5 % | 0.0 pp |
| `whispers2t` | `webrtc` | `synthetic` | 25.0 % | 25.0 % | 0.0 pp |

Only the rapid-burst synthetic applause case shows movement, and the
movement is one item out of eight (12.5 pp) — not the ≥30 % relative
drop demanded by plan D4 for promoting a preset to production default.
`whispers2t`'s internal `no_speech_prob` filter already keeps its
hallucination low (25 % synthetic, 0 % real), so the upstream gate
change has no headroom there.

### Finding 3: All 8 presets are recall-safe

The calibration analysis reports **zero recall regressions** across all
144 cells. No preset drops `speech_recall` or `short_utterance_recall`
below the `baseline_off` value for any (backend, engine, corpus) cell.
This is consistent with the high-precision AND design and confirms the
3 new presets do not break short utterances or normal speech.

### Finding 4: 4 presets tie on the Pareto frontier

Mean metrics across the full matrix:

| Preset | Mean False Trigger | Mean Hallucination | Pareto-dominant? |
|---|---|---|---|
| `on_moderate` | 22.9 % | 15.3 % | yes |
| `on_aggressive` | 22.9 % | 15.3 % | yes |
| `on_relaxed_rms` | 22.9 % | 15.3 % | yes |
| `on_low_freq_aware` | 22.9 % | 15.3 % | yes |
| `baseline_off` / `observe_defaults` / `on_conservative` / `on_speech_safe` | 25.0 % | 16.0 % | no |

The four pareto-dominant presets are tied because the only cell that
moved is the same one in all four (`parakeet_ja × webrtc × synthetic`).
The mean delta vs `baseline_off` is **-2.1 pp false_trigger** and
**-0.7 pp hallucination** — well under the ≥30 % relative threshold
that would justify changing the production default.

## Decision

> **DSP detector cannot meet the PR-B AC target with the current
> candidate presets. Keep `--transient-filter=off` as the CLI default;
> reframe Issue #295 PR-B AC to record the empirical achievable bound;
> open a Phase 2 SED epic for low-frequency / non-broadband transient
> detection.**

This is the rule D4 verdict from the calibration plan, evaluated against
the live sweep data:

| Rule D4 criterion | Observed | Verdict |
|---|---|---|
| Any preset achieves ≥30 % hallucination drop on `webrtc × parakeet_ja × real` | best 0.0 pp | **fail** |
| All positive recalls ≥ 95 % | yes | pass |
| Short utterance recall = 100 % | yes | pass |

Two of three criteria pass, but the headline criterion (the target
cell) fails completely. Promoting any preset to production default
would deliver virtually no hallucination benefit while imposing a
non-zero CPU cost and an `observable` change in default behaviour on
all users — the tradeoff is not worth shipping.

## Implications

| Item | Status after this calibration |
|---|---|
| CLI default `--transient-filter` | **`off` maintained** (no preset earns the change) |
| BASELINE_INVARIANTS bounds | **unchanged** (default unchanged, so no tighten) |
| Issue #295 PR-B AC | **reframed in v6** with empirical bound for `webrtc × desk_tap (real)` |
| Phase 2 SED epic | **to be opened** as the rightful path for low-frequency thumps |
| `#302` lookahead | **unchanged** (reject default ON is not happening, so lookahead remains backlog) |
| Documented recommended preset | `on_moderate` for users who explicitly enable `on` mode on rapid-burst applause scenes — the slight false_trigger improvement is real, just too small to justify defaults |

## Reproducibility

```bash
git checkout feat/issue-295-pr-b-calibration       # or the merge commit
uv sync --extra engines-torch --extra engines-nemo --extra dev
export LIVECAP_NON_SPEECH_CORPUS_DIR=/path/to/your/non_speech_corpus

uv run python -m benchmarks.non_speech_filter.sweep \
    --backend silero,tenvad,webrtc \
    --engine whispers2t,parakeet_ja,reazonspeech \
    --corpus-dir "$LIVECAP_NON_SPEECH_CORPUS_DIR" \
    --device cuda \
    --output-dir benchmark_results/non_speech_filter/sweep/calibration-2026-06-07

uv run python -m benchmarks.non_speech_filter.calibration \
    benchmark_results/non_speech_filter/sweep/calibration-2026-06-07/transient_sweep_*.csv \
    --output docs/benchmarks/calibration-results-2026-06-07.md
```

The private real corpus is not redistributable; see
`docs/benchmarks/non-speech-filter.md` → "Reference corpus" for the
manifest schema and recommended public substitutes (ESC-50 for applause,
FSD50K for non-speech events) that approximate this corpus.

The sweep wall-clock budget on a single RTX 4090 was ~16.5 min for 144
cells with engine-load amortisation. Plan accordingly for slower GPUs
or larger matrices.
