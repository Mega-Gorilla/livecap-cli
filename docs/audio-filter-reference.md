# Audio Filter Reference

User-facing reference for the audio processing filters that sit on the
real-time and file transcription pipelines in `livecap-cli`. Each entry
covers **what the filter does, where it runs, the CLI surface, the
measured effectiveness, and whether it is production-ready or
experimental**.

For deep benchmark methodology and raw numbers see:

- `docs/benchmarks/non-speech-filter.md` — Phase 1 evaluation harness,
  per-clip baselines, sweep harness usage.
- `docs/benchmarks/calibration-results-2026-06-07.md` — the calibration
  sweep that determined the production status of each layer.

---

## Pipeline overview

```
mic / file audio  (16 kHz mono float32)
       │
       ▼
NoiseGate (#291)              ─── per-sample peak gate, production
       │
       ▼
TransientDetector (#295 PR-B) ─── DSP applause/transient detector
       │                          *** EXPERIMENTAL, off by default ***
       ▼
VAD (Silero / TenVAD / WebRTC)─── speech vs non-speech segmentation
       │
       ▼
EnergyGate (#292)             ─── per-segment RMS gate, production
       │
       ▼
ASR Engine (whispers2t / parakeet_ja / reazonspeech / ...)
       │
       ▼
[future Layer 3+4: confidence filter + prompt-context reset]
```

Each filter targets a different failure mode and they compose
multiplicatively — enabling a downstream filter does not remove the
need for upstream ones.

---

## 1. NoiseGate (#291)

| Property | Value |
|---|---|
| **Purpose** | Per-sample peak gate that mutes audio below an absolute dB threshold (with attack/release smoothing). Removes click-level background noise so the VAD does not waste cycles on silence. |
| **Pipeline position** | First (raw audio in). |
| **Default state** | **OFF** (opt-in via `--noise-gate`). |
| **CLI surface** | `--noise-gate` (enable) <br> `--noise-gate-threshold` (default `-35` dB) <br> `--noise-gate-close-threshold` (default `threshold - 6` dB, hysteresis) <br> `--noise-gate-attack` (default `0.5` ms) <br> `--noise-gate-release` (default `100` ms) <br> `--noise-gate-floor` (default `-inf`, hard mute) |
| **Production-ready** | **Yes** — calibrated, hysteresis-aware, in `main`. |
| **Effective against** | Steady background noise floor; broadband click-level noise. |
| **Not effective against** | Sustained high-energy non-speech (laughter, applause); speech artefacts; engine hallucinations. |
| **When to enable** | Live microphone inputs in noisy rooms; recordings with audible hum or hiss. |
| **When NOT to enable** | Already-clean recordings; very quiet speech (the gate can clip syllable openings if threshold is too aggressive). |
| **Benchmark reference** | `docs/benchmarks/non-speech-filter.md` — PR-0 baselines were established with NoiseGate disabled, so its absolute impact is not in those tables. |

**Tuning workflow**: `livecap-cli levels --duration 5` measures your
environment's RMS floor and prints a suggested `--noise-gate-threshold`
value.

---

## 2. TransientDetector (#295 PR-B) — EXPERIMENTAL

> ⚠️ **Not a production hallucination mitigation candidate.** The
> 2026-06-07 calibration sweep showed `0 pp` improvement on the
> real-corpus AC target cell (`WebRTC × parakeet_ja × desk_tap`).
> Keep `--transient-filter=off` for production. See
> `docs/benchmarks/calibration-results-2026-06-07.md` for the
> empirical evidence.

| Property | Value |
|---|---|
| **Purpose** | DSP detector that AND-combines six per-frame features (spectral flatness, spectral centroid, ZCR, onset strength, voiced confidence, RMS dBFS) to flag rapid-burst applause-like frames. |
| **Pipeline position** | After NoiseGate, before VAD. |
| **Default state** | **OFF**. Calibration retained this default — no candidate preset earned promotion. |
| **CLI surface** | `--transient-filter {off,observe,on}` (default `off`) <br> `--transient-flatness-min` (default `0.30`) <br> `--transient-centroid-min-hz` (default `2500`) <br> `--transient-zcr-min` (default `0.12`) <br> `--transient-onset-ratio` (default `3.0`) <br> `--transient-voiced-max` (default `0.25`) <br> `--transient-rms-min-db` (default `-35`) |
| **Production-ready** | **No — experimental.** Not deprecated (there is no replacement yet) but not recommended for production hallucination mitigation. |
| **Effective against** | Synthetic rapid-burst applause (7+ claps in < 1 s, all at high SNR). One cell out of 144 in the calibration matrix moves: `parakeet_ja × WebRTC × synthetic burst` hallucination 75 % → 62.5 % under `on_moderate`/`on_aggressive`/`on_low_freq_aware`/`on_relaxed_rms`. |
| **Not effective against** | **Real desk taps, knocks, applause from live audiences, scattered claps, low-frequency thumps.** The 6-feature AND combination cannot fire on a clip whose spectral centroid sits below 2500 Hz on 100 % of frames *and* whose flatness sits at 0 % *and* whose RMS sits at 1 % simultaneously — widening any single threshold leaves the others as independent blockers. |
| **When to enable** | DSP-feature experimentation, calibration sweeps, observing per-frame counters in `observe` mode for environment-specific data collection. |
| **When NOT to enable** | Any production deployment whose goal is reducing real-world ASR hallucinations. The CLI emits an experimental notice at startup when the flag is set to `observe` or `on`. |
| **Modes** | `off`: detector not constructed, pipeline unchanged. <br> `observe`: features computed + telemetry counters updated, audio passes through unmodified. <br> `on`: flagged frames zeroed-out before the VAD sees them (causal best-effort — see `TransientDetector.process()` docstring for the chunked-streaming contract). |
| **Benchmark reference** | `docs/benchmarks/non-speech-filter.md` → "Calibration follow-up (2026-06-07)" + `docs/benchmarks/calibration-results-2026-06-07.md` for the 144-cell matrix. |

**`on_moderate` positioning**: The four Pareto-dominant on-mode presets
(`on_moderate`, `on_aggressive`, `on_relaxed_rms`, `on_low_freq_aware`)
tie on mean false_trigger and mean hallucination. `on_moderate` is
recorded as the **best observed DSP preset for synthetic rapid-burst
tests only** — explicitly **not** a production hallucination mitigation
recommendation.

**Roadmap**: Phase 2 SED (sound-event detection — YAMNet, EfficientAT,
or equivalent learned model) is the planned successor for `desk_tap`-
style transients the DSP design structurally cannot catch. When Phase 2
SED ships, this layer will either be removed or its slot will be reused
by the SED backend; the experimental status of `--transient-filter`
holds until that transition.

---

## 3. VAD backend (Silero / TenVAD / WebRTC)

| Property | Value |
|---|---|
| **Purpose** | Speech vs non-speech segmentation. The core decision that determines what audio segments the ASR engine sees. |
| **Pipeline position** | Core (after the upstream gates). |
| **Default state** | Backend choice depends on environment; default is `silero` for general use. |
| **CLI surface** | `--vad-backend {silero,tenvad,webrtc}` (or via stream config) |
| **Production-ready** | **Yes** — all three backends are production-ready. Trade-offs differ. |
| **Effective against** | Distinguishing speech from silence and sustained background noise. |
| **Not effective against** | High-energy non-speech events that resemble speech onsets (applause bursts, knocks). Each backend has different tolerance — `WebRTC` is most permissive, `TenVAD` is intermediate, `Silero` is strictest. |
| **Per-backend effectiveness** (private real corpus, PR-0 baseline) | `silero`: 0 % false_trigger, 100 % positive recall (but 0 % on synthetic sub-1 s utterances by design) <br> `tenvad`: 0 % false_trigger on real, 25 % on synthetic burst <br> `webrtc`: **50 % false_trigger on real** (the cell PR-B / Phase 2 SED targets), 75 % on synthetic burst |
| **When to choose which** | `silero` for the strictest no-false-trigger default; `tenvad` when sub-second utterances matter; `webrtc` for legacy compatibility — but be aware of its higher false_trigger rate. |
| **Benchmark reference** | `docs/benchmarks/non-speech-filter.md` → "Reference corpus" and "Per-backend per-clip triggered" tables. |

VAD selection is the largest single lever on hallucination risk —
switching from `webrtc` to `silero` on the same audio typically removes
more false triggers than any post-VAD gate can.

---

## 4. EnergyGate (#292)

| Property | Value |
|---|---|
| **Purpose** | Per-segment RMS gate that drops VAD segments whose RMS energy is below a threshold before they reach the ASR engine. |
| **Pipeline position** | Post-VAD, pre-engine. |
| **Default state** | **ON** (default threshold `-45` dBFS). Set `--engine-min-rms` to `-inf` to disable. |
| **CLI surface** | `--engine-min-rms` (default `-45.0` dBFS) |
| **Production-ready** | **Yes** — calibrated, validated against PR-0 baselines. |
| **Effective against** | Low-energy ASR false triggers: brief breath sounds, faint background events that the VAD mis-segments as speech. |
| **Not effective against** | High-energy non-speech (applause, knocks) — those sit well above the energy floor and pass through. |
| **When to tune** | `livecap-cli levels --duration 5` analyses your environment and prints a suggested value; default `-45` works for typical recordings. |
| **When NOT to relax (raise the floor)** | If your speakers are very quiet — raising the floor risks dropping legitimate quiet utterances. |
| **Benchmark reference** | Issue #292 / PR #294 measured the false-trigger reduction across all three backends. |

---

## Comparison table

| Filter | Pipeline position | Default | Production-ready | Hallucination on real desk_tap (WebRTC × parakeet_ja) |
|---|---|---|---|---|
| NoiseGate | Pre-VAD | OFF (opt-in) | Yes | n/a (not its target) |
| **TransientDetector** | Pre-VAD | **OFF (experimental)** | **No** | **No improvement (50 % → 50 %, 0 pp)** |
| VAD backend | Core | Silero | Yes | Backend choice is the single largest lever |
| EnergyGate | Post-VAD | ON (-45 dBFS) | Yes | Already at floor (engine-internal defense varies) |

---

## Decision: which filters do I enable?

The most defensible production stack today:

1. **`--noise-gate`** if your microphone picks up steady background noise
   (most live mic setups).
2. **`--vad-backend silero`** unless you have a specific reason to use a
   more permissive backend.
3. **EnergyGate at default `-45` dBFS** (no flag needed; on by default).
4. **`--transient-filter=off`** (default). Do **not** enable for
   production hallucination mitigation. Enable to `observe` only when
   you are collecting DSP-feature data for calibration work.

If your audience still gets hallucinations on knocks, taps, or applause
after the above, the next step is **not** to tune the transient detector
further — it is to wait for Phase 2 SED. Calibration data shows the DSP
6-feature AND combination is structurally unable to fire on those
sources.

---

## Known limitations and Phase 2 roadmap

| Failure mode | Layer that should solve it | Status |
|---|---|---|
| Background noise floor | NoiseGate | Solved (PR #291) |
| Low-energy false ASR | EnergyGate | Solved (PR #292) |
| Speech / non-speech boundary | VAD | Solved (Silero/TenVAD/WebRTC) |
| **Rapid-burst applause** | TransientDetector `on` (modest help on synthetic only) | Partial — DSP saturated |
| **Desk taps / knocks / scattered claps** | Phase 2 SED (planned) | **Not yet implemented** |
| ASR engine internal hallucination on speech-like noise | Phase 1 Layer 3 (confidence filter) | Planned (PR-A) |
| Whisper prompt-context drift | Phase 1 Layer 4 (prompt reset) | Planned (PR-A) |

**Phase 2 SED epic** (filed separately after this calibration): integrate
a learned sound-event-detection model (YAMNet, EfficientAT, or
equivalent) to handle the transient classes the DSP design cannot cover.
The empirical evidence for opening that epic lives in
`docs/benchmarks/calibration-results-2026-06-07.md`.

---

## Cross-references

- Phase 1 multi-layered defense epic: [Issue #295](https://github.com/Mega-Gorilla/livecap-cli/issues/295)
- Per-clip / per-backend triggered tables: `docs/benchmarks/non-speech-filter.md`
- DSP calibration empirical record: `docs/benchmarks/calibration-results-2026-06-07.md`
- VAD comparison: `docs/benchmarks/non-speech-filter.md` → "Reference corpus"
- NoiseGate calibration tool: `livecap-cli levels --help`
