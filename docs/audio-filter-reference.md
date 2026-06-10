# Audio Filter Reference

User-facing reference for the audio processing filters that sit on the
**real-time transcription pipeline** in `livecap-cli` (microphone or
streaming input). Each entry covers **what the filter does, where it
runs, the CLI surface, the measured effectiveness, and whether it is
production-ready or experimental**.

> **File-mode transcription** (`livecap-cli transcribe <file> -o out.srt`,
> no `--realtime`) currently uses a separate batch pipeline (see
> `livecap_cli/transcription/file_pipeline.py`) and does **not**
> construct NoiseGate, the transient detector, or the post-VAD
> EnergyGate. The CLI flags documented below are only effective in
> realtime mode. Integrating these filters into the file pipeline is a
> separate, unscheduled follow-up.

For deep benchmark methodology and raw numbers see:

- `docs/benchmarks/non-speech-filter.md` — Phase 1 evaluation harness,
  per-clip baselines, sweep harness usage.
- `docs/benchmarks/calibration-results-2026-06-07.md` — the DSP
  calibration sweep that determined the production status of each layer.
- `docs/research/phase2-sed-evaluation-2026-06-10.md` — the Phase 2 SED
  off-line evaluation (Issue #305 PR-D0) that confirmed a learned model
  can solve the WebRTC hallucination case, then closed as wontfix because
  Silero / TenVAD already do.

---

## TL;DR — for production, use Silero or TenVAD

If you have time for one line of advice: pass
`--vad-backend silero` (the default) and you will not see ASR
hallucinations on desk taps, knocks, or applause. The
[hallucination mechanism](#why-does-asr-hallucinate) section below
explains why this is enough, and why WebRTC is not recommended.

---

## Pipeline overview (realtime mode)

```
mic / streaming audio  (16 kHz mono float32)
       │
       ▼
NoiseGate (#291)              ─── per-sample peak gate, production
       │
       ▼
TransientDetector (#295 PR-B) ─── DSP applause/transient detector
       │                          *** EXPERIMENTAL, off by default ***
       ▼
VAD (Silero / TenVAD / WebRTC)─── speech vs non-speech segmentation
       │                          *** Silero/TenVAD recommended ***
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

## Why does ASR hallucinate?

ASR (automatic speech recognition) engines are statistical text
generators. When the upstream stages let a non-speech audio segment
through (e.g. a desk tap), the engine has no general way to refuse — it
maps the audio to the most likely Japanese (or other-language) phrase
under its model, which is typically a common short word like "はい" or
"ありがとうございます". That word appears in the live caption even
though nobody spoke. This is the "hallucination" phenomenon.

The realtime pipeline catches non-speech in three places before the
engine sees it:

1. **VAD backend** (the largest lever)
2. **Engine-internal `no_speech_prob`** (only available on Whisper-family
   models)
3. **Engine-confidence post-filter** (planned, Layer 3+4)

The single most important observation is that **the VAD backend choice
dominates everything else**. Empirical data from the 2026-06-07
calibration sweep (144 cells: 3 backends × 3 engines × 2 corpora × 8
DSP presets):

| VAD backend × ASR engine × real corpus | False trigger | Hallucination |
|---|---|---|
| `silero` × `whispers2t` / `parakeet_ja` / `reazonspeech` | **0 %** | **0 %** |
| `tenvad` × `whispers2t` / `parakeet_ja` / `reazonspeech` | **0 %** | **0 %** |
| `webrtc` × `whispers2t` | 50 % | 0 % (engine-internal `no_speech_prob` absorbs) |
| **`webrtc` × `parakeet_ja`** | **50 %** | **50 %** |
| **`webrtc` × `reazonspeech`** | **50 %** | **50 %** |

→ Hallucination happens **only when a permissive VAD lets a non-speech
segment through to an ASR engine without internal non-speech defence**.
Silero and TenVAD are learned VADs and refuse the desk-tap / applause
class outright. WebRTC is a rule-based VAD (Gaussian mixture on spectral
energy) and accepts the same audio because its energy envelope happens
to resemble a vowel onset.

Two consequences for users:

- If you are on Silero or TenVAD, **you do not have a hallucination
  problem to solve.** None of the DSP transient detector or the
  (closed) Phase 2 SED epic targets your stack.
- If you are on WebRTC and your engine is `parakeet_ja` or
  `reazonspeech`, **the highest-leverage fix is to switch backend**, not
  to enable extra filters.

Background:
[PR-B calibration results](benchmarks/calibration-results-2026-06-07.md),
[Phase 2 SED PR-D0 decision document](research/phase2-sed-evaluation-2026-06-10.md),
[Issue #305 close note](https://github.com/Mega-Gorilla/livecap-cli/issues/305).

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

**Phase 2 SED status (updated 2026-06-10)**: The follow-up Phase 2 SED
epic ([Issue #305](https://github.com/Mega-Gorilla/livecap-cli/issues/305))
was closed as `not planned` after the PR-D0 off-line evaluation
([PR #306](https://github.com/Mega-Gorilla/livecap-cli/pull/306)). The
evaluation confirmed that a learned SED model (EfficientAT `mn04_as`)
*can* solve the WebRTC × `desk_tap` hallucination case, but the broader
empirical record shows that Silero and TenVAD already achieve `0 %`
hallucination across all three engines on the same corpus — so the SED
investment would only benefit users staying on WebRTC + `parakeet_ja` /
`reazonspeech`. See
[`docs/research/phase2-sed-evaluation-2026-06-10.md`](research/phase2-sed-evaluation-2026-06-10.md)
for the decision document and the `benchmarks/sed/` package that can be
reactivated if real-world WebRTC demand surfaces.

---

## 3. VAD backend (Silero / TenVAD / WebRTC)

| Property | Value |
|---|---|
| **Purpose** | Speech vs non-speech segmentation. The core decision that determines what audio segments the ASR engine sees. |
| **Pipeline position** | Core (after the upstream gates). |
| **Default state** | `silero` (production recommended). |
| **CLI surface** | `--vad-backend {silero,tenvad,webrtc}` |
| **Production recommendation** | **Silero (default) or TenVAD**. Both achieve `0 %` hallucination on the real desk-tap / applause corpus across all three engines. |
| **WebRTC status** | **Not recommended for production with `parakeet_ja` or `reazonspeech`.** WebRTC is a rule-based VAD that false-triggers at `50 %` on the real desk-tap clip, and the affected engines have no internal `no_speech_prob` defence, so the false trigger flows through to a hallucinated transcription. Retained as a lightweight option for environments where the learned VADs cannot be loaded (e.g. no PyTorch). See the [hallucination mechanism section](#why-does-asr-hallucinate) for the explanation. |

### Per-backend numbers (private real corpus, PR-B 2026-06-07 sweep)

| Backend | False trigger | Hallucination (worst engine) | Production-ready? |
|---|---|---|---|
| **`silero`** (default) | **0 %** | **0 %** | ✅ Yes |
| **`tenvad`** | **0 %** | **0 %** | ✅ Yes |
| `webrtc` | **50 %** | **50 %** (with `parakeet_ja` / `reazonspeech`); `0 %` only with `whispers2t` | ⚠ Lightweight option only — see below |

### When to choose which backend

- **`silero` (default)**: the strictest no-false-trigger backend. Use
  unless you have a specific reason to switch. PyTorch is required.
- **`tenvad`**: a lighter learned VAD with the same hallucination
  resilience as Silero on the test corpus. Use when Silero's footprint
  is a problem and PyTorch is still available.
- **`webrtc`**: a pure-C frame-based VAD with the smallest binary
  footprint and lowest per-frame latency. **Not recommended for
  production with `parakeet_ja` or `reazonspeech`** because both engines
  hallucinate on WebRTC's false-trigger desk-tap segments at `50 %`.
  Acceptable for `whispers2t` because Whisper's internal
  `no_speech_prob` mechanism absorbs the false trigger.

VAD selection is the largest single lever on hallucination risk —
switching from `webrtc` to `silero` on the same audio removes more
false triggers than any post-VAD gate can.

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
| VAD backend | Core | **Silero (production)** | Silero / TenVAD ✅, WebRTC ⚠ (lightweight only) | **Silero / TenVAD already solve this case (0 % across all engines)** |
| EnergyGate | Post-VAD | ON (-45 dBFS) | Yes | Already at floor (engine-internal defense varies) |

---

## Decision: which filters do I enable?

The most defensible production stack today:

1. **`--vad-backend silero`** (the default). On any non-trivial real
   audio, this single choice removes the entire desk-tap / applause
   hallucination problem. Use `tenvad` as an equivalent lighter
   alternative.
2. **`--noise-gate`** if your microphone picks up steady background
   noise (most live mic setups).
3. **EnergyGate at default `-45` dBFS** (no flag needed; on by default).
4. **`--transient-filter=off`** (default). Do **not** enable for
   production hallucination mitigation. Enable to `observe` only when
   you are collecting DSP-feature data for calibration work.
5. **Avoid `--vad-backend webrtc`** with `parakeet_ja` or `reazonspeech`
   unless you have an external reason (PyTorch unavailable, embedded
   binary-size constraint). With `whispers2t` the engine's internal
   defence absorbs WebRTC's false triggers, so the combination is safe.

If you are on Silero or TenVAD and still get hallucinations on
something the test corpus does not cover (e.g. notification sounds,
background music), please open an issue with a short audio clip and
the VAD backend in use — that data is what would re-open Phase 2 SED
or motivate the planned PR-A engine-confidence filter.

---

## Known limitations and remaining work

| Failure mode | Layer that should solve it | Status |
|---|---|---|
| Background noise floor | NoiseGate | Solved (PR #291) |
| Low-energy false ASR | EnergyGate | Solved (PR #292) |
| Speech / non-speech boundary | VAD (Silero / TenVAD) | Solved on the tested corpus; WebRTC remains permissive by design |
| **Rapid-burst applause (synthetic)** | TransientDetector `on` (modest help) | Partial — DSP saturated |
| **Desk taps / knocks / scattered claps on WebRTC** | Switch to Silero / TenVAD, *or* the (now-closed) Phase 2 SED epic | **Workaround available (backend switch); SED epic closed as `not planned`** |
| ASR engine internal hallucination on speech-like noise (parakeet_ja / reazonspeech only) | Planned Layer 3 — engine-confidence filter (PR-A) | **Next planned work** |
| Whisper prompt-context drift | Planned Layer 4 — prompt reset (PR-A) | Next planned work |

**On Phase 2 SED (Issue #305)**: closed as `not planned` on 2026-06-10
after PR-D0 confirmed a learned model can solve the WebRTC × desk-tap
case but Silero / TenVAD already do across the same engine set. The
SED evaluation harness (`benchmarks/sed/`) and decision document
([`docs/research/phase2-sed-evaluation-2026-06-10.md`](research/phase2-sed-evaluation-2026-06-10.md))
remain in the repository as a reusable basis if real-world WebRTC
demand surfaces. The next planned defence-in-depth work is **PR-A**,
which is backend-independent and targets the
`parakeet_ja` / `reazonspeech` engines that lack an internal
`no_speech_prob` mechanism.

---

## Cross-references

- Phase 1 multi-layered defense epic: [Issue #295](https://github.com/Mega-Gorilla/livecap-cli/issues/295)
- Phase 2 SED epic (closed `not planned`): [Issue #305](https://github.com/Mega-Gorilla/livecap-cli/issues/305)
- Phase 2 SED PR-D0 decision document: `docs/research/phase2-sed-evaluation-2026-06-10.md`
- Per-clip / per-backend triggered tables: `docs/benchmarks/non-speech-filter.md`
- DSP calibration empirical record: `docs/benchmarks/calibration-results-2026-06-07.md`
- VAD comparison: `docs/benchmarks/non-speech-filter.md` → "Reference corpus"
- NoiseGate calibration tool: `livecap-cli levels --help`
