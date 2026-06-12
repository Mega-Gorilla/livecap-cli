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

## 5. Confidence Filter (PR-A, post-ASR)

| Property | Value |
|---|---|
| **Purpose** | Drops ASR output that the engine itself judged as low-confidence / non-speech, before the text reaches the subtitle stream. Uses the engine-internal signals that PR-A.0 ([#309](https://github.com/Mega-Gorilla/livecap-cli/pull/309)) exposed on `TranscriptionResult.engine_confidence`. |
| **Pipeline position** | **Post-ASR** (unique — only filter that runs after the engine). |
| **Default state** | **ON** (default `--confidence-filter on`). Use `off` to fully disable, `observe` to log decisions without dropping. |
| **CLI surface** | `--confidence-filter {off, observe, on}` (default `on`) |
| **Env var override** | `LIVECAP_CONFIDENCE_FILTER={off,observe,on}` takes precedence over the CLI flag. Useful for scripts / docker compose `.env` files. |
| **Production-ready** | **Yes** for WhisperS2T / Parakeet_ja / Voxtral / Canary / Parakeet 英語 / ReazonSpeech / **qwen3asr** (7 engine 対応、PR-A.0/A.4.1/A.4.2/A.4.3/A.5.1/A.5.2)。各 engine の検証 scope は以下:<br>• WhisperS2T / Parakeet_ja: PR #309 real-machine smoke verify (20× / 167× separation) + **PR-A.3 calibration sweep ([PR #312 MERGED]) で 54 cell validate** (旧 3 engine = whispers2t / parakeet_ja / reazonspeech が対象)<br>• Voxtral: **PR-A.4.1 ([#313 MERGED])** で smoke verify (margin +1.0) + 12 cell stream pipeline benchmark<br>• Canary: **PR-A.4.2 ([#315 MERGED])** で smoke verify (14.5× margin) + 12 cell stream pipeline benchmark<br>• Parakeet 英語: **PR-A.4.3 ([PR #316])** で smoke verify (**49× margin**) + 12 cell stream pipeline benchmark (`webrtc × synthetic × on` で Hall.(post) 75% → 12.5% 実証)<br>• ReazonSpeech: **PR-A.5.1 ([PR #319])** で smoke verify (int8 +0.13 / float32 +0.10 margin、両方 Case A) + 12 cell stream pipeline benchmark (`webrtc × real × on` で Hall.(post) **50% → 0%** 実証、Issue #295 元 motivation の最後の cell 完了)<br>• **qwen3asr**: **PR-A.5.2 ([Issue #318])** で両言語 smoke verify (EN +0.65 / JA +0.42 margin、両方 Case A) + 12 cell stream pipeline benchmark (Hall.(pre) 0% 全 cell、Canary と同 engine 固有 fail-safe pattern、`repetition_penalty=1.1 + no_repeat_ngram_size=3` で両言語 failure mode 解消) |
| **Effective against** | Engine-produced hallucinations on non-speech audio that the upstream VAD let through (e.g. WebRTC × desk-tap / applause). |
| **Not effective against** | qwen3asr の auto-detect mode (`--language=auto` / language 未指定) — wrapper fallback path に入り `engine_confidence` 全 None で fail-open。production user は `--language en/ja/...` 明示推奨。 |
| **When to tune** | Per-engine thresholds are fixed at smoke verify values. Override programmatically via `FilterConfig(no_speech_threshold=..., token_conf_threshold=..., avg_logprob_threshold=...)` (no CLI flag yet; PR-A.3 calibration doc 参照)。 |

### Engine support

| Engine | Filter material | Threshold (default) | Behavior with `--confidence-filter on` |
|---|---|---|---|
| **whispers2t** | `no_speech_prob` | `> 0.5` reject | Real-machine: speech 0.036 (pass) vs non-speech 0.63-0.66 (drop). 20× separation. |
| **parakeet_ja** | `token_confidence_mean` | `< 0.005` reject | Real-machine: speech 0.01-0.10 (pass) vs non-speech 0.0000029-0.0003 (drop). 3-4 orders of magnitude separation. |
| **voxtral** | `avg_logprob` (strict-gated) | `< -1.0` reject | PR-A.4.1 real-machine smoke (2026-06-11): speech mean -0.42 (pass) vs non-speech mean -1.53 (drop). Margin +1.0, midpoint -1.02. Strict-gated: only evaluated when `no_speech_prob` and `token_confidence_mean` are both `None` — so WhisperS2T / Parakeet_ja never enter this path. |
| **canary** | `token_confidence_mean` | `< 0.005` reject | PR-A.4.2 real-machine smoke (2026-06-11): native English speech mean 0.0724 (pass, 14.5× threshold). Greedy decoding + `confidence_cfg.preserve_token_confidence` 経由で NeMo `multitask_greedy_decoding.pack_hypotheses` から `torch.Tensor` token_confidence を取得、`.tolist()` で list 化して mean 計算。日本語など非対応言語入力では engine 自体が empty text を返す fail-safe (filter は介入不要)。 |
| **parakeet (英語)** | `token_confidence_mean` | `< 0.005` reject | PR-A.4.3 [#316] real-machine smoke (2026-06-11): native English speech mean **0.2452** (pass, **49× threshold**). NeMo TDT decoding + `preserve_alignments=True` + `confidence_cfg.preserve_token_confidence` 経由で `hypothesis.token_confidence` (List[float]) を取得、Parakeet_ja と同 helper で mean 計算。Section 2 stream pipeline で `webrtc × synthetic × on` の Hall.(post) 75% → 12.5% を実証。非英語入力 (日本語等) では language mismatch による低 confidence で false reject の可能性、`--confidence-filter off` で opt-out 可能。 |
| **reazonspeech** | `avg_logprob` (engine-specific threshold) | `< -0.2` reject | **PR-A.5.1 [#317] real-machine smoke (2026-06-11)**: speech mean -0.14 (int8) / -0.16 (float32) vs non-speech mean -0.30 / -0.45 → margin +0.13 (int8) / +0.10 (float32)、両 model で Case A clean separation。sherpa-onnx 1.12.39 で `OfflineRecognitionResult.ys_log_probs` を取得、mean を `EngineConfidence.avg_logprob` (Voxtral と同 semantics) に populate。Section 2 で `webrtc × real × on` の Hall.(post) **50% → 0%** を実証 (Issue #295 元 motivation の最後の cell 完了)。**engine-specific threshold** (`avg_logprob_thresholds["reazonspeech"] = -0.2`) で Voxtral 用 `-1.0` と分離 (両者の margin が桁違いのため)。 |
| **qwen3asr** | `avg_logprob` (engine-specific threshold) | `< -0.3` reject | **PR-A.5.2 ([Issue #318]) real-machine smoke (2026-06-12)**: 両言語 verified — EN: speech mean -0.05、non-speech -0.71、margin **+0.65** / JA: speech mean -0.12、non-speech -0.63、margin **+0.42**、両言語 Case A clean separation。**Wrapper bypass で `Qwen3ASRForConditionalGeneration.generate(output_scores=True, repetition_penalty=1.1, no_repeat_ngram_size=3)` を直接呼び**、`compute_transition_scores(normalize_logits=True)` 経由で per-token logprob を取得、Voxtral 同 helper で mean 計算。**`repetition_penalty=1.1 + no_repeat_ngram_size=3`** で両言語の failure mode (EN: system prompt leak、JA: 256-token repetition loop) を完全解消、Section 2 で Hall.(pre) 0% 全 cell (Canary と同 engine 固有 fail-safe pattern)。`_asr_language is None` (auto-detect mode) と auto-detect 用 fail-open path 残置。WER 軽微退行 (LLM typical 0.5-1%) は filter benefit 優先で allow。`--confidence-filter off` は **post-ASR reject のみ** 無効化し、`repetition_penalty=1.1 + no_repeat_ngram_size=3` の generation 側変更は固定で残る (Voxtral greedy / Canary greedy と同 framing)。 |
| mock | — (test fixture only) | — | — |

### 3 modes

- **`on`** (default) — filter applies, rejected outputs are silently dropped (no subtitle).
- **`observe`** — judgments are logged structured (`source_id`, `engine`, `text`, `decision`, `reason`, `engine_confidence`) but no drop happens. Use this when collecting calibration data without affecting users.
- **`off`** — filter is a no-op, no logging. Equivalent to PR-A.0 / pre-PR-A.1 behavior.

### Escape hatch

Two ways to disable the filter without touching the CLI:

1. **Environment variable**: `LIVECAP_CONFIDENCE_FILTER=off livecap-cli transcribe ...` (or set in the shell once)
2. **CLI flag**: `livecap-cli transcribe --confidence-filter off ...`

The env var takes precedence over the CLI flag, so `LIVECAP_CONFIDENCE_FILTER=on` will keep the filter active even if a script passes `--confidence-filter off`.

### Startup banner

Every realtime session emits one INFO log line on startup so users see the active mode:

```
Confidence filter: ON (whispers2t no_speech_prob > 0.5, parakeet (ja/en) / canary token_conf < 0.005, voxtral avg_logprob < -1.0, reazonspeech avg_logprob < -0.2). Disable: --confidence-filter off or LIVECAP_CONFIDENCE_FILTER=off
```

The `voxtral avg_logprob < -1.0` clause is omitted when the user explicitly opts out by passing `FilterConfig(avg_logprob_threshold=None)` (PR-A.4.1). The `parakeet (ja/en) / canary` clause shows the shared `token_conf_threshold` used by Parakeet_ja (TDT-CTC hybrid)、**Parakeet 英語** (TDT only, PR-A.4.3 [#316])、Canary (AED multitask, PR-A.4.2) — all three populate `EngineConfidence.token_confidence_mean` via NeMo greedy decoding with `preserve_token_confidence=True`. The **`reazonspeech avg_logprob < -0.2`** clause is the engine-specific threshold (PR-A.5.1 [#317]、`FilterConfig.avg_logprob_thresholds["reazonspeech"]`)、separated from Voxtral's `-1.0` because the two engines' avg_logprob distributions differ by an order of magnitude.

### When NOT to disable

For `webrtc × parakeet_ja` (the historical 50 % hallucination cell) the filter is the only engine-side defense — turning it `off` reverts to the pre-PR-A.1 behavior where 50 % of `desk_tap` audio produced phantom transcripts.

For `silero` / `tenvad` users the filter doesn't fire on any production-typical audio (the VAD already removes the non-speech before it reaches the engine), so leaving it `on` is essentially free.

### PR-A 系列 完成サマリ (2026-06-11、Issue #311 v2.1 完了時点)

Confidence filter は Phase 1 多段防御 epic ([#295 CLOSED]) の Layer 5 として PR-A.0/A.1/A.3 で本体実装、PR-A.4.1/A.4.2 で対応 engine を **4 つに拡大**して完成形に到達:

| Engine | Filter 状態 | Signal | Threshold | 寄与 PR | 実測 margin |
|---|---|---|---|---|---|
| **WhisperS2T** | ✅ Production | `no_speech_prob` | `> 0.5` | [#309/#310] | 20× (speech 0.036 vs non-speech 0.66) |
| **Parakeet_ja** | ✅ Production | `token_confidence_mean` | `< 0.005` | [#309/#310] | 167× (speech 0.05 vs non-speech 0.0000029) |
| **Voxtral** | ✅ Production (strict-gated) | `avg_logprob` | `< -1.0` | [#313] PR-A.4.1 | +1.0 (speech mean -0.42 vs non-speech mean -1.53) |
| **Canary** | ✅ Production | `token_confidence_mean` | `< 0.005` (Parakeet 共用) | [#315] PR-A.4.2 | 14.5× (speech 0.0724 vs threshold 0.005) |
| **Parakeet (英語)** | ✅ **Production** | `token_confidence_mean` | `< 0.005` (Parakeet 共用) | **[#316] PR-A.4.3** | **49× (speech 0.2452 vs threshold 0.005)** |
| **ReazonSpeech** | ✅ **Production** | `avg_logprob` (engine-specific threshold) | `< -0.2` reject | **[#317] PR-A.5.1** | **margin +0.13 (int8) / +0.10 (float32)、両方 Case A** |
| **qwen3asr** | ✅ **Production** | `avg_logprob` (engine-specific threshold、wrapper bypass) | `< -0.3` reject | **[#318] PR-A.5.2** | **両言語 verified — EN +0.65 / JA +0.42 margin、両方 Case A** |
| mock | — (test fixture only) | — | — | — | — |

#### Parakeet 英語の判明経緯 → PR-A.4.3 実装完了 (PR #316)

旧 docs では「NeMo RNNT path に token_confidence 未実装」を理由に PR-A.5 candidate としていたが、本 PR の調査で:

1. **NeMo source 確認** (`rnnt_decoding.py:95-106` で `preserve_token_confidence` documented、`tdt_loop_labels_computer.py:104-371` で実装 confirmed)
2. **NeMo の制約条件**: `preserve_frame_confidence=True` 時は **`preserve_alignments=True` 同時設定必須** (`rnnt_decoding.py:280-282`)
3. **PR #309 時点の実装漏れ**: `preserve_frame_confidence=True` のみ設定 → 拒否 → 「構造的限界」と誤認
4. **実機 probe**: Path 1 (Hybrid CTC) と同じ pattern (preserve_alignments + preserve_frame_confidence + confidence_cfg) を非 hybrid model に適用 → **Parakeet 英語で `token_confidence_mean = 0.2452` populate を確認**

→ 「構造的限界」ではなく **「設定漏れ」**だった。**本 PR ([#316]) で PR-A.4.3 として実装完了** — `parakeet_engine.py` に Path 1.5 (pure RNNT/TDT 用) 追加、Section 1 smoke (margin 49×) + Section 2 stream pipeline (`webrtc × synthetic × on` で Hall.(post) 75% → 12.5%) + unit test 更新で validate 済。詳細は [`docs/research/parakeet-english-confidence-smoke-2026-06-11.md`](research/parakeet-english-confidence-smoke-2026-06-11.md)。

### Production user の選択 (engine と filter benefit の対応)

| User の engine 選択 | Confidence filter benefit |
|---|---|
| WhisperS2T / Parakeet_ja / Voxtral / Canary / Parakeet 英語 / ReazonSpeech / **qwen3asr** (language 明示) | ✅ 自動 hallucination 抑制 (default on で active、**7 engine 対応**で PR-A 系列完成) |
| qwen3asr の auto-detect mode (`--language=auto`) | ⚠ wrapper fallback path で fail-open → `--language en/ja/...` 明示で **PR-A.5.2 avg_logprob path** に乗せる、または Silero / TenVAD VAD で VAD 段階の非音声除去を併用推奨 ([Layer 3](#3-vad-backend-pre-engine))。 |

### Defense-in-depth の到達点

Phase 1 多段防御 epic 完了時点 (2026-06-11):

- **Layer 1** NoiseGate ([#291] MERGED): 連続帯域ノイズ抑制 ✅
- **Layer 2** TransientDetector ([#300]/[#304] MERGED): 拍手/タップ DSP 早期 drop、default `off` (PR-B calibration 結論) ⚠ Experimental
- **Layer 3** VAD backend ([#302]/[#307] MERGED): Silero / TenVAD で production-grade な speech 判定 ✅
- **Layer 4** EnergyGate ([#292] MERGED): 短時間 utterance の energy 判定 ✅
- **Layer 5** Confidence Filter (PR-A 系列、本 doc の範囲): **7 engine 対応**で完成 ✅ (PR-A.5.2 [#318] で qwen3asr を追加、PR-A 系列 すべての production engine が対応)

5 layer × 7 engine の組み合わせで、`webrtc × parakeet_ja` の歴史的 50% hallucination cell が **WebRTC 構成 + `--confidence-filter on` (default)** により 0% まで抑制される現状を達成。Parakeet 英語 (PR-A.4.3) も `webrtc × synthetic × on` で Hall.(post) 75% → 12.5%、ReazonSpeech (PR-A.5.1) は `webrtc × real × on` で 50% → 0% 完全解消 (Issue #295 元 motivation 完了)、**qwen3asr (PR-A.5.2) は両言語 (en/ja) margin +0.42〜+0.65 で Case A、engine 固有 fail-safe で Hall.(pre) 0% 全 cell** を実証 (詳細は [qwen3asr decision doc](research/qwen3asr-confidence-smoke-2026-06-12.md))。`silero` / `tenvad` (production-default VAD) では Hall.(pre) は元々 0%、本 filter は冗長安全網として機能。

---

## Comparison table

| Filter | Pipeline position | Default | Production-ready | Hallucination on real desk_tap (WebRTC × parakeet_ja) |
|---|---|---|---|---|
| NoiseGate | Pre-VAD | OFF (opt-in) | Yes | n/a (not its target) |
| **TransientDetector** | Pre-VAD | **OFF (experimental)** | **No** | **No improvement (50 % → 50 %, 0 pp)** |
| VAD backend | Core | **Silero (production)** | Silero / TenVAD ✅, WebRTC ⚠ (lightweight only) | **Silero / TenVAD already solve this case (0 % across all engines)** |
| EnergyGate | Post-VAD | ON (-45 dBFS) | Yes | Already at floor (engine-internal defense varies) |
| **Confidence Filter** | **Post-ASR** | **ON (default)** | **Yes** (whispers2t / parakeet_ja / voxtral / canary / parakeet 英語 / reazonspeech / **qwen3asr**, 7 engine、PR-A 系列完成) | **Drops the phantom transcript at the engine output**。検証 scope は engine 別: **PR-A.3 ([PR #312]) 54-cell calibration sweep** (旧 3 engine)、**PR-A.4.1 ([PR #313]) 12-cell** (Voxtral)、**PR-A.4.2 ([PR #315]) 12-cell** (Canary)、**PR-A.4.3 ([PR #316]) 12-cell** (Parakeet 英語)、**PR-A.5.1 ([PR #319]) 12-cell** (ReazonSpeech)、**PR-A.5.2 ([Issue #318]) 12-cell + 両言語 (en/ja) smoke** (qwen3asr)。webrtc × parakeet_ja で 50% → 0%、webrtc × voxtral × real で 50% → 0%、webrtc × parakeet_en × synthetic で 75% → 12.5%、webrtc × reazonspeech × real で 50% → 0% (Issue #295 元 motivation 完了)、**qwen3asr は両言語 margin +0.42〜+0.65 で Case A** を実測実証。 |

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
5. **`--confidence-filter=on`** (default since PR-A.1). Provides a
   final engine-internal defense for cases where the VAD lets non-
   speech through. **7 engine** (WhisperS2T / Parakeet_ja / Voxtral /
   Canary / Parakeet 英語 / ReazonSpeech / qwen3asr、PR-A 系列完成)
   で clean signal separation を実測実証:
   - WhisperS2T: `no_speech_prob` で **20×** (PR-A.0 smoke verify)
   - Parakeet_ja: `token_confidence_mean` で **167×** (PR-A.0 smoke verify、7 engine 全体で最大)
   - Voxtral: `avg_logprob` で **+1.0 margin** (PR-A.4.1 [#313 MERGED] smoke + 12 cell stream pipeline)
   - Canary: `token_confidence_mean` で **14.5×** (PR-A.4.2 [#315 MERGED] smoke + 12 cell stream pipeline)
   - Parakeet 英語: `token_confidence_mean` で **49×** (PR-A.4.3 [PR #316] smoke + 12 cell stream pipeline)
   - ReazonSpeech: `avg_logprob` で **margin +0.10-0.13** (PR-A.5.1 [PR #319] smoke int8/float32 + 12 cell stream pipeline、engine-specific threshold `-0.2`)
   - **qwen3asr**: `avg_logprob` で **両言語 margin +0.42〜+0.65** (PR-A.5.2 [Issue #318] smoke EN+JP + 12 cell stream pipeline、wrapper bypass + `repetition_penalty=1.1 + no_repeat_ngram_size=3`、engine-specific threshold `-0.3`)
   default は Silero / TenVAD users にとって essentially zero-cost、
   webrtc 構成では `webrtc × parakeet_ja` / `webrtc × voxtral × real` / `webrtc × reazonspeech × real` を
   3 つとも 50 %→0 % まで改善、`webrtc × parakeet_en × synthetic` で
   75 %→12.5 % を実証。Use `observe` to collect calibration data
   without dropping, or `off` to revert to PR-A.0 behavior.
   `LIVECAP_CONFIDENCE_FILTER=off` env var also disables for the entire
   session.
6. **Avoid `--vad-backend webrtc`** with `parakeet_ja` or `reazonspeech`
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
