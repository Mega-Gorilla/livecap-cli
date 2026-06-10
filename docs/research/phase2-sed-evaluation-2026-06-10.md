# Phase 2 SED model evaluation — EfficientAT (mn04_as) — 2026-06-10

> **Status: PR-D0 verdict = GO for PR-D1 with two caveats** (corrected per
> codex-review on #306):
> - **Runtime: Conditional PASS** — CPU production-device path is well
>   within budget, but GPU p95 = 32.8 ms misses the original 30 ms
>   ceiling by 9 % (see §3.1). Production device = CPU.
> - **License: PASS at Auto-download tier** (not Bundle OK) — the
>   upstream EfficientAT release does not explicitly grant a license on
>   the model weights, so the integration ships via
>   `torch.hub.load_state_dict_from_url` rather than packaging the
>   `.pt` file in the wheel (see §4.2).
>
> Accuracy and Safety dimensions clear without reservation, subject to
> the provisional-gate disclaimer (6-clip corpus is statistically weak;
> PR-D1 must record a corpus-expansion judgement).

This document is the authoritative deliverable of PR-D0, the off-line model
evaluation phase of the Phase 2 Sound-Event Detection epic
([Issue #305](https://github.com/Mega-Gorilla/livecap-cli/issues/305) v3).
It answers the four PR-D0 acceptance dimensions
(Accuracy / Safety / Runtime / License) and records the reproducibility
baseline for PR-D1 / PR-D2.

Raw evidence underlying every number in this document lives at
[`benchmark_results/sed/2026-06-10/`](../../benchmark_results/sed/2026-06-10/)
(`probabilities.csv` + `probabilities_full.npz` + `latency.csv` +
`analysis.json` + `analysis.md`), and the analysis script is reproducible
via the command at the end of this file.

---

## 0. Setup

| Field | Value |
|---|---|
| Date | 2026-06-10 |
| Issue | [#305 v3](https://github.com/Mega-Gorilla/livecap-cli/issues/305) |
| PR phase | PR-D0 (off-line evaluation, no `livecap_cli/` integration) |
| Model variant | `mn04_as` (EfficientAT, MobileNetV3-based) |
| Model checkpoint | `mn04_as_mAP_432.pt` (4.07 MB) |
| EfficientAT commit | `a425fdce92572e602a1d5634799bd9f1f2efa806` |
| Hardware | RTX 4090 (CUDA 12.8) + Windows 11 |
| Torch version | 2.9.1+cu128 |
| Corpus | `.tmp/non_speech_corpus/` (6 clips, private, gitignored) |
| Window unit | 1.0 s (Issue #305 v3 primary metric unit) |
| Decision unit | clip-level max across windows |
| Total windows | 63 across 6 clips |

The `mn04_as` variant was chosen as the **smallest viable model** under the
50 MB checkpoint constraint. If Dimension 3 had failed, `dymn04_as` (7.72 MB,
slightly higher mAP) was the secondary probe. Since `mn04_as` passes all
four dimensions, the heavier variants are deferred to PR-D1 (the
co-existence-with-DSP integration may benefit from a larger model when the
production corpus is expanded).

---

## 1. Dimension 1 — Accuracy

### 1.1 Class mapping (decision)

Reject signal targets (AudioSet display name → index):

| Class | AudioSet index |
|---|---|
| Hands | 61 |
| Finger snapping | 62 |
| Clapping | 63 |
| Applause | 67 |
| Door | 354 |
| Sliding door | 357 |
| Slam | 358 |
| Knock | 359 |
| Tap | 360 |
| Thump, thud | 460 |

Speech-like suppression set (used by policy 3, `target − speech_like`):

| Class | AudioSet index |
|---|---|
| Speech | 0 |
| Male speech, man speaking | 1 |
| Female speech, woman speaking | 2 |
| Child speech, kid speaking | 3 |
| Conversation | 4 |
| Narration, monologue | 5 |
| Singing | 27 |

Rationale:
- The target set centres on the `desk_tap` empirical motivation
  (`Knock` / `Tap` / `Thump, thud`) plus the applause family
  (`Clapping` / `Applause` / `Hands`) that PR-B's sweep flagged as the
  realistic field signal.
- The Issue #305 v3 spec listed `Door` as a target — included here for
  completeness, even though the corpus has no door-only clip (no FP risk).
- Speech-like is broader than Issue #305 v3's literal list (added
  Male/Female/Child speech sub-classes) because EfficientAT's multi-label
  output can fire on child classes without lighting up the `Speech` parent.

The pinned indices are cross-checked against the canonical AudioSet
`class_labels_indices.csv` by
[`test_class_mapping.py::test_indices_match_efficientat_csv`](../../tests/integration/sed/test_class_mapping.py).

### 1.2 Threshold policies

Three aggregation policies were compared (Issue #305 v3 spec):

- `max(target)` — simplest baseline
- `sum(target)` — sensitive to multiple targets co-firing
- `target − speech_like` — explicit speech-aware suppression

### 1.3 Clip-level max reject scores

| Clip | Kind | `max` | `sum` | `target − speech_like` |
|---|---|---|---|---|
| applause_5_claps | negative | **0.738** | **1.298** | **0.717** |
| desk_tap | negative | **0.207** | **0.296** | **0.175** |
| short_utterances_mixed | positive | 0.025 | 0.057 | -0.759 |
| normal_speech_neko | positive | 0.026 | 0.060 | -0.787 |
| applause_then_speech | positive | 0.027 | 0.063 | -0.725 |
| overlapping_applause_speech | positive | 0.160 | 0.354 | -0.659 |

Critical observation: **`overlapping_applause_speech` scores 0.160 under
`max`** (above many candidate thresholds) but `-0.659` under
`target − speech_like`. Policy 3 correctly recognises that the speech
channel dominates the applause, sparing the clip from over-rejection. This
is exactly the failure mode PR-B's DSP detector could not distinguish.

### 1.4 Reject-signal-level P/R sweep

(Where "positive class" = clip should be rejected, i.e. `kind == "negative"`.)

| Threshold | `max` P / R | `sum` P / R | `target − speech_like` P / R |
|---|---|---|---|
| 0.01 | 0.33 / 1.00 | 0.33 / 1.00 | 1.00 / 1.00 |
| 0.05 | 0.67 / 1.00 | 0.33 / 1.00 | 1.00 / 1.00 |
| 0.10 | 0.67 / 1.00 | 0.67 / 1.00 | 1.00 / 1.00 |
| 0.15 | 0.67 / 1.00 | 0.67 / 1.00 | 1.00 / 1.00 |
| 0.20 | 1.00 / 1.00 | 0.67 / 1.00 | 1.00 / 0.50 |
| 0.30 | 1.00 / 0.50 | 0.50 / 0.50 | 1.00 / 0.50 |
| 0.50 | 1.00 / 0.50 | 1.00 / 0.50 | 1.00 / 0.50 |

### 1.5 Provisional gate verdict — **PASS**

| Field | Value |
|---|---|
| Target clip | `desk_tap` |
| Precision floor | 0.70 |
| Recall floor | 0.50 |
| Outcome | **PASS** |
| Passing policies | `max`, `target_minus_speech` |
| Chosen policy | `target_minus_speech` |
| Chosen threshold | 0.150 |
| Chosen precision | 1.00 |
| Chosen recall | 1.00 |
| `desk_tap` flagged at chosen threshold | ✅ |

**Recommended production preset for PR-D1**: policy
`target_minus_speech`, threshold in the band `[0.05, 0.15]` (centre 0.10).
Under that band, every clip in the 6-corpus is correctly classified —
both the negative reject signals (`applause_5_claps`, `desk_tap`) clear the
threshold, while every positive clip (including the adversarial overlap
case) stays below.

### 1.6 Class-level metrics @ threshold 0.05 (two-axis report, Issue #305 v3)

Target classes (we want high precision when these fire):

| Class | TP | FP | FN | TN | Precision | Recall |
|---|---|---|---|---|---|---|
| Hands | 1 | 0 | 1 | 4 | 1.00 | 0.50 |
| Finger snapping | 1 | 1 | 1 | 3 | 0.50 | 0.50 |
| Clapping | 1 | 1 | 1 | 3 | 0.50 | 0.50 |
| Applause | 0 | 0 | 2 | 4 | 0.00 | 0.00 |
| Door / Sliding door / Slam | 0 | 0 | 2 | 4 | 0.00 | 0.00 |
| Knock | 1 | 0 | 1 | 4 | **1.00** | 0.50 |
| Tap | 1 | 1 | 1 | 3 | 0.50 | 0.50 |
| Thump, thud | 0 | 0 | 2 | 4 | 0.00 | 0.00 |

Speech-like classes (FP/TN flip semantics — see note below):

| Class | TP | FP | FN | TN | Note |
|---|---|---|---|---|---|
| Speech | 0 | 4 | 2 | 0 | Fires on all positive clips (correct behaviour) and also on `overlapping_applause_speech` (correct) |
| Male speech, man speaking | 0 | 4 | 2 | 0 | Idem |
| Narration, monologue | 0 | 4 | 2 | 0 | Idem |
| Female speech / Conversation / Singing | 0 | 0 | 2 | 4 | Silent on this corpus (no female / sung clips) |

> The "speech-like" P/R framing treats `kind=="negative"` (reject signals)
> as the positive class, so on speech-like classes a high FP count is
> *expected and desirable*: it confirms the model is correctly recognising
> the speech channel of positive clips. The class table is reported
> verbatim from the metric module for transparency rather than as a
> production-relevant accuracy claim.

**Key class-level takeaways:**

1. The model's confidence on `Knock` (the desk_tap target class) is moderate
   — clip-level max 0.207 — but cleanly separated from speech-like clip max
   ~0.026. This 8× separation is what makes `target_minus_speech` work.
2. `Applause` / `Slam` / `Thump, thud` do **not** fire even when the clip
   contains the corresponding event (e.g. `applause_5_claps` lights up
   `Hands` and `Clapping` instead, not `Applause`). The 1-second window
   appears to bias the model toward percussion / contact-sound classes
   rather than crowd-event classes.
3. PR-D1 must consider: should we add `Hands` to the target set? It is
   already pinned and contributes to `max(target)`, but the implication is
   that EfficientAT's 1-second classification semantics are a noisier shadow
   of its 10-second AudioSet training regime. This is acceptable for PR-D0
   (the provisional gate passes), but PR-D1's expanded corpus may need to
   re-evaluate window length.

### 1.7 Accuracy verdict — PASS (provisional)

The Issue #305 v3 Accuracy gate is **satisfied** by the chosen policy.
The verdict is marked **provisional** per Issue #305 v3 rule because the
underlying corpus is only 6 clips:

> PR-D1 must record a corpus-expansion judgement (ESC-50 / FSD50K subset
> vs status quo) before promoting any production default.

---

## 2. Dimension 2 — Safety

Floor: `speech_recall ≥ 0.95` on the positive corpus,
`short_utterance_recall == 1.00` on `short_utterances_mixed`.

Under the chosen `target_minus_speech` policy at threshold 0.10:

| Positive clip | Clip-max score | Flagged as reject? |
|---|---|---|
| short_utterances_mixed | -0.759 | ❌ (correctly retained) |
| normal_speech_neko | -0.787 | ❌ (correctly retained) |
| applause_then_speech | -0.725 | ❌ (correctly retained) |
| overlapping_applause_speech | -0.659 | ❌ (correctly retained) |

| Metric | Value | Floor | Verdict |
|---|---|---|---|
| `speech_recall` | 4 / 4 = 1.00 | ≥ 0.95 | ✅ PASS |
| `short_utterance_recall` | 1 / 1 = 1.00 | == 1.00 | ✅ PASS |

### 2.1 Safety verdict — PASS

The `target − speech_like` policy is the architectural reason this passes:
even when raw target probability would over-fire (as on
`overlapping_applause_speech` where `max(target) = 0.160`), the
subtractive policy collapses the score to deeply negative territory and
the clip stays out of the reject set.

---

## 3. Dimension 3 — Runtime

Measured on RTX 4090 + Ryzen-class host, 50 iterations per percentile,
after 3 warmup iterations (`benchmarks.sed.latency.measure_all_axes`).

| Axis | Measured | Ceiling | Verdict |
|---|---|---|---|
| Checkpoint disk size | 4.07 MB (`mn04_as_mAP_432.pt`) | ≤ 50 MB | ✅ 12× under |
| Installed dep delta vs `engines-torch` | 0 bytes | ≤ 150 MB | ✅ (manual clone, no pip add) |
| Model parameter bytes | 3.93 MB | reference only | — |
| Runtime peak memory (tracemalloc) | 6.68 MB | ≤ 200 MB | ✅ 30× under |
| Cold start (load + first inference) | 74 ms | reference only | — |
| **CPU p50** | 26.9 ms | reference only | — |
| **CPU p95** | 29.0 ms | ≤ 100 ms | ✅ 3.4× under |
| GPU p50 | 30.5 ms | reference only | — |
| **GPU p95** | **32.8 ms** | **≤ 30 ms** | **❌ FAIL (9 % over)** |

### 3.1 GPU latency — fails the original ceiling

GPU p95 = 32.8 ms exceeds the Issue #305 v3 Go/no-go ceiling of 30 ms by
roughly 9 %. **This axis fails the originally pinned criterion**
(codex-review on #306 was correct to flag this as an inconsistency
between the verdict and the rule). Two observations are nonetheless
relevant for the production deployment plan:

1. **CPU runs faster than GPU at this model scale.** With only 3.9 M
   parameters, the CUDA kernel launch overhead (sample upload + memory
   copy + small-batch compute) dominates the pure computation.
   Measured CPU p95 (29.0 ms) is lower than GPU p95 (32.8 ms).
2. **The CPU axis clears the 100 ms streaming budget by 3.4×.** Production
   inference can therefore run on CPU within budget regardless of CUDA
   availability.

Implication: the GPU criterion is **failed but not blocking**. Production
inference will use CPU by default — for which the budget is met by a wide
margin. The GPU number is a research data point, not a production
constraint; PR-D1 should record the device choice rationale in CLI flag
documentation when the SED detector is wired up.

### 3.2 Runtime verdict — Conditional PASS (CPU device path)

| Axis | Result |
|---|---|
| Checkpoint disk size | ✅ PASS |
| Installed dependency delta | ✅ PASS |
| Runtime peak memory | ✅ PASS |
| CPU p95 (production target device) | ✅ PASS |
| **GPU p95 (original criterion)** | **❌ FAIL (9 % over 30 ms ceiling)** |

**Verdict: Conditional PASS** — five axes clear, the GPU axis fails the
original 30 ms criterion. The conditionality is that production device
selection moves to CPU (where the much looser 100 ms budget is satisfied
by 3.4×). A strict reading of the Issue #305 v3 gate would treat this as
a partial failure; this document records it honestly so PR-D1 can either
(a) accept CPU-only production with the criterion as-stated, or
(b) update the Issue #305 v3 GPU criterion to reference-only for
sub-10M-parameter SED models in a follow-up amendment.

---

## 4. Dimension 4 — License

### 4.1 Three-layer audit

| Layer | Asset | License | Verified | AGPL-3.0-only compatible? |
|---|---|---|---|---|
| 1. Code | EfficientAT repo (`fschmid56/EfficientAT`) | **MIT** | `LICENSE` file in clone (Copyright 2022 Florian Schmid) | ✅ Yes |
| 2. Checkpoint | `mn04_as_mAP_432.pt` (distributed via GitHub Releases at `v0.0.1`) | **Not explicitly stated in the upstream release**; the LICENSE covers the code repository, the README does not separately license the model weights. Standard convention treats the GitHub release artefact as inheriting the MIT code license, but this is not contractually pinned. | GitHub release page + LICENSE file | ✅ Yes for auto-download; **insufficient evidence to bundle** without further upstream clarification |
| 3. Training data | Google AudioSet (used to train the checkpoint) | **CC BY 4.0** | https://research.google.com/audioset/ + Phase 1 research | ✅ Yes (commercial use + redistribution allowed with attribution) |

### 4.2 License outcome category (Issue #305 v3 4-classification)

**Outcome: Auto-download OK (Category 2).**

The PR's prior draft listed this as **Bundle OK**; codex-review on #306
flagged that the checkpoint license is only inferred, not stated by the
upstream release notes. Issue #305 v3 requires explicit verification per
layer, so we downgrade to **Auto-download OK** — which matches what the
actual implementation already does (`torch.hub.load_state_dict_from_url`
fetches the checkpoint on first use rather than shipping it).

| Category | Match? | Why |
|---|---|---|
| Bundle OK | ❌ | Checkpoint license is not explicitly granted by the upstream release; bundling without that grant would carry redistribution risk under AGPL-3.0-only. |
| **Auto-download OK** | ✅ | `torch.hub.load_state_dict_from_url` fetches the artefact from the upstream URL on first use; the user receives the artefact directly from the upstream source under whatever terms that source is published. PR-D1's implementation already follows this pattern. |
| Manual user-provided only | not required | |
| NG | — | |

### 4.3 PR-D1 attribution obligations (when the model is auto-downloaded)

When PR-D1 adds the EfficientAT dependency, the following attribution
must accompany the distribution (e.g. in a `THIRD_PARTY_NOTICES.md`),
alongside documentation that the model is fetched from the upstream
release URL on first inference:

```
EfficientAT — MIT License, Copyright (c) 2022 Florian Schmid
https://github.com/fschmid56/EfficientAT

This product downloads pre-trained model checkpoints from the EfficientAT
GitHub releases on first use. The pre-trained weights are derived from
the Google AudioSet dataset (https://research.google.com/audioset/),
licensed under CC BY 4.0.
```

### 4.4 License verdict — PASS (Auto-download tier)

EfficientAT can ship with livecap-cli under the Auto-download OK tier
without further upstream license clarification. If a future need arises
to **bundle** the checkpoint into the release (e.g. for fully offline
installs), PR-D1 must first obtain explicit upstream confirmation that
the model weights are MIT-licensed (or open an issue against
`fschmid56/EfficientAT` requesting the clarification).

---

## 5. Overall Go/no-go decision

| Dimension | Verdict | Notes |
|---|---|---|
| 1. Accuracy | ✅ PASS (provisional) | `target_minus_speech` policy, threshold ~0.10, P=R=1.0 on 6 clips |
| 2. Safety | ✅ PASS | All 4 positive clips correctly retained |
| 3. Runtime | ⚠ **Conditional PASS** (CPU device path; GPU p95 fails 30 ms ceiling) | CPU p95 29 ms (3.4× under budget); GPU p95 32.8 ms (9 % over). Production device = CPU. |
| 4. License | ✅ PASS (Auto-download tier) | Checkpoint license not explicitly granted upstream → ship via `torch.hub` auto-download, not bundling |

### Verdict: **GO with two caveats — proceed to PR-D1 under the Auto-download + CPU device path.**

The Accuracy / Safety dimensions clear without reservation. The Runtime
dimension passes conditionally (CPU production device path is well
within budget; GPU does not meet the original 30 ms criterion but the
model is intentionally CPU-friendly at this scale). The License
dimension passes at the Auto-download OK tier rather than Bundle OK
because the upstream release does not explicitly grant a license on the
model weights — auto-download matches both the legal evidence we have
and the implementation already in use (`torch.hub.load_state_dict_from_url`).

EfficientAT `mn04_as` is the recommended Phase 2 SED model. PR-D1 should:

1. Add EfficientAT as a pip extra (`livecap-cli[engines-sed]`) using a
   pinned commit hash (`a425fdce92572e602a1d5634799bd9f1f2efa806`);
   the checkpoint is **auto-downloaded** on first use, not bundled.
2. Implement `livecap_cli/audio/sed_detector.py` mirroring the DSP
   detector's three-mode interface (`off` / `observe` / `on`),
   defaulting to `off` until PR-D2 calibration.
3. Default device = CPU; preserve CUDA option behind `--sed-device`.
4. Wire the `target_minus_speech` policy with threshold 0.10 as the
   initial production default candidate (PR-D2 will calibrate against
   the larger sweep matrix).
5. Add `THIRD_PARTY_NOTICES.md` with the attribution stub above.
6. Make a corpus-expansion judgement (ESC-50 / FSD50K subset vs status
   quo) — this is the **mandatory** Issue #305 v3 follow-up obligation.

---

## 6. PR-D1 risk register

Risks that PR-D1 must explicitly address (so they are not discovered
late in PR-D2):

| Risk | Likelihood | Impact | Mitigation in PR-D1 |
|---|---|---|---|
| Threshold drift in production audio (vs lab clips) | Medium | Could re-introduce hallucination | Sweep harness via `benchmarks/non_speech_filter/sweep.py` with SED preset |
| 1-second window biases model toward percussion classes (`Hands`/`Knock`) over event classes (`Applause`) | Already observed | Reject set may need re-tuning when corpus expands | Re-evaluate at 2 s / 5 s windows in PR-D1; metric calculation unit may need a v4 |
| EfficientAT's `helpers/utils.py` reads `class_labels_indices.csv` at import time relative to cwd | Low | Brittle integration | PR-D1 will vendor the CSV (or pin a fork) instead of relying on chdir |
| Streaming chunk boundary: events split across windows | Medium | Misses on short transients | Use overlapping windows (0.5 s hop on 1 s window) when `--sed-detector=on` |
| GPU p95 (33 ms) vs CPU p95 (30 ms) inversion | Confirmed | Adds a UX question | Document device default = CPU, allow CUDA opt-in for users with bigger headroom |
| AudioSet attribution missing from livecap-cli distributions | High if forgotten | License obligation | `THIRD_PARTY_NOTICES.md` is part of the PR-D1 deliverable definition |
| Corpus statistical weakness (provisional gate) | Known | Verdict numbers are not generalisable | Issue #305 v3 §"PR-D1" already mandates corpus-expansion judgement |
| Existing DSP detector co-existence | Medium | Two filters firing at the same VAD slot | Co-existence design in PR-D1: SED runs after DSP, DSP becomes opt-in by PR-D2 |
| `mn04_as` could be too small for noisier real-world recordings | Medium | Reduced production margin | PR-D1 should benchmark `dymn04_as` as a fallback against the same corpus |

---

## 7. Reproducibility

Exact reproduction of every number in this document:

```powershell
# 1. Set up scratch directory (Issue #305 v3 artifact policy keeps the corpus
#    and EfficientAT clone gitignored under .tmp/).
git clone https://github.com/fschmid56/EfficientAT.git .tmp/EfficientAT
# Pin the commit so a future repo rev does not break this reproduction.
git -C .tmp/EfficientAT checkout a425fdce92572e602a1d5634799bd9f1f2efa806

# 2. Provide the private non-speech corpus at .tmp/non_speech_corpus/
#    (existing PR-B path; corpus contents remain gitignored).
$env:LIVECAP_NON_SPEECH_CORPUS_DIR = (Resolve-Path .tmp/non_speech_corpus)
$env:LIVECAP_SED_EFFICIENTAT_PATH  = (Resolve-Path .tmp/EfficientAT)

# 3. Generate the raw evaluation artefacts.
uv run python -m benchmarks.sed `
    --model mn04_as `
    --output-dir benchmark_results/sed/2026-06-10/ `
    --latency-iters 50

# 4. Run the post-hoc analysis (writes analysis.json + analysis.md).
uv run python -m benchmarks.sed.analyze `
    --input-dir benchmark_results/sed/2026-06-10/
```

The four committed evidence files reproduce as:

| File | SHA stability | Purpose |
|---|---|---|
| `probabilities.csv` | exact (deterministic inference) | Per-clip × per-window class-level summary |
| `probabilities_full.npz` | exact | Raw per-window 527-vector tensors |
| `latency.csv` | varies (hardware-dependent) | 5-axis runtime measurement |
| `metadata.json` | varies (timestamp + path) | Reproducibility metadata |
| `analysis.json` | exact (function of probabilities) | Machine-readable verdict |
| `analysis.md` | exact (function of probabilities) | Human-readable tables |

---

## 8. Related artefacts

- Phase 2 SED epic: [Issue #305 v3](https://github.com/Mega-Gorilla/livecap-cli/issues/305)
- PR-B calibration (the empirical evidence that motivated this epic): [#304](https://github.com/Mega-Gorilla/livecap-cli/pull/304)
- DSP limit record (Phase 1 final disposition): [`docs/benchmarks/calibration-results-2026-06-07.md`](../benchmarks/calibration-results-2026-06-07.md)
- User-facing filter reference (will be updated in PR-D2): [`docs/audio-filter-reference.md`](../audio-filter-reference.md)
- Raw evaluation outputs: [`benchmark_results/sed/2026-06-10/`](../../benchmark_results/sed/2026-06-10/)
- Implementation: [`benchmarks/sed/`](../../benchmarks/sed/)
