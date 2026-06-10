# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Epic #64 (livecap-cli refactoring) - completion of all 6 phases.

This represents the completion of a major refactoring effort spanning 6 phases.
Package renamed from `livecap-core` to `livecap-cli`.

### Added

#### Engine confidence signal schema (Issue [#308] PR-A.0 / Phase 1 Layer 3)

- **`EngineConfidence` / `TranscriptionResult` dataclasses** added to
  `livecap_cli/engines/base_engine.py`:
  - `EngineConfidence`: `Optional[float]` fields for `no_speech_prob`,
    `avg_logprob`, `compression_ratio`, `token_confidence_mean`, plus a
    `raw: dict[str, float]` overflow bucket for engine-specific signals.
    `is_available` property returns `True` when at least one signal field
    is non-`None` (PR-A.1 filter precondition).
  - `TranscriptionResult`: `text`, `confidence`, `engine_confidence`
    (default = all-None `EngineConfidence()`). `__iter__` yields
    `(text, confidence)` so the legacy `text, confidence = result` tuple
    unpacking pattern continues to work — no caller migration required
    for existing engine adapters.
  - Both dataclasses are `frozen=True` (immutable) and re-exported via
    `livecap_cli.engines.__all__`.
- **Engine adapter expose paths** (engine-by-engine breakdown):
  - `whispers2t_engine.py`: extracts `no_speech_prob`, `avg_logprob`, and
    `compression_ratio` from the CTranslate2 backend result dict via the
    new pure-function `_extract_engine_confidence()`. Handles both
    top-level signals (current backend shape) and per-segment lists
    (legacy / future shape). Real-machine smoke verify (`normal_speech_neko`
    vs `desk_tap` vs `applause_5_claps`): `no_speech_prob` = 0.036 (speech)
    vs 0.63-0.66 (non-speech) — clean separation usable by the PR-A.1
    filter. Existing `confidence: float` calculation untouched.
  - `parakeet_engine.py`: hybrid `EncDecHybridRNNTCTCBPEModel` is now
    explicitly switched to the CTC decoder via the new
    `_configure_decoding_with_confidence()` helper (see "Changed"
    section below). Real-machine `token_confidence_mean` separation:
    0.01-0.10 (speech) vs 0.0000029-0.0003 (non-speech) — 3-4 orders
    of magnitude, usable by the PR-A.1 filter with a `> 0.005` threshold.
  - `reazonspeech_engine.py`: returns `EngineConfidence()` (all `None`)
    with a docstring `Note` explaining that sherpa-onnx Python bindings
    for transducer models do not expose per-token scores. Users who
    require engine-level hallucination defense are pointed to Silero /
    TenVAD backends. Real-machine smoke verify confirmed
    `is_available is False` on all corpus clips.
  - `qwen3asr_engine.py`, `voxtral_engine.py`, `canary_engine.py`,
    `benchmarks/non_speech_filter/mock_engine.py` (`MockEngine`): no-op
    migration — return `TranscriptionResult(text=..., confidence=...)`
    with default empty `EngineConfidence`. PR-A.1 filter will treat
    these engines as fail-open (`is_available is False` → pass-through).
- **Caller migration** (defensive `hasattr`-based dispatch retained for
  legacy `Tuple[str, float]` mocks):
  - `shared_engine_manager.py` `_process_request` uses `hasattr(result,
    'text')` primary branch, keeps tuple/dict legacy branches.
  - `benchmarks/non_speech_filter/mock_engine.py` `InstrumentedEngine`
    accepts both `TranscriptionResult` and the historical tuple shape so
    benchmark harnesses keep working unmodified.
  - `livecap_cli/transcription/stream.py` `TranscriptionEngine` Protocol
    return type updated to `EngineTranscriptionResult` (runtime import
    alias from `engines.base_engine` — kept out of `TYPE_CHECKING` so
    `typing.get_type_hints()` resolves it). Stream call sites at lines
    546 / 618 / 767 use `text, confidence = ...` unpacking and continue
    to work via the dataclass `__iter__`.
- **New unit tests** (do not require ASR models — pure-function pins
  the extraction logic):
  - `tests/core/engines/test_engine_confidence_schema.py` (17 cases):
    default values, `is_available` semantics across all four signal
    fields, frozen-mutation rejection, `__iter__` yields exactly two
    items (engine_confidence excluded from tuple unpacking), public
    re-export coverage.
  - `tests/core/engines/test_whispers2t_confidence_extraction.py`
    (17 cases): mock CTranslate2 result dicts covering top-level + per-
    segment mean aggregation, missing fields, non-numeric / `None`
    values, and non-dict segment entries.
  - `tests/core/engines/test_parakeet_confidence_extraction.py`
    (12 cases): `FakeHypothesis` mock pinning the token-confidence
    primary path and edge cases (string input, empty list, non-numeric
    values, completely empty hypothesis).
  - `tests/core/engines/test_parakeet_return_hypotheses.py` (5 cases):
    pins that `return_hypotheses=True` is passed to NeMo and that the
    legacy `score / len(y_sequence)` fallback is no longer populated.
  - `tests/core/engines/test_parakeet_decoding_strategy.py` (5 cases):
    pins hybrid-model detection, CTC switch via `decoder_type='ctc'`,
    fallback to strategy-only on `TypeError`, and exception resilience.
  - `tests/transcription/test_transcription_engine_protocol.py`
    (2 cases): pins that `typing.get_type_hints()` resolves
    `EngineTranscriptionResult` to `engines.base_engine.TranscriptionResult`
    rather than the `transcription.result` dataclass of the same name.

The new signals feed into PR-A.1 (`--confidence-filter {off,observe,on}`
post-filter) and PR-A.3 (calibration + production default). Together with
PR-B calibration (PR [#304]) and the PR #307 audio-filter-reference
rewrite, this lands the Phase 1 Layer 3 schema required to close Issue
[#295].

### Changed

#### Engine confidence filter — default ON (Issue [#308] PR-A.1)

Adds the post-ASR `livecap_cli.transcription.confidence_filter` module
that watches `engine_confidence` (PR-A.0) and silently drops outputs the
engine itself judged as non-speech, before they reach the subtitle stream.
Default is **on** for all realtime sessions.

- **Before** (PR-A.0): every engine output reached the subtitle stream,
  even when `no_speech_prob` was high (WhisperS2T) or
  `token_confidence_mean` was near zero (Parakeet_ja). The PR-B 144-cell
  matrix showed `webrtc × parakeet_ja × desk_tap` hallucinated 50 % of
  the time.
- **After** (PR-A.1): `StreamTranscriber` calls `apply_filter()` on the
  3 call sites (sync L566 / async L638 / interim L787). For
  `--confidence-filter on` (default), rejected outputs become `None`
  drops with a structured INFO log; the subtitle stream sees nothing.
  Real-machine smoke verify on the 6-clip PR-B corpus produced 100 %
  classification (all speech clips passed, both non-speech clips dropped
  on both whispers2t and parakeet_ja).
- **Migration**: existing scripts keep the same flags. To restore the
  previous behavior, pass `--confidence-filter off` or export
  `LIVECAP_CONFIDENCE_FILTER=off`. Engines that do not expose confidence
  signals (`reazonspeech`, `qwen3asr`, `voxtral`, `canary`) are
  pass-through regardless of the flag (fail-open by design).
- **Side effects**:
  - Per-engine thresholds (`whispers2t no_speech_prob > 0.5`,
    `parakeet_ja token_confidence_mean < 0.005`) are baked in from the
    PR-A.0 verify values; PR-A.3 will revisit them after a full 144-cell
    sweep.
  - The `--confidence-filter observe` mode emits the same structured
    decision log **for both pass and reject decisions** (codex-review
    #310 Item 4) — PR-A.3 calibration needs the speech-side
    `engine_confidence` distribution as well, not just the reject side,
    to evaluate threshold margins and speech-recall safety. The `on`
    mode keeps logging reject only to avoid production log spam. Log
    payload is JSON (stable schema documented in `_decision_to_dict()`
    of `confidence_filter.py`) so PR-A.3 parsers can read it as JSONL.
  - The `StreamTranscriber.__init__` gained a `filter_config:
    Optional[FilterConfig] = None` parameter; `None` constructs the
    default (on) at instantiation time. Direct API users who want the
    old behavior should pass `FilterConfig(mode="off")` explicitly.
  - `benchmarks/non_speech_filter/pipeline.py::build_pipeline` defaults
    to `FilterConfig(mode="off")` so existing sweep baselines remain
    bit-identical; PR-A.3 will pass `FilterConfig(mode="on")` to
    measure filter impact on the cell matrix.
  - **Scope clarification (codex-review #310 Item 3)**: this PR exposes
    `filter_config` on `build_pipeline()` only. Adding a
    `confidence_filter` axis to `benchmarks/non_speech_filter/sweep.py`
    (so that the existing preset/backend/engine matrix is multiplied by
    `{off, observe, on}`) is **deferred to PR-A.3**, together with the
    full 144-cell sweep run. The pipeline-level hook here is sufficient
    for PR-A.3 to construct sweep cells programmatically.
  - A startup INFO log line (`"Confidence filter: ON (...)"`) makes the
    active mode visible at every session start.

#### Parakeet_ja decoder strategy: RNNT greedy → CTC greedy_batch (Issue [#308] PR-A.0)

Investigation during PR #309 smoke verify uncovered that the
`nvidia/parakeet-tdt_ctc-0.6b-ja` checkpoint is an
`EncDecHybridRNNTCTCBPEModel` whose RNNT decoder (NeMo default) does not
implement `token_confidence`. The CTC decoder does. To make
`engine_confidence` actually populated for `parakeet_ja`, the adapter now
switches to the CTC decoder on `load_model`.

- **Before**: `parakeet_ja` used the RNNT decoder with `strategy=greedy`
  and the old `confidence_cfg` block — which NeMo silently rejected on
  the current version (`preserve_frame_confidence=True` requires
  `preserve_alignments=True`), so `token_confidence` was always `None`
  and the old `score / len(y_sequence)` fallback returned an
  empirically-inverted signal (speech `-71.5` vs applause `-47.3`).
- **After**: `_configure_decoding_with_confidence()` detects the hybrid
  model via `hasattr(self.model, 'cur_decoder')` and switches to
  `decoder_type='ctc'` with `strategy=greedy_batch`,
  `greedy.preserve_frame_confidence=True`, and a full `confidence_cfg`.
  `token_confidence_mean` is now populated with clean speech-vs-noise
  separation (0.01-0.10 vs 0.0000029-0.0003). A 3-stage fallback path
  protects against older NeMo versions and the non-hybrid English
  `parakeet` model.
- **Migration**: `EngineMetadata.default_params["parakeet_ja"]
  ["decoding_strategy"]` updated from `"greedy"` to `"greedy_batch"`
  (single source of truth, surfaced by GUI / diagnostics / docs). The
  English `parakeet` (pure RNNT) default remains `"greedy"`. Users who
  hard-coded `decoding_strategy="greedy"` on `parakeet_ja` will keep
  working (CTC greedy is slower but functional, NeMo emits a
  `greedy_batch` recommendation warning).
- **Side effects** measured on RTX 4090 with the
  `.tmp/non_speech_corpus/` 6-clip set:
  - **Latency** improves: CTC + `greedy_batch` runs 1.83× faster on the
    speech clip than the old RNNT `greedy` path (p50 81.4 ms vs
    149.8 ms).
  - **Transcription text** is preserved on 4/6 clips, slightly improved
    on 1/6 (`applause_5_claps` hallucinates fewer characters), and
    differs by 1 hiragana on 2/6 (e.g. 「とんと」 → 「とんど」). Not
    a regression for production usage; documented in
    `docs/research/parakeet-ja-confidence-spec-2026-06-10.md`.
- **Score fallback removed**: the previous
  `score / len(y_sequence) → avg_logprob` path inside
  `_extract_engine_confidence` is gone. When `token_confidence` is not
  available (older NeMo, non-hybrid model, or fallback path), the
  function returns `EngineConfidence()` honestly. The smoke-verified
  signal inversion made the old fallback actively harmful for the
  PR-A.1 filter.

This is an **engine behavior change** for `parakeet_ja`, but it
strengthens (not regresses) production behavior on every measured axis:
faster, comparable text, and a now-usable confidence signal.

#### Phase 2 SED model evaluation harness (Issue [#305] PR-D0)

- **New `benchmarks/sed/` package (research-only off-line evaluation;
  does not touch `livecap_cli/`)**:
  - `class_mapping.py` — pins the AudioSet 527-class taxonomy mapping for
    livecap reject signals. Defines `TARGET_CLASSES` (Hands / Finger
    snapping / Clapping / Applause / Door / Sliding door / Slam / Knock
    / Tap / Thump, thud — 10 classes) and `SPEECH_LIKE_CLASSES` (Speech
    family + Singing — 7 classes). Implements the three Issue #305 v3
    threshold policies (`max`, `sum`, `target_minus_speech`).
  - `inference.py` — loads EfficientAT pretrained models
    (`mn04_as` / `dymn04_as` / `dymn10_as`), resamples 16 kHz audio to
    32 kHz, slices into 1-second windows (Issue #305 v3 primary metric
    unit), and returns per-window 527-dim sigmoid probability matrices.
  - `metrics.py` — clip-level confusion-matrix metrics with hand-pinned
    semantics: class-level + reject-signal-level (Issue #305 v3 two-axis
    report), provisional-gate verdict (`precision ≥ 0.70` AND
    `recall ≥ 0.50` AND target clip flagged at the chosen threshold).
  - `latency.py` — 5-axis runtime measurement (checkpoint size, installed
    dep delta vs `engines-torch` baseline, runtime peak memory via
    `tracemalloc`, CPU + GPU p50/p95 latency, cold-start) per Issue #305
    v3 Dimension 3 refinement.
  - `orchestrator.py` — full evaluation pipeline (corpus → inference →
    CSVs + NPZ + JSON metadata).
  - `analyze.py` — post-hoc analysis: threshold sweep, class-level
    summary tables, provisional-gate verdict, `analysis.{json,md}` for
    decision-doc paste-in.
  - `cli.py` + `__main__.py` — `python -m benchmarks.sed` entry point.
  - `README.md` — manual EfficientAT setup, env vars, command reference.
- **New `tests/integration/sed/` (23 tests, `sed_evaluation` marker)**:
  - `test_class_mapping.py` (12 tests) — AudioSet index integrity, three
    policy semantics, validation; the
    `test_indices_match_efficientat_csv` test cross-checks the pinned
    indices against the canonical AudioSet CSV when EfficientAT is
    cloned.
  - `test_metrics.py` (10 tests) — synthetic 4-clip corpus with
    hand-derived precision/recall, gate truth-table (pass / precision
    fail / recall fail / target-not-flagged).
  - `test_inference_smoke.py` (1 test) — env-gated 1-clip smoke
    verifying `(n_windows, 527)` output shape; skipped automatically
    when the EfficientAT clone is absent.
  - New `sed_evaluation` pytest marker declared in `pyproject.toml`.
- **New `docs/research/phase2-sed-evaluation-2026-06-10.md` (~430
  lines)** — 4-dimension decision document covering Accuracy / Safety /
  Runtime / License (verdicts honest after codex-review on #306):
  - Accuracy: PASS (provisional). `target_minus_speech` policy at
    threshold ~0.10 yields precision=1.0, recall=1.0 on the 6-clip
    corpus; the critical `overlapping_applause_speech` case is correctly
    retained (`max(target)=0.16` would over-fire, but
    `target − speech_like = -0.66` correctly suppresses).
  - Safety: PASS. speech_recall = 1.00, short_utterance_recall = 1.00.
  - Runtime: **Conditional PASS** (CPU production-device path). CPU p95
    = 29.0 ms (3.4× under the 100 ms budget), checkpoint 4.07 MB
    (12× under 50 MB), runtime peak 6.68 MB (30× under 200 MB).
    **GPU p95 = 32.8 ms misses the original 30 ms ceiling by 9 %** —
    documented honestly rather than papered over; CPU runs faster than
    GPU at this 3.9 M-parameter scale, so production device = CPU and
    the CPU budget is satisfied.
  - License: PASS at the **Auto-download OK tier** (not Bundle OK —
    corrected after codex-review). The upstream EfficientAT release
    does not explicitly grant a license on the model weights, so the
    integration ships the checkpoint via `torch.hub` auto-download
    rather than packaging the `.pt` file; this matches both the legal
    evidence we have and the implementation already in use.
    Attribution stub recorded for PR-D1's `THIRD_PARTY_NOTICES.md`.
- **New `benchmark_results/sed/2026-06-10/` (committed per Issue #305 v3
  artifact policy)**: `probabilities.csv`, `probabilities_full.npz`,
  `latency.csv`, `metadata.json`, `analysis.json`, `analysis.md`.
- **`.gitignore` update**: changed `benchmark_results/` to
  `benchmark_results/*` with `!benchmark_results/sed/` exception so the
  PR-D0 evidence is committed while other benchmark outputs remain
  ignored. Added `.tmp/` to ignore the EfficientAT clone and any
  research scratch.
- **Issue #305 v2 → v3 body update** with six clarifications: metric
  calculation unit (window primary / clip-level max decision unit),
  license outcome 4-classification (Bundle OK / Auto-download OK /
  Manual user-provided only / NG), artifact commit policy, accuracy
  provisional-gate disclaimer, runtime constraint detailing
  (checkpoint / installed dep / runtime peak memory split), class-level
  + reject-signal-level two-axis metric report.
- **Scope discipline**: this PR does **not** modify any file under
  `livecap_cli/`. SED pipeline integration is PR-D1; default-decision
  is PR-D2; DSP detector disposition is PR-D2.

Verification: `pytest tests/integration/sed/` → 23 passed, 0 failed.
PR-relevant regression
(`tests/audio tests/integration/non_speech_filter tests/integration/vad
tests/transcription tests/core/cli tests/audio_sources`) → 307 passed,
5 skipped (env-gated), 0 failed — identical to the pre-PR baseline.

[#305]: https://github.com/Mega-Gorilla/livecap-cli/issues/305

#### Calibration follow-up: real-engine sweep + threshold tuning (Issue [#295] PR-B follow-up)

- **3 new hypothesis-driven candidate presets** appended to
  `benchmarks/non_speech_filter/sweep.py::default_named_presets()`:
  - `on_relaxed_rms` — drop `rms_min_db` floor from -35 to -45 to admit
    quieter real-corpus frames (real recordings sit at -41 to -46 dBFS
    overall, so the default floor was rejecting > 95 % of frames before
    the AND combination could fire).
  - `on_low_freq_aware` — widen the spectral centroid window
    (`centroid_min_hz` 2500 → 500) and tighten `voiced_max` (0.25 →
    0.15) to test whether `desk_tap`-style low-frequency thumps can be
    caught without dropping low-pitched speech.
  - `on_speech_safe` — tightest preset (`flatness_min` 0.45,
    `centroid_min_hz` 3000, `onset_ratio` 5.0) as a safety ceiling that
    confirms short-utterance recall stays at 100 % under aggressive
    filtering.
- **New `benchmarks/non_speech_filter/calibration.py` (~430 lines)**:
  reads the CSV emitted by `sweep.py` and produces a structured Markdown
  report containing (1) per-engine hallucination delta vs `baseline_off`
  (segmented by backend and corpus), (2) recall-regression flags for any
  preset/cell pair that dropped recall below the baseline, (3) a Pareto
  summary across presets with explicit dominance markers, and (4) a
  structured recommendation driven by Issue #295 PR-B follow-up plan
  rule D4 (≥30 % hallucination drop on `webrtc × parakeet_ja × real`
  with no recall regression → promote that preset; otherwise default
  off, document gap, propose Phase 2 SED).
- **Calibration findings** (full record in
  `docs/benchmarks/calibration-results-2026-06-07.md`):
  - 144 cells (8 presets × 3 backends × 3 engines × 2 corpora) ran in
    ~16.5 min on a single RTX 4090 with engine-load amortisation.
  - **`parakeet_ja × WebRTC × real desk_tap` hallucination unchanged
    at 50 % across all 8 presets** (the PR-B v4 AC target).
  - Same on `reazonspeech × WebRTC × real desk_tap` (50 % → 50 %).
  - `parakeet_ja × WebRTC × synthetic burst` hallucination drops
    75 % → 62.5 % (one item out of eight) on the 4 Pareto-dominant
    presets (`on_moderate`, `on_aggressive`, `on_relaxed_rms`,
    `on_low_freq_aware`).
  - **Zero recall regressions** in any of the 144 cells.
- **Default mode decision: `--transient-filter=off` is maintained.**
  Rule D4's headline criterion (≥30 % hallucination drop on the AC
  target cell) is unmet, so no preset earns a promotion to default.
- **`on_moderate` is documented as the best observed DSP preset for
  synthetic rapid-burst tests only** — explicitly **not** a production
  hallucination mitigation recommendation. Calibration showed zero
  improvement on the real-corpus target cell.
- **The DSP transient detector layer is positioned as `experimental`
  going forward**: not deprecated (no replacement exists yet) but not a
  production-hallucination-mitigation candidate. CLI invocations of
  `--transient-filter observe/on` now emit a one-line experimental
  notice to make the status visible at the moment of opt-in. Phase 2
  SED (sound-event detection) is the planned successor for
  `desk_tap`-style low-frequency transients.
- **New `docs/audio-filter-reference.md`**: user-facing reference for
  every audio filter in the pipeline (NoiseGate / TransientDetector /
  VAD / EnergyGate) — purpose, pipeline position, CLI surface, default
  state, measured effectiveness with citations, recommendation, known
  limitations. Single doc users can scan to decide which filter to
  enable.
- **No detector code change**. The sweep + analysis is pure data
  collection; this PR does not modify
  `livecap_cli/audio/transient_detector.py` or any production pipeline.
- **Issue #295 v6** reframes the PR-B AC line `WebRTC × desk_tap (real)
  false_trigger 50 % → 0 %` with the empirically demonstrated bound
  ("0.0 pp achievable with 6-feature AND DSP detector — Phase 2 SED is
  the correct route").
- **BASELINE_INVARIANTS bounds remain unchanged** (default unchanged
  → no tightening warranted).
- **Out of scope** (separate follow-ups): Phase 2 SED epic for low-
  frequency / non-broadband transient detection; detector architecture
  changes (AND → OR, weighted-sum, new features); `#302` lookahead
  delay (still gated on a future reject default ON decision that this
  calibration data argues against).
- Verification: full PR-relevant suite (`pytest tests/audio/
  tests/integration/non_speech_filter/ tests/integration/vad/
  tests/transcription/ tests/core/cli/ tests/audio_sources/`) → **300
  passed, 5 skipped (env-gated), 0 failed**.

#### Fixed: PR-B follow-up — async path bypass + causal best-effort spec (Issue [#295] PR-B follow-up)

- **`StreamTranscriber.transcribe_async()` が transient detector を bypass**
  していた問題を修正。`feed_audio()` と `transcribe_async()` の pre-VAD
  処理を `_apply_pre_vad_processing()` 共通 helper に集約し、両 path が
  必ず NoiseGate → Layer 1 detector → VAD の順で走るよう pin。
- **`tests/transcription/test_stream.py::TestTransientDetectorWiring`**
  を追加 (3 cases): sync/async 両 path の detector 起動、両 path の
  telemetry 完全一致を assert。再発防止層。
- **`TransientDetector.process()` docstring** に causal / no-lookahead
  仕様を明文化。`on` mode の chunked output は best-effort upper bound で
  あり、bit-exact reconstruction ではないことを記載。32 ms lookahead-
  delay 拡張は別 issue で track。
- **`tests/audio/test_transient_detector.py::TestStreamingEquivalence::
  test_on_mode_chunked_is_causal_best_effort`** を追加: telemetry が
  full / chunked で一致することと、`on` mode で flagged frame があれば
  energy が入力より低下することを assert (full vs chunked output の
  bit-exact equality は意図的に検証しない)。
- **default mode を `off` のまま維持** + ドキュメント整合: Issue #295
  / docs / CHANGELOG では旧 v3/v4 で「default observe」と記載していた
  が、PR-B 実装は安全側の `default off` を採用。calibration は
  `--transient-filter observe` で明示的に opt-in する運用を明記。
- 既存挙動への影響: なし (default `off` で detector 構築されないため)。
- Verification: `pytest tests/audio/test_transient_detector.py
  tests/transcription/test_stream.py tests/integration/non_speech_filter/`
  → 86 passed, 8 skipped (env-gated).

#### Layer 1: DSP Transient/Applause Detector (Issue [#295] PR-B)

- **新規 `livecap_cli/audio/transient_detector.py`**: 6 DSP feature
  (`spectral_flatness` / `spectral_centroid_hz` / `zero_crossing_rate` /
  `onset_strength` / `voiced_ratio` / `rms_db`) を AND 結合して
  applause-like フレームを検出する frame-based stateful detector。
  3 mode: `off` (構築されない) / `observe` (telemetry のみ、audio 不変) /
  `on` (applause-flag frame を zero-out)。
- **`StreamTranscriber` 統合**: 新引数
  `transient_detector: Optional[TransientDetector] = None`、`feed_audio`
  の NoiseGate 後 / VAD 前で起動。`reset()` / `close()` テレメトリにも
  対応 (EnergyGate と同じ pattern)。
- **CLI flags** (`transcribe` サブコマンド): `--transient-filter`
  (`off`/`observe`/`on`、default `off`) + 6 threshold flag。
- **Benchmark CLI flags** 同名で揃え、`build_pipeline()` に
  `transient_config` kwarg を追加。`None` を渡せば baseline pipeline は
  bit-identical に保たれる (PR-0 BASELINE_INVARIANTS regression なし)。
- **新規 `benchmarks/non_speech_filter/sweep.py`**: 5 named preset
  (`baseline_off` / `observe_defaults` / `on_conservative` /
  `on_moderate` / `on_aggressive`) を回す threshold sweep harness。
  CSV + Markdown 出力。
- **テスト**:
  - `tests/audio/test_transient_detector.py`: 26 unit test
    (feature 算出 / AND 決定 / streaming 等価性 / mode semantics /
    config validation)。
  - `tests/integration/non_speech_filter/test_transient_detector_integration.py`:
    7 integration test (observe = no-op on metrics, on-mode positive
    preservation, WebRTC burst no-regression)。
- **検証結果 (private real corpus + synthetic、mock engine)**:
  - observe mode は BASELINE_INVARIANTS と完全一致 (silero 0/0/0 %,
    tenvad 25/100/100 %, webrtc 75/100/100 %)。
  - on mode (moderate/aggressive) で **WebRTC × synthetic burst の
    false_trigger 75 % → 62.5 %**。
  - WebRTC × real desk_tap は default 閾値で 50 % のまま (per-clip 観測で
    `centroid_min_hz=2500` が desk_tap の低域成分を弾いていることを確認、
    calibration follow-up で対応)。
- **限界 (docs に明示)**:
  - 既定閾値は **synthetic rapid burst 想定**で、private real corpus の
    個別 clip (desk_tap、scattered 拍手) は未較正。
  - reject default ON 化は PR-B のスコープ外 — calibration sweep の
    結果が出てから別 PR で実施する設計。
- **Out of scope**:
  - PR-C で予定する Layer 2 (VADStateMachine cooldown) との signaling は
    実装しない (検出器は event を emit するが消費側は別 PR)。
  - PR-A 系の confidence filter / prompt reset は対象外。

#### Non-speech filter evaluation harness (Issue [#295] PR-0)

- **新規 `tests/integration/non_speech_filter/`**: Phase 1 多段防御
  (DSP transient detector / VADStateMachine cooldown 拡張 / Confidence filter /
  Prompt reset) の **baseline + regression 検出基盤** を導入。3 VAD backend
  (Silero / TenVAD / WebRTC) × 13 件の synthetic corpus (negative 8 + positive 5、
  うち短発話 2) で現状 pipeline (NoiseGate + VAD + EnergyGate) を計測し、
  `baselines/{backend}.json` に schema v1 で永続化。後続 PR-B/C/A はこの JSON を
  比較基準にする。
- **新規 `benchmarks/non_speech_filter/`**:
  `python -m benchmarks.non_speech_filter` で ad-hoc 評価可能な runner +
  Markdown/JSON レポート。`--engine whispers2t` 等を指定すれば
  `non_empty_hallucination_rate` (engine が非空 text を返した負例の割合) も計測。
  実音源は `LIVECAP_NON_SPEECH_CORPUS_DIR` で manifest+WAV を渡すと自動 load。
- **指標**: `false_asr_trigger_rate` / `speech_recall` /
  **`short_utterance_recall`** (最重要) / `non_empty_hallucination_rate` (opt-in) /
  `added_latency_p50_ms` / `_p95_ms`。
- **新 marker**: `evaluation_harness` (pyproject.toml に登録)。`-m evaluation_harness`
  で opt-in 実行、CI baseline tests のみ拾う。
- **既存コード touch ゼロ**: `livecap_cli/` 配下は無変更。`benchmarks/common/`
  (`DatasetManager` / `BenchmarkEngineManager`) は real-engine 経路でのみ利用。
- **動機**: Issue #295 v2 のレビュー指摘
  「**評価ハーネス先行整備** + pre-engine 優先 + DSP detector default off-by-default
  + 実装前 corpus 整備」を満たすため、Phase 1 PR-B/C/A の前提として独立着地させる。
- **限界 (docs/benchmarks/non-speech-filter.md に明記)**:
  - 合成 speech proxy は Silero VAD (実音声学習) で recall=0 になる構造的限界。
    Silero baseline を意味のある形で測るには実音源 fixture が必要。
  - WebRTC backend は binary 出力 (0.0/1.0) のため、Phase 1 PR-C で導入予定の
    hysteresis は no-op (duration-based cooldown のみ機能)。
- **検証**:
  - `uv run pytest tests/integration/non_speech_filter/ -m evaluation_harness`
    → 6 passed (3 backend × 2 tests), 6 skipped (env-var gated)。
  - `uv run python -m benchmarks.non_speech_filter --mode quick --backend silero,tenvad,webrtc`
    → JSON + Markdown 出力、Silero / TenVAD / WebRTC の baseline 差を可視化。
  - 既存 `tests/integration/vad/` + `tests/audio/` の 74 test に regression なし。

### Changed

#### **BREAKING** `StreamTranscriber` に engine-input low-energy gate (EnergyGate) を追加 (Issue [#292])

- **Before**: VAD segment は energy 不問で全て `engine.transcribe()` に渡され、低 RMS / 純ノイズ segment で hallucination ("うん"/"ピッ"/"え?"/"どうぞ" 等) が発生。
- **After**: `StreamTranscriber` の 3 callsites (`_transcribe_segment` / `_transcribe_segment_async` / `_transcribe_interim`) で共通 helper `_should_skip_low_energy(audio, kind)` を呼び、per-segment energy が threshold 未満なら `engine.transcribe()` を skip。
- **動機**: `#291` (NoiseGate 単位ミスマッチ) は NoiseGate 有効時の primary fix だが、NoiseGate は opt-in (default off) で大半のユーザーは VAD のみで防御。VAD false-positive segment が engine に渡って hallucination する経路が残っていた (Mega-Gorilla/livecap-gui#331 の root-cause の一つ)。実音源 pre-evaluation で parakeet_ja は **silent audio に対して 100% hallucination** を確認、EnergyGate を経由すると -45 dBFS threshold で 26% 削減 (stress test)。production 条件 (VAD default threshold) では VAD が一次防御として効き、本機能は副次防御として機能する。
- **API 変更点 (`StreamTranscriber.__init__`)**:
  - 新引数: `engine_min_rms_dbfs: float = -45.0` — threshold (dBFS)。`float("-inf")` で完全 opt-out。
  - 新引数: `engine_energy_metric: str = "max_frame_rms"` — 4 metric から選択 (`max_frame_rms` / `whole_rms` / `p95_frame_rms` / `top3_frame_rms`)。default は VAD padding 希釈に耐性の `max_frame_rms`。
  - 新引数: `engine_energy_frame_ms: float = 32.0` — frame-based metric の窓長 (ms)。
- **CLI 変更**:
  - `transcribe` に `--engine-min-rms` / `--engine-energy-metric` / `--engine-energy-frame-ms` の 3 flag を追加。`--engine-min-rms` には custom type を実装し `off` / `disabled` / `none` 文字列を `float("-inf")` に map (argparse の leading-`-` value 制約のため bare `-inf` は不可、`=-inf` か `off` を使う)。
  - `levels` に `--engine-min-rms-margin` flag を追加 (default `+6 dB`)。`suggested_engine_min_rms_dbfs` の margin を user 任意に調整可能。
- **`NoiseAnalysis` 変更**:
  - 新 field: `suggested_engine_min_rms_dbfs: float` (= `noise_rms_p95_db + engine_min_rms_margin_db`)。CLI `levels` で 1 回の calibration から peak-unit / RMS-unit 両方の suggested 値が得られる。
  - `analyze_noise_samples()` に optional `engine_min_rms_margin_db` キーワード引数を追加 (default `ENGINE_MIN_RMS_SAFETY_MARGIN_DB = 6.0`)。
- **新公開 API (`livecap_cli.audio`)**:
  - `ENGINE_MIN_RMS_SAFETY_MARGIN_DB = 6.0` (定数)
  - `ENERGY_METRICS = ("max_frame_rms", "whole_rms", "p95_frame_rms", "top3_frame_rms")`
  - `_segment_energy_dbfs(audio, sample_rate, metric, frame_ms) -> float` (helper、user-configurable metric/frame で per-segment energy を測定)
- **Telemetry**: `StreamTranscriber.close()` 時に drop counter の内訳 (`final_sync` / `final_async` / `interim`) を `logger.info` で 1 行サマリ。silent failure 防止。
- **Migration**:
  - 既存挙動を完全に維持したい場合: `StreamTranscriber(engine=..., engine_min_rms_dbfs=float("-inf"))` または CLI `--engine-min-rms off` で opt-out。
  - 通常はデフォルト (`-45.0`) で問題なく動作 (synthetic regression + 実音源プローブで通常会話・小声・ささやきレベル speech は pass を確認)。whisper 録音など特殊用途は閾値を下げる (`--engine-min-rms -50` 等) か opt-out。
- **検証**:
  - `tests/audio/test_analysis.py::TestSegmentEnergyDbfs` (10 cases) — 4 metric ごとの動作 / fallback / 物理的妥当性。
  - `tests/audio/test_energy_gate_regression.py` (新規 7 cases) — synthetic fixture (silent noise / speech-like burst / padded short utterance) で default threshold の drop/pass を assert。VAD padding 希釈に max_frame が耐性を持ち、whole_rms は希釈で false-drop することを documentation。
  - `tests/transcription/test_stream.py::TestEnergyGate` (10 cases) — 3 callsites (sync / async / interim) で `engine.transcribe()` が呼ばれない / 呼ばれることを mock の `call_count` で検証 + opt-out + invalid arg validation + close() log。
  - `tests/core/cli/test_cli.py::TestEnergyGateFlags` (12 cases) — `--engine-min-rms` の 4 parse パターン (numeric / off / disabled / =-inf) + invalid raises + metric choices + frame-ms parse + help text 可視性。
- **限界**:
  - EnergyGate は **silver bullet ではない**。実音源プローブで parakeet_ja は 73 silent windows で 100% hallucinate し、-45 dBFS threshold (max_frame_rms) で削減できるのは 26% のみ (transient を含む noisy silence は max frame で pass する)。完全防御には `--noise-gate` との併用 + VAD threshold 適切化が必要。
  - Engine choice が hallucination 耐性に大きく影響 (Parakeet 100% vs ReazonSpeech は known 低い)。
- **将来 follow-up**:
  - `BaseEngine` 側へのガード追加 (`StreamTranscriber` を経由しない advanced user 向け)
  - top-k metric の k=3 から user-configurable に拡張

#### **BREAKING** `NoiseAnalysis` / `analyze_noise_samples()` を peak-based calibration に置換 (Issue [#291])

- **Before**: `analyze_noise_samples(samples_db, sample_rate_hz)` →
  `suggested_threshold_db = noise_peak (chunk RMS p95) + 10 dB`
- **After**: `analyze_noise_samples(samples_db, peak_samples_db, sample_rate_hz)` →
  `suggested_threshold_db = peak_p95 (per-chunk |x|.max() p95) + 6 dB`
- **動機**: `NoiseGate` (`livecap_cli/audio/noise_gate.py`) の envelope follower
  は per-sample peak を追跡するが、calibration は chunk RMS を計測していた → unit
  mismatch により impulsive noise (キーボード/呼吸/breath bursts) で threshold が
  peak の下に潜り、無音時 hallucination ("あ"/"うん"/"ピッ") を引き起こしていた
  (Mega-Gorilla/livecap-gui#331 root-cause)。White noise の crest factor ≈ 11 dB
  が偶然 `+10` で吸収されていたが、impulsive noise では crest factor がより大きく
  破綻する。
- **API 変更点**:
  - `NoiseAnalysis` 新 field: `peak_p95_db: float` (per-chunk `|x|.max()` の 95%ile)
  - `NoiseAnalysis` 改名: `noise_peak_db` → `noise_rms_p95_db` (unit を field 名に明示)
  - `NoiseAnalysis` 削除: `safe_zone_min_db` (新 `suggested_threshold_db` と 1 dB 差で意味崩壊)
  - `analyze_noise_samples()` 新 required 引数: `peak_samples_db` (Optional=None の
    legacy default は無し: pre-1.0 backward-compat policy で旧バグ温存の flag は不可)
  - 新 module-level 定数: `livecap_cli.audio.PEAK_SAFETY_MARGIN_DB = 6.0`
  - `danger_zone` は据え置き (RMS-unit diagnostic として docstring で明記)
- **Migration**:
  ```python
  # 旧
  rms_db_list = [20*log10(rms(chunk)) for chunk in chunks]
  a = analyze_noise_samples(rms_db_list)

  # 新
  rms_db_list  = [20*log10(rms(chunk))    for chunk in chunks]
  peak_db_list = [20*log10(|chunk|.max()) for chunk in chunks]
  a = analyze_noise_samples(rms_db_list, peak_db_list)
  # a.noise_peak_db    -> a.noise_rms_p95_db
  # a.safe_zone_min_db -> (削除; suggested_threshold_db を直接使用)
  # a.peak_p95_db      -> 新 field; threshold の基準
  ```
  CLI `levels` は内部で per-chunk peak を収集するように移行済みのため、
  外部から CLI を呼ぶ場合の変更は不要 (JSON schema のみ変更)。
- **検証**: `tests/audio/test_noise_gate_calibration.py` (新規) で synthetic
  impulsive noise を旧 / 新 threshold それぞれで NoiseGate に通し、旧で gate
  が開く / 新で閉じ続けることを assert する end-to-end 回帰テストを追加。
- **GUI ペア PR**: livecap-gui 側は `core/noise_statistics.py` を削除し本 API
  に委譲する PR を別 issue (Mega-Gorilla/livecap-gui#335 の対) で受ける。
- **将来 follow-up**: NoiseGate の envelope follower を calibration 入力に対して
  simulate し envelope の 95%ile を取れば margin を 1-2 dB に縮められる ([#283]
  と組で別 issue 化予定)。

#### **BREAKING** `NoiseGate` デフォルト `release_ms` 変更 (Issue [#283] PR C)

- **Before** (PR #279 / PR #281 / PR #282): `release_ms=30`
- **After**: `release_ms=100`
- **動機**: PR #282 で導入された hard-mute と短い release の組み合わせが、aggressive な閾値 (-25/-17 dB) で whisper 系エンジンの fragmentation ハルシネーション (「んんん...」loop) を引き起こす。A/B 実測で `release_ms=100` により完全解消を確認 (316→102, 299→96 chars)
- **Migration**: 旧挙動を明示的に再現するには `release_ms=30` を直接渡す:
  ```python
  NoiseGate(release_ms=30)  # pre-PR-C default
  ```
  CLI の場合:
  ```bash
  livecap-cli transcribe --noise-gate --noise-gate-release 30 ...
  ```
- **検証結果**: `docs/benchmarks/noise-gate-ab.md` に更新後のテーブル掲載

#### **BREAKING** `NoiseGate` 既定挙動変更 (Issue [#280] PR B)

- **Before** (PR [#279] / PR [#281]): 単一閾値 + `-60 dB` soft-mute
- **After** (PR B): 自動ヒステリシス (`threshold_db - 6 dB`) + hard-mute (出力ゼロ)
- **動機**: PR #281 の A/B 検証で、PR #281 までの挙動が whisper 系エンジンで flicker によるハルシネーション暴走 ("どうもどうも..." 等) を引き起こすことが実証された
- **Migration**: 過去挙動を明示的に再現するには以下を指定:
  ```python
  NoiseGate(
      threshold_db=-35,
      close_threshold_db=-35,  # single-threshold (ヒステリシス無効)
      noise_floor_db=-60,      # soft-mute
  )
  ```
  CLI の場合:
  ```bash
  livecap-cli transcribe --noise-gate \
      --noise-gate-threshold -35 \
      --noise-gate-close-threshold -35 \
      --noise-gate-floor -60 \
      ...
  ```

新規オプション (既存呼び出しは無変更で動作、挙動のみ変化):

- `NoiseGate` / `transcribe` CLI に `close_threshold_db` / `--noise-gate-close-threshold` を追加 (ヒステリシス制御)
- `NoiseGate` / `transcribe` CLI に `noise_floor_db` / `--noise-gate-floor` を追加 (ゲート閉鎖時の減衰量制御)
- 初期化ログが resolved 値 (open/close/noise_floor) を出力するように改善 (ポリシー準拠)

既知の follow-up:

- `release_ms=30` は PR B の新しい gate 挙動 (hard-mute による clean silence) に対して短すぎるため、攻撃的な閾値で fragmentation hallucination が発生することがあります。`--noise-gate-release 100` または `200` で回避可能。デフォルト値の変更は別 issue で対応予定。

### Added

#### Noise Gate & Calibration ([#278], [#279], [#280], [#281])

- `livecap_cli.audio.NoiseGate` — 音量ベースのリアルタイムノイズゲート（サンプル単位エンベロープフォロワー、numba JIT で < 0.1 ms / 100 ms chunk）。VAD 前段に挿入してハルシネーションを抑制。
- `transcribe` サブコマンドに `--noise-gate` / `--noise-gate-threshold` / `--noise-gate-attack` / `--noise-gate-release` オプションを追加。
- `livecap-cli levels` サブコマンド — マイク入力レベルを dB 単位でリアルタイム表示し、環境ノイズから推奨閾値を算出。
  - `--duration N` — N 秒後に自動停止（非対話モード）。
  - `--json` — `NoiseAnalysis` を JSON で stdout に出力（GUI / スクリプト連携向け）。
- `livecap_cli.audio.analysis` モジュール — `NoiseAnalysis` dataclass と `analyze_noise_samples()` 関数（CLI / GUI 共通キャリブレーション API）。
- 推奨閾値アルゴリズム: `noise_peak (95%ile) + 10 dB`（[livecap-gui PR #294](https://github.com/Mega-Gorilla/livecap-gui/pull/294) の実測に基づく保守的マージン）。「死のゾーン」(`noise_floor ± 5 dB`) を回避する設計。

**段階導入について**: PR #281 は **キャリブレーション API 基盤の先行導入** (Issue #280 C-3 + C-4) です。NoiseGate 本体の安定化 ([Issue #280](https://github.com/Mega-Gorilla/livecap-cli/issues/280) の C-1 ヒステリシス + C-2 hard-mute) は follow-up PR で提供予定。現行実装 (単一閾値 + `-60 dB` soft-mute) では、閾値が speech peak 付近の場合に flicker で逆にハルシネーションを誘発することがあります。特に `whispers2t` エンジンで影響が大きく、`reazonspeech` / `parakeet_ja` / `qwen3asr` は影響を受けにくいことが A/B テストで確認されています (PR #281 comments 参照)。暫定対応として、低 SNR 環境では `levels` の推奨値より保守的な値の使用、または別エンジンの利用を推奨します。

#### Phase 6: CLI Subcommand Structure ([#74], [#201])

New CLI with subcommand architecture:

| Command | Description |
|---------|-------------|
| `livecap-cli info` | Display installation diagnostics |
| `livecap-cli devices` | List audio input devices |
| `livecap-cli engines` | List available ASR engines |
| `livecap-cli translators` | List available translators |
| `livecap-cli transcribe` | Transcribe audio (file or realtime) |

**transcribe options:**
- `<file> -o <output.srt>` - File transcription to SRT
- `--realtime --mic <id>` - Realtime microphone transcription
- `--translate <id> --target-lang <lang>` - Translation support
- `--vad <auto|silero|tenvad|webrtc>` - VAD backend selection
- `--engine <id>` - ASR engine selection
- `--device <auto|gpu|cpu>` - Device selection

**Package extras:**
- `recommended`: Google translation (deep-translator)
- `all`: All optional dependencies

#### Phase 5: Engine Optimizations ([#73], [#194], [#196], [#197])

- Template Method pattern for `BaseEngine` with standardized lifecycle
- Progress reporting during model loading (0-100%)
- Model memory caching for faster subsequent loads
- Library preloading for reduced import time
- Standardized cleanup and resource management

#### Phase 4: Translation Support ([#72], [#180], [#181], [#182], [#184], [#186])

**Translators:**
- `google` - Google Translate ([#180])
- `opus_mt` - Helsinki-NLP Opus-MT local models ([#181])
- `riva_instruct` - NVIDIA Riva Translate 4B Instruct ([#182])

**Features:**
- Context-aware translation with sentence buffering
- `StreamTranscriber` translation integration ([#184])
- `FileTranscriptionPipeline` translation integration ([#186])
- Configurable timeout via `LIVECAP_TRANSLATION_TIMEOUT`
- Async translation deadlock prevention ([#189])

#### Phase 3: Package Structure ([#71])

- Reorganized module structure under `livecap_cli/`
- Clear separation: `engines/`, `vad/`, `transcription/`, `translation/`
- Unified public API exports in `__init__.py`

#### Phase 2: API Unification ([#70])

- `TranscriptionResult` dataclass replacing `TranscriptionEventDict`
- `VADConfig` dataclass for VAD parameters
- `EngineFactory.create_engine(engine_type, device, **options)` API
- Consistent error handling with `TranscriptionError`, `EngineError`

#### Phase 1: Realtime Transcription ([#69], [#65], [#66], [#67], [#68])

**Core components:**
- `StreamTranscriber` - VAD + ASR streaming orchestration ([#65])
- `VADProcessor` - Pluggable VAD with state machine ([#66])
- `TranscriptionResult` / `InterimResult` - Unified result types ([#67])
- `AudioSource` / `FileSource` / `MicrophoneSource` - Audio abstraction ([#68])

**VAD backends:**
- Silero VAD (default, neural network-based)
- WebRTC VAD (fast, low memory)
- TenVAD (optimized for Japanese)

**Language optimization:**
- `VADProcessor.from_language("ja")` - Auto-select optimal VAD
- Benchmark-based presets for Japanese and English

#### File Transcription

- `FileTranscriptionPipeline` for batch processing
- SRT subtitle output format
- FFmpeg integration for audio extraction
- Translation integration for bilingual subtitles

### Changed

#### Breaking Changes

| Before | After |
|--------|-------|
| Package: `livecap-core` | Package: `livecap-cli` |
| Module: `livecap_core` | Module: `livecap_cli` |
| CLI: `livecap-core --info` | CLI: `livecap-cli info` |
| CLI: `livecap-core --as-json` | CLI: `livecap-cli info --as-json` |

#### API Changes

- `TranscriptionEventDict` → `TranscriptionResult` dataclass
- Engine creation unified to `EngineFactory.create_engine()`
- VAD configuration via `VADConfig` instead of dict
- `detect_device()` returns `str` instead of `Tuple` ([#175])

### Deprecated

- `TranscriptionEventDict` (use `TranscriptionResult`)
- `languages.py` module (use `langcodes` for BCP-47) ([#173])

### Removed

- Old flag-based CLI interface (`--info`, `--ensure-ffmpeg`, `--as-json`)
- `livecap-core` entry point
- `livecap_core` module name
- `Languages.get_engines_for_language()` (use `EngineMetadata`) ([#171])

### Fixed

- GitHub Actions workflows updated for module rename ([#201])
- Integration test path filters updated
- Async translation deadlock in concurrent scenarios ([#189])
- Translation timeout handling improvements ([#187])
- OPUS-MT context disabled by default for stability ([#191])

### Security

- No security issues in this release

---

## Migration Guide

### From `livecap-core` to `livecap-cli`

#### 1. Update package installation

```bash
# Before
pip install livecap-core[engines-torch]

# After
pip install livecap-cli[engines-torch]

# Or use the recommended bundle:
pip install livecap-cli[recommended]
```

#### 2. Update imports

```python
# Before
from livecap_core import StreamTranscriber, EngineFactory
from livecap_core.vad import VADProcessor, VADConfig

# After
from livecap_cli import StreamTranscriber, EngineFactory
from livecap_cli.vad import VADProcessor, VADConfig
```

#### 3. Update CLI commands

```bash
# Before
livecap-core --info
livecap-core --as-json

# After
livecap-cli info
livecap-cli info --as-json

# New commands
livecap-cli devices
livecap-cli engines
livecap-cli translators
livecap-cli transcribe input.mp4 -o output.srt
livecap-cli transcribe --realtime --mic 0
```

#### 4. Update result handling (if using old dict API)

```python
# Before (TranscriptionEventDict)
result = {"text": "...", "start": 0.0, "end": 1.0}
print(result["text"])

# After (TranscriptionResult dataclass)
# result is now a dataclass with attributes
print(result.text)
print(result.start_time)
print(result.end_time)
print(result.to_srt_entry(index=1))
```

---

## Issue References

- Epic: [#64] - livecap-cli リファクタリング
- Phase 1: [#69] - リアルタイム文字起こし実装
- Phase 2: [#70] - API 統一と Config 簡素化
- Phase 3: [#71] - パッケージ構造整理
- Phase 4: [#72] - 翻訳機能実装
- Phase 5: [#73] - エンジン最適化
- Phase 6: [#74] - 依存関係整理・CLI・パッケージ名変更
- Docs: [#75] - ドキュメント更新

---

[Unreleased]: https://github.com/Mega-Gorilla/livecap-cli/compare/main...HEAD

[#64]: https://github.com/Mega-Gorilla/livecap-cli/issues/64
[#65]: https://github.com/Mega-Gorilla/livecap-cli/issues/65
[#66]: https://github.com/Mega-Gorilla/livecap-cli/issues/66
[#67]: https://github.com/Mega-Gorilla/livecap-cli/issues/67
[#68]: https://github.com/Mega-Gorilla/livecap-cli/issues/68
[#69]: https://github.com/Mega-Gorilla/livecap-cli/issues/69
[#70]: https://github.com/Mega-Gorilla/livecap-cli/issues/70
[#71]: https://github.com/Mega-Gorilla/livecap-cli/issues/71
[#72]: https://github.com/Mega-Gorilla/livecap-cli/issues/72
[#73]: https://github.com/Mega-Gorilla/livecap-cli/issues/73
[#74]: https://github.com/Mega-Gorilla/livecap-cli/issues/74
[#75]: https://github.com/Mega-Gorilla/livecap-cli/issues/75
[#171]: https://github.com/Mega-Gorilla/livecap-cli/pull/171
[#173]: https://github.com/Mega-Gorilla/livecap-cli/pull/173
[#175]: https://github.com/Mega-Gorilla/livecap-cli/pull/175
[#180]: https://github.com/Mega-Gorilla/livecap-cli/pull/180
[#181]: https://github.com/Mega-Gorilla/livecap-cli/pull/181
[#182]: https://github.com/Mega-Gorilla/livecap-cli/pull/182
[#184]: https://github.com/Mega-Gorilla/livecap-cli/pull/184
[#186]: https://github.com/Mega-Gorilla/livecap-cli/pull/186
[#187]: https://github.com/Mega-Gorilla/livecap-cli/pull/187
[#189]: https://github.com/Mega-Gorilla/livecap-cli/pull/189
[#191]: https://github.com/Mega-Gorilla/livecap-cli/pull/191
[#194]: https://github.com/Mega-Gorilla/livecap-cli/pull/194
[#196]: https://github.com/Mega-Gorilla/livecap-cli/pull/196
[#197]: https://github.com/Mega-Gorilla/livecap-cli/pull/197
[#201]: https://github.com/Mega-Gorilla/livecap-cli/pull/201
[#278]: https://github.com/Mega-Gorilla/livecap-cli/issues/278
[#279]: https://github.com/Mega-Gorilla/livecap-cli/pull/279
[#280]: https://github.com/Mega-Gorilla/livecap-cli/issues/280
[#281]: https://github.com/Mega-Gorilla/livecap-cli/pull/281
[#283]: https://github.com/Mega-Gorilla/livecap-cli/issues/283
[#291]: https://github.com/Mega-Gorilla/livecap-cli/issues/291
[#292]: https://github.com/Mega-Gorilla/livecap-cli/issues/292
[#295]: https://github.com/Mega-Gorilla/livecap-cli/issues/295
