# Phase 2 SED off-line evaluation (Issue #305 PR-D0)

This directory holds the **research-only** off-line evaluation pipeline for the
Phase 2 Sound-Event Detection (SED) epic. The pipeline measures whether the
candidate model (default: **EfficientAT**, fallback: YAMNet) can ship in
livecap-cli under the four PR-D0 acceptance dimensions:

1. **Accuracy** — can the model identify `desk_tap` / `knock` / `applause`?
2. **Safety** — does it preserve speech recall ≥ 95 % and short-utterance recall = 100 %?
3. **Runtime** — checkpoint ≤ 50 MB, dep delta ≤ 150 MB, runtime peak ≤ 200 MB, CPU p95 ≤ 100 ms, GPU p95 ≤ 30 ms
4. **License** — code + checkpoint + AudioSet-derived rights all AGPL-3.0-only compatible

**Scope guardrail:** This package does *not* import from `livecap_cli.audio.*`
or touch any production code. Integration is PR-D1; default-decision is PR-D2.

---

## Quick start

### 1. Clone EfficientAT (one-time setup, not committed)

```powershell
git clone https://github.com/fschmid56/EfficientAT.git .tmp/EfficientAT
# Record the commit hash for the decision document
git -C .tmp/EfficientAT rev-parse HEAD
```

### 2. Provide the private non-speech corpus (existing PR-B path)

```powershell
$env:LIVECAP_NON_SPEECH_CORPUS_DIR = (Resolve-Path .tmp/non_speech_corpus)
```

The corpus must contain `manifest.json` plus 6 WAV files
(`applause_5_claps.wav`, `desk_tap.wav`, `short_utterances_mixed.wav`,
`normal_speech_neko.wav`, `applause_then_speech.wav`,
`overlapping_applause_speech.wav`). Same layout as PR-B; corpus contents are
gitignored.

### 3. Run the full evaluation

```powershell
$env:LIVECAP_SED_EFFICIENTAT_PATH = (Resolve-Path .tmp/EfficientAT)
uv run python -m benchmarks.sed --model mn04_as `
    --output-dir benchmark_results/sed/2026-06-10/
```

Outputs (committed per Issue #305 v3 artifact policy):

| File | Content |
|---|---|
| `probabilities.csv` | Per-clip × per-window × 527-class probabilities |
| `latency.csv` | 5-axis runtime measurement (checkpoint size / dep delta / peak memory / CPU p50/p95 / GPU p50/p95 / cold start) |
| `metadata.json` | Model variant, EfficientAT commit hash, hardware, env summary |

---

## EfficientAT model variants

| Variant | Disk size | AudioSet mAP | Notes |
|---|---|---|---|
| `mn04_as` | 3.88 MB | 43.2 | Smallest viable; default for PR-D0 |
| `dymn04_as` | 7.72 MB | 45.0 | Balanced; secondary probe |
| `dymn10_as` | 40.54 MB | 47.7 | Highest accuracy under 50 MB cap |

Each variant can be measured independently with `--model <name>` and a
distinct `--output-dir`.

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `LIVECAP_NON_SPEECH_CORPUS_DIR` | (unset) | Path to the private non-speech corpus directory (manifest.json + WAVs) |
| `LIVECAP_SED_EFFICIENTAT_PATH` | `.tmp/EfficientAT` | Path to the cloned EfficientAT repository |

---

## Module layout

```
benchmarks/sed/
├── __init__.py        # Public exports
├── __main__.py        # python -m benchmarks.sed entry
├── cli.py             # argparse + orchestration
├── class_mapping.py   # AudioSet 527 names + TARGET / SPEECH_LIKE + 3 threshold policies
├── inference.py       # EfficientAT loader, 16k→32k resample, 1s window iterator
├── metrics.py         # Class-level + reject-signal-level precision/recall
└── latency.py         # 5-axis runtime measurement
```

The accompanying tests live under
[`tests/integration/sed/`](../../tests/integration/sed/) and are marked
`@pytest.mark.sed_evaluation`. The inference smoke test is skipped unless
`LIVECAP_SED_EFFICIENTAT_PATH` resolves to a valid clone.

---

## Decision document

The final 4-dimension Go/no-go verdict is recorded in
[`docs/research/phase2-sed-evaluation-2026-06-10.md`](../../docs/research/phase2-sed-evaluation-2026-06-10.md).

The decision document is the authoritative artefact of PR-D0; the CSV files
in `benchmark_results/sed/` are the underlying evidence.

---

## Related

- Phase 2 SED epic: [Issue #305](https://github.com/Mega-Gorilla/livecap-cli/issues/305) (v3)
- Phase 1 PR-B calibration (the data that motivated this epic): [#304](https://github.com/Mega-Gorilla/livecap-cli/pull/304)
- DSP limit empirical record: [`docs/benchmarks/calibration-results-2026-06-07.md`](../../docs/benchmarks/calibration-results-2026-06-07.md)
- User-facing filter reference (updated to SED in PR-D2): [`docs/audio-filter-reference.md`](../../docs/audio-filter-reference.md)
