# SED PR-D0 analysis — mn04_as

Generated from `probabilities_full.npz` (commit `a425fdce92572e602a1d5634799bd9f1f2efa806`).

## 1. Provisional gate (Issue #305 v3)

- Target clip: `desk_tap`
- Precision floor: 0.70
- Recall floor: 0.50
- **Result: PASS**
- Passing policies: max, target_minus_speech
- Chosen (policy, threshold): (target_minus_speech, 0.15000000596046448)
- Chosen (precision, recall): (1.0, 1.0)

> Provisional gate satisfied. Corpus is 6 clips (statistically weak) — PR-D1 must record a corpus-expansion judgement (ESC-50 / FSD50K subset vs status quo).

## 2. Reject-signal-level P/R sweep

| Threshold | P (max)/R (max) | P (sum)/R (sum) | P (target_minus_speech)/R (target_minus_speech) |
|---|---|---|---|
| 0.01 | 0.33/1.00 | 0.33/1.00 | 1.00/1.00 |
| 0.02 | 0.33/1.00 | 0.33/1.00 | 1.00/1.00 |
| 0.03 | 0.67/1.00 | 0.33/1.00 | 1.00/1.00 |
| 0.05 | 0.67/1.00 | 0.33/1.00 | 1.00/1.00 |
| 0.08 | 0.67/1.00 | 0.67/1.00 | 1.00/1.00 |
| 0.10 | 0.67/1.00 | 0.67/1.00 | 1.00/1.00 |
| 0.15 | 0.67/1.00 | 0.67/1.00 | 1.00/1.00 |
| 0.20 | 1.00/1.00 | 0.67/1.00 | 1.00/0.50 |
| 0.30 | 1.00/0.50 | 0.50/0.50 | 1.00/0.50 |
| 0.50 | 1.00/0.50 | 1.00/0.50 | 1.00/0.50 |

## 3. Clip-level max reject scores

| Clip | max | sum | target_minus_speech |
|---|---|---|---|
| applause_5_claps | 0.7383 | 1.2978 | 0.7174 |
| desk_tap | 0.2068 | 0.2956 | 0.1750 |
| short_utterances_mixed | 0.0254 | 0.0573 | -0.7588 |
| normal_speech_neko | 0.0259 | 0.0601 | -0.7872 |
| applause_then_speech | 0.0268 | 0.0629 | -0.7248 |
| overlapping_applause_speech | 0.1603 | 0.3539 | -0.6590 |

## 4. Class-level metrics @ threshold 0.05

### Target classes

| Class | Index | TP | FP | FN | TN | Precision | Recall | (threshold 0.05) |
|---|---|---|---|---|---|---|---|---|
| Hands | 61 | 1 | 0 | 1 | 4 | 1.00 | 0.50 | |
| Finger snapping | 62 | 1 | 1 | 1 | 3 | 0.50 | 0.50 | |
| Clapping | 63 | 1 | 1 | 1 | 3 | 0.50 | 0.50 | |
| Applause | 67 | 0 | 0 | 2 | 4 | 0.00 | 0.00 | |
| Door | 354 | 0 | 0 | 2 | 4 | 0.00 | 0.00 | |
| Sliding door | 357 | 0 | 0 | 2 | 4 | 0.00 | 0.00 | |
| Slam | 358 | 0 | 0 | 2 | 4 | 0.00 | 0.00 | |
| Knock | 359 | 1 | 0 | 1 | 4 | 1.00 | 0.50 | |
| Tap | 360 | 1 | 1 | 1 | 3 | 0.50 | 0.50 | |
| Thump, thud | 460 | 0 | 0 | 2 | 4 | 0.00 | 0.00 | |

### Speech-like classes (should ideally have low TP)

| Class | Index | TP | FP | FN | TN | Precision | Recall | (threshold 0.05) |
|---|---|---|---|---|---|---|---|---|
| Speech | 0 | 0 | 4 | 2 | 0 | 0.00 | 0.00 | |
| Male speech, man speaking | 1 | 0 | 4 | 2 | 0 | 0.00 | 0.00 | |
| Female speech, woman speaking | 2 | 0 | 0 | 2 | 4 | 0.00 | 0.00 | |
| Child speech, kid speaking | 3 | 0 | 0 | 2 | 4 | 0.00 | 0.00 | |
| Conversation | 4 | 0 | 0 | 2 | 4 | 0.00 | 0.00 | |
| Narration, monologue | 5 | 0 | 4 | 2 | 0 | 0.00 | 0.00 | |
| Singing | 27 | 0 | 0 | 2 | 4 | 0.00 | 0.00 | |
