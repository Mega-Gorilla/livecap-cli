# Confidence threshold calibration harness

新規 ASR engine の `confidence_filter` threshold を audio corpus から自動最適化する CLI tooling。

**Issue #338** ([https://github.com/Mega-Gorilla/livecap-cli/issues/338](https://github.com/Mega-Gorilla/livecap-cli/issues/338)) で導入。**Issue #334** の PR-2 / PR-3 / PR-4 を加速 (observe mode 1-2 月運用 → ~1-2 週で完了) する目的。

## 概要

| Stage | CLI | Input | 目的 |
|---|---|---|---|
| **Stage 1** (PR-α) | `parse_observe.py` | `LIVECAP_CONFIDENCE_FILTER=observe` で蓄積した JSON log + user 提供 label | 既存 observe 運用の data を即時 sweep |
| **Stage 2** (PR-β) | `sweep.py` | `LIVECAP_CALIBRATION_CORPUS_DIR/manifest.jsonl` (audio + label) | user 提供 corpus で active calibration |
| Stage 2 helper (PR-β) | `build_corpus.py` | YouTube URL or local audio + 元原稿 | corpus を自動 chunking + label (yt-dlp + Silero VAD + 原稿 fuzzy match) |

## Stage 1 Quickstart (PR-α)

### 1. observe mode で運用、log を蓄積

```bash
LIVECAP_CONFIDENCE_FILTER=observe \
  uv run livecap-cli transcribe \
    --engine reazonspeech \
    --mic 0 \
    2>&1 | tee observe.log
```

log file の各行は以下の format ([`livecap_cli/transcription/confidence_filter.py:_decision_to_dict()`](../../livecap_cli/transcription/confidence_filter.py) 参照):

```
... INFO ... confidence_filter[observe]: {"source_id": "mic_001_chunk_00042", "engine": "reazonspeech", "text": "...", "decision": "pass", "reason": null, "engine_confidence": {"no_speech_prob": null, "avg_logprob": -0.15, ...}}
```

### 2. user 側で label を作成 (`labels.jsonl`)

実 observe log の `source_id` は **`StreamTranscriber.source_id`** (default `"default"`、mic / file 単位で複数 segment が同じ値を共有する) なので、multi-utterance log では 3 つの match strategy が用意されています:

| Match strategy | labels.jsonl key | 用途 |
|---|---|---|
| **A. composite (推奨)** | `source_id` + `occurrence_index` | multi-utterance log、各 segment を sequence index で識別 |
| **B. text match** | `source_id` + `text` (exact、case-sensitive) | log の text と一致させたい時 |
| **C. source-only (legacy)** | `source_id` のみ | 1 source = 1 sample の単純 case (重複時は警告 + last-wins) |

**重要 (silent label corruption 回避、PR #339 2nd round fix)**: 同じ `source_id` で **一度でも `occurrence_index` / `text` を使った label がある場合**、その source の source-only fallback (C) は **無効化** される。これにより `occurrence_index` を一部だけ label 済 (e.g. occurrence 0/1 のみ label、2 は未 label) な場合、未ラベル occurrence は **誤って source-only label に当てられず、unmatched skip** される。calibration data に silent corruption を入れないための safety net。

```jsonl
# A. composite key (推奨、multi-utterance log を sample 単位 join 可能)
{"source_id": "default", "occurrence_index": 0, "label": "speech"}
{"source_id": "default", "occurrence_index": 1, "label": "non_speech", "subtype": "applause"}
{"source_id": "default", "occurrence_index": 2, "label": "noisy_speech"}

# B. text match
{"source_id": "default", "text": "hello world", "label": "speech"}

# C. source-only (legacy、1 source = 1 sample の単純 case のみ)
{"source_id": "mic_001_chunk_00042", "label": "speech"}
```

Parser は **A → B → C** の順に match を試みる。label は `"speech"` / `"non_speech"` / `"noisy_speech"` のいずれか (`noisy_speech` は `speech` と同等扱い、reject されたら false reject)。

### 3. sweep 実行

```bash
uv run python -m benchmarks.confidence_calibration.parse_observe \
    --log observe.log \
    --labels labels.jsonl \
    --engine reazonspeech \
    --signal avg_logprob \
    --output report.json
```

threshold range は signal 種別から default 推定 (avg_logprob: -1.0 〜 -0.05、no_speech_prob: 0.1 〜 0.95、token_confidence_mean: 0.001 〜 0.5)、`--threshold-min` / `--threshold-max` / `--step` で override 可能。

**`--engine` 値について**: CLI には [`livecap_cli/engines/metadata.py:_ENGINES`](../../livecap_cli/engines/metadata.py) の **`id` field** (例: `reazonspeech` / `qwen3asr` / `whispers2t`) を渡す。observe log の `engine` field は実際には `engine.get_engine_name()` の **display string** (`"ReazonSpeech K2 (CPU, Int8)"` 等) が入るが、parser 側で `normalize_engine_id()` (= `_engine_id_from_name()` 相当の lower + first whitespace word) + ID alias で吸収して match させる。

Engine ID 対応表:

| metadata.py `id` (CLI ID) | display string 例 | normalize 経由 |
|---|---|---|
| `reazonspeech` | `"ReazonSpeech K2 (CPU, Int8)"` | `"reazonspeech"` (identity) |
| `whispers2t` | `"WhisperS2T base"` | `"whispers2t"` |
| `voxtral` | `"voxtral"` | `"voxtral"` (identity) |
| `canary` | `"Canary 1B Flash"` | `"canary"` |
| **`qwen3asr`** | `"Qwen3-ASR 0.6B"` / `"Qwen3-ASR 1.7B"` | `"qwen3asr"` (alias 経由、`"qwen3-asr"` から bridge) |

`--engine qwen3-asr` (hyphen 付き) も alias で受け入れるが、CLI 主表記は **`qwen3asr`** (metadata.py id と整合)。

### 4. report.json 解読

```json
{
  "engine": "reazonspeech",
  "signal_field": "avg_logprob",
  "direction": "reject_if_less",
  "sample_count": {"speech": 30, "non_speech": 20, "noisy_speech": 15},
  "excluded_count": 0,
  "recommended_threshold": -0.25,
  "recommended_metrics": {
    "threshold": -0.25,
    "tp": 19, "fp": 2, "tn": 43, "fn": 1,
    "precision": 0.905,
    "recall": 0.95,
    "f1": 0.927,
    "youden_j": 0.906,
    "false_reject_rate": 0.044
  },
  "criterion": "f1",
  "sweep": [...],
  "metadata": {"quantization": "float32", "language": "ja"}
}
```

`recommended_threshold` が data driven な推奨値。`sweep` array で全 threshold の trade-off を確認可能。

## Stage 2 (PR-β、未実装)

`sweep.py` + `build_corpus.py` は PR-β で landed 予定。詳細は Issue #338 を参照。

## Signal direction

| Signal | direction | reject 条件 |
|---|---|---|
| `avg_logprob` | `reject_if_less` | 値 < threshold で reject (低 confidence) |
| `token_confidence_mean` | `reject_if_less` | 同上 |
| `no_speech_prob` | `reject_if_greater` | 値 > threshold で reject (非音声確信度高) |

## Confusion matrix の意味

`non_speech` を positive class とする:

| | filter reject | filter pass |
|---|---|---|
| **non_speech** (positive) | TP (正しい reject) | FN (false pass、軽微) |
| **speech / noisy_speech** (negative) | **FP (false reject、user 痛い)** | TN (正しい pass) |

- **`precision`**: reject の正確性 (= TP / (TP+FP))、高いほど false reject 少
- **`recall`**: non_speech 検出率 (= TP / (TP+FN))、高いほど false pass 少
- **`f1`**: precision/recall の調和平均
- **`youden_j`**: sensitivity + specificity - 1、ROC 上の最適点
- **`false_reject_rate`**: speech を reject する割合 (user 体感、低いほど良い)

## Recommended threshold の選び方 (`--criterion`)

| Criterion | 適用 case |
|---|---|
| `f1` (default) | バランス、precision/recall ともに重要 |
| `youden_j` | ROC 最適点、binary classification の慣用 |
| `precision` | false reject (FP) を最も避けたい (user 痛い) |
| `recall` | false pass (FN) を最も避けたい (非音声混入避けたい) |

### Tie-break (同点時の選択)

複数 threshold が同 criterion 値で同点の場合、**direction-aware** に **より conservative (= 少数しか reject されない)** な threshold を選ぶ:

| Direction | Tie-break | 理由 |
|---|---|---|
| `reject_if_less` (avg_logprob 等) | threshold **小** を選ぶ | 小 threshold → 少数しか `value < threshold` にならない → reject 少 → false reject 抑制 |
| `reject_if_greater` (no_speech_prob) | threshold **大** を選ぶ | 大 threshold → 少数しか `value > threshold` にならない → reject 少 → false reject 抑制 |

これは本 harness の主目的 (Issue #334 noisy_speech false reject 抑制) と整合する選択。完全分離 case で F1=1.0 が複数 threshold で達成される時、より conservative (= 緩い、reject 少) な値が selected される。

## Corpus / labels の準備方針

**raw audio は repo に commit しない** (Issue #338 設計判断、私訳著作権存続音源含む)。

- 各 user / contributor が手元で audio を取得 (URL list は [`docs/research/calibration-corpus-sources.md`](../../docs/research/calibration-corpus-sources.md) で PR-β 完了後 documenting 予定)
- `LIVECAP_CALIBRATION_CORPUS_DIR` env var で corpus directory を指定 (既存 `LIVECAP_NON_SPEECH_CORPUS_DIR` pattern 踏襲)
- label 付与は **user 手動 + Whisper 補助** (PR-β `build_corpus.py` 提供予定)、Phase 1 では observe mode log を base に手動 label 付与で十分

## 関連リソース

- 親 issue: [Issue #338](https://github.com/Mega-Gorilla/livecap-cli/issues/338)
- 加速対象: [Issue #334](https://github.com/Mega-Gorilla/livecap-cli/issues/334) PR-2 / PR-3 / PR-4
- 既存 sweep precedent: [`benchmarks/non_speech_filter/sweep.py`](../non_speech_filter/sweep.py) (argparse + grid sweep canonical pattern)
- observe mode 仕様: [`livecap_cli/transcription/confidence_filter.py`](../../livecap_cli/transcription/confidence_filter.py) (`_decision_to_dict()`、`apply_filter()`)
- adding-an-engine guide: [`docs/contributor/adding-an-engine.md`](../../docs/contributor/adding-an-engine.md) §5 (threshold calibration template)
