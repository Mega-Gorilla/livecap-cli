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

```jsonl
{"source_id": "mic_001_chunk_00042", "label": "speech"}
{"source_id": "mic_001_chunk_00043", "label": "non_speech", "subtype": "applause"}
{"source_id": "mic_001_chunk_00044", "label": "noisy_speech"}
```

label は `"speech"` / `"non_speech"` / `"noisy_speech"` のいずれか (`noisy_speech` は `speech` と同等扱い、reject されたら false reject)。

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
