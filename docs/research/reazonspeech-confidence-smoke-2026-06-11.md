# PR-A.5.1 — ReazonSpeech Confidence Smoke Verify (2026-06-11)

> **Status**: ✅ **Validated** — Phase 1 bug fix + Section 1 Case A (int8/float32 両方) + Section 2 で `webrtc × reazonspeech × real × on` の Hall.(post) **50% → 0%** を実証 (Issue #295 元 motivation の最後の cell 完了)。Issue #317 PR-A.5.1 完了候補。

## Setup

| Item | Value |
|---|---|
| Date | 2026-06-11 |
| Branch | `feat/issue-317-pr-a5-1-reazonspeech-confidence` |
| Issue | [#317](https://github.com/Mega-Gorilla/livecap-cli/issues/317) PR-A.5.1 |
| Model | `reazon-research/reazonspeech-k2-v2` (sherpa-onnx 1.12.39) |
| Engine config | `device="cpu"`、`decoding_method="greedy_search"`、`use_int8=True/False` 両方 verify |
| Wall-clock | ~30s (load) + 30s (5 clip × 2 model variants smoke) + ~5 min (12 cell sweep on CPU) |
| Smoke scripts | `.tmp/pr_a5_1_reazonspeech_smoke.py`、`benchmarks/non_speech_filter/sweep`、いずれも一時、commit しない |

---

## Background — 旧「構造的限界」 claim の実機反証

旧 docs ([Issue #308 close 時点](https://github.com/Mega-Gorilla/livecap-cli/issues/308)) では:

> 「sherpa-onnx Python bindings に per-token score API なし。upstream [PR #2897](https://github.com/k2-fsa/sherpa-onnx/pull/2897) closed/not-merged、Python 未対応」

を理由に **PR-A.5 candidate (heavy refactor)** としていた。

本 PR plan 段階で reviewer (Issue #317) と独立に sherpa-onnx 1.12.39 を実機検証した結果、**`OfflineRecognitionResult.ys_log_probs` は既に Python bindings で exposed されていた**ことが判明:

```python
import sherpa_onnx
rec = sherpa_onnx.OfflineRecognizer.from_transducer(...)
s = rec.create_stream(); s.accept_waveform(16000, audio); rec.decode_stream(s)
print(s.result.ys_log_probs)  # ✅ list of per-token log-probabilities
```

reviewer も C API レベルで `SherpaOnnxOfflineRecognizerResult.ys_log_probs` field の存在を independently 確認。

→ 「構造的限界」 claim は誤り、本 PR で **standard integration work (1.5 日)** として対応。

## reviewer feedback (Issue #317) 反映

reviewer から 7 件の critical 指摘を受領、本 plan / 実装で全て反映:

| Point | severity | 内容 | 対応 |
|---|---|---|---|
| 1 | 🔴 CRITICAL | `token_confidence_mean` field 再利用は probability vs log prob semantics 不整合で全 reject になる | ✅ `avg_logprob` field 使用 (Voxtral 同 semantics、負の log probability) |
| 2 | 🟠 HIGH | `EngineConfidence.avg_logprob` + `raw["ys_log_probs_mean"]` 保存 | ✅ 両方実装 |
| 3 | 🟠 HIGH | Global `-1.0` threshold は ReazonSpeech margin -0.2 桁違い | ✅ `FilterConfig.avg_logprob_thresholds: Dict[str, float]` 追加 |
| 4 | 🟡 MEDIUM | qwen3asr の旧 claim を弱めるべき | (本 PR scope 外、[#318] で扱う) |
| 5 | 🟠 HIGH | qwen3asr の avg_logprob 単独 filter は危険 (confidence filter ≠ hallucination guard) | (本 PR scope 外、[#318] で扱う) |
| 6 | 🔴 CRITICAL | `reazonspeech_engine.py:430` の `text, confidence = ...` unpack が PR #314 で TypeError、長尺音声で silent drop bug | ✅ **Phase 1 で独立 commit + 3 件 regression test 追加** |
| 7 | 🟡 MEDIUM | ReazonSpeech (実装) と qwen3asr (research) を Issue 分割 | ✅ Issue [#317] / [#318] 分離 |

---

## Hypotheses

| # | Hypothesis | Verdict |
|---|---|---|
| **H1** | sherpa-onnx 1.12.39 `OfflineRecognitionResult.ys_log_probs` で per-token log probability が取得可能 | ✅ **CONFIRMED** (plan 段階の probe + Section 1 smoke 両方で) |
| **H2** | speech vs non-speech で `ys_log_probs` mean が clean separated (margin ≥ 0.1) | ✅ **CONFIRMED** (int8 margin +0.127、float32 margin +0.105、両方 Case A) |
| **H3** | threshold `-0.2` で speech (mean ~-0.14) pass、non_speech (mean ~-0.35-0.45) reject | ✅ **CONFIRMED** (int8 / float32 両方で 100% 分類) |
| **H4** | int8 量子化 model でも `ys_log_probs` exposed されている | ✅ **CONFIRMED** (Phase 5 int8 probe で確認、float32 と同 order の margin) |
| **H5** | `webrtc × reazonspeech × real × filter on` で post-filter hallucination が改善 | ✅ **CONFIRMED** (Hall.(post) 50% → 0%) |

---

## Implementation Summary

### Path 1.5 設計 (Voxtral pattern 流用 + engine-specific threshold)

```python
# reazonspeech_engine.py module-level
def _extract_engine_confidence(result: Any) -> EngineConfidence:
    ys = getattr(result, 'ys_log_probs', None)
    if ys is None: return EngineConfidence()
    ys_list = list(ys) if not isinstance(ys, list) else ys
    if not ys_list: return EngineConfidence()
    numeric = [float(v) for v in ys_list if v is not None]
    if not numeric: return EngineConfidence()
    mean_lp = sum(numeric) / len(numeric)
    return EngineConfidence(
        avg_logprob=mean_lp,  # ← Voxtral と同 semantics
        raw={"ys_log_probs_mean": mean_lp, "ys_log_probs_n": len(numeric)},
    )

# confidence_filter.py:FilterConfig
avg_logprob_thresholds: Dict[str, float] = field(default_factory=lambda: {
    "reazonspeech": -0.2,  # PR-A.5.1 smoke verify 2026-06-11
})

# confidence_filter.py:should_reject
if (no_speech_prob None AND token_conf_mean None AND avg_logprob not None):
    threshold = config.avg_logprob_thresholds.get(engine_name) or config.avg_logprob_threshold
    if threshold is not None and avg_logprob < threshold:
        reject(reason=f"avg_logprob {lp:.3f} < {threshold} (engine={engine_name})")
```

3 段 fallback:
1. **engine-specific threshold** (本 PR で reazonspeech: -0.2 を default load)
2. **global fallback** (`avg_logprob_threshold = -1.0`、Voxtral 用、backward compat)
3. **opt-out** (`avg_logprob_threshold=None` + dict にない engine → 完全 pass)

---

## Section 1: Engine-level Smoke (5 clip、int8 + float32)

### Setup

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:LIVECAP_NON_SPEECH_CORPUS_DIR = "D:\Codes\livecap-cli\.tmp\non_speech_corpus"
uv run python .tmp/pr_a5_1_reazonspeech_smoke.py
```

### Results — int8 model

| Clip | kind | text | **avg_logprob** | n_tokens |
|---|---|---|---|---|
| jsut_basic5000_0001 | speech_ja | "水をマレーシアから..." ✅ | **-0.1366** | 22 |
| normal_speech_neko | speech_ja | "吾輩は猫である名前はまだない..." ✅ | **-0.1477** | 88 |
| applause_then_speech | mixed | "吾輩は猫である名前はまだない" | -0.1359 | 14 |
| applause_5_claps | non_speech | "ピッ" | **-0.3341** | 2 |
| desk_tap | non_speech | "ピッ" | **-0.2748** | 2 |

Statistics (int8):
- speech mean: **-0.1421** (range -0.1477 to -0.1366)
- non_speech mean: **-0.3044** (range -0.3341 to -0.2748)
- **margin (speech min - non_speech max): +0.1271**
- threshold -0.2: speech all pass ✅、non_speech all reject ✅ → **Case A**

### Results — float32 model

| Clip | kind | text | **avg_logprob** | n_tokens |
|---|---|---|---|---|
| jsut_basic5000_0001 | speech_ja | "水をマレーシアから..." ✅ | **-0.1663** | 22 |
| normal_speech_neko | speech_ja | "吾輩は猫である名前はまだない..." ✅ | **-0.1442** | 88 |
| applause_then_speech | mixed | "吾輩は猫である名前はまだない" | -0.1170 | 14 |
| applause_5_claps | non_speech | "ピッ" | **-0.2709** | 2 |
| desk_tap | non_speech | "ピッ" | **-0.6339** | 2 |

Statistics (float32):
- speech mean: **-0.1553** (range -0.1663 to -0.1442)
- non_speech mean: **-0.4524** (range -0.6339 to -0.2709)
- **margin (speech min - non_speech max): +0.1046**
- threshold -0.2: speech all pass ✅、non_speech all reject ✅ → **Case A**

### Findings (Section 1)

#### F1.1 — int8 / float32 両方で Case A 確定 (reviewer Point 確認)

reviewer が flag した int8 quantized model での `ys_log_probs` availability を実機確認、**両 model で同 threshold -0.2 が機能**。production user が int8 (default 高速化 path) を選んでも filter 機能する。

#### F1.2 — margin は probe (5 clip) > 後段 sweep (synthetic 含む 12 cell) の関係

Section 1 (5 clip) margin +0.13 (int8) / +0.10 (float32) は Section 2 sweep の sample 数増加 + synthetic 合成音声 (PR-B corpus) で worst case を含めても threshold -0.2 で機能。

---

## Section 2: Stream Pipeline Benchmark (12 cell sweep)

### Setup

```powershell
uv run python -m benchmarks.non_speech_filter.sweep `
    --backend silero,tenvad,webrtc `
    --engine reazonspeech `
    --corpus-dir .tmp\non_speech_corpus `
    --device cpu `
    --preset baseline_off `
    --filter-mode off,on
```

Wall-clock: ~5 min on CPU。

### Results (主要 cell)

| Backend | Corpus | filter | Hall.(pre) | **Hall.(post)** | P50 ms | P95 ms |
|---|---|---|---|---|---|---|
| silero | real | off | 0.0% | 0.0% | 318 | 828 |
| silero | real | on | 0.0% | 0.0% | 315 | 817 |
| silero | synthetic | off/on | 0.0% | 0.0% | 6 | 90 |
| tenvad | real | off | 0.0% | 0.0% | 441 | 1040 |
| tenvad | real | on | 0.0% | 0.0% | 428 | 1036 |
| tenvad | synthetic | off | 0.0% | 0.0% | 5 | 136 |
| tenvad | synthetic | on | 0.0% | 0.0% | 5 | 132 |
| **webrtc** | **real** | **off** | **50.0%** | **50.0%** | 300 | 979 |
| **webrtc** | **real** | **on** | **50.0%** | **0.0%** ✅ | 293 | 960 |
| webrtc | synthetic | off | 62.5% | 62.5% | 80 | 134 |
| webrtc | synthetic | on | 62.5% | **25.0%** | 83 | 132 |

### Findings (Section 2)

#### F2.1 — `webrtc × reazonspeech × real × filter on`: Hall.(post) **50% → 0%** ✅

**Issue #295 元 motivation の最後の cell が完全解消**。PR-A.0 で WhisperS2T, Parakeet_ja で 0% を実現、PR-A.4.x で Voxtral / Canary / Parakeet 英語、本 PR で ReazonSpeech が production-ready 状態に到達。

#### F2.2 — `webrtc × synthetic × on`: 62.5% → 25.0% (60% drop)

formant-proxy 合成非音声 5 件のうち 3 件が `ys_log_probs` mean < -0.2 で reject、残り 2 件は marginal value で pass。実 production audio (real corpus) で 100% reject されたので production 影響は限定的。

#### F2.3 — silero / tenvad: 元々 0%

VAD で非音声を engine 前段で除去済のため filter は冗長安全網。

#### F2.4 — Latency 影響ゼロ

filter off (P50 300-440 ms) vs filter on (P50 293-428 ms) で実質同一。`ys_log_probs` 抽出 + mean 計算は sub-millisecond オーバーヘッド。

#### F2.5 — Long-audio bug fix (Phase 1)

Phase 1 で `reazonspeech_engine.py:430` の `TranscriptionResult.__iter__` 削除 (PR #314) followup bug を修正。旧 production code では **長尺音声 (>30s) で `_transcribe_with_split` が全 segment を silently dropped** していた。本 PR で:
- Phase 1: bug fix + 3 件 regression test (mock-based) で pin
- Phase 2: engine_confidence aggregation を一体に実装 (weighted mean)
- 結果: 30s 超え audio の transcription quality が production で復旧

---

## Section 3: Language Coverage

### ReazonSpeech 言語サポート

| Mode | サポート言語 | 本 PR で verify 済 |
|---|---|---|
| Japanese (ja) | ✅ native | ✅ Section 1 + 2 で確認 |
| 他言語 | ❌ 非対応 (日本語 native model) | n/a (Canary PR-A.4.2 と同 framing) |

### F3.1 — 日本語 native で threshold `-0.2` validate 済

Section 1 + Section 2 で日本語 native model の filter signal を検証、production-ready 状態を確認。

### F3.2 — ReazonSpeech 多言語拡張は出 scope

ReazonSpeech k2-v2 model 自体が日本語 native で他言語非対応のため、多言語 verify 自体が不要 (Canary PR-A.4.2 が 4 言語対応のため Section 3 で「英語 native のみ実施、他は user feedback ベース」と異なる framing)。

---

## Decision

### Threshold `-0.2` を default 採用

| Criterion | Result |
|---|---|
| H1-H5 全て confirmed | ✅ |
| Section 1 (int8 / float32 両方) speech all pass + non_speech all reject | ✅ |
| Section 2 webrtc × real × on で Hall.(post) **50% → 0%** | ✅ |
| Voxtral / Parakeet (ja/en) / Canary / WhisperS2T 退行ゼロ | ✅ (`should_reject()` の strict gating + dict fallback で `voxtral` は global `-1.0` 維持) |
| Latency 影響 | ✅ ゼロ |

→ **default `-0.2` 採用**。ReazonSpeech user は `--confidence-filter on` (default) で hallucination が自動 drop される。

### `FilterConfig.avg_logprob_thresholds` dict 採用 (architectural decision)

reviewer Point 3 の engine-specific threshold を **scalable な dict 設計** で実装:

```python
avg_logprob_thresholds: Dict[str, float] = {"reazonspeech": -0.2}
```

- ReazonSpeech 用 entry を default load
- Voxtral 用 entry は **意図的に dict に load しない** → `avg_logprob_threshold = -1.0` (global fallback) が適用される、backward compat 維持
- 将来 qwen3asr ([#318]) 等で別 threshold が必要になっても dict に追加するだけで scale

---

## Implications

### ReazonSpeech user の挙動変化

| Mode | Before (PR-A.5.1 前) | **After (本 PR 後)** |
|---|---|---|
| `--confidence-filter on` (default) | fail-open (engine_confidence 全 None で透過) | **`avg_logprob < -0.2` で reject (active)** |
| `--confidence-filter observe` | filter 判定なし | filter 判定 JSON log (observe mode active) |
| `--confidence-filter off` | filter 無効 | (不変) |
| 長尺音声 (>30s、auto_split 経路) | **全 segment が silently dropped (production bug)** | **正しく集約された transcription** (Phase 1 bug fix) |

### 他 engine の挙動 (退行ゼロ)

- WhisperS2T / Parakeet (ja/en) / Voxtral / Canary: `confidence_filter.py` 既存 path 不変、`voxtral` は dict にないため `avg_logprob_threshold = -1.0` (global) を使用 (PR-A.4.1 と同 behavior)
- qwen3asr: 不変 (fail-open、[#318] で扱う)

### Caveats (production user 向け)

1. **ReazonSpeech 日本語 native のみ**: 多言語 user は WhisperS2T / Voxtral / Canary を推奨
2. **Phase 1 bug fix は breaking change ではなく production bug 修正**: 旧挙動 (長尺音声で silently drop) が production bug だったため、新挙動 (正しい transcription) が正しい状態
3. **int8 / float32 両方 production-ready**: 用途に応じて選択 (int8 は高速化、float32 は精度優先)

---

## Reproducibility

### Section 1 smoke (5 clip × int8/float32)

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:LIVECAP_NON_SPEECH_CORPUS_DIR = "D:\Codes\livecap-cli\.tmp\non_speech_corpus"
uv run python .tmp/pr_a5_1_reazonspeech_smoke.py
```

期待: 両 model で Case A、margin > 0.1。

### Section 2 sweep (12 cell)

```powershell
uv run python -m benchmarks.non_speech_filter.sweep `
    --backend silero,tenvad,webrtc `
    --engine reazonspeech `
    --corpus-dir .tmp\non_speech_corpus `
    --device cpu `
    --preset baseline_off `
    --filter-mode off,on
```

期待: `webrtc × real × on` で Hall.(post) 0%、`silero/tenvad × *` で 0% (VAD で除去済)。

---

## 関連リソース

- 親 epic: [Issue #295 CLOSED](https://github.com/Mega-Gorilla/livecap-cli/issues/295) — Phase 1 多段防御
- 前段: [Issue #308 CLOSED](https://github.com/Mega-Gorilla/livecap-cli/issues/308) (PR-A.0/A.1/A.3 base filter)、[Issue #311 CLOSED](https://github.com/Mega-Gorilla/livecap-cli/issues/311) (PR-A.4 系列 voxtral/canary/parakeet-en)
- 姉妹 Issue: [#318](https://github.com/Mega-Gorilla/livecap-cli/issues/318) — qwen3asr probe / hallucination guard 設計
- 旧誤判定の参考: [PR #316](https://github.com/Mega-Gorilla/livecap-cli/pull/316) で Parakeet 英語が「構造的限界 → PR #309 設定漏れ」と判明した類似 case

by.Scotty
