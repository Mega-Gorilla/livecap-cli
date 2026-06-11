# PR-A.4.1 — Voxtral Confidence Smoke Verify (2026-06-11)

> **Status**: ✅ **Case A** (clear margin) — `FilterConfig.avg_logprob_threshold` default を `None` → `-1.0` に変更採用。

## Setup

| Item | Value |
|---|---|
| Date | 2026-06-11 |
| Branch | `feat/issue-311-pr-a4-1-voxtral-confidence` |
| Issue | [#311 v2.1](https://github.com/Mega-Gorilla/livecap-cli/issues/311) PR-A.4.1 |
| Model | `mistralai/Voxtral-Mini-3B-2507` (HF cache、~6 GB) |
| Engine config | `device="cuda"`, `language="en"`, `do_sample=False` (greedy) |
| GPU | NVIDIA RTX 4090, CUDA 12.8, PyTorch 2.9.1+cu128 |
| Corpus | PR-B 6 clip @ `.tmp/non_speech_corpus/`、`manifest.json` schema list 直接 |
| Wall-clock | ~30s (load) + ~10s (6 clip transcribe) |
| Smoke script | `.tmp/pr_a4_1_voxtral_smoke.py` (永続化しない一時 script) |

---

## Hypotheses

| # | Hypothesis | Verdict |
|---|---|---|
| **H1** | Voxtral は speech に対して `avg_logprob ≥ -1.0` を出す | ✅ **CONFIRMED** (speech 4 clip max=-0.354、min=-0.523、worst でも -0.523 > -1.0) |
| **H2** | Voxtral は non-speech に対して `avg_logprob < -1.0` を出す | ✅ **CONFIRMED** (applause_5_claps: -1.525 << -1.0、desk_tap: empty → fail-open) |
| **H3** | speech vs non-speech に clear margin (≥ 1.0) あり | ✅ **CONFIRMED** (margin +1.002) |
| **H4** | `_extract_engine_confidence()` で special token (EOS/PAD/BOS) を除外する logic は正しく機能する | ✅ **CONFIRMED** (desk_tap は EOS のみ生成 → 全 token special → `EngineConfidence()` fallback で fail-open) |

---

## Results (実測 6 clip)

| Clip | kind | avg_logprob | is_available | Voxtral text (preview) |
|---|---|---|---|---|
| `short_utterances_mixed.wav` | positive (speech) | **-0.523** | True | `"Yes, okay, um. Yes, okay, um."` |
| `normal_speech_neko.wav` | positive (speech) | **-0.360** | True | `"I am a baby. I have no name..."` (吾輩は猫である 翻訳) |
| `applause_then_speech.wav` | positive (speech) | **-0.444** | True | `"I am a cat. My name is not yet known."` |
| `overlapping_applause_speech.wav` | positive (speech) | **-0.354** | True | `"I am a cat. I have no name..."` |
| `applause_5_claps.wav` | negative (non-speech) | **-1.525** | True | `"."` (低信頼度 filler) |
| `desk_tap.wav` | negative (non-speech) | **None** | **False** | (empty、全 token special) |

### Distribution summary

| Group | n | min | max | mean |
|---|---|---|---|---|
| **speech (positive)** | 4 | -0.523 | -0.354 | -0.420 |
| **non-speech (negative)** | 1 (+ 1 fail-open) | -1.525 | -1.525 | -1.525 |

### Margin / threshold

| Metric | Value |
|---|---|
| `margin = speech_min - non_speech_max` | `-0.523 - (-1.525) = +1.002` |
| `midpoint = (speech_min + non_speech_max) / 2` | `-1.024` |
| **Recommended threshold** | **`-1.0`** (clean number、Whisper 慣習値とも一致) |
| Case classification | **Case A — Clear margin (margin > 1.0)** |

---

## Decision

### `FilterConfig.avg_logprob_threshold` default を `-1.0` に変更

| Criterion | Result |
|---|---|
| H1-H4 全て確認 | ✅ |
| margin ≥ 1.0 (Case A 条件) | ✅ +1.002 |
| 全 speech clip が threshold pass | ✅ (worst -0.523 > -1.0) |
| 全 non-speech clip が threshold reject or fail-open | ✅ (-1.525 < -1.0、desk_tap は EOS のみで fail-open) |
| WhisperS2T / Parakeet_ja に副作用なし | ✅ (strict gate: 両方 None でない限り avg_logprob 評価しない) |

→ **default `-1.0` 採用**。Voxtral user は `--confidence-filter on` (= default) で hallucination が自動 drop される。

### Strict gating の再確認

```python
# confidence_filter.py::should_reject() より抜粋
if (
    ec.no_speech_prob is None              # WhisperS2T はここで block
    and ec.token_confidence_mean is None   # Parakeet_ja はここで block
    and ec.avg_logprob is not None
    and config.avg_logprob_threshold is not None
    and ec.avg_logprob < config.avg_logprob_threshold
):
    return True, f"avg_logprob {ec.avg_logprob:.3f} < {config.avg_logprob_threshold}"
```

→ Voxtral 以外 (= avg_logprob のみ populate する engine 以外) は新 threshold の影響を受けない。

---

## Implications

### Voxtral user の挙動変化

| Mode | Before (PR-A.4.1 前) | After (本 PR 後) |
|---|---|---|
| `--confidence-filter off` | filter 無効、Voxtral 出力そのまま | (不変) |
| `--confidence-filter observe` | filter 判定なし、log 不出力 | filter 判定 log 出力 (pass/reject 両方) |
| `--confidence-filter on` (default) | filter 判定なし (fail-open)、Voxtral hallucination 透過 | Voxtral hallucination drop (avg_logprob < -1.0 で reject) |
| Python API | `FilterConfig()` で avg_logprob 判定 off (`avg_logprob_threshold=None`) | `FilterConfig()` で active (`avg_logprob_threshold=-1.0`)、明示 None で opt-out 可能 |

### 他 engine の挙動 (不変)

- WhisperS2T: `no_speech_prob` populate → strict gate で avg_logprob 評価 skip → **退行ゼロ**
- Parakeet_ja: `token_confidence_mean` populate → 同上 → **退行ゼロ**
- ReazonSpeech / qwen3asr / Canary / mock: engine_confidence 全 None → `is_available=False` → fail-open → **不変**

### Caveats (production user 向け)

1. **言語非依存性**: 本 smoke verify は `language="en"` で 6 clip 英語 transcription。Voxtral は 8 言語サポート (`en, es, fr, pt, hi, de, nl, it`)。他言語の avg_logprob 分布が同じ範囲か別途 verify が望ましい (PR-A.4.1 scope 外、user feedback で対応)。
2. **`do_sample=True` 時の semantics 違い**: smoke verify は greedy mode で実施。sampling mode では avg_logprob 値が sample 軌跡依存になるため、threshold `-1.0` の妥当性が変わる可能性 — `do_sample=True` 時の caveat を transcribe() docstring に記載済。
3. **HF cache 容量**: Voxtral-Mini-3B model は初回 download で ~6 GB の disk 消費。

---

## Reproducibility

```powershell
# 環境変数
$env:PYTHONIOENCODING = "utf-8"
$env:LIVECAP_NON_SPEECH_CORPUS_DIR = "D:\Codes\livecap-cli\.tmp\non_speech_corpus"

# 一時 script (本 PR には含めない)
uv run python .tmp/pr_a4_1_voxtral_smoke.py
```

期待出力: 上記 Results table を再現。Voxtral model は HF cache 初回 download 約 1-2 分、その後 transcribe 5-10 秒。

---

## 関連リソース

- 親 Issue: [#311 v2.1](https://github.com/Mega-Gorilla/livecap-cli/issues/311) — PR-A.4 (voxtral + canary scope)
- Feasibility comment: [#311#issuecomment-4677116650](https://github.com/Mega-Gorilla/livecap-cli/issues/311#issuecomment-4677116650)
- 前段 PR-A.0 schema: [PR #309 MERGED](https://github.com/Mega-Gorilla/livecap-cli/pull/309)
- 前段 PR-A.1 filter impl: [PR #310 MERGED](https://github.com/Mega-Gorilla/livecap-cli/pull/310)
- 前段 PR-A.3 calibration: [PR #312 MERGED](https://github.com/Mega-Gorilla/livecap-cli/pull/312) (engine 全体 sweep)
- 本 PR-A.4.1 (本 doc 永続化先): `feat/issue-311-pr-a4-1-voxtral-confidence`

by.Scotty
