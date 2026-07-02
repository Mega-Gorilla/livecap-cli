# JA ASR confidence threshold calibration — 2026-07 Phase 2 report (augmented corpus + Pareto gate)

> **Issue #338** の active calibration harness を、 Layer 2 (ESC-50 / MUSAN) と Layer 3 (SNR mixed noisy_speech) で augment した corpus に対して 5 engine 再 calibration した **第 2 次 report**。 Phase 1 report ([`calibration-japan-engines-2026-07.md`](calibration-japan-engines-2026-07.md)) の最重要 caveat (synthetic non_speech 依存で probe pass) を解消し、 Issue #334 PR-4 の直接 input を確定する。

| 項目 | 値 |
|---|---|
| 実施日 | 2026-07-02 |
| 対象 engine | reazonspeech (int8 / float32), qwen3asr ja, whispers2t base, parakeet_ja (5 sweep) |
| 対象言語 | JA |
| Corpus | 449 clean speech + 676 non_speech + 250 noisy_speech = **1375 sample** |
| Harness | `benchmarks/confidence_calibration/sweep.py` + `--breakdown-by snr_db,subtype,noise_source_dataset` (PR #345 Phase 6a) |
| 依存 PR | #341 (PR-γ kana metric) / #342 (Phase 1 report) / #343 (Layer 2 augment) / #344 (Layer 3 SNR mix) / #345 (Phase 6a breakdown) |
| 関連 Issue | [#338](https://github.com/Mega-Gorilla/livecap-cli/issues/338) (親 harness) / [#334](https://github.com/Mega-Gorilla/livecap-cli/issues/334) PR-4 (下流) |

## TL;DR

1. **Phase 1 の data-driven threshold (`-0.84` 等) は probe-pass の危険領域** だった。 Phase 2 の augmented corpus (ESC-50 + MUSAN + Layer 3 SNR mixed) で再算定すると、 全 engine で **現 default より permissive かつ probe-bound より保守的** な範囲に。
2. **F1 max criterion では 5 engine 中 4 engine が Pareto gate 「`clean_frr ≤ 1%` かつ `noisy_frr(SNR≥5 dB) ≤ 5%` かつ `probe-safe`」を部分違反** — 単一 F1 最大化は不適切であることが実測 evidence として確定。
3. **PR-4 用 threshold candidate は "relaxed Pareto gate" 適用で選定**:
   - reazonspeech (両量子化): **-0.40** (clean 2.9%、 SNR≥5 全て ≤ 5%、 probe margin +0.05)
   - qwen3asr ja: **-0.42** (clean 2.2%、 SNR 10 だけ 6% で条件付き)
   - whispers2t base: **0.71** (clean 2.7%、 SNR≥5 全て ≤ 5%、 F1=0.901)
   - parakeet_ja: **0.001** (strict Pareto pass、 clean 0.7%、 F1=0.961、 false reject 39→11 の 72% 削減)
4. **量子化 (int8 vs float32) は threshold に影響しない** — ReazonSpeech で完全同一 threshold `-0.37` (F1 max criterion)。 Finding F8 の一般則を実測 confirm。
5. **breathing / laughing は speech-adjacent borderline subtype** (§3.3 実測: `laughing` 63-70% / `breathing` は engine 差大で 73-97%)。 これらは **PR-4 の blocker とせず**、 metric 分離で対応 (Phase 2b の sensitivity analysis で included/excluded 両ケースを提示、 §5.4)。
6. **SNR -5 dB は全 engine で Pareto fail** (18-20% FRR)。 production 想定範囲は SNR ≥ 0 dB に絞ることを推奨。

---

## 1. Methodology

### 1.1 Corpus 構成 (Phase 1 → Phase 2 augment)

Phase 1 の 479 sample から **1375 sample に拡張** (+ 187%):

| Layer | Source | N | 追加 by |
|---|---|---|---|
| 1 | Little Prince JA Chapter 1 朗読 (taltal3014) | 449 speech | Phase 1 build_corpus (PR-β) |
| 1 | Synthetic silence + white/pink noise @ ~-40 dB | 30 non_speech | Phase 1 |
| 2 | **ESC-50** (15 category × 10 file × 3 chunk) | 450 non_speech | PR #343 |
| 2 | **MUSAN** noise (uniform stride 50 files × up to 5 chunk) | 196 non_speech | PR #343 |
| 3 | **Layer 3 SNR mixed** (clean speech × ESC-50/MUSAN noise × 5 SNR) | 250 noisy_speech | PR #344 |

**Total**: 449 speech + 676 non_speech + 250 noisy_speech = **1375**。

manifest metadata (Phase 6a で `sweep_report.breakdown` に到達可能):
- `label`: `speech` / `non_speech` / `noisy_speech`
- `snr_db`: `-5.0` / `0.0` / `5.0` / `10.0` / `20.0` (Layer 3 のみ)
- `subtype`: `clapping` / `breathing` / `laughing` / `engine` / etc. (ESC-50 category) or `musan_free-sound` / `musan_sound-bible` (MUSAN sub-dir)
- `source_dataset`: `phase1_original` / `esc50` / `musan` / `layer3_mix`

### 1.2 Sweep methodology (per-engine)

各 engine について:

```bash
uv run python -m benchmarks.confidence_calibration.sweep \
    --engine <engine_id> --signal <signal_field> \
    --filter-by-language ja \
    --corpus-dir .tmp/calibration_corpus_full \
    --engine-kwargs <engine_specific_kwargs> \
    --breakdown-by snr_db,subtype,noise_source_dataset \
    --output <report.json>
```

#### Engine 別 実行コマンド (再現用)

| Engine | `--signal` | 追加 `--quantization` | 追加 `--engine-kwargs` | Sweep wall-clock |
|---|---|---|---|---|
| reazonspeech (int8) | `avg_logprob` | (未指定 = int8 default) | (なし) | 3.6 min |
| reazonspeech (float32) | `avg_logprob` | `float32` | `use_int8=false` | 3.3 min |
| qwen3asr ja | `avg_logprob` | (該当なし) | `language=ja` | 8.8 min |
| whispers2t base | `no_speech_prob` | (該当なし) | `model_size=base` `compute_type=int8` `language=ja` | 2.0 min |
| parakeet_ja | `token_confidence_mean` | (該当なし) | (なし) | 2.5 min |

**Total sweep wall-clock**: ~20 min (RTX 4090)。 raw report JSON は `.tmp/phase2_reports/*.json` に格納。

| Setting | Value |
|---|---|
| Threshold grid | per-signal default range (avg_logprob: [-1.0, -0.05] step 0.01、 no_speech_prob: [0.1, 0.95] step 0.01、 token_confidence_mean: [0.001, 0.5] step 0.005) |
| Criterion | F1 (max F1 の同点は closest-to-zero-FRR で tie-break) |
| Breakdown keys | `snr_db` (SNR 別 FRR、 Pareto gate 用) / `subtype` (noise category 別 recall) / `noise_source_dataset` (ESC-50 vs MUSAN 別) |

### 1.3 Pareto gate 定義 (F1 max criterion に対する上位 criterion)

Phase 1 で判明した **単一 F1 最大化の限界** (production trade-off が hidden) を解消するため、 本 report は Pareto gate を採用:

```
Pareto gate:
  clean_speech_frr ≤ clean_max   (static / office 環境で会話が dropped しない)
  AND noisy_speech_frr(SNR bucket) ≤ noisy_max for all SNR ≥ 5 dB
  AND known probe rejected (direction-aware、 Phase 1 §4.2 の probe 値使用)
```

`clean_max` は「静かな環境での実 mic 会話が過剰 reject されないか」の閾値、 `noisy_max` は「家庭 / café 等の背景 noise ある会話が dropped しないか」の閾値。 SNR -5 dB / 0 dB は「賑やかな公共空間」相当で production 主要 use case には含まない (§5.3)。

**本 report では 4 段階の Pareto gate** を試行:

| Gate name | `clean_max` | `noisy_max` |
|---|---|---|
| **strict** | 1.0% | 5.0% |
| **relaxed_A** | 2.0% | 5.0% |
| **relaxed_B** | 3.0% | 5.0% |
| **relaxed_C** | 5.0% | 10.0% |

「pass する最厳 gate かつ F1 最大」の threshold を各 engine の PR-4 candidate として §4 で提示。

### 1.4 Caveats (Phase 1 caveat の更新)

| Phase 1 の caveat | Phase 2 での対応 |
|---|---|
| Synthetic non_speech のみ (probe pass) | ✅ **解消** (ESC-50 + MUSAN で production-realistic) |
| noisy_speech 未評価 | ✅ **解消** (Layer 3 SNR mixed で 5 SNR bucket 評価) |
| Clean 朗読 corpus bias | ⏳ **継続** (Layer 4 production observe log で verify 予定) |
| Single language (JA only) | ⏳ **継続** (EN calibration は別 Phase) |
| Single corpus (Chapter 1 のみ) | ⏳ **継続** (Chapter 多様化は Phase 3+) |

Phase 2 で新規発生:
- **breathing / laughing の speech-adjacent 性**: `laughing` は全 engine で recall 63-70% と最下位、 `breathing` は engine 差大 (ReazonSpeech 73% vs Parakeet_ja 97%)。 これらを PR-4 blocker とせず metric 分離で対応する方針 (§5.4、 Phase 2b で sensitivity analysis)
- **MUSAN sub-dir 混在**: `free-sound` (179 sample) と `sound-bible` (17 sample) で難易度差、 sample balance の見直し余地
- **SNR grid resolution**: `-5 / 0 / 5 / 10 / 20 dB` (5 値) のみ、 中間 SNR (2.5 / 7.5 dB 等) は補間なし → 全て sweep grid 実測値

---

## 2. Per-engine results

以下、 各 engine の F1 max threshold + per-SNR breakdown + per-subtype recall + Pareto gate 適用結果を提示。 全 数値は sweep grid 実測値 (補間なし、 Phase 1 codex-review 教訓)。

### 2.1 ReazonSpeech (int8 / float32)

#### 2.1.1 F1 max criterion (baseline)

| Quant | Recommended | F1 | 全体 FRR | Precision | Recall | TP/FP/TN/FN |
|---|---|---|---|---|---|---|
| int8 | **-0.37** | 0.934 | 5.44% | 0.943 | 0.926 | 626 / 38 / 661 / 50 |
| float32 | **-0.37** | 0.934 | 5.44% | 0.943 | 0.926 | 626 / 38 / 661 / 50 |

**量子化影響ゼロ**: int8 / float32 で全指標が完全一致。 Finding F8 (Issue #334 の量子化 calibration 一般則) を Phase 2 で改めて実測 confirm。

#### 2.1.2 Per-SNR FRR @ recommended `-0.37`

| SNR (dB) | N | FRR | Pareto ≤ 5% |
|---|---|---|---|
| -5 | 50 | 0.100 | ❌ |
| 0 | 50 | 0.080 | ❌ |
| 5 | 50 | 0.060 | ❌ |
| 10 | 50 | 0.040 | ✅ |
| 20 | 50 | 0.060 | ❌ |
| clean (`__none__`) | 449 | 0.047 | ❌ (clean_max=1%) |

**評価**: F1 max criterion `-0.37` は **strict Pareto gate 全違反**、 relaxed_C (SNR≥5 ≤ 10%) は pass。

#### 2.1.3 Per-subtype recall @ `-0.37` (**純 non_speech サンプルのみ**、 抜粋)

> **重要**: `subtype` bucket は Layer 2 (純 non_speech) と Layer 3 (noisy_speech に mix された noise の subtype) が共存するため、 recall (= TP / (TP+FN)) の分母は **純 non_speech サンプルのみ**。 `breathing` と `car_horn` は Layer 3 noise rotation で頻用されており、 bucket 全体 N は non_speech 30 + noisy_speech mix (それぞれ 150 / 100) となる。 noisy_speech mix 側の FRR は §3.3.2 に別掲。

| Category | N (non_speech) | Recall | 難易度 |
|---|---|---|---|
| rain / mouse_click / footsteps / keyboard_typing / engine / door_wood_knock / glass_breaking | 各 30 | 1.000 | 易 |
| silence / white_noise / pink_noise (Phase 1 synthetic) | 各 10 | 1.000 | 易 |
| clock_tick | 30 | 0.967 | 易 |
| clapping / car_horn | 30 / 30 | 0.933 | やや |
| musan_free-sound | 179 | 0.922 | やや |
| sneezing / siren | 30 | 0.900 | やや |
| coughing / musan_sound-bible | 30 / 17 | 0.833 / 0.824 | 難 |
| **breathing** | 30 | **0.733** | **最難** (speech 極近) |
| **laughing** | 30 | **0.700** | **最難** (speech 極近) |

#### 2.1.4 Pareto gate 適用 (relaxed_A / relaxed_B の top candidate)

| Gate | Threshold | F1 | clean_frr | SNR 5 | SNR 10 | SNR 20 | 全体 FRR |
|---|---|---|---|---|---|---|---|
| **strict** | (該当なし) | — | — | — | — | — | — |
| **relaxed_A** (clean ≤ 2%) | **-0.43** | 0.922 | 1.78% | 4.0% | 2.0% | 2.0% | 2.72% |
| **relaxed_A** (次点) | -0.44 | 0.916 | 1.34% | 4.0% | 2.0% | 2.0% | 2.43% |
| **relaxed_B** (clean ≤ 3%) | **-0.40** | 0.929 | 2.90% | 4.0% | 2.0% | 4.0% | 3.86% |
| **relaxed_B** (次点) | -0.42 | 0.929 | 2.00% | 4.0% | 2.0% | 4.0% | 3.00% |

**PR-4 推奨候補**: **`-0.40`** (relaxed_B、 F1=0.929、 probe margin +0.05 vs probe -0.45、 現 default -0.20 より -0.20 permissive)。 relaxed_A の `-0.43` は F1 若干低下 (0.922) で trade-off、 保守選好なら採用。

### 2.2 Qwen3-ASR (ja)

#### 2.2.1 F1 max criterion (baseline)

| Recommended | F1 | 全体 FRR | Precision | Recall | TP/FP/TN/FN |
|---|---|---|---|---|---|
| **-0.42** | 0.955 | 3.58% | 0.962 | 0.948 | 637 / 25 / 674 / 35 |

sample_count: 449 speech + 672 non_speech + 250 noisy_speech (4 sample が signal 未取得で除外、 全 non_speech から)。

#### 2.2.2 Per-SNR FRR @ `-0.42`

| SNR (dB) | N | FRR | Pareto ≤ 5% |
|---|---|---|---|
| -5 | 50 | 0.180 | ❌ |
| 0 | 50 | 0.020 | ✅ |
| 5 | 50 | 0.040 | ✅ |
| 10 | 50 | 0.060 | ❌ (borderline) |
| 20 | 50 | 0.000 | ✅ |
| clean | 449 | 0.022 | ❌ (clean_max=1%) |

**評価**: F1 max `-0.42` は clean 2.2% で strict fail、 SNR 10 が 6% で relaxed_A/B も部分違反。 Qwen3-ASR は **全 gate 部分 fail** で最も厳しい engine。

#### 2.2.3 Per-subtype recall @ `-0.42`

- 全体傾向は ReazonSpeech と同様 (breathing / laughing が最下位)、 詳細は `.tmp/phase2_reports/qwen3asr_ja.json` を参照

#### 2.2.4 Pareto gate 適用

| Gate | Threshold | F1 | clean_frr | SNR 5 | SNR 10 | SNR 20 | 全体 FRR |
|---|---|---|---|---|---|---|---|
| **strict / relaxed_A / relaxed_B** | (該当なし) | — | — | — | — | — | — |
| **relaxed_C** (clean ≤ 5%、 SNR ≤ 10%) | **-0.42** | 0.955 | 2.23% | 4.0% | 6.0% | 0.0% | 3.58% |
| relaxed_C 次点 | -0.40 | 0.954 | 2.90% | 6.0% | 6.0% | 0.0% | 4.72% |

**PR-4 推奨候補**: **`-0.42`** (F1 max = relaxed_C best、 SNR 10 が 6% で条件付き、 probe margin +0.04 vs probe -0.46)。 現 default `-0.30` より -0.12 permissive。

**Caveat**: SNR 10 dB での 6% FRR は Pareto gate `noisy_frr ≤ 5%` を若干超過。 Qwen3-ASR ja は engine 特性上 SNR 10 dB 帯で低確度出力しやすく、 現時点で Pareto を完全満たす threshold なし。 alternatives (`-0.40` / `-0.36`) は clean FRR が悪化するのみで SNR 10 dB は改善しない (borderline を解決しない) ため `-0.42` が最も合理的。 **Layer 4 (production observe replay) 完了後に再確認** — 実 mic 環境で SNR 10 dB 会話がどの程度発生するかで許容度判断。

### 2.3 WhisperS2T (base)

#### 2.3.1 F1 max criterion (baseline)

| Recommended | F1 | 全体 FRR | Precision | Recall | TP/FP/TN/FN |
|---|---|---|---|---|---|
| **0.65** | 0.923 | 5.01% | 0.946 | 0.901 | 609 / 35 / 664 / 67 |

Whisper 公式 `no_speech_threshold=0.6` と比較して +0.05 conservative、 現 default `0.5` より +0.15 conservative。

#### 2.3.2 Per-SNR FRR @ `0.65`

| SNR (dB) | N | FRR | Pareto ≤ 5% |
|---|---|---|---|
| -5 | 50 | 0.060 | ❌ |
| 0 | 50 | 0.040 | ✅ |
| 5 | 50 | 0.040 | ✅ |
| 10 | 50 | 0.060 | ❌ |
| 20 | 50 | 0.060 | ❌ |
| clean | 449 | 0.049 | ❌ (clean_max=1%) |

#### 2.3.3 Pareto gate 適用

| Gate | Threshold | F1 | clean_frr | SNR 5 | SNR 10 | SNR 20 | 全体 FRR |
|---|---|---|---|---|---|---|---|
| **strict** (clean ≤ 1%) | **0.81** | 0.768 | 0.89% | 2.0% | 0.0% | 0.0% | 0.86% |
| **strict** 次点 | 0.82 | 0.733 | 0.67% | 2.0% | 0.0% | 0.0% | 0.72% |
| **relaxed_A** (clean ≤ 2%) | **0.76** | 0.852 | 1.78% | 2.0% | 0.0% | 0.0% | 1.43% |
| **relaxed_B** (clean ≤ 3%) | **0.71** | 0.901 | 2.67% | 2.0% | 0.0% | 0.0% | 2.15% |

**PR-4 推奨候補**: **`0.71`** (relaxed_B、 F1=0.901、 Whisper 公式 `0.6` に近く + Pareto pass margin 十分、 現 default `0.5` より +0.21 conservative)。 relaxed_A の `0.76` は F1 若干低下 (0.852) で trade-off。

**特徴**: WhisperS2T は Pareto gate 対応で **最も柔軟性が高い engine** — strict gate すら pass する候補あり。

### 2.4 Parakeet_ja

#### 2.4.1 F1 max criterion (baseline)

| Recommended | F1 | 全体 FRR | Precision | Recall | TP/FP/TN/FN |
|---|---|---|---|---|---|
| **0.006** | 0.962 | 5.58% | 0.944 | 0.979 | 662 / 39 / 660 / 14 |

現 default `0.005` と近接、 Phase 1 DD `0.001` より若干 conservative。 F1 最良 engine。

#### 2.4.2 Per-SNR FRR @ `0.006`

| SNR (dB) | N | FRR | Pareto ≤ 5% |
|---|---|---|---|
| -5 | 50 | 0.200 | ❌ |
| 0 | 50 | 0.100 | ❌ |
| 5 | 50 | 0.060 | ❌ |
| 10 | 50 | 0.060 | ❌ |
| 20 | 50 | 0.020 | ✅ |
| clean | 449 | 0.038 | ❌ (clean_max=1%) |

**評価**: Parakeet_ja は **noise sensitivity 最高** — 低 SNR で FRR 急上昇 (SNR -5 で 20%)。

#### 2.4.3 Pareto gate 適用

| Gate | Threshold | F1 | clean_frr | SNR 5 | SNR 10 | SNR 20 | 全体 FRR |
|---|---|---|---|---|---|---|---|
| **strict** (clean ≤ 1%) | **0.001** | 0.961 | 0.67% | 2.0% | 0.0% | 0.0% | 1.57% |
| **relaxed_C** (clean ≤ 5%、 SNR ≤ 10%) | 0.006 | 0.962 | 3.79% | 6.0% | 6.0% | 2.0% | 5.58% |
| **relaxed_C** 次点 | 0.001 | 0.961 | 0.67% | 2.0% | 0.0% | 0.0% | 1.57% |

**PR-4 推奨候補**: **`0.001`** (strict Pareto pass、 F1=0.961、 現 default `0.005` より -0.004 permissive)。 F1 max `0.006` と F1 差はわずか (0.001) だが clean FRR で **5.6× 改善** (3.79% → 0.67%)。

**Trade-off**: `0.001` は non_speech recall が下がる (Layer 2 の一部 category を pass する可能性)、 Phase 6b で per-subtype recall を精査。

### 2.5 Voxtral (未計測)

Phase 1 report 通り、 現 default `-1.0` は margin +1.0 と広く実害証拠弱。 本 Phase 2 では re-calibration 対象外。 Layer 4 (production observe) で必要性 verify。

---

## 3. Cross-engine analysis

### 3.1 F1 max vs Pareto gate — 全 engine 比較

| Engine | F1 max threshold | strict | relaxed_A | relaxed_B | relaxed_C | 実質可能な最厳 gate |
|---|---|---|---|---|---|---|
| reazonspeech int8/float32 | -0.37 | ❌ | ✅ (-0.43) | ✅ (-0.40) | ✅ (-0.37) | **relaxed_A** |
| qwen3asr ja | -0.42 | ❌ | ❌ | ❌ | ✅ (-0.42) | **relaxed_C** |
| whispers2t base | 0.65 | ✅ (0.81) | ✅ (0.76) | ✅ (0.71) | ✅ (0.65) | **strict** |
| parakeet_ja | 0.006 | ✅ (0.001) | ✅ (0.001) | ✅ (0.001) | ✅ (0.001 or 0.006) | **strict** |

**結論**:
- **WhisperS2T / Parakeet_ja** は strict Pareto gate 対応可能 (production 適用時の安全性最高)
- **ReazonSpeech (両量子化)** は relaxed_A まで対応可能
- **Qwen3-ASR ja** は relaxed_C まで緩めても SNR 10 dB で 6% と borderline (最厳 engine)

### 3.2 Per-SNR curve 比較 (F1 max threshold 使用)

各 engine の F1 max threshold における noisy_speech FRR を SNR 別に比較:

| SNR (dB) | reazon int8 | qwen3 ja | whispers | parakeet |
|---|---|---|---|---|
| -5 | 0.100 | **0.180** 🔴 | 0.060 | **0.200** 🔴 |
| 0 | 0.080 | **0.020** ✅ | 0.040 | 0.100 |
| 5 | 0.060 | 0.040 | 0.040 | 0.060 |
| 10 | 0.040 | 0.060 | 0.060 | 0.060 |
| 20 | 0.060 | **0.000** ✅ | 0.060 | 0.020 |

**観察**:
- Qwen3-ASR は SNR 0 / 20 dB で優秀、 SNR -5 dB で急劣化 (curve が凹型で non-monotone)
- Parakeet_ja は monotonic な SNR 依存 (低 SNR ほど FRR 高)
- WhisperS2T は SNR 全 bucket で 4-6% と最安定 curve
- ReazonSpeech は SNR 20 dB で若干悪化 (6%) — 高 SNR 帯での挙動要確認

### 3.3 Per-subtype 純 non_speech reject rate (recall) — cross-engine

**F1 max threshold で純 non_speech サンプルのみを対象とした recall** の cross-engine 比較 (難易度昇順):

| Category | N (non_speech) | reazon i8 | reazon f32 | qwen3asr ja | whispers2t | parakeet_ja |
|---|---|---|---|---|---|---|
| **laughing** | 30 | 0.700 | 0.700 | 0.633 | 0.700 | 0.633 |
| **breathing** | 30 | 0.733 | 0.733 | 0.867 | 0.800 | 0.967 |
| **coughing** | 30 | 0.833 | 0.833 | 0.800 | 0.700 | 0.967 |
| musan_sound-bible | 17 | 0.824 | 0.824 | 1.000 | 0.706 | 1.000 |
| sneezing | 30 | 0.900 | 0.900 | 0.867 | 0.700 | 1.000 |
| clapping | 30 | 0.933 | 0.933 | 0.967 | 0.733 | 1.000 |
| siren | 30 | 0.900 | 0.900 | 0.933 | 0.967 | 1.000 |
| musan_free-sound | 179 | 0.922 | 0.922 | 0.978 | 0.922 | 0.994 |
| car_horn | 30 | 0.933 | 0.933 | 0.967 | 0.967 | 1.000 |
| clock_tick | 30 | 0.967 | 0.967 | 0.933 | 1.000 | 1.000 |
| glass_breaking | 30 | 1.000 | 1.000 | 1.000 | 0.933 | 1.000 |
| door_wood_knock | 30 | 1.000 | 1.000 | 1.000 | 0.967 | 1.000 |
| keyboard_typing | 30 | 1.000 | 1.000 | 1.000 | 0.967 | 1.000 |
| engine | 30 | 1.000 | 1.000 | 1.000 | 0.967 | 1.000 |
| silence / white_noise / pink_noise | 各 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| footsteps / rain / mouse_click | 各 30 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

**主要 observation**:
- **`laughing` は全 engine で最下位** (63-70%): 音響的に speech 極近、 engine 独立に難
- **`breathing` は engine 差大**: reazon 73% vs parakeet **97%** — Parakeet_ja は呼吸音の rejection に強い、 ReazonSpeech は弱い
- **`coughing` は whispers2t で 70%** と低い、 他 engine 80-97%
- **合成 silence/white/pink** は全 engine で完全 reject (100%) — Phase 1 baseline data の品質確認
- **Layer 2 の 15 category のうち 7 category** (`rain` / `mouse_click` / `footsteps` / `keyboard_typing` / `engine` / `door_wood_knock` / `glass_breaking`) は engine 独立に 100% reject 可能 (easy category、 誤検出の心配なし)
- **Parakeet_ja が全体的に最も強い** (14 category で 100%、 laughing / coughing のみ低)、 **WhisperS2T が最も弱い** (7 category で 100% 未満)

### 3.3.2 Layer 3 noisy_speech FRR by mixed noise subtype

**Layer 3 noise rotation の実挙動**: `noise_pool[i % len]` (sequential 選択、 現行 `gen_mixed_noisy_speech.py` の Plan D3) と filename sort 順の相互作用で、 50 speech × 5 SNR = 250 Layer 3 sample は **`breathing` (150) と `car_horn` (100) の 2 subtype のみ** で構成された:

| subtype | n_noisy_speech | reazon i8 | reazon f32 | qwen3asr ja | whispers2t | parakeet_ja |
|---|---|---|---|---|---|---|
| breathing | 150 | 0.053 | 0.053 | 0.073 | **0.027** | 0.080 |
| car_horn | 100 | 0.090 | 0.090 | **0.040** | 0.090 | 0.100 |

**observation**:
- **breathing 混合 speech**: 全 engine で 3-8% FRR、 「呼吸音のある会話」 は production で頻出だが Pareto gate 5% を borderline pass。 WhisperS2T (2.7%) が最良、 Parakeet_ja (8%) が最下位
- **car_horn 混合 speech**: 全 engine で 4-10% FRR、 Qwen3-ASR (4%) が最良、 Parakeet_ja (10%) が最下位

**⚠️ critical caveat (§5.8 で詳述)**: Layer 3 は breathing + car_horn の 2 subtype のみ実測。 他の noise category (clapping / coughing / engine / rain 等) との mix は **未評価**。 現 Pareto gate は breathing/car_horn 基準の一般化。 Phase 2b で noise rotation を uniform stride 化 or random seeded 化して他 subtype も cover する予定。

### 3.4 ESC-50 vs MUSAN 差異

ReazonSpeech int8 @ -0.37 での比較:

- `__none__` bucket (clean speech 449 + Phase 1 synthetic 30 + ESC-50 450 + MUSAN 196 の混合): recall 0.926
- `esc50` bucket (Layer 3 250 mixed 中の ESC-50 由来 noise を使ったもの): FRR 6.8%

MUSAN 単独の subtype (`musan_free-sound` 179、 `musan_sound-bible` 17) は recall 92% / 82% と若干 sound-bible の方が難。 sound-bible は sound effect library で speech-like transient 多め → 予想通り。

---

## 4. Recommendations for Issue #334 PR-4

### 4.1 Final threshold candidate 表

**推奨 PR-4 default 変更** (Pareto gate 適用、 relaxed_B 主に):

| Engine | 現 default | Phase 1 DD | Phase 2 F1 max | **PR-4 推奨** | Δ vs current | Pareto gate |
|---|---|---|---|---|---|---|
| reazonspeech int8/float32 | -0.20 | -0.84 (probe pass 危険) | -0.37 | **`-0.40`** | -0.20 (permissive 化) | relaxed_B (clean 2.9%, SNR≥5 ≤ 4%) |
| qwen3asr ja | -0.30 | -0.97 (probe pass 危険) | -0.42 | **`-0.42`** (F1 max = 最厳可能) | -0.12 (permissive 化) | relaxed_C (clean 2.2%, SNR 10 が 6% で条件付き) |
| whispers2t base | 0.50 | 0.88 | 0.65 | **`0.71`** | +0.21 (conservative 化) | relaxed_B (clean 2.7%, SNR≥5 ≤ 2%) |
| parakeet_ja | 0.005 | 0.001 | 0.006 | **`0.001`** | -0.004 | strict (clean 0.67%、 F1=0.961、 false reject 72% 削減) |

**代替 conservative 選好時** (Pareto gate strict 寄り):

| Engine | Alternative | Gate | F1 |
|---|---|---|---|
| reazonspeech | `-0.43` | relaxed_A (clean 1.78%) | 0.922 |
| whispers2t | `0.76` | relaxed_A (clean 1.78%) | 0.852 |

**Rejected alternative (Parakeet_ja `0.005` 維持案)**:

現 default `0.005` を維持する案は本 report では **不採用**。 理由: `0.001` (Pareto strict pass) と `0.005` (F1 max 0.006 近傍) を比較すると、 F1 差は +0.001 (0.961 vs 0.962) と誤差範囲だが、 clean_frr は **5.6× 差** (0.67% vs 3.79%)、 false reject 数は **11 vs 39** (72% 削減) と user 体感で顕著。 「reject すべき non_speech recall が 97.9% → 94.1% (3.9pt 低下)」 の trade-off は許容範囲 (§4.3 の CHANGELOG Migration guide で明記)。 Layer 4 で実際に non_speech recall 低下が問題になる場合のみ再検討。

### 4.2 PR-4 実装 outline

**⚠️ key 名 verify 済**: `livecap_cli/transcription/confidence_filter.py:162` の実装 dict key は **`"qwen3-asr"`** (ハイフン付き)。 report 執筆当初 `"qwen3asr"` と誤記していたため訂正。 PR-4 実装時にこの key で dict 更新しないと閾値が effect しない (default fallback される)。

`livecap_cli/transcription/confidence_filter.py:FilterConfig` の default 変更 (概略、 詳細は PR-4 で確認):

```python
# Before (現 default)
avg_logprob_thresholds = {
    "reazonspeech": -0.2,
    "qwen3-asr": -0.3,        # ← ハイフン付き key
    "voxtral": -1.0,
}
no_speech_prob_thresholds = {
    "whispers2t": 0.5,
}
token_confidence_mean_thresholds = {
    "parakeet_ja": 0.005,
    "parakeet_en": 0.005,     # 変更なし (EN Phase 別)
    "canary": 0.005,          # 変更なし
}

# After (Phase 2 report 反映)
avg_logprob_thresholds = {
    "reazonspeech": -0.40,    # -0.20 → -0.40 (Phase 2 Pareto relaxed_B、 F1=0.929)
    "qwen3-asr": -0.42,       # -0.30 → -0.42 (Phase 2 F1 max = 最厳可能 gate、 F1=0.955、
                              # SNR 10 dB 6% borderline は Layer 4 で再確認)
    "voxtral": -1.0,          # 変更なし (未計測)
}
no_speech_prob_thresholds = {
    "whispers2t": 0.71,       # 0.50 → 0.71 (Phase 2 Pareto relaxed_B、 F1=0.901、
                              # Whisper 公式 0.6 近傍)
}
token_confidence_mean_thresholds = {
    "parakeet_ja": 0.001,     # 0.005 → 0.001 (Phase 2 Pareto strict、 F1=0.961、
                              # false reject 39 → 11 (72%削減)、
                              # 引き換えに non_speech recall 97.9% → 94.1% (-3.9pt))
}
```

### 4.3 PR-4 実装チェックリスト (単一 PR 前提、 user レビュー反映)

**必須変更**:

- [ ] `livecap_cli/transcription/confidence_filter.py:FilterConfig` default 4 threshold 更新 (`reazonspeech` / `qwen3-asr` / `whispers2t` / `parakeet_ja`)
- [ ] `docs/reference/api.md` の閾値表を Phase 2 値に更新
- [ ] `docs/reference/cli.md` の閾値表を Phase 2 値に更新
- [ ] `CHANGELOG.md` `[Unreleased] → ### Changed` section に **Before / After / Migration** guide を追加:
  ```
  # Before
  ReazonSpeech: -0.20, Qwen3-ASR: -0.30, WhisperS2T: 0.50, Parakeet_ja: 0.005
  # After
  ReazonSpeech: -0.40, Qwen3-ASR: -0.42, WhisperS2T: 0.71, Parakeet_ja: 0.001
  # Migration
  現 default を維持したい場合は FilterConfig(avg_logprob_thresholds={"reazonspeech": -0.20, ...}) を明示指定。
  Parakeet_ja は non_speech recall を意図的に下げて false reject を大幅削減した trade-off (詳細は Phase 2 report §2.4)。
  ```
- [ ] tests で新 default を pin (`tests/transcription/test_confidence_filter.py` 相当、 現状 test あれば更新)
- [ ] Phase 2 report ([`calibration-japan-engines-phase2-2026-07.md`](calibration-japan-engines-phase2-2026-07.md)) を PR body で reference

**推奨**:

- [ ] full calibration suite regression 実行、 既存 test 全 retain
- [ ] `livecap-cli info` output に新閾値が反映されることを smoke verify
- [ ] livecap-gui 側の release note に「confidence_filter 閾値変更」を記載

### 4.4 Migration strategy

1. **Phase A** (本 report merge 後、 別 PR): PR-4 で `FilterConfig` default 変更 + Migration guide 追加
2. **Phase B**: Release note で挙動変化を明示、 livecap-gui 側 dependency update
3. **Phase C** (Layer 4 完了後): production observe log で verify、 特に Qwen3-ASR ja の SNR 10 dB borderline を再検討、 必要なら微調整

### 4.5 Downstream (Issue #334 の他 PR との関係)

- **PR-2** (元 plan の ReazonSpeech noisy_speech corpus 整備) → 本 Phase 2 で **代替済**、 close 可
- **PR-3** (元 plan の Qwen3-ASR language 分布測定) → 同、 close 可
- **PR-4** (default threshold 変更) → 本 report が **直接 input**、 上記 §4.1 の推奨 threshold を採用予定
- **Epic 1** (Refuse text pattern audit) → Phase 4 (production observe log) の後で再検討
- **Epic 2** (Multi-signal harness) → v3.2+ research

---

## 5. Caveats & open questions

### 5.1 Corpus scope (JA Little Prince Ch.1 のみ)

Phase 1 と同、 corpus は Little Prince Chapter 1 (~21 min 朗読) と ESC-50/MUSAN 環境音のみ。 Phonetic / lexical coverage に偏りあり、 Chapter 多様化は Phase 3+ で対応予定。

### 5.2 Layer 4 (production observe replay) 未実施

本 report の Pareto gate は synthetic augmentation (ESC-50 / MUSAN + Layer 3 mix) ベース。 実 production 環境 (家庭 mic / café 等の実 background noise) との FRR 一致は Layer 4 で verify 必要。 PR-4 merge 後 monitoring 期間を設ける。

### 5.3 SNR -5 dB / 0 dB は production scope 外

Pareto gate は SNR ≥ 5 dB のみを対象。 SNR -5 dB は「工事現場」相当で ASR 自体が困難、 SNR 0 dB は「賑やかな街」相当でも user 期待値が低い。 本 report では **SNR ≥ 5 dB を production 主要 scope** と定義。

### 5.4 breathing / laughing の noise source 妥当性 (metric 分離アプローチ)

全 engine で純 non_speech recall 63-73% (`breathing` / `laughing`) は他 category より **20-30 pt 低い**。 これらは音響的に speech に極近く、 「reject すべき non_speech」 と定義するかが微妙。 **本 report の判断**: Layer 3 noise source からの除外 (元 A 案) は Phase 2b で保留、 代わりに **metric 分離** で対応する:

- **理由 1**: `breathing/laughing` の低 recall は 「non_speech として reject すべきか」 という **label 定義** の問題であり、 これを解決するには engine の閾値ではなく label 側で対応 (borderline sample を noisy_speech へ再分類 or 別 label 導入)
- **理由 2**: Layer 3 除外は 「speech + breathing/laughing noise を noisy_speech として保持すべきか」 という **別問題**。 production では呼吸・笑い声は実際に mic に混入するため、 完全除外すると評価が unrealistic に容易化する
- **理由 3**: PR-4 の主判定は **breathing/laughing 含む現行 corpus** で行い、 supplement として excluded case の改善幅を Phase 2b で計測

**Phase 2b で実施予定** (別 PR、 PR-4 blocker としない):
1. **`breathing/laughing` を除外した subset で全 5 engine 再 sweep** → 純 non_speech recall と Pareto profile の改善幅を quantify
2. Included vs excluded 両 report を並置公開 (PR-4 の主判定は included ベース、 excluded は sensitivity analysis)
3. 「笑い声・呼吸音を transcribe すべきか」 の user 期待値は別 issue で扱う (Epic 1 の refuse text pattern audit と同 track)

### 5.5 SNR grid resolution (5 値のみ)

`-5 / 0 / 5 / 10 / 20 dB` の 5 bucket のみ、 中間 SNR (2.5 / 7.5 / 15 dB 等) は評価なし。 Phase 3 では grid を細分化検討 (計算 cost trade-off あり)。

### 5.6 EN 対応は別 Phase

Qwen3-ASR EN / Parakeet en / Canary en の calibration は Phase 4 以降で対応。 本 Phase 2 は JA scope に集中。

### 5.7 Layer 3 noise diversity limit (breathing + car_horn のみ実測)

`gen_mixed_noisy_speech.py` の Plan D3 (`noise_pool[i % len(noise_pool)]` sequential rotation) と `select_noise_pool` の filename sort 順の相互作用で、 50 speech × 5 SNR = 250 Layer 3 sample は **`breathing` (150、 60%) と `car_horn` (100、 40%) の 2 subtype のみ** に集中した。 他の Layer 2 noise category (`clapping` / `coughing` / `engine` / `rain` / `mouse_click` 等) との mix は **未評価**。

**影響**:
- 本 report の per-SNR FRR は 「speech + breathing/car_horn mix」 に対するもので、 他 noise type との mix に対する generalization は未検証
- Pareto gate `noisy_frr(SNR≥5) ≤ 5%` は breathing/car_horn 基準の一般化
- Parakeet_ja の SNR 敏感度が特に高い (§2.4.2) のは、 breathing の speech-adjacent 性が悪影響したか、 他 noise でも同様かは要検証

**Phase 2b で fix (別 PR)**:
1. `gen_mixed_noisy_speech.py` の noise rotation を **uniform stride** (`noise_pool[i * len // n_samples]`) or **random seeded** に変更 → 全 Layer 2 category を偏りなく Layer 3 に反映
2. 再 sweep で cross-noise-type SNR curve を quantify、 現 PR-4 threshold の妥当性を validate
3. 必要なら PR-4 candidate threshold を微調整 (Phase 2b の別 PR)

### 5.8 F1 max criterion は残す (Pareto gate の baseline として)

Pareto gate 採用でも F1 max criterion は sweep report に残す。 これは:
- Pareto gate が satisfy 不能な engine (Qwen3-ASR SNR 10 borderline) で fallback として機能
- Phase 1 report との比較で「Pareto がどれくらい F1 を sacrifice するか」を quantify (例: ReazonSpeech -0.37 F1=0.934 → -0.40 F1=0.929、 差 -0.005)
- 将来 corpus 拡張で全 engine が strict Pareto pass になった場合、 F1 と Pareto の乖離が縮小する見込み

---

## 6. Related

### PR 依存関係

- 上流 PR #341 (PR-γ kana metric、 merged `207e889`)
- 上流 PR #342 (Phase 1 report、 merged `6199194`) — 本 report が直接更新
- 上流 PR #343 (Layer 2 ESC-50 / MUSAN augment、 merged `9e1c634`)
- 上流 PR #344 (Layer 3 SNR mixed、 merged `18e9bf8`)
- 上流 PR #345 (Phase 6a sweep breakdown、 merged `7d07386`)
- 下流 PR (未着手): PR-4 threshold 変更 — 本 report §4.1 を input

### Issue

- 親 Issue: [#338](https://github.com/Mega-Gorilla/livecap-cli/issues/338) (active calibration harness、 Phase 2 完了で close 候補)
- 下流 Issue: [#334](https://github.com/Mega-Gorilla/livecap-cli/issues/334) (audit、 PR-4 で threshold 変更後に close 候補)
- Audit 契機: [livecap-gui #362](https://github.com/Mega-Gorilla/livecap-gui/issues/362) / [#366](https://github.com/Mega-Gorilla/livecap-gui/issues/366)

### Reference docs

- Phase 1 report: [`calibration-japan-engines-2026-07.md`](calibration-japan-engines-2026-07.md)
- Contributor guide (PR-6 の output、 新 engine 追加時参照): [`../contributor/adding-an-engine.md`](../contributor/adding-an-engine.md)
- Harness README: [`../../benchmarks/confidence_calibration/README.md`](../../benchmarks/confidence_calibration/README.md)

### Raw data (user verification 用)

音源の妥当性 (特に Layer 3 の SNR-mixed noisy_speech) を user が耳検 / 波形検で確認可能なパス:

| Layer | 音源 dir (`.tmp/calibration_corpus_full/`) | N | filename example |
|---|---|---|---|
| 1 clean speech | `ja_clean/` | 449 | `segment_0000.wav` ... `segment_0448.wav` |
| 1 synthetic non_speech | `ja_non_speech/` | 30 | `noise_000_silence_0.5s.wav`、 `noise_010_white_noise_1.5s.wav` |
| 2 ESC-50 non_speech | `ja_non_speech_esc50/` | 450 | `breathing_1-100032-A-23_chunk0.wav`、 `clapping_1-100032-A-22_chunk1.wav` |
| 2 MUSAN noise | `ja_non_speech_musan/` | 196 | `noise-free-sound-0000_chunk0.wav`、 `noise-sound-bible-0004_chunk2.wav` |
| **3 SNR-mixed noisy_speech** | `ja_noisy_speech/` | **250** | `segment_0000_snr-5dB_breathing.wav` (SNR -5 dB)、 `segment_0000_snr10dB_breathing.wav` (SNR 10 dB) |

**Layer 3 の filename pattern**: `{speech_stem}_snr{db_str}dB_{noise_subtype}.wav`
- 例: `segment_0000_snr10dB_breathing.wav` = clean 音源 `segment_0000.wav` に breathing noise を SNR 10 dB で混合
- 現状の noise 分布: breathing 150 (60%) + car_horn 100 (40%) — §5.8 の noise diversity 課題

**推奨検証手順** (Layer 3 音源の耳検):
```bash
# SNR 20 dB (静オフィス相当、 clean にほぼ近い) を再生
soundfile-play .tmp/calibration_corpus_full/ja_noisy_speech/segment_0000_snr20dB_breathing.wav

# SNR 0 dB (賑やかな公共空間相当、 noise と speech 同レベル) を再生
soundfile-play .tmp/calibration_corpus_full/ja_noisy_speech/segment_0000_snr0dB_breathing.wav

# SNR -5 dB (工事現場相当、 noise が優勢) を再生
soundfile-play .tmp/calibration_corpus_full/ja_noisy_speech/segment_0000_snr-5dB_breathing.wav
```

同一 speech (`segment_0000`) が 5 SNR で並ぶため、 SNR の 5 dB 刻みで背景 noise がどのように変わるか耳で確認可能。 元 clean 音源 (`segment_0000.wav`) と比較すれば mixing 品質を判定できる。

### Raw sweep report

- 5 engine × 全 corpus の sweep report JSON: `.tmp/phase2_reports/*.json` (git 外、 dev only)
- Intermediate summary (本 report 執筆前の analysis 素材): `.tmp/phase2_reports/summary.md`
- Calibration corpus manifest (git 外、 dev only): `.tmp/calibration_corpus_full/manifest.jsonl` (1375 entries)

## まとめ (Phase 6b 完了 marker)

| 観点 | 詳細 |
|---|---|
| **Phase 6b (本 report 執筆) 完了** | 5 engine の Phase 2 Pareto gate 適用 threshold を確定、 PR-4 直接 input を提示 |
| **Phase 1 caveat 解消状況** | ✅ synthetic non_speech 依存、 ✅ noisy_speech 未評価 (Layer 3 で 5 SNR bucket)、 ⏳ Layer 4 production observe は継続課題 |
| **PR-4 で採用する threshold** | reazon `-0.40` / qwen3-asr `-0.42` / whispers2t `0.71` / parakeet_ja `0.001` |
| **Pareto gate satisfy 度** | WhisperS2T / Parakeet_ja は strict、 ReazonSpeech は relaxed_B、 Qwen3-ASR は relaxed_C 部分 |
| **量子化影響** | ゼロ (ReazonSpeech int8/float32 で完全同一)、 Finding F8 一般則の実測 confirm |
| **次の action** | Issue #334 PR-4 の実装 (`FilterConfig` default 更新 + CHANGELOG Migration guide)、 Layer 4 は別 Phase |
