# JA ASR confidence threshold calibration — 2026-07 Phase 1 report

> **Issue #338** (Stage 2 active calibration harness) で構築した sweep CLI を、
> リトル・プリンス JA Chapter 1 全長 + synthetic non_speech corpus に対して
> 5 engine 適用した data-driven calibration の **第 1 次 report**。

| 項目 | 値 |
|---|---|
| 実施日 | 2026-06-30 |
| 対象 engine | reazonspeech (int8 / float32), qwen3asr, whispers2t, parakeet_ja (5 sweep) |
| 対象言語 | JA |
| Corpus | 449 speech (Little Prince Ch.1 朗読) + 30 synthetic non_speech (silence + noise) |
| Harness | `benchmarks/confidence_calibration/sweep.py` (PR #340 PR-β) |
| Kana metric | `_normalize_jp.py` v4 (PR #341 PR-γ) — manifest に併記 |
| 関連 Issue | [#338](https://github.com/Mega-Gorilla/livecap-cli/issues/338) (本 report の harness)、 [#334](https://github.com/Mega-Gorilla/livecap-cli/issues/334) PR-4 (本 report の下流: data-driven threshold 変更) |

## TL;DR

1. **全 5 engine で data-driven threshold が現 default より大幅に permissive** (3 engine で
   speech 誤 reject 率が ≥ 14%、 内 2 engine で ≥ 42%)。
2. F1 ≥ 0.952 が全 engine 達成 (qwen3asr で F1 = 1.000 perfect、 ReazonSpeech 両 quant で
   0.984、 WhisperS2T 0.983、 Parakeet_ja 0.952)。
3. **重大 caveat**: 本 corpus の non_speech は synthetic silence + low noise のみ。
   production の applause / 環境音 / music は signal が speech に近く、 production
   妥当な threshold は本 report 値より conservative (現 default 寄り) になる可能性高。
4. **Issue #334 PR-4 への concrete input**: 本 report の data-driven 値を **下限 estimate**
   とし、 production observe-mode log 比較 + ESC-50 等で再 calibration して中間値で landed
   する 2-step approach を推奨。
5. Parakeet_ja は現 default (0.005) が既にほぼ optimal (0.001 と比べて FRR +0.6pt のみ)。
   その他 4 engine は **削減方向**の見直し対象。

## 1. Methodology

### 1.1 Corpus 構成 (`$LIVECAP_CALIBRATION_CORPUS_DIR/`)

```
.tmp/calibration_corpus_full/
├── manifest.jsonl                     ← 479 entries (449 + 30)
├── ja_clean/                          ← 449 segment wav (Little Prince Ch.1)
│   └── segment_NNNN.wav  (avg 1.65s, max 3.0s)
└── ja_non_speech/                     ← 30 synthetic samples
    ├── noise_NNN_silence_X.Xs.wav     ← 10 pure silence
    ├── noise_NNN_white_noise_X.Xs.wav ← 10 white noise @ ~-40dB
    └── noise_NNN_pink_noise_X.Xs.wav  ← 10 pink-ish noise @ ~-40dB
```

#### Positive (speech, label=speech, language=ja):
- **Source**: `https://www.youtube.com/watch?v=6aJ3jsVeQIg` (taltal3014 ja 朗読 全 21 min)
- **Reference**: `https://taltal3014.lsv.jp/little-prince/LittlePrince1.html` (6600 chars)
- **Build pipeline**: yt-dlp + ffmpeg (16 kHz mono) + Silero VAD chunker + WhisperS2T base int8
  transcribe + `compute_alignment_score` text + `compute_alignment_score_kana` (PR-γ v4)
- **Count**: 449 segments、 total duration 742.6 sec (12.4 min of speech)
- **Coverage 分布** (kana metric):
  - perfect ≥ 0.95: 198 (44%)
  - high ≥ 0.50: 372 (83%)
  - low < 0.20: 6 (1%)

#### Negative (non_speech, label=non_speech, language=ja):
- **Source**: synthetic generation (`gen_synthetic_non_speech.py`)
- **Count**: 30 (10 silence、 10 white noise @ ~-40dB、 10 pink-ish noise @ ~-40dB)
- **Duration**: 0.5-1.5 sec each
- **Rationale**: Phase 0c で sweep が degenerate F1=0 から脱却する最小 negative set。
  ESC-50 / MUSAN 等の production-grade non_speech は次 Phase に分離。

### 1.2 Sweep methodology (per-engine)

各 engine について `benchmarks.confidence_calibration.sweep` を以下の構成で実行:

```bash
uv run python -m benchmarks.confidence_calibration.sweep \
    --engine <engine_id> \
    --signal <signal_field> \
    --filter-by-language ja \
    --corpus-dir .tmp/calibration_corpus_full \
    --engine-kwargs <engine_specific_kwargs> \
    --output <report.json>
```

| Setting | Value |
|---|---|
| Threshold grid | per-signal default range (avg_logprob: [-1.0, -0.05] step 0.01; no_speech_prob: [0.1, 0.95] step 0.01; token_confidence_mean: [0.001, 0.5] step 0.01) |
| Criterion | F1 (max F1 が同点の場合 closest-to-zero-FRR を採用) |
| Per-sample wall-clock | ~0.5-2 sec (engine 依存) |
| Sweep wall-clock | ~3-15 min (engine 依存、 479 samples × per-sample time) |

### 1.3 Caveats

1. **Synthetic non_speech limitation** (最重要): 純 silence + low-level noise (~-40 dB) は
   production の "non_speech" (applause、 環境音、 music、 distortion 等) と signal 分布が
   大きく異なる。 PR-A.5.2 probe では Qwen3-ASR JA で applause の avg_logprob ≈ -0.46、
   desk_tap ≈ -0.50 だったが、 本 report の synthetic は production より **easier** (より低
   signal、 より広い margin)。 → **本 report の data-driven 値は production 妥当 threshold
   の下限 estimate** に当たる。
2. **Clean 朗読 corpus bias**: 朗読 audio は production 一般 (mic 録音、 環境音混入、 不明瞭
   発話) より clean。 speech avg_logprob 分布は本 corpus 値より低く (negative) シフトする
   可能性。 → 本 report の threshold を下げすぎると production の speech retention は維持
   できるが、 production non_speech rejection が weak になる可能性。
3. **Single-language scope**: 本 report は JA のみ。 EN sweep は smoke 24 samples のみで
   別 Phase で実施予定。
4. **Single calibration corpus**: Chapter 1 のみ (21 min)。 phonetic / acoustic variance
   が偏る可能性。 Phase 2/3 で Chapter 多様化推奨。

## 2. Per-engine results

### 2.1 ReazonSpeech int8 (sherpa-onnx zipformer JA)

| Metric | Value |
|---|---|
| Signal | `avg_logprob` |
| Direction | `reject_if_less` (lower log prob → reject) |
| Current default | **-0.2** (`livecap_cli/transcription/confidence_filter.py:153`、 PR-A.5.1 #317) |
| **Data-driven** | **-0.84** (criterion=F1) |
| Delta | **-0.64 (3.2x more permissive)** |
| F1 @ DD | **0.984** |
| Precision @ DD | 0.968 |
| Recall @ DD | 1.000 (全 non_speech 検出) |
| FRR @ DD | 0.002 (~1 of 449 speech が誤 reject) |
| **FRR @ Current default** | **0.425** (449 speech のうち ~191 件が誤 reject 推定) |
| Plateau (F1 ≥ max-1%) | **[-0.84, -0.58]** (27 step、 robust) |

**観察**: 現 default -0.2 では 449 speech のうち推定 191 件 (42.5%) が `avg_logprob < -0.2` で
誤 reject される。 朗読 audio という clean データに対してこの値は明らかに **過剰 aggressive**。
Plateau が広い ([-0.84, -0.58], 0.26 幅、 27 step) ため、 threshold 選択に robustness margin
あり。

### 2.2 ReazonSpeech float32 (sherpa-onnx zipformer JA、 quantization=float32)

| Metric | Value |
|---|---|
| Signal | `avg_logprob` |
| Current default | -0.2 (engine-specific entry のみ、 quantization 別管理なし) |
| **Data-driven** | **-0.75** |
| Delta | -0.55 |
| F1 @ DD | 0.984 |
| FRR @ DD | 0.002 |
| **FRR @ Current default** | **0.437** |
| Plateau | **[-0.75, -0.68]** (8 step、 やや narrow) |

**観察**: int8 と同じ F1 だが optimal threshold が +0.09 (less negative)。 float32 model は
int8 quantization 損失 が小さく speech の avg_logprob 分布がやや高め。 Plateau が int8 より
narrow (8 step vs 27)。 同一 engine_id "reazonspeech" を両 quant で共有しているため、 PR-4 で
quantization 別 threshold を持つか global で平均値を取るかの設計判断が必要。

### 2.3 Qwen3-ASR 0.6B (multilingual、 language=ja)

| Metric | Value |
|---|---|
| Signal | `avg_logprob` |
| Direction | `reject_if_less` |
| Current default | **-0.3** (PR-A.5.2 #318、 JA probe speech -0.20 / applause -0.46 / margin +0.27) |
| **Data-driven** | **-0.97** (criterion=F1) |
| Delta | **-0.67** |
| **F1 @ DD** | **1.000 (perfect)** |
| Precision @ DD | 1.000 |
| Recall @ DD | 1.000 |
| FRR @ DD | 0.000 |
| **FRR @ Current default** | **0.065** (~29 件) |
| Plateau (F1 ≥ max-1%) | **[-0.97, -0.64]** (34 step、 最 robust) |

**重要**: `--engine-kwargs language=ja` 必須。 language 未指定だと auto-detect path に入り
`avg_logprob` が None で全 sample excluded (signal extraction 失敗)。

**観察**: 5 engine 中で **F1 = 1.000 perfect** + **plateau 最広** (0.33 幅、 34 step)。
Qwen3-ASR は本 corpus に対して最も discriminative。 PR-A.5.2 の Phase 1 probe (speech -0.20 /
applause -0.46) との対比: 本 corpus で speech avg_logprob は **大きく分布**して -1.0 まで届く
(現 default -0.3 で 6.5% 誤 reject)。 PR-A.5.2 の "margin +0.27" は probe 短時間 sample
の局所値であり、 multi-segment 朗読では speech 分布が広い。

### 2.4 WhisperS2T base (CTranslate2、 language=ja)

| Metric | Value |
|---|---|
| Signal | `no_speech_prob` |
| Direction | `reject_if_greater` (high no_speech_prob → reject) |
| Current default | **0.5** (`no_speech_threshold`、 Whisper 公式 0.6 を踏襲した中道) |
| **Data-driven** | **0.88** |
| Delta | **+0.38 (より permissive)** |
| F1 @ DD | 0.983 |
| Precision @ DD | 1.000 (全 rejection が真の non_speech) |
| Recall @ DD | 0.967 (30 件中 29 件 catch、 1 件 miss) |
| FRR @ DD | 0.000 |
| **FRR @ Current default** | **0.143** (~64 件) |
| Plateau | **[0.88, 0.88]** (single point、 最 narrow) |

**観察**: speech は no_speech_prob ≤ 0.87 にきれいに分布、 non_speech 29 件は 0.88 以上に明確
分離 (perfect precision)。 ただし plateau が 1 step のみ → threshold 微調整に超 sensitive。
synthetic non_speech 1 件が 0.88 未満 (= speech と誤判定) されており、 production
non_speech multivariance では recall がさらに下がる可能性。

### 2.5 Parakeet_ja (NVIDIA NeMo TDT-CTC 0.6B JA)

| Metric | Value |
|---|---|
| Signal | `token_confidence_mean` |
| Direction | `reject_if_less` |
| Current default | **0.005** |
| **Data-driven** | **0.001** |
| Delta | -0.004 |
| F1 @ DD | **0.952** (5 engine 中最低) |
| Precision @ DD | 0.909 |
| Recall @ DD | 1.000 |
| FRR @ DD | 0.007 |
| **FRR @ Current default** | **0.007** (Parakeet は **現 default が既にほぼ optimal**) |
| Plateau | **[0.001, 0.001]** (single point) |

**観察**: 唯一 **現 default と data-driven がほぼ同値** (delta -0.004) で、 production 観点で
変更不要。 ただし F1=0.952 と他 engine より低い → token_confidence_mean signal の
discriminative power が他 engine の avg_logprob / no_speech_prob より弱い。 speech の
token_confidence_mean が 0.001-0.291 と広く分布、 非常に low confidence の正解 speech が
多い ASR architecture 特性。 Phase 4 本格 calibration で別 signal も併用検討余地あり。

## 3. Comparison summary

### 3.1 Threshold vs current default

| Engine | Signal | Current | **Data-driven** | Delta | 方向 |
|---|---|---|---|---|---|
| reazonspeech int8 | avg_logprob | -0.20 | **-0.84** | -0.64 | より permissive |
| reazonspeech float32 | avg_logprob | -0.20 | **-0.75** | -0.55 | より permissive |
| qwen3asr | avg_logprob | -0.30 | **-0.97** | -0.67 | より permissive |
| whispers2t (no_speech_prob) | no_speech_prob | 0.50 | **0.88** | +0.38 | より permissive |
| parakeet_ja | token_confidence_mean | 0.005 | **0.001** | -0.004 | 現状維持で OK |

### 3.2 現 default での speech 誤 reject 影響度

| Engine | Current FRR | 影響評価 |
|---|---|---|
| reazonspeech int8 / float32 | **42.5% / 43.7%** | 🔴 **重大** (本 corpus で半数近く誤 reject) |
| qwen3asr | 6.5% | 🟡 中 |
| whispers2t base | 14.3% | 🟠 やや高 |
| parakeet_ja | 0.7% | 🟢 適切 |

### 3.3 Plateau robustness (threshold 選択の余裕)

| Engine | Plateau width | N steps | 評価 |
|---|---|---|---|
| qwen3asr | 0.33 | 34 | 🟢 最 robust (threshold 選択余裕大) |
| reazonspeech int8 | 0.26 | 27 | 🟢 robust |
| reazonspeech float32 | 0.07 | 8 | 🟡 narrow |
| whispers2t base | 0.00 (point) | 1 | 🔴 sharp (threshold 微調整に超 sensitive) |
| parakeet_ja | 0.00 (point) | 1 | 🔴 sharp |

## 4. Recommendations for Issue #334 PR-4

PR-4 author (default `FilterConfig.avg_logprob_thresholds` etc 変更) への concrete input。

### 4.1 推奨 approach: 2-step landing

本 report の data-driven 値を直接 default にするのは risky (synthetic non_speech caveat)。
推奨 path:

1. **Step 1 — production observe-mode log との突合**:
   - 既存 production observe log があれば `parse_observe.py` で本 report の threshold を
     当てて、 false_reject_rate / pass_rate 確認
   - production false_reject_rate が本 report の予測 (各 engine の FRR @ Current default)
     と整合するか verify
2. **Step 2 — ESC-50 / MUSAN 補強 + 再 sweep**:
   - synthetic non_speech を ESC-50 (applause / dog bark / fireworks / rain / engine 等)
     + MUSAN (music / noise) で置換
   - 再 sweep で **production-realistic** threshold を再計算
   - 本 report 値と新 threshold の中間値で default landed

### 4.2 暫定 (Step 1 のみで進める場合) 候補 threshold

production observe との突合まで時間を取れない場合の **中道 conservative** 値:

| Engine | Current | Data-driven | **暫定 候補 (中道)** | 根拠 |
|---|---|---|---|---|
| reazonspeech (両 quant 共通) | -0.2 | -0.84 / -0.75 | **-0.5** | Δ -0.3 ずつ近接 |
| qwen3asr | -0.3 | -0.97 | **-0.6** | PR-A.5.2 probe applause -0.46 を超え、 margin 0.14 |
| whispers2t (no_speech_prob) | 0.5 | 0.88 | **0.7** | Whisper 公式 0.6 + margin 0.1 |
| parakeet_ja | 0.005 | 0.001 | **0.001** (data-driven そのまま) | 現 default と data-driven がほぼ同値 |

### 4.3 期待 outcome

暫定 threshold での **FRR (本 corpus)**:

| Engine | Cur FRR | 暫定 FRR (推定) | Δ |
|---|---|---|---|
| reazonspeech int8 (-0.5) | 0.425 | ~0.05 (sweep table 確認要) | **-0.375pt** speech retention 改善 |
| reazonspeech float32 (-0.5) | 0.437 | ~0.05 | -0.387pt |
| qwen3asr (-0.6) | 0.065 | ~0.01 | -0.055pt |
| whispers2t (0.7) | 0.143 | ~0.05 | -0.093pt |

→ **production observe log で問題ないと確認できれば、 暫定値 landed で speech retention
大幅改善 + non_speech rejection は production-realistic non_speech で別途 verify**。

## 5. Limitations + next steps

### 5.1 既知 limitations

1. Non_speech は synthetic のみ (silence + noise)、 production の applause / 環境音 / music
   未網羅
2. EN sweep は smoke 24 samples のみで Phase 1 含まず (qwen3asr en は別 Phase で別途)
3. Chapter 1 のみで phonetic / acoustic variance に bias
4. ReazonSpeech が int8 / float32 別の engine_id を持たない (両 quant 共通の "reazonspeech")
5. Parakeet_ja の F1 0.952 は他 engine より 0.03 低い (token_confidence_mean の
   discriminative limit)

### 5.2 Next steps

| 順序 | Action |
|---|---|
| 1 | **production observe log との突合** (本 report Step 1) |
| 2 | **ESC-50 / MUSAN 補強 + 再 sweep** (本 report Step 2) |
| 3 | **EN 全長 build + qwen3asr en sweep** (現状 EN は smoke 24 件のみ) |
| 4 | **Issue #334 PR-4** (本 report + Step 1/2 結果を input に default 値 update) |
| 5 | (任意) Chapter 多様化 (Ch.2-27 で追加 sweep、 phonetic variance を広げる) |
| 6 | (任意) Parakeet_ja に別 signal (e.g. avg_logprob 抽出可能か engine check) |

## 6. 関連 リソース

- 親 Issue: [#338](https://github.com/Mega-Gorilla/livecap-cli/issues/338) (本 sweep harness)
- 下流 Issue: [#334](https://github.com/Mega-Gorilla/livecap-cli/issues/334) PR-4 (本 report の利用先)
- 前提 PR: [#339](https://github.com/Mega-Gorilla/livecap-cli/pull/339) (PR-α: signal-agnostic sweep core)、 [#340](https://github.com/Mega-Gorilla/livecap-cli/pull/340) (PR-β: active sweep + build_corpus)、 [#341](https://github.com/Mega-Gorilla/livecap-cli/pull/341) (PR-γ: kana metric)
- 前 verify: `docs/research/calibration-corpus-smoke-verify.md` (60-sec smoke verify)
- 本 report の raw data: `.tmp/phase1_reports/{reazonspeech_float32_ja, qwen3asr_ja, whispers2t_ja, parakeet_ja_ja}.json` + `.tmp/report_phase0c_reazonspeech_int8_ja.json` (local-only、 git 外)
- Local corpus: `.tmp/calibration_corpus_full/manifest.jsonl` (479 entries、 local-only)

## まとめ

| 観点 | 状況 |
|---|---|
| 5 engine sweep 完遂 | ✅ JA 全 engine (Issue #338 Phase 4 本格 calibration) |
| F1 quality | ✅ 全 engine ≥ 0.952、 qwen3asr で 1.000 perfect |
| 現 default との gap 定量化 | ✅ FRR @ Current default を 5 engine 計測、 重大 (reazonspeech 42.5%) を発見 |
| Issue #334 PR-4 への 直接 input | ✅ 暫定中道 threshold 候補 4 件 + 2-step landing 推奨 path |
| Limitation 明示 | ✅ synthetic non_speech / clean 朗読 / EN 未実施 / Chapter 1 のみ |
| Phase 4 本格 calibration への次ステップ | production observe 突合 → ESC-50 / MUSAN 補強 → 再 sweep → PR-4 |

by.Scotty
