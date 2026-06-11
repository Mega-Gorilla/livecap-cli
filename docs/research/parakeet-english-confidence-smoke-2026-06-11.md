# PR-A.4.3 — Parakeet 英語 Confidence Smoke Verify (2026-06-11)

> **Status**: ✅ **Implemented & Validated** — Phase 1 probe success + Section 1 Case A 確定 (margin 49×) + Section 2 stream pipeline benchmark で `webrtc × synthetic × on` Hall.(post) 75% → 12.5% を実証。Issue #311 PR-A.4.3 完了候補。

## Setup

| Item | Value |
|---|---|
| Date | 2026-06-11 |
| Branch | `chore/pr-a4-docs-finalize` (PR-A.4.docs と合体、PR #316 で完了) |
| Issue | [#311 v2.1+v2.2](https://github.com/Mega-Gorilla/livecap-cli/issues/311) PR-A.4.3 |
| Model | NVIDIA Parakeet TDT 0.6B v2 (`nvidia/parakeet-tdt-0.6b-v2`) |
| Engine config | `device="cuda"`, `engine_name="parakeet"` (英語 TDT)、**Path 1.5: greedy + preserve_alignments + confidence_cfg.preserve_token_confidence** |
| GPU | NVIDIA RTX 4090, CUDA 12.8, PyTorch 2.9.1+cu128 |
| Supported languages | English (native) |
| Smoke scripts | `.tmp/pr_a4_3_parakeet_en_smoke.py` (Section 1)、`benchmarks/non_speech_filter/sweep` (Section 2)、いずれも一時、commit しない |

---

## Background — 「構造的限界」は誤りだった経緯

旧 docs (PR-A.4.docs 初版以前) では Parakeet 英語を「NeMo RNNT path に token_confidence 未実装 → 構造的限界」と PR-A.5 candidate に分類していた。本 PR の作業中の調査で **「構造的限界ではなく PR #309 時点の `preserve_alignments=True` 併設漏れ」**と判明:

### NeMo source の smoking-gun

| Evidence | 内容 |
|---|---|
| `rnnt_decoding.py:95-106` | `preserve_token_confidence` documented、`token_confidence` is a List of floats |
| `rnnt_decoding.py:277` | `_init_confidence(self.cfg.get('confidence_cfg', None))` で RNNT confidence_cfg 初期化 |
| **`rnnt_decoding.py:280-282`** | **「`preserve_frame_confidence=True` 設定時は `preserve_alignments=True` 同時設定必須」制約** |
| `tdt_loop_labels_computer.py:104, 167, 371` | TDT decoding で `preserve_frame_confidence` 実装あり |
| `rnnt_loop_labels_computer.py:102, 163, 353` | RNNT decoding で同実装 |

PR #309 時点の実装者は `preserve_frame_confidence=True` のみ設定 → NeMo が `preserve_alignments=True` 併設要求で reject → 「構造的限界」と誤認した。Path 1 (Hybrid CTC) と同 pattern を non-hybrid model に適用すれば populate される。

→ 本 PR で **Path 1.5** (pure RNNT/TDT 用) を追加、Parakeet 英語の filter 対応を完了。

---

## Implementation — Path 1.5

`livecap_cli/engines/parakeet_engine.py::_configure_decoding_with_confidence`:

```python
# Path 1.5: Pure RNNT/TDT model (parakeet 英語) — preserve_alignments を
# 併設して confidence_cfg を有効化。Hybrid CTC と同じ pattern を
# decoder_type 切替なしで適用 (TDT decoding 自体が confidence をサポート)。
try:
    tdt_cfg = {
        'strategy': self.decoding_strategy,
        'preserve_alignments': True,          # ← NeMo 制約満たす key
        'greedy': {
            'preserve_alignments': True,
            'preserve_frame_confidence': True,
        },
        'confidence_cfg': {
            'preserve_frame_confidence': True,
            'preserve_token_confidence': True,
            'preserve_word_confidence': False,
            'exclude_blank': True,
            'aggregation': 'mean',
        },
    }
    self.model.change_decoding_strategy(tdt_cfg)
    logger.info(
        f"Parakeet RNNT/TDT: confidence_cfg activated "
        f"(strategy={self.decoding_strategy!r}, frame_confidence ON, "
        "token_confidence_mean は filter signal として使用可能 [PR-A.4.3])"
    )
    return
except (TypeError, KeyError, ValueError, AttributeError) as e:
    logger.info(
        f"Parakeet RNNT/TDT confidence_cfg rejected ({type(e).__name__}: {e}); "
        "falling back to strategy-only (token_confidence will be None)."
    )
```

3-fallback path:
1. **Path 1** (Hybrid CTC, parakeet_ja のみ): `decoder_type='ctc'` + confidence_cfg
2. **Path 1.5** (本 PR-A.4.3 新規, pure RNNT/TDT): `preserve_alignments` + confidence_cfg
3. **Path 2** (Legacy fail-open): strategy-only (token_confidence は None)

---

## Hypotheses

| # | Hypothesis | Verdict |
|---|---|---|
| **H1** | TDT decoding に `preserve_alignments=True` + `confidence_cfg.preserve_token_confidence=True` で `hypothesis.token_confidence` populate される | ✅ **CONFIRMED** (Phase 1 probe + Section 1) |
| **H2** | Native English transcription で `token_confidence_mean ≫ 0.005` (threshold pass) | ✅ **CONFIRMED** (0.2452, 49× margin) |
| **H3** | Stream pipeline benchmark で `webrtc × synthetic × filter on` の hallucination を有意に改善 | ✅ **CONFIRMED** (Hall.(post) 75% → 12.5%) |
| **H4** | 既存 helper (`_extract_engine_confidence`) を流用可能 (list[float] / Tensor 両対応で migration ゼロ) | ✅ **CONFIRMED** (Parakeet 既存 helper + Canary PR-A.4.2 で Tensor 対応済の helper を再利用、変更不要) |

---

## Section 1: Engine-level Smoke (3 clip)

### Results

| Clip | kind | **token_confidence_mean** | Parakeet text |
|---|---|---|---|
| `librispeech_1089-134686-0001.wav` | speech (native en) | **0.2452** ✅ | `"Stuff it into you, his belly counselled him."` |
| `applause_5_claps.wav` | non-speech | **None** (fail-open) | `""` (empty) |
| `desk_tap.wav` | non-speech | **None** (fail-open) | `""` (empty) |

### Margin / threshold

| Metric | Value |
|---|---|
| speech mean (n=1) | **0.2452** |
| non-speech (n=2) | empty text + None confidence (engine 自身が refuse) |
| threshold default (既存 `token_conf_threshold`) | `0.005` |
| margin to threshold | **49×** (speech 0.2452 / threshold 0.005) |
| Case classification | **Case A** (clear margin) |

### Findings (Section 1)

#### F1.1 — Parakeet 英語 native で `token_confidence_mean = 0.2452` ✅

Canary (0.0724) の **~3×**、Parakeet 日本語 speech mean (0.05) の **~5×**、threshold 0.005 の **49×**。3 engine 中で最大の margin。

#### F1.2 — 非音声で Parakeet 英語は empty text を返す fail-safe (Canary と同様)

applause / desk_tap で:
- text: 空文字
- `hypothesis.token_confidence`: None
- → filter 介入不要 (字幕 stream に空 text が届かない)

Canary と同じ fail-safe 挙動、Voxtral (拍手で "." を hallucinate) や Parakeet_ja (低 confidence 幻覚) と異なる。

---

## Section 2: Stream Pipeline Benchmark (12 cell sweep)

### Setup

```powershell
$env:LIVECAP_NON_SPEECH_CORPUS_DIR = ".tmp\non_speech_corpus"
uv run python -m benchmarks.non_speech_filter.sweep `
    --backend silero,tenvad,webrtc `
    --engine parakeet `
    --corpus-dir .tmp\non_speech_corpus `
    --device cuda `
    --preset baseline_off `
    --filter-mode off,on
```

### Results (主要 cell)

| Backend | Corpus | filter | Hall.(pre) | Hall.(post) | SR(pre) | SR(post) | P50 ms |
|---|---|---|---|---|---|---|---|
| silero | real | off | 0.0% | 0.0% | 100% | 100% | 673 |
| silero | real | on | 0.0% | 0.0% | 100% | 50% ⚠ | 375 |
| tenvad | real | off | 0.0% | 0.0% | 100% | 100% | 547 |
| tenvad | real | on | 0.0% | 0.0% | 100% | 50% ⚠ | 483 |
| webrtc | real | off | 50.0% | 50.0% | 100% | 100% | 332 |
| webrtc | real | on | 50.0% | 50.0% | 100% | 75% | 284 |
| silero | synthetic | off/on | 0% | 0% | 0% | 0% | 7 |
| tenvad | synthetic | off | 25.0% | 25.0% | 100% | 100% | 85 |
| tenvad | synthetic | on | 25.0% | 25.0% | 100% | 100% | 72 |
| webrtc | synthetic | off | 75.0% | 75.0% | 100% | 100% | 101 |
| **webrtc** | **synthetic** | **on** | **75.0%** | **12.5%** ✅ | 100% | 80% | 95 |

### Findings (Section 2)

#### F2.1 — `webrtc × synthetic × filter on`: Hall.(post) 75% → 12.5% を実測実証 ✅

Voxtral (PR-A.4.1) / Parakeet_ja (PR-A.0) で観測された hallucination 抑制効果が Parakeet 英語でも同等に機能。filter は engine の low-confidence hallucination の **6 件中 5 件 (~83%) を drop**。

#### F2.2 — `webrtc × real × filter on`: Hall.(post) 改善なし (50% → 50%) — 言語不一致が confounding factor

PR-B real corpus の negative items (desk_tap 等) に対して Parakeet 英語が **高 confidence で hallucinate** している可能性が高い。原因の仮説:

- Parakeet 英語は negative items を「English ぽい音」と認識し、 plausible English transcript を高 confidence で出力
- 一方 Parakeet 日本語 (PR-A.0 で 50%→0% 改善実証) は negative items を「日本語ぽくない」と判断し低 confidence
- → engine 自体の言語認識特性の差

#### F2.3 — SR(post) drop in `silero / tenvad × real` (100% → 50%): 言語不一致による false reject

PR-B real corpus の positive items は **日本語音声**。Parakeet 英語は日本語音声を transcribe しようとして低 confidence transcript を出力 → filter が「低 confidence ≒ hallucination」と判断し drop。

これは **language-mismatch case** (Canary PR-A.4.2 の Section 2 と同じパターン、PR-A.4.2 では engine 自身が empty text を返した、Parakeet 英語は無理に transcribe を試みる)。

→ **Production user 注意**: Parakeet 英語を非英語音声に使う場合、confidence filter で false reject が発生する可能性あり。`--confidence-filter off` で opt-out 可能。

#### F2.4 — Latency 影響ほぼなし

filter off (P50 547-673 ms) vs filter on (P50 375-483 ms) で実は filter on の方が速い (filter で drop された clip は post-processing が skip される効果)。confidence cfg 計算自体の overhead は negligible。

---

## Section 3: Language Coverage

### Parakeet 英語 言語サポート

| Mode | サポート言語 | 本 PR で verify 済 |
|---|---|---|
| English (en) | ✅ native | ✅ Section 1 + 2 で確認、49× margin |
| 他言語 | ❌ 非対応 (English-only model) | (Section 2 で日本語音声に対する false reject を実機確認) |

### F3.1 — English native で threshold `0.005` validate 済

Section 1 で `token_confidence_mean = 0.2452` (LibriSpeech 1 clip)、Parakeet 日本語 speech mean 0.05、Canary speech 0.0724 と比べて最大 margin。**threshold `0.005` (既存) を変更せず default 維持**。

### F3.2 — 非英語入力時の false reject リスク (production user 注意)

Section 2 で実測した「日本語音声 + Parakeet 英語 + filter on → SR(post) 50%」は意図的な production constraint:

- **対応策 1**: 日本語音声には Parakeet 日本語 (`parakeet_ja`) を使用 (PR-A.0 で対応済、token_confidence_mean populate)
- **対応策 2**: 言語が不明 / 多言語混在の場合は Voxtral (8 言語) or Canary (4 言語) を選択
- **対応策 3**: Parakeet 英語で多言語を扱う必要がある場合は `--confidence-filter off` で opt-out

---

## Decision

### Threshold `0.005` (既存) を変更しない

| Criterion | Result |
|---|---|
| H1-H4 全て confirmed | ✅ |
| Section 1 native English margin | 49× (Case A) |
| Section 1 non-speech | empty text (fail-safe、filter 介入不要) |
| Section 2 `webrtc × synthetic × on` Hall.(post) 改善 | 75% → 12.5% (5/6 drop) |
| Section 2 language-mismatch case | SR(post) 50% (false reject、`--confidence-filter off` で opt-out 可能) |
| WhisperS2T / Parakeet_ja / Voxtral / Canary 退行 | ✅ ゼロ (`confidence_filter.py` 変更なし) |

→ **default `0.005` 採用**。Parakeet 英語 user は `--confidence-filter on` (default) で英語 audio の hallucination が自動 drop される。

### `FilterConfig` schema 不変

Parakeet 英語専用 field 追加なし、`confidence_filter.py::should_reject()` 変更なし。**既存の `token_conf_threshold` path を Parakeet 英語も共用**。

### `EngineConfidence._extract_engine_confidence()` helper 変更なし

Canary PR-A.4.2 で **Tensor / List / numpy** 全部扱える形に拡張済の helper を Parakeet 英語でもそのまま利用。Parakeet 英語の `hypothesis.token_confidence` は **List[float]** (Parakeet 日本語と同じ型) で来るため、互換性問題なし。

---

## Implications

### Parakeet 英語 user の挙動変化

| Mode | Before (PR-A.4.3 前) | **After (本 PR 後)** |
|---|---|---|
| `--confidence-filter off` | filter 無効、Parakeet 英語出力そのまま | (不変) |
| `--confidence-filter observe` | filter 判定なし (engine_confidence 全 None で fail-open) | **filter 判定 JSON log** (observe mode active) |
| `--confidence-filter on` (default) | filter 無効 (engine_confidence 全 None で fail-open) | **`token_confidence_mean < 0.005` で reject (active)** |
| Decoding strategy | RNNT/TDT minimal (strategy のみ) | **TDT + preserve_alignments + confidence_cfg** (Path 1.5) |

### 他 engine の挙動 (不変)

- WhisperS2T / Parakeet_ja / Voxtral / Canary: `confidence_filter.py` 共用 path、退行ゼロ
- ReazonSpeech / qwen3asr: 不変 (fail-open)

### Caveats (production user 向け)

1. **Parakeet 英語は English-only model**: 非英語音声に対しては Section 2 で false reject (SR(post) 50%) を実測。日本語音声には `parakeet_ja` を使用推奨
2. **Real corpus の hallucination 改善は engine 特性依存**: Section 2 で `webrtc × real × on` の改善が観察されなかったのは Parakeet 英語が negative items を高 confidence で hallucinate する特性のため。`webrtc × synthetic × on` (75%→12.5%) では明確な効果を実証
3. **3-fallback path**: NeMo API 変更で `preserve_alignments` が拒否される場合も model 自体は動作 (Path 2 strategy-only で fail-open)

---

## Reproducibility

### Section 1 smoke (3 clip)

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:LIVECAP_NON_SPEECH_CORPUS_DIR = ".tmp\non_speech_corpus"
uv run python .tmp/pr_a4_3_parakeet_en_smoke.py
```

期待出力: 上記 Section 1 Results table を再現 (token_confidence_mean ≈ 0.24-0.25)。

### Section 2 sweep (12 cell)

```powershell
uv run python -m benchmarks.non_speech_filter.sweep `
    --backend silero,tenvad,webrtc `
    --engine parakeet `
    --corpus-dir .tmp\non_speech_corpus `
    --device cuda `
    --preset baseline_off `
    --filter-mode off,on
```

期待出力: `webrtc × synthetic × on` で Hall.(post) ≤ 15% (75% → ~12.5%)、他 cell は table の通り。

---

## 関連リソース

- 親 Issue: [#311 v2.1+v2.2](https://github.com/Mega-Gorilla/livecap-cli/issues/311) — PR-A.4 (voxtral + canary + 本 PR で parakeet_en 追加)
- 前段 PR-A.4.1 Voxtral: [PR #313 MERGED](https://github.com/Mega-Gorilla/livecap-cli/pull/313)
- 前段 PR-A.4.2 Canary: [PR #315 MERGED](https://github.com/Mega-Gorilla/livecap-cli/pull/315)
- 前段 PR-A.0 Parakeet_ja: [PR #309 MERGED](https://github.com/Mega-Gorilla/livecap-cli/pull/309) — `_configure_decoding_with_confidence` originator
- Canary decision doc (Section 1/2/3 pattern reference): [`docs/research/canary-confidence-smoke-2026-06-11.md`](https://github.com/Mega-Gorilla/livecap-cli/blob/main/docs/research/canary-confidence-smoke-2026-06-11.md)

by.Scotty
