# Calibration Corpus Sources

> 本 doc は **Issue [#338](https://github.com/Mega-Gorilla/livecap-cli/issues/338) PR-β** で導入された active calibration harness (`benchmarks/confidence_calibration/build_corpus.py` + `sweep.py`) で使用する **audio corpus の source URL 一覧** と、各 source の license / 取得方法 / 推奨用途を documenting します。
>
> **raw audio は repo に commit しません** (Issue #338 設計判断、本 doc は URL list のみ)。各 user / contributor は手元で取得し、`LIVECAP_CALIBRATION_CORPUS_DIR` env var 経由で参照してください。

## Corpus directory layout (推奨)

```
$LIVECAP_CALIBRATION_CORPUS_DIR/
  ├── manifest.jsonl            ← build_corpus.py が生成
  ├── ja_clean/                  ← JA clean speech segments
  ├── ja_non_speech/             ← JA non-speech (applause / silence / etc.)
  ├── ja_noisy/                  ← JA noisy speech (環境音 + speech)
  ├── en_clean/                  ← EN clean speech segments
  ├── en_non_speech/             ← EN non-speech
  └── _raw/                      ← yt-dlp download + ffmpeg 中間 wav (gitignore 対象)
```

## Primary corpus (Issue #338 推奨、user 提供)

### JA clean speech — リトル・プリンス Chapter 1 朗読

| Item | URL |
|---|---|
| Audio | https://www.youtube.com/watch?v=6aJ3jsVeQIg |
| Reference text | https://taltal3014.lsv.jp/little-prince/LittlePrince1.html |
| Scope | Chapter 1 単独 |
| License | **訳者著作権存続** (内藤濯訳 PD 化予定 2046)、**private 利用に限定** |

#### Build command

```bash
uv run python -m benchmarks.confidence_calibration.build_corpus \
    --source "https://www.youtube.com/watch?v=6aJ3jsVeQIg" \
    --reference-text "https://taltal3014.lsv.jp/little-prince/LittlePrince1.html" \
    --output-dir "$LIVECAP_CALIBRATION_CORPUS_DIR/ja_clean" \
    --language ja --label speech
```

### EN clean speech — The Little Prince Chapter 1 朗読

| Item | URL |
|---|---|
| Audio | https://www.youtube.com/watch?v=fxvOPdOYyeo (**0:06 開始 trim 必須**) |
| Reference text | https://esl-bits.eu/Novellas.for.ESL.Students/LittlePrince/01/text.html |
| Scope | Chapter 1 単独 (全 27 章中) |
| License | 教育目的 fair use、**private 利用に限定** |

#### Build command

```bash
uv run python -m benchmarks.confidence_calibration.build_corpus \
    --source "https://www.youtube.com/watch?v=fxvOPdOYyeo" \
    --reference-text "https://esl-bits.eu/Novellas.for.ESL.Students/LittlePrince/01/text.html" \
    --output-dir "$LIVECAP_CALIBRATION_CORPUS_DIR/en_clean" \
    --language en --label speech \
    --start-offset-sec 6.0 \
    --max-duration-sec 900  # Chapter 1 相当 ~15 min、duration 確認後調整
```

## Alternative sources (PD / CC、再現性確保)

Primary corpus が dead link 化 / 著作権で利用不可になった場合の **置換候補**:

### JA clean speech

| Source | License | URL | 用途 |
|---|---|---|---|
| **Common Voice ja** | CC-0 | https://commonvoice.mozilla.org/en/datasets | 100+ 時間の crowdsourced ja audio、PD |
| **JSUT corpus (basic5000)** | CC-BY-SA 4.0 | https://sites.google.com/site/shinnosuketakamichi/publication/jsut | studio 録音 ja、5000 発話 |
| **ReazonSpeech** (huggingface) | 商用利用可 | https://huggingface.co/datasets/reazon-research/reazonspeech | 大規模 ja audio + transcript、ASR 学習用 |

### EN clean speech

| Source | License | URL | 用途 |
|---|---|---|---|
| **LibriSpeech** | CC-BY 4.0 | https://www.openslr.org/12/ | 1000 時間の朗読 EN、ASR 標準 corpus |
| **Common Voice en** | CC-0 | https://commonvoice.mozilla.org/en/datasets | 同上 ja の英語版 |

### Non-speech (両言語共通)

| Source | License | URL | 用途 |
|---|---|---|---|
| **ESC-50** | CC-BY-NC 4.0 | https://github.com/karolpiczak/ESC-50 | 50 class × 40 samples、環境音 (applause, glass break, door knock 等)、non_speech 補強の標準 |
| **UrbanSound8K** | CC-BY-NC 4.0 | https://urbansounddataset.weebly.com/urbansound8k.html | 8 class × ~1000 件 urban sound |
| **MUSAN noise** | (mixed) | https://www.openslr.org/17/ | 多様な noise type、speaker overlap test |

## 著作権 / Privacy 取扱方針 (PR-β 設計判断)

1. **raw audio は repo にコミット禁止** — `LIVECAP_CALIBRATION_CORPUS_DIR` 経由で user 手元参照
2. **本 doc には URL のみ** 記載、著作権ある content (原稿 text / audio file) は repo に置かない
3. **訳者著作権 / 朗読権** がある source は private 利用に限定 — calibration 自体は user 手元で完結
4. **PD / CC license alternative** を並記 — primary source が dead link 化しても再現可能
5. **test fixtures** は CC-0 / PD source の超小 sample (< 5MB) のみ `tests/fixtures/calibration/` に置く (本 PR 内では未追加、将来必要時)

## Calibration target overview

JA / EN corpus を build した後、5 engine の sweep を実行:

| Engine | Signal | Lang | Priority | Quantization |
|---|---|---|---|---|
| `reazonspeech` | `avg_logprob` | ja | 🔴 P0 | int8 / float32 両方 |
| `qwen3asr` | `avg_logprob` | ja | 🟠 P1 | (HF default) |
| `qwen3asr` | `avg_logprob` | en | 🟠 P1 | (HF default) |
| `parakeet_ja` | `token_confidence_mean` | ja | 🟡 P2 | (NeMo default) |
| `whispers2t` | `no_speech_prob` | ja | 🟡 P2 | int8 / float32 |

各 engine の build + sweep workflow は `benchmarks/confidence_calibration/README.md` の Stage 2 quickstart を参照。

## 関連リソース

- 親 issue: [Issue #338](https://github.com/Mega-Gorilla/livecap-cli/issues/338) (calibration harness)
- 加速対象: [Issue #334](https://github.com/Mega-Gorilla/livecap-cli/issues/334) PR-2 / PR-3 / PR-4
- 既存 corpus pattern: [`tests/integration/non_speech_filter/test_baseline.py`](../../tests/integration/non_speech_filter/test_baseline.py) (`LIVECAP_NON_SPEECH_CORPUS_DIR` precedent)
- adding-an-engine guide: [`docs/contributor/adding-an-engine.md`](../contributor/adding-an-engine.md) §5 (threshold calibration template)
