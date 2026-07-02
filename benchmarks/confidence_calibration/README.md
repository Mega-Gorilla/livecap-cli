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

Engine ID 対応表 (全 7 engine、PR #339 3rd round で完備):

| metadata.py `id` (CLI ID) | display string 例 | normalize 経路 |
|---|---|---|
| `reazonspeech` | `"ReazonSpeech K2 (CPU, Int8)"` / `"ReazonSpeech K2 v2"` | first-word |
| `whispers2t` | `"WhisperS2T base"` / `"WhisperS2T large-v3"` | first-word |
| `parakeet` | `"NVIDIA Parakeet TDT 0.6B v2"` | **prefix map** (provider 名 NVIDIA 始まり) |
| `parakeet_ja` | `"NVIDIA Parakeet TDT CTC 0.6B JA"` | **prefix map (長い側優先)**、`parakeet` より先に match |
| `canary` | `"NVIDIA Canary 1B Flash"` | **prefix map** |
| `voxtral` | `"MistralAI Voxtral Mini 3B"` | **prefix map** (provider 名 MistralAI 始まり) |
| `qwen3asr` | `"Qwen3-ASR 0.6B"` / `"Qwen3-ASR 1.7B"` | first-word `"qwen3-asr"` + alias `"qwen3-asr"` → `"qwen3asr"` |

`normalize_engine_id()` の 3 段階 normalize:
1. `strip + lower`
2. **Multi-word prefix map** (長い側優先で `NVIDIA Parakeet TDT CTC` → `parakeet_ja` を `NVIDIA Parakeet` → `parakeet` より先に評価)
3. **First-word + alias** (`qwen3-asr` → `qwen3asr` 等)

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

## Stage 2 Quickstart (PR-β、active calibration)

Stage 1 は **既に observe 運用していて log が貯まっている前提**。Stage 2 (本節) は **user 提供 audio file から直接 calibration** する path。リトル・プリンス朗読 + 原稿で end-to-end の sweep を実行可能。

### 1. 環境準備

```bash
# yt-dlp + ffmpeg が install されていること
uv sync --extra dev   # yt-dlp dev dep が install される

# Corpus directory
export LIVECAP_CALIBRATION_CORPUS_DIR="$HOME/.calibration_corpus"
mkdir -p "$LIVECAP_CALIBRATION_CORPUS_DIR"
```

### 2. JA Chapter 1 corpus build

```bash
uv run python -m benchmarks.confidence_calibration.build_corpus \
    --source "https://www.youtube.com/watch?v=6aJ3jsVeQIg" \
    --reference-text "https://taltal3014.lsv.jp/little-prince/LittlePrince1.html" \
    --output-dir "$LIVECAP_CALIBRATION_CORPUS_DIR/ja_clean" \
    --language ja --label speech \
    --engine-kwargs "model_size=base" "language=ja"   # CPU 軽量化 + 言語 hint
```

> **`--engine-kwargs model_size=base` 推奨**: alignment 用 WhisperS2T を軽量
> 設定 (~150 MB) に切り替え。default は `large-v3` (~1.5 GB、CPU では数倍遅い)。
> alignment は coverage fuzzy match なので `base` で十分 (PR #340 review 指摘 3)。
>
> **`--engine-kwargs language=ja` 推奨**: `--language ja` は manifest metadata 用で、
> alignment ASR には渡りません。WhisperS2T default 言語が `ja` のため省略しても
> 動作しますが、明示することで他 engine への切替時に挙動が安定します
> (PR #340 review 2nd round 指摘、smoke verify docs §6 と整合)。

Build flow:
1. `yt-dlp` で audio download (cache 済なら skip)
2. `ffmpeg` で 16 kHz mono wav 変換
3. **Silero VAD** で speech segment 切り出し (threshold + hysteresis)
4. 各 segment で WhisperS2T (or 指定 engine) で transcribe
5. 原稿 text と `difflib.SequenceMatcher.find_longest_match()` で fuzzy match、
   **coverage score** (= matched substring 長 / transcribed 長、0.0-1.0) を計算
6. `manifest.jsonl` を **upsert** (idempotent + resumable、`--force` でも path
   重複なし、他 source の entry は保持)

### 3. EN Chapter 1 corpus build (0:06 trim 必須)

```bash
uv run python -m benchmarks.confidence_calibration.build_corpus \
    --source "https://www.youtube.com/watch?v=fxvOPdOYyeo" \
    --reference-text "https://esl-bits.eu/Novellas.for.ESL.Students/LittlePrince/01/text.html" \
    --output-dir "$LIVECAP_CALIBRATION_CORPUS_DIR/en_clean" \
    --language en --label speech \
    --start-offset-sec 6.0 \
    --max-duration-sec 900 \
    --engine-kwargs "model_size=base" "language=en"   # language=en は EN audio に必須
```

> **`--engine-kwargs language=en` 必須 (EN audio)**: `--language en` は manifest
> metadata 用で、alignment ASR には渡りません。`language` hint なしだと
> WhisperS2T が EN audio を `ja` と auto-detect し、日本語 hallucination で
> alignment coverage が壊滅的に低下することを Phase 4 smoke verify で確認済み
> (`docs/research/calibration-corpus-smoke-verify.md` §6、PR #340 review 2nd round
> 指摘)。

### 4. Non-speech / noisy_speech 補強 (~20-30 件)

ESC-50 等の PD corpus or 手元 audio を **手動で** `$LIVECAP_CALIBRATION_CORPUS_DIR/{ja,en}_non_speech/` に配置、`manifest.jsonl` に entry 追記:

```bash
# 例: ESC-50 から applause sample を copy
cp ESC-50/audio/1-100032-A-0.wav "$LIVECAP_CALIBRATION_CORPUS_DIR/ja_non_speech/applause_001.wav"
echo '{"path": "ja_non_speech/applause_001.wav", "label": "non_speech", "language": "ja", "subtype": "applause"}' \
    >> "$LIVECAP_CALIBRATION_CORPUS_DIR/manifest.jsonl"
```

詳細は [`docs/research/calibration-corpus-sources.md`](../../docs/research/calibration-corpus-sources.md) (PD alternative source 一覧) を参照。 なお **Phase 2 では ESC-50 / MUSAN の自動 augmentation CLI が用意されています** (§4.6 参照、 15 category × 10 sample + MUSAN noise 50 sample を script 一発で追加可能)。

### 4.5. (任意) kana-level alignment metric を追加 (PR-γ、JA 表記揺れ吸収)

JA 朗読 corpus では、ASR の音響出力が正しくても **表記揺れ** (kanji ↔ katakana ↔ 算用数字) だけで text-level coverage が低くなる現象が Phase 4 smoke verify で観測されました (例: 「1人で」 vs 「一人で」、 「サハラ砂漠」 vs 「さはらさばく」 reference)。`recompute_alignment.py` で **既存 manifest に kana-level coverage を追加** することで、acoustic confidence と lexical surface form を分離できます。

```bash
uv run python -m benchmarks.confidence_calibration.recompute_alignment \
    --manifest "$LIVECAP_CALIBRATION_CORPUS_DIR/manifest.jsonl" \
    --reference-text-ja "https://taltal3014.lsv.jp/little-prince/LittlePrince1.html" \
    --reference-text-en "https://esl-bits.eu/Novellas.for.ESL.Students/LittlePrince/01/text.html"
```

各 entry に **3 つの kana field** が追加されます (text-level field は **不変**、forensic safe):

- `alignment_score_kana` — kana 化した両側で計算した coverage (0.0–1.0)
- `reference_text_matched_kana` — 一致した kana span (reference 側)
- `transcribed_text_kana` — 正規化後の transcribed (debugging 用)

正規化 pipeline: NFKC → CJK 隣接の Arabic 数字 run → kanjize で 漢数字化 (`1200 → 千二百`、 `1人 → 一人` 等) → pykakasi で hiragana 化 → 句読点 strip。 PR #341 codex-review 訂正の v4 反映: v1 blanket mask (`一人` と `二人` 同一視 false-high) → v2 per-char (compound `千二百` を `10002100` と誤変換) → v3 kanji→arabic (`一緒`/`十分`/`一番`/`一人` の pykakasi 自然な読みを壊す) → v4 arabic→kanji で全方位対応 (`一緒` 等の idiom は無変更、 EN の `Chapter 1` も無変更、 `1人 ↔ 一人` 等の cross-form は kanjize で kanji 化されて pykakasi の compound rules で正規化、 詳細は [`_normalize_jp.py`](_normalize_jp.py) docstring 参照)。

> **License note (PR-γ)**: kana metric は **`pykakasi` (GPL-3.0-or-later)** と **`kanjize` (MIT)** に依存します。本 repo は AGPL-3.0-only ですが、 両 lib とも `[project.optional-dependencies] dev` (`uv sync --extra dev` でインストール) 限定の dev / benchmark 依存です。**production runtime は両 lib を一切 import しません** (`tests/test_production_no_pykakasi.py` で static grep guard、 両 lib を parametrize で chec)。新規 `build_corpus` invoke も kana field を自動で書込みます (PR-γ 後)。

> **EN audio**: pykakasi は ASCII を pass-through するため、EN entry の kana score は text-level score とほぼ等価です (NFKC + punctuation strip の差のみ)。

### 4.6. (Phase 2) 自動 augmentation via ESC-50 / MUSAN CLI (production-realistic non_speech)

Issue #338 Phase 1 report ([`docs/research/calibration-japan-engines-2026-07.md`](../../docs/research/calibration-japan-engines-2026-07.md)) の最重要 caveat は「synthetic non_speech (silence + white/pink noise) は production non_speech (applause / laughing / engine / mouse click 等) より **easier** で、 data-driven threshold が probe を pass してしまう」。 Phase 2 は **ESC-50** (環境音 50 category) と **MUSAN** noise を augment して、 production-realistic な threshold を再算定します。

**dataset (raw audio は git 外、 `.tmp/` 配下に配置)**:

| Dataset | License | 用途 | URL |
|---|---|---|---|
| ESC-50 | **CC BY-NC 4.0 (Non-Commercial)** | dev/calibration のみ | https://github.com/karolpiczak/ESC-50 |
| MUSAN noise | **CC BY 4.0** | dev/calibration のみ | https://www.openslr.org/17/ |

**ESC-50 augment** (~450 sample、 15 category × 10 file × 3 chunk):

```bash
# 事前 download (~600 MB、 dev-only、 git 外の .tmp/ に配置)
mkdir -p .tmp/esc50_source
# (browser or curl で ESC-50-master.zip を .tmp/esc50_source/ に配置し unzip)
# もしくは --download flag で自動化:

uv run python -m benchmarks.confidence_calibration.gen_esc50_non_speech \
    --source-dir .tmp/esc50_source/ESC-50-master \
    --output-dir "$LIVECAP_CALIBRATION_CORPUS_DIR" \
    --samples-per-category 10
# → 450 entries added to manifest (15 category × 10 file × 3 chunk), wavs in ja_non_speech_esc50/
```

対象 15 category (Plan D2、 production-realistic):
- Human non-speech: `laughing` / `sneezing` / `coughing` / `breathing` / `clapping` / `footsteps`
- Natural: `rain`
- Interior: `door_wood_knock` / `mouse_click` / `keyboard_typing` / `clock_tick` / `glass_breaking`
- Exterior: `engine` / `car_horn` / `siren`

`--categories` で override 可能 (comma-separated)、 `--force` で既存 ESC-50 entry を全削除して再 augment (safe re-run)、 `--dry-run` で書込前 preview。

**MUSAN noise augment** (~50-250 sample、 `music/` と `speech/` は除外):

```bash
# 事前 download (~11 GB、 dev-only、 --source-dir 推奨 or --download)
mkdir -p .tmp/musan_source
# (browser or curl で musan.tar.gz を .tmp/musan_source/ に配置し展開)

uv run python -m benchmarks.confidence_calibration.gen_musan_noise \
    --source-dir .tmp/musan_source/musan \
    --output-dir "$LIVECAP_CALIBRATION_CORPUS_DIR" \
    --samples 50
# → 50 files × up to 5 chunks each = ~150-250 entries added
```

`--max-chunks-per-file` で file 当たり chunk 数調整、 `--samples` で file 選択総数 (uniform stride で deterministic)。 `music/` と `speech/` サブセットは意図的に除外 (music は BGM 判断が別問題、 speech は false positive)。

**manifest schema (Phase 2 additive fields)**:

Phase 2 augmented entry は Phase 1 の 14 field に加えて 3 field を持ちます (backward compatible、 既存 entry は field なしのまま `pipeline.load_calibration_corpus()` が metadata dict に格納):

```jsonc
{
  "path": "ja_non_speech_esc50/clapping_1-100032-A-22_chunk0.wav",
  "label": "non_speech",
  "language": "ja",
  "subtype": "clapping",              // ESC-50 category name / MUSAN sub-dir
  "transcribed_text": "",              // sweep 時に engine で埋める
  "alignment_score": 0.0,
  "alignment_score_kana": 0.0,
  "duration_sec": 1.5,
  // Phase 2 additive attribution fields
  "source_dataset": "esc50",           // "esc50" | "musan"
  "source_file": "1-100032-A-22.wav",  // 元 file 名 (attribution)
  "source_license": "CC BY-NC 4.0"
}
```

> **License note (Phase 2)**: ESC-50 は **CC BY-NC 4.0 (Non-Commercial)**、 MUSAN は **CC BY 4.0**。 両 dataset とも **dev/calibration 限定** の raw audio 依存です。 **production runtime (`livecap_cli/`) は audio dataset を一切 import しません** (Python パッケージではない data のため import しようがない)。 `.tmp/` 配下の raw audio は既存の `.gitignore` rule で保護されており、 git push は物理的に不可能です。

#### Layered evaluation 内での本 CLI の位置づけ (PR #343 review 方針反映)

本 CLI は **Layer 2** に相当し、 **これだけで PR-4 default threshold を確定するには不十分** です。 Phase 2 report / PR-4 では以下 layered evaluation を統合する必要があります:

| Layer | 対象 | 現状 |
|---|---|---|
| **Layer 1** | Phase 1 baseline: clean speech + synthetic silence/noise | ✅ Phase 1 report 完了 (PR #342) |
| **Layer 2** | **本 CLI**: ESC-50 / MUSAN non_speech hard negative augment | ✅ PR #343 (本 PR) |
| **Layer 3** | Clean speech に ESC-50/MUSAN noise を SNR mix した noisy_speech corpus | ⏳ follow-up PR (別途 `gen_mixed_noisy_speech.py` 追加予定) |
| **Layer 4** | Production observe log による candidate threshold の replay | ⏳ `parse_observe.py` 既存、 user 環境の log 待ち |
| **Layer 5** | VAD + noise gate + confidence_filter + ASR の realtime E2E 統合確認 | ⏳ Issue #338 Phase 3 |

**ESC-50 acoustic realism caveat**: ESC-50 は 5 sec の **curated classification clip** で、 実マイクの room impulse / distance / gain / VAD segmentation / realtime chunking は再現しません。 公式 README にも preprocessing / bandlimiting 由来の caveat あり。 → production 分布の完全代替とは見ず、 **hard negative の一次候補** として使用します。

**PR-4 で採用すべき metrics 分解** (単一 F1 最大化を避け、 Pareto 条件で threshold を選ぶ):

- `clean_speech_frr` — clean 朗読 corpus (Phase 1 baseline)
- `noisy_speech_frr` by SNR — Layer 3 corpus (SNR 別に分解)
- `non_speech_pass_rate` by subtype — ESC-50 category / MUSAN sub-dir 別
- `hallucination_rate` on non_speech — text 生成率 (empty vs non-empty)
- **known probe pass/reject** — Phase 1 §4.2 の applause -0.46 / desk_tap -0.50 等 (Pareto の必須 gate)

具体的 Pareto gate 例: `clean_speech_frr ≤ 1%` かつ `noisy_speech_frr ≤ 5%` かつ `known_probe reject`。 単一 F1 最大は environmental sound reject と noisy speech drop の trade-off を隠すため PR-4 では採用しない予定です。

**MUSAN と ESC-50 の役割分担** (方針レビュー反映): MUSAN は主に **背景ノイズ耐性**、 ESC-50 は **speech-like transient hard negative** (`clapping` / `coughing` / `keyboard_typing` / `mouse_click` 等) 向け。 両方を混ぜて augment することで補完関係を確保します。

### 4.7. (Layer 3) SNR-mixed noisy_speech CLI (`noisy_speech_frr by SNR` の evidence gathering)

Layered evaluation の Layer 3 (Layer 1 clean speech + Layer 2 non_speech hard negative の次) として、 **clean speech に Layer 2 noise を目標 SNR で混合** した `label=noisy_speech` corpus を生成する CLI。 PR #343 codex-review 2nd round で確認された **Pareto gate `noisy_speech_frr by SNR ≤ 5%`** の direct evidence 収集用。

**用途**: PR-4 で default threshold を下げる時、 clean speech FRR は改善しても production の実 mic で背景 noise がある会話 (`noisy_speech`) が過剰 reject される trade-off を SNR 別に定量化。

```bash
uv run python -m benchmarks.confidence_calibration.gen_mixed_noisy_speech \
    --output-dir "$LIVECAP_CALIBRATION_CORPUS_DIR" \
    --samples 50 \
    --snr-db-list "-5,0,5,10,20" \
    --noise-datasets esc50,musan \
    --speech-language ja
# → 50 speech × 5 SNR = 250 layer3_mix entries added
```

> **Language 引数について** (codex-review 対応): `--speech-language` は clean speech の filter と生成 entry の `language` field **両方**を制御します (mixed noisy_speech の language は clean speech の language と一致するのが自然、 別引数だと mismatch で `sweep.py --filter-by-language` を汚染するため意図的に単一引数)。 EN speech で mix したい場合は `--speech-language en` を指定するだけで output entry も `language="en"` になります。

**Default 設計** (Plan D1-D3):
- **SNR grid**: `-5 / 0 / 5 / 10 / 20 dB` (5 値)
  - `-5 / 0`: extreme low、 ASR degrade 境界
  - `5 / 10`: 家庭 / café の typical SNR、 Pareto gate 主戦場
  - `20`: 静オフィス、 clean と同等挙動を期待する境界
- **サンプル数**: 50 speech × 5 SNR = **250 entries** (各 SNR bucket 50 sample で FRR の 95%CI ≈ ±10%、 `≤ 5%` Pareto gate を判定可能な最小)
- **noise 分配**: 同一 speech sample を全 SNR で mix (paired within-subject 比較)、 speech 全体は noise pool を deterministic rotation (`noise_pool[i % len]`) で spread
- **paired evaluation**: SNR effect の pure comparison が可能 (異なる speech per SNR より statistical power 高い)

**Prerequisites** (loud fail if missing):
- `{output_dir}/manifest.jsonl` に `label=speech + language=<--speech-language>` が `>= --samples` 件必要 (Phase 1 build_corpus で生成)
- Layer 2 output (`{lang}_non_speech_esc50/` or `{lang}_non_speech_musan/` 等、 `source_dataset in <--noise-datasets>` の entry) が `>= 1` 件必要 (§4.6 の gen_esc50_non_speech / gen_musan_noise で生成)

**Output layout (multi-language 対応、 codex-review 2nd round 反映)**:

Mixed wavs は `{output_dir}/{speech_language}_noisy_speech/` に配置 → JA と EN を同 corpus で augment しても path 衝突なし:

```
$LIVECAP_CALIBRATION_CORPUS_DIR/
├── manifest.jsonl                            # 全 layer 混在 (label で区別)
├── ja_clean/                                  # Phase 1 speech (JA)
├── en_clean/                                  # Phase 1 speech (EN)
├── ja_non_speech_esc50/                       # Phase 2 (ESC-50)
├── ja_non_speech_musan/                       # Phase 2 (MUSAN)
├── ja_noisy_speech/                           # Layer 3 (JA speech + noise)
│   ├── segment_0000_snr-5dB_clapping.wav
│   ├── segment_0000_snr0dB_clapping.wav
│   └── ...
└── en_noisy_speech/                           # Layer 3 (EN speech + noise)
    ├── segment_0000_snr-5dB_clapping.wav
    └── ...
```

`--speech-language` は **language の single source of truth** — clean speech の filter、 output entry の `language` field、 output subdir 全てを一元制御。 別引数の設計は path collision と manifest 汚染の原因になるため意図的に排除しています。

**SNR mixing 数学的定義**: RMS-based (`_mix_snr.py:mix_at_snr`):
```
scale = sqrt(P_speech / (P_noise * 10^(SNR_dB/10)))
mixed = speech + noise * scale
```
Length matching は tile (noise が短い場合) / truncate (noise が長い場合)、 clip detection で `|mix|.max() > 1.0` 時は 0.95 peak に renormalize (SNR ratio 保持)。 unit test で 5 SNR grid × 3 signal pair の実測精度 ±0.5 dB を pin。

**Manifest schema (Layer 3 additive 4 field)**:

Phase 2 の 17 field を保持し、 Layer 3 で以下 4 field 追加:

```jsonc
{
  ...existing 17 field including source_dataset="layer3_mix"...,
  "snr_db": 10.0,                            // target SNR (dB)
  "noise_source_dataset": "esc50",           // "esc50" | "musan"
  "noise_source_file": "1-104089-A-22.wav",  // noise 元 file
  "noise_source_path": "ja_non_speech_esc50/clapping_1-104089-A-22_chunk0.wav"
}
```

`label="noisy_speech"` は `_core.py:_normalize_label()` で `speech` 扱い → filter に reject されると FRR contribution となり、 confusion matrix に自然に統合されます (既存 test `test_core.py::test_noisy_speech_treated_as_speech` で pin 済)。

**⚠️ Sweep report 側の gap (Phase 6 で対応)**: 現 `sweep.py:measure_signals()` は corpus metadata から 3 field (`text` / `language` / `is_available`) のみ pass-through、 `snr_db` は sweep report まで到達しません。 per-SNR FRR 集計は Phase 6 で以下のいずれかで対応:
- **option A**: `sweep.py` に `--per-metadata-key snr_db` flag 追加 (post-process group-by output)
- **option B**: 新規 `analyze_by_snr.py` CLI で manifest + engine 再 transcribe → per-SNR FRR 独立集計 (`recompute_alignment.py` パターン踏襲)

いずれも本 Layer 3 CLI PR とは scope 分離、 別 PR で isolated review。

**License note (Layer 3)**: 出力音声は **derivative** — clean speech (public domain の朗読) + Layer 2 noise (ESC-50 CC BY-NC 4.0 or MUSAN CC BY 4.0)。 `.tmp/` 配下で保護、 git push 事故物理的不可、 production runtime 依存なし。

### 5. Sweep 実行 (5 engine、JP モデル中心 + EN は Qwen3-ASR 並行)

```bash
# ReazonSpeech (P0、量子化別)
for quant in true false; do
  uv run python -m benchmarks.confidence_calibration.sweep \
      --engine reazonspeech --signal avg_logprob \
      --filter-by-language ja \
      --quantization $([ "$quant" = "true" ] && echo "int8" || echo "float32") \
      --engine-kwargs "use_int8=$quant" \
      --output "report_reazonspeech_$([ "$quant" = "true" ] && echo "int8" || echo "float32")_ja.json"
done

# Qwen3-ASR (P1、ja + en)
uv run python -m benchmarks.confidence_calibration.sweep \
    --engine qwen3asr --signal avg_logprob --filter-by-language ja \
    --engine-kwargs "language=Japanese" --output report_qwen3asr_ja.json

uv run python -m benchmarks.confidence_calibration.sweep \
    --engine qwen3asr --signal avg_logprob --filter-by-language en \
    --engine-kwargs "language=English" --output report_qwen3asr_en.json

# Parakeet_ja (P2)
uv run python -m benchmarks.confidence_calibration.sweep \
    --engine parakeet_ja --signal token_confidence_mean --filter-by-language ja \
    --threshold-min 0.001 --threshold-max 0.5 --step 0.005 \
    --output report_parakeet_ja.json

# WhisperS2T (P2)
uv run python -m benchmarks.confidence_calibration.sweep \
    --engine whispers2t --signal no_speech_prob --filter-by-language ja \
    --engine-kwargs "language=ja" --output report_whispers2t_ja.json
```

各 report の `recommended_threshold` + `false_reject_rate` + sample 分布を集計して、Issue #334 PR-4 の input report (`docs/research/calibration-japan-engines-*.md`) を作成。

### 5.5. (Phase 6a) 混同行列を metadata で分解する `--breakdown-by`

Phase 2 report で PR-4 の Pareto gate `noisy_speech_frr by SNR ≤ 5%` / `non_speech_pass_rate by subtype` を評価するには、 sweep report を **metadata の値ごとに分解** して混同行列を取り出す必要があります。 `--breakdown-by` はこの分解を **1 回の sweep 実行で** 実現します:

```bash
uv run python -m benchmarks.confidence_calibration.sweep \
    --engine reazonspeech --signal avg_logprob --filter-by-language ja \
    --breakdown-by snr_db,subtype,noise_source_dataset \
    --output report_reazonspeech_int8_ja.json
```

指定した各 key について、 report に per-value の閾値 sweep が追加されます:

```json
{
  ...既存の全 field...,
  "sweep": [...],  // 全 sample の全体 sweep (既存)
  "breakdown": {
    "snr_db": {
      "key": "snr_db",
      "value_counts": {"10.0": 50, "0.0": 50, "-5.0": 50, "5.0": 50, "20.0": 50, "__none__": 449},
      "sweep_by_value": {
        "10.0": [{"threshold": -0.2, "tp": ..., "false_reject_rate": ...}, ...],
        "0.0":  [...],
        "-5.0": [...],
        ...
      }
    },
    "subtype": {
      "key": "subtype",
      "value_counts": {"clapping": 30, "coughing": 30, "engine": 30, "__none__": 449, ...},
      "sweep_by_value": {"clapping": [...], "coughing": [...], ...}
    },
    "noise_source_dataset": {...}
  }
}
```

**設計**:
- `--breakdown-by` は comma-separated key list、 空文字 / duplicate は fail-fast
- 対象 key を manifest metadata から持たない sample (例: clean speech は `snr_db` field なし) は **`"__none__"` bucket** に集約
- typo などで全 sample が該当 key を持たない場合は warning log + `"__none__"` bucket のみで継続 (fail-close ではない)
- **`--breakdown-by` 未指定時は Phase 1 report と完全 backward compat** (`"breakdown": {}` の空 dict のみ追加、 追加 schema なし)

**Phase 2 report での使い方**:
- `report["breakdown"]["snr_db"]["sweep_by_value"]["10.0"]` から SNR 10 dB での per-threshold FRR を取得 → Pareto gate `noisy_speech_frr by SNR ≤ 5%` を validate
- `report["breakdown"]["subtype"]["sweep_by_value"]["clapping"]` から拍手だけの pass rate を取得 → hard negative category の効果測定
- 全 breakdown は 1 sweep で取得可能、 5 engine × ~1 hour GPU の再実行不要

### Quantization / language metadata

`--quantization` / `--filter-by-language` / `--engine-kwargs` で各 sweep の metadata を report に embed:

```json
{
  ...
  "metadata": {
    "engine_normalized": "reazonspeech",
    "engine_display": "ReazonSpeech K2 (CPU, Int8)",
    "corpus_dir": "/home/user/.calibration_corpus",
    "corpus_size_loaded": 120,
    "samples_with_signal": 118,
    "quantization": "int8",
    "language": "ja",
    "engine_kwargs": {"use_int8": true}
  }
}
```



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
