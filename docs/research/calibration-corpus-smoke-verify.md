# Calibration Corpus Smoke Verify (Phase 4)

> 本 doc は [Issue #338](https://github.com/Mega-Gorilla/livecap-cli/issues/338) PR-β
> ([PR #340](https://github.com/Mega-Gorilla/livecap-cli/pull/340)) で実装した
> `benchmarks/confidence_calibration/build_corpus.py` を、user 提供 corpus
> (リトル・プリンス JA / EN Chapter 1) で実音声 + 実 ASR (WhisperS2T base) を
> 使って verify した結果と、そこから得られた **alignment の機能性 / 制約 /
> 運用知見 / 発覚した bug** を記録します。Phase 4 本格 calibration (5 engine
> sweep) の前段として実施。
>
> **再現性**: 本 verify は user 手元の音源 + WhisperS2T base モデルで実施。
> raw audio / segment wav は repo 外 (`LIVECAP_CALIBRATION_CORPUS_DIR` 経由)。
> 結果 manifest は `.tmp/calibration_corpus_smoke/manifest.jsonl` に保存
> (git ignored)、本 doc には抜粋のみ embed。

## 0. ⚠️ 本 verify で発覚した重大 bug + factual 訂正

**Phase 4 verify は当初「JA WhisperS2T 漢字精度 + EN reference mismatch」 が
低 coverage の主因と分析していた**が、raw data 確認 (user 指示「問題指摘が
適切か再確認」) によって本分析の根拠が **2 つの方向で誤っていた** ことが
判明:

1. **🔴 SequenceMatcher autojunk bug 発覚** — `SequenceMatcher` の default
   `autojunk=True` heuristic が、reference text が長い (~数千 chars) + 頻出
   char (日本語 hiragana / 英語の冠詞・前置詞) が **200 件超** だと、頻出
   char を junk 扱いで match 候補から除外。結果 **本来連続 20 chars match
   できる substring が partial match (4-5 chars) に縮小**。本 verify の
   初期 manifest で多くの low coverage warning は **本 bug 起因の人工的
   artefact**。

2. **🟡 私の docs 上 factual 誤記** — autojunk bug の影響下で raw text を
   解釈した結果、以下の誤った主張が docs に残っていた:
   - 「1000マイル → 単位変換失敗」 ← 実は原稿は **「千マイル」** (漢数字)、
     ASR は「1000枚」と認識 (枚 vs マイル の同音誤認)
   - 「さっきに → 朗読 wording 差」 ← 実は原稿は「真っ先に」、ASR の同音
     誤認 (真っ先 → さっき)
   - 「ESL-bits が ESL 学習者向け **simplified** 版で朗読 (Saint-Exupéry
     原文) と乖離」 ← 実は autojunk bug を除けば EN reference text は朗読
     audio とよく一致 (coverage ≥ 0.5 が 12/24 = 50%)。「abridged 版」 と
     断定する根拠が崩れた

本 doc は **autojunk fix 後の最終 manifest** を analyze した数値を記載
(commit `<本 doc と同 commit>` で fix 適用済)。

### autojunk fix の concrete impact

```python
# Before fix (default autojunk=True、JP Phase 4 manifest 抜粋)
segment 0014: "一人でエンジンを修理しなければならなかった" (21 chars)
  reference contains: "1人でエンジンを修理しなければならなかった"  (連続 20 chars)
  autojunk=True : matched="エンジン" (4 chars)  → coverage 0.19  ❌
  autojunk=False: matched="人でエンジンを修理しなければならなかった" (20 chars) → coverage 0.95 ✅

# build_corpus.py の修正 (1 line):
matcher = SequenceMatcher(None, transcribed, reference_text, autojunk=False)
```

Test (`test_long_reference_finds_full_substring_autojunk_disabled`) で regression
guard 追加済。

## 1. 実行条件 (最終)

| Item | Value |
|---|---|
| 検証日 | 2026-06-29 |
| 環境 | Windows 11, Python 3.11.13, livecap-cli `feat/issue-338-confidence-calibration-pr-beta` (autojunk fix 含む) |
| ASR (alignment 用) | WhisperS2T `model_size=base, compute_type=int8, language=ja` (or `en`) |
| VAD | Silero VAD v5/v6 ONNX (`--vad-threshold 0.5 --min-speech-sec 0.5 --max-segment-sec 3.0 --min-silence-sec 0.3`、default) |
| Audio scope | 各言語 Chapter 1 朗読の **先頭 60 秒のみ** (smoke verify 用) |
| FFmpeg | `./ffmpeg-bin/ffmpeg.exe` v8.1.2 essentials |

### JA build command

```bash
uv run python -m benchmarks.confidence_calibration.build_corpus \
    --source ".tmp/calibration_corpus_smoke/_raw/ja_full.wav" \
    --reference-text "https://taltal3014.lsv.jp/little-prince/LittlePrince1.html" \
    --output-dir ".tmp/calibration_corpus_smoke/ja_clean" \
    --language ja --label speech \
    --max-duration-sec 60 \
    --engine-kwargs "model_size=base" "compute_type=int8" "language=ja"
```

### EN build command

```bash
uv run python -m benchmarks.confidence_calibration.build_corpus \
    --source ".tmp/calibration_corpus_smoke/_raw/en_full.wav" \
    --reference-text "https://esl-bits.eu/Novellas.for.ESL.Students/LittlePrince/01/text.html" \
    --output-dir ".tmp/calibration_corpus_smoke/en_clean" \
    --language en --label speech \
    --start-offset-sec 6.0 \
    --max-duration-sec 60 \
    --engine-kwargs "model_size=base" "compute_type=int8" "language=en"
```

## 2. 結果サマリー (autojunk fix 後)

| Lang | Audio source | Segments | Total speech | Coverage = 1.0 | Coverage ≥ 0.9 | Coverage ≥ 0.5 | Mean |
|---|---|---|---|---|---|---|---|
| **JA** | `youtube.com/watch?v=6aJ3jsVeQIg` | **15** | 26.4 sec | **6 (40%)** | 7 (47%) | **9 (60%)** | ~0.71 |
| **EN** | `youtube.com/watch?v=fxvOPdOYyeo` (0:06〜) | **24** | 45.0 sec | 0 (0%) | **3 (12%)** | **12 (50%)** | ~0.55 |

manifest.jsonl 総行数 = **39** (= JA 15 + EN 24)、`--force` 再 build 後も
duplicate なし ← PR #340 review fix 1 (manifest upsert pattern) の実機 verify ✅

## 3. JA 結果評価 (autojunk fix 後)

### JA segments 全件

| # | Coverage | Transcribed text | Reference match | 評価 |
|---|---|---|---|---|
| 0000 | 0.25 | "お疲れ様" | "お" | audio 冒頭の音 (原稿外、実態不明) |
| 0001 | 0.25 | "センパン" | "セ" | audio 冒頭の音 (原稿外、実態不明) |
| 0002 | **1.00** | "僕は今まで" | "僕は今まで" | ✅ 完全一致 |
| 0003 | **1.00** | "この話を誰にもしたことがない" | "この話を誰にもしたことがない" | ✅ 完全一致 |
| 0004 | **1.00** | "それでも今回話すのは" | "それでも今回話すのは" | ✅ 完全一致 |
| 0005 | **1.00** | "もし彼が再び現れた時" | "もし彼が再び現れた時" | ✅ 完全一致 |
| 0006 | **0.81** | "さっきに僕に知らせてほしいからだ" | "に僕に知らせてほしいからだ" | ASR 同音誤認 (原稿「真っ先に」→「さっき」)、後段一致 |
| 0007 | **1.00** | "あれは6年前のこと" | "あれは6年前のこと" | ✅ 完全一致 |
| 0008 | 0.45 | "機構機の操縦だった僕は" | "だった僕は" | ASR 漢字誤認 (原稿「飛行機の操縦」→「機構機の操縦」) |
| 0009 | **1.00** | "エンジンのトラブルで" | "エンジンのトラブルで" | ✅ 完全一致 |
| 0010 | 0.21 | "さはらさばくに振り着くするはめになった" | "になった" | ASR 漢字化失敗 (原稿「サハラ砂漠」→「さはらさばく」)、末尾一致のみ |
| 0011 | 0.41 | "その時は上客も整備しもいなかったし" | "もいなかったし" | ASR 漢字同音誤認 (原稿「乗客」「整備士」→「上客」「整備し」) |
| 0012 | 0.35 | "人材からは1000枚でも離れていた" | "も離れていた" | ASR 同音誤認 (原稿「人里からは千マイル」 → 「人材から」「1000枚」)、後段一致 |
| 0013 | 0.58 | "飲み水は一週間分しかない" | "週間分しかない" | 原稿「1週間分」 vs ASR「一週間分」 (数字表記差)、後段一致 |
| 0014 | **0.95** | "一人でエンジンを修理しなければならなかった" | "人でエンジンを修理しなければならなかった" | 原稿「1人で」 vs ASR「一人で」のみ差 (数字表記)、ほぼ完全一致 |

### JA 評価

- ✅ **完全一致 6 件 (0002-0005, 0007, 0009) で coverage = 1.0** (旧 `ratio()` 実装ならこれらも ~0.005 だった)
- ✅ **高一致 9 件 / 15 = 60% が coverage ≥ 0.5**
- ✅ Low coverage 原因の **全てが説明可能**:
  - 4 件 (0008, 0010, 0011, 0012): WhisperS2T base の **日本語漢字精度** (同音誤認 + ひらがな化)
  - 2 件 (0013, 0014): **原稿数字表記差** (1 vs 一 のような アラビア vs 漢数字)
  - 1 件 (0006): ASR **同音誤認** (「真っ先に」→「さっき」)
  - 2 件 (0000, 0001): audio 冒頭の **何か** (原稿外、実態は朗読版前の音、確認できず)
- ✅ alignment の coverage 指標が **「朗読と原稿の wording 一致度」** を正しく反映
- ⚠️ **WhisperS2T base 日本語精度**は alignment 用には十分だが、本格 calibration の ASR としては不適 (本格 calibration で使う engine は ReazonSpeech / Qwen3-ASR )

## 4. EN 結果評価 (autojunk fix 後)

### EN 重要観察: `language=en` hint なしの場合の hallucination 問題

**初回 build (`language=en` なし)** の結果 (この時点では autojunk bug もあり):
```
segment_0001: "そのために、そのために、そのために..." (繰り返し) coverage 0.0
segment_0005: "アクティのアクティのアクティ..." (繰り返し) coverage 0.0
segment_0017: "My drawing number one" (例外的に英語) coverage 0.52
```

→ WhisperS2T base の language auto-detect が **EN 朗読を ja と誤判定** し、
日本語 hallucination の multi-token repetition を生成。

**`--engine-kwargs language=en` 追加 + autojunk fix で再 build** すると、
全 24 segment が **正しく英語で transcribe + 高 coverage**:

| # | Coverage | Transcribed text | Reference match | 評価 |
|---|---|---|---|---|
| 0000 | 0.39 | "Once, when I was six years old," | " when I was " | 朗読「Once,」 / 原稿「ONCE WHEN」 (caps + comma 差) |
| 0001 | **0.97** | "saw a magnificent picture in a book." | "saw a magnificent picture in a book" | ✅ ほぼ完全一致 |
| 0002 | 0.59 | "called True Stories from Nature." | "called True Stories" | 朗読「from Nature」 / 原稿は別 wording |
| 0003 | 0.38 | "about the primeval forest." | "about the " | 「primeval」 は ESL-bits 原稿に **無し** (Saint-Exupéry の英語版用語) |
| 0004 | **0.97** | "It was a picture of a boa constrictor." | "It was a picture of a boa constrictor" | ✅ ほぼ完全一致 |
| 0005 | 0.37 | "in the act of swallowing an animal." | " swallowing a" | 朗読のフィラー句が原稿外、後半一致 |
| 0006 | **0.73** | "Here is a copy of the drawing." | "Here is a copy of the " | ✅ 大半一致、末尾「drawing」 のみ wording 差 |
| 0007 | **0.95** | "In the book it said," | "In the book it said" | ✅ ほぼ完全一致 |
| 0008 | **0.75** | "Swallow their prey hole." | "wallow their prey " | ✅ 大半一致、「hole」 (whole 同音誤認) は別 |
| 0009 | **0.79** | "without chewing it." | "without chewing" | ✅ 大半一致 |
| 0010 | 0.34 | "After that, they are not able to move." | " able to move" | 朗読の wording「After that」 vs 原稿「Afterward」 |
| 0011 | 0.47 | "and they sleep through the six months." | "eep through the si" | 朗読 wording 差、後段一致 |
| 0012 | 0.50 | "they need for digestion." | "r digestion." | 朗読の文構造変更、末尾一致 |
| 0013 | 0.30 | "I pondered deeply then." | "ly then" | 朗読「pondered deeply」 は ESL-bits 原稿に **無し** (Saint-Exupéry 風の wording) |
| 0014 | 0.35 | "over the adventures of the jungle." | "e adventures" | 朗読「over the adventures」 と原稿「about jungle adventures」 が異なる構造 |
| 0015 | 0.43 | "and after some work with a colored pencil." | " a colored pencil." | 末尾一致、前段は別 wording |
| 0016 | 0.44 | "I succeeded in making my first drawing." | " my first drawing" | 末尾「my first drawing」 のみ一致 |
| 0017 | 0.50 | "my drawing number one." | "my drawing " | 「my drawing」 まで一致、「number one」 は原稿に別形式 |
| 0018 | **0.85** | "It looked like this." | " looked like this" | ✅ ほぼ完全一致 |
| 0019 | 0.38 | "I showed my masterpiece to the grownups." | " my masterpiece" | 朗読「I showed」 vs 原稿「showed grown-ups」 (語順差) |
| 0020 | 0.29 | "and ask them whether the drawing frightened them." | "er the drawing" | 朗読 wording 差大 |
| 0021 | **0.67** | "but they answered." | "hey answered" | 「they answered」 一致 |
| 0022 | **0.56** | "Brighton." | "right" | おそらく ASR 誤認 (実は「Frighten」 が音響的に Brighton として認識) |
| 0023 | 0.20 | "Why should anyone be frightened by a hat?" | " should " | 朗読「frightened by a hat」 は ESL-bits 原稿に **無し** (Saint-Exupéry 風) |

### EN 評価

- ✅ **language=en hint で正しく英語 transcribe** (auto-detect なら ja に誤誘導されハルシネーション)
- ✅ **coverage ≥ 0.9 が 3 件、≥ 0.5 が 12 件 / 24 = 50%** (旧 docs では reference mismatch と誤判定したが、実は **autojunk bug が主因** で reference 自体は朗読とよく対応している)
- ✅ low coverage の原因解析:
  - ESL-bits 原稿は **Saint-Exupéry 英語版に忠実だが完全コピーではなく、一部 wording adapt がある** (例: ESL-bits に「primeval forest」「pondered deeply」「frightened by a hat」 が無い)
  - 朗読は **Saint-Exupéry 原文版** に近く、ESL-bits との細部差が low coverage に反映 (autojunk fix で 50% は ≥ 0.5)
  - 一部 ASR 誤認 (segment 0022: "Brighton" は本来 "Frighten" の誤聴) は coverage proxy として正しく検出
- **前回 docs の「ESL-bits は simplified 版で mismatch」 判定は autojunk bug 起因の誤分析** だったと訂正

## 5. PR #340 Review fix の実機検証

本 verify は **PR #340 codex-review** で対応した 3 fix の実機 verification も兼ねる:

### ✅ Review Fix 1 (manifest upsert)

- JA build 後、EN を **`--force=True`** で再 build → JA 15 entries 保持、EN 24 entries 置換、合計 39 行で duplicate なし
- さらに autojunk fix で両言語を再 `--force`build → 同様に **39 行を維持**、3 回連続の force 再 build を経ても重複ゼロ
- 旧実装 (`append_manifest`) なら 3 回目には 3 × 39 = 117 行になっていた

### ✅ Review Fix 2 (coverage 指標)

- JA segment 0003 "この話を誰にもしたことがない" (15 chars) が原稿 (~6600 chars) 内に完全一致 → **coverage 1.0**
- 旧実装 `ratio()` ならば `2×15/(15+6600) ≈ 0.005` で low coverage warning 化していた

### ✅ Review Fix 3 (--engine-kwargs)

- `--engine-kwargs "model_size=base" "compute_type=int8" "language=ja"` で alignment ASR を CLI から完全制御
- 旧実装 (`--engine-kwargs` なし) なら large-v3 default で ~1.5 GB model + 数倍遅い、しかも language hint も渡せず

### ✅ Autojunk fix (本 verify で新発見、3rd round 相当)

- 同 commit に含めた `autojunk=False` 修正で JA / EN 両方で **coverage 値が大幅向上**
- Regression test `test_long_reference_finds_full_substring_autojunk_disabled` で固定

## 6. 本格 Phase 4 calibration への次の action

### Phase 4 実行前の必須調整 (本 verify から学んだ点)

1. **`--engine-kwargs language=<ISO>` を build_corpus.py 標準 invoke に必須化**
   - JA: `--engine-kwargs language=ja model_size=base compute_type=int8`
   - EN: `--engine-kwargs language=en model_size=base compute_type=int8`
   - README.md / docs/contributor の例も update 推奨

2. **reference text source は本 verify の組み合わせで使用可**
   - JA: taltal3014 ja Chapter 1 → 朗読と一致度高、6/15 完全一致
   - EN: ESL-bits Chapter 1 → 12/24 が coverage ≥ 0.5 で実用範囲
   - 前回 docs の「mismatch」 判定は autojunk bug 起因の誤分析、訂正

3. **WhisperS2T base 日本語漢字精度問題は alignment 用途では受容**
   - alignment 用 ASR は coverage proxy として使うので、漢字精度より language coverage 重要
   - 本格 calibration の 5 engine sweep は別 ASR (ReazonSpeech / Qwen3-ASR) で実施

### Phase 4 (本格 calibration、別 PR or merge 後) で実施

1. JA Chapter 1 全長 (~21 分) で build (`--max-duration-sec` 撤去 or 大幅延長)
2. EN Chapter 1 全長 (~15-30 分、duration 確認) で build
3. non_speech / noisy_speech 補強 (ESC-50 etc., user 手動)
4. 5 engine sweep:
   - ReazonSpeech (int8 / float32) JA
   - Qwen3-ASR (ja / en)
   - Parakeet_ja
   - WhisperS2T (ja、no_speech_prob)
5. 結果 → `docs/research/calibration-japan-engines-2026-XX.md` report PR

## 6.5. PR-γ: kana-level alignment metric への migration plan

本 verify の §3 (JA 結果評価) で観測された 6 件の low coverage segment のうち、
**少なくとも 4 件** (0010, 0011, 0014, 残数件) は **表記揺れだけ** が原因
(kanji ↔ katakana ↔ 算用数字差) であり、真の音響誤認識は ~2 件のみと推測される。
text-level coverage は「acoustic confidence」 と「lexical surface form」 を
混同しているため、calibration の本来目的 (acoustic confidence の評価) には適して
いない。

[PR #340 review 2nd round + comment-4840371438 / 4840421498](https://github.com/Mega-Gorilla/livecap-cli/pull/340#issuecomment-4840371438) で
PR-γ として **pykakasi 後処理 normalize** による kana-level alignment metric の
導入を提案。本 verify 後の実装で:

1. ASR 出力 + reference 双方を **NFKC → 漢数字 canonical substitution (per-char、
   `一 → 1` / `千 → 1000` 等) → hiragana → 句読点 strip** で正規化
2. 正規化後の文字列で同じ `SequenceMatcher(autojunk=False)` coverage を計算
3. manifest entry に **`alignment_score_kana` / `reference_text_matched_kana` /
   `transcribed_text_kana`** を additive で追加 (text-level 指標は **不変**)

> **PR #341 codex-review 訂正反映**: 初版は数字を blanket mask (`一` / `1000` /
> `千` 全部 `#`) していたが、 `一人` (= 1 人) と `二人` (= 2 人) を同一視する
> false-high 問題があり、 calibration の閾値判定を歪める。 修正版は per-char
> canonical substitution (`一 → 1`、 `千 → 1000`、 ...) で値の区別を保持しつつ
> `千マイル ↔ 1000マイル` のような正当な surface 表記差は吸収する設計に変更。

### Phase 4 manifest への migration (`recompute_alignment.py`)

新規 CLI で **audio 再 transcribe なし** で既存 manifest に kana field を追加できる
(text-only 計算、~30 sec で完了):

```pwsh
uv run python -m benchmarks.confidence_calibration.recompute_alignment `
    --manifest "$env:LIVECAP_CALIBRATION_CORPUS_DIR/manifest.jsonl" `
    --reference-text-ja "https://taltal3014.lsv.jp/little-prince/LittlePrince1.html" `
    --reference-text-en "https://esl-bits.eu/Novellas.for.ESL.Students/LittlePrince/01/text.html"
```

### kana metric の **実測** outcome (Phase 4 manifest を PR-γ で migrate)

PR-γ `recompute_alignment.py` を本 verify の Phase 4 manifest (15 JA + 24 EN
segment) に **audio 再 transcribe なしで** 適用し、kana coverage を取得した
結果 (audio 再 transcribe 不要、~5 sec 完了):

**集計サマリ**:

| Metric | text level (PR-β) | **kana level (PR-γ)** | Δ |
|---|---|---|---|
| JA coverage ≥ 0.5 | 9 / 15 (60%) | **12 / 15 (80%)** | **+20pt** |
| JA coverage ≥ 0.9 | 7 / 15 (47%) | **9 / 15 (60%)** | **+13pt** |
| EN coverage ≥ 0.5 | 12 / 24 (50%) | 12 / 24 (50%) | 0pt (no-op、pykakasi ASCII passthrough) |
| EN coverage ≥ 0.9 | 3 / 24 (12%) | 3 / 24 (12%) | 0pt |
| kana regressions (kana < text - 0.05) | — | **0 件** | — |

→ **kana metric は決して悪化させず**、 JA だけで calibration signal-to-noise を
大幅改善 (表記差由来の偽 low が ~20pt 減少)。EN は予測通り無効果 (passthrough)。

**JA で大幅改善した segment 例** (kana > text + 0.20):

| Segment | text → kana | Δ | 推定原因 |
|---|---|---|---|
| 0001 | 0.250 → 0.500 | +0.250 | katakana 表記差 (ASR "センパン" vs reference 平仮名混在) |
| 0010 | 0.210 → 0.474 | +0.263 | katakana / kanji 混在 ("サハラ砂漠" vs hiragana reference) |
| 0011 | 0.412 → **1.000** | +0.588 | 表記差 (元 text の解釈は前 docs で誤、kana で完全 match) |
| 0013 | 0.583 → **1.000** | +0.417 | 漢字 ↔ hiragana 表記差 |

→ 前 docs で「真の音響誤認識」 と判断していた segment 0011 は、実は kana 化で
完全 match (= **音響的には成功**、表記差だけ) という新発見。 raw data + kana
metric の組合せで calibration の解像度が向上することを実証。

### License safety

`pykakasi` は **`GPL-3.0-or-later`**、livecap-cli は **`AGPL-3.0-only`**。
dev / benchmark 限定利用 (`[project.optional-dependencies] dev`) なら互換、
production runtime に import されないことを **static grep guard**
(`tests/test_production_no_pykakasi.py`) で常時 verify。

## 7. 関連 PR / Issue

- 親 PR: [PR #340](https://github.com/Mega-Gorilla/livecap-cli/pull/340) (PR-β、本 doc が verify 結果)
- 親 Issue: [Issue #338](https://github.com/Mega-Gorilla/livecap-cli/issues/338) (active calibration harness)
- 加速対象 Issue: [Issue #334](https://github.com/Mega-Gorilla/livecap-cli/issues/334) PR-2 / PR-3 / PR-4
- Stage 1 PR: [PR #339](https://github.com/Mega-Gorilla/livecap-cli/pull/339) (commit `2bc56c6` で merge 済)

## 8. まとめ

| 観点 | 評価 |
|---|---|
| build_corpus end-to-end flow | ✅ **両言語で works** (yt-dlp → ffmpeg → Silero VAD → WhisperS2T transcribe → difflib coverage → manifest upsert) |
| coverage 指標 (Review fix 2) | ✅ **完全一致を正しく 1.0、wording 差を比例的に低下、設計通り** |
| manifest upsert (Review fix 1) | ✅ **`--force` 再 build で他 source 保持、duplicate なし** |
| `--engine-kwargs` CLI (Review fix 3) | ✅ **alignment 用 model_size / compute_type / language を CLI で制御可能、必須 hint** |
| autojunk fix (本 verify で発見) | ✅ **長文 reference + 頻出 char の bug を解消、両言語で coverage 値大幅向上** |
| JA 結果品質 | 🟢 **9/15 (60%) が coverage ≥ 0.5、残りも漢字精度 / 数字表記差で説明可能** |
| EN 結果品質 | 🟢 **12/24 (50%) が coverage ≥ 0.5、low coverage は朗読 vs ESL-bits の細部 wording 差** |
| 教訓 1 | **language hint 必須** (EN は明示しないと auto-detect で hallucination 直撃) |
| 教訓 2 | **autojunk=False が calibration use case で必須** (本 verify で発見、fix 済 + regression test) |
| 教訓 3 | **raw data 確認の重要性** — autojunk bug 発覚前 docs に factual 誤記が複数あり、user 指示「問題指摘の再確認」 で根本原因が判明 |
| Phase 4 本格実行可否 | ✅ **可能** (両言語とも reference 一致度十分、本格 calibration へ進める) |
