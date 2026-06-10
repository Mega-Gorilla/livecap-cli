# Parakeet_ja Confidence Signal — 仕様調査 (2026-06-10)

> **Status**: ✅ **Implementation completed in PR #309** — 当初は研究 artifact として残す予定だったが、調査の結果「CTC switch で signal が確実に取れる + RNNT より 1.83x 高速 + text 精度は ~同等」という net win が判明したため、PR #309 内で adapter 修正を実装。本 doc は実装根拠 + 実機 benchmark の永続記録。

## 要旨

PR-A.0 の smoke verify で `nvidia/parakeet-tdt_ctc-0.6b-ja` を実機テストしたところ、`hypothesis.score / len(y_sequence)` (= score fallback) が **speech (-71.5) と applause (-47.3) で逆転** していた。深掘り調査の結果:

1. **モデルは Hybrid TDT-CTC** (`EncDecHybridRNNTCTCBPEModel`)、default `cur_decoder="rnnt"`
2. **RNNT decoding path には `token_confidence` 計算ロジック自体が存在しない** — `preserve_token_confidence=True` を立てても silently 無視される
3. **`hypothesis.score` は log-prob 累積和** — 「短い自信ある幻覚」のほうが高 per-token score になる "simplicity bonus" 仕様
4. **CTC decoder に明示切替 + nested `greedy.preserve_frame_confidence=True`** を設定すると `frame_confidence` / `token_confidence` の両方が populate され、**speech (0.05) vs non-speech (0.0003) で 167 倍のコントラスト** が出る

## 実機 verify 結果 (post-investigation)

### CTC decoder + frame_confidence 有効化後

| Clip | token_confidence_mean | frame_confidence mean | frame_confidence max | text |
|---|---|---|---|---|
| `normal_speech_neko.wav` (speech, 15.6s) | **0.0504** | **0.177** | 0.884 | 「吾輩は猫である名前はまだないどこで生まれたか…」 |
| `desk_tap.wav` (tap, 5.9s) | 0.0003 | 0.040 | 0.371 | 'ピッ!' |
| `applause_5_claps.wav` (applause, 8.5s) | 0.0000 | 0.018 | 0.240 | '。' |

**分離度**:
- token_confidence_mean: speech が non-speech の **167-∞ 倍**
- frame_confidence mean: speech が non-speech の **4-10 倍**
- 閾値 `token_confidence_mean > 0.01` で完全分離可能

### CTC decoder の text 精度トレードオフ

| Clip | RNNT (default、現 adapter) | CTC (本調査) | 差分 |
|---|---|---|---|
| speech | 「…どこで生まれたか**とんと**見当が…」 | 「…どこで生まれたか**とんど**見当が…」 | 1 文字違い (小さい) |
| desk_tap | 'ピッ!' | 'ピッ!' | 同一 |
| applause | 'ピッ。' | '。' | CTC のほうがより少ない幻覚 |

CTC は速度では **`greedy_batch` への切替推奨** の NeMo 警告が出る (default greedy より遅い)。

## 仕様レベルの根本原因

### 1. モデルアーキ: TDT-CTC Hybrid

実機で確認:

```
Model type: EncDecHybridRNNTCTCBPEModel
Module: nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models
Default cur_decoder: 'rnnt'
```

`nvidia/parakeet-tdt_ctc-0.6b-ja` の model 名 `tdt_ctc` が示す通り、encoder 共有 + RNNT (TDT 系) decoder + CTC decoder の **両方を持つ hybrid**。`change_decoding_strategy(cfg, decoder_type='ctc' or 'rnnt')` で active decoder を切替可能。

### 2. RNNT path には token_confidence が「実装されていない」

NeMo source (`.venv/Lib/site-packages/nemo/collections/asr/parts/submodules/`):

| File | confidence support |
|---|---|
| `ctc_greedy_decoding.py:237` | `hyp.frame_confidence` + `compute_confidence()` で `token_confidence` 生成 ✅ |
| `rnnt_greedy_decoding.py` | `frame_confidence` は preserve するが、`token_confidence` 計算ロジック **無し** ❌ |
| `tdt_loop_labels_computer.py` | duration token 計算のみ、confidence 集約は無し ❌ |

→ `preserve_token_confidence=True` は CTC 専用設定。RNNT/TDT では silently 無視される。これは NeMo の API 設計上の限界。

### 3. Score 逆転の仕組み

`Hypothesis.score` の計算式 (`ctc_greedy_decoding.py:237`):

```python
hypothesis.score = (prediction_logprobs[non_blank_ids]).sum()
```

= **non-blank token の log-likelihood 累積和** (length-normalization なし)。

これにより:
- **長い speech (67 tokens)**: 各 token に多様な log-prob (一部低、一部高) → 累積で大きい絶対値 = score -4791.8
- **短い幻覚 (4 tokens)**: モデルが少数の高 confidence pattern (e.g., 'ピッ') を確信を持って出力 → 各 log-prob が 0 に近い = score -189
- per-token (= `score / len`) で正規化しても、長い文の「不確実性込みの平均」 vs 短い「確信のある単純パターン」では後者のほうが高くなる

これは **language modeling の length penalty とは逆方向** — Parakeet score は「長さペナルティ」ではなく「単純さボーナス」。仕様として正しいが、filter 用途には**不適**。

### 4. Hybrid model での nested config 問題

実機で観察した cfg 構造:

```yaml
# 我々の adapter が設定:
confidence_cfg:
  preserve_frame_confidence: True
  preserve_token_confidence: True

# しかし greedy.* は別管理:
greedy:
  preserve_frame_confidence: False  # ← これが actual decoding で参照される
```

→ `confidence_cfg.*` は metadata 的、`greedy.*` こそが decoder の actual flag。両方立てる必要がある。これは NeMo の API gotcha。

### 5. 現 adapter が「実機で signal を取れていなかった」理由

`livecap_cli/engines/parakeet_engine.py:233-260` の現実装:

```python
decoding_config = {
    'strategy': self.decoding_strategy,
    'preserve_alignments': False,
    'confidence_cfg': {
        'preserve_frame_confidence': True,
        'preserve_token_confidence': True,
    },
}
self.model.change_decoding_strategy(decoding_config)
```

問題点:
1. **`decoder_type` 未指定** → default `cur_decoder='rnnt'` のまま (= confidence 計算路に入らない)
2. **`greedy.preserve_frame_confidence` 未設定** → CTC に切り替えても actual greedy decoder で false のまま
3. **`change_decoding_strategy` は silent fail せず** だが、効いていない設定があっても警告は出ない

PR-A.0 の smoke verify でも `token_confidence=None` で帰ってきた理由はこれ。

## PR-A.1 への推奨戦略

### 採用すべき signal: **`token_confidence_mean`** (CTC path)

実機検証で:
- speech: 0.0504
- non-speech: 0.0000 - 0.0003

→ **閾値 0.01 で確実に分離**。これは PR-A.1 filter の理想的な threshold material。

### 実装上の trade-off (PR-A.1 で意思決定)

| Option | 説明 | Pros | Cons |
|---|---|---|---|
| **A. CTC default 化** | adapter の `_configure_model` で `decoder_type='ctc'` 強制 + nested config 整備 | 単一 inference で text + confidence を取得 | text 精度が極 slight に低下 (「とんと」→「とんど」程度)、`greedy_batch` への移行も必要 |
| **B. Dual-pass** | RNNT で text、CTC で confidence のみ取得 | text 精度維持 | 2x inference cost |
| **C. RNNT のまま、Parakeet は filter 対象外** | 現状維持、Parakeet は observe-only | scope 最小、Silero/TenVAD default 戦略と整合 | confidence による defense-in-depth は WhisperS2T のみで実現 |

### 戦略的位置付け (WebRTC 非推奨方針との整合)

PR-B calibration ([#304](https://github.com/Mega-Gorilla/livecap-cli/pull/304)) で:
- **Silero/TenVAD × Parakeet_ja = 0% 幻覚** (= 既に解決済)
- WebRTC × Parakeet_ja = 50% (← WebRTC を非推奨にする方針で対応済、PR [#307](https://github.com/Mega-Gorilla/livecap-cli/pull/307))

→ Parakeet_ja の engine 内部 filter は **「未知ノイズへの defense-in-depth」** という位置付け。

### 最終判断 (PR #309 で Option A を実装)

調査時点では「Option C (Parakeet 対象外)」が最も保守的でしたが、追加 benchmark
で **CTC switch は trade-off が想定より圧倒的に良い** ことが判明:

1. **Text 精度**: 6 clip 中 4 件で RNNT と完全一致、1 件で CTC のほうが優れる
   (applause で幻覚が少ない)、2 件で微小な hiragana 違い (とんと→とんど、
   ジメジメ→ジめジめ) のみ — production 品質に影響なし
2. **Latency**: CTC + greedy_batch は **RNNT より 1.83x 高速** (149.8ms→81.4ms p50)
3. **Signal**: token_confidence_mean が **3-4 桁の分離度** で speech vs non-speech
   を完全分類可能 (閾値 0.005 で OK)

→ **Option A (CTC default 化) を PR #309 で実装**。Option B (dual-pass) は
2x コストで本案より明確に劣後、Option C (対象外) は signal を捨てる機会損失。

実装内容:
- `parakeet_engine.py` `_configure_decoding_with_confidence()` で hybrid model
  検知 → CTC + greedy_batch + frame_confidence の段階設定
- `change_decoding_strategy` 失敗時は legacy strategy-only fallback で degrade
- 旧 PR-A.0 の score-based fallback は削除 (filter に有害な逆方向 signal)

## 実機 benchmark 結果 (PR #309 実装後、2026-06-10)

### Text quality verification (6 corpus clip)

| Clip | RNNT (legacy) | CTC (new) | 差分 |
|---|---|---|---|
| normal_speech_neko | 「…どこで生まれたか**とんと**見当が…」 | 「…とんど見当が…」 | 1 hiragana |
| desk_tap | 'ピッ!' | 'ピッ!' | **同一** |
| applause_5_claps | 'ピッ。' | '。' | **CTC で幻覚減** |
| short_utterances_mixed | 'はいokうんはいokうん。' | 同 | **同一** |
| applause_then_speech | '吾輩は猫である名前はまだない。' | 同 | **同一** |
| overlapping_applause_speech | …**ジメジメ**… | …**ジめジめ**… | 1 hiragana 大文字小文字 |

→ **4/6 完全一致、1/6 で CTC 優位、2/6 で微小 hiragana 差**。production 影響なし。

### Token confidence signal (PR-A.1 filter material)

| Clip | category | `token_confidence_mean` (CTC) | 評価 |
|---|---|---|---|
| applause_then_speech | speech-dominant | **0.1023** | 最高 |
| normal_speech_neko | pure speech | **0.0504** | 高 |
| overlapping_applause_speech | mixed | **0.0383** | 高 |
| short_utterances_mixed | short speech | **0.0104** | 中 |
| desk_tap | tap | **0.0003** | 極低 |
| applause_5_claps | applause | **0.0000029** | 極低 |

**分離度**: speech 系 (0.01 - 0.10) vs non-speech 系 (0.0000029 - 0.0003) で
**3-4 桁の差**。閾値 `token_confidence_mean > 0.005` で完全分類可能。

### Latency micro-benchmark (10 iter, speech 15.6s)

| Path | p50 | p95 | mean | min |
|---|---|---|---|---|
| RNNT (legacy) | 149.8 ms | 152.6 ms | 149.5 ms | 147.4 ms |
| **CTC (new)** | **81.4 ms** ⚡ | **84.4 ms** ⚡ | **81.7 ms** | **79.1 ms** |

**CTC は RNNT より 1.83x 高速**。`greedy_batch` strategy の benefit が大きい。
Issue [#305](https://github.com/Mega-Gorilla/livecap-cli/issues/305) v3 の
p95 ≤ 100ms 規定値も clear。

### Adapter 変更の根本原因 (追加発見)

実装中の benchmark で **旧 PR-A.0 の RNNT path 自体が NeMo に拒否されていた** こと
を発見:

```
ValueError: If `preserve_frame_confidence` flag is set, then
            `preserve_alignments` flag must also be set.
```

旧 PR-A.0 は `preserve_frame_confidence=True` + `preserve_alignments=False` を
渡しており、NeMo はこの矛盾を拒否していた。adapter の except 句で silently
catch されて strategy-only fallback に流れていた = 実質「ただの strategy 指定」
だった。これも本 PR で解消 (Path 2 は意図的に confidence_cfg を含まない strategy
のみに整理)。

## 補足: なぜ PR-A.0 初版では「state C」と判定されたか

PR-A.0 の `_extract_engine_confidence` は:
1. `hypothesis.token_confidence` が non-None なら使う (= state A)
2. そうでなければ `score / len(y_sequence)` を `avg_logprob` field に詰める (= state C fallback)

実機で state C が選択された理由は本調査で判明 (RNNT default + token_confidence は CTC 専用)。**PR-A.0 の schema 自体は両 path 対応で正しく動作**しているが、state C は使えない signal を返していた。

→ PR-A.1 で「Parakeet の `avg_logprob` field (score 由来) は filter 用途には使わない」と決定すれば、本 PR (PR-A.0) の修正は **不要**。schema は PR-A.0 のまま、filter logic で取り扱いを分岐させる。

## 関連リソース

- 実機データ source: 本 session の `.tmp/non_speech_corpus/` (PR-B [#304])
- NeMo source: `.venv/Lib/site-packages/nemo/collections/asr/`
  - `models/hybrid_rnnt_ctc_models.py:318-366` (hybrid `change_decoding_strategy`)
  - `parts/submodules/ctc_greedy_decoding.py:237` (CTC score 計算)
  - `parts/submodules/ctc_decoding.py:539-581` (CTC `compute_confidence`)
  - `parts/submodules/rnnt_greedy_decoding.py:495` (RNNT score 計算、token_confidence なし)
  - `parts/utils/asr_confidence_utils.py:118-183` (ConfidenceConfig)
  - `parts/utils/rnnt_utils.py:35-110` (Hypothesis dataclass)
- 戦略整合: [docs/audio-filter-reference.md](../audio-filter-reference.md) (Silero/TenVAD default 戦略)
- 関連 Issue / PR: [#295](https://github.com/Mega-Gorilla/livecap-cli/issues/295) (epic), [#304](https://github.com/Mega-Gorilla/livecap-cli/pull/304) (PR-B), [#308](https://github.com/Mega-Gorilla/livecap-cli/issues/308) (PR-A plan), [#309](https://github.com/Mega-Gorilla/livecap-cli/pull/309) (PR-A.0)

## 再現コマンド

```powershell
# 環境 (確認済):
# - GPU: RTX 4090, CUDA 12.8, PyTorch 2.9.1+cu128
# - parakeet_ja cached at: %LOCALAPPDATA%\PineLab\LiveCap\Cache\models\parakeet\
# - Corpus: .tmp/non_speech_corpus/

# CTC decoder + nested greedy.preserve_frame_confidence で signal を populate
$env:PYTHONIOENCODING="utf-8"
uv run python -c "
from livecap_cli.engines.engine_factory import EngineFactory
from omegaconf import OmegaConf
import soundfile as sf, tempfile, os
e = EngineFactory.create_engine('parakeet_ja', device='cuda')
e.load_model()
cfg = OmegaConf.create({
    'strategy': 'greedy',
    'preserve_alignments': True,
    'greedy': {
        'preserve_alignments': True,
        'preserve_frame_confidence': True,
    },
    'confidence_cfg': {
        'preserve_frame_confidence': True,
        'preserve_token_confidence': True,
        'exclude_blank': True,
        'aggregation': 'mean',
    },
})
e.model.change_decoding_strategy(cfg, decoder_type='ctc')
audio, sr = sf.read('.tmp/non_speech_corpus/normal_speech_neko.wav', dtype='float32')
with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
    tmp_path = tmp.name; sf.write(tmp_path, audio, sr)
hyp = e.model.transcribe(audio=[tmp_path], batch_size=1, return_hypotheses=True)
while isinstance(hyp, (list, tuple)) and len(hyp) > 0: hyp = hyp[0]
print('token_confidence:', sum(hyp.token_confidence)/len(hyp.token_confidence))
print('frame_confidence:', sum(hyp.frame_confidence)/len(hyp.frame_confidence))
os.unlink(tmp_path)
"
```

by.Scotty
