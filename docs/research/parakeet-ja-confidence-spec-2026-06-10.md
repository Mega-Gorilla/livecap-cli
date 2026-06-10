# Parakeet_ja Confidence Signal — 仕様調査 (2026-06-10)

> **Status**: 🔬 Research artifact — PR-A.0 ([#309](https://github.com/Mega-Gorilla/livecap-cli/pull/309)) smoke verify で発覚した「score 逆転現象」の根本原因と、PR-A.1 (engine confidence filter) で実際に使える signal を実機 + source レベルで特定した記録。本 PR 内では adapter 変更を行わず、PR-A.1 着手時の decision doc として参照する。

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

→ Parakeet_ja の engine 内部 filter は **「未知ノイズへの defense-in-depth」** という位置付け。緊急度は低い。

**推奨**: **Option C を PR-A.1 default、Option A を opt-in 拡張として実装可能性を残す**。

具体的には PR-A.1 で:
1. `--confidence-filter` flag は WhisperS2T 専用と明記
2. Parakeet_ja は `engine_confidence.is_available is False` (現状の score fallback ではなく) を返すよう adapter 修正 — fail-open 経由で filter 対象外化
3. Option A (CTC default 化) は **別 issue** として登録、PR-A.1/A.3 完了後の余裕で着手判断

これにより:
- PR-A.1 scope は単純化 (WhisperS2T 1 engine 対象)
- Parakeet_ja の filter 改善は将来の選択肢として温存
- production 動作 (Silero/TenVAD × parakeet_ja の 0% 幻覚) は不変

## 補足: なぜ PR-A.0 では「state C」と判定されたか

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
