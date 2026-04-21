# NoiseGate A/B Benchmark Results

本ドキュメントは livecap-cli の `NoiseGate` コンポーネントを複数 ASR エンジン × 複数閾値で定量評価した実測記録です。[PR #281](https://github.com/Mega-Gorilla/livecap-cli/pull/281) と [PR #282](https://github.com/Mega-Gorilla/livecap-cli/pull/282) の効果の empirical な根拠であり、`Issue #280` / `Issue #283` の議論を裏付けます。

再現は [`scripts/benchmarks/noise_gate_ab_test.py`](../../scripts/benchmarks/noise_gate_ab_test.py) から可能です (生 JSON は本リポジトリに含まれません)。

## 1. 目的

`NoiseGate` は VAD の前段で環境ノイズを減衰させる音量ベースのゲートです。ナイーブな単一閾値 + soft-mute 実装は whisper 系 ASR エンジンで flicker ハルシネーションを誘発することが判明し、本ベンチマークはその現象を定量化し、改善の妥当性を裏付けるために実施されました。

## 2. テストセットアップ

- **音声ソース**: [livecap-gui の reference audio](https://github.com/Mega-Gorilla/livecap-gui/tree/main/experiments/noise_filter_comparison/test_data)
  - `neko_reference.wav` — クリーン環境、16 kHz mono, 16.09 秒、noise_floor ≈ -55 dB, speech_peak ≈ -30 dB
  - `neko_reference_noisy.wav` — ノイズ環境、15.94 秒、noise_floor ≈ -33 dB, speech_peak ≈ -12 dB
- **リファレンステキスト**: 『吾輩は猫である』冒頭 (両音声で共通)
- **ASR エンジン**: whispers2t (base, CPU), reazonspeech, qwen3asr, parakeet_ja
- **VAD**: 各エンジンの `from_language("ja")` 既定 (whispers2t → silero、他 → tenvad)
- **閾値 (noise-gate-threshold)**: `baseline (no gate)`, `-35`, `-25`, `-20`, `-17` dB
- **メトリクス**:
  - `n_entries` — 転記エントリ数
  - `total_chars` — 全エントリの文字数合計 (baseline 比の倍率がハルシネーション bloat の指標)
  - `max_char_run` — 同一文字の連続最大長 (5 以上で単一文字 loop hallucination のサイン)
- **評価対象ファイル**: `neko_reference_noisy.wav` (より厳しい条件、本表示は基本的にこちら)

## 3. PR #281 baseline 結果 (4 engines × 5 thresholds)

`neko_reference_noisy.wav` での実測。PR #281 時点 (PR #279 と同等の NoiseGate: 単一閾値 + `-60 dB` soft-mute)。

> **再現について**: PR #282 マージ以降は、現行 `NoiseGate` の既定値が auto hysteresis + hard-mute に変わっています。**以下の表の値を現行 `main` で直接再現するには、ハーネスに `--gate-mode pre-prb` を渡してください** (`close_threshold_db=threshold_db`, `noise_floor_db=-60` を明示的に使う互換モード)。
>
> Python / torch / numba / whispers2t のバージョン差により char 数は±20% 程度ずれる可能性がありますが、**閾値 -20 dB での暴走 (300+ chars) vs 他閾値の正常 (100 chars 前後) という qualitative な切り分けは再現されます**。

| 閾値 | whispers2t | reazonspeech | qwen3asr | parakeet_ja |
|---|---|---|---|---|
| baseline (no gate) | 6e / 99c | 5e / 87c | 7e / 98c | 2e / 93c |
| -35 dB (default) | 6e / 102c | 5e / 85c | 4e / 92c | 3e / 94c |
| -25 dB | 6e / 105c | 7e / 87c | 6e / 96c | 7e / 98c |
| **-20 dB** | **7e / 423c 🔥** | 6e / 83c | 6e / 93c | 6e / 103c |
| -17 dB (user mic test) | 9e / 103c | 6e / 83c | 6e / 88c | 6e / 102c |

**注記 (e = entries, c = total_chars)**。

### 出力サンプル (whispers2t @ -20 dB)

```text
[131.97-133.89] どうもどうもどうもどうもどうもどうもどうもどうもどうもどうも
                どうもどうもどうもどうもどうもどうもどうもどうもどうもどうも
                ... (×110 回リピート、330 文字)
```

→ whisper の YouTube dataset バイアス (「どうも」等の反復フィラー) が、flicker で断片化された音声から呼び起こされる現象。

## 4. PR #282 follow-up 結果 (whispers2t / parakeet_ja × 5 thresholds)

`neko_reference_noisy.wav` での実測。PR #282 後 (hysteresis + hard-mute 既定)。ハーネスの既定 `--gate-mode post-prb` がこの state を再現します。

| 閾値 | whispers2t (PR #281) | whispers2t (**PR #282**) | Δ | parakeet_ja (PR #281) | parakeet_ja (**PR #282**) | Δ |
|---|---|---|---|---|---|---|
| baseline | 99c | 103c | +4 | 93c | 93c | 0 |
| -35 dB (default) | 102c | 101c | -1 | 94c | 94c | 0 |
| -25 dB | 105c | **316c 🔥** | **+211** (新種 fragmentation) | 98c | 98c | 0 |
| **-20 dB** | **423c 🔥** | **88c ✅** | **-335** (主目標達成) | 103c | **95c ✅** | -8 |
| -17 dB | 103c | **300c 🔥** | **+197** (新種 fragmentation) | 102c | **95c ✅** | -7 |

### 観察

- **whispers2t -20 dB の暴走は解消** (423 → 88 chars、**-79% 削減**)
- **`parakeet_ja` は全 configurations で改善または同等** (他エンジンへの副作用なし)
- **新種の regression**: `whispers2t @ -25 / -17 dB` で fragmentation 発生。hard-mute で silence が clean になった結果、phrase 間の brief pause で gate が閉じ、短フラグメントから whisper が「んんん...」ループを生成。これは `release_ms` 調整で解消可能 (次節)。

## 5. release_ms スイープ結果 (whispers2t × 2 thresholds × 3 release_ms)

Issue #283 の根拠となった検証。`neko_reference_noisy.wav` + whispers2t で `release_ms` を変えた結果 (測定時の既定は `release_ms=30`)。

| 閾値 | release_ms=30 (旧既定) | release_ms=100 | release_ms=200 |
|---|---|---|---|
| -25 dB | 316c 🔥 | **102c ✅** | **102c ✅** |
| -17 dB | 299c 🔥 | **96c ✅** | **86c ✅** |

→ **`release_ms=100`** で fragmentation が完全解消 (baseline 99 chars 同等に回復)。200 ms との差は 0-10 chars のみ。この発見が PR C ([Issue #283](https://github.com/Mega-Gorilla/livecap-cli/issues/283)) の根拠となった。

## 6. PR C 結果 (既定 release_ms=100 の採用後)

PR C で `release_ms` 既定値を `30 → 100` に変更した結果、`--noise-gate-release` を明示指定せずとも aggressive 閾値で fragmentation が発生しなくなった。

| 閾値 | PR #281 | PR #282 | **PR C (現既定)** | 差分 (PR #282 → PR C) |
|---|---|---|---|---|
| baseline (no gate) | 99c | 103c | 103c | 0 |
| -35 dB (default) | 102c | 101c | 101c | 0 |
| -25 dB | 105c | 316c 🔥 | **102c ✅** | **-214** |
| **-20 dB** | **423c 🔥** | 88c ✅ | 98c ✅ | +10 (維持) |
| -17 dB | 103c | 300c 🔥 | **100c ✅** | **-200** |

→ **PR B (hysteresis + hard-mute) と PR C (release_ms=100) の組合せで、ベースラインとほぼ同等 (98-102 chars) の安定を実現**。whisper 系でも aggressive な閾値を安全に使える。

## 7. 主要な発見

### 発見 1: ハルシネーション暴走は whisper 特異現象

4 エンジン中、PR #281 時点で `-20 dB` で 423 chars に bloat したのは **whispers2t のみ**。reazonspeech (K2 framework, CTC decoder) / qwen3asr (Qwen Transformer) / parakeet_ja (NeMo TDT-CTC) はいずれも 83-103 chars の範囲で安定していた。

**原因仮説**: OpenAI Whisper は YouTube ビデオ大規模 dataset で訓練されており、特定のフィラーフレーズ (「ご視聴ありがとうございました」「どうも」等) への強いバイアスを持つ。NoiseGate の soft-mute (×0.001 残留) + flicker が作る音声アーチファクトが、このバイアスのトリガーになる。

### 発見 2: PR #282 で whisper の暴走を解消

`NoiseGate` の既定を「自動ヒステリシス + hard-mute」に変更したところ、whispers2t @ -20 dB の暴走 (423 chars) が 88 chars に改善。目標達成。

### 発見 3: Hard-mute + 短い release_ms が新種 fragmentation を生む

PR #282 の副次効果として、aggressive な閾値 (-25 / -17 dB) では `release_ms=30` が短すぎて phrase 間で gate が閉じ、短フラグメントから whisper が別種の「んんん...」loop ハルシネーションを生成する現象が判明。`release_ms=100` で完全解消。これは `Issue #283` で追跡。

## 8. 再現方法

前提として [livecap-gui の test_data](https://github.com/Mega-Gorilla/livecap-gui/tree/main/experiments/noise_filter_comparison/test_data) を入手する。

```bash
# PR #282 以降の現行挙動 (Section 4 と対応)
uv run python scripts/benchmarks/noise_gate_ab_test.py \
    --test-data-dir /path/to/livecap-gui/experiments/noise_filter_comparison/test_data \
    --engine whispers2t \
    --files neko_reference_noisy.wav \
    --output /tmp/ab_post-prb.json
    # --gate-mode post-prb がデフォルト

# PR #281 時点の挙動を simulate (Section 3 と対応)
# close_threshold_db=threshold_db と noise_floor_db=-60 が明示的に適用される
uv run python scripts/benchmarks/noise_gate_ab_test.py \
    --test-data-dir /path/to/livecap-gui/experiments/noise_filter_comparison/test_data \
    --engine whispers2t \
    --files neko_reference_noisy.wav \
    --gate-mode pre-prb \
    --output /tmp/ab_pre-prb.json

# 別エンジンに切替
uv run python scripts/benchmarks/noise_gate_ab_test.py \
    --test-data-dir /path/to/livecap-gui/experiments/noise_filter_comparison/test_data \
    --engine reazonspeech \
    --files neko_reference_noisy.wav \
    --output /tmp/ab_reazonspeech.json
```

出力 JSON のスキーマ (`gate_mode` フィールドを含む) はスクリプトの docstring を参照。

### 再現性についての注意

- **`--gate-mode pre-prb`** は **PR #282 以降の main で Section 3 相当の qualitative 結果** (whisper @ -20 dB で bloat が発生する事実、baseline が 100 chars 前後であること) を再現します。
- **厳密な char 数の一致** は保証されません — Python / PyTorch / numba / whispers2t などの runtime バージョン、whisper のサンプリング非決定性で ±20% 程度のずれが出ます。
- **本当に PR #281 時点の char 数 (423 等) を byte-identical に再現** する場合は、当時の `main` revision (例: PR #281 マージコミット `2024d50`) を checkout して実行してください。

## 9. 関連

| 資料 | 内容 |
|---|---|
| [Issue #280](https://github.com/Mega-Gorilla/livecap-cli/issues/280) | NoiseGate Tier 1 拡張 (closed) |
| [PR #281](https://github.com/Mega-Gorilla/livecap-cli/pull/281) | C-3 + C-4: キャリブレーション API + `levels --json`/`--duration` |
| [PR #282](https://github.com/Mega-Gorilla/livecap-cli/pull/282) | C-1 + C-2: hysteresis + hard-mute |
| [Issue #283](https://github.com/Mega-Gorilla/livecap-cli/issues/283) | `release_ms` 既定値の再評価 (本ベンチマークが根拠) |
| PR C | `release_ms` 既定を `30 → 100` に変更 (Section 6 の結果を達成) |
| [livecap-gui PR #294](https://github.com/Mega-Gorilla/livecap-gui/pull/294) | 「死のゾーン」の実証 |
| [AGENTS.md § Backward Compatibility Policy](../../AGENTS.md) | pre-1.0 policy (既定値の変更を許容する根拠) |

## 10. ベンチマーク環境

- **OS**: Windows 11 Pro (日本語ロケール / cp932)
- **PyTorch**: `2.9.1+cpu` (CUDA 無効、全エンジンを CPU 推論で比較)
- **whispers2t**: base モデル
- **numba JIT**: 有効 (`@numba.njit(cache=True)`, < 0.1 ms / 100 ms chunk)
- **計測日**: 2026-04-21
