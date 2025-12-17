# Issue #73: Phase 5 エンジン最適化

> **Status**: 🚧 IN PROGRESS (Phase 5A ✅, Phase 5B 進行中)
> **作成日**: 2025-12-17
> **親 Issue**: #64 [Epic] livecap-cli リファクタリング
> **依存**: #71 [Phase3] パッケージ構造整理（完了）

---

## 1. 概要

BaseEngine の過剰な複雑さを解消し、各エンジン実装を最適化する。

### 1.1 現状の問題

| 問題 | 影響 | 詳細 |
|------|------|------|
| 6段階フェーズ管理 | 複雑さ | `LoadPhase` enum + `ModelLoadingPhases` クラス |
| GUI向け i18n キー | 不要 | `model_init_dialog.*` の fallback 47件 |
| 進捗報告の密結合 | 拡張性 | `report_progress()` が `LoadPhase` に依存 |

### 1.2 対象ファイル

```
livecap_core/engines/
├── base_engine.py              # 387行（主要リファクタリング対象）
├── model_loading_phases.py     # 138行（削除候補）
├── whispers2t_engine.py        # WhisperS2T 実装
├── reazonspeech_engine.py      # ReazonSpeech 実装
├── parakeet_engine.py          # Parakeet 実装
├── canary_engine.py            # Canary 実装
└── voxtral_engine.py           # Voxtral 実装
```

---

## 2. 設計方針

### 2.1 codex-review の分析結果（2025-12-12）

> **重要**: 以下の指摘を計画に反映

1. **API 戻り値は維持**: `transcribe() -> Tuple[str, float]` を変更しない（StreamTranscriber との整合性）
2. **段階的移行**: 一括削除ではなく、依存を外しながら移行
3. **計測指標の明確化**: 「高速化」「効率化」の評価基準を定義

### 2.2 設計原則

```python
# Before: 複雑な6段階フェーズ
def load_model(self):
    phase_info = ModelLoadingPhases.get_phase_info(LoadPhase.CHECK_DEPENDENCIES)
    self.report_progress(phase_info.progress_start, self.get_status_message("checking_dependencies"), LoadPhase.CHECK_DEPENDENCIES)
    self._check_dependencies()
    self.report_progress(phase_info.progress_end, phase=LoadPhase.CHECK_DEPENDENCIES)
    # ... 6段階続く

# After: シンプルなフック型進捗報告
# 既存の set_progress_callback() との互換性を維持
def load_model(self) -> None:
    """モデルをロード（進捗報告は set_progress_callback() で事前設定）"""
    def report(percent: int, message: str = ""):
        if self.progress_callback:
            self.progress_callback(percent, message)
        if message:
            logger.info(f"[{self.engine_name}] [{percent}%] {message}")

    report(0, "Checking dependencies...")
    self._check_dependencies()

    report(10, "Preparing model directory...")
    models_dir = self._prepare_model_directory()

    # ... シンプルな進捗報告（LoadPhase への依存なし）
```

> **Note**: 既存 API との互換性のため `set_progress_callback()` を維持。
> `load_model(progress_callback=...)` 形式は将来の拡張として検討可能。

---

## 3. 実装フェーズ

### Phase 5A: BaseEngine 簡素化

#### 5A-1: i18n キー fallback 削除

**変更内容**:
- `base_engine.py` の `register_fallbacks({...})` ブロック削除（47行）
- `get_status_message()` を削除し、直接文字列を使用
- エンジン固有のメッセージは各エンジンで定義

**影響範囲**:
- `base_engine.py`: `register_fallbacks()` と `get_status_message()` を削除
- **各エンジン実装**: `self.get_status_message(...)` 呼び出しを直接文字列に置換（全エンジンで修正必要）

#### 5A-2: LoadPhase enum 依存の削減

**変更内容**:
- `report_progress()` から `phase` パラメータを削除
- `ModelLoadingPhases.get_phase_by_progress()` 呼び出しを削除
- 進捗報告を `(percent, message)` のみに簡素化

**影響範囲**:
- `base_engine.py`: `report_progress()` シグネチャ変更
- 各エンジン: `report_progress()` 呼び出しの `phase=` 引数削除

#### 5A-3: model_loading_phases.py の削除

**前提条件**:
- 5A-1, 5A-2 完了後、`LoadPhase`/`ModelLoadingPhases` への参照がゼロであること

**変更内容**:
- `livecap_core/engines/model_loading_phases.py` を削除
- `base_engine.py` の import 文を削除

**検証**:
```bash
rg "LoadPhase|ModelLoadingPhases|model_loading_phases" livecap_core/
# 結果が空であることを確認
```

**理由**:
- 内部実装詳細であり、公開 API ではない
- `_deprecated/` への移動は技術的負債を残すため不採用
- 万一必要になれば git history から復元可能

### Phase 5B: エンジン個別最適化

#### 計測指標

| 指標 | 説明 | 計測方法 |
|------|------|----------|
| `load_time_cold` | コールドスタート時のモデルロード時間 | `time.perf_counter()` |
| `load_time_cached` | キャッシュ済みモデルのロード時間 | 同上 |
| `first_inference_latency` | 最初の推論レイテンシ | 同上 |
| `rtf` | Real-Time Factor | `inference_time / audio_duration` |
| `peak_ram_mb` | CPU RAM ピーク使用量 | `tracemalloc`（※1） |
| `peak_vram_mb` | GPU VRAM ピーク使用量 | `torch.cuda.max_memory_allocated()`（※2） |

> **※1**: `tracemalloc` は Python 管理メモリのみ計測。Torch/ONNX 等のネイティブメモリは捕捉できない場合あり。
> **※2**: Torch ベースエンジンのみ対応。非 Torch エンジン（ONNX 等）では skip/NA を許容。

#### ベースライン計測結果 (2025-12-17)

**環境**: RTX 4070 Ti (11.6 GB VRAM), Python 3.11

##### WhisperS2T (GPU)

| Model | Load(ms) | VRAM(MB) | Infer(5s, ms) | RTF |
|-------|----------|----------|---------------|-----|
| tiny | 4161 | N/A | 281 | 0.056x |
| base | 372 | N/A | 276 | 0.055x |
| small | 645 | N/A | 309 | 0.062x |
| **large-v3-turbo** | 4869 | N/A | **201** | **0.040x** |

> **Note**: RTF < 1.0 はリアルタイムより高速。large-v3-turbo が最も高速（RTF 0.040x = 25倍速）
> **VRAM 計測について**: WhisperS2T は CTranslate2 ベースのため、`torch.cuda.max_memory_allocated()` では実 VRAM を計測できない。実際の VRAM 使用量は nvidia-smi 等で別途確認が必要。

##### NeMo エンジン (GPU)

| Engine | Load(ms) | VRAM(MB) | Infer(5s, ms) | RTF |
|--------|----------|----------|---------------|-----|
| **Canary** | 67587 | 6830 | 402 | 0.080x |
| **Parakeet** | 62528 | 2417 | 373 | 0.075x |

> **Note**: 初回ロード時間が長い（60秒以上）。NeMo フレームワークのオーバーヘッドが主因。
> Parakeet は VRAM 効率が良好（2.4GB vs Canary の 6.8GB）。

##### 他エンジン

| Engine | Status | Note |
|--------|--------|------|
| ReazonSpeech | N/A | `sherpa_onnx` 依存（`libonnxruntime.so` 不足）で環境未対応 |
| Voxtral | Load OK, OOM | ロード成功（VRAM 8923MB）だが推論時に OOM。RTX 4070 Ti（11.6GB）では不足 |

#### 計測結果のまとめ

| エンジン | ロード時間 | VRAM | RTF | 評価 |
|----------|------------|------|-----|------|
| WhisperS2T (large-v3-turbo) | 4.9s | N/A※ | **0.040x** | 最速推論、実用最適 |
| WhisperS2T (base) | **0.4s** | N/A※ | 0.055x | 最速ロード |
| Parakeet | 62.5s | 2417MB | 0.075x | VRAM 効率◎ |
| Canary | 67.6s | 6830MB | 0.080x | NeMo オーバーヘッド大 |
| Voxtral | 22.1s | 8923MB | OOM | 12GB+ VRAM 推奨 |

> ※ WhisperS2T は CTranslate2 ベースのため torch VRAM 計測対象外

#### エンジン別改善ポイント

| エンジン | 改善候補 | 優先度 | 計測結果からの考察 |
|----------|----------|--------|-------------------|
| **WhisperS2T** | バッチサイズ最適化、メモリキャッシュ戦略 | 高 | 既に十分高速（RTF 0.04x） |
| **ReazonSpeech** | 不要なロギング削除、推論パス最適化 | 中 | 環境依存で未計測 |
| **Parakeet** | 初期化の高速化（遅延ロード検討） | 中 | 62秒のロード時間改善が課題 |
| **Canary** | 初期化の高速化 | 中 | 67秒のロード＋高 VRAM（6.8GB） |
| **Voxtral** | VRAM 最適化または 12GB+ 環境限定 | 低 | 現状 11.6GB では OOM |

---

## 4. 受け入れ基準

### Phase 5A 完了条件 ✅

- [x] `base_engine.py` から `register_fallbacks()` ブロック削除 — PR #194
- [x] `get_status_message()` メソッド削除 — PR #194
- [x] `report_progress()` から `phase` パラメータ削除 — PR #194
- [x] 全エンジンが新しい `report_progress()` シグネチャに対応 — PR #194
- [x] `model_loading_phases.py` 削除 — PR #194
- [x] 全テストが通る（233 passed）

### Phase 5B 完了条件

- [x] ベースライン計測データが記録されている — PR #196
- [ ] 各エンジンの `load_time_cached` が改善または維持
- [ ] RTF が改善または維持
- [ ] メモリ使用量が悪化していない
- [x] 全テストが通る（233 passed）

---

## 5. 移行手順

### Step 1: 準備 ✅

1. ✅ 計画ドキュメント作成（本ファイル）

### Step 2: Phase 5A 実装 ✅

1. ✅ ブランチ作成: `refactor/issue-73-phase5a-base-engine`
2. ✅ i18n キー fallback 削除（47行）
3. ✅ `get_status_message()` 呼び出しを文字列に置換（全4エンジン）
4. ✅ `report_progress()` の `phase` パラメータ削除
5. ✅ `model_loading_phases.py` 削除（138行）
6. ✅ テスト実行（233 passed）
7. ✅ PR #194 作成・レビュー・マージ

### Step 3: Phase 5B 実装（現在のステップ）

1. ⬜ ブランチ作成: `refactor/issue-73-phase5b-engine-optimization`
2. ⬜ コード分析・改善ポイント特定
3. ⬜ 改善実装
4. ⬜ テスト実行
5. ⬜ PR 作成・レビュー

---

## 6. リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| GUI 側でフェーズ管理に依存 | 高 | GUI リポジトリを確認、必要なら互換レイヤー |
| 進捗報告の削除で UX 低下 | 中 | callback 形式で維持、デフォルトは logger 出力 |
| エンジン最適化で回帰 | 中 | ベースライン計測と比較、全テスト通過を必須に |

---

## 7. 関連リソース

- [refactoring-plan.md](./refactoring-plan.md) - 全体リファクタリング計画
- [Issue #73](https://github.com/Mega-Gorilla/livecap-cli/issues/73) - GitHub Issue
- [Issue #64](https://github.com/Mega-Gorilla/livecap-cli/issues/64) - Epic Issue
