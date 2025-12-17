# Issue #73: Phase 5 エンジン最適化

> **Status**: 📋 PLANNING
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
def load_model(self, progress_callback: Optional[Callable[[int, str], None]] = None) -> None:
    """モデルをロード（オプショナルな進捗報告）"""
    def report(percent: int, message: str = ""):
        if progress_callback:
            progress_callback(percent, message)
        logger.info(f"[{self.engine_name}] [{percent}%] {message}")

    report(0, "Checking dependencies...")
    self._check_dependencies()

    report(10, "Preparing model directory...")
    models_dir = self._prepare_model_directory()

    # ... シンプルな進捗報告
```

---

## 3. 実装フェーズ

### Phase 5A: BaseEngine 簡素化

#### 5A-1: i18n キー fallback 削除

**変更内容**:
- `base_engine.py` の `register_fallbacks({...})` ブロック削除（47行）
- `get_status_message()` を削除し、直接文字列を使用
- エンジン固有のメッセージは各エンジンで定義

**影響範囲**:
- `base_engine.py` のみ
- 各エンジン実装への変更不要（`get_status_message()` 呼び出しを文字列に置換）

#### 5A-2: LoadPhase enum 依存の削減

**変更内容**:
- `report_progress()` から `phase` パラメータを削除
- `ModelLoadingPhases.get_phase_by_progress()` 呼び出しを削除
- 進捗報告を `(percent, message)` のみに簡素化

**影響範囲**:
- `base_engine.py`: `report_progress()` シグネチャ変更
- 各エンジン: `report_progress()` 呼び出しの `phase=` 引数削除

#### 5A-3: model_loading_phases.py の非推奨化

**変更内容**:
- `model_loading_phases.py` を使用箇所がなくなった後に削除
- または `_deprecated/` に移動して段階的に削除

**条件**:
- GUI 側が独自にフェーズ管理を持つか確認
- 外部からの参照がないことを確認

### Phase 5B: エンジン個別最適化

#### 計測指標

| 指標 | 説明 | 計測方法 |
|------|------|----------|
| `load_time_cold` | コールドスタート時のモデルロード時間 | `time.perf_counter()` |
| `load_time_cached` | キャッシュ済みモデルのロード時間 | 同上 |
| `first_inference_latency` | 最初の推論レイテンシ | 同上 |
| `rtf` | Real-Time Factor | `inference_time / audio_duration` |
| `peak_ram_mb` | CPU RAM ピーク使用量 | `tracemalloc` |
| `peak_vram_mb` | GPU VRAM ピーク使用量 | `torch.cuda.max_memory_allocated()` |

#### ベースライン計測

```bash
# 計測コマンド例
uv run pytest tests/integration/engines -m engine_smoke --benchmark
```

#### エンジン別改善ポイント

| エンジン | 改善候補 | 優先度 |
|----------|----------|--------|
| **WhisperS2T** | バッチサイズ最適化、メモリキャッシュ戦略 | 高 |
| **ReazonSpeech** | 不要なロギング削除、推論パス最適化 | 中 |
| **Parakeet** | 初期化の高速化（遅延ロード検討） | 中 |
| **Canary** | 初期化の高速化 | 中 |
| **Voxtral** | 初期化の高速化 | 低 |

---

## 4. 受け入れ基準

### Phase 5A 完了条件

- [ ] `base_engine.py` から `register_fallbacks()` ブロック削除
- [ ] `get_status_message()` メソッド削除
- [ ] `report_progress()` から `phase` パラメータ削除
- [ ] 全エンジンが新しい `report_progress()` シグネチャに対応
- [ ] `model_loading_phases.py` の使用箇所がゼロ
- [ ] 全テストが通る

### Phase 5B 完了条件

- [ ] ベースライン計測データが記録されている
- [ ] 各エンジンの `load_time_cached` が改善または維持
- [ ] RTF が改善または維持
- [ ] メモリ使用量が悪化していない
- [ ] 全テストが通る

---

## 5. 移行手順

### Step 1: 準備（現在のステップ）

1. ✅ 計画ドキュメント作成（本ファイル）
2. ⬜ ベースライン計測の実施
3. ⬜ 計測結果の記録

### Step 2: Phase 5A 実装

1. ブランチ作成: `refactor/issue-73-phase5a-base-engine`
2. i18n キー fallback 削除
3. `get_status_message()` 呼び出しを文字列に置換
4. `report_progress()` の `phase` パラメータ削除
5. 各エンジンの対応修正
6. テスト実行・修正
7. PR 作成・レビュー

### Step 3: Phase 5B 実装

1. ブランチ作成: `refactor/issue-73-phase5b-engine-optimization`
2. 各エンジンの計測
3. ボトルネック特定
4. 改善実装
5. 改善後の計測・比較
6. PR 作成・レビュー

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
