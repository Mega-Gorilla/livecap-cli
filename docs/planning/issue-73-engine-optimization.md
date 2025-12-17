# Issue #73: Phase 5 エンジン最適化

> **Status**: ✅ COMPLETE (Phase 5A ✅, Phase 5B-1 ✅, Phase 5B-2 ✅, Phase 5B-3 ✅)
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

| Engine | Load Cold(ms) | Load Cached(ms) | VRAM(MB) | Infer(5s, ms) | RTF |
|--------|---------------|-----------------|----------|---------------|-----|
| **Canary** | 67587 | **18052** | 6830 | 402 | 0.080x |
| **Parakeet** | 62528 | **27472** | 2417 | 373 | 0.075x |

> **Note**:
> - **Load Cold**: 初回ロード（モデルダウンロード含む）
> - **Load Cached**: キャッシュ済みモデルのロード（実運用時の値）
> - キャッシュ済みでも 18〜27秒かかるのは NeMo フレームワークの初期化オーバーヘッドが主因
> - Parakeet は VRAM 効率が良好（2.4GB vs Canary の 6.8GB）

##### 他エンジン

| Engine | Status | Note |
|--------|--------|------|
| ReazonSpeech | N/A | `sherpa_onnx` 依存（`libonnxruntime.so` 不足）で環境未対応 |
| Voxtral | Load OK, OOM | ロード成功（VRAM 8923MB）だが推論時に OOM。RTX 4070 Ti（11.6GB）では不足 |

#### 計測結果のまとめ

| エンジン | ロード時間（cached） | VRAM | RTF | 評価 |
|----------|---------------------|------|-----|------|
| WhisperS2T (large-v3-turbo) | 4.9s | N/A※ | **0.040x** | 最速推論、実用最適 |
| WhisperS2T (base) | **0.4s** | N/A※ | 0.055x | 最速ロード |
| Parakeet | **27.5s** | 2417MB | 0.075x | VRAM 効率◎ |
| Canary | **18.1s** | 6830MB | 0.080x | NeMo オーバーヘッド大 |
| Voxtral | 22.1s | 8923MB | OOM | 12GB+ VRAM 推奨 |

> ※ WhisperS2T は CTranslate2 ベースのため torch VRAM 計測対象外
> ※ NeMo エンジンの初回ロードはダウンロード込みで 60秒以上

#### エンジン別改善ポイント（Phase 5B-3 候補）

| エンジン | 改善候補 | 優先度 | 計測結果からの考察 |
|----------|----------|--------|-------------------|
| **WhisperS2T** | バッチサイズ最適化、メモリキャッシュ戦略 | 低 | 既に十分高速（RTF 0.04x, ロード 0.4〜5s） |
| **ReazonSpeech** | 不要なロギング削除、推論パス最適化 | 中 | 環境依存で未計測 |
| **Parakeet** | 初期化の高速化（遅延ロード検討） | 中 | キャッシュ済み 27.5s、NeMo 初期化がボトルネック |
| **Canary** | 初期化の高速化 | 中 | キャッシュ済み 18.1s＋高 VRAM（6.8GB） |
| **Voxtral** | VRAM 最適化または 12GB+ 環境限定 | 低 | 現状 11.6GB では OOM |

> **Note**: NeMo エンジンのロード時間（18〜27秒）は NeMo フレームワーク自体の初期化が主因。
> 大幅な改善には NeMo の遅延インポートや軽量ラッパーの検討が必要だが、投資対効果は要検討。

#### NeMo ロード時間の詳細分析（2025-12-17 追加調査）

##### 時間内訳

| Phase | Canary | Parakeet | 備考 |
|-------|--------|----------|------|
| Import (torch + nemo.collections.asr) | 5.1s | 5.1s | プロセス内で共通・1回のみ |
| `load_model()` | 13.6s | 9.6s | モデルサイズ依存 |
| **合計** | **18.7s** | **14.7s** | |

> **発見**: Import が全体の約27%を占める。`load_model()` が主要ボトルネック。

##### 既存の最適化機能

本リポジトリには以下の最適化機能が**既に実装済み**:

| 機能 | ファイル | 効果 |
|------|----------|------|
| `LibraryPreloader` | `engines/library_preloader.py` | NeMo を**バックグラウンドで事前インポート**（5.1s をユーザー待機時間から除外可能） |
| `ModelMemoryCache` | `engines/model_memory_cache.py` | モデルを**メモリにキャッシュ**し再利用時は `load_model()` スキップ |

##### NeMo 高速化手法（外部調査）

| 手法 | 効果 | 対象 | 参考 |
|------|------|------|------|
| bfloat16 autocasting | 推論10x高速化 | 推論時間 | NeMo 2.0+ で自動適用 |
| CUDA Graphs | カーネル起動オーバーヘッド削減 | 推論時間 | NeMo 2.0+ で利用可能 |
| Label-looping algorithm | RNN-T/TDT 高速デコード | 推論時間 | Parakeet-TDT で採用済み |
| ONNX/TensorRT export | ロード高速化＋推論最適化 | 全体 | 要エクスポート実装 |
| NVIDIA Riva | 本番環境向け最適化 | 全体 | インフラ変更必要 |

参考: [NVIDIA Blog - 10x ASR Speed](https://developer.nvidia.com/blog/accelerating-leaderboard-topping-asr-models-10x-with-nvidia-nemo/)

##### 投資対効果分析

| 最適化オプション | 削減効果 | 実装工数 | ROI | 推奨 |
|-----------------|----------|----------|-----|------|
| **既存 LibraryPreloader 活用** | ~5s (Import) | なし（既存） | **高** | ✅ |
| **既存 ModelMemoryCache 活用** | ~14s (再利用時) | なし（既存） | **高** | ✅ |
| Lazy import 追加 | ~5s | 0.5日 | 中 | △ |
| ONNX export 実装 | 要計測 | 2-3日 | 中 | △ |
| Riva 導入 | 大幅改善 | 1週間+ | 低 | ✗ |

##### 結論

1. **即効性の高い改善**: 既存の `LibraryPreloader` と `ModelMemoryCache` の活用を推奨
   - アプリ起動時に `LibraryPreloader.start_preloading('canary')` を呼び出し
   - ユーザー操作中にバックグラウンドで Import 完了
   - モデル切り替え時もキャッシュから即座に復元

2. **追加投資が必要な改善**: ONNX/TensorRT export は工数対効果を要検討
   - `load_model()` の 9-14 秒は NeMo フレームワーク自体の初期化が主因
   - NeMo の `restore_from()` 内部処理の短縮は困難

3. **Phase 5B-3 の方針**: 大規模な最適化実装は保留し、既存機能の活用を優先

#### Phase 5B-3 実装結果（2025-12-17）

##### 実装内容

| 変更 | ファイル | 効果 |
|------|----------|------|
| `LibraryPreloader` 追加 | `parakeet_engine.py` | Canary と同様にバックグラウンド事前インポート |
| Strong Cache opt-in | `canary_engine.py`, `parakeet_engine.py`, `voxtral_engine.py` | `LIVECAP_ENGINE_STRONG_CACHE=1` で強参照キャッシュ有効化 |

##### 計測結果

環境変数 `LIVECAP_ENGINE_STRONG_CACHE=1` を設定した場合:

| メトリック | 時間 | 備考 |
|-----------|------|------|
| 初回ロード | 11.8s | Import 済みの状態 |
| **再ロード（Strong Cache）** | **22ms** | **667倍高速化** |

> **効果**: 同一プロセス内でエンジンを再作成する場合、`restore_from()` を完全にスキップし、
> キャッシュから即座にモデルを復元。14.7s → 22ms の劇的な改善。

##### 使用方法

```bash
# 強参照キャッシュを有効化
export LIVECAP_ENGINE_STRONG_CACHE=1

# または Python コード内で
import os
os.environ['LIVECAP_ENGINE_STRONG_CACHE'] = '1'
```

> **注意**: 強参照キャッシュは VRAM を保持し続けるため、メモリ制約がある環境では注意が必要。

##### 制限事項

- **Voxtral**: `(model, processor)` の tuple は `weakref` 不可のため、環境変数に関わらず常に強参照でキャッシュされます。
  これは `ModelMemoryCache.set()` のフォールバック動作（L89-92）によるものです。

---

## 4. 受け入れ基準

### Phase 5A 完了条件 ✅

- [x] `base_engine.py` から `register_fallbacks()` ブロック削除 — PR #194
- [x] `get_status_message()` メソッド削除 — PR #194
- [x] `report_progress()` から `phase` パラメータ削除 — PR #194
- [x] 全エンジンが新しい `report_progress()` シグネチャに対応 — PR #194
- [x] `model_loading_phases.py` 削除 — PR #194
- [x] 全テストが通る（233 passed）

### Phase 5B-1 完了条件（コードクリーンアップ）✅

- [x] `check_nemo_availability()` を共通モジュールに抽出 — PR #196
- [x] 重複インポートの整理 — PR #196
- [x] ログモジュールの統一 (`logger = logging.getLogger(__name__)`) — PR #196
- [x] 不要コードの削除（ReazonSpeech GPU クリーンアップ） — PR #196
- [x] 全テストが通る（233 passed）

### Phase 5B-2 完了条件（ベースライン計測）✅

- [x] 全エンジンのベースライン計測データが記録されている — PR #196
- [x] Cold/Cached ロード時間の区別が明確 — 本コミット
- [x] 計測方法の制限事項が文書化されている（CTranslate2 VRAM 等）

### Phase 5B-3 完了条件（最適化実装）✅

- [x] `LibraryPreloader` を Parakeet に追加（Canary と同様）
- [x] Strong Cache opt-in 実装（`LIVECAP_ENGINE_STRONG_CACHE` 環境変数）
- [x] 再ロード時間の劇的改善を確認（14.7s → 22ms）
- [x] 全テストが通る（233 passed）

> **結果**: 同一プロセス内での再ロードが 667倍高速化。
> 初回ロードの短縮は NeMo フレームワーク依存のため限界あり。

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

### Step 3: Phase 5B-1 実装 ✅

1. ✅ ブランチ作成: `refactor/issue-73-phase5b-code-cleanup`
2. ✅ `nemo_utils.py` 作成（共通関数抽出）
3. ✅ インポート整理、ログ統一
4. ✅ 不要コード削除
5. ✅ テスト実行（233 passed）
6. ✅ PR #196 作成・レビュー・マージ

### Step 4: Phase 5B-2 実装 ✅

1. ✅ 全エンジンのベースライン計測
2. ✅ Cold/Cached ロード時間の計測・記録
3. ✅ 計測結果を計画ドキュメントに記録
4. ✅ 計測制限事項の文書化（CTranslate2 VRAM 等）

### Step 5: Phase 5B-3 実装 ✅

1. ✅ 改善対象エンジンの選定（投資対効果の検討）
2. ✅ 改善実装（LibraryPreloader 追加、Strong Cache opt-in）
3. ✅ 改善後の計測・ベースラインとの比較（再ロード 14.7s → 22ms）
4. ✅ テスト実行（233 passed）
5. ✅ PR 作成・レビュー — PR #197

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
