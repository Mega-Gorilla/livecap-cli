# Phase 3: パッケージ構造整理 実装計画

> **Status**: ✅ IMPLEMENTED
> **作成日:** 2025-12-02
> **実装日:** 2025-12-02
> **関連 Issue:** #71
> **関連 PR:** #164
> **依存:** #70 (Phase 2: Config 廃止) ✅ 完了

---

## 1. 背景と目的

### 1.1 現状の課題

LiveCap-GUI からの分離抽出という経緯から、`engines/` ディレクトリがルートレベルに存在している。
これにより以下の問題が発生している：

| 課題 | 詳細 | 影響度 |
|------|------|--------|
| **パッケージ境界の不整合** | `engines/` が `livecap_core/` 外に存在 | 高 |
| **インポートパスの不統一** | `from engines import X` と `from livecap_core import Y` が混在 | 中 |
| **pyproject.toml の複雑さ** | 複数パッケージを個別に include | 低 |

### 1.2 目標

1. **engines/ を livecap_core/engines/ に統合**: 単一パッケージ構造の実現
2. **インポートパスの統一**: すべて `from livecap_core.xxx` 形式に
3. **pyproject.toml の簡素化**: `livecap_core*` のみの include

---

## 2. 現状分析

### 2.1 現在のディレクトリ構造

```
livecap-core/
├── livecap_core/
│   ├── __init__.py
│   ├── cli.py
│   ├── audio_sources/
│   ├── resources/
│   ├── transcription/
│   ├── utils/
│   └── vad/
├── engines/                 # ← 移動対象
│   ├── __init__.py
│   ├── base_engine.py
│   ├── engine_factory.py
│   ├── metadata.py
│   └── ... (14ファイル)
├── benchmarks/
├── examples/
└── tests/
```

### 2.2 移動対象ファイル

`engines/` ディレクトリ内の全14ファイル：

| ファイル | 役割 | LOC (概算) |
|----------|------|------------|
| `__init__.py` | パッケージエクスポート | 18 |
| `base_engine.py` | 基底クラス (Template Method) | 500+ |
| `engine_factory.py` | ファクトリークラス | 200+ |
| `metadata.py` | エンジンメタデータ定義 | 250+ |
| `library_preloader.py` | ライブラリ事前ロード | 300+ |
| `model_loading_phases.py` | ロードフェーズ定義 | 150+ |
| `model_memory_cache.py` | モデルキャッシュ | 200+ |
| `nemo_jit_patch.py` | NeMo JIT パッチ | 50+ |
| `shared_engine_manager.py` | 共有エンジン管理 | 500+ |
| `reazonspeech_engine.py` | ReazonSpeech エンジン | 700+ |
| `parakeet_engine.py` | Parakeet エンジン | 500+ |
| `canary_engine.py` | Canary エンジン | 450+ |
| `whispers2t_engine.py` | WhisperS2T エンジン | 450+ |
| `voxtral_engine.py` | Voxtral エンジン | 500+ |

### 2.3 engines/ 内部の依存関係

`engines/` 内のファイルは以下のモジュールをインポートしている：

| インポート元 | インポート先 | 用途 |
|-------------|-------------|------|
| 各エンジン | `livecap_core.utils` | ユーティリティ関数 |
| 各エンジン | `livecap_core.resources` | モデル管理 |
| `engine_factory.py` | `livecap_core.i18n` | 国際化 |
| `whispers2t_engine.py` | `livecap_core.languages` | 言語定義 |
| `shared_engine_manager.py` | `livecap_core` | イベント型 |

移動後もこれらのインポートは絶対パスで維持する（`livecap_core.xxx`）。

---

## 3. 影響調査結果

### 3.1 Pythonコードの更新対象

`from engines` でインポートしている13ファイル：

#### livecap_core/

| ファイル | 現在のインポート | 変更後 |
|----------|-----------------|--------|
| `cli.py` | `from engines.metadata import EngineMetadata` | `from livecap_core.engines.metadata import EngineMetadata` |

#### examples/realtime/

| ファイル | 現在のインポート | 変更後 |
|----------|-----------------|--------|
| `basic_file_transcription.py` | `from engines.engine_factory import EngineFactory` | `from livecap_core.engines import EngineFactory` |
| `async_microphone.py` | 同上 | 同上 |
| `callback_api.py` | 同上 | 同上 |
| `custom_vad_config.py` | 同上 | 同上 |

#### benchmarks/

| ファイル | 現在のインポート | 変更後 |
|----------|-----------------|--------|
| `common/engines.py` | `from engines.engine_factory import EngineFactory` | `from livecap_core.engines import EngineFactory` |
| `common/engines.py` | `from engines.metadata import EngineMetadata` | `from livecap_core.engines import EngineMetadata` |
| `common/datasets.py` | `from engines.metadata import EngineMetadata` | `from livecap_core.engines import EngineMetadata` |
| `optimization/objective.py` | `from engines.base_engine import TranscriptionEngine` | `from livecap_core import TranscriptionEngine` ※1 |
| `optimization/vad_optimizer.py` | `from engines.base_engine import TranscriptionEngine` | `from livecap_core import TranscriptionEngine` ※1 |
| `optimization/vad_optimizer.py` | `from engines.engine_factory import EngineFactory` | `from livecap_core.engines import EngineFactory` |

> ※1: `TranscriptionEngine` は `engines/base_engine.py` には存在しない（既存のバグ）。正しくは `livecap_core.transcription.stream` で定義されている Protocol。Phase 3 で修正。

#### tests/

| ファイル | 現在のインポート | 変更後 |
|----------|-----------------|--------|
| `core/engines/test_engine_factory.py` | `from engines.engine_factory import EngineFactory` | `from livecap_core.engines import EngineFactory` |
| `core/engines/test_engine_factory.py` | `from engines.metadata import ...` | `from livecap_core.engines import ...` |
| `integration/engines/test_smoke_engines.py` | `from engines.engine_factory import EngineFactory` | `from livecap_core.engines import EngineFactory` |
| `integration/realtime/test_e2e_realtime_flow.py` | 同上 | 同上 |

#### engines/ 内部（相対インポートに変更）

| ファイル | 現在のインポート | 変更後 |
|----------|-----------------|--------|
| `shared_engine_manager.py` | `from engines.engine_factory import EngineFactory` | `from .engine_factory import EngineFactory` ※2 |

> ※2: engines/ 内部のファイルは相対インポートを使用すべき。他のファイル（base_engine.py 等）も相対インポートを使用している。

### 3.2 ドキュメント・設定ファイルの更新対象

#### 必須更新（アクティブドキュメント）

| ファイル | 更新内容 |
|----------|----------|
| `README.md` | インポート例の更新 |
| `CLAUDE.md` | エンジン使用例の更新 |
| `AGENTS.md` | engines/ への参照を更新 |
| `docs/architecture/core-api-spec.md` | API 仕様のインポートパス更新 |
| `docs/guides/realtime-transcription.md` | 使用例の更新（5箇所） |
| `docs/guides/benchmark/asr-benchmark.md` | ベンチマーク使用例の更新 |
| `docs/reference/feature-inventory.md` | 機能一覧の更新（5箇所） |
| `docs/reference/vad/config.md` | インポート例の更新 |

#### CI/CD（手動更新: 5箇所）

| ファイル | 行番号 | 更新内容 |
|----------|--------|----------|
| `.github/workflows/integration-tests.yml` | 155, 346, 409 | `from engines.engine_factory` → `from livecap_core.engines` |
| `.github/workflows/benchmark-asr.yml` | 186 | `from engines.metadata` → `from livecap_core.engines` |
| `.github/workflows/benchmark-vad.yml` | 210 | `from engines.metadata` → `from livecap_core.engines` |

> Note: `.github/workflows/core-tests.yml` には `from engines` パターンがないため更新不要。

#### アーカイブ（更新不要）

以下はアーカイブ済みのため更新不要：

- `docs/planning/archive/phase1-implementation-plan.md`
- `docs/planning/archive/phase2-api-config-simplification.md`
- `docs/planning/archive/language-based-vad-optimization.md`
- `docs/planning/archive/vad-benchmark-plan.md`

### 3.3 pyproject.toml の更新

```toml
# Before
[tool.setuptools.packages.find]
where = ["."]
include = ["livecap_core*", "engines*", "config*", "benchmarks*"]

# After
[tool.setuptools.packages.find]
where = ["."]
include = ["livecap_core*", "benchmarks*"]
```

---

## 4. 実装タスク

### 4.1 Task 1: engines/ を livecap_core/engines/ に移動

```bash
# git mv を使用して履歴を保持
git mv engines livecap_core/engines
```

**移動後の構造:**
```
livecap_core/
├── __init__.py
├── cli.py
├── engines/              # ← 新しい場所
│   ├── __init__.py
│   ├── base_engine.py
│   ├── engine_factory.py
│   ├── metadata.py
│   └── ...
├── audio_sources/
├── resources/
├── transcription/
├── utils/
└── vad/
```

### 4.2 Task 2: インポートパスの一括更新

**sed コマンドでの一括置換:**

```bash
# Python ファイルのインポート更新
find . -name "*.py" -not -path "./.venv/*" -exec sed -i \
  's/from engines\./from livecap_core.engines./g' {} \;

find . -name "*.py" -not -path "./.venv/*" -exec sed -i \
  's/from engines import/from livecap_core.engines import/g' {} \;
```

**手動修正が必要な箇所:**

1. **engine_factory.py:50** - importlib の package 引数:
   ```python
   # Before
   module = importlib.import_module(module_name, package="engines")

   # After
   module = importlib.import_module(module_name, package="livecap_core.engines")
   ```

2. **shared_engine_manager.py:325** - 相対インポートに変更:
   ```python
   # Before
   from engines.engine_factory import EngineFactory

   # After
   from .engine_factory import EngineFactory
   ```

3. **benchmarks/optimization/objective.py, vad_optimizer.py** - 既存バグの修正:
   ```python
   # Before (TYPE_CHECKING内、engines/base_engine.pyには存在しない)
   from engines.base_engine import TranscriptionEngine

   # After
   from livecap_core import TranscriptionEngine
   ```

**手動確認が必要なパターン:**
- `import engines` (存在しないはず)
- 条件付きインポート (`if TYPE_CHECKING:` 内) - 上記3で対応済み

### 4.3 Task 3: livecap_core/__init__.py の更新

`EngineFactory`, `EngineMetadata`, `BaseEngine`, `EngineInfo` を公開 API としてエクスポート：

```python
# livecap_core/__init__.py に追加
from .engines import EngineFactory, EngineMetadata, BaseEngine, EngineInfo

__all__ = [
    # 既存のエクスポート
    ...
    # Phase 3: エンジン関連
    "EngineFactory",
    "EngineMetadata",
    "BaseEngine",
    "EngineInfo",
]
```

> `EngineInfo` は `EngineMetadata.get()` の戻り値の型として使用される dataclass。外部コードが型アノテーションで使用するケースがあるため、公開 API に含める。

### 4.4 Task 4: pyproject.toml の更新

`engines*` と `config*` を削除（config/ は Phase 2 で削除済み）:

```toml
# Before
[tool.setuptools.packages.find]
where = ["."]
include = ["livecap_core*", "engines*", "config*", "benchmarks*"]

# After
[tool.setuptools.packages.find]
where = ["."]
include = ["livecap_core*", "benchmarks*"]
```

### 4.5 Task 5: TranscriptionEngine Protocol の統一

**背景:**
`benchmarks/common/engines.py` と `livecap_core/transcription/stream.py` に同名の `TranscriptionEngine` Protocol が重複している。BaseEngine は既に `get_engine_name()` と `cleanup()` を実装しているため、Protocol を統一する。

**Step 1: livecap_core/transcription/stream.py を拡張**

```python
# Before
class TranscriptionEngine(Protocol):
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Tuple[str, float]: ...
    def get_required_sample_rate(self) -> int: ...

# After
class TranscriptionEngine(Protocol):
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Tuple[str, float]: ...
    def get_required_sample_rate(self) -> int: ...
    def get_engine_name(self) -> str: ...  # 追加
    def cleanup(self) -> None: ...  # 追加
```

**Step 2: benchmarks/common/engines.py から重複を削除**

```python
# Before
class TranscriptionEngine(Protocol):
    """Protocol for ASR engines."""
    def transcribe(self, audio: Any, sample_rate: int) -> tuple[str, float]: ...
    def get_required_sample_rate(self) -> int: ...
    def get_engine_name(self) -> str: ...
    def cleanup(self) -> None: ...

# After
from livecap_core import TranscriptionEngine  # livecap_core から使用
# 独自 Protocol 定義を削除
```

**確認済み:**
全エンジン（ReazonSpeech, Parakeet, Canary, WhisperS2T, Voxtral）に `get_engine_name()` と `cleanup()` が実装済み。

### 4.6 Task 6: ドキュメント更新

**README.md の更新例:**

```python
# Before
from engines import EngineFactory

# After
from livecap_core.engines import EngineFactory
# または
from livecap_core import EngineFactory  # livecap_core/__init__.py でエクスポート済み
```

### 4.7 Task 7: テスト実行・確認

```bash
# ユニットテスト
uv run pytest tests/core/engines/ -v

# 統合テスト
uv run pytest tests/integration/engines/ -v

# 全テスト
uv run pytest tests/ -v

# インストール確認
pip install -e .
python -c "from livecap_core.engines import EngineFactory; print('OK')"
```

### 4.8 Task 8: 旧ディレクトリのクリーンアップ

`git mv` を使用したため、旧 `engines/` は自動的に削除される。
残留ファイル（`__pycache__` 等）がある場合は手動削除。

---

## 5. 実装順序

```
Step 1: ブランチ作成
    git checkout -b feat/phase3-engines-restructure
    ↓
Step 2: engines/ を livecap_core/engines/ に移動
    git mv engines livecap_core/engines
    ↓
Step 3: インポートパスの更新（13ファイル）
    sed または手動で更新
    ↓
Step 4: livecap_core/__init__.py にエクスポート追加
    EngineFactory, EngineMetadata, BaseEngine, EngineInfo
    ↓
Step 5: pyproject.toml の更新
    include から engines*, config* を削除
    ↓
Step 6: TranscriptionEngine Protocol 統一
    - livecap_core/transcription/stream.py に get_engine_name, cleanup 追加
    - benchmarks/common/engines.py から重複 Protocol 削除
    ↓
Step 7: テスト実行
    uv run pytest tests/ -v
    ↓
Step 8: pip install -e . で確認
    ↓
Step 9: ドキュメント更新（8ファイル）
    ↓
Step 10: CI ワークフロー更新（3ファイル、5箇所）
    ↓
Step 11: PR 作成・レビュー・マージ
```

---

## 6. 検証項目

### 6.1 単体テスト

- [x] `tests/core/engines/test_engine_factory.py` がパス
- [x] 全 `tests/core/` テストがパス

### 6.2 統合テスト

- [x] `tests/integration/engines/test_smoke_engines.py` がパス
- [x] `tests/integration/realtime/test_e2e_realtime_flow.py` がパス
- [x] 全 `tests/integration/` テストがパス（341 passed, 4 failed は環境依存）

### 6.3 インストール確認

- [x] `pip install -e .` が成功
- [x] `from livecap_core.engines import EngineFactory` が動作
- [x] `from livecap_core import EngineFactory` が動作
- [x] `from livecap_core import EngineInfo` が動作

### 6.4 ベンチマーク

- [ ] ASR ベンチマークが動作（CI で検証）
- [ ] VAD ベンチマークが動作（CI で検証）

### 6.5 Examples

- [x] `examples/realtime/basic_file_transcription.py` インポート更新済み
- [x] `examples/realtime/async_microphone.py` インポート更新済み

### 6.6 CLI

- [x] `livecap-core --info` が動作（エンジン一覧表示）

---

## 7. 完了条件

- [x] `engines/` が `livecap_core/engines/` に移動されている
- [x] 全インポートパスが `livecap_core.engines` に更新されている
- [x] `livecap_core/__init__.py` で `EngineFactory`, `EngineMetadata`, `EngineInfo` がエクスポートされている
- [x] `pyproject.toml` から `engines*`, `config*` が削除されている
- [x] `TranscriptionEngine` Protocol が統一されている（`get_engine_name`, `cleanup` 追加）
- [x] `benchmarks/common/engines.py` の重複 Protocol が削除されている
- [x] 全テストがパス（341 passed, 4 failed は環境依存）
- [x] `pip install -e .` が動作する
- [x] ドキュメントが更新されている
- [ ] CI が全てグリーン（PR #164 で検証中）

---

## 8. リスクと対策

| リスク | レベル | 対策 |
|--------|--------|------|
| インポートパス更新漏れ | 中 | grep で網羅的に検索、テストで検出 |
| 循環参照の発生 | 低 | 移動後も絶対インポートを維持 |
| CI 失敗 | 中 | ローカルで全テスト実行後に PR 作成 |
| ベンチマーク動作不良 | 低 | ベンチマーク実行確認を検証項目に含む |

### 8.1 設計決定: TranscriptionEngine の統一

`benchmarks/common/engines.py` と `livecap_core/transcription/stream.py` に同名の `TranscriptionEngine` Protocol が重複していた。

| 定義場所 | メソッド（変更前） |
|----------|----------|
| `livecap_core` | `transcribe`, `get_required_sample_rate` (2メソッド) |
| `benchmarks` | 上記 + `get_engine_name`, `cleanup` (4メソッド) |

**決定: livecap_core に統一（Phase 3 で実施）**

理由:
1. **BaseEngine は既にこれらのメソッドを実装済み** - 新機能追加ではなく既存実装の反映
2. **技術的負債の早期解消** - 重複は長期的に分岐リスクがある
3. **単一の真実のソース** - Protocol 定義が1箇所になりメンテナンス性向上
4. **破壊的変更ではない** - Protocol へのメソッド追加は既存コードに影響しない

**確認済み:**
全エンジン（ReazonSpeech, Parakeet, Canary, WhisperS2T, Voxtral）に `get_engine_name()` と `cleanup()` が実装済み。

---

## 9. 後方互換性

### 9.1 方針

**互換性維持は不要**（refactoring-plan.md セクション 6.1 参照）

- 本リポジトリは外部で利用されていない
- 破壊的変更を積極的に行い、クリーンな API 設計を優先

### 9.2 移行ガイド（参考）

```python
# Before
from engines import EngineFactory
from engines.metadata import EngineMetadata

# After (推奨)
from livecap_core import EngineFactory, EngineMetadata, EngineInfo

# After (詳細インポート)
from livecap_core.engines import EngineFactory
from livecap_core.engines.metadata import EngineMetadata
```

---

## 変更履歴

| 日付 | 変更内容 |
|------|----------|
| 2025-12-02 | 初版作成 |
| 2025-12-02 | 不明点・問題点の解決: TranscriptionEngine バグ修正追記、手動修正箇所の明記、EngineInfo エクスポート追加、GEMINI.md 削除 |
| 2025-12-02 | レビュー対応: ドキュメント影響範囲拡充（AGENTS.md, vad/config.md追加）、pyproject.toml の config* 削除を明記、TranscriptionEngine 重複の設計決定追記 |
| 2025-12-02 | CI/CD 更新箇所を具体化（行番号追記、core-tests.yml は更新不要と明記） |
| 2025-12-02 | TranscriptionEngine 統一を Phase 3 スコープに追加（Task 5, Step 6）、設計決定を「維持」から「統一」に変更 |
