# WhisperS2T エンジン統合 実装計画

> **Status**: PLANNING
> **作成日:** 2025-12-04
> **関連 Issue:** #165
> **依存:** #71 (Phase 3: パッケージ構造整理) ✅ 完了

---

## 1. 背景と目的

### 1.1 現状の課題

現在 `metadata.py` に5つの別エンジンとして WhisperS2T が定義されている。
しかし `WhisperS2TEngine` は既に `model_size` パラメータで任意のモデルを指定可能な実装になっている。

| 課題 | 詳細 | 影響度 |
|------|------|--------|
| **冗長なメタデータ定義** | 同じクラスが5つのエントリとして定義 | 中 |
| **モデル追加の手間** | 新モデル追加時に新エントリが必要 | 中 |
| **一貫性の欠如** | 他のエンジンはパラメータで切り替え可能 | 低 |
| **compute_type の最適化不足** | CPU で `float32` 使用（`int8` が1.5倍高速） | 中 |

### 1.2 目標

1. **5つのエントリを1つに統合**: `whispers2t` + `model_size` パラメータ
2. **新モデルの追加**: large-v1, large-v2, large-v3-turbo, distil-large-v3
3. **compute_type パラメータ追加**: デフォルト `auto` でデバイス最適化

---

## 2. 現状分析

### 2.1 現在の metadata.py 定義

```python
"whispers2t_tiny": EngineInfo(id="whispers2t_tiny", default_params={"model_size": "tiny", ...}),
"whispers2t_base": EngineInfo(id="whispers2t_base", default_params={"model_size": "base", ...}),
"whispers2t_small": EngineInfo(id="whispers2t_small", default_params={"model_size": "small", ...}),
"whispers2t_medium": EngineInfo(id="whispers2t_medium", default_params={"model_size": "medium", ...}),
"whispers2t_large_v3": EngineInfo(id="whispers2t_large_v3", default_params={"model_size": "large-v3", ...}),
```

### 2.2 現在の whispers2t_engine.py

```python
class WhisperS2TEngine(BaseEngine):
    def __init__(
        self,
        device: Optional[str] = None,
        language: str = "ja",
        model_size: str = "base",  # ← 既にパラメータ化済み
        batch_size: int = 24,
        use_vad: bool = True,
        **kwargs,
    ):
        self.device, self.compute_type = detect_device(device, "WhisperS2T")
        # ...
```

### 2.3 使用箇所の調査結果

| カテゴリ | ファイル数 | 主なパターン |
|----------|-----------|--------------|
| **tests/** | 4 | `whispers2t_base`, `whispers2t_large_v3` |
| **examples/** | 4 | `whispers2t_base`, `startswith("whispers2t_")` |
| **benchmarks/** | 3 | `whispers2t_large_v3`, `startswith("whispers2t_")` |
| **CI** | 1 | `whispers2t_base` |
| **docs/** | 17 | 各種言及 |

---

## 3. 変更概要

### 3.1 エンジンID の変更

| Before | After |
|--------|-------|
| `whispers2t_tiny` | `whispers2t` + `model_size="tiny"` |
| `whispers2t_base` | `whispers2t` + `model_size="base"` (デフォルト) |
| `whispers2t_small` | `whispers2t` + `model_size="small"` |
| `whispers2t_medium` | `whispers2t` + `model_size="medium"` |
| `whispers2t_large_v3` | `whispers2t` + `model_size="large-v3"` |

### 3.2 新規追加モデル

| モデル | サイズ | 特徴 |
|--------|--------|------|
| `large-v1` | 1.55GB | 初代大型モデル |
| `large-v2` | 1.55GB | v1の改良版 |
| `large-v3-turbo` | ~1.6GB | v3ベース、8倍高速 (2024年10月リリース) |
| `distil-large-v3` | ~756MB | v3比1%以内のWERで6倍高速 |

### 3.3 新規追加パラメータ: `compute_type`

CTranslate2 の量子化タイプを制御するパラメータ。

| 値 | 説明 |
|----|------|
| `auto` (デフォルト) | デバイスに応じて最適値を自動選択 |
| `int8` | 整数8bit (CPU推奨、1.5倍高速) |
| `int8_float16` | 混合精度 (GPU高速) |
| `float16` | 半精度浮動小数点 (GPU標準) |
| `float32` | 単精度浮動小数点 (精度重視) |

**自動選択ロジック:**
- CPU: `int8` (float32比で1.5倍高速、メモリ35%削減)
- GPU: `float16` (標準的な精度と速度のバランス)

---

## 4. 実装タスク

### 4.1 Task 1: `EngineInfo` dataclass 拡張

**ファイル:** `livecap_core/engines/metadata.py`

```python
@dataclass
class EngineInfo:
    """エンジン情報"""
    id: str
    display_name: str
    description: str
    supported_languages: List[str]
    requires_download: bool = False
    model_size: Optional[str] = None
    device_support: List[str] = field(default_factory=lambda: ["cpu"])
    streaming: bool = False
    default_params: Dict[str, Any] = field(default_factory=dict)
    module: Optional[str] = None
    class_name: Optional[str] = None
    available_model_sizes: Optional[List[str]] = None  # 追加
```

### 4.2 Task 2: `metadata.py` の WhisperS2T エントリ統合

5つのエントリを1つに統合し、`compute_type` を追加:

```python
"whispers2t": EngineInfo(
    id="whispers2t",
    display_name="WhisperS2T",
    description="Multilingual ASR model with selectable model sizes (tiny to large-v3-turbo)",
    supported_languages=["ja", "en", "zh-CN", "zh-TW", "ko", "de", "fr", "es", "ru", "ar", "pt", "it", "hi"],
    requires_download=True,
    model_size=None,  # 複数サイズ対応のため None
    device_support=["cpu", "cuda"],
    streaming=True,
    module=".whispers2t_engine",
    class_name="WhisperS2TEngine",
    available_model_sizes=[
        # 標準モデル
        "tiny", "base", "small", "medium",
        # 大型モデル
        "large-v1", "large-v2", "large-v3",
        # 高速モデル
        "large-v3-turbo", "distil-large-v3",
    ],
    default_params={
        "model_size": "base",
        "compute_type": "auto",  # NEW: デフォルトは自動最適化
        "batch_size": 24,
        "use_vad": True,
    },
),
```

### 4.3 Task 3: `whispers2t_engine.py` に `compute_type` パラメータ追加

```python
class WhisperS2TEngine(BaseEngine):
    def __init__(
        self,
        device: Optional[str] = None,
        language: str = "ja",
        model_size: str = "base",
        compute_type: str = "auto",  # NEW
        batch_size: int = 24,
        use_vad: bool = True,
        **kwargs,
    ):
        self.device = detect_device(device, "WhisperS2T")  # str のみ受け取る（将来 #166 で対応）
        self.compute_type = self._resolve_compute_type(compute_type)  # NEW
        # ...

    def _resolve_compute_type(self, compute_type: str) -> str:
        """compute_typeを解決（autoの場合はデバイスに応じて最適化）"""
        if compute_type != "auto":
            return compute_type  # ユーザー指定を尊重

        # auto: デバイスに応じた最適値
        # CPU: int8 (1.5x faster than float32, 35% less memory)
        # GPU: float16 (standard precision/speed balance)
        return "int8" if self.device == "cpu" else "float16"
```

**注意:** 現時点では `detect_device()` の戻り値を引き続き受け取るが、`self.compute_type` は `_resolve_compute_type()` で上書きする。#166 で `detect_device()` をリファクタリング後、よりクリーンな実装になる。

### 4.4 Task 4: 使用箇所の更新

#### 4.4.1 tests/

| ファイル | 変更内容 |
|----------|----------|
| `core/engines/test_engine_factory.py` | `whispers2t_base` → `whispers2t` |
| `integration/engines/test_smoke_engines.py` | 各バリエーションを `model_size` パラメータで指定 |
| `integration/realtime/test_e2e_realtime_flow.py` | `whispers2t_base` → `whispers2t` |
| `asr/test_runner.py` | `whispers2t_large_v3` → `whispers2t` + `model_size="large-v3"` |
| `vad/test_runner.py` | 同上 |

#### 4.4.2 examples/

| ファイル | 変更内容 |
|----------|----------|
| `realtime/basic_file_transcription.py` | `whispers2t_base` → `whispers2t` |
| `realtime/async_microphone.py` | 同上 |
| `realtime/callback_api.py` | 同上、`startswith("whispers2t_")` → `== "whispers2t"` |
| `realtime/custom_vad_config.py` | 同上 |

#### 4.4.3 benchmarks/

| ファイル | 変更内容 |
|----------|----------|
| `asr/runner.py` | `whispers2t_large_v3` → `whispers2t` (+ model_size) |
| `vad/runner.py` | 同上 |
| `common/engines.py` | `startswith("whispers2t_")` → `== "whispers2t"` |

#### 4.4.4 CI

| ファイル | 変更内容 |
|----------|----------|
| `.github/workflows/integration-tests.yml` | `whispers2t_base` → `whispers2t` |

#### 4.4.5 core

| ファイル | 変更内容 |
|----------|----------|
| `livecap_core/engines/engine_factory.py` | docstring 更新 |
| `livecap_core/engines/shared_engine_manager.py` | 必要に応じて更新 |

### 4.5 Task 5: ドキュメント更新

| ファイル | 変更内容 |
|----------|----------|
| `README.md` | 新しい使用方法に更新 |
| `CLAUDE.md` | エンジン一覧更新 |
| `docs/guides/realtime-transcription.md` | 使用例更新 |
| `docs/architecture/core-api-spec.md` | API 仕様更新 |
| `docs/reference/feature-inventory.md` | エンジン一覧更新 |

---

## 5. 実装順序

```
Step 1: ブランチ作成
    git checkout -b feat/whispers2t-consolidation
    ↓
Step 2: EngineInfo dataclass に available_model_sizes 追加
    livecap_core/engines/metadata.py
    ↓
Step 3: WhisperS2T エントリ統合 (5→1)
    - 5つのエントリを削除
    - 統合エントリを追加
    ↓
Step 4: whispers2t_engine.py に compute_type 追加
    - compute_type パラメータ追加
    - _resolve_compute_type() メソッド追加
    ↓
Step 5: テストコード更新
    - test_engine_factory.py
    - test_smoke_engines.py
    - test_e2e_realtime_flow.py
    ↓
Step 6: examples 更新 (4ファイル)
    ↓
Step 7: benchmarks 更新 (3ファイル)
    ↓
Step 8: CI ワークフロー更新
    ↓
Step 9: テスト実行
    uv run pytest tests/ -v
    ↓
Step 10: pip install -e . で確認
    ↓
Step 11: ドキュメント更新 (5ファイル)
    ↓
Step 12: PR 作成・レビュー・マージ
```

---

## 6. 新しい使用方法

```python
from livecap_core import EngineFactory, EngineMetadata

# 基本使用（デフォルト: base, compute_type=auto）
engine = EngineFactory.create_engine("whispers2t", device="cuda")

# モデルサイズ指定
engine = EngineFactory.create_engine("whispers2t", device="cuda", model_size="large-v3")
engine = EngineFactory.create_engine("whispers2t", device="cuda", model_size="large-v3-turbo")

# compute_type 明示指定（上級ユーザー向け）
engine = EngineFactory.create_engine("whispers2t", device="cpu", compute_type="int8")
engine = EngineFactory.create_engine("whispers2t", device="cuda", compute_type="int8_float16")
engine = EngineFactory.create_engine("whispers2t", device="cuda", compute_type="float32")  # 精度重視

# 利用可能なモデルサイズの確認
info = EngineMetadata.get("whispers2t")
print(info.available_model_sizes)
# ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large-v3-turbo", "distil-large-v3"]
```

---

## 7. 多言語エンジン判定の変更

```python
# Before
if engine_type.startswith("whispers2t_") or engine_type in ("canary", "voxtral"):
    engine_options["language"] = lang

# After
if engine_type in ("whispers2t", "canary", "voxtral"):
    engine_options["language"] = lang
```

---

## 8. 検証項目

### 8.1 単体テスト

- [ ] `tests/core/engines/test_engine_factory.py` がパス
- [ ] 全 `tests/core/` テストがパス

### 8.2 統合テスト

- [ ] `tests/integration/engines/test_smoke_engines.py` がパス
- [ ] `tests/integration/realtime/test_e2e_realtime_flow.py` がパス
- [ ] 全 `tests/integration/` テストがパス

### 8.3 機能テスト

- [ ] `EngineFactory.create_engine("whispers2t")` が動作
- [ ] `model_size` パラメータで各サイズが指定可能
- [ ] `compute_type="auto"` がデバイスに応じて正しく解決される
- [ ] `compute_type` 明示指定が正しく反映される
- [ ] 新モデル (`large-v3-turbo`, `distil-large-v3`) の動作確認

### 8.4 CLI

- [ ] `livecap-core --info` で `whispers2t` が表示される

### 8.5 Examples

- [ ] 全 examples が正常に動作

### 8.6 CI

- [ ] 全ワークフローがグリーン

---

## 9. 完了条件

- [ ] `metadata.py` の WhisperS2T エントリが1つに統合されている
- [ ] `EngineInfo` に `available_model_sizes` フィールドが追加されている
- [ ] `whispers2t_engine.py` に `compute_type` パラメータが追加されている
- [ ] `_resolve_compute_type()` で自動最適化が実装されている
- [ ] 全使用箇所が更新されている
- [ ] 全テストがパス
- [ ] ドキュメントが更新されている
- [ ] CI が全てグリーン

---

## 10. リスクと対策

| リスク | レベル | 対策 |
|--------|--------|------|
| 使用箇所の更新漏れ | 中 | grep で網羅的に検索、テストで検出 |
| 新モデルの動作不良 | 低 | smoke test で確認 |
| compute_type の誤設定 | 低 | ユニットテストでカバー |
| CI 失敗 | 中 | ローカルで全テスト実行後に PR 作成 |

---

## 11. 関連 Issue

- **#166**: `detect_device()` リファクタリング（本 Issue 完了後に実施）
  - 戻り値を `Tuple[str, str]` → `str` に変更
  - `compute_type` は WhisperS2T 内部で解決するため不要に

---

## 12. 参考資料

- [WhisperS2T GitHub](https://github.com/shashikg/WhisperS2T)
- [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper)
- [CTranslate2 Quantization](https://opennmt.net/CTranslate2/quantization.html)
- [deepdml/faster-whisper-large-v3-turbo-ct2](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2)
- [Systran/faster-distil-whisper-large-v3](https://huggingface.co/Systran/faster-distil-whisper-large-v3)

---

## 変更履歴

| 日付 | 変更内容 |
|------|----------|
| 2025-12-04 | 初版作成 |
