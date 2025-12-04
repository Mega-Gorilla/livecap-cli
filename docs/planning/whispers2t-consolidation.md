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
| **言語サポートの制限** | Whisper公式99言語のうち13言語のみ定義 | 中 |

### 1.2 目標

1. **5つのエントリを1つに統合**: `whispers2t` + `model_size` パラメータ
2. **新モデルの追加**: large-v1, large-v2, large-v3-turbo, distil-large-v3
3. **compute_type パラメータ追加**: デフォルト `auto` でデバイス最適化
4. **言語サポート拡張**: Whisper公式99言語を全てサポート

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
| **tests/** | 4 | `whispers2t_base`, `whispers2t_large_v3`, `startswith("whispers2t_")` |
| **examples/** | 4 | `whispers2t_base`, `startswith("whispers2t_")` |
| **benchmarks/** | 3 | `whispers2t_large_v3`, `startswith("whispers2t_")` |
| **CI** | 1 | `whispers2t_base`, `startswith("whispers2t_")` |
| **docs/** | 15 | 各種言及（アーカイブ含む） |
| **livecap_core/** | 2 | `library_preloader.py`, `languages.py` |

### 2.4 WhisperS2T モデル識別子の調査結果 (2025-12-04 調査)

**調査内容:** WhisperS2T ライブラリがモデル識別子をどのように解決するかを確認。

**発見事項:**

1. **`_MODELS` 辞書の制限**
   - WhisperS2T の `hf_utils.py` には以下のショートネームのみ定義:
     ```python
     _MODELS = {
         "tiny.en", "tiny", "base.en", "base", "small.en", "small",
         "medium.en", "medium", "large-v1", "large-v2", "large-v3", "large"
     }
     ```
   - `large-v3-turbo` と `distil-large-v3` は **含まれていない**

2. **HuggingFace パス形式の対応**
   - `download_model()` は `/` を含む識別子を直接 `repo_id` として扱う
   - 例: `deepdml/faster-whisper-large-v3-turbo-ct2` → 直接ダウンロード可能

3. **n_mels パラメータの問題**
   - `load_model()` は `model_identifier == 'large-v3'` の場合のみ `n_mels=128` を設定
   - HuggingFace パス形式では **自動設定されない**
   - v3ベースのモデルは 128 メルバンクが必須（80では Shape mismatch エラー）

**ローカルテスト結果:**

| モデル | n_mels=80 (デフォルト) | n_mels=128 (明示指定) |
|--------|------------------------|------------------------|
| large-v3-turbo | ❌ Shape mismatch | ✅ Success |
| distil-large-v3 | ❌ Shape mismatch | ✅ Success |

**対応方針:**

livecap-core の `whispers2t_engine.py` で以下を実装:
1. ショートネーム → HuggingFace パスのマッピング
2. v3ベースモデルには `n_mels=128` を明示的に設定

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

| モデル | サイズ | 特徴 | CT2モデル | n_mels |
|--------|--------|------|-----------|--------|
| `large-v1` | 1.55GB | 初代大型モデル | ✅ [Systran/faster-whisper-large-v1](https://huggingface.co/Systran/faster-whisper-large-v1) | 80 |
| `large-v2` | 1.55GB | v1の改良版 | ✅ [Systran/faster-whisper-large-v2](https://huggingface.co/Systran/faster-whisper-large-v2) | 80 |
| `large-v3-turbo` | ~1.6GB | v3ベース、8倍高速 (2024年10月) | ✅ [deepdml/faster-whisper-large-v3-turbo-ct2](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2) | **128** |
| `distil-large-v3` | ~756MB | v3比1%以内のWERで6倍高速 | ✅ [Systran/faster-distil-whisper-large-v3](https://huggingface.co/Systran/faster-distil-whisper-large-v3) | **128** |

> **確認済み (2025-12-04)**: すべてのCT2モデルがHuggingFaceに存在し、必要なファイル（config.json, model.bin, tokenizer.json）が揃っていることを確認。

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

### 3.4 言語サポート拡張

Whisper公式の99言語を全てサポート。

**設計方針:**

| 項目 | 方針 |
|------|------|
| **定義場所** | `metadata.py` に `WHISPER_LANGUAGES` を一箇所で定義 |
| **supported_languages** | `WHISPER_LANGUAGES` をそのまま使用（他エンジンと同じパターン） |
| **バリデーション** | 一致しなければ `ValueError`（正規化なし） |
| **エラーメッセージ** | シンプルに `"Unsupported language: {lang}"` |

**責務分離:**

| レイヤー | 責務 |
|---------|------|
| **Languages クラス** | 入力の正規化（"zh-CN" → "zh" 等）、UI表示 |
| **WhisperS2TEngine** | `WHISPER_LANGUAGES` に含まれるか検証するのみ |

### 3.5 入力バリデーション方針

無効な `model_size`、`compute_type`、`language` が指定された場合の挙動：

| パラメータ | 無効値の挙動 | 理由 |
|-----------|-------------|------|
| `model_size` | `ValueError` を送出 | ダウンロード失敗を防ぐため早期検出 |
| `compute_type` | `ValueError` を送出 | ランタイムエラーを防ぐため早期検出 |
| `language` | `ValueError` を送出 | Whisper実行エラーを防ぐため早期検出 |

### 3.6 モデル識別子マッピングと n_mels 設定

WhisperS2T の制限により、livecap-core 側でモデル識別子の変換と n_mels の設定が必要。

**3.6.1 モデルマッピング定数:**

```python
# whispers2t_engine.py に追加
MODEL_MAPPING = {
    # 既存モデル（WhisperS2Tの_MODELSに含まれる）
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large-v1": "large-v1",
    "large-v2": "large-v2",
    "large-v3": "large-v3",
    # 新規追加モデル（HuggingFaceパス形式で指定）
    "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
}

# 128メルバンクが必要なモデル
MODELS_REQUIRING_128_MELS = frozenset({
    "large-v3",
    "large-v3-turbo",
    "distil-large-v3",
})
```

**3.6.2 n_mels 設定ロジック:**

```python
def _get_n_mels(self, model_size: str) -> int:
    """モデルサイズに応じた n_mels 値を取得"""
    return 128 if model_size in MODELS_REQUIRING_128_MELS else 80
```

**3.6.3 load_model 呼び出し:**

```python
# 現在
model = whisper_s2t.load_model(
    model_identifier=self.model_size,
    ...
)

# 変更後
model_identifier = MODEL_MAPPING.get(self.model_size, self.model_size)
n_mels = self._get_n_mels(self.model_size)

model = whisper_s2t.load_model(
    model_identifier=model_identifier,
    n_mels=n_mels,
    ...
)
```

### 3.7 設計決定事項 (2025-12-04 決定)

実装前の不明点を洗い出し、以下の方針で決定。

**3.7.1 engine_name の統一**

| 項目 | Before | After |
|------|--------|-------|
| `self.engine_name` | `f'whispers2t_{model_size}'` | `"whispers2t"` |

**理由:**
- 統合の目的は「5つのエントリを1つにまとめる」こと
- キャッシュキーには `model_size` が含まれるので区別可能
- ユーザー向け表示は `get_engine_name()` で対応

**3.7.2 デフォルト model_size**

| 項目 | 値 | 理由 |
|------|-----|------|
| `default_params["model_size"]` | `"large-v3"` | benchmark 互換性維持（現在 `whispers2t_large_v3` がデフォルト） |

**3.7.3 benchmark / examples の更新方針**

| カテゴリ | Before | After | 備考 |
|----------|--------|-------|------|
| benchmark runner | `whispers2t_large_v3` | `whispers2t` | デフォルト `large-v3` で互換性維持 |
| benchmark tests | `assert "whispers2t_large_v3"` | `assert "whispers2t"` | |
| examples デフォルト | `whispers2t_base` | `whispers2t` | シンプルに統一 |

**3.7.4 追加更新箇所**

| ファイル | 更新内容 |
|---------|----------|
| `engine_factory.py` | docstring のエンジン一覧を更新 |
| `whispers2t_engine.py` | `get_supported_languages()` で `WHISPER_LANGUAGES` を使用 |
| `whispers2t_engine.py` | `CPU_SPEED_ESTIMATES` に主要モデルを追加 |
| `whispers2t_engine.py` | `get_engine_name()` の `size_map` を拡張 |

**3.7.5 CPU_SPEED_ESTIMATES 拡張**

```python
CPU_SPEED_ESTIMATES = {
    'base': '3-5x real-time',
    'large-v3': '0.1-0.3x real-time (VERY SLOW)',
    'large-v3-turbo': '~0.5x real-time',
}
```

**3.7.6 size_map 拡張**

```python
size_map = {
    'tiny': 'Tiny',
    'base': 'Base',
    'small': 'Small',
    'medium': 'Medium',
    'large-v1': 'Large-v1',
    'large-v2': 'Large-v2',
    'large-v3': 'Large-v3',
    'large-v3-turbo': 'Large-v3 Turbo',
    'distil-large-v3': 'Distil Large-v3',
}
```

### 3.8 言語コード変換の調査結果 (2025-12-04 調査)

各ASRエンジンが期待する言語コード形式を調査し、`Languages` クラスとの連携方針を決定。

**3.8.1 各ASRエンジンの言語コード要件**

| エンジン | サポート言語 | 期待形式 | `zh-CN`対応 |
|---------|------------|---------|------------|
| **WhisperS2T** | 99言語 | ISO 639-1 (`zh`) | ❌ KeyError |
| **Canary** | 4言語 | ISO 639-1 (`en`, `de`, `fr`, `es`) | N/A（中国語未サポート） |
| **Voxtral** | 8言語 + auto | ISO 639-1 | N/A（中国語未サポート） |
| **ReazonSpeech** | 1言語 | 固定 (`ja`) | N/A |
| **Parakeet** | 1言語 | 固定 (`en`/`ja`) | N/A |

**結論**: すべてのASRエンジンが **ISO 639-1 の2文字コード** を期待。地域コード（`zh-CN`, `zh-TW`）は直接サポートされない。

**3.8.2 Languages クラスの役割**

```python
# languages.py
"zh-CN": LanguageInfo(
    code="zh-CN",        # UI表示用
    asr_code="zh",       # ASRエンジン用 ← これが重要
)
"zh-TW": LanguageInfo(
    code="zh-TW",
    asr_code="zh",       # 繁体字も簡体字も同じASRコード
)
```

| メソッド | 役割 |
|---------|------|
| `Languages.normalize()` | UI入力の正規化（`"ZH-CN"` → `"zh-CN"`） |
| `Languages.get_info().asr_code` | UI言語コード → ASR言語コードへの変換 |

**3.8.3 WhisperS2T での言語バリデーション方針**

```python
# __init__ での処理
lang_info = Languages.get_info(language)
asr_language = lang_info.asr_code if lang_info else language

if asr_language not in WHISPER_LANGUAGES:
    raise ValueError(f"Unsupported language: {language}")
```

**この方式の利点**:
- `"zh-CN"` → `asr_code="zh"` → バリデーション OK
- `"vi"` → `Languages.get_info()` は None → fallback で `"vi"` → バリデーション OK
- `"invalid"` → fallback → バリデーション NG → `ValueError`

**3.8.4 関連 Issue**

- **#168**: `languages.py` の `supported_engines` 不整合修正
  - Canary/Voxtral が日本語未サポートなのに `ja` の `supported_engines` に含まれている問題
  - 本 Issue (#165) の範囲外として切り出し

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

**4.2.1 `WHISPER_LANGUAGES` をインポート:**

```python
# whisper_languages.py からインポート（Task 5 で作成）
from .whisper_languages import WHISPER_LANGUAGES
```

> **Note:** `WHISPER_LANGUAGES` の定義は Task 5 (`whisper_languages.py`) で行う。
> 実装順序は Step 4 → Step 5 の順で、先に `whisper_languages.py` を作成してからインポートする。

**4.2.2 5つのエントリを1つに統合:**

```python
"whispers2t": EngineInfo(
    id="whispers2t",
    display_name="WhisperS2T",
    description="Multilingual ASR model with selectable model sizes (tiny to large-v3-turbo)",
    supported_languages=WHISPER_LANGUAGES,  # 99言語
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
        "model_size": "large-v3",  # benchmark互換性維持（旧whispers2t_large_v3がデフォルト）
        "compute_type": "auto",
        "batch_size": 24,
        "use_vad": True,
    },
),
```

### 4.3 Task 3: `whispers2t_engine.py` 更新

```python
from .whisper_languages import WHISPER_LANGUAGES_SET  # O(1) lookup 用
from livecap_core.languages import Languages

# モデル識別子マッピング（WhisperS2Tの_MODELSにないモデルはHuggingFaceパスで指定）
MODEL_MAPPING = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large-v1": "large-v1",
    "large-v2": "large-v2",
    "large-v3": "large-v3",
    "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
}

VALID_MODEL_SIZES = frozenset(MODEL_MAPPING.keys())
VALID_COMPUTE_TYPES = frozenset({"auto", "int8", "int8_float16", "float16", "float32"})

# 128メルバンクが必要なモデル（v3ベース）
MODELS_REQUIRING_128_MELS = frozenset({"large-v3", "large-v3-turbo", "distil-large-v3"})

# CPU速度推定値
CPU_SPEED_ESTIMATES = {
    'base': '3-5x real-time',
    'large-v3': '0.1-0.3x real-time (VERY SLOW)',
    'large-v3-turbo': '~0.5x real-time',
}

class WhisperS2TEngine(BaseEngine):
    def __init__(
        self,
        device: Optional[str] = None,
        language: str = "ja",
        model_size: str = "large-v3",  # benchmark互換性維持
        compute_type: str = "auto",
        batch_size: int = 24,
        use_vad: bool = True,
        **kwargs,
    ):
        # engine_name を統一（旧: f'whispers2t_{model_size}'）
        self.engine_name = "whispers2t"

        # 言語コードの変換とバリデーション
        # 1. UI言語コード（zh-CN等）→ ASR言語コード（zh等）への変換
        lang_info = Languages.get_info(language)
        asr_language = lang_info.asr_code if lang_info else language

        # 2. WHISPER_LANGUAGES_SET でバリデーション（O(1) lookup）
        if asr_language not in WHISPER_LANGUAGES_SET:
            raise ValueError(f"Unsupported language: {language}")
        # model_size, compute_type バリデーション
        if model_size not in VALID_MODEL_SIZES:
            raise ValueError(f"Unsupported model_size: {model_size}")
        if compute_type not in VALID_COMPUTE_TYPES:
            raise ValueError(f"Unsupported compute_type: {compute_type}")

        self.language = language  # 元のコードを保持（ログ/デバッグ用）
        self._asr_language = asr_language  # 変換後のコード（transcribe()で使用）
        self.model_size = model_size

        # detect_device() は Tuple[str, str] を返すため、最初の要素のみ使用
        # 注: #166 完了後は戻り値が str になる
        device_result = detect_device(device, "WhisperS2T")
        self.device = device_result[0] if isinstance(device_result, tuple) else device_result

        self.compute_type = self._resolve_compute_type(compute_type)
        # ...

    def _resolve_compute_type(self, compute_type: str) -> str:
        """compute_typeを解決（autoの場合はデバイスに応じて最適化）"""
        if compute_type != "auto":
            return compute_type  # ユーザー指定を尊重
        return "int8" if self.device == "cpu" else "float16"

    def _get_n_mels(self) -> int:
        """モデルサイズに応じた n_mels 値を取得"""
        return 128 if self.model_size in MODELS_REQUIRING_128_MELS else 80

    def _get_model_identifier(self) -> str:
        """モデルサイズを WhisperS2T 用の識別子に変換"""
        return MODEL_MAPPING.get(self.model_size, self.model_size)

    def _load_model_from_path(self, model_path: Path) -> Any:
        """モデルをロード（n_mels を明示的に指定）"""
        import whisper_s2t

        model = whisper_s2t.load_model(
            model_identifier=self._get_model_identifier(),
            backend='CTranslate2',
            device=self.device,
            compute_type=self.compute_type,
            n_mels=self._get_n_mels(),  # v3ベースモデルには128を指定
        )
        return model

    def get_supported_languages(self) -> list:
        """サポートされる言語のリストを取得"""
        return list(WHISPER_LANGUAGES)

    def get_engine_name(self) -> str:
        """エンジン名を取得（ユーザー向け表示用）"""
        size_map = {
            'tiny': 'Tiny', 'base': 'Base', 'small': 'Small', 'medium': 'Medium',
            'large-v1': 'Large-v1', 'large-v2': 'Large-v2', 'large-v3': 'Large-v3',
            'large-v3-turbo': 'Large-v3 Turbo', 'distil-large-v3': 'Distil Large-v3',
        }
        return f"WhisperS2T {size_map.get(self.model_size, self.model_size.title())}"
```

### 4.4 Task 4: `LibraryPreloader` 更新

**ファイル:** `livecap_core/engines/library_preloader.py`

`SharedEngineManager` は `engine_type.split('_')[0]` を渡すため、統合後は `start_preloading("whispers2t")` になる。
現在の `_get_required_libraries` は `whispers2t_base` / `whispers2t_large_v3` 固定のため更新が必要。

```python
# Before (86-96行目)
library_map = {
    'parakeet': {'matplotlib', 'nemo'},
    'parakeet_ja': {'matplotlib', 'nemo'},
    'canary': {'matplotlib', 'nemo'},
    'voxtral': {'transformers'},
    'whispers2t_base': {'whisper_s2t'},      # ← 旧エンジンID
    'whispers2t_large_v3': {'whisper_s2t'},  # ← 旧エンジンID
    'reazonspeech': {'sherpa_onnx'},
}

# After
library_map = {
    'parakeet': {'matplotlib', 'nemo'},
    'parakeet_ja': {'matplotlib', 'nemo'},
    'canary': {'matplotlib', 'nemo'},
    'voxtral': {'transformers'},
    'whispers2t': {'whisper_s2t'},  # ← 統合エンジンID
    'reazonspeech': {'sherpa_onnx'},
}
```

> **⚠️ 既存バグの発見 (2025-12-04):**
> 現在の `whispers2t_engine.py` (行65) は `LibraryPreloader.start_preloading('whispers2t')` を呼び出しているが、
> `library_preloader.py` の `library_map` には `'whispers2t'` エントリが存在しない（`'whispers2t_base'` と `'whispers2t_large_v3'` のみ）。
> 結果として、WhisperS2T の事前ロードが現状で機能していない可能性がある。
> 本タスクで `'whispers2t'` エントリを追加することで修正される。

### 4.5 Task 5: `languages.py` 更新

**ファイル:** `livecap_core/languages.py`

`supported_engines` から `whispers2t_*` を削除し、`whispers2t` に統一:

```python
# Before (各言語で)
supported_engines=["reazonspeech", "whispers2t_base", "whispers2t_tiny",
                   "whispers2t_small", "whispers2t_medium", "whispers2t_large_v3",
                   "canary", "voxtral"],

# After
supported_engines=["reazonspeech", "whispers2t", "canary", "voxtral"],
```

**対象言語:** ja, en, zh-CN, zh-TW, ko, de, fr, es, ru, ar, pt, it, hi, nl（計14言語）

> **注意:** `Languages.get_engines_for_language()` の結果に影響するため、UI/CLIの言語→エンジン対応が正しく動作することを確認すること。

### 4.6 Task 5: WHISPER_LANGUAGES の独立モジュール化と EngineMetadata 連携

**問題:**
- `EngineMetadata.supported_languages` は現在13言語のみ定義
- CLI/API (`get_engine_info()`, `get_engines_for_language()`) で99言語が表示されない
- 例: `get_engines_for_language("vi")` が空リストを返す

**解決策:** `WHISPER_LANGUAGES` を独立モジュールに移動し、循環インポートを避けつつ99言語を `EngineMetadata` で公開

**4.6.1 新規ファイル作成: `livecap_core/engines/whisper_languages.py`**

```python
"""Whisper supported languages (99 languages from OpenAI Whisper tokenizer.py)"""

# ISO 639-1 codes supported by Whisper
WHISPER_LANGUAGES = (
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl",
    "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk",
    "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr",
    "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn",
    "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne",
    "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn",
    "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi",
    "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my",
    "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
    "yue",
)

# Frozen set for O(1) lookup
WHISPER_LANGUAGES_SET = frozenset(WHISPER_LANGUAGES)
```

**4.6.2 metadata.py の更新:**

```python
# Before (Task 4.2 で追加した WHISPER_LANGUAGES をインポートに変更)
WHISPER_LANGUAGES = [...]

# After
from .whisper_languages import WHISPER_LANGUAGES

# EngineInfo の supported_languages で使用
"whispers2t": EngineInfo(
    ...
    supported_languages=list(WHISPER_LANGUAGES),  # 99言語
    ...
)
```

**4.6.3 whispers2t_engine.py の更新:**

```python
# Before (Task 4.3 で metadata.py からインポート)
from .metadata import WHISPER_LANGUAGES

# After
from .whisper_languages import WHISPER_LANGUAGES_SET

# バリデーションで使用
if asr_language not in WHISPER_LANGUAGES_SET:
    raise ValueError(f"Unsupported language: {language}")
```

**4.6.4 影響確認:**

| API | Before | After |
|-----|--------|-------|
| `get_engine_info("whispers2t")["supported_languages"]` | 13言語 | 99言語 |
| `get_engines_for_language("vi")` | `[]` | `["whispers2t"]` |
| `get_engines_for_language("yue")` | `[]` | `["whispers2t"]` |
| CLI `--info` | whispers2t 表示 | whispers2t 表示（変更なし） |

**4.6.5 追加検証項目:**

- [ ] `livecap_core/engines/whisper_languages.py` が作成されている
- [ ] `metadata.py` が `whisper_languages` からインポートしている
- [ ] `whispers2t_engine.py` が `whisper_languages` からインポートしている
- [ ] 循環インポートが発生しない
- [ ] `EngineFactory.get_engine_info("whispers2t")["supported_languages"]` が99言語を含む
- [ ] `EngineFactory.get_engines_for_language("vi")` に `whispers2t` が含まれる

### 4.7 Task 6: 使用箇所の更新

#### 4.7.1 tests/

| ファイル | 変更内容 |
|----------|----------|
| `core/engines/test_engine_factory.py` | `whispers2t_base` → `whispers2t` |
| `core/cli/test_cli.py` | CLI出力の期待値確認・更新 |
| `integration/engines/test_smoke_engines.py` | 各バリエーションを `model_size` パラメータで指定、`startswith` 削除 |
| `integration/realtime/test_e2e_realtime_flow.py` | `whispers2t_base` → `whispers2t`、`startswith` 削除 |
| `benchmark_tests/asr/test_runner.py` | `whispers2t_large_v3` → `whispers2t`（assert文更新） |
| `benchmark_tests/vad/test_runner.py` | 同上 |

**smoke test の model_size 指定方法:**

`EngineSmokeCase` dataclass に `model_size` フィールドを追加:

```python
@dataclass(frozen=True)
class EngineSmokeCase:
    id: str
    engine: str
    language: str
    audio_stem: str
    device: str | None
    requires_gpu: bool = False
    min_vram_gb: float | None = None
    model_size: str | None = None  # 追加: WhisperS2T用

# 使用例
EngineSmokeCase(
    id="whispers2t_cpu_en",
    engine="whispers2t",           # 統合エンジンID
    language="en",
    audio_stem="en/librispeech_1089-134686-0001",
    device="cpu",
    model_size="base",             # model_size をパラメータで指定
),
EngineSmokeCase(
    id="whispers2t_large_v3_gpu_en",
    engine="whispers2t",
    language="en",
    audio_stem="en/librispeech_1089-134686-0001",
    device="cuda",
    requires_gpu=True,
    model_size="large-v3",
),
```

`_build_engine_options()` で `model_size` を処理:

```python
def _build_engine_options(case: EngineSmokeCase) -> dict:
    options = {}
    if case.engine in ("whispers2t", "canary", "voxtral"):
        options["language"] = case.language
    if case.model_size is not None:
        options["model_size"] = case.model_size
    return options
```

**benchmark テスト assert 文の変更例:**
```python
# Before
assert engines == ["parakeet_ja", "whispers2t_large_v3"]

# After (デフォルト model_size="large-v3" により互換性維持)
assert engines == ["parakeet_ja", "whispers2t"]
```

#### 4.7.2 examples/

| ファイル | 変更内容 |
|----------|----------|
| `realtime/basic_file_transcription.py` | `whispers2t_base` → `whispers2t`、`startswith` → `==` |
| `realtime/async_microphone.py` | 同上 |
| `realtime/callback_api.py` | 同上 |
| `realtime/custom_vad_config.py` | 同上 |

#### 4.7.3 benchmarks/

| ファイル | 変更内容 |
|----------|----------|
| `asr/runner.py` | `whispers2t_large_v3` → `whispers2t` (+ model_size) |
| `vad/runner.py` | 同上 |
| `common/engines.py` | `startswith("whispers2t_")` → `== "whispers2t"` (160行目) |

#### 4.7.4 CI

| ファイル | 変更内容 |
|----------|----------|
| `.github/workflows/integration-tests.yml` | `whispers2t_base` → `whispers2t`、`startswith` 削除 (158, 355行目) |

#### 4.7.5 core

| ファイル | 変更内容 |
|----------|----------|
| `livecap_core/engines/engine_factory.py` | docstring 更新 |
| `livecap_core/engines/shared_engine_manager.py` | 必要に応じて更新 |

### 4.8 Task 7: ドキュメント更新

**全文検索で置換・確認が必要なファイル:**

```bash
grep -r "whispers2t_" docs/ --include="*.md" | grep -v "archive/"
```

| ファイル | 変更内容 |
|----------|----------|
| `README.md` | 新しい使用方法に更新 |
| `CLAUDE.md` | エンジン一覧更新 |
| `docs/guides/realtime-transcription.md` | 使用例更新 |
| `docs/guides/benchmark/asr-benchmark.md` | ベンチマーク使用例更新 |
| `docs/guides/benchmark/vad-benchmark.md` | 同上 |
| `docs/guides/benchmark/vad-optimization.md` | 同上 |
| `docs/architecture/core-api-spec.md` | API 仕様更新 |
| `docs/reference/feature-inventory.md` | エンジン一覧更新 |
| `docs/reference/vad/config.md` | 使用例更新 |
| `docs/reference/vad/comparison.md` | エンジン言及更新 |
| `docs/planning/refactoring-plan.md` | 必要に応じて更新 |
| `docs/planning/phase3-package-restructure.md` | 必要に応じて更新 |

> **アーカイブ (`docs/planning/archive/*`)** は更新不要。

---

## 5. 実装順序

```
Step 1: ブランチ作成
    git checkout -b feat/whispers2t-consolidation
    ↓
Step 2: CT2モデル存在確認
    新モデル (large-v1, large-v2, large-v3-turbo, distil-large-v3) の
    CTranslate2変換済みモデルが利用可能か確認
    ↓
Step 3: EngineInfo dataclass に available_model_sizes 追加
    livecap_core/engines/metadata.py
    ↓
Step 4: whisper_languages.py 作成
    livecap_core/engines/whisper_languages.py を新規作成
    WHISPER_LANGUAGES (tuple) と WHISPER_LANGUAGES_SET (frozenset) を定義
    ↓
Step 5: WhisperS2T エントリ統合 (5→1)
    metadata.py から whisper_languages をインポート
    - 5つのエントリを削除
    - 統合エントリを追加（supported_languages=WHISPER_LANGUAGES）
    ↓
Step 6: whispers2t_engine.py 更新
    - whisper_languages から WHISPER_LANGUAGES_SET をインポート
    - language バリデーション追加
    - compute_type パラメータ追加
    - _resolve_compute_type() メソッド追加
    - model_size, compute_type バリデーション追加
    - detect_device() 戻り値のタプル対応
    ↓
Step 7: LibraryPreloader 更新
    library_map に 'whispers2t' エントリ追加
    ↓
Step 8: languages.py 更新
    全14言語の supported_engines を更新
    ↓
Step 9: テストコード更新
    - test_engine_factory.py
    - test_smoke_engines.py
    - test_e2e_realtime_flow.py
    - test_cli.py (CLI出力確認)
    ↓
Step 10: examples 更新 (4ファイル)
    ↓
Step 11: benchmarks 更新 (3ファイル)
    ↓
Step 12: CI ワークフロー更新
    ↓
Step 13: テスト実行
    uv run pytest tests/ -v
    ↓
Step 14: pip install -e . で確認
    ↓
Step 15: ドキュメント更新 (全文検索で漏れなく)
    ↓
Step 16: CLI出力確認
    livecap-core --info で whispers2t が正しく表示されることを確認
    ↓
Step 17: PR 作成・レビュー・マージ
```

---

## 6. 新しい使用方法

```python
from livecap_core import EngineFactory, EngineMetadata

# 基本使用（デフォルト: base, compute_type=auto, language=ja）
engine = EngineFactory.create_engine("whispers2t", device="cuda")

# モデルサイズ指定
engine = EngineFactory.create_engine("whispers2t", device="cuda", model_size="large-v3")
engine = EngineFactory.create_engine("whispers2t", device="cuda", model_size="large-v3-turbo")

# 言語指定（99言語対応）
engine = EngineFactory.create_engine("whispers2t", device="cuda", language="vi")  # Vietnamese
engine = EngineFactory.create_engine("whispers2t", device="cuda", language="th")  # Thai
engine = EngineFactory.create_engine("whispers2t", device="cuda", language="yue") # Cantonese

# compute_type 明示指定（上級ユーザー向け）
engine = EngineFactory.create_engine("whispers2t", device="cpu", compute_type="int8")
engine = EngineFactory.create_engine("whispers2t", device="cuda", compute_type="int8_float16")
engine = EngineFactory.create_engine("whispers2t", device="cuda", compute_type="float32")  # 精度重視

# 利用可能なモデルサイズの確認
info = EngineMetadata.get("whispers2t")
print(info.available_model_sizes)
# ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large-v3-turbo", "distil-large-v3"]

# 対応言語の確認
print(len(info.supported_languages))  # 99
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

### 8.1 事前確認

- [ ] 新モデル (`large-v1`, `large-v2`) の CT2 変換済みモデルが利用可能
- [ ] 新モデル (`large-v3-turbo`, `distil-large-v3`) の CT2 変換済みモデルが利用可能

### 8.2 単体テスト

- [ ] `tests/core/engines/test_engine_factory.py` がパス
- [ ] 全 `tests/core/` テストがパス
- [ ] 入力バリデーションテスト（無効な model_size, compute_type, language）

### 8.3 統合テスト

- [ ] `tests/integration/engines/test_smoke_engines.py` がパス
- [ ] `tests/integration/realtime/test_e2e_realtime_flow.py` がパス
- [ ] 全 `tests/integration/` テストがパス

### 8.4 機能テスト

- [ ] `EngineFactory.create_engine("whispers2t")` が動作
- [ ] `model_size` パラメータで各サイズが指定可能
- [ ] `compute_type="auto"` がデバイスに応じて正しく解決される（CPU→int8, GPU→float16）
- [ ] `compute_type` 明示指定が正しく反映される
- [ ] 無効な `model_size` で `ValueError` が発生
- [ ] 無効な `compute_type` で `ValueError` が発生
- [ ] 新モデル (`large-v3-turbo`, `distil-large-v3`) の動作確認
- [ ] v3ベースモデルで `n_mels=128` が正しく設定される
- [ ] 非v3モデルで `n_mels=80` が正しく設定される
- [ ] `MODEL_MAPPING` によるHuggingFaceパス変換が動作
- [ ] `LibraryPreloader.start_preloading("whispers2t")` が正しく動作

### 8.5 言語サポート

- [ ] `EngineFactory.create_engine("whispers2t", language="ja")` が動作
- [ ] `EngineFactory.create_engine("whispers2t", language="vi")` が動作（新規言語）
- [ ] `EngineFactory.create_engine("whispers2t", language="yue")` が動作（広東語）
- [ ] 無効な言語コード（例: `"xxx"`）で `ValueError` が発生
- [ ] `EngineMetadata.get("whispers2t").supported_languages` が99言語を含む
- [ ] `EngineFactory.get_engines_for_language("ja")` に `whispers2t` が含まれる
- [ ] `EngineFactory.get_engines_for_language("vi")` に `whispers2t` が含まれる（新規言語）
- [ ] `EngineFactory.get_engines_for_language("yue")` に `whispers2t` が含まれる（広東語）

### 8.6 languages.py

- [ ] `Languages.get_engines_for_language("ja")` に `whispers2t` が含まれる
- [ ] 旧エンジンID (`whispers2t_base` 等) が結果に含まれない

### 8.7 CLI

- [ ] `livecap-core --info` で `whispers2t` が表示される
- [ ] `livecap-core --info` で旧エンジンID が表示されない
- [ ] `tests/core/cli/test_cli.py` がパス

### 8.8 Examples

- [ ] 全 examples が正常に動作

### 8.9 CI

- [ ] 全ワークフローがグリーン

---

## 9. 完了条件

### 9.1 whisper_languages.py (新規)
- [ ] `livecap_core/engines/whisper_languages.py` が作成されている
- [ ] `WHISPER_LANGUAGES` (tuple) が99言語を含む
- [ ] `WHISPER_LANGUAGES_SET` (frozenset) が定義されている

### 9.2 metadata.py
- [ ] `whisper_languages` から `WHISPER_LANGUAGES` をインポートしている
- [ ] WhisperS2T エントリが1つに統合されている（`"whispers2t"`）
- [ ] `supported_languages` が `list(WHISPER_LANGUAGES)` を使用している
- [ ] `EngineInfo` に `available_model_sizes` フィールドが追加されている
- [ ] デフォルト `model_size` が `"large-v3"` に設定されている

### 9.3 whispers2t_engine.py
- [ ] `whisper_languages` から `WHISPER_LANGUAGES_SET` をインポートしている
- [ ] `engine_name` が `"whispers2t"` に統一されている
- [ ] 言語バリデーションが追加されている
- [ ] `compute_type` パラメータが追加されている
- [ ] `_resolve_compute_type()` で自動最適化が実装されている
- [ ] `MODEL_MAPPING` でHuggingFaceパス変換が実装されている
- [ ] `_get_n_mels()` でv3ベースモデルに128が設定される
- [ ] `model_size`、`compute_type`、`language` の入力バリデーションが実装されている
- [ ] `detect_device()` のタプル戻り値が正しく処理されている
- [ ] `get_supported_languages()` が `WHISPER_LANGUAGES` を返す
- [ ] `get_engine_name()` の `size_map` が全モデルサイズを含む
- [ ] `CPU_SPEED_ESTIMATES` が主要モデルを含む

### 9.4 関連ファイル
- [ ] `LibraryPreloader` が `whispers2t` に対応している
- [ ] `languages.py` の全言語で `supported_engines` が更新されている
- [ ] `engine_factory.py` の docstring が更新されている
- [ ] benchmark runner/tests が更新されている
- [ ] examples が更新されている
- [ ] CI ワークフローが更新されている

### 9.5 検証
- [ ] 全使用箇所が更新されている（`grep -r "whispers2t_"` で確認）
- [ ] 全テストがパス
- [ ] benchmark テストがパス（デフォルト `large-v3` で互換性維持）
- [ ] ドキュメントが更新されている
- [ ] CI が全てグリーン

---

## 10. リスクと対策

| リスク | レベル | 対策 |
|--------|--------|------|
| 使用箇所の更新漏れ | 中 | `grep -r "whispers2t_"` で網羅的に検索、テストで検出 |
| 新モデルのCT2モデル不在 | 中 | 実装前にHugging Faceで存在確認 |
| detect_device() タプル処理ミス | 中 | 型チェックで安全に処理 |
| LibraryPreloader 事前ロード失敗 | 中 | 統合テストで確認 |
| languages.py 更新漏れ | 中 | 全言語をリストアップして確認 |
| compute_type の誤設定 | 低 | バリデーションとユニットテストでカバー |
| 無効言語コードの検出漏れ | 低 | シンプルな `in` チェックで確実に検出 |
| CI 失敗 | 中 | ローカルで全テスト実行後に PR 作成 |

---

## 11. 関連 Issue

- **#166**: `detect_device()` リファクタリング（本 Issue 完了後に実施）
  - 戻り値を `Tuple[str, str]` → `str` に変更
  - `compute_type` は WhisperS2T 内部で解決するため不要に
  - 本 Issue では暫定対処（タプルの最初の要素を使用）

- **#168**: `languages.py` の `supported_engines` 不整合修正
  - Canary/Voxtral が日本語未サポートなのに `ja` の `supported_engines` に含まれている問題
  - 本 Issue (#165) の範囲外として切り出し
  - 各エンジンの言語コード調査結果に基づく

---

## 12. 参考資料

- [WhisperS2T GitHub](https://github.com/shashikg/WhisperS2T)
- [OpenAI Whisper GitHub](https://github.com/openai/whisper)
- [Whisper tokenizer.py (言語リスト)](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py)
- [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper)
- [CTranslate2 Quantization](https://opennmt.net/CTranslate2/quantization.html)
- [deepdml/faster-whisper-large-v3-turbo-ct2](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2)
- [Systran/faster-distil-whisper-large-v3](https://huggingface.co/Systran/faster-distil-whisper-large-v3)

---

## 変更履歴

| 日付 | 変更内容 |
|------|----------|
| 2025-12-04 | 初版作成 |
| 2025-12-04 | レビュー対応: LibraryPreloader更新追加、languages.py更新追加、detect_device()タプル処理明記、入力バリデーション追加、CT2モデル確認手順追加、ドキュメント網羅性向上、CLIテスト確認追加 |
| 2025-12-04 | 言語サポート拡張: WHISPER_LANGUAGES定数追加（99言語）、supported_languagesを直接使用、言語バリデーション追加、検証項目拡充 |
| 2025-12-04 | WhisperS2T調査結果追加: モデル識別子マッピング（MODEL_MAPPING）、n_mels設定（v3ベースは128）、CT2モデル存在確認済み、ローカルテスト結果追加 |
| 2025-12-04 | 設計決定事項追加（セクション3.7）: engine_name統一("whispers2t")、デフォルトmodel_size="large-v3"（benchmark互換性維持）、benchmark/examples更新方針、CPU_SPEED_ESTIMATES/size_map拡張、完了条件の詳細化 |
| 2025-12-04 | 言語コード変換調査結果追加（セクション3.8）: 各ASRエンジンの言語コード要件調査、Languages.asr_code活用方針決定、#168切り出し |
| 2025-12-04 | Task 5追加: WHISPER_LANGUAGESの独立モジュール化（whisper_languages.py）、EngineMetadata/CLI/APIで99言語表示対応、get_engines_for_language("vi")等で新言語が返却されるよう修正 |
| 2025-12-04 | 実装前最終確認: Task 4.2/4.3のコード例をインポート形式に修正、smoke test の model_size 指定方法を追記（EngineSmokeCase フィールド追加）、LibraryPreloader既存バグ発見・メモ追加 |
