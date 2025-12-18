# Phase 2: Config 廃止と API 簡素化 実装計画

> **Status**: ✅ COMPLETED
> **作成日:** 2025-12-01
> **完了日:** 2025-12-02
> **関連 Issue:** #70
> **関連 PR:** #158 (マージ済み)
> **依存:** #69 (Phase 1: リアルタイム文字起こし実装) ✅ 完了

---

## 1. 背景と目的

### 1.1 現状の課題

Phase 1 で `StreamTranscriber` + `VADProcessor` + `VADConfig` を実装した結果、以下の問題が明らかになった：

| 課題 | 詳細 | 影響度 |
|------|------|--------|
| **Config の存在意義** | Phase 1 のアーキテクチャは Config なしで動作する | 致命的 |
| VAD 設定の二重定義 | `silence_detection` と `VADConfig` が重複 | 高 |
| GUI 由来の複雑さ | `multi_source`, `vad_state_machine` 等は不要 | 高 |
| 型安全性の欠如 | dict ベースの Config は型が曖昧 | 中 |

### 1.2 方針転換

**当初の計画:** Config スキーマの簡素化・リネーム

**新しい方針:** Config システムの廃止

### 1.3 目標

1. **DEFAULT_CONFIG の廃止**: dict ベースの Config を削除
2. **EngineFactory の簡素化**: 必要最小限のパラメータのみ
3. **dataclass ベースの設定**: `VADConfig` パターンを踏襲
4. **config/ ディレクトリの削除**: 不要なコードを完全削除

---

## 2. 現状分析

### 2.1 Phase 1 のアーキテクチャ（Config 不使用）

```python
# 現在の使い方 - Config を使っていない
from livecap_cli import StreamTranscriber, MicrophoneSource
from livecap_cli.vad import VADConfig
from engines import EngineFactory

engine = EngineFactory.create_engine("whispers2t_base", device="cuda")
vad_config = VADConfig(threshold=0.5, min_speech_ms=250)

with StreamTranscriber(engine=engine, vad_config=vad_config) as transcriber:
    with MicrophoneSource(sample_rate=16000) as mic:
        for result in transcriber.transcribe_sync(mic):
            print(result.text)
```

### 2.2 Config が使われている箇所

#### EngineFactory 関連

| 箇所 | 使用内容 | 廃止後の対応 |
|------|----------|-------------|
| `EngineFactory.create_engine()` | `language_engines` マッピング、auto 解決 | **auto 廃止**、エンジン明示指定を必須化 |
| `EngineFactory._configure_engine_specific_settings()` | エンジン固有設定 | **メソッド廃止**、`EngineMetadata.default_params` に統合 |
| `EngineFactory._prepare_config()` | Config の正規化 | **メソッド廃止** |
| `EngineFactory.resolve_auto_engine()` | auto → 実エンジン解決 | **メソッド廃止** |
| `EngineFactory.get_default_engine_for_language()` | 言語別デフォルト | **メソッド廃止**（`EngineMetadata` で代替可能） |

#### エンジンクラス（多言語対応）

| 箇所 | 使用内容 | 廃止後の対応 |
|------|----------|-------------|
| `WhisperS2TEngine.transcribe()` | `config["transcription"]["input_language"]` | `__init__(language=...)` で受け取り、`self.language` を使用 |
| `CanaryEngine.transcribe()` | `config["transcription"]["input_language"]` | `__init__(language=...)` で受け取り、`self.language` を使用 |
| `VoxtralEngine.transcribe()` | `config["transcription"]["input_language"]` | `__init__(language=...)` で受け取り、`self.language` を使用 |
| `ReazonSpeechEngine.__init__()` | `config["transcription"]["reazonspeech_config"]` | `__init__(use_int8=..., ...)` で直接受け取る |
| `ParakeetEngine.__init__()` | `config["parakeet"]["model_name"]` | `__init__(model_name=...)` で直接受け取る |

#### その他

| 箇所 | 使用内容 | 廃止後の対応 |
|------|----------|-------------|
| `benchmarks/common/engines.py` | `_build_config()` で Config 構築 | `**engine_options` で直接渡す |
| `cli.py --dump-config` | 診断出力 | `--info` に置き換え |
| `examples/*.py` | 設定の取得 | 直接パラメータ指定 |

> **重要な設計決定**:
> 1. `engine_type="auto"` は廃止。`EngineMetadata.get_engines_for_language()` でエンジン検索可能
> 2. `EngineMetadata.default_params` がエンジン固有パラメータの唯一の定義場所
> 3. 多言語エンジンは `language` パラメータを `__init__` で受け取る

### 2.3 削除対象ファイル

```
config/                              # 完全削除
├── __init__.py
└── core_config_builder.py

livecap_cli/config/                 # 完全削除（ディレクトリごと）
├── __init__.py                      # 削除
├── defaults.py                      # 削除
├── schema.py                        # 削除
└── validator.py                     # 削除
```

> **注意**: `livecap_cli/config/` は完全削除が可能。VADConfig は `livecap_cli/vad/config.py` に定義されており、別ディレクトリのため影響なし。削除前に Section 10.2 の更新対象ファイルを先に修正する必要がある。

---

## 3. 新しいアーキテクチャ

### 3.1 EngineFactory の簡素化

```python
# engines/engine_factory.py
class EngineFactory:
    """音声認識エンジンファクトリー"""

    @classmethod
    def create_engine(
        cls,
        engine_type: str,  # 必須、"auto" は廃止
        device: str | None = None,
        **engine_options,
    ) -> BaseEngine:
        """
        エンジンを作成

        Args:
            engine_type: エンジンタイプ（必須）
            device: デバイス（"cuda", "cpu", None=自動）
            **engine_options: エンジン固有オプション
                - model_size: WhisperS2T 用
                - model_name: Parakeet/Voxtral 用
                - use_int8: ReazonSpeech 用

        Returns:
            BaseEngine インスタンス

        Raises:
            ValueError: engine_type="auto" が指定された場合（非推奨）

        Example:
            # 利用可能なエンジンを確認
            from engines.metadata import EngineMetadata
            engines = EngineMetadata.get_engines_for_language("ja")
            # → ["reazonspeech", "parakeet_ja", "whispers2t_base", ...]

            # 明示的にエンジンを指定
            engine = EngineFactory.create_engine("reazonspeech", device="cuda")
        """
        if engine_type == "auto":
            raise ValueError(
                "engine_type='auto' is deprecated. "
                "Use EngineMetadata.get_engines_for_language() to find available engines."
            )
        ...
```

> **設計根拠**: `LANGUAGE_DEFAULTS` を廃止する理由
> 1. `EngineMetadata` に各エンジンの `supported_languages` が既に定義されている
> 2. `EngineMetadata.get_engines_for_language()` で言語→エンジン検索が可能
> 3. ユーザーが明示的にエンジンを選択することで、意図しないエンジン選択を防止

### 3.2 VADConfig（変更なし）

```python
# livecap_cli/vad/config.py - 既存のまま維持
@dataclass(frozen=True, slots=True)
class VADConfig:
    threshold: float = 0.5
    neg_threshold: float | None = None
    min_speech_ms: int = 250
    min_silence_ms: int = 100
    speech_pad_ms: int = 100
    max_speech_ms: int = 0
    interim_min_duration_ms: int = 2000
    interim_interval_ms: int = 1000
```

### 3.3 CLI の簡素化

```python
# livecap_cli/cli.py
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="livecap-core",
        description="LiveCap Core installation diagnostics.",
    )
    parser.add_argument("--info", action="store_true", help="Show installation info")
    parser.add_argument("--ensure-ffmpeg", action="store_true")
    parser.add_argument("--as-json", action="store_true")
    # --dump-config は削除
    ...
```

#### `--info` 出力内容（具体化）

Config 検証を削除し、以下の情報を表示：

| 項目 | 説明 | 実装 |
|------|------|------|
| **FFmpeg** | パス / 利用可能フラグ | `get_ffmpeg_manager().resolve_executable()` |
| **Models root** | モデル保存先ディレクトリ | `get_model_manager().models_root` |
| **Cache root** | キャッシュディレクトリ | `get_model_manager().cache_root` |
| **CUDA** | CUDA 利用可否 | `torch.cuda.is_available()` |
| **VAD backends** | 利用可能な VAD 一覧 | `get_available_presets()` |
| **ASR engines** | 登録済みエンジン一覧 | `EngineMetadata.get_all()` |
| **Translator** | 翻訳機能の状態 | `diagnose_i18n()` |

```bash
# 出力例
$ livecap-core --info
LiveCap Core diagnostics:
  FFmpeg: /usr/bin/ffmpeg
  Models root: ~/.cache/livecap/models
  Cache root: ~/.cache/livecap
  CUDA available: yes (RTX 4090)
  VAD backends: silero, tenvad, webrtc
  ASR engines: reazonspeech, parakeet, parakeet_ja, whispers2t_base, ...
  Translator: not registered (fallback only)
```

---

## 4. 実装タスク

### 4.1 EngineFactory の簡素化

#### Task 1.1: EngineFactory のリファクタリング

**ファイル:** `engines/engine_factory.py`

**削除するメソッド:**
- `_prepare_config()` - Config 依存
- `_configure_engine_specific_settings()` - `EngineMetadata.default_params` で代替
- `resolve_auto_engine()` - auto 廃止
- `get_default_engine_for_language()` - `EngineMetadata` で代替可能

**変更内容:**
- `build_core_config()` の import と呼び出しを削除
- `create_engine()` で `engine_type="auto"` を `ValueError` で拒否
- `create_engine()` を `**engine_options` + `EngineMetadata.default_params` で簡素化

#### Task 1.2: エンジン固有オプションの対応

**方針**: `_configure_engine_specific_settings()` を廃止し、`EngineMetadata.default_params` に統合

```python
# 新しい create_engine() の実装
@classmethod
def create_engine(cls, engine_type: str, device: str | None = None, **engine_options):
    metadata = EngineMetadata.get(engine_type)
    if not metadata:
        raise ValueError(f"Unknown engine type: {engine_type}")

    # デフォルトパラメータと engine_options をマージ（engine_options が優先）
    merged_options = {**metadata.default_params, **engine_options}

    engine_class = cls._get_engine_class(engine_type)
    return engine_class(device=device, **merged_options)
```

```python
# 使用例
engine = EngineFactory.create_engine("whispers2t_large_v3", device="cuda")
# → default_params から model_size="large-v3" が自動適用

engine = EngineFactory.create_engine("reazonspeech", device="cuda", use_int8=True)
# → default_params + use_int8=True がマージされる
```

#### Task 1.3: EngineMetadata.default_params の拡充

**ファイル:** `engines/metadata.py`

##### パラメータの分類方針

調査の結果、エンジンパラメータを以下の3カテゴリに分類して管理する：

| カテゴリ | 定義 | 対応方針 |
|---------|------|---------|
| **A: ユーザー向け** | 一般ユーザーが変更する可能性が高い | `default_params` に追加 |
| **B: 内部詳細** | エンジン固有の実装詳細 | エンジンクラス内でハードコード維持 |
| **C: 上級者向け** | 特殊なケースでのみ変更 | `**kwargs` 経由で上書き可能 |

##### カテゴリA: `default_params` に追加するパラメータ

| エンジン | パラメータ | 型 | デフォルト値 | 備考 |
|---------|-----------|-----|-------------|------|
| **reazonspeech** | `use_int8` | bool | `False` | int8量子化 |
| | `num_threads` | int | `4` | 処理スレッド数 |
| | `decoding_method` | str | `"greedy_search"` | デコード方式 |
| **whispers2t_*** | `batch_size` | int | `24` | バッチサイズ |
| | `use_vad` | bool | `True` | 内蔵VAD |
| **canary** | `model_name` | str | `"nvidia/canary-1b-v2"` | モデル名 |
| | `beam_size` | int | `1` | ビームサーチ |
| **parakeet** | `model_name` | str | `"nvidia/parakeet-tdt-0.6b-v3"` | モデル名 |
| | `decoding_strategy` | str | `"greedy"` | デコード戦略 |
| **parakeet_ja** | `decoding_strategy` | str | `"greedy"` | デコード戦略 |
| **voxtral** | `model_name` | str | `"mistralai/Voxtral-Mini-3B-2507"` | モデル名 |

##### カテゴリB: エンジン内部に維持（`**kwargs` で上書き可能）

ReazonSpeech 固有の音声処理パラメータ。99%のユーザーは変更不要だが、上級者は `**kwargs` 経由で上書き可能：

| パラメータ | 型 | デフォルト値 | 用途 |
|-----------|-----|-------------|------|
| `auto_split_duration` | float | `30.0` | 長音声の自動分割時間 |
| `padding_duration` | float | `0.9` | パディング時間 |
| `padding_threshold` | float | `5.0` | パディング閾値 |
| `min_audio_duration` | float | `0.3` | 最小音声長 |
| `short_audio_duration` | float | `1.0` | 短い音声の閾値 |
| `extended_padding_duration` | float | `2.0` | 拡張パディング |
| `decode_timeout` | float | `5.0` | デコードタイムアウト |

##### 更新後の `EngineMetadata.default_params`

```python
# engines/metadata.py - 実装済み
"reazonspeech": EngineInfo(
    ...
    default_params={
        # カテゴリA パラメータ（エンジンの__init__で使用）
        # Note: temperature, beam_size は sherpa-onnx API で未使用のため削除
        "use_int8": False,
        "num_threads": 4,
        "decoding_method": "greedy_search",
    }
),
"parakeet": EngineInfo(
    ...
    default_params={
        "model_name": "nvidia/parakeet-tdt-0.6b-v3",
        "decoding_strategy": "greedy",
    }
),
"parakeet_ja": EngineInfo(
    ...
    default_params={
        "model_name": "nvidia/parakeet-tdt_ctc-0.6b-ja",
        "decoding_strategy": "greedy",
    }
),
"canary": EngineInfo(
    ...
    default_params={
        "model_name": "nvidia/canary-1b-v2",
        "beam_size": 1,
    }
),
"voxtral": EngineInfo(
    ...
    default_params={
        "temperature": 0.0,
        "do_sample": False,
        "max_new_tokens": 448,
        "model_name": "mistralai/Voxtral-Mini-3B-2507",
    }
),
"whispers2t_base": EngineInfo(
    ...
    default_params={
        "model_size": "base",
        "batch_size": 24,
        "use_vad": True,
    }
),
# whispers2t_tiny, whispers2t_small, whispers2t_medium, whispers2t_large_v3 も同様に更新
```

> **設計原則**: `EngineMetadata.default_params` がエンジン固有パラメータの**唯一の定義場所**とする。これにより：
> - 単一の真実の源 (Single Source of Truth) を実現
> - 新エンジン追加時は `metadata.py` のみを更新
> - `EngineMetadata.get_all()` でデフォルトパラメータを一覧表示可能
> - カテゴリBは `default_params` に含めず、必要時のみ `**kwargs` で上書き

#### Task 1.4: BaseEngine + 各エンジンクラスの `__init__` 修正

**背景**: 現在のエンジンは `config["transcription"]["input_language"]` から言語を取得している。Config 廃止後は `**engine_options` で直接パラメータを受け取る必要がある。

**影響ファイル:**
- `engines/base_engine.py` ← **BaseEngine の修正（重要）**
- `engines/whispers2t_engine.py`
- `engines/canary_engine.py`
- `engines/voxtral_engine.py`
- `engines/reazonspeech_engine.py`
- `engines/parakeet_engine.py`

##### BaseEngine の修正

```python
# Before
class BaseEngine(ABC):
    def __init__(self, device: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.device = device
        self.config = config or {}
        # ... config.get('engines', {}) で設定を取得

# After
class BaseEngine(ABC):
    def __init__(self, device: Optional[str] = None, **kwargs):
        self.device = device
        # self.config は完全削除（後方互換性は無視）
        # 各エンジンは kwargs から必要なパラメータを取り出す
```

> **設計決定**: `self.config` は**完全削除**する（後方互換性は無視して良いため）。
>
> **移行順序**: 子クラス（各エンジン）を先に修正し、最後に BaseEngine を修正することで段階的に移行可能。

##### 各エンジンクラスの修正

**変更内容:**

| エンジン | 現在の `__init__` | 修正後 |
|----------|-------------------|--------|
| WhisperS2TEngine | `(device, config)` | `(device, language="ja", use_vad=True, model_size="base", batch_size=24, **kwargs)` |
| CanaryEngine | `(device, config)` | `(device, language="en", model_name=..., beam_size=1, **kwargs)` |
| VoxtralEngine | `(device, config)` | `(device, language="auto", model_name=..., **kwargs)` |
| ReazonSpeechEngine | `(device, config)` | `(device, use_int8=False, num_threads=4, decoding_method="greedy_search", **kwargs)` |
| ParakeetEngine | `(device, config)` | `(device, model_name=..., decoding_strategy="greedy", **kwargs)` |

> **`**kwargs` の役割**: カテゴリB（内部詳細）パラメータを受け取るため。
> 例: ReazonSpeechEngine で `decode_timeout=10.0` を上書きしたい場合、`**kwargs` 経由で渡す。

**コード例（WhisperS2TEngine）:**

```python
# Before
class WhisperS2TEngine(BaseEngine):
    def __init__(self, device=None, config=None):
        self.config = config or {}
        model_size = self.config.get('whispers2t', {}).get('model_size', 'base')
        ...

    def transcribe(self, audio_data, sample_rate):
        input_language = self.config.get('transcription', {}).get('input_language', 'ja')
        ...

# After
class WhisperS2TEngine(BaseEngine):
    def __init__(
        self,
        device: str | None = None,
        language: str = "ja",
        use_vad: bool = True,
        model_size: str = "base",
        **kwargs,
    ):
        self.language = language
        self.use_vad = use_vad
        self.model_size = model_size
        ...

    def transcribe(self, audio_data, sample_rate):
        input_language = self.language  # self.config ではなく self.language を使用
        ...
```

> **注意**: `**kwargs` を受け取ることで、`EngineMetadata.default_params` からの追加パラメータを柔軟に処理可能。

### 4.2 config/ ディレクトリの削除

#### Task 2.1: config/ の削除

**削除ファイル:**
- `config/__init__.py`
- `config/core_config_builder.py`

#### Task 2.2: livecap_cli/config/ の簡素化

**削除ファイル:**
- `livecap_cli/config/defaults.py`
- `livecap_cli/config/schema.py`
- `livecap_cli/config/validator.py`

**更新ファイル:**
- `livecap_cli/config/__init__.py` - 空または削除

### 4.3 依存コードの更新

#### Task 3.1: benchmarks/common/engines.py

**依存**: Task 1.1-1.4 完了後に実施

**変更内容:**
- `_build_config()` メソッドを削除
- `_create_engine()` で `**engine_options` を使用

```python
# Before
def _build_config(self, engine_id: str, language: str) -> dict[str, Any]:
    config: dict[str, Any] = {
        "transcription": {
            "input_language": language,
        }
    }
    if engine_id.startswith("whispers2t_"):
        config["whispers2t"] = {"use_vad": False}
    return config

def _create_engine(self, engine_id, device, language):
    config = self._build_config(engine_id, language)
    engine = EngineFactory.create_engine(engine_id, device, config)
    ...

# After
def _create_engine(self, engine_id, device, language):
    # engine_options を構築
    engine_options: dict[str, Any] = {
        "language": language,
    }

    # WhisperS2T の場合は VAD を無効化（純粋な ASR 性能を測定）
    if engine_id.startswith("whispers2t_"):
        engine_options["use_vad"] = False

    engine = EngineFactory.create_engine(
        engine_type=engine_id,
        device=device,
        **engine_options,
    )
    ...
```

**変更しない箇所:**
- `get_engine(engine_id, device, language)` - シグネチャ維持
- `get_model_memory(engine_id, device, language)` - シグネチャ維持
- `unload_engine(engine_id, device, language)` - シグネチャ維持
- キャッシュキー `f"{engine_id}_{device}_{language}"` - 維持

> **注意**: `language` パラメータは多言語エンジンで実際に使用されるため、維持が必要。

#### Task 3.2: Examples の更新

**影響ファイル:**
- `examples/realtime/basic_file_transcription.py`
- `examples/realtime/async_microphone.py`
- `examples/realtime/callback_api.py`
- `examples/realtime/custom_vad_config.py`

```python
# Before
from livecap_cli.config.defaults import get_default_config
config = get_default_config()
config["transcription"]["engine"] = engine_type
engine = EngineFactory.create_engine(engine_type, device, config)

# After
engine = EngineFactory.create_engine(engine_type, device=device)
```

#### Task 3.3: CLI の更新

**ファイル:** `livecap_cli/cli.py`

- `--dump-config` を削除
- `--info` に置き換え（FFmpeg, モデルパス等の情報表示）
- `ConfigValidator` の使用を削除

#### Task 3.4: テスト・CI の更新

**削除テスト:**
- `tests/core/config/test_config_defaults.py`
- `tests/core/config/test_core_config_builder.py`

**更新テスト:**
- `tests/core/engines/test_engine_factory.py`
- `tests/integration/engines/test_smoke_engines.py`

**新規テスト（追加）:**

```python
# tests/core/engines/test_engine_factory.py に追加

def test_create_engine_auto_raises_error():
    """engine_type='auto' が ValueError を発生させることを確認"""
    with pytest.raises(ValueError, match="deprecated"):
        EngineFactory.create_engine("auto", device="cuda")

def test_create_engine_with_kwargs():
    """**engine_options が正しくマージされることを確認"""
    engine = EngineFactory.create_engine("reazonspeech", device="cpu", use_int8=True)
    assert engine.use_int8 == True

def test_create_engine_default_params_applied():
    """EngineMetadata.default_params がデフォルト適用されることを確認"""
    engine = EngineFactory.create_engine("reazonspeech", device="cpu")
    assert engine.use_int8 == False  # default_params のデフォルト値
    assert engine.num_threads == 4
```

**CI の更新（同一 PR で実施）:**
- `.github/workflows/integration-tests.yml`
  - `from livecap_cli.config.defaults import get_default_config` を削除
  - 直接パラメータ指定に変更

> **重要**: コード変更と CI 更新を同一 PR で実施すること。別 PR にするとマージ順序によって CI が壊れる期間が発生する。

### 4.4 その他の影響コード

#### Task 4.1: FileTranscriptionPipeline

**ファイル:** `livecap_cli/transcription/file_pipeline.py`

- `config` パラメータを削除（現在も未使用）

#### Task 4.2: engines/*.py

各エンジンの `config` パラメータ使用状況を確認し、必要に応じて更新。

#### Task 4.3: engines/__init__.py のエクスポート更新

**ファイル:** `engines/__init__.py`

`EngineMetadata` を公開 API としてエクスポート（`get_engines_for_language()` を使用するため）：

```python
# engines/__init__.py
from .metadata import EngineMetadata
from .engine_factory import EngineFactory

__all__ = ["EngineFactory", "EngineMetadata", ...]
```

---

## 5. 移行手順

```
Step 1a: EngineMetadata.default_params の拡充 (Task 1.3)
         → 他への影響なし、単独でテスト可能
    ↓
Step 1b: BaseEngine + 各エンジンクラスの __init__ 修正 (Task 1.4)
         → config 依存を削除、**kwargs 対応
    ↓
Step 1c: EngineFactory の簡素化 (Task 1.1-1.2)
         → Step 1a, 1b に依存
    ↓
Step 2: benchmarks/common/engines.py の更新 (Task 3.1)
    ↓
Step 3: Examples の更新 (Task 3.2)
    ↓
Step 4: CLI の更新 (Task 3.3)
         → --dump-config 削除、--info 追加
    ↓
Step 5: テスト・CI の更新 (Task 3.4)
         → CI (.github/workflows/) も同一 PR で更新
    ↓
Step 6: config/ ディレクトリの削除
    ↓
Step 7: livecap_cli/config/ の削除
    ↓
Step 8: 全テスト実行・確認
```

---

## 6. 検証項目

### 6.1 単体テスト

- [x] `test_engine_factory.py` がパス
- [x] Config 関連テストを削除済み

### 6.2 統合テスト

- [x] `test_smoke_engines.py` がパス
- [x] `test_file_transcription_pipeline.py` がパス
- [x] `test_e2e_realtime_flow.py` がパス

### 6.3 ベンチマーク

- [x] ASR ベンチマークが動作
- [x] VAD ベンチマークが動作
- [x] 最適化ベンチマークが動作

### 6.4 Examples 動作確認

- [x] `basic_file_transcription.py` が動作
- [x] `async_microphone.py` が動作
- [x] `callback_api.py` が動作
- [x] `custom_vad_config.py` が動作

### 6.5 CLI

- [x] `livecap-core --info` が動作
- [x] `livecap-core --ensure-ffmpeg` が動作

---

## 7. 削除対象の完全リスト

### 7.1 ファイル削除

| ファイル | 理由 |
|----------|------|
| `config/__init__.py` | Config 廃止 |
| `config/core_config_builder.py` | Config 廃止 |
| `livecap_cli/config/__init__.py` | Config 廃止（VADConfig は `livecap_cli/vad/config.py` のため影響なし） |
| `livecap_cli/config/defaults.py` | Config 廃止 |
| `livecap_cli/config/schema.py` | Config 廃止 |
| `livecap_cli/config/validator.py` | Config 廃止 |
| `tests/core/config/test_config_defaults.py` | Config 廃止 |
| `tests/core/config/test_core_config_builder.py` | Config 廃止 |

### 7.2 コード削除

| ファイル | 削除内容 |
|----------|----------|
| `engines/engine_factory.py` | `_prepare_config()`, `build_core_config` インポート |
| `livecap_cli/cli.py` | `--dump-config`, `ConfigValidator` |
| `livecap_cli/transcription/file_pipeline.py` | `config` パラメータ |

---

## 8. 完了条件

- [x] `DEFAULT_CONFIG` が完全に削除されている
- [x] `config/` ディレクトリが削除されている
- [x] `livecap_cli/config/` が削除または空になっている
- [x] `EngineFactory` が Config なしで動作する
- [x] 全テストがパス
- [x] 全ベンチマークが動作
- [x] Examples が動作

---

## 9. リスクと対策

| リスク | レベル | 対策 |
|--------|--------|------|
| 見落としたコード依存 | 低 | Grep で網羅的に検索済み（下記参照） |
| エンジン固有設定の欠落 | 中 | 各エンジンの使用状況を個別確認 |
| テスト失敗 | 中 | 段階的に実行、各ステップで確認 |
| Examples 動作不良 | 低 | 全 Examples の動作確認を検証項目に含む |

### 9.1 後方互換性と CHANGELOG

Phase 2 は**破壊的変更**を含む。CHANGELOG に以下を記載すること：

#### Breaking Changes

```markdown
## [UNRELEASED] - Phase 2: Config 廃止

### Breaking Changes

- **`engine_type="auto"` 廃止**: `EngineFactory.create_engine()` で `engine_type="auto"` を指定すると `ValueError` が発生します。`EngineMetadata.get_engines_for_language()` を使用して利用可能なエンジンを確認してください。

- **`livecap_cli.config` モジュール削除**: 以下のインポートは動作しなくなります：
  - `from livecap_cli.config import get_default_config`
  - `from livecap_cli.config import merge_config`
  - `from livecap_cli.config import ConfigValidator`

- **エンジン `__init__` シグネチャ変更**: 全エンジンの `__init__` が `(device, config)` から `(device, **kwargs)` に変更されます。

### Migration Guide

# Before
engine = EngineFactory.create_engine("auto", device="cuda", config=my_config)

# After
from engines.metadata import EngineMetadata
engines = EngineMetadata.get_engines_for_language("ja")  # → ["reazonspeech", ...]
engine = EngineFactory.create_engine("reazonspeech", device="cuda")
```

> **注意**: livecap-core は内部ライブラリのため、Migration Guide は簡潔で十分。エラーメッセージで移行先を案内する。

---

## 10. 影響調査結果

### 10.1 削除対象ファイル（影響なし）

Config 廃止に伴い削除するファイル。これらは他から参照されないため影響なし。

| ファイル | 理由 |
|----------|------|
| `config/__init__.py` | Config 廃止 |
| `config/core_config_builder.py` | Config 廃止 |
| `livecap_cli/config/__init__.py` | Config 廃止（VADConfig は `livecap_cli/vad/config.py` のため影響なし） |
| `livecap_cli/config/defaults.py` | Config 廃止 |
| `livecap_cli/config/schema.py` | Config 廃止 |
| `livecap_cli/config/validator.py` | Config 廃止 |
| `tests/core/config/test_config_defaults.py` | Config 廃止 |
| `tests/core/config/test_core_config_builder.py` | Config 廃止 |

### 10.2 更新が必要なファイル

Config を参照している箇所と、具体的な変更内容。

#### コードファイル

| ファイル | 現在の使用 | 変更内容 |
|----------|-----------|----------|
| `engines/engine_factory.py` | `build_core_config()`, `_prepare_config()` | `**engine_options` + `EngineMetadata.default_params` |
| `engines/whispers2t_engine.py` | `config["transcription"]["input_language"]` | `__init__(language=...)` で直接受け取る |
| `engines/canary_engine.py` | `config["transcription"]["input_language"]` | `__init__(language=...)` で直接受け取る |
| `engines/voxtral_engine.py` | `config["transcription"]["input_language"]` | `__init__(language=...)` で直接受け取る |
| `engines/reazonspeech_engine.py` | `config["transcription"]["reazonspeech_config"]` | `__init__(use_int8=..., ...)` で直接受け取る |
| `engines/parakeet_engine.py` | `config["parakeet"]["model_name"]` | `__init__(model_name=...)` で直接受け取る |
| `benchmarks/common/engines.py` | `_build_config()` | `**engine_options` で渡す |
| `livecap_cli/cli.py` | `--dump-config`, `ConfigValidator` | `--info` に置き換え、Validator 削除 |
| `examples/realtime/basic_file_transcription.py` | `get_default_config()` | 直接パラメータ指定に変更 |
| `examples/realtime/async_microphone.py` | `get_default_config()` | 直接パラメータ指定に変更 |
| `examples/realtime/callback_api.py` | `get_default_config()` | 直接パラメータ指定に変更 |
| `examples/realtime/custom_vad_config.py` | `get_default_config()` | 直接パラメータ指定に変更 |
| `tests/integration/engines/test_smoke_engines.py` | `_build_config()` 関数 | `**engine_options` で直接指定 |
| `tests/integration/transcription/test_file_transcription_pipeline.py` | `config=get_default_config()` | `config` パラメータ削除 |
| `tests/integration/realtime/test_e2e_realtime_flow.py` | `config["transcription"]` 操作 | Config 操作を削除 |

#### ドキュメントファイル

| ファイル | 現在の使用 | 変更内容 |
|----------|-----------|----------|
| `README.md` | `get_default_config()` サンプルコード | 新 API でのサンプルに更新 |
| `CLAUDE.md` | `--dump-config` 例、Configuration セクション | `--info` に更新、Config 説明を削除 |
| `AGENTS.md` | `--dump-config` 例 | `--info` に更新 |
| `GEMINI.md` | `--dump-config` 例 | `--info` に更新 |
| `docs/architecture/core-api-spec.md` | Config API 仕様、サンプルコード多数 | 新 API 仕様に全面更新 |
| `docs/reference/feature-inventory.md` | Config 使用例多数 | 新 API でのサンプルに更新 |

### 10.3 誤検知（影響なし）

Grep で検出されたが、実際には影響がない箇所。

| ファイル | 理由 |
|----------|------|
| `livecap_cli/vad/config.py` | VADConfig dataclass（維持対象） |

### 10.4 評価サマリー

- **削除ファイル**: 8 ファイル（`livecap_cli/config/__init__.py` 追加）
- **更新ファイル（コード）**: 15 ファイル（エンジンクラス 5 ファイル追加）
- **更新ファイル（ドキュメント）**: 6 ファイル
- **影響範囲**: 中程度、エンジンクラス修正が追加で必要

> **注意**: `docs/architecture/core-api-spec.md` と `docs/reference/feature-inventory.md` は Config 参照が多いため、Phase 2 完了後に全面的な見直しが必要です。

---

## 変更履歴

| 日付 | 変更内容 |
|------|----------|
| 2025-12-01 | 初版作成（Config 簡素化計画） |
| 2025-12-01 | **方針転換: Config 廃止に変更** |
| 2025-12-01 | セクション 10「影響調査結果」追加、リスク評価詳細化 |
| 2025-12-01 | CLI `--info` 出力内容を具体化、ドキュメント更新対象を追加 |
| 2025-12-02 | **auto 廃止、LANGUAGE_DEFAULTS 廃止**: エンジン明示指定を必須化 |
| 2025-12-02 | Task 1.3 追加: `EngineMetadata.default_params` 拡充、設計原則を明記 |
| 2025-12-02 | Task 1.4 追加: 各エンジンクラスの `__init__` 修正（`language` パラメータ対応） |
| 2025-12-02 | Task 3.1 更新: benchmarks の `**engine_options` 対応、影響調査結果を更新 |
| 2025-12-02 | Task 1.3 拡充: パラメータ3カテゴリ分類（A:ユーザー向け、B:内部詳細、C:上級者向け）を追加 |
| 2025-12-02 | 全エンジンの `default_params` 追加パラメータを網羅（WhisperS2T: batch_size/use_vad、Canary: model_name/beam_size、Parakeet: decoding_strategy） |
| 2025-12-02 | **実装前最終レビュー完了**: Step 細分化（1a/1b/1c）、BaseEngine 修正追加、CI 同一 PR 対応、CHANGELOG 記載追加 |
| 2025-12-02 | 追加詳細: 新規テスト例（Task 3.4）、engines/__init__.py エクスポート（Task 4.3）、self.config 完全削除を明記 |
| 2025-12-02 | **✅ 実装完了**: PR #158 マージ。全タスク完了、全テストパス |
| 2025-12-02 | 実装時の修正: ReazonSpeech の `temperature`, `beam_size` は sherpa-onnx API で未使用のため削除 |
| 2025-12-02 | 追加修正: WhisperS2T の `transcribe()` に `batch_size` パラメータ追加 (PR #159) |
