# Phase 2: Config 廃止と API 簡素化 実装計画

> **Status**: 📋 PLANNING
> **作成日:** 2025-12-01
> **更新日:** 2025-12-02
> **関連 Issue:** #70
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
from livecap_core import StreamTranscriber, MicrophoneSource
from livecap_core.vad import VADConfig
from engines import EngineFactory

engine = EngineFactory.create_engine("whispers2t_base", device="cuda")
vad_config = VADConfig(threshold=0.5, min_speech_ms=250)

with StreamTranscriber(engine=engine, vad_config=vad_config) as transcriber:
    with MicrophoneSource(sample_rate=16000) as mic:
        for result in transcriber.transcribe_sync(mic):
            print(result.text)
```

### 2.2 Config が使われている箇所

| 箇所 | 使用内容 | 廃止後の対応 |
|------|----------|-------------|
| `EngineFactory.create_engine()` | `language_engines` マッピング、auto 解決 | **auto 廃止**、エンジン明示指定を必須化 |
| `EngineFactory._configure_engine_specific_settings()` | エンジン固有設定 | `**engine_options` で対応 |
| `benchmarks/common/engines.py` | `transcription.input_language` | 引数で直接指定 |
| `cli.py --dump-config` | 診断出力 | `--info` に置き換え |
| `examples/*.py` | 設定の取得 | 直接パラメータ指定 |

> **重要な設計決定**: `engine_type="auto"` は廃止します。各エンジンの対応言語は `EngineMetadata.get_engines_for_language()` で確認でき、ユーザーは明示的にエンジンを選択します。

### 2.3 削除対象ファイル

```
config/                              # 完全削除
├── __init__.py
└── core_config_builder.py

livecap_core/config/                 # 大部分を削除
├── __init__.py                      # 簡素化
├── defaults.py                      # 削除
├── schema.py                        # 削除
└── validator.py                     # 削除
```

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
# livecap_core/vad/config.py - 既存のまま維持
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
# livecap_core/cli.py
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

変更内容:
- `_prepare_config()` を削除
- `build_core_config()` の呼び出しを削除
- `_configure_engine_specific_settings()` を `**engine_options` で置き換え
- `resolve_auto_engine()` を削除
- `get_default_engine_for_language()` を削除（`EngineMetadata` で代替可能）
- `create_engine()` で `engine_type="auto"` を `ValueError` で拒否

#### Task 1.2: エンジン固有オプションの対応

**影響エンジン:**
- WhisperS2T: `model_size` パラメータ
- Parakeet: `model_name` パラメータ
- Voxtral: `model_name` パラメータ

```python
# 使用例
engine = EngineFactory.create_engine(
    "whispers2t_large_v3",
    device="cuda",
    model_size="large-v3",  # エンジン固有オプション
)
```

### 4.2 config/ ディレクトリの削除

#### Task 2.1: config/ の削除

**削除ファイル:**
- `config/__init__.py`
- `config/core_config_builder.py`

#### Task 2.2: livecap_core/config/ の簡素化

**削除ファイル:**
- `livecap_core/config/defaults.py`
- `livecap_core/config/schema.py`
- `livecap_core/config/validator.py`

**更新ファイル:**
- `livecap_core/config/__init__.py` - 空または削除

### 4.3 依存コードの更新

#### Task 3.1: benchmarks/common/engines.py

```python
# Before
config = {
    "transcription": {
        "input_language": language,
    }
}
engine = EngineFactory.create_engine(engine_id, device, config)

# After
engine = EngineFactory.create_engine(engine_id, device=device)
```

> **注意**: `language` は `EngineFactory` に渡す必要がなくなります。必要に応じてベンチマーク側でエンジン選択ロジックを実装してください。

#### Task 3.2: Examples の更新

**影響ファイル:**
- `examples/realtime/basic_file_transcription.py`
- `examples/realtime/async_microphone.py`
- `examples/realtime/callback_api.py`
- `examples/realtime/custom_vad_config.py`

```python
# Before
from livecap_core.config.defaults import get_default_config
config = get_default_config()
config["transcription"]["engine"] = engine_type
engine = EngineFactory.create_engine(engine_type, device, config)

# After
engine = EngineFactory.create_engine(engine_type, device=device)
```

#### Task 3.3: CLI の更新

**ファイル:** `livecap_core/cli.py`

- `--dump-config` を削除
- `--info` に置き換え（FFmpeg, モデルパス等の情報表示）
- `ConfigValidator` の使用を削除

#### Task 3.4: テストの更新

**削除テスト:**
- `tests/core/config/test_config_defaults.py`
- `tests/core/config/test_core_config_builder.py`

**更新テスト:**
- `tests/core/engines/test_engine_factory.py`
- `tests/integration/engines/test_smoke_engines.py`

### 4.4 その他の影響コード

#### Task 4.1: FileTranscriptionPipeline

**ファイル:** `livecap_core/transcription/file_pipeline.py`

- `config` パラメータを削除（現在も未使用）

#### Task 4.2: engines/*.py

各エンジンの `config` パラメータ使用状況を確認し、必要に応じて更新。

---

## 5. 移行手順

```
Step 1: EngineFactory のリファクタリング
    ↓
Step 2: benchmarks/common/engines.py の更新
    ↓
Step 3: Examples の更新
    ↓
Step 4: CLI の更新（--dump-config 削除）
    ↓
Step 5: テストの削除・更新
    ↓
Step 6: config/ ディレクトリの削除
    ↓
Step 7: livecap_core/config/ の削除
    ↓
Step 8: 全テスト実行・確認
```

---

## 6. 検証項目

### 6.1 単体テスト

- [ ] `test_engine_factory.py` がパス
- [ ] Config 関連テストを削除済み

### 6.2 統合テスト

- [ ] `test_smoke_engines.py` がパス
- [ ] `test_file_transcription_pipeline.py` がパス
- [ ] `test_e2e_realtime_flow.py` がパス

### 6.3 ベンチマーク

- [ ] ASR ベンチマークが動作
- [ ] VAD ベンチマークが動作
- [ ] 最適化ベンチマークが動作

### 6.4 Examples 動作確認

- [ ] `basic_file_transcription.py` が動作
- [ ] `async_microphone.py` が動作
- [ ] `callback_api.py` が動作
- [ ] `custom_vad_config.py` が動作

### 6.5 CLI

- [ ] `livecap-core --info` が動作
- [ ] `livecap-core --ensure-ffmpeg` が動作

---

## 7. 削除対象の完全リスト

### 7.1 ファイル削除

| ファイル | 理由 |
|----------|------|
| `config/__init__.py` | Config 廃止 |
| `config/core_config_builder.py` | Config 廃止 |
| `livecap_core/config/defaults.py` | Config 廃止 |
| `livecap_core/config/schema.py` | Config 廃止 |
| `livecap_core/config/validator.py` | Config 廃止 |
| `tests/core/config/test_config_defaults.py` | Config 廃止 |
| `tests/core/config/test_core_config_builder.py` | Config 廃止 |

### 7.2 コード削除

| ファイル | 削除内容 |
|----------|----------|
| `engines/engine_factory.py` | `_prepare_config()`, `build_core_config` インポート |
| `livecap_core/cli.py` | `--dump-config`, `ConfigValidator` |
| `livecap_core/transcription/file_pipeline.py` | `config` パラメータ |

---

## 8. 完了条件

- [ ] `DEFAULT_CONFIG` が完全に削除されている
- [ ] `config/` ディレクトリが削除されている
- [ ] `livecap_core/config/` が削除または空になっている
- [ ] `EngineFactory` が Config なしで動作する
- [ ] 全テストがパス
- [ ] 全ベンチマークが動作
- [ ] Examples が動作

---

## 9. リスクと対策

| リスク | レベル | 対策 |
|--------|--------|------|
| 見落としたコード依存 | 低 | Grep で網羅的に検索済み（下記参照） |
| エンジン固有設定の欠落 | 中 | 各エンジンの使用状況を個別確認 |
| テスト失敗 | 中 | 段階的に実行、各ステップで確認 |
| Examples 動作不良 | 低 | 全 Examples の動作確認を検証項目に含む |

---

## 10. 影響調査結果

### 10.1 削除対象ファイル（影響なし）

Config 廃止に伴い削除するファイル。これらは他から参照されないため影響なし。

| ファイル | 理由 |
|----------|------|
| `config/__init__.py` | Config 廃止 |
| `config/core_config_builder.py` | Config 廃止 |
| `livecap_core/config/defaults.py` | Config 廃止 |
| `livecap_core/config/schema.py` | Config 廃止 |
| `livecap_core/config/validator.py` | Config 廃止 |
| `tests/core/config/test_config_defaults.py` | Config 廃止 |
| `tests/core/config/test_core_config_builder.py` | Config 廃止 |

### 10.2 更新が必要なファイル

Config を参照している箇所と、具体的な変更内容。

#### コードファイル

| ファイル | 現在の使用 | 変更内容 |
|----------|-----------|----------|
| `engines/engine_factory.py` | `build_core_config()` 呼び出し | `LANGUAGE_DEFAULTS` クラス定数に置き換え |
| `livecap_core/cli.py` | `--dump-config`, `ConfigValidator` | `--info` に置き換え、Validator 削除 |
| `examples/realtime/basic_file_transcription.py` | `get_default_config()` | 直接パラメータ指定に変更 |
| `examples/realtime/async_microphone.py` | `get_default_config()` | 直接パラメータ指定に変更 |
| `examples/realtime/callback_api.py` | `get_default_config()` | 直接パラメータ指定に変更 |
| `examples/realtime/custom_vad_config.py` | `get_default_config()` | 直接パラメータ指定に変更 |
| `tests/integration/engines/test_smoke_engines.py` | `_build_config()` 関数 | `language` 引数で直接指定 |
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
| `livecap_core/vad/config.py` | VADConfig dataclass（維持対象） |
| `benchmarks/common/engines.py` | 別の config 変数（`language` 引数化で対応済み） |

### 10.4 評価サマリー

- **削除ファイル**: 7 ファイル
- **更新ファイル（コード）**: 9 ファイル
- **更新ファイル（ドキュメント）**: 6 ファイル
- **影響範囲**: 限定的、安全に実装可能

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
