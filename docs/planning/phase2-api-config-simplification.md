# Phase 2: Config 廃止と API 簡素化 実装計画

> **Status**: 📋 PLANNING
> **作成日:** 2025-12-01
> **更新日:** 2025-12-01
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
| `EngineFactory.create_engine()` | `language_engines` マッピング | クラス定数に移動 |
| `EngineFactory._configure_engine_specific_settings()` | エンジン固有設定 | コンストラクタ引数で対応 |
| `benchmarks/common/engines.py` | `transcription.input_language` | 引数で直接指定 |
| `cli.py --dump-config` | 診断出力 | `--info` に置き換え |
| `examples/*.py` | 設定の取得 | 直接パラメータ指定 |

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

    # 言語別デフォルトエンジン（クラス定数）
    LANGUAGE_DEFAULTS: dict[str, str] = {
        "ja": "reazonspeech",
        "en": "parakeet",
        "zh": "whispers2t_base",
        "ko": "whispers2t_base",
        "de": "voxtral",
        "fr": "voxtral",
        "es": "voxtral",
        "default": "whispers2t_base",
    }

    @classmethod
    def create_engine(
        cls,
        engine_type: str = "auto",
        device: str | None = None,
        language: str = "ja",
        **engine_options,
    ) -> BaseEngine:
        """
        エンジンを作成

        Args:
            engine_type: エンジンタイプ（"auto" で言語から自動選択）
            device: デバイス（"cuda", "cpu", None=自動）
            language: 入力言語（engine_type="auto" 時に使用）
            **engine_options: エンジン固有オプション
                - model_size: WhisperS2T 用
                - model_name: Parakeet/Voxtral 用

        Returns:
            BaseEngine インスタンス
        """
        if engine_type == "auto":
            engine_type = cls.LANGUAGE_DEFAULTS.get(
                language,
                cls.LANGUAGE_DEFAULTS["default"]
            )
        ...

    @classmethod
    def get_default_engine(cls, language: str) -> str:
        """言語のデフォルトエンジンを取得"""
        return cls.LANGUAGE_DEFAULTS.get(language, cls.LANGUAGE_DEFAULTS["default"])
```

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

---

## 4. 実装タスク

### 4.1 EngineFactory の簡素化

#### Task 1.1: EngineFactory のリファクタリング

**ファイル:** `engines/engine_factory.py`

変更内容:
- `_prepare_config()` を削除
- `build_core_config()` の呼び出しを削除
- `LANGUAGE_DEFAULTS` をクラス定数として定義
- `create_engine()` の引数を簡素化
- `_configure_engine_specific_settings()` を `**engine_options` で置き換え

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
engine = EngineFactory.create_engine(engine_id, device=device, language=language)
```

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
engine = EngineFactory.create_engine(engine_type, device=device, language=language)
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

| リスク | 対策 |
|--------|------|
| 見落としたコード依存 | Grep で `get_default_config`, `build_core_config`, `DEFAULT_CONFIG` を検索 |
| エンジン固有設定の欠落 | 各エンジンの使用状況を個別確認 |
| テスト失敗 | 段階的に実行、各ステップで確認 |

---

## 変更履歴

| 日付 | 変更内容 |
|------|----------|
| 2025-12-01 | 初版作成（Config 簡素化計画） |
| 2025-12-01 | **方針転換: Config 廃止に変更** |
