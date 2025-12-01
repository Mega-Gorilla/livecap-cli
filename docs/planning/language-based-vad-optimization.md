# 言語別VAD最適化 実装計画

Issue: #139
Status: **PLANNING**

## 確定事項

| 項目 | 決定 | 備考 |
|------|------|------|
| TenVADライセンス警告 | そのまま出力 | ライセンス上の重要な警告のため |
| VADデフォルトインストール | JaVAD以外を`[vad]`に含める | 言語別VAD最適化の前提条件 |
| 未サポート言語 | Sileroフォールバック + INFOログ | "No optimized preset registered for language '...', falling back to Silero" |
| APIの公開範囲 | 追加エクスポート不要 | `VADProcessor`は既にエクスポート済み |
| テスト依存関係 | JaVAD以外は必須 | スキップ不要 |
| StreamTranscriber言語不一致 | **Option B: StreamTranscriberには追加しない** | 責任の分離、明示的な設定を優先 |

## 概要

Phase D VADパラメータ最適化（#126）の調査結果に基づき、言語別に最適なVADバックエンドを自動選択する機能を実装する。

### 背景

#126 で実施したベンチマーク結果により、言語別の最適VADが明確になった：

| 言語 | 最適VAD | スコア | Silero比較 |
|------|---------|--------|------------|
| 日本語 | **tenvad** | 7.2% CER | -1.9% |
| 英語 | **webrtc** | 3.3% WER | -2.6% |

現在のデフォルト（Silero）は汎用的だが、言語固有の最適化で大幅な精度改善が可能。

## 現状分析

### 既存実装

#### VADProcessor (`livecap_core/vad/processor.py`)
- `backend` パラメータで任意のVADバックエンドを注入可能
- デフォルトは Silero VAD
- `config` パラメータで VADConfig を指定可能

```python
class VADProcessor:
    def __init__(
        self,
        config: Optional[VADConfig] = None,
        backend: Optional[VADBackend] = None,
    ):
        ...
```

#### presets.py (`livecap_core/vad/presets.py`)
- `VAD_OPTIMIZED_PRESETS`: 最適化済みパラメータの辞書
- `get_best_vad_for_language(language)`: 言語に最適なVADを返す（**既存**）
- `get_optimized_preset(vad_type, language)`: 特定VAD+言語のプリセット取得

```python
# 既存の関数
def get_best_vad_for_language(language: str) -> tuple[str, dict[str, Any]] | None:
    """Get the best performing VAD preset for a language."""
    ...
```

#### StreamTranscriber (`livecap_core/transcription/stream.py`)
- `vad_processor` パラメータで外部からVADProcessorを注入可能
- `vad_config` パラメータでVAD設定を指定可能
- 現在 `language` パラメータは存在しない

```python
class StreamTranscriber:
    def __init__(
        self,
        engine: TranscriptionEngine,
        vad_config: Optional[VADConfig] = None,
        vad_processor: Optional[VADProcessor] = None,
        source_id: str = "default",
        max_workers: int = 1,
    ):
        ...
```

#### VADバックエンド
| バックエンド | クラス | パラメータ |
|-------------|--------|-----------|
| Silero | `SileroVAD` | `threshold`, `onnx` |
| TenVAD | `TenVAD` | `hop_size`, `threshold` |
| WebRTC | `WebRTCVAD` | `mode`, `frame_duration_ms` |

## 設計

### Phase 1: VADProcessor.from_language()

`VADProcessor` にクラスメソッドを追加：

```python
@classmethod
def from_language(
    cls,
    language: str,
    fallback_to_silero: bool = True,
) -> "VADProcessor":
    """言語に最適なVADを使用してVADProcessorを作成

    Args:
        language: 言語コード ("ja", "en", etc.)
        fallback_to_silero: バックエンドが利用できない場合にSileroにフォールバック

    Returns:
        VADProcessor with optimized backend and config

    Raises:
        ImportError: fallback_to_silero=False で必要なバックエンドがない場合
    """
```

#### 実装ロジック

```python
@classmethod
def from_language(cls, language: str, fallback_to_silero: bool = True) -> "VADProcessor":
    from .presets import get_best_vad_for_language
    from .config import VADConfig

    # 1. 最適なVADとプリセットを取得
    result = get_best_vad_for_language(language)

    if result is None:
        # プリセットがない言語 → Sileroにフォールバック
        logger.info(
            f"No optimized preset registered for language '{language}', "
            "falling back to Silero"
        )
        return cls()

    vad_type, preset = result
    vad_config = VADConfig.from_dict(preset["vad_config"])
    backend_params = preset.get("backend", {})

    # 2. バックエンドを作成
    backend = cls._create_backend(vad_type, backend_params, fallback_to_silero)

    logger.info(f"Selected {vad_type} optimized for language '{language}'")
    return cls(config=vad_config, backend=backend)

@classmethod
def _create_backend(
    cls,
    vad_type: str,
    backend_params: dict,
    fallback_to_silero: bool,
) -> VADBackend:
    """バックエンドを作成、必要に応じてフォールバック"""
    try:
        if vad_type == "silero":
            from .backends.silero import SileroVAD
            return SileroVAD(**backend_params)
        elif vad_type == "tenvad":
            from .backends.tenvad import TenVAD
            return TenVAD(**backend_params)
        elif vad_type == "webrtc":
            from .backends.webrtc import WebRTCVAD
            return WebRTCVAD(**backend_params)
        else:
            raise ValueError(f"Unknown VAD type: {vad_type}")
    except ImportError as e:
        if fallback_to_silero:
            logger.warning(f"{vad_type} not available, falling back to Silero: {e}")
            from .backends.silero import SileroVAD
            return SileroVAD()
        raise
```

### Phase 2: 使用例とドキュメント

**設計決定: Option B** - StreamTranscriberには`language`パラメータを追加しない

理由：
1. **責任の分離**: VAD設定は`VADProcessor`、`StreamTranscriber`はVAD+ASRの統合に専念
2. **明示的な設定**: ユーザーが意識的に言語を選択、設定内容が明確
3. **既存APIとの一貫性**: `vad_processor`パラメータによる注入パターンを踏襲
4. **言語不一致問題の回避**: エンジンとVADの言語が異なる場合の検証が不要

#### 推奨される使用パターン

```python
from livecap_core.vad import VADProcessor
from livecap_core.transcription import StreamTranscriber
from engines import EngineFactory

# 1. 言語に最適化されたVADを作成
vad = VADProcessor.from_language("ja")

# 2. エンジンを作成
engine = EngineFactory.create_engine("parakeet_ja", device="cuda")
engine.load_model()

# 3. StreamTranscriberにVADを注入
transcriber = StreamTranscriber(
    engine=engine,
    vad_processor=vad,
)

# 4. 文字起こし実行
with MicrophoneSource() as mic:
    for result in transcriber.transcribe_sync(mic):
        print(f"[{result.start_time:.2f}s] {result.text}")
```

この設計により：
- VADとエンジンの言語設定が分離され、それぞれ独立して設定可能
- ユーザーは設定内容を明確に理解できる
- 既存の`vad_processor`パラメータとの整合性が保たれる

### Phase 3: テスト

#### ユニットテスト (`tests/vad/test_processor.py` に追加)

```python
class TestVADProcessorFromLanguage:
    """from_language ファクトリメソッドのテスト"""

    def test_from_language_ja_uses_tenvad(self):
        """日本語はTenVADを使用"""
        # TenVADが利用可能な場合
        processor = VADProcessor.from_language("ja")
        assert "tenvad" in processor.backend_name

    def test_from_language_en_uses_webrtc(self):
        """英語はWebRTCを使用"""
        processor = VADProcessor.from_language("en")
        assert "webrtc" in processor.backend_name

    def test_from_language_unknown_falls_back_to_silero(self):
        """未知の言語はSileroにフォールバック"""
        processor = VADProcessor.from_language("zh")
        assert "silero" in processor.backend_name

    def test_from_language_fallback_when_backend_unavailable(self):
        """バックエンドが利用できない場合のフォールバック"""
        # モックで依存関係がないシナリオをテスト
        ...

    def test_from_language_no_fallback_raises(self):
        """fallback_to_silero=Falseで依存関係がない場合は例外"""
        with pytest.raises(ImportError):
            VADProcessor.from_language("ja", fallback_to_silero=False)
```

#### 統合テスト (`tests/vad/test_from_language_integration.py`)

```python
class TestVADProcessorFromLanguageIntegration:
    """from_language() の統合テスト"""

    def test_from_language_with_stream_transcriber(self):
        """StreamTranscriberとの統合動作確認"""
        vad = VADProcessor.from_language("ja")
        transcriber = StreamTranscriber(
            engine=mock_engine,
            vad_processor=vad,
        )
        assert "tenvad" in transcriber._vad.backend_name

    def test_from_language_processes_audio_correctly(self):
        """最適化VADで音声処理が正常に動作"""
        vad = VADProcessor.from_language("ja")
        # 実際の音声データでセグメント検出をテスト
        segments = vad.process_chunk(test_audio, sample_rate=16000)
        assert segments is not None
```

## タスク分解

### Phase 1: VADProcessor.from_language() (推定: 2-3h)

- [ ] `livecap_core/vad/processor.py`
  - [ ] `from_language()` クラスメソッド追加
  - [ ] `_create_backend()` ヘルパーメソッド追加
  - [ ] ログ出力追加（選択されたVAD、フォールバック時の警告）

- [ ] `livecap_core/vad/__init__.py`
  - [ ] エクスポート確認（変更不要の可能性）

- [ ] `tests/vad/test_processor.py`
  - [ ] `TestVADProcessorFromLanguage` クラス追加
  - [ ] 正常系テスト（ja, en）
  - [ ] フォールバックテスト
  - [ ] 例外テスト

### Phase 2: 統合テスト (推定: 1h)

- [ ] `tests/vad/test_from_language_integration.py`
  - [ ] StreamTranscriberとの統合テスト
  - [ ] 実際の音声処理テスト

**Note**: StreamTranscriberへの`language`パラメータ追加は行わない（Option B決定）

### Phase 3: ドキュメント・仕上げ (推定: 1.5h)

- [ ] `docs/guides/vad-optimization.md` 更新
  - [ ] `from_language()` 使用例追加
  - [ ] StreamTranscriberとの統合例追加

- [ ] `examples/realtime/custom_vad_config.py` 更新
  - [ ] `--language` オプション追加（`VADProcessor.from_language()` を使用）
  - [ ] 言語別最適化プロファイル例の追加
  - [ ] 使用例をdocstringに追記

- [ ] `livecap_core/vad/__init__.py` docstring更新

- [ ] Issue #139 更新
  - [ ] 完了報告
  - [ ] クローズ

## リスクと軽減策

### リスク1: TenVAD/WebRTCの依存関係

**リスク**: TenVAD (`ten-vad`) や WebRTC (`webrtcvad`) がインストールされていない環境でのエラー

**軽減策** (確定):
- `[vad]` に TenVAD/WebRTC を含める（JaVAD以外は必須）
- `fallback_to_silero=True` をデフォルトに（万が一の場合）
- TenVADのライセンス警告はそのまま出力

### リスク2: presets.pyの言語カバレッジ

**リスク**: ja, en 以外の言語に対するプリセットがない

**軽減策** (確定):
- 未知の言語はSileroにフォールバック
- INFOログで通知: "No optimized preset registered for language '{lang}', falling back to Silero"
- 将来的に他言語のベンチマークを実施してプリセット追加可能

## 完了条件

- [ ] `presets.py` のスコアがPhase D-4の結果に更新されている
- [ ] `pyproject.toml` の `[vad]` に TenVAD/WebRTC が含まれている
- [ ] `VADProcessor.from_language("ja")` で TenVAD が使用される
- [ ] `VADProcessor.from_language("en")` で WebRTC が使用される
- [ ] 未サポート言語でINFOログが出力され、Sileroにフォールバック
- [ ] フォールバック機構が正常に動作する
- [ ] StreamTranscriberへの`vad_processor`注入で正常動作
- [ ] 全テストがパス
- [ ] CI がパス
- [ ] ドキュメント更新済み

## 関連ファイル

| ファイル | 変更内容 |
|---------|---------|
| `livecap_core/vad/presets.py` | スコア更新（Phase 0） |
| `pyproject.toml` | VAD依存関係更新（Phase 0） |
| `livecap_core/vad/processor.py` | `from_language()` 追加（Phase 1） |
| `tests/vad/test_processor.py` | ユニットテスト追加（Phase 1） |
| `tests/vad/test_from_language_integration.py` | 統合テスト追加（Phase 2） |
| `docs/guides/vad-optimization.md` | 使用例追加（Phase 3） |
| `examples/realtime/custom_vad_config.py` | `--language`オプション追加（Phase 3） |

**変更なし**: `livecap_core/transcription/stream.py` - Option B採用により変更不要

## 前提タスク: presets.pyスコア更新

### 問題

`presets.py` のスコアはPhase D-2（Bayesian最適化時）の値で、Phase D-4（実ベンチマーク）の結果と乖離がある。

| VAD | 言語 | presets.py (D-2) | Benchmark (D-4) |
|-----|------|-----------------|-----------------|
| silero | ja | 6.47% | 8.2% |
| tenvad | ja | 7.06% | **7.2%** |
| webrtc | ja | 7.05% | 7.7% |
| silero | en | 3.96% | 4.0% |
| tenvad | en | 3.40% | 3.4% |
| webrtc | en | 3.31% | **3.3%** |

**影響**: `get_best_vad_for_language("ja")` が silero を返すが、実際の最適は tenvad。

### 更新内容

`livecap_core/vad/presets.py` の `metadata.score` をPhase D-4の結果で更新：

```python
VAD_OPTIMIZED_PRESETS = {
    "silero": {
        "ja": {
            "metadata": {
                "score": 0.082,  # 6.47% → 8.2%
                ...
            },
        },
        "en": {
            "metadata": {
                "score": 0.040,  # 3.96% → 4.0%
                ...
            },
        },
    },
    "tenvad": {
        "ja": {
            "metadata": {
                "score": 0.072,  # 7.06% → 7.2%
                ...
            },
        },
        "en": {
            "metadata": {
                "score": 0.034,  # 3.40% → 3.4%
                ...
            },
        },
    },
    "webrtc": {
        "ja": {
            "metadata": {
                "score": 0.077,  # 7.05% → 7.7%
                ...
            },
        },
        "en": {
            "metadata": {
                "score": 0.033,  # 3.31% → 3.3%
                ...
            },
        },
    },
}
```

### タスク追加

**Phase 0: 前提タスク** (推定: 1h)

- [ ] `livecap_core/vad/presets.py`
  - [ ] metadata.score をPhase D-4の結果で更新
  - [ ] コメントに測定条件を追記（standard mode, parakeet系エンジン）
- [ ] `pyproject.toml` VAD依存関係の更新
  - [ ] `[vad]` に webrtcvad, ten-vad を追加（JaVAD以外を必須化）
  - [ ] `[vad-javad]` を新設（オプショナル）
- [ ] 動作確認
  - [ ] `get_best_vad_for_language("ja")` → tenvad
  - [ ] `get_best_vad_for_language("en")` → webrtc
  - [ ] `uv sync --extra vad` で TenVAD/WebRTC がインストールされる

## 参考

- Issue #126: VADパラメータ最適化
- Issue #64: Epic livecap-cli リファクタリング
- `livecap_core/vad/presets.py`: 最適化済みパラメータ
- VAD Benchmark Run #19782802125: Phase D-4 結果
