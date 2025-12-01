# 言語別VAD最適化 実装計画

Issue: #139
Status: **COMPLETED**

## 進捗

| Phase | 内容 | 状態 | PR |
|-------|------|------|-----|
| Phase 0 | 前提タスク（presets更新、VAD依存関係バンドル） | ✅ 完了 | #143 |
| Phase 1 | VADProcessor.from_language() 実装 | ✅ 完了 | #145 |
| Phase 2 | 統合テスト | ✅ 完了 | #146 |
| Phase 3 | ドキュメント・仕上げ | ✅ 完了 | #151 |

## 確定事項

| 項目 | 決定 | 備考 |
|------|------|------|
| TenVADライセンス警告 | そのまま出力 | ライセンス上の重要な警告のため |
| VADデフォルトインストール | JaVAD以外をデフォルト依存関係に含める | 言語別VAD最適化の前提条件 |
| 未サポート言語 | **ValueError例外をraise** | サポート言語: ja, en のみ。他言語は`VADProcessor()`を使用 |
| フォールバック機構 | **廃止** | `fallback_to_silero`パラメータは実装しない。明示的なエラーを優先 |
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
def from_language(cls, language: str) -> "VADProcessor":
    """言語に最適なVADを使用してVADProcessorを作成

    Args:
        language: 言語コード ("ja", "en")

    Returns:
        VADProcessor with optimized backend and config

    Raises:
        ValueError: 未サポート言語の場合
        ImportError: 必要なVADバックエンドがインストールされていない場合
    """
```

#### 実装ロジック

```python
@classmethod
def from_language(cls, language: str) -> "VADProcessor":
    from .presets import get_available_presets, get_best_vad_for_language

    # 1. 最適なVADとプリセットを取得
    result = get_best_vad_for_language(language)

    if result is None:
        # プリセットがない言語 → エラー（サポート言語を動的に取得）
        available = get_available_presets()
        supported = sorted(set(lang for _, lang in available))
        raise ValueError(
            f"No optimized preset for language '{language}'. "
            f"Supported languages: {', '.join(supported)}. "
            f"Use VADProcessor() for default Silero VAD."
        )

    vad_type, preset = result
    vad_config = VADConfig.from_dict(preset["vad_config"])
    backend_params = preset.get("backend", {})

    # 2. バックエンドを作成
    backend = cls._create_backend(vad_type, backend_params)

    logger.info(f"Created VADProcessor with {vad_type} optimized for '{language}'")
    return cls(config=vad_config, backend=backend)

@classmethod
def _create_backend(cls, vad_type: str, backend_params: dict) -> VADBackend:
    """バックエンドを作成

    Args:
        vad_type: VADバックエンドの種類 ("silero", "tenvad", "webrtc")
        backend_params: プリセットから取得したバックエンド固有パラメータ

    Raises:
        ImportError: VADバックエンドがインストールされていない場合
        ValueError: 未知のVADタイプの場合

    Note:
        SileroVADのthresholdはVADConfigで管理されるため、
        バックエンドには渡さない（onnx=Trueのみ指定）
    """
    if vad_type == "silero":
        from .backends.silero import SileroVAD
        return SileroVAD(onnx=True, **backend_params)
    elif vad_type == "tenvad":
        from .backends.tenvad import TenVAD
        return TenVAD(**backend_params)
    elif vad_type == "webrtc":
        from .backends.webrtc import WebRTCVAD
        return WebRTCVAD(**backend_params)
    else:
        raise ValueError(f"Unknown VAD type: {vad_type}")
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
        processor = VADProcessor.from_language("ja")
        assert "tenvad" in processor.backend_name

    def test_from_language_en_uses_webrtc(self):
        """英語はWebRTCを使用"""
        processor = VADProcessor.from_language("en")
        assert "webrtc" in processor.backend_name

    def test_from_language_unsupported_raises_valueerror(self):
        """未サポート言語はValueError"""
        with pytest.raises(ValueError, match="No optimized preset"):
            VADProcessor.from_language("zh")

    def test_from_language_missing_backend_raises_importerror(self):
        """バックエンドがインストールされていない場合はImportError"""
        # モックでTenVADがインストールされていないシナリオをテスト
        with pytest.raises(ImportError):
            ...
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

### Phase 1: VADProcessor.from_language() ✅ 完了 (PR #145)

- [x] `livecap_core/vad/processor.py`
  - [x] `from_language()` クラスメソッド追加
  - [x] `_create_backend()` ヘルパーメソッド追加
  - [x] INFOログ出力追加（選択されたVAD）

- [x] `livecap_core/vad/__init__.py`
  - [x] エクスポート確認（変更不要）

- [x] `tests/vad/test_processor.py`
  - [x] `TestVADProcessorFromLanguage` クラス追加
  - [x] 正常系テスト（ja → TenVAD, en → WebRTC）
  - [x] 未サポート言語でValueErrorテスト
  - [x] プリセットパラメータ検証テスト（vad_config, backend params）
  - [x] `TestVADProcessorCreateBackend` クラス追加

- [x] `pyproject.toml` VAD依存関係リファクタリング
  - [x] silero-vad, webrtcvad, ten-vad をデフォルト依存関係に移動
  - [x] `[vad]`, `[vad-javad]`, `[vad-all]` extras を削除
  - [x] javad を `benchmark` / `optimization` extras に統合

- [x] CI ワークフロー更新
  - [x] `core-tests.yml` から `--extra vad` を削除
  - [x] `integration-tests.yml` から `--extra vad` を削除
  - [x] `benchmark-asr.yml` から `[vad]` を削除
  - [x] `benchmark-vad.yml` から `[vad]` を削除

- [x] `README.md` 更新
  - [x] インストール手順から `--extra vad` を削除
  - [x] extras テーブルを更新

### Phase 2: 統合テスト ✅ 完了 (PR #146)

- [x] `tests/integration/vad/test_from_language_integration.py`
  - [x] StreamTranscriberとの統合テスト（ja/en）
  - [x] 実際の音声処理・セグメント検出テスト
  - [x] コールバック方式フローテスト
  - [x] VADSegmentバリデーションテスト

- [x] テストインフラ修正
  - [x] `tests/__init__.py`, `tests/integration/__init__.py` 追加（モジュール解決修正）
  - [x] `tests/vad/test_backends.py` Silero fixture スキップ条件修正
  - [x] `tests/vad/test_processor.py` AttributeError ハンドリング追加

**Note**: StreamTranscriberへの`language`パラメータ追加は行わない（Option B決定）

### Phase 3: ドキュメント・仕上げ ✅ 完了 (PR #151)

- [x] `docs/guides/vad-optimization.md` タイトル修正
  - [x] 「VAD Bayesian 最適化によるパラメータチューニングガイド」にリネーム
  - [x] `realtime-transcription.md` への参照リンク追加

- [x] `docs/guides/realtime-transcription.md` 更新
  - [x] インストール手順更新（`[vad]` extra 不要）
  - [x] 「言語別 VAD 最適化（推奨）」セクション追加
  - [x] `VADProcessor.from_language()` の使い方
  - [x] サポート言語と推奨 VAD の表
  - [x] エラーハンドリング（ValueError）
  - [x] TenVAD ライセンス警告の説明
  - [x] 関連ドキュメントセクション更新

- [x] `examples/realtime/custom_vad_config.py` 更新
  - [x] `--language` オプション追加（`VADProcessor.from_language()` を使用）
  - [x] 使用例を docstring に追記

- [x] `livecap_core/vad/__init__.py` docstring 更新
  - [x] `from_language()` の使い方追加
  - [x] サポート言語一覧追加

- [ ] Issue #139 更新
  - [ ] 完了報告
  - [ ] クローズ

## リスクと軽減策

### リスク1: TenVAD/WebRTCの依存関係

**リスク**: TenVAD (`ten-vad`) や WebRTC (`webrtcvad`) がインストールされていない環境でのエラー

**軽減策** (確定):
- TenVAD/WebRTC をデフォルト依存関係に含める（JaVAD以外は必須）
- TenVADのライセンス警告はそのまま出力

### リスク2: presets.pyの言語カバレッジ

**リスク**: ja, en 以外の言語に対するプリセットがない

**軽減策** (確定):
- 未サポート言語は `ValueError` で明示的にエラー
- エラーメッセージに代替手段を提示: "Use VADProcessor() for default Silero VAD"
- 将来的に他言語のベンチマークを実施してプリセット追加可能

## 完了条件

- [x] `presets.py` のスコアがPhase D-4の結果に更新されている (Phase 0)
- [x] `pyproject.toml` のデフォルト依存関係に TenVAD/WebRTC が含まれている (Phase 1, PR #145)
- [x] `VADProcessor.from_language("ja")` で TenVAD が使用される (Phase 1, PR #145)
- [x] `VADProcessor.from_language("en")` で WebRTC が使用される (Phase 1, PR #145)
- [x] `VADProcessor.from_language("zh")` で `ValueError` が発生する (Phase 1, PR #145)
- [x] プリセットパラメータが正しく適用される（vad_config, backend params） (Phase 1, PR #145)
- [x] StreamTranscriberへの`vad_processor`注入で正常動作 (Phase 2, PR #146)
- [x] 全テストがパス (Phase 2, PR #146)
- [x] CI がパス (Phase 1, PR #145)
- [x] ドキュメント更新済み (Phase 3, PR #151)

## 関連ファイル

| ファイル | 変更内容 | Phase | 状態 |
|---------|---------|-------|------|
| `livecap_core/vad/presets.py` | スコア更新 | 0 | ✅ |
| `pyproject.toml` | VAD依存関係をデフォルトに移動、extras削除 | 0, 1 | ✅ |
| `.github/workflows/core-tests.yml` | libc++インストール追加、`--extra vad`削除 | 0, 1 | ✅ |
| `.github/workflows/integration-tests.yml` | `--extra vad`削除 | 1 | ✅ |
| `.github/workflows/benchmark-asr.yml` | `[vad]`削除 | 1 | ✅ |
| `.github/workflows/benchmark-vad.yml` | `[vad]`削除 | 1 | ✅ |
| `README.md` | VAD説明更新、libc++手順追加、extras更新 | 0, 1 | ✅ |
| `tests/core/vad/test_presets.py` | Phase D-4スコア対応 | 0 | ✅ |
| `livecap_core/vad/processor.py` | `from_language()`, `_create_backend()` 追加 | 1 | ✅ |
| `tests/vad/test_processor.py` | ユニットテスト追加（11テスト）、AttributeError ハンドリング追加 | 1, 2 | ✅ |
| `tests/vad/test_backends.py` | Silero fixture スキップ条件修正 | 2 | ✅ |
| `tests/integration/vad/test_from_language_integration.py` | 統合テスト追加（7テスト） | 2 | ✅ |
| `tests/__init__.py` | pytest モジュール解決修正 | 2 | ✅ |
| `tests/integration/__init__.py` | pytest モジュール解決修正 | 2 | ✅ |
| `tests/integration/vad/__init__.py` | VAD 統合テストパッケージ | 2 | ✅ |
| `docs/guides/vad-optimization.md` | タイトル修正、realtime-transcription.mdへの参照追加 | 3 | ✅ |
| `docs/guides/realtime-transcription.md` | 言語別VAD最適化セクション追加、インストール手順更新 | 3 | ✅ |
| `examples/realtime/custom_vad_config.py` | `--language`オプション追加 | 3 | ✅ |
| `livecap_core/vad/__init__.py` | docstring更新（`from_language()` 使用例追加） | 3 | ✅ |

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

**Phase 0: 前提タスク** ✅ 完了 (PR #143)

- [x] `livecap_core/vad/presets.py`
  - [x] metadata.score をPhase D-4の結果で更新
  - [x] コメントに測定条件を追記（standard mode, parakeet系エンジン）
- [x] `pyproject.toml` VAD依存関係の更新
  - [x] silero-vad, webrtcvad, ten-vad をデフォルト依存関係に移動（PR #145 で更新）
  - [x] javad を `benchmark` / `optimization` extras に統合
  - [x] libc++要件をコメントで追記
- [x] `.github/workflows/core-tests.yml`
  - [x] libc++1インストールステップ追加（TenVADテストを有効化）
- [x] `README.md`
  - [x] VAD extra説明を更新（silero-vad, webrtcvad, ten-vad）
  - [x] libc++インストール手順を追加
- [x] `tests/core/vad/test_presets.py`
  - [x] Phase D-4スコアに合わせてテスト更新
- [x] 動作確認
  - [x] `get_best_vad_for_language("ja")` → tenvad
  - [x] `get_best_vad_for_language("en")` → webrtc
  - [x] `uv sync` で TenVAD/WebRTC がインストールされる（デフォルト依存）
  - [x] CIでTenVADテストが実行される（OSErrorスキップではなく実際にテスト）

## 参考

- Issue #126: VADパラメータ最適化
- Issue #64: Epic livecap-cli リファクタリング
- `livecap_core/vad/presets.py`: 最適化済みパラメータ
- VAD Benchmark Run #19782802125: Phase D-4 結果
