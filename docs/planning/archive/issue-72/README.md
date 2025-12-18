# Issue #72: 翻訳プラグインシステム実装

## 概要

翻訳プラグインシステムを設計し、Google Translate、OPUS-MT、Riva-Translate-4B-Instruct の3つの翻訳エンジンを実装する。

## 関連ドキュメント

| ドキュメント | 内容 |
|-------------|------|
| [`../issue-72-architecture.md`](../issue-72-architecture.md) | 技術設計（Phase 1-4 完了分） |
| [`../issue-72-examples.md`](../issue-72-examples.md) | 使用例 |
| [`../issue-72-testing.md`](../issue-72-testing.md) | テスト計画 |

## 実装ステータス

### Phase 1-4 完了（2025-12-11）

| コンポーネント | ステータス | Phase | 備考 |
|---------------|-----------|-------|------|
| `livecap_cli/translation/` | ✅ 完了 | 1 | ディレクトリ作成 |
| `translation/base.py` | ✅ 完了 | 1 | BaseTranslator ABC |
| `translation/result.py` | ✅ 完了 | 1 | TranslationResult dataclass |
| `translation/metadata.py` | ✅ 完了 | 1 | TranslatorMetadata |
| `translation/factory.py` | ✅ 完了 | 1 | TranslatorFactory |
| `translation/exceptions.py` | ✅ 完了 | 1 | 例外クラス階層 |
| `translation/retry.py` | ✅ 完了 | 1 | リトライデコレータ |
| `translation/lang_codes.py` | ✅ 完了 | 1 | 言語コードユーティリティ |
| `translation/impl/google.py` | ✅ 完了 | 2 | GoogleTranslator |
| `translation/impl/opus_mt.py` | ✅ 完了 | 3 | OpusMTTranslator |
| `translation/impl/riva_instruct.py` | ✅ 完了 | 4 | RivaInstructTranslator |
| `utils/__init__.py` VRAM 追加 | ✅ 完了 | 1 | get_available_vram 等 |
| `pyproject.toml` 依存追加 | ✅ 完了 | 1 | translation-local, translation-riva |
| `translation/__init__.py` エクスポート | ✅ 完了 | 2 | TranslatorFactory 等（※1） |
| `tests/core/translation/` | ✅ 完了 | 2-4 | ユニットテスト (120+) |
| `tests/conftest.py` マーカー追加 | ✅ 完了 | 2 | network, slow, gpu |
| `examples/translation/` | ✅ 完了 | 4 | サンプルスクリプト (5件) |

**※1**: 翻訳 API は `livecap_cli.translation` パッケージからインポート。トップレベル `livecap_cli` へのエクスポートは Phase 6 で検討。

### Phase 5 完了（2025-12-12）

| コンポーネント | ステータス | Phase | 備考 |
|---------------|-----------|-------|------|
| `TranscriptionResult` 翻訳フィールド | ✅ 完了 | 5 | `translated_text`, `target_language` |
| `StreamTranscriber` translator 統合 | ✅ 完了 | 5 | パラメータ追加、バリデーション |
| 文脈バッファ管理 | ✅ 完了 | 5 | `deque(maxlen=100)` 実装 |
| 翻訳エラーハンドリング | ✅ 完了 | 5 | 警告ログ + `translated_text=None` |
| 翻訳タイムアウト | ✅ 完了 | 5 | 5秒タイムアウト実装 |
| `tests/core/transcription/test_stream_translation.py` | ✅ 完了 | 5 | 翻訳統合ユニットテスト |
| `tests/core/transcription/test_result.py` | ✅ 完了 | 5 | 翻訳フィールドテスト |
| `examples/realtime/realtime_translation.py` | ✅ 完了 | 5 | リアルタイム翻訳サンプル |

### Phase 6a 完了（2025-12-12）

| コンポーネント | ステータス | Phase | 備考 |
|---------------|-----------|-------|------|
| `FileTranscriptionPipeline` translator 統合 | ✅ 完了 | 6a | パラメータ追加、文脈管理 |
| `FileSubtitleSegment` 翻訳フィールド | ✅ 完了 | 6a | `translated_text`, `target_language` |
| 翻訳 SRT 出力オプション | ✅ 完了 | 6a | `write_translated_subtitles` パラメータ |
| `tests/core/transcription/test_file_pipeline_translation.py` | ✅ 完了 | 6a | ユニットテスト（25件） |
| `examples/batch/batch_translation.py` | ✅ 完了 | 6a | バッチ翻訳サンプル |

### Phase 6b スキップ

| コンポーネント | ステータス | Phase | 備考 |
|---------------|-----------|-------|------|
| トップレベルエクスポート | ⏭️ スキップ | 6b | 現状 `livecap_cli.translation` で十分 |

**スキップ理由**: 機能的な価値がなく、名前空間の肥大化を避けるため。ユーザー要望があれば実装。

### Phase 7 計画（将来）

| コンポーネント | ステータス | Phase | 備考 |
|---------------|-----------|-------|------|
| 非同期翻訳オプション | 📋 計画 | 7 | 必要性が確認されてから |

### 既存コード（参照のみ）

| コンポーネント | ステータス | ファイル |
|---------------|-----------|----------|
| `TranslationRequestEventDict` | ✅ 既存 | `transcription_types.py` |
| `TranslationResultEventDict` | ✅ 既存 | `transcription_types.py` |
| `create_translation_result_event()` | ✅ 既存 | `transcription_types.py` |
| `LoadPhase.TRANSLATION_MODEL` | ✅ 既存 | `model_loading_phases.py` |
| `EngineMetadata.to_iso639_1()` | ✅ 既存 | `engines/metadata.py` |

## Phase 5: StreamTranscriber 翻訳統合

StreamTranscriber にリアルタイム翻訳機能を統合し、ASR + 翻訳のシームレスなパイプラインを提供する。

### 設計決定事項

| 項目 | 決定 | 理由 |
|------|------|------|
| 統合方式 | translator パラメータ追加 | 継承より合成、後方互換性 |
| TranscriptionResult | `translated_text` + `target_language` 追加 | `language` を source として再利用 |
| デフォルト言語 | なし（translator 設定時は必須） | 明示的指定でミス防止 |
| context_sentences | translator のプロパティから取得 | 各エンジンに最適な設定を尊重 |
| コンテキストバッファ | `deque(maxlen=MAX)` で制限 | 長時間セッションのメモリ保護 |
| 翻訳エラー | `translated_text=None` + 警告ログ | 主機能（文字起こし）を保護 |
| 非同期翻訳 | Phase 5 では同期のみ | 複雑性を避け、Phase 6 で検討 |
| ライフサイクル | 呼び出し側が管理 | engine と同じパターン、一貫性 |

### 実装上の注意点

#### 1. context_sentences の公開アクセス

現在 `BaseTranslator._default_context_sentences` はプライベート属性。Phase 5 実装時に公開プロパティを追加:

```python
# BaseTranslator に追加
@property
def default_context_sentences(self) -> int:
    """文脈として使用するデフォルトの文数"""
    return self._default_context_sentences
```

#### 2. コンテキストバッファのサイズ制限

長時間セッションでのメモリ成長を防ぐため `collections.deque` を使用:

```python
from collections import deque

MAX_CONTEXT_BUFFER = 100  # 最大100文を保持

class StreamTranscriber:
    def __init__(self, ...):
        self._context_buffer: deque[str] = deque(maxlen=MAX_CONTEXT_BUFFER)
```

#### 3. 同期翻訳の性能制限

Phase 5 では同期翻訳のみサポート。以下の制限事項を認識:

| 翻訳エンジン | 想定レイテンシ | リアルタイム性への影響 |
|-------------|--------------|---------------------|
| Google | 100-300ms | 低（許容範囲） |
| OPUS-MT (CPU) | 50-200ms | 低（許容範囲） |
| Riva-4B (GPU) | 500-2000ms | **高**（ASR ブロック可能性） |

**軽減策**（Phase 5 暫定）:
- 翻訳タイムアウト（5秒）を設定し、超過時は `translated_text=None` で継続
- Riva-4B 使用時は警告ログを出力
- 本格的な非同期対応は Phase 6 で実装

#### 4. 言語ペアの事前バリデーション

`translator.get_supported_pairs()` が空でない場合、初期化時に警告:

```python
if translator:
    pairs = translator.get_supported_pairs()
    if pairs and (source_lang, target_lang) not in pairs:
        logger.warning(
            "Language pair (%s -> %s) may not be supported by %s",
            source_lang, target_lang, translator.get_translator_name()
        )
```

**Note**: Google は全ペア対応（`get_supported_pairs()` が空）のため、警告は出ない。

#### 5. 破壊的変更の影響範囲

`TranscriptionResult` は Phase 1 で追加された新 API のため、破壊的変更を容認:

| 変更対象 | 更新必要性 |
|---------|----------|
| `livecap_cli/transcription/result.py` | ✅ フィールド追加 |
| `livecap_cli/transcription/stream.py` | ✅ パラメータ追加、翻訳処理追加 |
| `tests/core/transcription/test_result.py` | ✅ 新フィールドのテスト追加 |
| `tests/core/transcription/test_stream.py` | ✅ 翻訳統合テスト追加 |
| 外部依存コード | ❌ なし（新 API のため） |

### 主要変更

1. **TranscriptionResult の拡張**
   ```python
   @dataclass(frozen=True, slots=True)
   class TranscriptionResult:
       text: str
       start_time: float
       end_time: float
       is_final: bool = True
       confidence: float = 1.0
       language: str = ""           # ASR 検出言語（= 翻訳元言語）
       source_id: str = "default"
       # Phase 5 追加
       translated_text: Optional[str] = None   # 翻訳結果
       target_language: Optional[str] = None   # 翻訳先言語
   ```

   **Note**: `source_language` は追加しない。既存の `language` フィールドを翻訳元言語として再利用。

2. **StreamTranscriber の拡張**
   ```python
   from collections import deque

   MAX_CONTEXT_BUFFER = 100

   class StreamTranscriber:
       def __init__(
           self,
           engine: TranscriptionEngine,
           translator: Optional[BaseTranslator] = None,
           source_lang: Optional[str] = None,  # translator 設定時は必須
           target_lang: Optional[str] = None,  # translator 設定時は必須
           vad_config: Optional[VADConfig] = None,
           ...
       ):
           self._translator = translator
           self._source_lang = source_lang
           self._target_lang = target_lang
           self._context_buffer: deque[str] = deque(maxlen=MAX_CONTEXT_BUFFER)

           # バリデーション
           if translator:
               if not translator.is_initialized():
                   raise ValueError("Translator not initialized. Call load_model() first.")
               if source_lang is None or target_lang is None:
                   raise ValueError("source_lang and target_lang are required when translator is set.")
               # 言語ペアの事前警告
               pairs = translator.get_supported_pairs()
               if pairs and (source_lang, target_lang) not in pairs:
                   logger.warning(
                       "Language pair (%s -> %s) may not be supported by %s",
                       source_lang, target_lang, translator.get_translator_name()
                   )
   ```

3. **翻訳パイプラインの追加**
   ```python
   def _process_segment(self, segment: VADSegment) -> TranscriptionResult:
       # ASR
       text, confidence = self._engine.transcribe(...)

       # 翻訳（translator が設定されている場合）
       translated_text = None
       target_language = None
       if self._translator and text.strip():
           try:
               # 公開プロパティから context_sentences を取得
               context_len = self._translator.default_context_sentences
               context = list(self._context_buffer)[-context_len:]
               trans_result = self._translator.translate(
                   text,
                   self._source_lang,
                   self._target_lang,
                   context=context,
               )
               translated_text = trans_result.text
               target_language = self._target_lang
               self._context_buffer.append(text)
           except TimeoutError:
               logger.warning("Translation timed out, continuing without translation")
           except Exception as e:
               logger.warning(f"Translation failed: {e}")
               # 翻訳失敗しても文字起こし結果は返す

       return TranscriptionResult(
           text=text,
           translated_text=translated_text,
           target_language=target_language,
           ...
       )
   ```

### 使用例

```python
from livecap_cli import StreamTranscriber, EngineFactory, MicrophoneSource
from livecap_cli.translation import TranslatorFactory

# ASR エンジン初期化
engine = EngineFactory.create_engine("whispers2t_base", device="cuda")
engine.load_model()

# Translator 初期化（呼び出し側がライフサイクル管理）
translator = TranslatorFactory.create_translator("google")
# ローカルモデルの場合: translator.load_model()

# StreamTranscriber に translator を渡す
with StreamTranscriber(
    engine=engine,
    translator=translator,
    source_lang="ja",   # 必須
    target_lang="en",   # 必須
) as transcriber:
    with MicrophoneSource() as mic:
        for result in transcriber.transcribe_sync(mic):
            print(f"[{result.language}] {result.text}")
            if result.translated_text:
                print(f"[{result.target_language}] {result.translated_text}")
            else:
                print("(translation unavailable)")

# クリーンアップ（呼び出し側が管理）
# translator.cleanup()  # ローカルモデルの場合
engine.cleanup()
```

### 翻訳なしモード（後方互換）

```python
# translator を渡さない場合は従来通り動作
with StreamTranscriber(engine=engine) as transcriber:
    for result in transcriber.transcribe_sync(mic):
        print(result.text)
        # result.translated_text は None
```

### 実装タスク

1. `BaseTranslator` に `default_context_sentences` プロパティ追加
2. `TranscriptionResult` に `translated_text`, `target_language` フィールド追加
3. `StreamTranscriber.__init__` に translator 関連パラメータ追加
4. 初期化時のバリデーション実装（言語ペア警告含む）
5. 文脈バッファ管理の実装（`deque(maxlen=MAX_CONTEXT_BUFFER)`）
6. `_transcribe_segment` / `_transcribe_segment_async` での翻訳処理追加
7. 翻訳タイムアウト処理の実装
8. 翻訳エラー時の警告ログ実装
9. ユニットテスト作成
10. 統合テスト作成
11. サンプルスクリプト作成

### 変更ファイル

| ファイル | 操作 | 説明 |
|---------|------|------|
| `livecap_cli/translation/base.py` | 更新 | `default_context_sentences` プロパティ追加 |
| `livecap_cli/transcription/result.py` | 更新 | 翻訳フィールド追加 |
| `livecap_cli/transcription/stream.py` | 更新 | translator 統合、deque バッファ、タイムアウト |
| `tests/core/translation/test_base.py` | 更新 | プロパティのテスト |
| `tests/core/transcription/test_result.py` | 更新 | 新フィールドのテスト |
| `tests/core/transcription/test_stream.py` | 更新 | 翻訳統合テスト |
| `tests/integration/test_stream_translation.py` | 新規 | ASR+翻訳統合テスト |
| `examples/realtime/realtime_translation.py` | 新規 | リアルタイム翻訳例 |

## Phase 6: FileTranscriptionPipeline 翻訳統合

Phase 5 で実装した StreamTranscriber の翻訳統合パターンを FileTranscriptionPipeline に適用する。

### 設計決定事項

| 項目 | 決定 | 理由 |
|------|------|------|
| 統合方式 | `process_file(s)` に translator パラメータ追加 | 現行 API 維持（`__init__` は `ffmpeg_manager/segmenter` のみ）|
| FileSubtitleSegment | `translated_text` + `target_language` を末尾 Optional 追加 | 既存フィールド (`index/start/end/text/metadata`) との互換性 |
| 文脈管理 | ファイル内バッファ、ファイル間リセット | バッチ処理に最適化 |
| タイムアウト | デフォルト無効の `translation_timeout` オプション | ネットワークハング対策、Phase 5 の `TRANSLATION_TIMEOUT` 再利用可 |
| SRT 出力 | 翻訳版を別ファイル出力オプション | 柔軟性確保 |
| 非同期翻訳 | **Phase 7 へ延期** | 必要性が確認されてから実装 |

### Phase 6a: FileTranscriptionPipeline 翻訳統合（必須）

#### FileSubtitleSegment の拡張

現行の `FileSubtitleSegment` は以下の構造（`slots=True`）:

```python
@dataclass(slots=True)
class FileSubtitleSegment:
    index: int
    start: float      # ※ start_time ではない
    end: float        # ※ end_time ではない
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
```

Phase 6a で末尾に Optional フィールドを追加:

```python
@dataclass(slots=True)
class FileSubtitleSegment:
    index: int
    start: float
    end: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # Phase 6a 追加（末尾 Optional で後方互換）
    translated_text: Optional[str] = None
    target_language: Optional[str] = None
```

**後方互換性の注意**:
- `FileSubtitleSegment` はトップレベル export 済み（`livecap_cli.FileSubtitleSegment`）
- 末尾 Optional 追加は安全だが、位置引数で全フィールドを渡している既存コードがあれば更新推奨
- `slots=True` のため、新フィールドは `__slots__` に自動追加される

#### FileTranscriptionPipeline の拡張

現行 API では `__init__` は `ffmpeg_manager/segmenter` のみを受け、ASR の `segment_transcriber` は `process_file(s)` に渡す設計。この設計を維持し、translator 系パラメータも `process_file(s)` に追加する。

```python
class FileTranscriptionPipeline:
    def __init__(
        self,
        *,
        ffmpeg_manager: Optional[FFmpegManager] = None,
        segmenter: Optional[Segmenter] = None,
    ) -> None:
        # 既存の初期化（変更なし）
        ...

    def process_file(
        self,
        file_path: str | Path,
        *,
        segment_transcriber: SegmentTranscriber,
        # Phase 6a 追加
        translator: Optional[BaseTranslator] = None,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        translation_timeout: Optional[float] = None,  # デフォルト無効
        write_subtitles: bool = True,
        write_translated_subtitles: bool = False,     # 翻訳版 SRT
        ...
    ) -> FileProcessingResult:
        # バリデーション
        if translator:
            if not translator.is_initialized():
                raise ValueError("Translator not initialized")
            if source_lang is None or target_lang is None:
                raise ValueError("source_lang and target_lang required")
        ...

    def process_files(
        self,
        file_paths: Sequence[str | Path],
        *,
        segment_transcriber: SegmentTranscriber,
        # Phase 6a 追加（process_file と同じパラメータ）
        translator: Optional[BaseTranslator] = None,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        translation_timeout: Optional[float] = None,
        write_subtitles: bool = True,
        write_translated_subtitles: bool = False,
        ...
    ) -> list[FileProcessingResult]:
        ...
```

**タイムアウトオプション**:
- `translation_timeout: Optional[float] = None` でデフォルト無効
- 指定時は Phase 5 の `TRANSLATION_TIMEOUT` 相当の処理を適用
- ネットワークハングや重いモデル（Riva-4B）対策として有用

#### 文脈管理の違い

| 項目 | StreamTranscriber | FileTranscriptionPipeline |
|------|-------------------|---------------------------|
| 文脈蓄積 | セッション中継続 | ファイル内のみ |
| リセット | `reset()` 呼び出し時 | ファイル処理完了時に自動リセット |
| 最大サイズ | `MAX_CONTEXT_BUFFER=100` | 同じ定数を共有 |

```python
def process_file(self, file_path, *, segment_transcriber, translator=None, ...):
    # 翻訳が有効な場合、ファイル処理開始時に文脈をリセット
    context_buffer: deque[str] = deque(maxlen=MAX_CONTEXT_BUFFER)

    for segment in self._transcribe_segments(...):
        # 翻訳処理（translator が設定されている場合）
        if translator and segment.text.strip():
            # 文脈を使って翻訳
            ...
            context_buffer.append(segment.text)
        yield segment

    # ファイル処理完了時に文脈バッファは自動的にスコープアウト
```

#### SRT 出力オプション

```python
# 元言語と翻訳を両方出力
pipeline = FileTranscriptionPipeline()

result = pipeline.process_file(
    "audio.wav",
    segment_transcriber=engine.transcribe,
    translator=translator,
    source_lang="ja",
    target_lang="en",
    write_subtitles=True,              # 元言語の字幕（audio.srt）
    write_translated_subtitles=True,   # 翻訳版の字幕（audio_en.srt）
)
```

#### 使用例

```python
from livecap_cli import FileTranscriptionPipeline
from livecap_cli.engines import EngineFactory
from livecap_cli.translation import TranslatorFactory

# エンジン初期化
engine = EngineFactory.create_engine("whispers2t_base", device="cuda")
engine.load_model()

# Translator 初期化
translator = TranslatorFactory.create_translator("google")

# パイプライン実行
pipeline = FileTranscriptionPipeline()

result = pipeline.process_file(
    "audio.wav",
    segment_transcriber=engine.transcribe,
    translator=translator,
    source_lang="ja",
    target_lang="en",
)

for segment in result.subtitles:
    print(f"[JA] {segment.text}")
    if segment.translated_text:
        print(f"[EN] {segment.translated_text}")
```

#### 実装タスク (Phase 6a)

1. `FileSubtitleSegment` に `translated_text`, `target_language` フィールド追加（末尾 Optional）
2. `process_file` / `process_files` に translator 関連パラメータ追加
3. パラメータバリデーション実装（translator 設定時の言語必須チェック等）
4. 文脈バッファ管理の実装（ファイル内スコープ、ファイル間リセット）
5. `_transcribe_segments` での翻訳処理追加
6. 翻訳エラー時の警告ログ実装
7. `translation_timeout` オプションの実装（デフォルト無効）
8. `write_translated_subtitles` オプションの実装
9. ユニットテスト作成（`tests/core/transcription/` に配置）
10. 統合テスト更新（`tests/integration/transcription/` に配置）
11. サンプルスクリプト作成

#### 変更ファイル (Phase 6a)

| ファイル | 操作 | 説明 |
|---------|------|------|
| `livecap_cli/transcription/file_pipeline.py` | 更新 | `FileSubtitleSegment` 翻訳フィールド追加、`process_file(s)` translator 統合 |
| `tests/core/transcription/test_file_pipeline_translation.py` | 新規 | 翻訳統合ユニットテスト |
| `tests/integration/transcription/test_file_transcription_pipeline.py` | 更新 | 翻訳統合テスト追加 |
| `examples/batch/batch_translation.py` | 新規 | バッチ翻訳サンプル |

**テスト配置方針**:
- ユニットテスト: `tests/core/transcription/` に配置（モック使用、外部依存なし）
- 統合テスト: `tests/integration/transcription/` に配置（実際の ASR/翻訳エンジン使用）

### Phase 6b: トップレベルエクスポート（オプション）

翻訳 API を `livecap_cli` トップレベルからインポート可能にする。

#### 現状

```python
# Phase 5 現在
from livecap_cli.translation import TranslatorFactory, TranslationResult, BaseTranslator
```

#### Phase 6b 後

```python
# Phase 6b 後
from livecap_cli import TranslatorFactory, TranslationResult, BaseTranslator
```

#### エクスポート対象

| クラス | 説明 |
|--------|------|
| `TranslatorFactory` | Translator 生成ファクトリ |
| `TranslationResult` | 翻訳結果 dataclass |
| `BaseTranslator` | Translator 基底クラス |

#### 実装タスク (Phase 6b)

1. `livecap_cli/__init__.py` に翻訳 API エクスポート追加
2. `__all__` リスト更新
3. ドキュメント更新

**動的 import の注意**:
- トップレベル `livecap_cli/__init__.py` で重い依存（torch, transformers 等）を即座に引かないよう、遅延 import を維持する
- `TranslatorFactory` 等は `livecap_cli.translation` サブモジュールからの re-export とし、実際のインポートはサブモジュール参照時に発生させる
- 例: `from livecap_cli.translation import TranslatorFactory` をトップレベルで `TranslatorFactory = ...` として公開

#### 変更ファイル (Phase 6b)

| ファイル | 操作 | 説明 |
|---------|------|------|
| `livecap_cli/__init__.py` | 更新 | 翻訳 API エクスポート（遅延 import 維持） |

## Phase 7: 非同期翻訳（将来計画）

非同期翻訳オプションは **Phase 7 へ延期**。以下の理由により、現時点では実装しない。

### 延期理由

1. **Phase 5 の同期翻訳で大半のユースケースをカバー**
   - Google/OPUS-MT: 低レイテンシ（100-300ms）で問題なし
   - Riva-4B: 5秒タイムアウトで graceful degradation

2. **実装コストが高い**
   - 翻訳結果の順序保証
   - コールバック設計
   - エラー伝播の複雑性

3. **ユーザーからの要望を待つ**
   - 実際のボトルネックが確認されてから対応

### 将来的な設計案（参考）

```python
# Phase 7 で検討する設計
StreamTranscriber(
    engine=engine,
    translator=translator,
    source_lang="ja",
    target_lang="en",
    async_translation=True,
    translation_callback=on_translation_complete,
)
```

検討事項:
- 翻訳結果の順序保証（結果キューの管理）
- コールバック設計（エラー時の挙動）
- タイムアウト処理との整合性

## 変更ファイル一覧（全 Phase）

| ファイル | 操作 | 説明 |
|---------|------|------|
| `livecap_cli/translation/__init__.py` | 新規 | Public API |
| `livecap_cli/translation/base.py` | 新規 | BaseTranslator |
| `livecap_cli/translation/result.py` | 新規 | TranslationResult |
| `livecap_cli/translation/metadata.py` | 新規 | TranslatorMetadata |
| `livecap_cli/translation/factory.py` | 新規 | TranslatorFactory |
| `livecap_cli/translation/exceptions.py` | 新規 | 例外クラス階層 |
| `livecap_cli/translation/retry.py` | 新規 | リトライデコレータ |
| `livecap_cli/translation/lang_codes.py` | 新規 | 言語コード正規化 |
| `livecap_cli/translation/impl/__init__.py` | 新規 | Impl package |
| `livecap_cli/translation/impl/google.py` | 新規 | GoogleTranslator |
| `livecap_cli/translation/impl/opus_mt.py` | 新規 | OpusMTTranslator |
| `livecap_cli/translation/impl/riva_instruct.py` | 新規 | RivaInstructTranslator |
| `livecap_cli/__init__.py` | 更新 | Translation exports |
| `livecap_cli/utils/__init__.py` | 更新 | VRAM 確認ユーティリティ追加 |
| `pyproject.toml` | 更新 | 依存関係追加 |
| `tests/core/translation/` | 新規 | ユニットテスト |
| `tests/integration/test_translation.py` | 新規 | 統合テスト |
| `tests/conftest.py` | 更新 | テストマーカー追加 |

## リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| Google Translate レート制限 | 高頻度使用で失敗 | リトライ + バックオフ |
| OPUS-MT モデル変換失敗 | 初回起動が遅い | 事前変換済みモデル提供 |
| Riva-4B VRAM 不足 | GPU 8GB 必要 | 明確なエラーメッセージ + 警告 |
| ASR + Riva-4B 同時ロード | VRAM 超過 | OPUS-MT CPU デフォルト、構成ガイド |
| 文脈抽出の精度 | 翻訳結果から対象文を特定困難 | 区切り文字の工夫 |

## 完了条件

### Phase 1-4（✅ 完了）

- [x] BaseTranslator ABC が定義されている
- [x] TranslatorFactory が動作する
- [x] GoogleTranslator が動作する
- [x] OpusMTTranslator が動作する（モデルロード含む）
- [x] RivaInstructTranslator が動作する（GPU 環境）
- [x] 文脈挿入が全エンジンで機能する
- [x] `TranslationResult.to_event_dict()` が既存イベント型に変換できる
- [x] VRAM 確認ユーティリティが追加されている
- [x] VRAM 不足時の警告が実装されている
- [x] ユニットテストがパスする（120+ テスト）
- [x] `livecap_cli.translation` から export されている
- [x] サンプルスクリプトが作成されている

### Phase 5（✅ 完了 2025-12-12）

- [x] `BaseTranslator.default_context_sentences` プロパティが追加されている
- [x] `TranscriptionResult` に `translated_text`, `target_language` フィールドが追加されている
- [x] `StreamTranscriber` に `translator`, `source_lang`, `target_lang` パラメータが追加されている
- [x] translator 設定時の初期化バリデーションが実装されている
- [x] 言語ペアの事前警告が実装されている
- [x] 文脈バッファ管理が `deque(maxlen=100)` で実装されている
- [x] 翻訳タイムアウト処理（5秒）が実装されている
- [x] 翻訳エラー時に `translated_text=None` + 警告ログが出力される
- [x] translator なしの後方互換動作が維持されている
- [x] ユニットテストがパスする
- [x] リアルタイム翻訳のサンプルスクリプトが作成されている

### Phase 6a（✅ 完了）

- [x] `FileSubtitleSegment` に `translated_text`, `target_language` フィールドが追加されている
- [x] `FileTranscriptionPipeline` に `translator`, `source_lang`, `target_lang` パラメータが追加されている
- [x] translator 設定時の初期化バリデーションが実装されている
- [x] 文脈バッファ管理がファイル間リセットで実装されている
- [x] 翻訳エラー時に `translated_text=None` + 警告ログが出力される
- [x] `write_translated_subtitles` オプションが実装されている
- [x] translator なしの後方互換動作が維持されている
- [x] ユニットテストがパスする（25件）
- [x] バッチ翻訳のサンプルスクリプトが作成されている

### Phase 6b（⏭️ スキップ）

トップレベルエクスポートは現状不要と判断。`livecap_cli.translation` からのインポートで十分。

- [ ] ~~`TranslatorFactory` が `livecap_cli` トップレベルからエクスポートされている~~
- [ ] ~~`TranslationResult` が `livecap_cli` トップレベルからエクスポートされている~~
- [ ] ~~`BaseTranslator` が `livecap_cli` トップレベルからエクスポートされている~~
- [ ] ~~`__all__` リストが更新されている~~

**スキップ理由**: 機能的な価値がなく、名前空間の肥大化を避けるため。

### Phase 7（📋 将来計画）

- [ ] 非同期翻訳の必要性が確認されている
- [ ] 非同期翻訳モードが実装されている（必要性確認後）

## 参考資料

- [deep-translator PyPI](https://pypi.org/project/deep-translator/)
- [CTranslate2 OPUS-MT Guide](https://opennmt.net/CTranslate2/guides/opus_mt.html)
- [Helsinki-NLP/opus-mt-ja-en](https://huggingface.co/Helsinki-NLP/opus-mt-ja-en)
- [nvidia/Riva-Translate-4B-Instruct](https://huggingface.co/nvidia/Riva-Translate-4B-Instruct)
- [Google Cloud Translation](https://cloud.google.com/blog/products/ai-machine-learning/google-cloud-translation-ai)

---

**作成日**: 2025-12-11
**最終更新**: 2025-12-12
**Issue**: #72
**現在の Phase**: 6a (FileTranscriptionPipeline 翻訳統合)
**次の Phase**: 6b (トップレベルエクスポート) → 7 (非同期翻訳 - 必要時)
