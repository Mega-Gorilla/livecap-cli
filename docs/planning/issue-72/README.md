# Issue #72: 翻訳プラグインシステム実装

## 概要

翻訳プラグインシステムを設計し、Google Translate、OPUS-MT、Riva-Translate-4B-Instruct の3つの翻訳エンジンを実装する。

## 関連ドキュメント

| ドキュメント | 内容 |
|-------------|------|
| [architecture.md](./architecture.md) | 技術設計（Phase 1-4 完了分） |
| [examples.md](./examples.md) | 使用例 |
| [testing.md](./testing.md) | テスト計画 |

## 実装ステータス

### Phase 1-4 完了（2025-12-11）

| コンポーネント | ステータス | Phase | 備考 |
|---------------|-----------|-------|------|
| `livecap_core/translation/` | ✅ 完了 | 1 | ディレクトリ作成 |
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

**※1**: 翻訳 API は `livecap_core.translation` パッケージからインポート。トップレベル `livecap_core` へのエクスポートは Phase 6 で検討。

### Phase 5 未実装

| コンポーネント | ステータス | Phase | 備考 |
|---------------|-----------|-------|------|
| `TranscriptionResult` 翻訳フィールド | ❌ 未実装 | 5 | `translated_text`, `target_language` |
| `StreamTranscriber` translator 統合 | ❌ 未実装 | 5 | パラメータ追加、バリデーション |
| 文脈バッファ管理 | ❌ 未実装 | 5 | translator の context_sentences 使用 |
| 翻訳エラーハンドリング | ❌ 未実装 | 5 | 警告ログ、結果は返す |
| `tests/core/transcription/test_stream.py` 更新 | ❌ 未実装 | 5 | 翻訳統合ユニットテスト |
| `tests/integration/test_stream_translation.py` | ❌ 未実装 | 5 | ASR+翻訳統合テスト |
| `examples/realtime/realtime_translation.py` | ❌ 未実装 | 5 | リアルタイム翻訳サンプル |

### Phase 6 未実装（将来）

| コンポーネント | ステータス | Phase | 備考 |
|---------------|-----------|-------|------|
| `FileTranscriptionPipeline` 翻訳統合 | ❌ 未実装 | 6 | ファイル文字起こし+翻訳 |
| 非同期翻訳オプション | ❌ 検討中 | 6 | `async_translation` パラメータ |
| トップレベルエクスポート | ❌ 検討中 | 6 | `livecap_core` から直接インポート |

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
| `livecap_core/transcription/result.py` | ✅ フィールド追加 |
| `livecap_core/transcription/stream.py` | ✅ パラメータ追加、翻訳処理追加 |
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
from livecap_core import StreamTranscriber, EngineFactory, MicrophoneSource
from livecap_core.translation import TranslatorFactory

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
| `livecap_core/translation/base.py` | 更新 | `default_context_sentences` プロパティ追加 |
| `livecap_core/transcription/result.py` | 更新 | 翻訳フィールド追加 |
| `livecap_core/transcription/stream.py` | 更新 | translator 統合、deque バッファ、タイムアウト |
| `tests/core/translation/test_base.py` | 更新 | プロパティのテスト |
| `tests/core/transcription/test_result.py` | 更新 | 新フィールドのテスト |
| `tests/core/transcription/test_stream.py` | 更新 | 翻訳統合テスト |
| `tests/integration/test_stream_translation.py` | 新規 | ASR+翻訳統合テスト |
| `examples/realtime/realtime_translation.py` | 新規 | リアルタイム翻訳例 |

## Phase 6: 拡張機能（将来）

Phase 5 完了後、必要に応じて実装を検討する機能。

### 6.1 FileTranscriptionPipeline 翻訳統合

ファイル文字起こしパイプラインへの翻訳機能統合。

```python
pipeline = FileTranscriptionPipeline(
    segment_transcriber=engine.transcribe,
    translator=translator,
    source_lang="ja",
    target_lang="en",
)

for result in pipeline.process("audio.wav"):
    print(f"{result.text} → {result.translated_text}")
```

### 6.2 非同期翻訳オプション

翻訳がボトルネックになる場合の非同期翻訳モード。

```python
StreamTranscriber(
    engine=engine,
    translator=translator,
    source_lang="ja",
    target_lang="en",
    async_translation=True,  # 非同期モード
    translation_callback=on_translation_complete,
)
```

検討事項:
- 翻訳結果の順序保証
- コールバック設計
- タイムアウト処理

### 6.3 トップレベルエクスポート

`livecap_core` トップレベルへの翻訳 API エクスポート。

```python
# 現状（Phase 5）
from livecap_core.translation import TranslatorFactory

# Phase 6 検討
from livecap_core import TranslatorFactory  # トップレベルから直接
```

## 変更ファイル一覧（全 Phase）

| ファイル | 操作 | 説明 |
|---------|------|------|
| `livecap_core/translation/__init__.py` | 新規 | Public API |
| `livecap_core/translation/base.py` | 新規 | BaseTranslator |
| `livecap_core/translation/result.py` | 新規 | TranslationResult |
| `livecap_core/translation/metadata.py` | 新規 | TranslatorMetadata |
| `livecap_core/translation/factory.py` | 新規 | TranslatorFactory |
| `livecap_core/translation/exceptions.py` | 新規 | 例外クラス階層 |
| `livecap_core/translation/retry.py` | 新規 | リトライデコレータ |
| `livecap_core/translation/lang_codes.py` | 新規 | 言語コード正規化 |
| `livecap_core/translation/impl/__init__.py` | 新規 | Impl package |
| `livecap_core/translation/impl/google.py` | 新規 | GoogleTranslator |
| `livecap_core/translation/impl/opus_mt.py` | 新規 | OpusMTTranslator |
| `livecap_core/translation/impl/riva_instruct.py` | 新規 | RivaInstructTranslator |
| `livecap_core/__init__.py` | 更新 | Translation exports |
| `livecap_core/utils/__init__.py` | 更新 | VRAM 確認ユーティリティ追加 |
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
- [x] `livecap_core.translation` から export されている
- [x] サンプルスクリプトが作成されている

### Phase 5（❌ 未完了）

- [ ] `BaseTranslator.default_context_sentences` プロパティが追加されている
- [ ] `TranscriptionResult` に `translated_text`, `target_language` フィールドが追加されている
- [ ] `StreamTranscriber` に `translator`, `source_lang`, `target_lang` パラメータが追加されている
- [ ] translator 設定時の初期化バリデーションが実装されている
- [ ] 言語ペアの事前警告が実装されている
- [ ] 文脈バッファ管理が `deque(maxlen=...)` で実装されている
- [ ] 翻訳タイムアウト処理が実装されている
- [ ] 翻訳エラー時に `translated_text=None` + 警告ログが出力される
- [ ] translator なしの後方互換動作が維持されている
- [ ] ユニットテストがパスする
- [ ] 統合テストがパスする
- [ ] リアルタイム翻訳のサンプルスクリプトが作成されている

### Phase 6（❌ 将来検討）

- [ ] `FileTranscriptionPipeline` に翻訳機能が統合されている
- [ ] （オプション）非同期翻訳モードが実装されている
- [ ] （オプション）トップレベル `livecap_core` へ翻訳 API エクスポート

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
**現在の Phase**: 5 (StreamTranscriber 翻訳統合)
**次の Phase**: 6 (FileTranscriptionPipeline 翻訳統合、非同期翻訳)
