# Issue #273: ResultCoalescer — 短い ASR 結果のテキスト結合

> **Status**: 📋 PLANNED
> **作成日**: 2026-03-15
> **Issue**: #273
> **影響範囲**: `livecap_cli/transcription/` (VAD 変更なし)

---

## 1. 概要

短い発話（「はい」「うん」等）が独立した ASR 結果として即確定され、字幕が1行1単語で不自然になる問題を、ASR 出力後のテキスト結合（ResultCoalescer）で解決する。

### 1.1 方針決定の経緯

| 検討アプローチ | 採否 | 理由 |
|---|---|---|
| 音声マージ（VAD セグメント結合） | ❌ | VAD 最適化プリセット（100 trials/preset）を壊すリスク、パディング膨張問題（1.9-2.1x） |
| ASR 結果テキスト結合 | ✅ | VAD/ASR パイプラインに非干渉、文字数で直接判定可能 |
| 短い出力を無視 | ❌ | 情報を失う（「はい」「OK」は意味のある発話） |

### 1.2 パイプライン変更

```
変更前: VAD → ASR+翻訳 → TranscriptionResult → 即時出力
変更後: VAD → ASR(翻訳なし) → [ResultCoalescer] → 翻訳 → 出力
```

coalescer 無効時は従来パスを維持（後方互換）。

---

## 2. 実装計画

### Phase 1: ResultCoalescer クラス（新規ファイル）

**ファイル**: `livecap_cli/transcription/result_coalescer.py`

```python
class ResultCoalescer:
    def __init__(self, max_words=1, max_chars_single_token=4, merge_window_s=2.0): ...
    def push(self, result, now) -> list[TranscriptionResult]: ...
    def flush(self, now, force=False) -> Optional[TranscriptionResult]: ...
    def _is_short(self, text) -> bool: ...
    def _merge(self, a, b) -> TranscriptionResult: ...
    def reset(self) -> None: ...
```

**判定ロジック `_is_short()`**:
1. 句読点終端（。！？.!?）→ マージしない
2. `split()` で2語以上 → `len(words) <= max_words` で判定
3. 単一トークン → `len(text) <= max_chars_single_token` で判定

**タスク**:
- [ ] `result_coalescer.py` 作成
- [ ] `_is_short()` 実装（句読点 + word_count + char_count 多段判定）
- [ ] `push()` 実装（pending 管理、gap 判定、マージ）
- [ ] `flush()` 実装（タイムアウト + force flush）
- [ ] `_merge()` 実装（テキスト結合、language フィールド保持、translated_text=None）
- [ ] `__init__.py` の `__all__` 更新

### Phase 2: StreamTranscriber 統合（同期パス）

**ファイル**: `livecap_cli/transcription/stream.py`

**変更点**:
- `__init__()`: `result_coalescer` オプション引数追加
- `_transcribe_segment()`: coalescer 有効時は翻訳スキップ
- `_apply_translation_sync()`: coalescer 出力に `_translate_text()` 適用
- `feed_audio()`: coalescer 経由で結果を emit + flush(now) によるタイムアウト
- `finalize()`: 戻り値型を `list[TranscriptionResult]` に変更（coalescer の flush 含む）
- `transcribe_sync()`: `for result in self.finalize(): yield result`
- `reset()`: coalescer.reset() 呼び出し追加

**タスク**:
- [ ] `__init__` に `result_coalescer` 引数追加
- [ ] `_transcribe_segment()` に coalescer 分岐追加
- [ ] `_apply_translation_sync()` 新規メソッド
- [ ] `feed_audio()` に coalescer 経路追加
- [ ] `finalize()` の戻り値を `list[TranscriptionResult]` に変更
- [ ] `transcribe_sync()` を `for result in self.finalize(): yield result` に変更
- [ ] `reset()` に coalescer.reset() 追加

### Phase 3: StreamTranscriber 統合（非同期パス）

**ファイル**: `livecap_cli/transcription/stream.py`

**変更点**:
- `_transcribe_segment_async()`: coalescer 有効時は翻訳スキップ
- `_apply_translation_async()`: `_do_translate_direct()` を executor 経由で呼ぶ（二重 submit 回避、タイムアウト付き）
- `transcribe_async()`: coalescer 経由で結果を yield + flush + finalize 直接処理

**タスク**:
- [ ] `_apply_translation_async()` 新規メソッド
- [ ] `_transcribe_segment_async()` に coalescer 分岐追加
- [ ] `transcribe_async()` に coalescer 経路追加（flush + finalize）

### Phase 4: テスト

**ファイル**: `tests/transcription/test_result_coalescer.py`（新規）

**テストケース**:
- [ ] `_is_short()` — 全言語パターン（ja/en/zh/ko + 句読点）
- [ ] `push()` — ケース1: 短文 + 窓内後続 → マージ
- [ ] `push()` — ケース2: 短文 + 窓外後続 → flush + 新判定
- [ ] `push()` — ケース3: 連続短文のマージ（再保留）
- [ ] `push()` — ケース4: 十分な長さ → 即確定
- [ ] `flush()` — タイムアウト判定 + force flush
- [ ] `_merge()` — テキスト結合、confidence、language、translated_text=None
- [ ] StreamTranscriber 統合テスト — coalescer 有効/無効での feed_audio() 動作
- [ ] StreamTranscriber 統合テスト — finalize() が list を返すこと
- [ ] 翻訳付きパステスト — coalescer 経由の翻訳適用

---

## 3. 設計上の重要決定

### 3.1 finalize() の戻り値型変更

`finalize()` の戻り値型を `Optional[TranscriptionResult]` → `list[TranscriptionResult]` に変更する。coalescer 導入後は最大2件の結果が出る（最終 VAD セグメント + 保留中の flush）ため、リスト返却が自然。後方互換を維持するための二重経路（キュー + 戻り値）は複雑さの源になるため採用しない。`finalize()` の外部利用は限定的（examples 内のみ）であり、`if final:` → `for final in finals:` の変更で対応可能。

### 3.2 sync/async 翻訳経路の分離

| パス | 翻訳メソッド | 理由 |
|---|---|---|
| 同期（feed_audio, finalize） | `_apply_translation_sync()` → `_translate_text()` | 内部で executor.submit + timeout |
| 非同期（transcribe_async） | `_apply_translation_async()` → `_do_translate_direct()` via executor | 二重 submit 回避 |

### 3.3 判定ロジック（外部ライブラリ不要）

`_is_short()` は多段ヒューリスティクス。budoux 等の外部ライブラリは不要（budoux は「はい」を2分割する問題あり）。将来 ICU BreakIterator が必要になった場合は `_is_short()` のみ差し替え。

| テキスト | 判定 | 分岐 |
|---|---|---|
| はい | 短い | 1トークン, 2文字 ≤ 4 |
| そうですね | 短くない | 1トークン, 5文字 > 4 |
| はい。 | 短くない | 句読点終端 |
| yes | 短い | 1トークン, 3文字 ≤ 4 |
| I agree | 短くない | 2語 > 1 |
| 好的 | 短い | 1トークン, 2文字 ≤ 4 |
| 네 | 短い | 1トークン, 1文字 ≤ 4 |

### 3.4 設定パラメータ

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `max_words` | 1 | スペース区切り言語（2語以上）の閾値 |
| `max_chars_single_token` | 4 | 非スペース言語 / 単一語の文字数閾値 |
| `merge_window_s` | 2.0 | セグメント間ギャップ上限 兼 保留タイムアウト |

---

## 4. 影響範囲

| ファイル | 変更種別 | 内容 |
|---|---|---|
| `livecap_cli/transcription/result_coalescer.py` | **新規** | ResultCoalescer クラス |
| `livecap_cli/transcription/stream.py` | 変更 | coalescer 統合（sync/async 別経路） |
| `livecap_cli/transcription/__init__.py` | 変更 | `__all__` にエクスポート追加 |
| `tests/transcription/test_result_coalescer.py` | **新規** | ユニットテスト |
| `livecap_cli/vad/` | **変更なし** | — |
| `livecap_cli/engines/` | **変更なし** | — |

### API 変更

- `finalize()` の戻り値型: `Optional[TranscriptionResult]` → `list[TranscriptionResult]`
- `result_coalescer` は新規オプション引数（既存コードへの影響なし）

---

## 5. 将来の拡張（本 Issue スコープ外）

- **ICU BreakIterator**: `_is_short()` のバックエンド差し替え
- **音声レベルの SegmentCoalescer**: ASR 精度改善がベンチマークで実証された場合
- **言語別テキスト結合ルール**: `_merge()` の結合方式を言語に応じて切り替え
- **エンジン別 coalesce_policy**: `EngineInfo` に設定を追加
