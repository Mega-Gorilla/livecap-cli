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

coalescer は常時有効。`StreamTranscriber.__init__()` の `result_coalescer` 引数は `Optional[ResultCoalescer]` 型で、`None`（デフォルト）の場合はデフォルト設定の `ResultCoalescer()` を内部生成する。`None` は「無効」ではなく「デフォルト生成」を意味する（mutable default argument 問題の回避）。`if self._coalescer:` 分岐を持たず単一パスで実装する。非短文は `push()` で即返却されるため遅延なし。マージ不要の場合は `ResultCoalescer(max_chars_single_token=0, max_words=0)` で全結果が即確定される。

---

## 2. 実装計画

### Phase 1: ResultCoalescer クラス（新規ファイル）

**ファイル**: `livecap_cli/transcription/result_coalescer.py`

```python
# スペース区切りが必要な言語セット
_SPACE_DELIMITED_LANGS = frozenset({"en", "fr", "de", "es", "it", "pt", "nl", "ru", "ko", "pl", "sv", ...})

class ResultCoalescer:
    def __init__(self, max_words=1, max_chars_single_token=4, merge_window_s=2.0): ...
    def push(self, result, now) -> list[TranscriptionResult]: ...
    def flush(self, now, force=False) -> Optional[TranscriptionResult]: ...
    def _is_short(self, text) -> bool: ...
    def _join_text(self, a, b, language) -> str: ...  # 言語ベースのスペース挿入
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
- [ ] `_join_text()` 実装（言語ベースのスペース挿入/直接結合切り替え）
- [ ] `_merge()` 実装（`_join_text()` 使用、confidence、language 保持、translated_text=None）
- [ ] `__init__.py` の `__all__` 更新

### Phase 2: StreamTranscriber 統合（同期パス）

**ファイル**: `livecap_cli/transcription/stream.py`

**変更点**:
- `__init__()`: `result_coalescer: Optional[ResultCoalescer] = None` 引数追加（`None` = デフォルト生成、mutable default 回避）
- `_emit_result()`: キュー投入 + `on_result` コールバック呼び出しヘルパー（`feed_audio()` 用）
- `_transcribe_segment()`: 翻訳を常にスキップ（翻訳は coalescer 出力後に実行）
- `_apply_translation_sync()`: coalescer 出力に `_translate_text()` 適用
- `feed_audio()`: coalescer 経由で結果を emit + flush(now) によるタイムアウト
- `finalize()`: 戻り値型を `list[TranscriptionResult]` に変更。VAD finalize → push → force flush の順序で結果を収集して返す
- `transcribe_sync()`: `for final in self.finalize(): yield final`（list 返却で drain 不要）
- `reset()`: coalescer.reset() 呼び出し追加

**タスク**:
- [ ] `__init__` に `result_coalescer: Optional[ResultCoalescer] = None` 引数追加（`None` → `ResultCoalescer()` 生成）
- [ ] `_emit_result()` ヘルパーメソッド新規作成（`feed_audio()` 用）
- [ ] `_transcribe_segment()` から翻訳呼び出しを除去
- [ ] `_apply_translation_sync()` 新規メソッド
- [ ] `feed_audio()` を coalescer 経由に書き換え（分岐なし単一パス）
- [ ] `finalize()` を `list[TranscriptionResult]` 返却に変更（VAD finalize → push → force flush）
- [ ] `transcribe_sync()` の finalize 呼び出しを `for final in self.finalize(): yield final` に変更
- [ ] `reset()` に coalescer.reset() 追加

### Phase 3: StreamTranscriber 統合（非同期パス）

**ファイル**: `livecap_cli/transcription/stream.py`

**変更点**:
- `_transcribe_segment_async()`: 翻訳を常にスキップ（同期パスと同じ方針）
- `_apply_translation_async()`: `_do_translate_direct()` を executor 経由で呼ぶ（二重 submit 回避、タイムアウト付き）
- `transcribe_async()`: coalescer 経由で結果を yield + flush + finalize 直接処理（分岐なし単一パス）

**タスク**:
- [ ] `_apply_translation_async()` 新規メソッド
- [ ] `_transcribe_segment_async()` から翻訳呼び出しを除去
- [ ] `transcribe_async()` を coalescer 経由に書き換え（flush + push + finalize）

### Phase 4: テスト

**ファイル**: `tests/transcription/test_result_coalescer.py`（新規）

**テストケース**:
- [ ] `_is_short()` — 全言語パターン（ja/en/zh/ko + 句読点）
- [ ] `push()` — ケース1: 短文 + 窓内後続 → マージ
- [ ] `push()` — ケース2: 短文 + 窓外後続 → flush + 新判定
- [ ] `push()` — ケース3: 連続短文のマージ（再保留）
- [ ] `push()` — ケース4: 十分な長さ → 即確定
- [ ] `flush()` — タイムアウト判定 + force flush
- [ ] `_join_text()` — 言語ベースのスペース挿入（en: "yes I agree", ja: "はい今日は", ko: "네 알겠습니다"）
- [ ] `_merge()` — `_join_text()` 使用、confidence、language、translated_text=None
- [ ] StreamTranscriber 統合テスト — feed_audio() が coalescer 経由で結果を emit すること
- [ ] StreamTranscriber 統合テスト — finalize() が `list[TranscriptionResult]` を返すこと
- [ ] StreamTranscriber 統合テスト — finalize() が VAD finalize → push → force flush の順序で動作すること
- [ ] StreamTranscriber 統合テスト — finalize() で pending と最終 VAD セグメントがマージされること
- [ ] StreamTranscriber 統合テスト — `ResultCoalescer(max_chars_single_token=0, max_words=0)` でマージ無効化
- [ ] 翻訳付きパステスト — coalescer 経由の翻訳適用

---

## 3. 設計上の重要決定

### 3.1 finalize() の戻り値変更と単一パス設計

`finalize()` の戻り値型を **`list[TranscriptionResult]`** に変更する。coalescer 常時有効により `if self._coalescer:` 分岐が不要になり、`_emit_result()` 経路とのドレイン問題も発生しない。

`finalize()` の動作:
1. 最終 VAD セグメントを ASR → `coalescer.push()` → 確定分を list に追加（pending と最終セグメントのマージ機会を保持）
2. coalescer に残った保留分を `flush(force=True)` → list に追加
3. list を返す（0〜2 件）

**順序が重要**: 先に pending を force flush すると、最終 VAD セグメントが pending の merge_window 内にあってもマージできなくなる。最終セグメントを先に push() することで、end-of-stream 直前の短文結合を正しく処理する。

呼び出し側の変更:
```python
# Before
final = transcriber.finalize()
if final:
    yield final

# After
for final in transcriber.finalize():
    yield final
```

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
| 알겠습니다 | 短くない | 1トークン, 5文字 > 4 |

**注**: 韓国語はスペース区切り言語のため `_SPACE_DELIMITED_LANGS` に含める（結合時: "네 알겠습니다"）。

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

- `StreamTranscriber.__init__()` に `result_coalescer: Optional[ResultCoalescer] = None` パラメータ追加（`None` = デフォルト生成、mutable default 回避）
- **`finalize()` の戻り値型を `list[TranscriptionResult]` に変更**（破壊的変更）
- `_transcribe_segment()` / `_transcribe_segment_async()` は翻訳を行わなくなる（翻訳は coalescer 出力後に実行）
- coalescer は常時有効。マージ不要時は `ResultCoalescer(max_chars_single_token=0, max_words=0)` を渡す

### `finalize()` 変更の移行対象

`finalize()` の戻り値を `Optional[TranscriptionResult]` → `list[TranscriptionResult]` に変更する際の影響箇所一覧:

**実装コード** (Phase 2/3 で対応):
| ファイル | 行 | 現行パターン | 変更後 |
|---|---|---|---|
| `stream.py` | 644 | `final = self.finalize(); if final: yield final` | `for final in self.finalize(): yield final` |
| `stream.py` | 687 | `transcribe_async` 内の finalize 処理 | Phase 3 で直接 push/flush に書き換え |

**テスト** (Phase 4 で対応):
| ファイル | 行 | 変更内容 |
|---|---|---|
| `tests/transcription/test_stream.py` | 263 | `result = .finalize(); assert result is not None` → `results = .finalize(); assert len(results) == 1` |
| `tests/transcription/test_stream.py` | 275 | `assert result is None` → `assert results == []` |
| `tests/integration/realtime/test_mock_realtime_flow.py` | 294 | `if final:` → `for final in ...:` |
| `tests/integration/realtime/test_e2e_realtime_flow.py` | 409 | 同上 |
| `tests/integration/vad/test_from_language_integration.py` | 266 | 同上 |

**サンプルコード・ドキュメント** (Phase 4 完了後に対応):
| ファイル | 行 | 変更内容 |
|---|---|---|
| `examples/realtime/callback_api.py` | 143 | `if final:` → `for final in ...:` |
| `docs/guides/realtime-transcription.md` | 141 | 同上 |
| `docs/architecture/core-api-spec.md` | 545, 768 | 型シグネチャ + コード例の更新 |
| `docs/reference/api.md` | 243 | 説明の更新 |
| `docs/reference/feature-inventory.md` | 148 | コード例の更新 |

---

## 5. 将来の拡張（本 Issue スコープ外）

- **ICU BreakIterator**: `_is_short()` のバックエンド差し替え
- **音声レベルの SegmentCoalescer**: ASR 精度改善がベンチマークで実証された場合
- **エンジン別 coalesce_policy**: `EngineInfo` に設定を追加
