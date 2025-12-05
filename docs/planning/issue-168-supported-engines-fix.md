# Issue #168: supported_engines 不整合修正

> **Status**: IN PROGRESS
> **作成日:** 2025-12-05
> **更新日:** 2025-12-05
> **関連 Issue:** #168
> **関連 PR:** #171
> **依存:** #165 (WhisperS2T統合) ✅ 完了

---

## 1. 背景と目的

### 1.1 Issue #168 の概要

Issue #168 では以下の2つの問題が報告されていた：

1. **`languages.py` の `supported_engines` に誤ったエンジンが含まれている**
2. **`asr_code` の一貫した活用がない**

### 1.2 調査結果

#### 問題1: `supported_engines` の誤り → ✅ 既に解決済み

現在の `languages.py` では、`canary` と `voxtral` は正しい言語のみに設定されている：

```python
# 現在の正しい設定
"ja": supported_engines=["reazonspeech", "whispers2t", "parakeet_ja"]  # canary, voxtral 含まず
"en": supported_engines=["parakeet", "whispers2t", "canary", "voxtral"]  # 正しい
"de": supported_engines=["whispers2t", "canary", "voxtral"]  # 正しい
```

#### 問題2: `asr_code` 未対応 → ✅ PR #171 で修正済み

`EngineMetadata.get_engines_for_language()` が `asr_code` を考慮していないため、
地域コード付き言語（`zh-CN`, `zh-TW` など）で正しくエンジンを取得できない。

| 入力 | 修正前 | 修正後 |
|------|--------|--------|
| `"zh-CN"` | `[]` ❌ | `["whispers2t"]` ✅ |
| `"zh-TW"` | `[]` ❌ | `["whispers2t"]` ✅ |
| `"pt-BR"` | `[]` ❌ | `["whispers2t", "voxtral"]` ✅ |

#### 問題3: 二重管理とデッドコード → ❌ 要対応（新規発見）

調査の結果、より根本的な問題が発見された：

**3a. 二重管理の問題**

言語→エンジンのマッピングが2箇所で管理されている：

| 場所 | 関数 | 実コード利用 |
|------|------|-------------|
| `languages.py` | `Languages.get_engines_for_language()` | **0箇所** |
| `metadata.py` | `EngineMetadata.get_engines_for_language()` | **3箇所** |

**3b. `riva` エンジンのデッドコード**

`Languages._LANGUAGES` には `riva` への参照があるが、エンジン自体が存在しない：

```python
# languages.py に定義されているが...
"es-ES": supported_engines=["riva"],  # ← riva エンジンは未実装
"es-US": supported_engines=["riva"],  # ← riva エンジンは未実装
"pt-BR": supported_engines=["riva"],  # ← riva エンジンは未実装
```

`livecap_core/engines/` に `riva` エンジンの実装は存在しない（デッドコード）。

**3c. データ不整合**

| 言語 | `Languages` の結果 | `EngineMetadata` の結果 | 実態 |
|------|-------------------|------------------------|------|
| `pt-BR` | `["riva"]` | `["whispers2t", "voxtral"]` | riva未実装、後者が正しい |
| `es-ES` | `["riva"]` | `["whispers2t", "canary", "voxtral"]` | riva未実装、後者が正しい |

---

## 2. 原因分析

### 2.1 データフロー

```
ユーザー入力: "zh-CN"
    ↓
Languages.normalize("zh-CN") → "zh-CN" (地域コード保持)
    ↓
LanguageInfo.asr_code → "zh" (ASR用2文字コード)
    ↓
WHISPER_LANGUAGES contains "zh" → True
```

### 2.2 問題の根本原因

1. **歴史的経緯**: `Languages.supported_engines` は UI 用に手動管理されていた
2. **エンジン統合**: WhisperS2T 統合により `EngineMetadata` が正式なデータソースに
3. **移行未完了**: 古い `Languages.supported_engines` が残存
4. **riva の計画変更**: `riva` エンジンは計画されていたが未実装のまま放置

---

## 3. 修正方針

### 3.1 Phase 1: `asr_code` 対応（✅ PR #171 で完了）

`EngineMetadata.get_engines_for_language()` を `asr_code` 対応に修正。

```python
# metadata.py: get_engines_for_language() の修正
lang_info = Languages.get_info(normalized)
asr_code = lang_info.asr_code if lang_info else normalized

for engine_id, info in cls._ENGINES.items():
    if asr_code in info.supported_languages:
        result.append(engine_id)
```

### 3.2 Phase 2: `Languages.get_engines_for_language()` 廃止（本対応）

**方針**: `Languages.get_engines_for_language()` を廃止し、`EngineMetadata` に一元化

**理由**:
1. **実利用なし**: 実コードで0箇所しか使われていない
2. **上位互換が存在**: `EngineMetadata` 版が100言語対応で機能的に優れる
3. **データ不整合**: `riva` 参照がデッドコード
4. **Single Source of Truth**: エンジン情報は `EngineMetadata` に集約すべき

**実装内容**:
1. `Languages.get_engines_for_language()` に `@deprecated` を追加
2. `LanguageInfo.supported_engines` フィールドを削除
3. ドキュメントを `EngineMetadata.get_engines_for_language()` に統一

---

## 4. 実装計画

### 4.1 Phase 1: asr_code 対応（✅ 完了）

| ファイル | 変更内容 | 状態 |
|---------|----------|------|
| `livecap_core/engines/metadata.py` | `get_engines_for_language()` を `asr_code` 対応に修正 | ✅ 完了 |
| `tests/core/engines/test_engine_factory.py` | `asr_code` 変換のテスト追加（6テスト） | ✅ 完了 |

### 4.2 Phase 2: Languages.get_engines_for_language() 廃止

| ファイル | 変更内容 | 状態 |
|---------|----------|------|
| `livecap_core/languages.py` | `get_engines_for_language()` に `@deprecated` 追加 | 未着手 |
| `livecap_core/languages.py` | `LanguageInfo.supported_engines` フィールド削除 | 未着手 |
| `docs/architecture/core-api-spec.md` | API ドキュメント更新 | 未着手 |
| `docs/reference/feature-inventory.md` | リファレンス更新 | 未着手 |

### 4.3 Phase 2 修正コード

```python
# languages.py: get_engines_for_language() の廃止

import warnings

@classmethod
def get_engines_for_language(cls, code: str) -> List[str]:
    """
    指定言語をサポートするエンジンリストを取得

    .. deprecated:: 2.1.0
       代わりに EngineMetadata.get_engines_for_language() を使用してください。
       この関数は16言語のみ対応で、100言語対応の EngineMetadata 版を推奨します。
    """
    warnings.warn(
        "Languages.get_engines_for_language() is deprecated. "
        "Use EngineMetadata.get_engines_for_language() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # EngineMetadata に委譲
    from livecap_core.engines.metadata import EngineMetadata
    return EngineMetadata.get_engines_for_language(code)
```

```python
# languages.py: LanguageInfo から supported_engines を削除

@dataclass
class LanguageInfo:
    """言語情報の完全定義"""
    code: str
    display_name: str
    english_name: str
    native_name: str
    flag: str
    iso639_1: Optional[str]
    iso639_3: Optional[str]
    windows_lcid: Optional[int]
    google_code: Optional[str]
    translation_code: str
    asr_code: str
    # supported_engines: List[str]  # ← 削除
    translation_services: List[str] = field(default_factory=list)
```

---

## 5. 影響範囲

### 5.1 Phase 1 の影響（asr_code 対応）

| UI言語コード | `asr_code` | 修正前 | 修正後 |
|-------------|------------|--------|--------|
| `zh-CN` | `zh` | `[]` | `["whispers2t"]` |
| `zh-TW` | `zh` | `[]` | `["whispers2t"]` |
| `pt-BR` | `pt` | `[]` | `["whispers2t", "voxtral"]` |
| `es-ES` | `es` | `[]` | `["whispers2t", "canary", "voxtral"]` |
| `es-US` | `es` | `[]` | `["whispers2t", "canary", "voxtral"]` |

### 5.2 Phase 2 の影響（廃止）

- **破壊的変更**: `Languages.get_engines_for_language()` が `DeprecationWarning` を発生
- **データ削除**: `LanguageInfo.supported_engines` フィールドが消失
- **riva 参照削除**: デッドコードの `riva` 参照がすべて削除される

### 5.3 後方互換性

- **Phase 1**: 破壊的変更なし（動作改善のみ）
- **Phase 2**: `DeprecationWarning` のみ、機能は維持（EngineMetadata に委譲）

---

## 6. 完了条件

### Phase 1（✅ 完了）

- [x] `EngineMetadata.get_engines_for_language("zh-CN")` が `["whispers2t"]` を返す
- [x] `EngineMetadata.get_engines_for_language("zh-TW")` が `["whispers2t"]` を返す
- [x] `EngineMetadata.get_engines_for_language("pt-BR")` が `["whispers2t", "voxtral"]` を返す
- [x] 新規テストが追加されている（6テスト）
- [x] 全既存テストがパス
- [ ] CI全ジョブ合格（Windows self-hosted 待ち）

### Phase 2（未着手）

- [ ] `Languages.get_engines_for_language()` が `DeprecationWarning` を発生
- [ ] `Languages.get_engines_for_language()` が `EngineMetadata` に委譲
- [ ] `LanguageInfo.supported_engines` フィールドが削除されている
- [ ] `riva` デッドコード参照がすべて削除されている
- [ ] ドキュメントが更新されている
- [ ] 全テストがパス

---

## 7. 関連情報

### 7.1 関連 Issue/PR

- #165: WhisperS2T エンジン統合（完了）
- #166: `detect_device()` リファクタリング（未着手）
- #169, #170: WhisperS2T統合関連PR（完了）
- #171: Issue #168 修正 PR（Phase 1 完了、Phase 2 進行中）

### 7.2 参考資料

- `languages.py`: 言語定義マスター、`asr_code` フィールド
- `metadata.py`: エンジンメタデータ、`supported_languages`
- `whisper_languages.py`: WhisperS2T の100言語定義

### 7.3 利用状況調査結果

```
Languages.get_engines_for_language():
  - 実コード利用: 0箇所
  - ドキュメント参照: core-api-spec.md, feature-inventory.md

EngineMetadata.get_engines_for_language():
  - 実コード利用: 3箇所
    - engine_factory.py:211
    - shared_engine_manager.py:83
    - benchmarks/common/engines.py:251
  - テスト: 多数
```
