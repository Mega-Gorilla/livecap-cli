# Issue #168: Languages.get_engines_for_language() 廃止

> **Status**: PLANNING
> **作成日:** 2025-12-05
> **関連 Issue:** #168
> **前提:** Phase 1（asr_code 対応）は PR #171 で完了済み

---

## 1. 目的

`Languages.get_engines_for_language()` を廃止し、`EngineMetadata.get_engines_for_language()` に一元化する。

## 2. 背景

### 2.1 現状の問題

| 問題 | 詳細 |
|------|------|
| **二重管理** | 言語→エンジンのマッピングが2箇所で管理されている |
| **利用状況** | `Languages` 版: 0箇所、`EngineMetadata` 版: 3箇所 |
| **`riva` デッドコード** | `Languages` に `riva` 参照があるがエンジン未実装 |
| **データ不整合** | `pt-BR`: Languages=`["riva"]` vs EngineMetadata=`["whispers2t", "voxtral"]` |

### 2.2 廃止理由

1. **実利用なし**: `Languages.get_engines_for_language()` は実コードで使用されていない
2. **上位互換が存在**: `EngineMetadata` 版が100言語対応で機能的に優れる
3. **Single Source of Truth**: エンジン情報は `EngineMetadata` に集約すべき

---

## 3. 実装計画

### 3.1 修正対象ファイル

| ファイル | 変更内容 |
|---------|----------|
| `livecap_core/languages.py` | `get_engines_for_language()` に `@deprecated` 追加、`EngineMetadata` に委譲 |
| `livecap_core/languages.py` | `LanguageInfo.supported_engines` フィールド削除 |
| `docs/architecture/core-api-spec.md` | API ドキュメント更新 |
| `docs/reference/feature-inventory.md` | リファレンス更新 |

### 3.2 コード変更

#### languages.py: get_engines_for_language() の廃止

```python
import warnings

@classmethod
def get_engines_for_language(cls, code: str) -> List[str]:
    """
    指定言語をサポートするエンジンリストを取得

    .. deprecated:: 2.1.0
       代わりに EngineMetadata.get_engines_for_language() を使用してください。
    """
    warnings.warn(
        "Languages.get_engines_for_language() is deprecated. "
        "Use EngineMetadata.get_engines_for_language() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    from livecap_core.engines.metadata import EngineMetadata
    return EngineMetadata.get_engines_for_language(code)
```

#### languages.py: LanguageInfo から supported_engines を削除

```python
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
    # supported_engines: List[str]  ← 削除
    translation_services: List[str] = field(default_factory=list)
```

---

## 4. 影響範囲

### 4.1 破壊的変更

- `Languages.get_engines_for_language()` が `DeprecationWarning` を発生
- `LanguageInfo.supported_engines` フィールドへのアクセスが失敗

### 4.2 後方互換性

- 機能は維持（`EngineMetadata` に委譲）
- 警告のみ、エラーにはならない

---

## 5. 完了条件

- [ ] `Languages.get_engines_for_language()` が `DeprecationWarning` を発生
- [ ] `Languages.get_engines_for_language()` が `EngineMetadata` に委譲
- [ ] `LanguageInfo.supported_engines` フィールドが削除されている
- [ ] `riva` デッドコード参照がすべて削除されている
- [ ] ドキュメントが更新されている
- [ ] 全テストがパス

---

## 6. 関連情報

- Phase 1（asr_code 対応）: PR #171 で完了
- Issue #168: https://github.com/Mega-Gorilla/livecap-core/issues/168
