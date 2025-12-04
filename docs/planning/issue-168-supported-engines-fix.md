# Issue #168: supported_engines 不整合修正

> **Status**: PLANNING
> **作成日:** 2025-12-05
> **関連 Issue:** #168
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

#### 問題2: `asr_code` 未対応 → ❌ 要修正

`EngineMetadata.get_engines_for_language()` が `asr_code` を考慮していないため、
地域コード付き言語（`zh-CN`, `zh-TW` など）で正しくエンジンを取得できない。

| 入力 | `Languages.get_engines_for_language()` | `EngineMetadata.get_engines_for_language()` |
|------|----------------------------------------|---------------------------------------------|
| `"zh-CN"` | `["whispers2t"]` ✅ | `[]` ❌ |
| `"zh-TW"` | `["whispers2t"]` ✅ | `[]` ❌ |
| `"pt-BR"` | `["riva"]` | `[]` ❌ |

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

### 2.2 現在の問題点

`EngineMetadata.get_engines_for_language()` の実装（`metadata.py:201-212`）：

```python
normalized = Languages.normalize(lang_code) or lang_code
# ...
for engine_id, info in cls._ENGINES.items():
    if normalized in info.supported_languages:  # ← "zh-CN" で比較
        result.append(engine_id)
```

**問題:** `normalized` (= "zh-CN") を直接 `supported_languages` と比較しているが、
WhisperS2T の `supported_languages` には "zh" しか含まれていない。

### 2.3 二重管理の問題

現在、言語→エンジンのマッピングは2箇所で管理されている：

| 場所 | 用途 | 問題点 |
|------|------|--------|
| `languages.py` の `supported_engines` | UI表示、16言語のみ | 手動管理、100言語未対応 |
| `metadata.py` の `EngineInfo.supported_languages` | エンジンの真のサポート言語 | `asr_code` 変換なし |

---

## 3. 修正方針

### 3.1 方針A: `EngineMetadata.get_engines_for_language()` で `asr_code` を使用（推奨）

**メリット:**
- 最小限の変更
- `metadata.py` の `supported_languages` は正確な値を維持
- `asr_code` の変換ロジックを一箇所に集約

**デメリット:**
- `Languages` への依存が増える

### 3.2 方針B: `metadata.py` の `supported_languages` に地域コードも追加

**メリット:**
- 依存関係なし

**デメリット:**
- データの冗長性（`zh`, `zh-CN`, `zh-TW` をすべて追加）
- 100言語 × 地域バリアント = 管理が煩雑

### 3.3 採用: 方針A

---

## 4. 実装計画

### 4.1 修正対象ファイル

| ファイル | 変更内容 |
|---------|----------|
| `livecap_core/engines/metadata.py` | `get_engines_for_language()` を `asr_code` 対応に修正 |
| `tests/core/engines/test_engine_factory.py` | `asr_code` 変換のテスト追加 |

### 4.2 修正コード

```python
# metadata.py: get_engines_for_language() の修正

@classmethod
def get_engines_for_language(cls, lang_code: str) -> list:
    """
    指定言語をサポートするエンジンIDリストを取得

    Args:
        lang_code: 言語コード（"ja", "zh-CN", "en" など）

    Returns:
        エンジンIDのリスト

    Note:
        地域コード付き言語（zh-CN, zh-TW など）は asr_code（zh）に
        変換してから比較する。これにより WhisperS2T の100言語サポートが
        正しく機能する。
    """
    from livecap_core.languages import Languages

    # 言語コードを正規化
    normalized = Languages.normalize(lang_code) or lang_code
    if not normalized:
        return []

    # asr_code を取得（zh-CN → zh, pt-BR → pt など）
    lang_info = Languages.get_info(normalized)
    asr_code = lang_info.asr_code if lang_info else normalized

    result = []
    for engine_id, info in cls._ENGINES.items():
        # asr_code で比較（WhisperS2T等の多言語エンジン対応）
        if asr_code in info.supported_languages:
            result.append(engine_id)
        # フォールバック: 正規化コードでも比較（riva等の地域コード対応エンジン）
        elif normalized in info.supported_languages:
            result.append(engine_id)
    return result
```

### 4.3 テストケース

```python
# test_engine_factory.py に追加

def test_get_engines_for_language_with_region_code():
    """Test that regional language codes are properly handled via asr_code."""
    from livecap_core.engines.metadata import EngineMetadata

    # zh-CN should find whispers2t (via asr_code "zh")
    zh_cn_engines = EngineMetadata.get_engines_for_language("zh-CN")
    assert "whispers2t" in zh_cn_engines

    # zh-TW should also find whispers2t (via asr_code "zh")
    zh_tw_engines = EngineMetadata.get_engines_for_language("zh-TW")
    assert "whispers2t" in zh_tw_engines

    # pt-BR should find whispers2t (via asr_code "pt")
    pt_br_engines = EngineMetadata.get_engines_for_language("pt-BR")
    assert "whispers2t" in pt_br_engines
```

---

## 5. 影響範囲

### 5.1 影響を受ける言語

| UI言語コード | `asr_code` | 修正前 | 修正後 |
|-------------|------------|--------|--------|
| `zh-CN` | `zh` | `[]` | `["whispers2t"]` |
| `zh-TW` | `zh` | `[]` | `["whispers2t"]` |
| `pt-BR` | `pt` | `[]` | `["whispers2t", "voxtral"]` |
| `es-ES` | `es` | `[]` | `["whispers2t", "canary", "voxtral"]` |
| `es-US` | `es` | `[]` | `["whispers2t", "canary", "voxtral"]` |

### 5.2 後方互換性

- **破壊的変更なし**: 既存の呼び出しは引き続き動作
- **動作改善のみ**: 以前は空配列を返していたケースで正しいエンジンを返す

---

## 6. 完了条件

- [ ] `EngineMetadata.get_engines_for_language("zh-CN")` が `["whispers2t"]` を返す
- [ ] `EngineMetadata.get_engines_for_language("zh-TW")` が `["whispers2t"]` を返す
- [ ] `EngineMetadata.get_engines_for_language("pt-BR")` が `["whispers2t", "voxtral"]` を返す
- [ ] 新規テストが追加されている
- [ ] 全既存テストがパス
- [ ] CI全ジョブ合格

---

## 7. 関連情報

### 7.1 関連 Issue/PR

- #165: WhisperS2T エンジン統合（完了）
- #166: `detect_device()` リファクタリング（未着手）
- #169, #170: WhisperS2T統合関連PR（完了）

### 7.2 参考資料

- `languages.py`: 言語定義マスター、`asr_code` フィールド
- `metadata.py`: エンジンメタデータ、`supported_languages`
- `whisper_languages.py`: WhisperS2T の100言語定義
