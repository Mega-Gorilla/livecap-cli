# Issue #168: languages.py 廃止と EngineMetadata への統合

> **Status**: READY FOR IMPLEMENTATION
> **作成日:** 2025-12-05
> **更新日:** 2025-12-05
> **関連 Issue:** #168
> **前提:** Phase 1（asr_code 対応）は PR #171 で完了済み
> **方針:** 破壊的変更を許容、シンプル化優先

---

## 1. 目的

`languages.py`（646行）を廃止し、必要最小限の機能のみを `metadata.py` に統合する。

**削減効果**: 646行 → 約5行（純削減 ~640行）

---

## 2. 背景

### 2.1 調査結果

`languages.py` の利用状況を調査した結果：

| 機能 | 実コード利用 |
|-----|-------------|
| `Languages.normalize()` | 1箇所 |
| `Languages.get_info()` | 2箇所 |
| `LanguageInfo.asr_code` | 2箇所 |
| **その他全てのメソッド** | **0箇所** |

### 2.2 実質的に必要な機能

646行のモジュールで実際に必要なのは：

- **ISO 639-1 変換**: `"zh-CN"` → `"zh"`（地域コードの除去）

これは **langcodes ライブラリ** で実現可能（約5行）。

---

## 3. 設計決定

### 3.1 廃止する機能

| 機能 | 廃止理由 |
|------|---------|
| **`Languages.normalize()`** | langcodes が大文字小文字を自動正規化するため不要 |
| **`Languages.get_info()`** | `to_iso639_1()` で代替 |
| **エイリアス** (`zh`→`zh-CN` 等) | 使用箇所0、暗黙変換は危険 |
| **`auto` 特殊処理** | 使用箇所0、各エンジンで扱う |
| **`LanguageInfo` データクラス** | 使用箇所0 |
| **表示名・国旗等のメタデータ** | 使用箇所0 |

### 3.2 スコープ: ASR エンジン専用

**本 Issue のスコープは ASR（音声認識）エンジンのみ**。翻訳サービスは対象外。

```
ユーザー入力: ISO 639-1 ("ja") または BCP-47 ("zh-CN")
     │
     ├─→ ASR Engine: to_iso639_1() で変換 → "zh"
     │   （WhisperS2T, Canary, Voxtral 等）
     │
     └─→ Translation Service: 変換不要、元の入力をそのまま使用
         （RIVA, Google Translate 等 - 将来実装）
```

**ASR と Translation の違い**:

| サービス種別 | 必要な形式 | 変換関数 | 実装時期 |
|-------------|-----------|---------|---------|
| ASR | ISO 639-1 (`zh`) | `to_iso639_1()` | **今回** |
| Translation | BCP-47 (`zh-CN`) | `to_bcp47()` | 将来 |

**翻訳サービスの考慮事項**:
- ユーザーが ISO 639-1 (`zh`) を入力した場合、翻訳サービスは BCP-47 (`zh-CN`) が必要
- `to_bcp47()` はデフォルト地域マッピングが必要（例: `zh` → `zh-CN`）
- 翻訳メタデータ（`translation_services` 等）は Issue #168 のスコープ外
- 翻訳モジュール実装時に別途設計

**補足: Google Translate の言語コード対応**:
- Google Translate は ISO 639-1 (`zh`) と BCP-47 (`zh-CN`) の両方を受け付ける
- 中国語の場合: `zh-CN`（簡体字）と `zh-TW`（繁体字）で翻訳結果が異なる
- ISO 639-1 のみ指定時のデフォルト動作は API に依存
- `to_bcp47()` 実装時に Google/RIVA 両対応を検討

### 3.3 設計方針（ASR）

```
ユーザー入力: "ja", "zh-CN", "zh-TW", "vi" など
      ↓ to_iso639_1()（地域コード除去のみ）
ISO 639-1: "ja", "zh", "zh", "vi"
      ↓ 各エンジンでバリデーション
WhisperS2T: WHISPER_LANGUAGES_SET でチェック → エラー or 成功
```

**ポイント**:
- **大文字小文字の正規化**: langcodes が自動で処理（`ZH-CN` → `zh`）
- **バリデーションは各エンジンに委譲**: Single Source of Truth は各エンジンの `supported_languages`
- **langcodes ライブラリ使用**: BCP-47 → ISO 639-1 変換を標準ライブラリに委譲
- **`auto` の扱い**: パススルー（各エンジンで処理）

### 3.4 関数名の設計

標準規格に基づいた明確な命名を採用：

| 処理 | 関数名 | 理由 |
|------|--------|------|
| BCP-47 → ISO 639-1 | **`to_iso639_1()`** | 標準規格名で変換先が明確 |
| ISO 639-1 → BCP-47 | `to_bcp47()` | 将来実装時（RIVA等） |

**ISO 639-3（3文字コード）の扱い**:
- `to_iso639_1()` は langcodes の `language` サブタグを返す
- ISO 639-1（2文字）が存在しない言語は ISO 639-3（3文字）を返す
- 例: `yue`（粵語/Cantonese）、`haw`（ハワイ語）
- Whisper は `yue` 等の3文字コードを明示的にサポート（`WHISPER_LANGUAGES` に含まれる）
- **実装方針**: 3文字コードも許容。バリデーションは各エンジンの `supported_languages` で行う。

**廃止した候補**:
- `to_asr_code()`: 「ASR」は用途であり、変換形式ではない
- `to_google_code()`: Google Translate は現在 ISO 639-1 を受け付けるため不要

### 3.5 ライブラリ調査結果

言語コード変換用ライブラリを調査した結果、**langcodes を採用**：

| ライブラリ | 最新版 | 判断 |
|-----------|--------|------|
| [langcodes](https://pypi.org/project/langcodes/) | 3.5.1 (2025-12) | **採用** - BCP-47完全対応、軽量 |
| [pycountry](https://github.com/pycountry/pycountry) | - | 不採用 - ISO全般、重すぎる |
| [iso639-lang](https://pypi.org/project/iso639-lang/) | 2025-07 | 不採用 - 今回のユースケースには過剰 |

**langcodes 採用理由**:
1. **エッジケース対応**: `zh-Hans`（スクリプトサブタグ）等も正しく処理
2. **コード削減**: マッピングテーブル不要、約5行で実装可能
3. **メンテナンスフリー**: 新しい地域コードへの対応が不要
4. **標準準拠**: BCP-47/ISO 639 の正式な解析
5. **軽量**: 純Python、追加依存なし

**トレードオフ（10行マッピング vs langcodes）**:

| 観点 | 10行マッピング | langcodes |
|------|---------------|-----------|
| 依存追加 | なし | +1（~50KB wheel） |
| エッジケース | `zh-Hans` 未対応 | 全BCP-47対応 |
| 大文字正規化 | 手動実装必要 | 自動対応 |
| メンテナンス | 手動追加 | 不要 |
| ビルド影響 | なし | 軽微（純Python） |

**採用判断**: エッジケース対応とメンテナンス性を優先。依存サイズ（~50KB）は許容範囲。

---

## 4. 実装

### 4.1 修正対象ファイル

| ファイル | 変更内容 |
|---------|----------|
| `pyproject.toml` | `langcodes` 依存を追加 |
| `livecap_core/languages.py` | **削除** |
| `livecap_core/__init__.py` | `Languages` エクスポート削除 |
| `livecap_core/engines/metadata.py` | `to_iso639_1()` を追加（~5行） |
| `livecap_core/engines/whispers2t_engine.py` | `EngineMetadata.to_iso639_1()` を使用 |
| `docs/architecture/core-api-spec.md` | `Languages` セクション削除 |
| `docs/reference/feature-inventory.md` | 言語API更新 |

### 4.2 pyproject.toml への追加

```toml
[project]
dependencies = [
    # ... 既存の依存 ...
    "langcodes>=3.4.0",
]
```

### 4.3 metadata.py への追加コード

```python
import langcodes

@classmethod
def to_iso639_1(cls, code: str) -> str:
    """BCP-47 言語コードを ISO 639-1 に変換。"""
    return langcodes.Language.get(code).language
```

**追加行数**: 約5行（import 1行 + 関数4行）

**対応例**:
```python
to_iso639_1("zh-CN")    # → "zh"
to_iso639_1("zh-TW")    # → "zh"
to_iso639_1("zh-Hans")  # → "zh" (スクリプトサブタグも対応)
to_iso639_1("pt-BR")    # → "pt"
to_iso639_1("ja")       # → "ja"
to_iso639_1("ZH-CN")    # → "zh" (大文字も自動正規化)
to_iso639_1("yue")      # → "yue" (Cantonese, パススルー)
to_iso639_1("auto")     # → "auto" (パススルー、各エンジンで処理)
```

**エラーケース**:
```python
to_iso639_1("invalid-code")  # → LanguageTagError 例外
to_iso639_1("")              # → LanguageTagError 例外
to_iso639_1("zz")            # → "zz" (パススルー、エンジンでバリデーション)
```

**無効コード時の挙動**:
- **形式エラー** (`invalid-code`, 空文字): `LanguageTagError` 例外を上位に伝播
- **未知の言語コード** (`zz`, `xx`): パススルー → 各エンジンのバリデーションで `ValueError`
- **設計方針**: 静かな失敗を避け、エラーは明示的に伝播。フォールバックは行わない。

### 4.4 whispers2t_engine.py の修正

```python
# Before
from livecap_core.languages import Languages
...
lang_info = Languages.get_info(language)
asr_language = lang_info.asr_code if lang_info else language

# After
from .metadata import EngineMetadata
...
asr_language = EngineMetadata.to_iso639_1(language)
```

### 4.5 get_engines_for_language() の修正

```python
# Before（languages.py に依存）
from livecap_core.languages import Languages
normalized = Languages.normalize(lang_code) or lang_code
lang_info = Languages.get_info(normalized)
asr_code = lang_info.asr_code if lang_info else normalized

# After（自己完結、1行）
iso_code = cls.to_iso639_1(lang_code)
```

---

## 5. 影響範囲

### 5.1 破壊的変更

| 変更 | 影響 |
|-----|------|
| `languages.py` 削除 | `from livecap_core.languages import Languages` が失敗 |
| `Languages` クラス削除 | 全メソッド使用不可 |
| `LanguageInfo` 削除 | データクラス使用不可 |

### 5.2 移行ガイド

```python
# Before
from livecap_core.languages import Languages
Languages.normalize("zh-TW")  # 正規化 → 廃止（不要）
info = Languages.get_info("ja")
asr_code = info.asr_code

# After
from livecap_core.engines import EngineMetadata
iso_code = EngineMetadata.to_iso639_1("zh-CN")  # "zh"
iso_code = EngineMetadata.to_iso639_1("ja")     # "ja"
```

### 5.3 テストへの影響

以下のテストは引き続きパスする必要がある：

- `test_zh_cn_finds_whispers2t_via_asr_code`
- `test_zh_tw_finds_whispers2t_via_asr_code`
- `test_pt_br_finds_whispers2t_via_asr_code`
- `test_es_es_finds_whispers2t_via_asr_code`
- `test_base_language_codes_still_work`
- `test_regional_codes_find_correct_engines`

---

## 6. 実装順序

```
Phase A: 依存追加と機能実装
  1. pyproject.toml に langcodes 依存を追加
  2. metadata.py に to_iso639_1() を追加

Phase B: 依存を切り替え
  3. whispers2t_engine.py を EngineMetadata.to_iso639_1() に移行
  4. get_engines_for_language() を自己完結に修正
  5. テスト実行・確認

Phase C: 削除
  6. languages.py を削除
  7. __init__.py から Languages を削除
  8. ドキュメント更新
```

---

## 7. 完了条件

- [ ] `pyproject.toml` に `langcodes` 依存が追加されている
- [ ] `EngineMetadata.to_iso639_1()` が実装されている（langcodes 使用）
- [ ] `whispers2t_engine.py` が `EngineMetadata.to_iso639_1()` を使用している
- [ ] `EngineMetadata.get_engines_for_language()` が自己完結している
- [ ] `languages.py` が削除されている
- [ ] `livecap_core/__init__.py` から `Languages` が削除されている
- [ ] 全テストがパス
- [ ] ドキュメントが更新されている

---

## 8. 将来の拡張（実装しない）

以下は将来必要になった時点で実装する：

| 機能 | 用途 | 実装時期 |
|------|------|---------|
| `to_bcp47()` | 翻訳サービス（RIVA, Google Translate）統合時 | 翻訳モジュール実装時 |

### 8.1 to_bcp47() の設計メモ

翻訳サービス（RIVA, Google Translate）実装時に必要となる逆変換：

```python
# ISO 639-1 → BCP-47（デフォルト地域付き）
def to_bcp47(code: str, default_region: Optional[str] = None) -> str:
    """
    ISO 639-1 を BCP-47 に変換。
    既に BCP-47 の場合はそのまま返す。
    """
    # langcodes で判定可能
    pass
```

**デフォルト地域マッピング（案）**:

| ISO 639-1 | デフォルト BCP-47 | 理由 |
|-----------|------------------|------|
| `zh` | `zh-CN` | 簡体字中国語をデフォルト |
| `pt` | `pt-BR` | ブラジルポルトガル語が主流 |
| `es` | `es-ES` | スペイン語（スペイン） |

**注意**: デフォルト地域の選択はビジネス要件による。翻訳モジュール実装時に決定。

---

## 9. 関連情報

### 9.1 参考リンク

- Phase 1（asr_code 対応）: PR #171 で完了
- Issue #168: https://github.com/Mega-Gorilla/livecap-core/issues/168

### 9.2 調査で判明した未使用機能一覧

<details>
<summary>クリックして展開</summary>

```
Languages.normalize()                  # 1箇所 → 廃止
Languages.get_info()                   # 2箇所 → to_iso639_1() に置換
Languages.get_display_name()           # 0箇所
Languages.get_google_code()            # 0箇所
Languages.from_windows_lcid()          # 0箇所
Languages.get_supported_codes()        # 0箇所
Languages.get_engines_for_language()   # 0箇所
Languages.is_auto()                    # 0箇所
Languages.is_valid()                   # 0箇所
Languages.get_all_codes()              # 0箇所
Languages.get_aliases()                # 0箇所
Languages.get_name()                   # 0箇所
Languages.get_languages_for_translation_service()  # 0箇所
Languages.get_translation_languages_dict()         # 非推奨
Languages.get_transcription_languages_dict()       # 非推奨

LanguageInfo.display_name              # 0箇所
LanguageInfo.english_name              # 0箇所
LanguageInfo.native_name               # 0箇所
LanguageInfo.flag                      # 0箇所
LanguageInfo.iso639_1                  # 0箇所
LanguageInfo.iso639_3                  # 0箇所
LanguageInfo.windows_lcid              # 0箇所
LanguageInfo.google_code               # 0箇所
LanguageInfo.translation_code          # 0箇所
LanguageInfo.supported_engines         # 0箇所
LanguageInfo.translation_services      # 0箇所
```

</details>
