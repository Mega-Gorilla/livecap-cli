# Issue #72: テスト計画

翻訳プラグインシステムのテスト計画。

> **Note**: 進捗状況については [`./issue-72/README.md`](./issue-72/README.md) を参照。

## テストマーカー

```python
# conftest.py に追加
def pytest_configure(config):
    config.addinivalue_line("markers", "network: requires network access (Google API)")
    config.addinivalue_line("markers", "slow: slow tests (real model loading)")
    config.addinivalue_line("markers", "gpu: requires CUDA GPU")
```

## CI 実行方針

| テスト種別 | マーカー | CI 実行 |
|-----------|---------|---------|
| ユニット（モック） | なし | ✅ 常に実行 |
| Google 実 API | `@pytest.mark.network` | ❌ スキップ |
| OPUS-MT 実モデル | `@pytest.mark.slow` | ⚙️ オプション |
| Riva GPU | `@pytest.mark.gpu` | ⚙️ self-hosted のみ |

```bash
# CI デフォルト
pytest tests/core/translation -m "not network and not slow and not gpu"

# ローカル開発時（GPU なし）
pytest tests/core/translation -m "not gpu"

# フル実行（GPU 環境）
pytest tests/core/translation
```

## テストファイル構成

```
tests/core/translation/
├── __init__.py
├── test_lang_codes.py           # 言語コード正規化
├── test_retry.py                # リトライデコレータ
├── test_metadata.py             # TranslatorMetadata
├── test_exceptions.py           # 例外クラス
├── test_result.py               # TranslationResult
├── test_factory.py              # TranslatorFactory
├── test_google_translator.py    # GoogleTranslator
├── test_opus_mt_translator.py   # OpusMTTranslator
└── test_riva_instruct_translator.py  # RivaInstructTranslator

tests/integration/
└── test_translation.py          # 統合テスト
```

## ユニットテスト

### test_lang_codes.py

```python
import pytest
from livecap_cli.translation.lang_codes import (
    to_iso639_1,
    normalize_for_google,
    normalize_for_opus_mt,
    get_language_name,
    get_opus_mt_model_name,
)

def test_to_iso639_1():
    assert to_iso639_1("ja") == "ja"
    assert to_iso639_1("ja-JP") == "ja"
    assert to_iso639_1("zh-CN") == "zh"
    assert to_iso639_1("ZH-TW") == "zh"  # 大文字も正規化

def test_normalize_for_google():
    assert normalize_for_google("ja") == "ja"
    assert normalize_for_google("zh") == "zh-CN"
    assert normalize_for_google("zh-TW") == "zh-TW"  # 繁体字は維持
    assert normalize_for_google("zh-Hant") == "zh-TW"

def test_get_opus_mt_model_name():
    assert get_opus_mt_model_name("ja", "en") == "Helsinki-NLP/opus-mt-ja-en"
    assert get_opus_mt_model_name("en", "ja") == "Helsinki-NLP/opus-mt-en-ja"
```

### test_retry.py

```python
import pytest
from unittest.mock import MagicMock
from livecap_cli.translation.retry import with_retry
from livecap_cli.translation.exceptions import TranslationNetworkError

def test_retry_success_first_attempt():
    mock_func = MagicMock(return_value="success")
    decorated = with_retry(max_retries=3)(mock_func)
    assert decorated() == "success"
    assert mock_func.call_count == 1

def test_retry_success_after_failure():
    mock_func = MagicMock(side_effect=[
        TranslationNetworkError("fail1"),
        TranslationNetworkError("fail2"),
        "success",
    ])
    decorated = with_retry(max_retries=3, base_delay=0.01)(mock_func)
    assert decorated() == "success"
    assert mock_func.call_count == 3

def test_retry_exhausted():
    mock_func = MagicMock(side_effect=TranslationNetworkError("always fail"))
    decorated = with_retry(max_retries=2, base_delay=0.01)(mock_func)
    with pytest.raises(TranslationNetworkError):
        decorated()
    assert mock_func.call_count == 2
```

### test_metadata.py

```python
from livecap_cli.translation.metadata import TranslatorMetadata

def test_get_existing_translator():
    info = TranslatorMetadata.get("google")
    assert info is not None
    assert info.translator_id == "google"
    assert info.requires_model_load is False

def test_get_nonexistent_translator():
    info = TranslatorMetadata.get("nonexistent")
    assert info is None

def test_get_translators_for_pair():
    translators = TranslatorMetadata.get_translators_for_pair("ja", "en")
    assert "google" in translators
    assert "opus_mt" in translators
```

### test_google_translator.py

```python
from unittest.mock import patch, MagicMock

def test_translate_basic_mock():
    """モックを使用した基本テスト"""
    with patch("deep_translator.GoogleTranslator") as mock_gt:
        mock_gt.return_value.translate.return_value = "こんにちは"
        translator = GoogleTranslator()
        result = translator.translate("Hello", "en", "ja")
        assert result.text == "こんにちは"
        assert result.original_text == "Hello"

def test_translate_empty_text():
    """空文字列の翻訳"""
    translator = GoogleTranslator()
    result = translator.translate("", "en", "ja")
    assert result.text == ""

def test_translate_same_language_raises():
    """同一言語でエラー"""
    from livecap_cli.translation.exceptions import UnsupportedLanguagePairError
    translator = GoogleTranslator()
    with pytest.raises(UnsupportedLanguagePairError):
        translator.translate("Hello", "en", "en")

@pytest.mark.network
def test_translate_basic_real():
    """実 API を使用したテスト（CI ではスキップ）"""
    translator = GoogleTranslator()
    result = translator.translate("Hello", "en", "ja")
    assert result.text  # 何らかの翻訳が返る
```

### test_opus_mt_translator.py

```python
@pytest.mark.slow
def test_opus_mt_load_model():
    """実モデルロードテスト"""
    translator = OpusMTTranslator(source_lang="en", target_lang="ja")
    translator.load_model()
    assert translator.is_initialized()

def test_opus_mt_model_name_generation():
    """モデル名生成テスト"""
    translator = OpusMTTranslator(source_lang="ja", target_lang="en")
    assert translator.model_name == "Helsinki-NLP/opus-mt-ja-en"

def test_opus_mt_model_name_override():
    """モデル名オーバーライドテスト"""
    translator = OpusMTTranslator(
        source_lang="ja",
        target_lang="en",
        model_name="custom/model-ja-en"
    )
    assert translator.model_name == "custom/model-ja-en"
```

### test_riva_instruct_translator.py

```python
@pytest.mark.gpu
def test_riva_instruct_with_context():
    translator = RivaInstructTranslator(device="cuda")
    translator.load_model()
    context = ["VRChatで友達とドライブした。", "彼はとてもスピードを出した。"]
    result = translator.translate(
        "そのせいで今日は少し疲れている。",
        "ja", "en",
        context=context
    )
    assert "tired" in result.text.lower() or "fatigue" in result.text.lower()
```

## 統合テスト

### test_translation.py

```python
# tests/integration/test_translation.py

@pytest.mark.slow
def test_opus_mt_full_pipeline():
    """OPUS-MT のフルパイプラインテスト"""
    translator = OpusMTTranslator(source_lang="ja", target_lang="en")
    translator.load_model()

    # 文脈付き翻訳
    context = ["昨日は友達と遊んだ。"]
    result = translator.translate(
        "今日は疲れている。",
        "ja", "en",
        context=context
    )
    assert result.text
    assert result.original_text == "今日は疲れている。"

@pytest.mark.slow
def test_translator_factory():
    """TranslatorFactory の統合テスト"""
    # Google（初期化不要）
    google = TranslatorFactory.create_translator("google")
    assert google.is_initialized()

    # OPUS-MT（要モデルロード）
    opus = TranslatorFactory.create_translator("opus_mt", source_lang="ja", target_lang="en")
    assert not opus.is_initialized()
    opus.load_model()
    assert opus.is_initialized()

def test_translation_result_to_event_dict():
    """TranslationResult のイベント変換テスト"""
    result = TranslationResult(
        text="Hello",
        original_text="こんにちは",
        source_lang="ja",
        target_lang="en",
    )
    event = result.to_event_dict()
    assert event["event_type"] == "translation_result"
    assert event["translated_text"] == "Hello"
    assert event["original_text"] == "こんにちは"
```

**Note**: StreamTranscriber との統合テストは Phase 5 で追加予定。
