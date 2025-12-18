"""
言語コード正規化ユーティリティ

BCP-47 言語コードの正規化と各翻訳エンジン向けの変換を提供。
Issue #168 で実装された EngineMetadata.to_iso639_1() と同じ langcodes ライブラリを使用。
"""

import langcodes

# Riva-4B プロンプト用の言語名マッピング
LANGUAGE_NAMES = {
    "ja": "Japanese",
    "en": "English",
    "zh": "Simplified Chinese",
    "zh-TW": "Traditional Chinese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Brazilian Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
}


def to_iso639_1(code: str) -> str:
    """
    BCP-47 言語コードを ISO 639-1 に変換

    langcodes ライブラリを使用（EngineMetadata.to_iso639_1 と同じ実装）

    Args:
        code: 言語コード（"ja", "zh-CN", "ZH-TW" など）

    Returns:
        ISO 639-1 言語コード（"ja", "zh" など）

    Examples:
        >>> to_iso639_1("ja")
        'ja'
        >>> to_iso639_1("zh-CN")
        'zh'
        >>> to_iso639_1("ZH-TW")  # 大文字も正規化
        'zh'
    """
    return langcodes.Language.get(code).language


def normalize_for_google(lang: str) -> str:
    """
    Google Translate 用に正規化

    Note: Google は zh-CN/zh-TW を区別するため、
          元の入力が zh-TW なら維持する

    Args:
        lang: 言語コード

    Returns:
        Google Translate 用の言語コード

    Examples:
        >>> normalize_for_google("ja")
        'ja'
        >>> normalize_for_google("zh")
        'zh-CN'
        >>> normalize_for_google("zh-TW")
        'zh-TW'
    """
    # 元の入力を保持（zh-TW の場合）
    if lang.lower() in ("zh-tw", "zh-hant"):
        return "zh-TW"
    # それ以外は ISO 639-1 に変換後、中国語のみ zh-CN に
    iso = to_iso639_1(lang)
    if iso == "zh":
        return "zh-CN"
    return iso


def normalize_for_opus_mt(lang: str) -> str:
    """
    OPUS-MT 用に正規化（ISO 639-1）

    Args:
        lang: 言語コード

    Returns:
        ISO 639-1 言語コード
    """
    return to_iso639_1(lang)


def get_language_name(lang: str) -> str:
    """
    Riva 用に言語名を取得

    Args:
        lang: 言語コード

    Returns:
        英語での言語名（例: "Japanese", "English"）
    """
    iso = to_iso639_1(lang)
    # zh-TW は特別扱い
    if lang.lower() in ("zh-tw", "zh-hant"):
        return LANGUAGE_NAMES.get("zh-TW", "Traditional Chinese")
    # LANGUAGE_NAMES に存在する場合はそれを返す
    if iso in LANGUAGE_NAMES:
        return LANGUAGE_NAMES[iso]
    # 未知の言語は langcodes から取得（language_data が必要）
    return langcodes.Language.get(lang).display_name()


def get_opus_mt_model_name(source: str, target: str) -> str:
    """
    OPUS-MT モデル名を生成

    Args:
        source: ソース言語コード
        target: ターゲット言語コード

    Returns:
        Helsinki-NLP モデル名（例: "Helsinki-NLP/opus-mt-ja-en"）
    """
    src = normalize_for_opus_mt(source)
    tgt = normalize_for_opus_mt(target)
    return f"Helsinki-NLP/opus-mt-{src}-{tgt}"
