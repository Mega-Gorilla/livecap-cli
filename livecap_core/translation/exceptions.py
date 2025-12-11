"""
翻訳エラーの例外クラス階層

翻訳処理で発生する各種エラーを分類するための例外クラスを定義。
"""


class TranslationError(Exception):
    """翻訳エラーの基底クラス"""

    pass


class TranslationNetworkError(TranslationError):
    """ネットワーク関連エラー（API 失敗、タイムアウト）"""

    pass


class TranslationModelError(TranslationError):
    """モデル関連エラー（ロード失敗、推論失敗）"""

    pass


class UnsupportedLanguagePairError(TranslationError):
    """未サポートの言語ペア"""

    def __init__(self, source: str, target: str, translator: str):
        self.source = source
        self.target = target
        self.translator = translator
        super().__init__(
            f"Language pair ({source} -> {target}) not supported by {translator}"
        )
