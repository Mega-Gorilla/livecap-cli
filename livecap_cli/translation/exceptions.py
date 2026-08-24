"""
翻訳エラーの例外クラス階層

翻訳処理で発生する各種エラーを分類するための例外クラスを定義。

Issue #402: エラーは**構造化フィールド**を持つ。翻訳対象のテキストは GET query
の一部として送られるため、通信ライブラリの例外文字列には発話内容が percent-encode
された URL ごと含まれる。それを握り潰さずに診断するには、元例外を chain するのでは
なく「何が起きたか」だけを型付きで持ち越す必要がある (`from None` と併用)。

**例外メッセージ・フィールドに翻訳対象テキストを入れてはならない。**
"""

from typing import Optional


class TranslationError(Exception):
    """翻訳エラーの基底クラス

    Attributes:
        provider: 翻訳エンジン識別子 ("google" 等)
        reason: 機械可読な失敗理由 ("http_status" / "transport" / "empty_result" 等)
        status_code: HTTP ステータス (該当する場合)

    Note:
        いずれのフィールドにも**翻訳対象テキストを含めない**。呼び出し側が
        ``exc_info=True`` でログを出しても発話が漏れないことが要件 (#402 D8)。
    """

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        reason: Optional[str] = None,
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.reason = reason
        self.status_code = status_code


class TranslationNetworkError(TranslationError):
    """ネットワーク関連エラー（API 失敗、タイムアウト）

    リトライする価値がある失敗。``retry.py`` はこの型のみを再試行する。
    """

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
            f"Language pair ({source} -> {target}) not supported by {translator}",
            provider=translator,
            reason="unsupported_language_pair",
        )
