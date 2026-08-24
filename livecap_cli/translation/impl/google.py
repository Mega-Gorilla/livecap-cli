"""Google Translate 実装 (Issue #402)

これは **Google のウェブ版をスクレイピングしている**。公式 API ではない。
Google 側の変更で壊れることを前提とし、壊れたときの調査手順は
``docs/troubleshooting/translation.md`` に置いてある。

なぜ deep-translator を使わないか
--------------------------------
以前は ``deep-translator`` 経由で同じエンドポイントを叩いていたが、同ライブラリは
``requests.get()`` を**ヘッダ無しで**呼ぶため User-Agent が ``python-requests/2.x``
になる。Google はこれを絞っており、**HTTP 200 のまま本文に "Error 500" ページ**を
返す。実測で UA 無しは 10 回中 5 回失敗、ブラウザ UA では 10/10 成功した。

``deep-translator`` には ``headers`` も ``session`` も渡す口が無く (引数は
``proxies`` のみ)、最新 1.11.4 でも該当コードは同一。最終リリースは 2023-06-28 で
上流の対応も見込めないため、この 1 経路だけを自前に置き換えた。

設計上の約束
-----------
* **リトライしない。** HTTP は 1 試行のみで、失敗は型で分類して投げる。何回試すかは
  用途を知っている呼び出し側が決める (realtime は fail fast、ファイルは再試行)。
* **翻訳対象テキストを例外・ログに出さない。** テキストは GET query に入るので、
  requests 由来の例外をそのまま chain すると発話が漏れる。``from None`` で切り、
  診断情報は構造化フィールドで持ち越す。
* **context を使わない。** 改行連結方式は Google では行単位に訳されて文が壊れる。
* **Session を再利用する。** 毎回新規接続だと字幕 1 本ごとに TLS ハンドシェイクが
  走る (実測 403ms → 191ms)。
* **インスタンスを複数の StreamTranscriber で共有しない。** ``requests.Session``
  の並行利用は安全と保証されていない。source ごとに生成すること。
"""

from __future__ import annotations

import logging
from html.parser import HTMLParser
from typing import Any, List, Optional, Tuple
from urllib.parse import urlencode

import requests

from ..base import BaseTranslator
from ..exceptions import (
    TranslationError,
    TranslationNetworkError,
    UnsupportedLanguagePairError,
)
from ..lang_codes import normalize_for_google, to_iso639_1
from ..result import TranslationResult

logger = logging.getLogger(__name__)

__all__ = ["GoogleTranslator"]

#: スクレイピング対象。``/m`` はスクリプト無しの軽量版で、翻訳結果が HTML に直接載る。
ENDPOINT = "https://translate.google.com/m"

#: 実在するブラウザの UA。``python-requests/2.x`` は絞られる (本 module の docstring)。
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

#: 翻訳結果を含む要素。過去に ``t0`` から変わっており、また変わり得る。
RESULT_CLASS = "result-container"

#: (connect, read)。realtime 字幕なので read は短く保つ。
DEFAULT_TIMEOUT: Tuple[float, float] = (3.0, 5.0)

#: percent-encode 後の URL 長上限。実測では ~16.3KB で HTTP 400 になる
#: (16254 bytes → 200 / 16454 bytes → 400)。余裕を持たせた値。
#: **文字数ではなくバイト長で測る** — 同じ 1500 文字でも ASCII 1.5KB、
#: 日本語 13.5KB、絵文字 18KB と大きく異なるため。
MAX_ENCODED_URL_BYTES = 12_000

#: 本文にエラーページが埋め込まれた 200 応答を判定する目印。翻訳結果そのものに
#: "Error 500" が含まれ得るので、**成功要素が取れなかった場合にのみ**参照する。
_ERROR_PAGE_MARKERS = ("Error 500 (Server Error)", "Error 502", "Error 503")

#: リトライする価値がある HTTP status。それ以外の 4xx は恒久的。
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class _ResultExtractor(HTMLParser):
    """``div.result-container`` の中身をテキストとして取り出す。

    beautifulsoup4 を使わないのは、それが deep-translator の推移的依存でしかなく、
    同ライブラリを外すと存在が保証されなくなるため (#402 D3)。

    ``convert_charrefs=True`` (既定) により ``&#39;`` 等は自動でアンエスケープされる。
    入れ子と ``<br>`` は現在の応答には現れないが、Google の出力は制御できないので
    深さカウントで防御しておく。
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._depth = 0
        self._done = False
        self._parts: List[str] = []
        self.found = False

    @property
    def _capturing(self) -> bool:
        return self._depth > 0 and not self._done

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if self._capturing:
            if tag == "div":
                self._depth += 1
            elif tag == "br":
                self._parts.append("\n")
            return
        # `_done` を見ないと、結果 div を閉じた直後の <div class="links-container">
        # (ページ末尾のリンク集) から再び拾ってしまう。
        if not self._done and tag == "div" and RESULT_CLASS in dict(attrs).get("class", ""):
            self._depth = 1
            self.found = True

    def handle_startendtag(self, tag: str, attrs: Any) -> None:
        if self._capturing and tag == "br":
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._capturing and tag == "div":
            self._depth -= 1
            if self._depth == 0:
                self._done = True

    def handle_data(self, data: str) -> None:
        if self._capturing:
            self._parts.append(data)

    @property
    def text(self) -> str:
        return "".join(self._parts)


def _extract_result(html: str) -> Optional[str]:
    """成功要素の中身。要素が無ければ ``None``。"""
    parser = _ResultExtractor()
    parser.feed(html)
    parser.close()
    return parser.text if parser.found else None


class GoogleTranslator(BaseTranslator):
    """Google Translate (ウェブ版のスクレイピング)

    Examples:
        >>> translator = GoogleTranslator()
        >>> result = translator.translate("こんにちは", "ja", "en")
        >>> print(result.text)
        "Hello"
        >>> translator.cleanup()
    """

    def __init__(
        self,
        default_context_sentences: int = 0,
        timeout: Optional[Tuple[float, float]] = None,
        transport: Optional[requests.Session] = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            default_context_sentences: **0 固定を推奨。** 本 adapter は context を
                使わない (下記 ``translate`` 参照)。値は互換のため受け取るだけ。
            timeout: ``(connect, read)`` 秒。既定は :data:`DEFAULT_TIMEOUT`。
            transport: 既存の ``requests.Session``。**渡した側が所有する** —
                :meth:`cleanup` は close しない。省略時は自前で生成し、自前のものは
                :meth:`cleanup` が close する。
        """
        super().__init__(default_context_sentences=default_context_sentences, **kwargs)
        self._timeout = timeout or DEFAULT_TIMEOUT
        self._owns_session = transport is None
        self._session = transport or self._build_session()
        self._initialized = True  # ウェブ版なのでモデルロード不要

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        session.headers.update({"User-Agent": BROWSER_UA})
        return session

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Optional[List[str]] = None,
    ) -> TranslationResult:
        """テキストを翻訳する (HTTP 1 試行のみ)。

        Args:
            text: 翻訳対象テキスト
            source_lang: ソース言語コード (BCP-47)
            target_lang: ターゲット言語コード (BCP-47)
            context: **無視される。** 本 adapter は文脈を使わない — 改行で連結して
                送ると Google は行単位に訳すため、分割された 1 文が壊れる
                (`'昨日は\\n雨が\\n降りました'` → `'Yesterday\\nrain\\nI got off'`)。
                さらに改行が統合されると文脈全体が結果として返る危険もある。
                引数は Protocol 互換のために受け取るだけ。

        Returns:
            TranslationResult

        Raises:
            UnsupportedLanguagePairError: 同一言語が指定された場合
            TranslationNetworkError: リトライする価値のある失敗 (5xx / 429 / 通信)
            TranslationError: 恒久的な失敗 (4xx / 解析不能 / 長すぎる入力)
        """
        if not text or not text.strip():
            return TranslationResult(
                text="",
                original_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
            )

        if to_iso639_1(source_lang) == to_iso639_1(target_lang):
            raise UnsupportedLanguagePairError(
                source_lang, target_lang, self.get_translator_name()
            )

        params = {
            "sl": normalize_for_google(source_lang),
            "tl": normalize_for_google(target_lang),
            "q": text,
            "hl": "en-US",
        }
        self._check_url_length(params)

        translated = self._request(params)

        return TranslationResult(
            text=translated,
            original_text=text,
            source_lang=source_lang,
            target_lang=target_lang,
        )

    def _check_url_length(self, params: dict) -> None:
        """送信前に長さを弾く。実際に 400 を食らってから気付かないため。"""
        encoded_length = len(ENDPOINT) + 1 + len(urlencode(params))
        if encoded_length > MAX_ENCODED_URL_BYTES:
            # 長さのみ報告する。テキストそのものは出さない (#402 D8)。
            raise TranslationError(
                "Text is too long for Google Translate: encoded request would be "
                f"{encoded_length} bytes (limit {MAX_ENCODED_URL_BYTES}).",
                provider="google",
                reason="request_too_long",
            )

    def _request(self, params: dict) -> str:
        """1 回だけ HTTP を投げ、結果テキストを返す。

        例外は必ず ``from None`` で chain を切る — ``requests`` の例外文字列には
        ``q=`` を含む URL 全体、つまり**発話内容**が入っており、呼び出し側が
        ``exc_info=True`` でログを出すと ``__cause__`` 経由で漏れるため (#402 D8)。
        """
        try:
            response = self._session.get(ENDPOINT, params=params, timeout=self._timeout)
        except requests.Timeout:
            raise TranslationNetworkError(
                "Google Translate request timed out",
                provider="google",
                reason="timeout",
            ) from None
        except requests.RequestException as exc:
            raise TranslationNetworkError(
                f"Google Translate request failed: {type(exc).__name__}",
                provider="google",
                reason="transport",
            ) from None

        status = response.status_code
        if status != 200:
            message = f"Google Translate request failed: HTTP {status}"
            if status in RETRYABLE_STATUS:
                raise TranslationNetworkError(
                    message, provider="google", reason="http_status", status_code=status
                ) from None
            raise TranslationError(
                message, provider="google", reason="http_status", status_code=status
            ) from None

        body = response.text
        result = _extract_result(body)

        if result is None:
            # 成功要素が取れなかったときに限りエラーページを疑う。翻訳結果に
            # "Error 500" が含まれる可能性があるため、順序が逆だと誤判定する。
            if any(marker in body for marker in _ERROR_PAGE_MARKERS):
                raise TranslationNetworkError(
                    "Google Translate returned an error page with HTTP 200",
                    provider="google",
                    reason="embedded_error_page",
                ) from None
            raise TranslationError(
                "Google Translate response did not contain a result element. "
                "The page layout likely changed - see docs/troubleshooting/translation.md.",
                provider="google",
                reason="layout_changed",
            ) from None

        if not result.strip():
            raise TranslationError(
                "Google Translate returned an empty result",
                provider="google",
                reason="empty_result",
            ) from None

        return result

    def cleanup(self) -> None:
        """自前で生成した Session だけを close する。

        注入された Session は**注入元が所有する**ので触らない (#402 D9)。
        """
        if self._owns_session and self._session is not None:
            self._session.close()

    def get_translator_name(self) -> str:
        return "google"

    def get_supported_pairs(self) -> List[Tuple[str, str]]:
        """空リスト = 全言語ペア対応。"""
        return []
