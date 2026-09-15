"""Google Translate 実装 (Issue #402 / #442)

これは **Google の非公式エンドポイントを叩いている**。公式 API ではない。
Google 側の変更で壊れることを前提とし、壊れたときの調査手順は
``docs/troubleshooting/translation.md`` に置いてある。

経路の変遷
---------
* **〜2026-08**: ``translate.google.com/m`` (スクリプト無しの HTML ページ) をスクレイピング。
  ``deep-translator`` が UA 無しで叩いて絞られたため自前 adapter に置き換えた (#402)。
* **2026-09**: ``/m`` が Google の abuse 検知に振り分けられ、**302 → ``www.google.com/sorry/``
  → 429 + reCAPTCHA** になった (#442)。ヘッダを揃えても変わらず、CAPTCHA はプログラム
  から突破できない。同じネットワーク・同じ UA で ``translate.googleapis.com/translate_a/single``
  (``client=gtx``、JSON) は通ったため、そちらへ切り替えた。**こちらも非公式である。**
  恒久的な保証は公式 Cloud Translation API (API キー必須) にしか無い。

なぜ deep-translator を使わないか
--------------------------------
``requests.get()`` を**ヘッダ無しで**呼ぶため User-Agent が ``python-requests/2.x``
になり、Google に絞られる。``headers`` も ``session`` も渡す口が無く、上流の対応も
見込めないため (最終リリース 2023-06)、この 1 経路だけを自前に置き換えた。

設計上の約束
-----------
* **リトライしない。** HTTP は 1 試行のみで、失敗は型で分類して投げる。何回試すかは
  用途を知っている呼び出し側が決める (realtime は fail fast、ファイルは再試行)。
* **bot 判定 (reCAPTCHA) は再試行しない型で投げる。** 待っても解消しない種類の 429 で、
  再送は「unusual traffic」の判定を強めるだけである (#442)。
* **翻訳対象テキストを例外・ログに出さない。** テキストは GET query に入るので、
  requests 由来の例外をそのまま chain すると発話が漏れる。``from None`` で切り、
  診断情報は構造化フィールドで持ち越す。sorry ページの URL も ``continue=`` に
  query 全体を含むため、**URL を例外メッセージに入れない**。
* **無効な言語コードは送信前に弾く。** gtx は ``tl=xx`` でも **HTTP 200 で原文を
  そのまま返す** (実測)。送ってからでは失敗を検出できない。
* **context を使わない。** 改行連結方式は Google では行単位に訳されて文が壊れる。
* **Session を再利用する。** 毎回新規接続だと字幕 1 本ごとに TLS ハンドシェイクが
  走る (実測 403ms → 191ms)。
* **インスタンスを複数の StreamTranscriber で共有しない。** ``requests.Session``
  の並行利用は安全と保証されていない。source ごとに生成すること。
"""

from __future__ import annotations

import json
from typing import Any, List, Optional, Tuple
from urllib.parse import urlencode, urlparse

import requests

from ..base import BaseTranslator
from ..exceptions import (
    TranslationError,
    TranslationNetworkError,
    UnsupportedLanguagePairError,
)
from ..lang_codes import is_known_language, normalize_for_google, to_iso639_1
from ..result import TranslationResult

__all__ = ["GoogleTranslator"]

#: 非公式の JSON エンドポイント。``client=gtx`` + ``dt=t`` + ``dj=1`` で
#: ``{"sentences": [{"trans": ..., "orig": ...}, ...], "src": ...}`` が返る。
#: 旧経路 ``translate.google.com/m`` は 2026-09 に reCAPTCHA で塞がれた (#442)。
ENDPOINT = "https://translate.googleapis.com/translate_a/single"

#: 毎回送る固定パラメータ。``dj=1`` でオブジェクト形式にする — ``dt=t`` だけの
#: 配列形式は位置依存で、要素が増減すると黙って壊れる。
FIXED_PARAMS = {"client": "gtx", "dt": "t", "dj": "1"}

#: 実在するブラウザの UA。``python-requests/2.x`` は絞られる (本 module の docstring)。
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

#: (connect, read)。realtime 字幕が主用途なので短く保つ。実測は Session 再利用時の
#: 中央値 155-191ms、観測した最悪 1331ms なので、read 2.5s はその倍近い余裕がある。
#: 合計 4.0s が :attr:`estimated_attempt_seconds` の見積値になる (保証ではない)。
DEFAULT_TIMEOUT: Tuple[float, float] = (1.5, 2.5)

#: percent-encode 後の URL 長上限。旧経路の実測では ~16.3KB で HTTP 400 になった
#: (16254 bytes → 200 / 16454 bytes → 400)。gtx でも GET なので同じ余裕で運用する。
#: **文字数ではなくバイト長で測る** — 同じ 1500 文字でも ASCII 1.5KB、
#: 日本語 13.5KB、絵文字 18KB と大きく異なるため。
MAX_ENCODED_URL_BYTES = 12_000

#: 本文にエラーページが埋め込まれた 200 応答を判定する目印。翻訳結果そのものに
#: "Error 500" が含まれ得るので、**JSON として読めなかった場合にのみ**参照する。
_ERROR_PAGE_MARKERS = ("Error 500 (Server Error)", "Error 502", "Error 503")

#: bot 判定の目印。実測した形は 2 つある (どちらも 2026-09-15):
#:
#: * ``/m``: 302 で ``www.google.com/sorry/`` へ飛ばされ、reCAPTCHA 付きの 429
#:   ("Our systems have detected unusual traffic from your computer network")
#: * ``gtx``: redirect 無しで**直接 429**、本文は ``<title>Sorry...</title>`` の HTML
#:   ("your computer or network may be sending automated queries")
#:
#: 後者は CAPTCHA も無く、素の rate limit と見分けるには本文の文言しか無い。
_BOT_CHALLENGE_MARKERS = (
    "captcha-form",
    "recaptcha",
    "unusual traffic",
    "automated queries",
    "<title>Sorry...</title>",
)

#: リトライする価値がある HTTP status。それ以外の 4xx は恒久的。
#: **429 でも bot 判定を伴うものは除く** — :func:`_is_bot_challenge` が先に捕まえる。
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def _extract_translation(body: str) -> Optional[str]:
    """gtx の JSON から翻訳文を取り出す。読めなければ ``None``。

    ``sentences[*].trans`` を**連結**する。複数文は要素が分かれ、改行は ``trans``
    の中に保持される (実測)。``trans`` を持たない要素 (``dt`` を増やすと transliteration
    だけの要素が末尾に付く) は無視する。

    ``None`` を返すのは「JSON ではない / dict ではない / ``sentences`` が list で
    ない / ``trans`` が 1 つも無い」のいずれか。呼び出し側はそれを
    ``layout_changed`` (恒久) か ``embedded_error_page`` (一時) に分類する。
    """
    try:
        payload = json.loads(body)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    sentences = payload.get("sentences")
    if not isinstance(sentences, list):
        return None
    parts = [s["trans"] for s in sentences if isinstance(s, dict) and isinstance(s.get("trans"), str)]
    if not parts:
        return None
    return "".join(parts)


def _is_bot_challenge(response: Any) -> bool:
    """Google の abuse 検知に振り分けられた応答か。

    * ``www.google.com/sorry/`` へ redirect された (``/m`` で実測した形)
    * 429 で本文に reCAPTCHA の目印がある (redirect 無しで返る場合に備える)

    素の 429 (目印無し) は**含めない** — 一時的な rate limit の可能性があり、
    従来どおり :data:`RETRYABLE_STATUS` として扱う。
    """
    url = urlparse(getattr(response, "url", "") or "")
    if url.netloc == "www.google.com" and url.path.startswith("/sorry/"):
        return True
    if response.status_code == 429:
        body = getattr(response, "text", "") or ""
        return any(marker in body for marker in _BOT_CHALLENGE_MARKERS)
    return False


class GoogleTranslator(BaseTranslator):
    """Google Translate (非公式 JSON エンドポイント)

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
        self._session = transport if transport is not None else requests.Session()
        self._initialized = True  # ウェブ経路なのでモデルロード不要

    @property
    def estimated_attempt_seconds(self) -> float:
        """1 回の翻訳にかかる時間の**見積**。保証ではない。

        ``connect + read`` を返すが、これは上限として保証できる値ではない —
        requests の read timeout は「サーバからバイトが届くまでの待ち時間」であり、
        レスポンス全体の wall-clock 上限ではないとドキュメントされている。少しずつ
        送り続けるサーバに対して総時間が伸びる可能性は、実装・版・OS に依存する。

        したがってこの値は :class:`RetryPolicy` が「次の試行を始めてよいか」を
        判断するための**保守的な見積**としてのみ使う。deadline を hard bound として
        保証するものではない。
        """
        connect, read = self._timeout
        return connect + read

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
            UnsupportedLanguagePairError: 同一言語、または**実在しない言語コード**が
                指定された場合。gtx は ``tl=xx`` でも 200 で原文を返すので、送信前に弾く
            TranslationNetworkError: リトライする価値のある失敗 (5xx / 素の 429 / 通信)
            TranslationError: 恒久的な失敗 (4xx / bot 判定 / 解析不能 / 長すぎる入力)
        """
        if not text or not text.strip():
            return TranslationResult(
                text="",
                original_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
            )

        # gtx は実在しないコードでも 200 で原文を返す (実測: tl=xx -> "Hello")。
        # 送ってからでは検出できないので、ここで弾く。
        if not (is_known_language(source_lang) and is_known_language(target_lang)):
            raise UnsupportedLanguagePairError(
                source_lang, target_lang, self.get_translator_name()
            )

        if to_iso639_1(source_lang) == to_iso639_1(target_lang):
            raise UnsupportedLanguagePairError(
                source_lang, target_lang, self.get_translator_name()
            )

        params = {
            **FIXED_PARAMS,
            "sl": normalize_for_google(source_lang),
            "tl": normalize_for_google(target_lang),
            "q": text,
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
        """1 回だけ HTTP を投げ、翻訳文を返す。

        例外は必ず ``from None`` で chain を切る — ``requests`` の例外文字列には
        ``q=`` を含む URL 全体、つまり**発話内容**が入っており、呼び出し側が
        ``exc_info=True`` でログを出すと ``__cause__`` 経由で漏れるため (#402 D8)。
        """
        try:
            # UA は **リクエストごとに**渡す。Session の headers を書き換えると
            # (a) 注入された Session では設定漏れが起きて #402 の障害が再発し、
            # (b) こちらが所有していないオブジェクトを恒久的に変更してしまう。
            response = self._session.get(
                ENDPOINT,
                params=params,
                timeout=self._timeout,
                headers={"User-Agent": BROWSER_UA},
            )
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

        # **status の分類より先に見る。** bot 判定は 429 で来るが、RETRYABLE_STATUS に
        # 任せると呼び出し側が sorry ページを叩き続け、判定を強める (#442)。
        # メッセージに URL を入れない — sorry の continue= に q= (発話) が入っている。
        if _is_bot_challenge(response):
            raise TranslationError(
                "Google Translate served a bot challenge (reCAPTCHA); retrying will "
                "not help. Wait a while, or use another translator (opus_mt / "
                "riva_instruct). See docs/troubleshooting/translation.md.",
                provider="google",
                reason="bot_challenge",
                status_code=status,
            ) from None

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
        result = _extract_translation(body)

        if result is None:
            # JSON として読めなかったときに限りエラーページを疑う。翻訳結果に
            # "Error 500" が含まれる可能性があるため、順序が逆だと誤判定する。
            if any(marker in body for marker in _ERROR_PAGE_MARKERS):
                raise TranslationNetworkError(
                    "Google Translate returned an error page with HTTP 200",
                    provider="google",
                    reason="embedded_error_page",
                ) from None
            raise TranslationError(
                "Google Translate response was not the expected JSON "
                "({\"sentences\": [{\"trans\": ...}]}). The endpoint contract likely "
                "changed - see docs/troubleshooting/translation.md.",
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
