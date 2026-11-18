"""tests/core/cli 共通の fixture。"""

from __future__ import annotations

import io
import sys

import pytest


class RedirectedStdout:
    """日本語 Windows で、パイプ / ファイルへリダイレクトされた stdout (#385)。

    CPython はこのとき stdout を locale のエンコーダ (cp932) + ``surrogateescape`` で
    開く (Windows 11 / Python 3.11 / ACP=932 で実測)。``newline=None`` は実物と同じく
    ``\\n`` を ``os.linesep`` へ変換するので、``-o`` が書くファイルとバイト単位で比べられる。

    **``with`` の中でだけ ``sys.stdout`` を差し替える。** fixture の setup で差し替えても、
    pytest の capture が call フェーズの開始時に ``sys.stdout`` を自分のものへ戻してしまう。
    """

    def __init__(self) -> None:
        self.buffer = io.BytesIO()
        self.stream = io.TextIOWrapper(
            self.buffer, encoding="cp932", errors="surrogateescape", newline=None
        )
        self._saved: list = []

    def __enter__(self) -> RedirectedStdout:
        self._saved.append(sys.stdout)
        sys.stdout = self.stream
        return self

    def __exit__(self, *exc) -> bool:
        sys.stdout = self._saved.pop()
        return False

    def written(self) -> bytes:
        self.stream.flush()
        return self.buffer.getvalue()


@pytest.fixture
def cp932_redirected_stdout(monkeypatch) -> RedirectedStdout:
    """:class:`RedirectedStdout` を返す。``with`` で囲んだ間だけ stdout になる。

    ``PYTHONIOENCODING`` は CLI の判断を変える (明示されていれば従う) ので、ここで外す。
    """
    monkeypatch.delenv("PYTHONIOENCODING", raising=False)
    return RedirectedStdout()
