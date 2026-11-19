"""発話ごとの一時 wav の確保と後始末 (Issue #419)。

engine は発話ごとに一時 wav を作り、ネイティブ ASR へ path で渡す。その寿命の規則は
次の 3 つで、engine 側は **:func:`reserve_temp_wav` で確保 → ``try`` の中で書き込みと
consumer 呼び出し → ``finally`` で :func:`discard_temp_wav`** の順に並べる。

- **名前は呼び出しごとに一意にする。** pid 固定名は同一プロセスの並行呼び出しで衝突し、
  一方の cleanup が他方の入力を消す。
- **書き込みも cleanup 対象の ``try`` の内側で行う。** 確保した時点でファイルは作られて
  いるので、``try`` の外で書き込みが失敗するとファイルが残る。
- **後始末の失敗で本来の例外を上書きしない。**

置き場所は engine ごとに違い、非 ASCII path の棚卸し表 (``tests/nonascii/registry.py`` の
``engine.*.utterance_wav``) が engine ごとの ``reserve_temp_wav(<置き場所>...)`` の
callsite で追跡している。そのため置き場所は引数で受け取り、ここでは決めない。
**このモジュールは ASCII path を保証しない** (それは ``livecap_cli.paths`` の役目)。
"""

from __future__ import annotations

import errno
import logging
import os
import secrets

logger = logging.getLogger(__name__)

__all__ = ["RESERVE_ATTEMPTS", "discard_temp_wav", "reserve_temp_wav"]

#: 名前の衝突 (``FileExistsError``) のときに別の名前を試す回数の上限。
#:
#: 名前は 64 bit の乱数なので、正常な filesystem で衝突が続くことは無い。上限は、
#: ``O_EXCL`` がいつも ``EEXIST`` を返すような異常な filesystem で返らなくならないため。
RESERVE_ATTEMPTS = 8

_FLAGS = os.O_CREAT | os.O_EXCL | os.O_WRONLY


def reserve_temp_wav(directory: str | os.PathLike[str], *, prefix: str = "tmp") -> str:
    """``directory`` に一意な名前の空ファイルを排他的に作り、その path を返す。

    名前は ``<prefix><16 桁の 16 進>.wav``。handle は閉じて返すので、呼び出し側は
    その path へ書き込み、``finally`` で :func:`discard_temp_wav` に渡す。

    ``tempfile.NamedTemporaryFile`` / ``mkstemp`` を使わない理由: CPython の
    ``tempfile._mkstemp_inner`` は Windows で ``PermissionError`` を受けても、
    ``os.access(dir, W_OK)`` (ACL を見ない) が真なら名前の衝突とみなして再試行を続ける。
    上限の ``TMP_MAX`` は約 21 億回なので、**ACL で作成を拒否されたディレクトリでは返らない**
    (書き込み probe の #468 と同じ原因)。ここでは ``os.open(O_CREAT | O_EXCL)`` を直接呼び、
    再試行するのは ``FileExistsError`` (名前の衝突) のときだけにする。

    Raises:
        OSError: 作れないとき。``PermissionError`` など ``FileExistsError`` 以外は 1 回目で
            そのまま送出する。``FileExistsError`` が :data:`RESERVE_ATTEMPTS` 回続いたときは
            ``FileExistsError`` を送出する。
    """
    base = os.fspath(directory)
    for _ in range(RESERVE_ATTEMPTS):
        path = os.path.join(base, f"{prefix}{secrets.token_hex(8)}.wav")
        try:
            fd = os.open(path, _FLAGS, 0o600)
        except FileExistsError:
            continue
        os.close(fd)
        return path
    raise FileExistsError(
        errno.EEXIST,
        f"could not reserve a unique temporary wav name in {RESERVE_ATTEMPTS} attempts",
        base,
    )


def discard_temp_wav(path: str | os.PathLike[str]) -> None:
    """一時 wav を削除する。**削除できなくても例外を投げない。**

    ``finally`` から呼ぶ前提である。ここで ``OSError`` を投げると、consumer の例外
    (呼び出し側が本当に知るべき失敗原因) がその ``OSError`` に置き換わり、正常終了なら
    転写結果が失われる。削除できなかった事実は WARNING に残す。

    既に無い (``FileNotFoundError``) のは正常扱いにする。
    """
    try:
        os.unlink(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        # exc の文字列は path を含むことが多いので、理由 (strerror) だけを添える
        logger.warning("Could not remove temporary wav %s: %s", path, exc.strerror or exc)
