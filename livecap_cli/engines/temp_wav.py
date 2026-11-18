"""発話ごとの一時 wav の後始末 (Issue #419)。

engine は発話ごとに一時 wav を作り、ネイティブ ASR へ path で渡す。その寿命の規則は
次の 3 つで、engine 側は **名前の確保 → ``try`` の中で書き込みと consumer 呼び出し →
``finally`` で** :func:`discard_temp_wav` の順に並べる。

- **名前は呼び出しごとに一意にする** (``tempfile.NamedTemporaryFile(delete=False)``)。
  pid 固定名は同一プロセスの並行呼び出しで衝突し、一方の cleanup が他方の入力を消す。
- **書き込みも cleanup 対象の ``try`` の内側で行う。** ``NamedTemporaryFile`` は名前を
  確保した時点でファイルを作るので、``try`` の外で書き込みが失敗するとファイルが残る。
- **後始末の失敗で本来の例外を上書きしない** (このモジュールの役目)。

置き場所 (``dir=``) は engine ごとに違い、非 ASCII path の棚卸し表
(``tests/nonascii/registry.py`` の ``engine.*.utterance_wav``) が engine ごとの
callsite で追跡している。そのため名前の確保は engine 側に残し、ここでは扱わない。
**このモジュールは ASCII path を保証しない** (それは ``livecap_cli.paths`` の役目)。
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

__all__ = ["discard_temp_wav"]


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
        logger.warning("Could not remove temporary wav %s: %s", path, exc)
