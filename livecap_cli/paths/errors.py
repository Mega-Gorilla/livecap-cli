"""ASCII path 保証まわりの例外 (Issue #375 PR 2、契約は #378 §6.8)。

**正本は :mod:`livecap_cli.resources.errors` にある。** ここで再定義しない —
同じ条件に 2 つのクラスがあると、捕捉側がどちらを掴めばよいか分からなくなる
(PR 2 の実装中に実際にやってしまい、実地確認で発覚した)。

置き場が ``resources`` 側なのは import 方向のため: ``paths`` -> ``resources`` は
成立するが逆は循環する。``configure_resources()`` が明示指定を freeze 時に弾く
時点で同じ例外が要るので、下側に置く必要がある。

メッセージ契約 (この順で必須):

1. **境界名** — だから ``boundary`` は必須キーワード引数である
2. **問題の path**
3. **何を試して各々なぜ失敗したか** (``errno`` / ``winerror`` 付き)
4. **env var を名指しした実行可能な対処**

``code`` は i18n フック用の安定識別子。メッセージ本文は英語にする。
"""
from __future__ import annotations

from typing import Optional

from livecap_cli.resources.errors import AsciiPathError, AsciiStagingUnavailableError

__all__ = [
    "AsciiPathError",
    "AsciiStagingUnavailableError",
    "TempEnvironmentConflictError",
]


class TempEnvironmentConflictError(AsciiPathError):
    """別 purpose の temp environment スコープが既に開いている。

    ``TEMP`` / ``TMP`` / ``TMPDIR`` / ``tempfile.tempdir`` は**プロセス全体**の状態
    なので、異なる行き先を同時に要求されたら**どちらかに嘘をつくことになる**。
    yield した path と ``tempfile`` が実際に書く場所がずれるより、送出する方がよい。
    """

    code = "temp_environment_conflict"

    def __init__(self, message: str, *, boundary: Optional[str] = None) -> None:
        super().__init__(message, boundary=boundary)
