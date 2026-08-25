"""Resource 層の例外 (Issue #375)。

``configuration`` と、PR 2 で入る staging module の**両方**が送出するため、
どちらでもない独立した module に置く。片方に置くともう片方が import すること
になり、PR 2 で循環 import になる。
"""
from __future__ import annotations

from typing import Sequence, Tuple

__all__ = [
    "ResourceConfigurationError",
    "AsciiPathError",
    "AsciiStagingUnavailableError",
]


class ResourceConfigurationError(RuntimeError):
    """resource configuration が確定できない / 矛盾している。

    R2 (Issue #375) により、**明示された入力が使えないときは候補 ladder へ黙って
    落ちない**。ホストが渡した root や運用者が設定した env が使えないことは、
    「別の場所を勝手に使ってよい」という意味ではないため。
    """


class AsciiPathError(RuntimeError):
    """ネイティブ境界へ渡せる ASCII path を提供できなかった (#378 §6.8)。

    ``livecap_cli.paths`` の全失敗がこれを基底に持つので、呼び出し側は
    ``except AsciiPathError`` で ASCII 保証の失敗だけをまとめて拾える。

    **``OSError`` 派生にしない。** 呼び出し側は I/O の失敗を ``except OSError`` で
    握り潰すことがあり、そこへ紛れ込むと「ASCII を保証できなかった」が黙って
    握られる — epic #380 が排除している silent degradation そのものになる。
    """

    code = "ascii_path_error"

    def __init__(self, message: str, *, boundary: str | None = None) -> None:
        super().__init__(message)
        self.boundary = boundary


class AsciiStagingUnavailableError(ResourceConfigurationError, AsciiPathError):
    """明示された staging root が staging 用の述語を満たさない。

    述語は **ASCII / 長さ / 作成・書き込み可能**の 3 つ
    (:func:`livecap_cli.resources.configuration.validate_staging_root`)。

    **基底が 2 つあるのは、この失敗が実際に両方だから。** ``configure_resources()``
    が明示指定を弾くときは configuration の失敗であり、境界呼び出し時に候補が
    全滅したときは ASCII path の失敗である。**同じ条件に 2 つのクラスを作ると、
    捕捉側がどちらを掴めばよいか分からなくなる** (実際、PR 2 の実装中に
    ``resources`` 側と ``paths`` 側へ別々に定義してしまい、実地確認で発覚した)。

    Attributes:
        attempts: ``(候補の説明, なぜ駄目だったか)`` の並び。**何を試したかを構造化
            して持つ**もので、呼び出し側がログや UI で並べ直せる。

    Note:
        判定の分担: **明示指定 (API ``staging_root`` /
        ``LIVECAP_CORE_ASCII_STAGING_DIR``) は ``configure_resources()`` が freeze 時に**
        弾く。候補 ladder は ``livecap_cli.paths.roots`` の責務で、そちらは ladder を
        降りてよい — 降りないのは明示指定の場合のみ。
    """

    code = "ascii_staging_unavailable"

    def __init__(
        self,
        message: str,
        *,
        boundary: str | None = None,
        attempts: "Sequence[Tuple[str, str]]" = (),
    ) -> None:
        super().__init__(message)
        self.boundary = boundary
        self.attempts: "Tuple[Tuple[str, str], ...]" = tuple(attempts)
