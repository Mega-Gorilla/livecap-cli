"""Resource 層の例外 (Issue #375)。

``configuration`` と、PR 2 で入る staging module の**両方**が送出するため、
どちらでもない独立した module に置く。片方に置くともう片方が import すること
になり、PR 2 で循環 import になる。
"""
from __future__ import annotations

__all__ = ["ResourceConfigurationError", "AsciiStagingUnavailableError"]


class ResourceConfigurationError(RuntimeError):
    """resource configuration が確定できない / 矛盾している。

    R2 (Issue #375) により、**明示された入力が使えないときは候補 ladder へ黙って
    落ちない**。ホストが渡した root や運用者が設定した env が使えないことは、
    「別の場所を勝手に使ってよい」という意味ではないため。
    """


class AsciiStagingUnavailableError(ResourceConfigurationError):
    """明示された staging root が staging 用の述語を満たさない。

    述語は **ASCII / 長さ / 作成・書き込み可能**の 3 つ
    (:func:`livecap_cli.resources.configuration.validate_staging_root`)。

    Note:
        PR 1 が判定するのは **明示指定 (API ``staging_root`` /
        ``LIVECAP_CORE_ASCII_STAGING_DIR``) だけ**である。候補 ladder は staging
        core (PR 2) の責務で、そちらは ladder を降りてよい — 降りないのは明示
        指定の場合のみ。
    """
