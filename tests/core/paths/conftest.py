"""`tests/core/paths` 共通のヘルパ (Issue #375 PR 2)。

**ボリューム root をテストに直書きしない。** ``"D:"`` は Windows でこそドライブ
レターだが、POSIX の ``splitdrive`` はドライブを認識しないので**ただの相対
ディレクトリ名**であり、:func:`~livecap_cli.paths.roots.validate_source_volume`
が正しく拒否する (受理すると cwd 依存の staging 先という、まさに直した欠陥を
POSIX 側に作ることになる)。

実際 ``"D:"`` を直書きしたテストは Windows でだけ緑になり、Linux CI で落ちた。
プラットフォーム差が出る値は、テスト側で 1 箇所にまとめて吸収する。
"""

from __future__ import annotations

import os

import pytest


def _a_volume_root() -> str:
    """そのプラットフォームで**ボリューム root として妥当な**値。"""
    return "D:" if os.name == "nt" else "/"


def _another_volume_root() -> str:
    """:func:`a_volume_root` とは**別の**ボリューム root。

    「同じ root へ降りた 2 つの staging 元をどちらも観測できる」ことを確かめる用。
    存在も書き込み可能性も要らない — 検証は絶対 path であることだけを見るし、
    使う側のテストは source volume 候補を強制的に reject する。
    """
    return "E:" if os.name == "nt" else "/mnt"


@pytest.fixture
def a_volume_root() -> str:
    """そのプラットフォームで妥当なボリューム root。"""
    return _a_volume_root()


@pytest.fixture
def another_volume_root() -> str:
    """``a_volume_root`` とは別のボリューム root。"""
    return _another_volume_root()
