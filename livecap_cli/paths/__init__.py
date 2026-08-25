"""ネイティブ境界へ渡す path の ASCII 保証 (Issue #375 PR 2、契約は #378 §6)。

いつ使うのか
----------
**ネイティブライブラリが narrow path で path を扱う境界だけ**に使う。
次の場合は使ってはいけない (#378 §6.1):

- ``*_buf`` / ``*_bytes`` / serialized-proto / file-object 版の API がある (= 方式①)
- CPython 経由のみで到達する (``open`` / ``pathlib`` / ``shutil`` / ``tarfile`` /
  ``json``)。実測で ``tarfile.extractall`` / ``urlretrieve`` / ``huggingface_hub``
  はすべて非 ASCII でも通っている (= 方式②)

**② で足りる境界に ③ を持ち込まないこと。**

2 つの API とその使い分け
-----------------------
====================================  ==========================================
用途                                  API
====================================  ==========================================
ネイティブが**自前で** ``%TEMP%`` へ  :func:`ascii_safe_temp_environment`
展開する (NeMo の untar 等)
**我々が**ファイルを作る              :func:`ascii_safe_workspace`
(発話ごとの wav 等)
====================================  ==========================================

非対称が 1 つある — **前者は退出時にディレクトリを消さず、後者は消す**。
プロセス全体の TEMP を向けている間は無関係なスレッドのファイルもそこへ落ちるため
(#386 のデータ消失そのもの)。詳細は各 module の docstring を参照。

既存のツリーを ASCII 領域へ staging する ``ascii_safe_path()`` は**まだ実装していない**。
設計は #378 §6 に確定しているが、**現時点で必要とする境界が無い** — 唯一の候補だった
sherpa-onnx は 1.13.6 への version bump で ②wide-path になった (#377)。消費者が
現れた時点で実装する。

明示的な非保証
------------
- **fork 安全ではない。** 子プロセスは :func:`reset_staging_root_cache` を呼ぶこと
- **ブロッキング**する。event loop スレッドから呼ばないこと
  (``asyncio.to_thread()`` を使う)
- 無関係な境界を直列化しない (グローバルなモデルロードロックではない)
- 消費側ライブラリのスレッド安全性については何も言わない
"""
from __future__ import annotations

from .errors import (
    AsciiPathError,
    AsciiStagingUnavailableError,
    TempEnvironmentConflictError,
)
from .reaper import DEFAULT_TTL_HOURS, reap_staging_root
from .roots import is_ascii_safe, reset_staging_root_cache, select_staging_root
from .temp_env import ascii_safe_temp_environment
from .workspace import ascii_safe_workspace

__all__ = [
    # 境界向け API
    "ascii_safe_temp_environment",
    "ascii_safe_workspace",
    # root 選定
    "select_staging_root",
    "is_ascii_safe",
    "reset_staging_root_cache",
    # 回収
    "reap_staging_root",
    "DEFAULT_TTL_HOURS",
    # 例外
    "AsciiPathError",
    "AsciiStagingUnavailableError",
    "TempEnvironmentConflictError",
]
