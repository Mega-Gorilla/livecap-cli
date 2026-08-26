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
- **支えるのはスコープ内で完了する同期境界だけ。** 各 entry は所有権マーカー兼 lease
  (:mod:`livecap_cli.paths.lease`) で保護されるが、Python のハンドルは既定で非継承
  (PEP 446) なので、**親のスコープより長生きする子プロセスは保護されない**。
  context の中で spawn した子は、抜ける前に終了 / join させること
- **``fork()`` は支えない。復旧手段も用意しない。** 「子で reset を呼べば安全になる」
  とは言えないためである。fork した子が引き継ぐ壊れた状態は 1 つではない:

  - :mod:`~livecap_cli.paths.temp_env` の ``RLock`` — 親で**別スレッドが保持したまま**
    fork すると、子では解放する主体が存在せず**デッドロックする**
  - 同 module の深度カウンタ — 子がスコープを抜けたと判断して ``%TEMP%`` を復元する
  - lease の file descriptor — 親子が**同じ open file description** を共有するので、
    子が閉じると**親の lease が外れる**
  - :mod:`~livecap_cli.paths.roots` の選定キャッシュ、:mod:`~livecap_cli.paths.reaper`
    の「root ごとに 1 回」記録、freeze 済みの resource configuration

  これらを一括で戻す API を用意しても**使う consumer が居ない**ので作らない。
  マルチプロセスが要るなら ``spawn`` を使うか、本 API を親でだけ使うこと
- **ブロッキング**する。event loop スレッドから呼ばないこと。async から使うときは
  **context の enter・境界処理・exit を同じ同期関数にまとめ、その関数全体を 1 回の
  ``asyncio.to_thread()`` で実行する**。**enter と exit を別々の呼び出しへ分割しない** —
  別の worker スレッドで走り得るが、``RLock`` はスレッド所有権を持つので取得した
  スレッド以外からは解放できない (詳細は ``docs/reference/api.md`` の「async から使う」)
- **``ascii_safe_temp_environment()`` はプロセス内で 1 つずつしか動かない。** ``TEMP`` が
  プロセス全体の状態なので、排他をスコープの全期間保持する — **別スレッドの呼び出しは
  ``boundary`` / ``purpose`` に関係なく直列化される** (待たされるのであって例外にはならない)。
  ``ascii_safe_workspace()`` は env を触らないので直列化されない。**スコープの外にある
  モデルロードや推論も直列化しない**ので、ウィンドウは最小にすること
- 消費側ライブラリのスレッド安全性については何も言わない
"""
from __future__ import annotations

from .errors import (
    AsciiPathError,
    AsciiStagingUnavailableError,
    TempEnvironmentConflictError,
)
from .temp_env import ascii_safe_temp_environment
from .workspace import ascii_safe_workspace

#: **公開面は境界 API 2 つと例外だけ。**
#:
#: root 選定 (:mod:`~livecap_cli.paths.roots`) と回収 (:mod:`~livecap_cli.paths.reaper`)
#: は**本 package の内部実装**である。ホストが選ばれた root を知りたい場合は
#: ``get_resource_configuration().staging_roots`` を読むこと — selector を直接呼ぶと
#: **configuration を freeze する副作用**があり、readback にはその副作用が無い。
#:
#: 以前は selector・回収・test 専用の reset まで top-level に出していたが、**production
#: consumer が 1 つも無い**まま公開面を広げていた (`docs/architecture/core-api-spec.md`
#: §3.4 が公開 API として挙げているのも下の 2 つだけである)。必要な内部 module は
#: ``from livecap_cli.paths import roots`` のように明示的に import する。
__all__ = [
    # 境界向け API
    "ascii_safe_temp_environment",
    "ascii_safe_workspace",
    # 例外
    "AsciiPathError",
    "AsciiStagingUnavailableError",
    "TempEnvironmentConflictError",
]
