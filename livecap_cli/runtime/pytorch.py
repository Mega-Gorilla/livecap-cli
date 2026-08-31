"""PyTorch ランタイムの起動時設定 (Issue #422)。

**扱う境界は 1 つ — CUDA Jiterator の kernel cache の置き場所である。**

PyTorch が Jiterator (nvrtc で生成する CUDA カーネル) のキャッシュ先を
Windows の narrow 文字列 API で扱っているため、その path が ACP (Windows ANSI
code page) の外側だと **CUDA 上の演算そのものが ``UnicodeDecodeError`` で失敗
する**。モデルは一切関係なく、``torch`` だけで再現する::

    x = torch.randn(16381, device="cuda")
    torch.fft.rfft(x).abs()      # -> UnicodeDecodeError: byte 0x83 ...

例外は**パスを一切名指ししない** (C++ 側のメッセージが ANSI で返り UTF-8 復号に
失敗している形) ので、epic #380 の言う「診断上 fail_silent」に該当する。
``cjk_kana`` (``ユーザー``) は cp932 の内側なので**再現しない** — ACP の外側で
のみ壊れるため、日本語 Windows での素朴な確認では見逃す。

決定と実装方針
--------------

キャッシュ先は ``PYTORCH_KERNEL_CACHE_PATH`` → ``%TEMP%\\torch\\kernels`` →
``%HOME%\\.cache\\torch\\kernels`` の順で決まる (``TMP`` / ``TMPDIR`` / ``USERPROFILE``
は**参照されない** — :func:`_default_cache_dir` の実測表を見よ)。
**既定では機能ごと無効化する** (``USE_PYTORCH_KERNEL_CACHE=0``)。実測 (#422 §2.1)
では **PyTorch 2.9.1 の通常の Windows 書き込み経路が cache を populate できない** —
``<name>_tmp_<pid>`` から最終名への rename が起きず、ルックアップは最終名で行われる
ため、**自分で書いたものを自分で読めない**。したがって無効化しても失われるものが無く、
「ASCII なキャッシュをどこへ永続配置するか」という設計 (TTL 回収の対象外にする /
世代管理 / 容量) を抱え込まずに破綻だけを消せる。

**明示指定は尊重する。** 外部で pre-populate されたキャッシュは実際にヒットする
(実測: 98.4 ms -> 20.2 ms) ので、``USE_PYTORCH_KERNEL_CACHE=1`` や明示 path を
黙って無効化しない。**明示された非 ASCII path と未知の値は fail loud** にする。

タイミング
----------

キャッシュ先が確定するのは **CUDA 初期化時ではなく、最初の Jiterator 実行時**である
(実測: CUDA 初期化後に env を差し替えても効いてしまう)。したがって:

- ``import torch`` より**後**でよい。engine / translator / VAD の構築時で間に合う
- 逆に、一度確定した後の変更は反映されない。しかも**確定済みかを読む公開 API が無い**ので、
  「もう手遅れか」を事後検出することはできない (``torch.cuda.is_initialized()`` は判定に
  使えない)。だから再呼び出しでは黙って再適用せず、**drift を fail loud にする**

契約
----

- **``torch`` を import しない / CUDA を初期化しない。** 環境変数を決めるだけである。
  CPU-only 環境と import コストを壊さないため、そして engine の ``__init__`` から
  呼べるようにするため
- **冪等**。2 回目以降は最初の決定を返す (drift 検査つき)
- **スレッド安全**
- **``import livecap_cli`` では自動実行しない。** ホストの ``configure_resources()``
  より先に走ると設定を横取りしかねない。本 module 自体は resources を触らないが、
  「起動時に勝手に走る初期化」を作らないという方針を守る
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple

from livecap_cli.resources.errors import AsciiPathError

logger = logging.getLogger(__name__)

__all__ = [
    "BOUNDARY",
    "ENV_USE_KERNEL_CACHE",
    "ENV_KERNEL_CACHE_PATH",
    "PyTorchRuntimeError",
    "PyTorchRuntimeDecision",
    "configure_pytorch_runtime",
    "current_pytorch_runtime",
]

#: 棚卸し表 (``tests/nonascii/registry.py``) の boundary_id と**同一文字列**である。
#: 例外メッセージとログの ``boundary=`` に必ず出す。
BOUNDARY = "framework.pytorch.cuda_jiterator_kernel_cache"

ENV_USE_KERNEL_CACHE = "USE_PYTORCH_KERNEL_CACHE"
ENV_KERNEL_CACHE_PATH = "PYTORCH_KERNEL_CACHE_PATH"

#: kernel cache の状態。
CACHE_DISABLED = "disabled"
CACHE_ENABLED = "enabled"
CACHE_NOT_APPLICABLE = "not_applicable"

#: PyTorch が populate に失敗する件 (#422 §2.1) を、有効化を選んだ利用者へ伝える文言。
_POPULATE_WARNING = (
    "PyTorch 2.9.1 on Windows does not populate this cache: it writes "
    "'<name>_tmp_<pid>' and never renames it to the final name that lookup uses, "
    "so kernels are recompiled every run and the files accumulate. Only a cache "
    "populated by other means will actually be used."
)


class PyTorchRuntimeError(AsciiPathError):
    """PyTorch ランタイムの設定を確定できない (Issue #422)。

    **``AsciiPathError`` を基底にするのは、この関数の仕事が「Jiterator 境界へ渡せる
    ASCII path を保証すること」だからである。** 送出する条件は 2 つあるが、どちらも
    「保証できなかった」に帰着する:

    1. 明示された cache path が非 ASCII / 利用不能 — そのまま path の失敗
    2. ``USE_PYTORCH_KERNEL_CACHE`` が ``0`` / ``1`` 以外 — 利用者の意図が読めず、
       しかも **PyTorch はそれを「有効」として扱う**ので、検証していない path を
       境界へ渡すことになる

    呼び出し側が ``except AsciiPathError`` で ASCII 保証の失敗をまとめて拾える状態を
    保つため、独立した family を作らない。
    """

    code = "pytorch_runtime_misconfigured"


@dataclass(frozen=True)
class PyTorchRuntimeDecision:
    """何をどう決めたか。**診断ログと snapshot に載せるための値である。**

    ``ignored`` / ``warnings`` を持つのは、「黙って上書きしない」を観測可能にする
    ため — 決定だけを返すと、運用者は自分の設定が効いたのかどうか分からない。
    """

    #: ``disabled`` / ``enabled`` / ``not_applicable``
    kernel_cache: str
    #: 有効時に PyTorch が使う path (絶対 path)。無効時と非対象 platform では None。
    #: **有効なら必ず値が入る** — 既定の置き場所を採る場合も、解決した値を
    #: ``PYTORCH_KERNEL_CACHE_PATH`` へ pin するためである (``explicit_enable`` 分岐)。
    kernel_cache_path: Optional[str]
    #: ``platform`` / ``default`` / ``explicit_disable`` / ``explicit_enable`` / ``explicit_path``
    source: str
    reason: str
    #: ``(env 変数名, なぜ無視したか)``。
    ignored: Tuple[Tuple[str, str], ...] = ()
    warnings: Tuple[str, ...] = ()
    #: **決定後の 2 変数の期待値** ``((名前, 値), ...)``。適用にも drift 検出にも使う。
    #: 値が ``None`` の項目は「設定されていないこと」を期待する。
    #:
    #: **dict にしない。** ``frozen=True`` は field の**再代入**しか止めないので、
    #: dict を持たせると ``decision.expected_env[...] = ...`` で**公開 decision 経由で
    #: drift 検査の期待値そのものを書き換えられる** (レビュー指摘)。immutable snapshot
    #: という契約に反するし、drift 検査は「誰かが環境を変えた」ことを見るための機構
    #: なので、その基準が可変では意味を成さない。``ignored`` / ``fallbacks`` と同じ
    #: tuple-of-tuples 表現に揃える。
    expected_env: Tuple[Tuple[str, Optional[str]], ...] = ()


# --- path の可用性判定 --------------------------------------------------------


def _reject_reason(path: Path) -> Optional[str]:
    """cache 先として使えない理由。使えるなら ``None``。

    順序は ``livecap_cli.paths.roots._reject_reason`` に合わせる — **filesystem を
    触る前に安い判定を済ませる**。あちらを再利用しないのは、``STAGING_ROOT_MAX_LEN``
    (``<root>\\<purpose>\\<uuid12>`` を載せる staging の予算) が本件に無関係だから
    である。PyTorch は自分でファイル名を決めるので、我々の purpose 予算は当たらない。
    """
    text = str(path)
    if not text.isascii():
        return "not ASCII"

    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        return f"cannot create: {error}"

    # **書き込み probe。** Windows の ACL 検査は当てにならないので実際に書く。
    try:
        handle, probe = tempfile.mkstemp(dir=path, prefix=".livecap-probe-")
    except OSError as error:
        return f"not writable: {error}"
    os.close(handle)
    try:
        os.unlink(probe)
    except OSError:  # pragma: no cover - 書けたのに消せないのは稀
        pass
    return None


def _default_cache_dir(environ: Mapping[str, str]) -> Optional[Path]:
    """``PYTORCH_KERNEL_CACHE_PATH`` が無いときに PyTorch が使う既定。``None`` = どこにも置かない。

    **上流の解決順を写す。** ここがずれると、**PyTorch が使わない path を検証して
    「安全」と答える**ことになり、保証が空洞になる。実測 (torch 2.9.1+cu128 /
    Windows 11 26200) で確定させた:

    ==========================================  ==========================================
    環境                                        書かれた場所
    ==========================================  ==========================================
    ``TEMP`` 未設定 / ``TMP`` = ASCII            **どこにも書かれない** (``TMP`` は見ない)
    ``TEMP`` 未設定 / ``HOME`` = ASCII           ``HOME\\.cache\\torch\\kernels``
    ``TEMP`` = ``""`` / ``HOME`` あり             ``HOME`` 側 (空文字は未設定と同じ)
    ``TEMP`` = ASCII / ``HOME`` = 非 ASCII        ``TEMP\\torch\\kernels``
    ``HOME`` 未設定 / ``USERPROFILE`` = ASCII     **どこにも書かれない** (``USERPROFILE`` も見ない)
    ==========================================  ==========================================

    したがって参照するのは **``TEMP`` と ``HOME`` の 2 つだけ**である。
    ``TMP`` / ``TMPDIR`` / ``USERPROFILE`` は**使われない**。

    ``tempfile.gettempdir()`` を使わないのは、あれが**初回参照でキャッシュ**され、
    ハーネスや呼び出し順に依存した値を返し得るためである。
    """
    temp = environ.get("TEMP")
    if temp:
        return Path(temp) / "torch" / "kernels"
    home = environ.get("HOME")
    if home:
        return Path(home) / ".cache" / "torch" / "kernels"
    return None


# --- 決定 (純関数) ------------------------------------------------------------


def _decide(environ: Mapping[str, str], platform: str) -> PyTorchRuntimeDecision:
    """環境から決定を導く。**プロセスの状態は一切書き換えない。**

    ``os.environ`` も module-level の cache も触らないので、決定表を**環境変数を
    汚さずに**網羅テストできる。適用と drift 検査は :func:`configure_pytorch_runtime`
    が持つ。

    **完全な純関数ではない**: 明示 path の可用性判定 (:func:`_reject_reason`) が
    ディレクトリ作成と書き込み probe を行う。**それが意図である** — Windows の ACL
    検査は当てにならないので、「使える」と答える前に実際に書く。
    """
    if platform != "win32":
        # ACP が無いので境界が存在しない。**何も約束しない** (drift 検査もしない)。
        return PyTorchRuntimeDecision(
            kernel_cache=CACHE_NOT_APPLICABLE,
            kernel_cache_path=None,
            source="platform",
            reason=f"{platform}: no ANSI code page, the Jiterator cache path is not narrowed",
        )

    raw_use = environ.get(ENV_USE_KERNEL_CACHE)
    raw_path = environ.get(ENV_KERNEL_CACHE_PATH)

    # --- USE の解釈 ---------------------------------------------------------
    # **PyTorch の解釈をそのまま採らない。** PyTorch は "0" 以外をすべて有効として
    # 扱うので、`false` / `no` / 空文字を書いた利用者は**無効化したつもりで有効化
    # している** (実測 #422 §2.3)。意図と実際が食い違うのに兆候がゼロなのは epic #380
    # が排除している形そのものなので、ここで断つ。
    if raw_use is not None and raw_use not in {"0", "1"}:
        raise PyTorchRuntimeError(
            f"{BOUNDARY}: {ENV_USE_KERNEL_CACHE}={raw_use!r} is not understood. "
            f"PyTorch treats every value except '0' as ENABLED, so this most likely "
            f"does the opposite of what you intended. Set '0' to disable the CUDA "
            f"Jiterator kernel cache or '1' to enable it.",
            boundary=BOUNDARY,
        )

    # --- USE=0: 無効化を明示 ------------------------------------------------
    if raw_use == "0":
        ignored: Tuple[Tuple[str, str], ...] = ()
        if raw_path is not None:
            ignored = (
                (
                    ENV_KERNEL_CACHE_PATH,
                    f"{ENV_USE_KERNEL_CACHE}=0 disables the cache entirely, so "
                    f"{ascii(raw_path)} is never used",
                ),
            )
        return PyTorchRuntimeDecision(
            kernel_cache=CACHE_DISABLED,
            kernel_cache_path=None,
            source="explicit_disable",
            reason=f"{ENV_USE_KERNEL_CACHE}=0 was set explicitly",
            ignored=ignored,
            expected_env=(
                (ENV_USE_KERNEL_CACHE, "0"),
                (ENV_KERNEL_CACHE_PATH, raw_path),
            ),
        )

    # --- 明示 path がある ---------------------------------------------------
    # **`USE` 未設定でも、明示 path 自体が opt-in である。** 置き場所をわざわざ
    # 指定した利用者を既定の無効化で黙って上書きしない。
    if raw_path is not None:
        # **空文字は「未設定」ではない。** 実測では PyTorch がこれを空のディレクトリ
        # 名として扱い、**キャッシュを黙って一切行わない** (非 ASCII な `%TEMP%` でも
        # 落ちない = 経路に入っていない)。一方 `Path("")` は `Path(".")` なので、
        # 素直に検証すると cwd を probe して「使える」と答えてしまう。
        # **設定が何もしていない**ことを利用者に伝える。
        if not raw_path.strip():
            raise PyTorchRuntimeError(
                f"{BOUNDARY}: {ENV_KERNEL_CACHE_PATH} is set but empty. PyTorch treats "
                f"that as an empty directory name and silently caches nothing, so the "
                f"setting does not do what it looks like. Set it to a writable ASCII "
                f"directory, unset it, or set {ENV_USE_KERNEL_CACHE}=0 to disable the "
                f"cache explicitly.",
                boundary=BOUNDARY,
            )

        # **絶対 path へ正規化して、その値を適用する。** 相対 path のままだと、
        # 初期化後に cwd が変わったとき**検証した場所と PyTorch が使う場所がずれる**。
        # PyTorch が cache 先を解決するのは最初の Jiterator 実行時なので、その間に
        # cwd が動く余地は実際にある。
        resolved = os.path.abspath(raw_path)
        reason = _reject_reason(Path(resolved))
        if reason is not None:
            raise PyTorchRuntimeError(
                f"{BOUNDARY}: {ENV_KERNEL_CACHE_PATH}={ascii(raw_path)} cannot be used "
                f"as the CUDA Jiterator kernel cache ({reason}). On Windows a cache "
                f"path outside the ANSI code page cannot be handed to PyTorch reliably: "
                f"it either fails with an UnicodeDecodeError that never names the path, "
                f"or silently caches nothing. Point it at a writable ASCII directory, "
                f"or set {ENV_USE_KERNEL_CACHE}=0 to disable the cache.",
                boundary=BOUNDARY,
            )
        normalized = () if resolved == raw_path else (
            f"{ENV_KERNEL_CACHE_PATH} normalised to an absolute path so that a later "
            f"working-directory change cannot move it: {ascii(raw_path)} -> {ascii(resolved)}",
        )
        return PyTorchRuntimeDecision(
            kernel_cache=CACHE_ENABLED,
            kernel_cache_path=resolved,
            source="explicit_path",
            reason=(
                f"{ENV_KERNEL_CACHE_PATH} was set explicitly and is a usable ASCII path"
                + (f" ({ENV_USE_KERNEL_CACHE}=1)" if raw_use == "1" else "")
            ),
            warnings=(_POPULATE_WARNING,) + normalized,
            expected_env=(
                (ENV_USE_KERNEL_CACHE, raw_use),
                (ENV_KERNEL_CACHE_PATH, resolved),
            ),
        )

    # --- USE=1 かつ path なし: 既定の置き場所を検証し、そこへ pin する -------
    if raw_use == "1":
        default = _default_cache_dir(environ)
        if default is None:
            raise PyTorchRuntimeError(
                f"{BOUNDARY}: {ENV_USE_KERNEL_CACHE}=1 asks for the CUDA Jiterator "
                f"kernel cache, but neither TEMP nor HOME is set, so PyTorch has "
                f"nowhere to put it and would cache nothing. Set "
                f"{ENV_KERNEL_CACHE_PATH} to a writable ASCII directory, or "
                f"{ENV_USE_KERNEL_CACHE}=0 to disable the cache explicitly.",
                boundary=BOUNDARY,
            )

        # **検証した場所を `PYTORCH_KERNEL_CACHE_PATH` へ固定する。**
        # 検証するだけでは保証にならない — PyTorch が cache 先を解決するのは最初の
        # Jiterator 実行時なので、それまでに `TEMP` / `HOME` が変われば**検証して
        # いない場所が使われる**。しかも解決の材料である `TEMP` / `HOME` は
        # `expected_env` に載らないので、drift 検査も素通りする。
        #
        # 外部コードだけの話ではない。本 repo には `ascii_safe_temp_environment()`
        # という `%TEMP%` を一時的に差し替える機構があり、その内側で最初の Jiterator
        # が走ると、**TTL 回収の対象である staging を PyTorch が static に握る**
        # (#386 型の寿命ずれ)。
        #
        # 実測 (torch 2.9.1+cu128 / RTX 4090): 確定後に `TEMP` を ACP 外へ変えると、
        # pin 無しでは `UnicodeDecodeError`、pin ありでは成功し cache は pin 先へ
        # 書かれた。**明示 path 分岐と同じ drift 保証**をこの分岐にも与える。
        resolved = os.path.abspath(str(default))
        reason = _reject_reason(Path(resolved))
        if reason is not None:
            raise PyTorchRuntimeError(
                f"{BOUNDARY}: {ENV_USE_KERNEL_CACHE}=1 asks for the CUDA Jiterator "
                f"kernel cache, but the location PyTorch would use, "
                f"{ascii(resolved)}, cannot be used ({reason}). On Windows a cache "
                f"path outside the ANSI code page cannot be handed to PyTorch reliably: "
                f"it either fails with an UnicodeDecodeError that never names the path, "
                f"or silently caches nothing. Set {ENV_KERNEL_CACHE_PATH} to a writable "
                f"ASCII directory, or {ENV_USE_KERNEL_CACHE}=0 to disable the cache.",
                boundary=BOUNDARY,
            )
        return PyTorchRuntimeDecision(
            kernel_cache=CACHE_ENABLED,
            kernel_cache_path=resolved,
            source="explicit_enable",
            reason=(
                f"{ENV_USE_KERNEL_CACHE}=1 was set explicitly and the location PyTorch "
                f"would use, {ascii(resolved)}, is usable"
            ),
            warnings=(
                _POPULATE_WARNING,
                f"{ENV_KERNEL_CACHE_PATH} was not set, so it has been pinned to the "
                f"location PyTorch would have used anyway ({ascii(resolved)}). Without "
                f"the pin a later TEMP/HOME change would silently move the cache to a "
                f"path that was never validated, because PyTorch resolves it at the "
                f"first Jiterator call rather than now.",
            ),
            expected_env=(
                (ENV_USE_KERNEL_CACHE, "1"),
                (ENV_KERNEL_CACHE_PATH, resolved),
            ),
        )

    # --- 既定: 明示が何も無い -----------------------------------------------
    # **境界そのものを通さない。** #422 §2.1 のとおり、この cache は Windows では
    # populate されないので、無効化しても失われるものが無い。
    return PyTorchRuntimeDecision(
        kernel_cache=CACHE_DISABLED,
        kernel_cache_path=None,
        source="default",
        reason=(
            "no explicit setting: disabled so that a non-ASCII cache path cannot break "
            "CUDA Jiterator operations. The cache is not populated on Windows anyway "
            f"(see {BOUNDARY})"
        ),
        expected_env=(
            (ENV_USE_KERNEL_CACHE, "0"),
            (ENV_KERNEL_CACHE_PATH, None),
        ),
    )


# --- 適用 ---------------------------------------------------------------------

_lock = threading.Lock()
_decision: Optional[PyTorchRuntimeDecision] = None


def _apply(expected: Tuple[Tuple[str, Optional[str]], ...]) -> None:
    for name, value in expected:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _drift(expected: Tuple[Tuple[str, Optional[str]], ...]) -> list[str]:
    return [
        f"{name}: expected {value!r}, found {os.environ.get(name)!r}"
        for name, value in expected
        if os.environ.get(name) != value
    ]


def configure_pytorch_runtime() -> PyTorchRuntimeDecision:
    """PyTorch の起動時設定を確定して適用する。**冪等・スレッド安全。**

    ``torch`` は import しない。CUDA も初期化しない。環境変数を決めるだけである。

    Returns:
        この プロセスで確定した :class:`PyTorchRuntimeDecision`。

    Raises:
        PyTorchRuntimeError: 設定が矛盾している / 明示された cache path が使えない /
            **確定後に環境変数が書き換えられた** (下記)。

    Note:
        **再呼び出しでは、黙って再適用せず drift を送出する。** キャッシュ先が確定
        するのは最初の Jiterator 実行時で、**確定済みかを読む公開 API が PyTorch に
        無い**。したがって再適用が効いたかどうかを保証できず、黙って上書きすると
        「直したつもり」のログだけが残る。誰が何を壊したかを見せる方がよい。
    """
    global _decision
    with _lock:
        if _decision is None:
            decision = _decide(os.environ, sys.platform)
            _apply(decision.expected_env)
            _decision = decision
            _log(decision)
            return decision

        if _decision.kernel_cache == CACHE_NOT_APPLICABLE:
            # 何も設定していないので、守るべき状態が無い。
            return _decision

        drift = _drift(_decision.expected_env)
        if drift:
            raise PyTorchRuntimeError(
                f"{BOUNDARY}: the PyTorch runtime environment changed after it was "
                f"configured ({'; '.join(drift)}). Re-applying it here would not be "
                f"honest: PyTorch resolves the kernel cache directory once, at the "
                f"first Jiterator call, and offers no way to tell whether that already "
                f"happened. Set these variables before the first engine, translator or "
                f"VAD is constructed.",
                boundary=BOUNDARY,
            )
        return _decision


def current_pytorch_runtime() -> Optional[PyTorchRuntimeDecision]:
    """このプロセスで確定済みの決定。まだ設定していなければ ``None``。

    **読むだけで、設定はしない。** 診断 (``info`` 出力やテスト) が「どう決まったか」を
    確認するためのもので、ここが ``None`` を返すなら「まだ誰も初期化していない」という
    事実そのものが答えである — :func:`configure_pytorch_runtime` を呼んでしまうと
    その事実が消える。
    """
    with _lock:
        return _decision


def _log(decision: PyTorchRuntimeDecision) -> None:
    """決定を 1 行で残す。

    **解決値を出す** (``raw args`` ではなく)。運用者が知りたいのは「自分の設定が
    どう解釈されたか」であり、無視された設定こそ観測できなければならない。
    """
    logger.info(
        "PyTorch runtime configured: boundary=%s kernel_cache=%s path=%s source=%s (%s)",
        BOUNDARY,
        decision.kernel_cache,
        ascii(decision.kernel_cache_path) if decision.kernel_cache_path else "-",
        decision.source,
        decision.reason,
    )
    for name, why in decision.ignored:
        logger.warning("PyTorch runtime: ignoring %s - %s", name, why)
    for warning in decision.warnings:
        logger.warning("PyTorch runtime: %s", warning)


def _reset_pytorch_runtime_for_tests() -> None:
    """確定済みの決定を捨てる。**テスト専用。**

    ``monkeypatch.setenv`` の効果を反映させたい場合に使う。production 用の再設定
    手段は用意しない — 確定は 1 プロセス 1 回という契約そのものを弱めるため。
    """
    global _decision
    with _lock:
        _decision = None
