"""GPU が idle になった後に、PyTorch allocator が握り続ける CUDA メモリを返す (Issue #462)。

**扱う境界は 1 つ — cuBLAS workspace が allocator segment を生存させることである。**

PyTorch は cuBLAS / cuBLASLt の workspace を **process 内の static map** に持ち、handle
ごとに 1 度確保したら以後解放しない。既定サイズは非 Hopper GPU で ``:4096:2:16:8``
(``2 * 4096 KiB + 8 * 16 KiB`` = **8,519,680 bytes = 8.125 MiB**) である。

この 8 MiB は小さいが、**それが乗っている allocator segment 全体を返せなくする**。実測
(RTX 4090 / PyTorch 2.9.1+cu128)::

    # 単純な matmul を 1 回 → del → gc.collect() → torch.cuda.empty_cache()
    allocated =   8,519,680    reserved =  20,971,520

    # Riva-Translate-4B を load → translate 1 回 → cleanup() → gc → empty_cache()
    allocated =   8,519,680    reserved = 8,359,247,872   (約 7.785 GiB)

つまり「7.8 GiB の cuBLAS workspace が残る」のではなく、**8.125 MiB の生存 workspace が
7.8 GiB の reservation を解放不能にしている**。モデル参照のリークではない — Riva を使わない
matmul だけでも同じ 8,519,680 bytes が残る。

なぜ translator の ``cleanup()`` でやらないか
---------------------------------------------

workspace map は **process 全体で 1 つ**なので、clear は「その translator の分だけ」では
ありえない。一方で:

- ``StreamTranscriber`` は翻訳 worker を drain してから caller が translator を cleanup する
  契約だが (#402)、**ASR 用 executor は ``shutdown(wait=False)``** で閉じる。translator 単体
  では「この process の CUDA work が全部止まった」ことを証明できない
- ホスト (livecap-gui) は複数 source / engine を同一 process で持ち得る

実行中の CUDA work の裏で workspace を消す安全性は保証できないので、**GPU が idle だと
知っている process owner が明示的に呼ぶ** API にする。``RivaInstructTranslator.cleanup()``
などの自動経路からは呼ばない (``tests/core/runtime/test_call_sites.py`` の audit で固定)。

契約
----

- **呼び出し側の前提**: application が持つ CUDA の推論 / 翻訳がすべて終了し、以後の
  enqueue も止まっていること。live CUDA graph が workspace の address を握っていないこと。
  **実行中に呼んではならない** (別スレッドの cuBLAS 呼び出しと競合する)
- 手順は ``gc.collect()`` → ``torch.cuda.synchronize()`` → cuBLAS workspace clear →
  ``torch.cuda.empty_cache()`` の順に固定する
- **``synchronize()`` に失敗したら、その場で手順を中断する (fail closed)。** 同期が成立して
  いない = 「走っている kernel が無い」ことを確認できていない状態であり、そこで process 全体の
  workspace を消すのは、この module が自分で書いている前提 (clear の前に必ず同期する) を破る。
  sticky な CUDA error や device の異常で driver 状態が不明なときに private API を叩かない
- **冪等**。2 回目以降も安全で、2 回目は解放量 0 の同じ構造を返す
- CUDA が無い / まだ初期化されていない場合は **no-op** (CUDA context を新たに作らない)
- private API が無い / 失敗した場合も **silent success にしない** — warning を出し、
  :class:`CudaMemoryRelease` の ``cublas_cleared`` / ``reason`` で呼び出し側へ返す
- 初期 scope は**単一 GPU** (現在の device)。multi-GPU 対応は名乗らない

private API に依存する範囲
--------------------------

PyTorch 2.9.1 に公開 API は無い (`pytorch#184084 <https://github.com/pytorch/pytorch/issues/184084>`_
で ``torch.cuda.clear_cublas_workspaces()`` / ``empty_cache(include_cublas_workspaces=True)``
が提案されている段階)。実在するのは ``torch._C._cuda_clearCublasWorkspaces`` だけで、
``torch.cuda._clear_cublas_workspaces`` は**存在しない** (実測)。

そこで **名前を決め打ちで呼ばず capability detection** する
(:data:`_CUBLAS_CLEAR_CANDIDATES`)。候補には**実在が確認できたものだけ**を入れる —
未確定の名前を先回りで入れても上流が別名を採れば拾えず、同名の private wrapper が現れたときに
互換性未確認のまま優先してしまう。公開 API が出たら、その semantics を見て adapter を足す。
見つからない版では ``cublas_cleared=False`` + ``reason='cublas-api-missing'`` を返し、
**完全解放できたと偽らない**。
"""

from __future__ import annotations

import gc
import logging
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["CudaMemoryRelease", "release_idle_cuda_memory"]

#: cuBLAS workspace を clear する候補 ``(表示名, 取得関数)``。先頭から探し、最初に見つかった
#: ものを使う。
#:
#: **実在が確認できたものだけを並べる。** 「将来こういう名前になるはず」を先回りで入れると、
#: (a) 上流が実際に採る名前と違えば拾えず、(b) 同名の private wrapper が現れたときに
#: **互換性を確認しないまま優先してしまう** (PR #465 レビュー MEDIUM)。
#:
#: 上流の公開 API は `pytorch#184084 <https://github.com/pytorch/pytorch/issues/184084>`_ で
#: ``torch.cuda.clear_cublas_workspaces()`` または ``empty_cache(include_cublas_workspaces=True)``
#: が提案されている段階である。**採用された版が出たら、その semantics を確認した上で**
#: ここへ adapter を足すこと (``empty_cache`` 型なら手順 4 と統合する必要がある)。
#: それまでは「公開 API が無いので private を使っている」ことを結果の ``cublas_api`` で見せる。
_CUBLAS_CLEAR_CANDIDATES: Tuple[Tuple[str, Callable[[Any], Any]], ...] = (
    # PyTorch 2.9.1 で実在する private binding (実測。`torch.cuda._clear_cublas_workspaces` は無い)
    ("torch._C._cuda_clearCublasWorkspaces", lambda torch: torch._C._cuda_clearCublasWorkspaces),
)

#: ``reason`` に入る値。呼び出し側が分岐できるよう**文字列を固定**する
REASON_RELEASED = "released"
REASON_TORCH_MISSING = "torch-missing"
REASON_CUDA_UNAVAILABLE = "cuda-unavailable"
REASON_CUDA_NOT_INITIALIZED = "cuda-not-initialized"
REASON_CUDA_SYNCHRONIZE_FAILED = "cuda-synchronize-failed"
REASON_CUBLAS_API_MISSING = "cublas-api-missing"
REASON_CUBLAS_API_FAILED = "cublas-api-failed"


@dataclass(frozen=True)
class CudaMemoryRelease:
    """:func:`release_idle_cuda_memory` が何をどこまでできたか。

    **「完全に解放できた」と偽らないための構造体である** — private API が無い環境では
    ``cublas_cleared=False`` のまま ``empty_cache()`` だけが走るので、呼び出し側 (GUI /
    CLI) が「解放済み」と表示してしまわないよう、実施可否と前後の実測値を返す。
    """

    #: 解放手順を実行したか (CUDA が使えて初期化済みだったか)
    attempted: bool
    #: cuBLAS workspace を実際に clear できたか
    cublas_cleared: bool
    #: clear に使った API 名 (:data:`_CUBLAS_CLEAR_CANDIDATES` の表示名)。使えなければ ``None``
    cublas_api: Optional[str]
    #: ``torch.cuda.memory_allocated()`` の前後 (bytes)
    allocated_before: int
    allocated_after: int
    #: ``torch.cuda.memory_reserved()`` の前後 (bytes)
    reserved_before: int
    reserved_after: int
    #: ``released`` / ``torch-missing`` / ``cuda-unavailable`` / ``cuda-not-initialized`` /
    #: ``cuda-synchronize-failed`` / ``cublas-api-missing`` / ``cublas-api-failed``
    reason: str
    #: 人間向けの補足 (private API が無い、synchronize に失敗した、など)
    warnings: Tuple[str, ...] = ()

    @property
    def released_bytes(self) -> int:
        """allocator が OS へ返した量 (``reserved`` の減少分)。増えていれば 0。"""
        return max(0, self.reserved_before - self.reserved_after)

    def to_dict(self) -> dict:
        """診断ログ / JSON 出力向けの dict (``released_bytes`` も含む)。"""
        payload = asdict(self)
        payload["warnings"] = list(self.warnings)
        payload["released_bytes"] = self.released_bytes
        return payload


def _import_torch():
    """``torch`` を返す。入っていなければ ``None``。

    **module 直下では import しない** — CPU-only 環境と import コストを壊さないため
    (``runtime.pytorch`` と同じ方針)。テストはここを差し替える。
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch は engines-torch extra
        return None
    return torch


def _memory(reader: Callable[[], int], fallback: int) -> int:
    """allocator の統計を読む。driver が異常なときでも中断理由の報告を潰さないための保険。"""
    try:
        return int(reader())
    except Exception:  # noqa: BLE001 - 統計が読めないことは報告の失敗理由にしない
        return fallback


def _resolve_cublas_clear(torch) -> Tuple[Optional[str], Optional[Callable[[], None]]]:
    """使える cuBLAS workspace clear を ``(表示名, 呼び出し可能)`` で返す。無ければ ``(None, None)``。"""
    for name, getter in _CUBLAS_CLEAR_CANDIDATES:
        try:
            func = getter(torch)
        except AttributeError:
            continue
        if callable(func):
            return name, func
    return None, None


def release_idle_cuda_memory() -> CudaMemoryRelease:
    """**GPU が idle な状態で**、PyTorch が握っている CUDA メモリを可能な限り返す (Issue #462)。

    ``torch.cuda.empty_cache()`` だけでは、生存している cuBLAS workspace (既定 8.125 MiB) が
    乗った allocator segment を返せない。Riva-Translate-4B では約 7.8 GiB が reserved の
    まま残る。本関数は workspace も clear してから ``empty_cache()`` する。

    **呼び出す前に、この process の CUDA 推論 / 翻訳をすべて停止し、worker を join / drain
    すること。** 実行中に呼ぶと、別スレッドの cuBLAS 呼び出しと競合する。engine /
    translator の ``cleanup()`` からは呼ばない (module docstring の「なぜ」を参照)。

    ``torch.cuda.synchronize()`` に失敗した場合は **その場で中断する** — 同期が成立して
    いない状態で process 全体の workspace を消さない (``reason='cuda-synchronize-failed'``)。

    Returns:
        CudaMemoryRelease: 実施可否と前後の実測値。``cublas_cleared`` が ``False`` の
        ときは ``reason`` / ``warnings`` に理由が入る (silent success にしない)。

    Example:
        >>> transcriber.close()            # 全 worker を止めてから
        >>> translator.cleanup()
        >>> result = release_idle_cuda_memory()
        >>> if not result.cublas_cleared:
        ...     ...  # 完全解放はできていない (process 再起動を案内する等)
    """
    torch = _import_torch()
    if torch is None:
        logger.debug("release_idle_cuda_memory: torch が無いので何もしない")
        return CudaMemoryRelease(
            attempted=False,
            cublas_cleared=False,
            cublas_api=None,
            allocated_before=0,
            allocated_after=0,
            reserved_before=0,
            reserved_after=0,
            reason=REASON_TORCH_MISSING,
        )

    if not torch.cuda.is_available():
        logger.debug("release_idle_cuda_memory: CUDA が無いので何もしない")
        return CudaMemoryRelease(
            attempted=False,
            cublas_cleared=False,
            cublas_api=None,
            allocated_before=0,
            allocated_after=0,
            reserved_before=0,
            reserved_after=0,
            reason=REASON_CUDA_UNAVAILABLE,
        )

    if not torch.cuda.is_initialized():
        # **context を作らない。** まだ初期化されていないなら allocator は何も持っておらず、
        # synchronize() を呼ぶとここで初めて CUDA context ができてしまう (逆効果)
        logger.debug("release_idle_cuda_memory: CUDA 未初期化なので何もしない")
        return CudaMemoryRelease(
            attempted=False,
            cublas_cleared=False,
            cublas_api=None,
            allocated_before=0,
            allocated_after=0,
            reserved_before=0,
            reserved_after=0,
            reason=REASON_CUDA_NOT_INITIALIZED,
        )

    allocated_before = int(torch.cuda.memory_allocated())
    reserved_before = int(torch.cuda.memory_reserved())
    warnings: list[str] = []
    reason = REASON_RELEASED

    # 1. Python 側の参照を落とす (del しただけの tensor は GC 後に allocator へ返る)
    gc.collect()

    # 2. 走っている kernel の完了を待つ。ここで待たずに workspace を消すと、
    #    in-flight な cuBLAS 呼び出しと競合し得る。
    #    **失敗したら中断する (fail closed)** — 同期できない = 「もう誰も走っていない」ことを
    #    確認できていない、かつ driver 状態が不明。そこで process 全体の private clear を
    #    叩くのは、この関数が自分で立てている前提を破る (PR #465 レビュー HIGH)
    try:
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - driver 側の異常も握って構造化結果で返す
        message = (
            f"torch.cuda.synchronize() に失敗したので解放を中断した ({exc})。"
            "同期できない状態で process 全体の cuBLAS workspace を消さない"
        )
        logger.warning("release_idle_cuda_memory: %s", message)
        return CudaMemoryRelease(
            attempted=True,
            cublas_cleared=False,
            cublas_api=None,
            allocated_before=allocated_before,
            allocated_after=_memory(torch.cuda.memory_allocated, allocated_before),
            reserved_before=reserved_before,
            reserved_after=_memory(torch.cuda.memory_reserved, reserved_before),
            reason=REASON_CUDA_SYNCHRONIZE_FAILED,
            warnings=(message,),
        )

    # 3. cuBLAS / cuBLASLt workspace を返す (**process 全体**)
    cublas_api, clear = _resolve_cublas_clear(torch)
    cublas_cleared = False
    if clear is None:
        reason = REASON_CUBLAS_API_MISSING
        message = (
            "この PyTorch には cuBLAS workspace を解放する API が無い "
            f"(試した: {', '.join(name for name, _ in _CUBLAS_CLEAR_CANDIDATES)})。"
            "empty_cache() だけ実行する — reserved は完全には戻らない"
        )
        warnings.append(message)
        logger.warning("release_idle_cuda_memory: %s", message)
    else:
        try:
            clear()
            cublas_cleared = True
        except Exception as exc:  # noqa: BLE001 - private API なので失敗も想定内
            reason = REASON_CUBLAS_API_FAILED
            cublas_api = None
            message = f"cuBLAS workspace の解放に失敗した ({exc})。empty_cache() だけ実行する"
            warnings.append(message)
            logger.warning("release_idle_cuda_memory: %s", message)

    # 4. workspace を返した後に allocator の空き segment を OS へ返す
    gc.collect()
    torch.cuda.empty_cache()

    allocated_after = int(torch.cuda.memory_allocated())
    reserved_after = int(torch.cuda.memory_reserved())
    result = CudaMemoryRelease(
        attempted=True,
        cublas_cleared=cublas_cleared,
        cublas_api=cublas_api,
        allocated_before=allocated_before,
        allocated_after=allocated_after,
        reserved_before=reserved_before,
        reserved_after=reserved_after,
        reason=reason,
        warnings=tuple(warnings),
    )
    logger.info(
        "release_idle_cuda_memory: reserved %d -> %d bytes (%.1f MiB 解放), allocated %d -> %d, "
        "cublas_cleared=%s (%s)",
        result.reserved_before,
        result.reserved_after,
        result.released_bytes / (1024 * 1024),
        result.allocated_before,
        result.allocated_after,
        result.cublas_cleared,
        result.cublas_api or result.reason,
    )
    return result
