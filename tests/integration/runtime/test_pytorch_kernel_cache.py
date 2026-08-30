"""PyTorch CUDA Jiterator kernel cache の**上流側の性質**を固定する (Issue #422)。

ここは LiveCap の修正を測るところ**ではない** — それは
``tests/nonascii`` の ``framework.pytorch.cuda_jiterator_kernel_cache`` 行が見る。
本 module が固定するのは、その修正の**前提**である 3 つの事実である。

1. ACP の外側の cache 先で CUDA の Jiterator 演算が壊れること
   (**`cjk_kana` では壊れない** — この非対称が「両 variant を要求する」根拠)
2. CPU 経路は影響を受けないこと
3. **cache が populate されないこと** — 既定で無効化しても失われるものが無い、
   という判断 (#422 §3.3) の根拠であり、**上流が直ったら落ちて再評価を促す**

**すべて fresh subprocess で回す。** PyTorch は cache 先を関数内 static として
一度だけ解決するので、同一プロセスで環境変数を変えながら matrix を回すと
**最初のケースに汚染される**。

上流が直ったらどれかが落ちる。それが**設計どおり**である — 落ちたら #422 §3.4 の
再評価 (永続 ASCII cache root の是非) を行い、本 module の期待値を更新すること。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

#: ACP (cp932 / cp1252) の外側。
OUTSIDE_ACP = "한국어Ω"
#: cp932 の**内側**。日本語 Windows では narrow path でも通ってしまう。
CJK_KANA = "ユーザー"

#: Jiterator 経路に入る演算。素の行列積などでは踏まない。
_SCRIPT = """
import os, pathlib, sys
cache = pathlib.Path(sys.argv[1]); cache.mkdir(parents=True, exist_ok=True)
os.environ["PYTORCH_KERNEL_CACHE_PATH"] = str(cache)
device = sys.argv[2]
import torch
x = torch.randn(16381, device=device)
try:
    torch.fft.rfft(x).abs()
    if device == "cuda":
        torch.cuda.synchronize()
    print("OK")
except UnicodeDecodeError as exc:
    print("UNICODE_DECODE_ERROR", exc)
"""


def _child_env() -> dict:
    """**親の cache 設定を継承しない。**

    親 (pytest プロセスや CI シェル) に ``USE_PYTORCH_KERNEL_CACHE=0`` が残っていると、
    ACP 外の path でも子が ``OK`` を返し、**上流欠陥の再現テストが「PyTorch が直った」と
    誤って主張する**。raw track の子は cache 先を自分で設定するので、この 2 変数は
    明示的に消してから渡す。
    """
    env = dict(os.environ)
    env.pop("USE_PYTORCH_KERNEL_CACHE", None)
    env.pop("PYTORCH_KERNEL_CACHE_PATH", None)
    return env


def _run(cache_dir: Path, device: str = "cuda") -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(cache_dir), device],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_child_env(),
        timeout=300,
    )
    assert proc.returncode == 0, f"子プロセスが落ちた: {proc.stderr[-2000:]}"
    return proc.stdout.strip().splitlines()[-1]


@pytest.fixture(scope="module", autouse=True)
def _requires_cuda_on_windows():
    if sys.platform != "win32":
        pytest.skip("narrow path 変換は Windows 固有 (ACP が無い環境では起きない)")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA が使えない (Jiterator 経路に入らない)")


def test_outside_acp_cache_path_breaks_cuda(tmp_path: Path) -> None:
    """**上流の欠陥そのもの。** ACP の外側の cache 先で CUDA 演算が落ちる。

    これが落ちるようになったら、PyTorch 側で narrow path 変換が直った可能性がある。
    #422 §3.4 の再評価を行うこと。
    """
    assert _run(tmp_path / OUTSIDE_ACP).startswith("UNICODE_DECODE_ERROR"), (
        "PyTorch が ACP 外の kernel cache path を受け付けるようになった。"
        "#422 の前提が変わったので、既定の無効化と永続 ASCII cache root の"
        "是非を再評価すること。"
    )


def test_cjk_kana_cache_path_does_not_reproduce(tmp_path: Path) -> None:
    """**`cjk_kana` だけでは再現しない** — 両 variant を要求する根拠。

    ``ユーザー`` は cp932 の内側なので、日本語 Windows では narrow path 変換が
    成功してしまう。この非対称を固定しておかないと、``required_variants`` から
    ``outside_acp`` が黙って落ちても誰も気付かない。
    """
    assert _run(tmp_path / CJK_KANA) == "OK", (
        "cp932 の内側の path でも壊れるようになった。"
        "非対称が崩れたので #422 の variant 設計を見直すこと。"
    )


def test_cpu_path_is_unaffected(tmp_path: Path) -> None:
    """**CPU では踏まない。** 境界は CUDA の Jiterator に限られる。"""
    assert _run(tmp_path / OUTSIDE_ACP, device="cpu") == "OK"


def test_cache_is_not_populated_on_windows(tmp_path: Path) -> None:
    """**2 プロセス判定** — 既定で無効化する判断の根拠 (#422 §3.3 / §3.4)。

    ``final-named > 0`` を条件にしてはならない。手動配置・古い PyTorch・別プロセスの
    残骸でも成立してしまう。**「今の PyTorch が書けること」と「次のプロセスが
    読めること」の両方**を、空の専用ディレクトリで確かめる。

    現状はどちらも成立しない: PyTorch 2.9.1 は ``<name>_tmp_<pid>`` へ書いてから
    最終名へ rename するが、``std::ofstream`` を閉じる前に ``std::rename()`` を
    呼ぶため Windows では rename が失敗する。ルックアップは最終名で行われるので、
    **自分で書いたものを自分で読めない**。

    **このテストが落ちたら上流が直ったということである。** その場合、無効化の代償が
    生まれるので #422 §3.3 の案 B (永続 ASCII cache root) を再検討すること。
    """
    cache = tmp_path / "kernels"

    assert _run(cache) == "OK"
    after_a = sorted(p.name for p in cache.iterdir())
    assert after_a, "process A が何も書かなかった - この判定自体が成立していない"

    final_named = [n for n in after_a if "_tmp_" not in n]
    assert not final_named, (
        f"PyTorch が最終名のファイルを書けるようになった: {final_named}。"
        "#422 §3.4 の再評価 trigger が成立したので、案 B (永続 ASCII cache root) を"
        "検討し、本テストの期待値を更新すること。"
    )

    assert _run(cache) == "OK"
    added = sorted(set(p.name for p in cache.iterdir()) - set(after_a))
    assert added, (
        "process B が新しい `_tmp_` を作らなかった = cache がヒットした。"
        "#422 §3.4 の再評価 trigger が成立している。"
    )
