"""``release_idle_cuda_memory()`` が実 GPU で reserved を baseline へ戻すこと (Issue #462)。

固定するのは 3 つ。

1. ``torch.cuda.empty_cache()` **だけでは戻らない** — matmul を 1 回するだけで
   cuBLAS workspace (既定 8,519,680 bytes) が生き残り、その segment が返らない
2. helper を呼ぶと allocated / reserved が baseline へ戻る
3. **idle な別モデルを壊さない** — helper の後もその tensor と推論結果が保たれる

**すべて fresh subprocess で回す。** cuBLAS workspace は process 内の static map なので、
同一 process で複数ケースを回すと前のケースの clear に汚染される (pytest 本体の process では
他のテストが CUDA を触っている可能性もある)。

上流 (PyTorch) が workspace の lifecycle を変えたら 1. が落ちる。それは**設計どおり**で、
落ちたら #462 の前提 (helper が要るかどうか) を再評価すること。
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

#: 非 Hopper GPU の既定 cuBLAS workspace (`:4096:2:16:8` = 2 * 4096 KiB + 8 * 16 KiB)
DEFAULT_CUBLAS_WORKSPACE_BYTES = 8_519_680

_RESIDUAL_AFTER_EMPTY_CACHE = """
import gc, json, torch

if not torch.cuda.is_available():
    print("RESULT " + json.dumps({"skip": "cuda unavailable"}))
    raise SystemExit(0)

from livecap_cli.runtime import release_idle_cuda_memory

base_allocated, base_reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
x = torch.randn(1024, 1024, device="cuda")
y = x @ x
del x, y
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
after_empty = {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved()}

result = release_idle_cuda_memory()
after_helper = {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved()}
again = release_idle_cuda_memory()  # 冪等

print("RESULT " + json.dumps({
    "baseline": {"allocated": base_allocated, "reserved": base_reserved},
    "after_empty": after_empty,
    "after_helper": after_helper,
    "result": result.to_dict(),
    "second": again.to_dict(),
}))
"""

_OTHER_MODEL_SURVIVES = """
import json, torch

if not torch.cuda.is_available():
    print("RESULT " + json.dumps({"skip": "cuda unavailable"}))
    raise SystemExit(0)

from livecap_cli.runtime import release_idle_cuda_memory

torch.manual_seed(0)
model = torch.nn.Linear(64, 64).cuda().eval()
keep = torch.arange(64, dtype=torch.float32, device="cuda")
with torch.no_grad():
    before = model(keep).clone()

# 「別の仕事」を終わらせて helper を呼ぶ — idle な model / tensor は生き残らなければならない
scratch = torch.randn(512, 512, device="cuda")
del scratch
result = release_idle_cuda_memory()

with torch.no_grad():
    after = model(keep).clone()

print("RESULT " + json.dumps({
    "kept_tensor_ok": bool(torch.equal(keep, torch.arange(64, dtype=torch.float32, device="cuda"))),
    "inference_matches": bool(torch.equal(before, after)),
    "reserved_after": torch.cuda.memory_reserved(),
    "result": result.to_dict(),
}))
"""


def _run(script: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = next(line for line in proc.stdout.splitlines() if line.startswith("RESULT "))
    payload = json.loads(line[len("RESULT ") :])
    if "skip" in payload:
        pytest.skip(payload["skip"])
    return payload


def test_empty_cache_leaves_the_cublas_workspace_and_the_helper_releases_it() -> None:
    payload = _run(_RESIDUAL_AFTER_EMPTY_CACHE)

    assert payload["after_empty"]["allocated"] == DEFAULT_CUBLAS_WORKSPACE_BYTES, (
        "empty_cache() の後に残るのは cuBLAS workspace ちょうどのはず: " f"{payload['after_empty']}"
    )
    assert payload["after_empty"]["reserved"] > payload["baseline"]["reserved"], "segment が返っていないこと"

    assert payload["after_helper"]["allocated"] == payload["baseline"]["allocated"]
    assert payload["after_helper"]["reserved"] == payload["baseline"]["reserved"]

    result = payload["result"]
    assert result["attempted"] and result["cublas_cleared"], result
    assert result["cublas_api"], "どの API を使ったか報告すること"
    assert result["released_bytes"] >= DEFAULT_CUBLAS_WORKSPACE_BYTES

    second = payload["second"]
    assert second["attempted"] and second["released_bytes"] == 0, "2 回目も安全で、解放量は 0"


def test_idle_model_and_tensor_survive_the_release() -> None:
    payload = _run(_OTHER_MODEL_SURVIVES)

    assert payload["kept_tensor_ok"], "helper が生きている tensor を壊した"
    assert payload["inference_matches"], "helper の後に同じ入力で結果が変わった"
    assert payload["reserved_after"] > 0, "使用中の model の分は reserved に残る"
    assert payload["result"]["cublas_cleared"], payload["result"]
