"""共有初期化がすべての torch 到達経路から呼ばれること (Issue #422)。

**`EngineFactory` に置くだけでは足りない。** engine クラスを直接生成する library
利用者を守れないためである (translator / VAD も同様)。したがって呼び出し位置は
共通の基底 ``__init__`` であり、ここではそれを**直接構築**で確かめる。

``load_model()`` を入口にできない理由も固定する:

- ``BaseEngine.load_model()`` は parakeet / reazonspeech が override している
- ``BaseTranslator.load_model()`` は基底が no-op で、ローカル translator 2 つが
  override している

最後に **audit test** を置く。新しい torch consumer が増えたときに「共有初期化を
通す入口か」を必ず判断させるためで、これが無いと抜けが黙って積み上がる。
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from livecap_cli.engines.base_engine import BaseEngine, TranscriptionResult
from livecap_cli.translation.base import BaseTranslator

PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "livecap_cli"


class _StubEngine(BaseEngine):
    """**Factory を経由しない**最小の具象 engine。"""

    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> TranscriptionResult:  # pragma: no cover
        raise NotImplementedError

    def get_engine_name(self) -> str:
        return "stub"

    def get_supported_languages(self) -> list:
        return []

    def get_required_sample_rate(self) -> int:
        return 16000


class _StubTranslator(BaseTranslator):
    """**Factory を経由しない**最小の具象 translator。"""

    def translate(self, text, source_lang, target_lang, context=None):  # pragma: no cover
        raise NotImplementedError

    def get_supported_pairs(self) -> List[Tuple[str, str]]:
        return []

    def get_translator_name(self) -> str:
        return "stub"


@pytest.fixture()
def calls(monkeypatch: pytest.MonkeyPatch) -> list:
    """共有初期化の呼び出し回数を数える。

    **各 module が import した名前を差し替える。** 定義元だけを差し替えても、
    ``from ... import configure_pytorch_runtime`` 済みの module には効かない。
    """
    recorded: list = []

    def spy():
        recorded.append(True)
        return None

    monkeypatch.setattr("livecap_cli.engines.base_engine.configure_pytorch_runtime", spy)
    monkeypatch.setattr("livecap_cli.translation.base.configure_pytorch_runtime", spy)
    return recorded


def test_engine_constructed_directly_configures_runtime(calls: list) -> None:
    _StubEngine(device="cpu")

    assert calls, (
        "engine クラスを直接生成しても共有初期化が走ること。"
        "EngineFactory に置くだけでは library 利用者を守れない。"
    )


def test_translator_constructed_directly_configures_runtime(calls: list) -> None:
    _StubTranslator()

    assert calls, "translator クラスを直接生成しても共有初期化が走ること"


def test_vad_backend_configures_runtime_before_importing_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VAD は engine を経由しない独立した torch 到達経路である。

    **``import torch`` より前に呼ぶ**ことまで見る — 順序が逆でも今は間に合う
    (確定は最初の Jiterator 実行時) が、「torch を触る直前に決める」という読み方を
    崩さないため。
    """
    pytest.importorskip("silero_vad", reason="silero-vad 未導入 (vad extra)")
    pytest.importorskip("torch")

    from livecap_cli.vad.backends import silero

    order: list = []

    monkeypatch.setattr(
        "livecap_cli.runtime.configure_pytorch_runtime",
        lambda: order.append("configure"),
    )
    monkeypatch.setattr(
        "silero_vad.load_silero_vad",
        lambda onnx=True: order.append("load_model") or object(),
    )

    silero.SileroVAD(onnx=True)

    assert order and order[0] == "configure", (
        f"VAD backend が共有初期化を先に呼んでいない: {order}"
    )


# --- audit ---------------------------------------------------------------------


def _python_files() -> list[Path]:
    return sorted(p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _imports_torch(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name == "torch" or a.name.startswith("torch.") for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "torch" or module.startswith("torch."):
                return True
    return False


#: **直接** torch を import する module (repo 相対)。
#:
#: 推移的な import (NeMo / transformers / whisper_s2t 経由) はここに現れないが、
#: それらはすべて engine / translator / VAD の構築を通るので共有初期化に守られる。
#: この一覧が守るのは「**新しい直接 consumer** が黙って増えないこと」である。
KNOWN_TORCH_IMPORTERS = {
    "livecap_cli/cli.py",
    "livecap_cli/engines/canary_engine.py",
    "livecap_cli/engines/nemo_jit_patch.py",
    "livecap_cli/engines/parakeet_engine.py",
    "livecap_cli/engines/qwen3asr_engine.py",
    "livecap_cli/engines/voxtral_engine.py",
    "livecap_cli/engines/whispers2t_engine.py",
    "livecap_cli/translation/impl/riva_instruct.py",
    "livecap_cli/utils/__init__.py",
    "livecap_cli/vad/backends/silero.py",
}


def test_torch_consumers_are_known() -> None:
    """**新しい torch consumer は、共有初期化を通す入口かどうかの判断を要求する。**

    増えたらこのテストが落ちるので、``KNOWN_TORCH_IMPORTERS`` を更新する前に
    「この経路は ``BaseEngine`` / ``BaseTranslator`` / VAD のどれかを通るか」を
    考えることになる。通らないなら ``configure_pytorch_runtime()`` を足す。
    """
    repo_root = PACKAGE_ROOT.parent
    found = {
        str(path.relative_to(repo_root)).replace("\\", "/")
        for path in _python_files()
        if _imports_torch(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    }

    added = sorted(found - KNOWN_TORCH_IMPORTERS)
    removed = sorted(KNOWN_TORCH_IMPORTERS - found)

    assert not added, (
        "torch を直接 import する module が増えた: "
        f"{added}。この経路が BaseEngine / BaseTranslator / Silero VAD のいずれかを"
        "通るなら共有初期化に守られているので KNOWN_TORCH_IMPORTERS へ足す。"
        "通らないなら configure_pytorch_runtime() を呼ぶこと (Issue #422)。"
    )
    assert not removed, (
        f"torch を import しなくなった module: {removed}。"
        "KNOWN_TORCH_IMPORTERS から外して、一覧が実態とずれないようにすること。"
    )


def _init_calls_super(class_node: ast.ClassDef) -> bool | None:
    """``__init__`` が ``super().__init__()`` を呼ぶか。``__init__`` が無ければ None。"""
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "__init__"
                    and isinstance(inner.func.value, ast.Call)
                    and isinstance(inner.func.value.func, ast.Name)
                    and inner.func.value.func.id == "super"
                ):
                    return True
            return False
    return None


@pytest.mark.parametrize(
    ("directory", "base"),
    [("engines", "BaseEngine"), ("translation/impl", "BaseTranslator")],
)
def test_subclasses_call_super_init(directory: str, base: str) -> None:
    """**基底 ``__init__`` を入口にした前提を固定する。**

    ``super().__init__()`` を呼ばない具象クラスが 1 つでも入ると、その engine /
    translator だけが共有初期化を通らず、**非 ASCII 環境でその経路だけ壊れる**。
    「ゲートは緑だが対象経路を通っていない」の典型形なので、静的に禁じる。
    """
    offenders: list[str] = []
    for path in sorted((PACKAGE_ROOT / directory).glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {b.id for b in node.bases if isinstance(b, ast.Name)}
            if base not in bases:
                continue
            if _init_calls_super(node) is False:
                offenders.append(f"{path.name}::{node.name}")

    assert not offenders, (
        f"{base} を継承しているのに super().__init__() を呼んでいない: {offenders}。"
        "共有 PyTorch 初期化 (Issue #422) がその経路だけ走らなくなる。"
    )
