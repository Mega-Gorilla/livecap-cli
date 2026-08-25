"""`reazon_model_files()` が**整合したセットだけ**を返すこと (Issue #377)。

ファイルごとに独立して glob していた頃は、壊れた int8 dir と完全な float32
ファイルが同居すると `encoder-*.int8.onnx` + `decoder-*.onnx` + `joiner-*.onnx`
という**混在セット**を返した。sherpa-onnx へ渡す 3 つの ONNX は同じ量子化で
そろっている必要がある。

int8 の decoder は量子化されないため ``.int8`` が付かない — これが
「per-file glob でも正しく見える」錯覚の元だった。
"""

from __future__ import annotations

from pathlib import Path

from .probes.native_models import reazon_model_files

INT8 = (
    "tokens.txt",
    "encoder-epoch-99-avg-1.int8.onnx",
    "decoder-epoch-99-avg-1.onnx",
    "joiner-epoch-99-avg-1.int8.onnx",
)
FLOAT32 = (
    "tokens.txt",
    "encoder-epoch-99-avg-1.onnx",
    "decoder-epoch-99-avg-1.onnx",
    "joiner-epoch-99-avg-1.onnx",
)


def _make(root: Path, names) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_bytes(b"")
    return root


def test_int8_only(tmp_path: Path):
    variant, files = reazon_model_files(_make(tmp_path / "int8", INT8))
    assert variant == "int8"
    assert files == INT8


def test_float32_only(tmp_path: Path):
    variant, files = reazon_model_files(_make(tmp_path / "f32", FLOAT32))
    assert variant == "float32"
    assert files == FLOAT32


def test_int8_is_preferred_when_both_are_present(tmp_path: Path):
    """軽い方 (154 MB) を選ぶ。測定内容は同じ。"""
    root = _make(tmp_path / "both", set(INT8) | set(FLOAT32))
    variant, _ = reazon_model_files(root)
    assert variant == "int8"


def test_incomplete_int8_falls_back_to_complete_float32(tmp_path: Path):
    """**混在セットを返さない。**

    int8 の encoder だけが残った dir に完全な float32 が同居する状況。
    per-file glob だった頃はここで int8 encoder + float32 decoder/joiner を
    返していた。
    """
    root = tmp_path / "mixed"
    _make(root, FLOAT32)
    (root / "encoder-epoch-99-avg-1.int8.onnx").write_bytes(b"")  # int8 は encoder だけ

    variant, files = reazon_model_files(root)

    assert variant == "float32"
    assert files == FLOAT32
    assert not any(".int8." in name for name in files)


def test_incomplete_directory_is_rejected(tmp_path: Path):
    """どちらのセットもそろわなければ ``None``。呼び出し側が次の候補へ進める。"""
    root = _make(tmp_path / "partial", ["tokens.txt", "encoder-epoch-99-avg-1.onnx"])
    assert reazon_model_files(root) is None


def test_missing_tokens_is_rejected(tmp_path: Path):
    """tokens.txt が無ければ、本 probe が守っている経路自体を通れない。"""
    root = _make(tmp_path / "no-tokens", [n for n in FLOAT32 if n != "tokens.txt"])
    assert reazon_model_files(root) is None
