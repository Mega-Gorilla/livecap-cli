"""ReazonSpeech の cache identity (Issue #409)。

`ModelMemoryCache` のキーが **`use_int8` とディレクトリの basename しか含んでいない**ため、
① 異なる models root の同名ディレクトリが衝突し、② モデルファイルを差し替えても古い
recognizer が返り、③ `num_threads` / `decoding_method` を変えても同じキーになる。

**`_load_model_from_path()` 越しに検査する。** identity builder を直接呼ぶだけだと
「engine が実際にそれを使っているか」を保証できない (#379 で踏んだ失敗モード)。

**「壊れた recognizer を保存しない」は本 issue のスコープ外**である
(#392 が post-load health check と保存ゲートを持つ)。ここでは identity だけを見る。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from livecap_cli.engines.model_memory_cache import ModelMemoryCache
from livecap_cli.engines.reazonspeech_engine import ReazonSpeechEngine

_EPOCHS = 99


def _file_names(use_int8: bool) -> dict[str, str]:
    """production と同じファイル名 (テスト側の期待値)。"""
    if use_int8:
        return {
            "tokens": "tokens.txt",
            "encoder": f"encoder-epoch-{_EPOCHS}-avg-1.int8.onnx",
            "decoder": f"decoder-epoch-{_EPOCHS}-avg-1.onnx",
            "joiner": f"joiner-epoch-{_EPOCHS}-avg-1.int8.onnx",
        }
    return {
        "tokens": "tokens.txt",
        "encoder": f"encoder-epoch-{_EPOCHS}-avg-1.onnx",
        "decoder": f"decoder-epoch-{_EPOCHS}-avg-1.onnx",
        "joiner": f"joiner-epoch-{_EPOCHS}-avg-1.onnx",
    }


def _make_model_dir(root: Path, *, use_int8: bool = True, tokens: bytes = b"a 0\nb 1\n") -> Path:
    """`sherpa-onnx-zipformer-...` 相当のモデルディレクトリを作る。"""
    root.mkdir(parents=True, exist_ok=True)
    for role, name in _file_names(use_int8).items():
        payload = tokens if role == "tokens" else f"{role}-bytes".encode()
        (root / name).write_bytes(payload)
    return root


class _FakeRecognizer:
    """`OfflineRecognizer` の代わり。**呼び出しごとに別オブジェクト**になる。"""

    def __init__(self, kwargs: dict[str, Any]):
        self.kwargs = kwargs


class _Builder:
    """`from_transducer` の fake。呼び出しを記録する。"""

    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs) -> _FakeRecognizer:
        self.calls.append(kwargs)
        return _FakeRecognizer(kwargs)


@pytest.fixture(autouse=True)
def _clean_cache():
    """`ModelMemoryCache` は class 変数を共有する**グローバル状態**である。"""
    ModelMemoryCache.clear()
    yield
    ModelMemoryCache.clear()


@pytest.fixture
def builder(monkeypatch: pytest.MonkeyPatch) -> _Builder:
    import sherpa_onnx

    fake = _Builder()
    monkeypatch.setattr(
        sherpa_onnx.OfflineRecognizer, "from_transducer", staticmethod(fake)
    )
    return fake


def _engine(*, use_int8: bool = True, num_threads: int = 4,
            decoding_method: str = "greedy_search") -> ReazonSpeechEngine:
    return ReazonSpeechEngine(
        device="cpu",
        use_int8=use_int8,
        num_threads=num_threads,
        decoding_method=decoding_method,
    )


class TestIdentityDistinguishes:
    """**同じ recognizer を返してはいけない**組み合わせ。"""

    def test_same_basename_in_a_different_root_is_a_miss(self, tmp_path: Path, builder):
        name = "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01"
        first = _make_model_dir(tmp_path / "rootA" / name)
        second = _make_model_dir(tmp_path / "rootB" / name)

        a = _engine()._load_model_from_path(first)
        b = _engine()._load_model_from_path(second)

        assert a is not b, "異なる models root の同名ディレクトリが衝突している"
        assert len(builder.calls) == 2

    def test_changing_tokens_content_is_a_miss(self, tmp_path: Path, builder):
        model_dir = _make_model_dir(tmp_path / "model", tokens=b"a 0\n")
        a = _engine()._load_model_from_path(model_dir)

        (model_dir / "tokens.txt").write_bytes(b"a 0\nb 1\nc 2\n")
        b = _engine()._load_model_from_path(model_dir)

        assert a is not b, "tokens.txt を差し替えても古い recognizer が返っている"

    @pytest.mark.parametrize("role", ["encoder", "decoder", "joiner"])
    def test_changing_an_onnx_stat_is_a_miss(self, tmp_path: Path, builder, role: str):
        model_dir = _make_model_dir(tmp_path / "model")
        a = _engine()._load_model_from_path(model_dir)

        target = model_dir / _file_names(True)[role]
        target.write_bytes(b"different-and-longer-bytes")
        b = _engine()._load_model_from_path(model_dir)

        assert a is not b, f"{role} を差し替えても古い recognizer が返っている"

    def test_changing_num_threads_is_a_miss(self, tmp_path: Path, builder):
        model_dir = _make_model_dir(tmp_path / "model")
        a = _engine(num_threads=4)._load_model_from_path(model_dir)
        b = _engine(num_threads=8)._load_model_from_path(model_dir)

        assert a is not b, "num_threads は from_transducer に渡るのにキーへ入っていない"
        assert builder.calls[0]["num_threads"] == 4
        assert builder.calls[1]["num_threads"] == 8

    def test_changing_decoding_method_is_a_miss(self, tmp_path: Path, builder):
        model_dir = _make_model_dir(tmp_path / "model")
        a = _engine(decoding_method="greedy_search")._load_model_from_path(model_dir)
        b = _engine(decoding_method="modified_beam_search")._load_model_from_path(model_dir)

        assert a is not b, "decoding_method は from_transducer に渡るのにキーへ入っていない"

    @pytest.mark.parametrize("package", ["sherpa-onnx", "sherpa-onnx-core"])
    def test_changing_a_native_version_is_a_miss(
        self, tmp_path: Path, builder, monkeypatch: pytest.MonkeyPatch, package: str
    ):
        """**core も含める。** native 処理には `sherpa-onnx-core` も関係する。"""
        from livecap_cli.engines import reazonspeech_cache

        model_dir = _make_model_dir(tmp_path / "model")
        a = _engine()._load_model_from_path(model_dir)

        real = reazonspeech_cache._package_version

        def bumped(name: str) -> str:
            return "99.99.99" if name == package else real(name)

        monkeypatch.setattr(reazonspeech_cache, "_package_version", bumped)
        b = _engine()._load_model_from_path(model_dir)

        assert a is not b, f"{package} の版が変わっても同じ recognizer が返っている"


class TestCacheHit:
    def test_same_inputs_reuse_the_recognizer(self, tmp_path: Path, builder):
        model_dir = _make_model_dir(tmp_path / "model")
        a = _engine()._load_model_from_path(model_dir)
        b = _engine()._load_model_from_path(model_dir)

        assert a is b, "同じ入力なのに cache hit していない"
        assert len(builder.calls) == 1, (
            "cache hit なのに from_transducer が再実行されている"
        )

    def test_legacy_v1_key_is_not_reused(self, tmp_path: Path, builder):
        """**v1 key (`reazonspeech_{use_int8}_{name}`) は読みも書きもしない。**"""
        model_dir = _make_model_dir(tmp_path / "model")
        sentinel = object()
        legacy_key = f"reazonspeech_True_{model_dir.name}"
        ModelMemoryCache.set(legacy_key, sentinel, strong=True)

        loaded = _engine()._load_model_from_path(model_dir)

        assert loaded is not sentinel, "legacy な v1 key を再利用している"
        assert len(builder.calls) == 1

        # production が**書いた**キーだけを見る (legacy key はテストが仕込んだもの)。
        written = set(ModelMemoryCache.get_stats()["cache_keys"]) - {legacy_key}
        assert written, "新しいキーで保存していない"
        assert all(k.startswith("reazonspeech:v2:") for k in written), (
            f"v1 形式のキーを書いている: {sorted(written)}"
        )


class TestFailLoud:
    """**identity は lookup より前に確定する。**

    ここを崩すと将来「identity 取得に失敗したら旧 key や簡易 key へ fallback する」
    実装が入り得る。
    """

    def test_missing_file_does_not_return_a_cached_entry(self, tmp_path: Path, builder):
        model_dir = _make_model_dir(tmp_path / "model")
        cached = _engine()._load_model_from_path(model_dir)
        assert cached is not None

        (model_dir / "tokens.txt").unlink()

        with pytest.raises(Exception) as excinfo:
            _engine()._load_model_from_path(model_dir)
        assert not isinstance(excinfo.value, AssertionError)
        assert len(builder.calls) == 1, "ファイルが欠けているのに再構築している"

    def test_model_changed_during_construction_is_not_cached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """構築中にモデルが変わったら、**古い identity のキーへ保存しない**。"""
        from livecap_cli.engines.reazonspeech_cache import ModelIdentityChangedError

        import sherpa_onnx

        model_dir = _make_model_dir(tmp_path / "model")

        def mutating(**kwargs):
            # 構築中に tokens.txt が差し替わる
            (model_dir / "tokens.txt").write_bytes(b"a 0\nb 1\nc 2\nd 3\n")
            return _FakeRecognizer(kwargs)

        monkeypatch.setattr(
            sherpa_onnx.OfflineRecognizer, "from_transducer", staticmethod(mutating)
        )

        with pytest.raises(ModelIdentityChangedError):
            _engine()._load_model_from_path(model_dir)

        assert ModelMemoryCache.get_stats()["cache_keys"] == [], (
            "構築中に変わったモデルをキャッシュしている"
        )


class TestKeyStability:
    """canonical serialization を固定する。`repr(dict)` に依存しない。"""

    def test_key_is_deterministic_and_order_independent(self, tmp_path: Path):
        from livecap_cli.engines.reazonspeech_cache import build_identity

        model_dir = _make_model_dir(tmp_path / "model")
        kwargs = dict(use_int8=True, num_threads=4, decoding_method="greedy_search")

        first = build_identity(model_dir, **kwargs)
        second = build_identity(model_dir, **kwargs)

        assert first.cache_key() == second.cache_key()
        assert first.cache_key().startswith("reazonspeech:v2:")
        # digest なので生の path が漏れない
        assert str(model_dir) not in first.cache_key()
