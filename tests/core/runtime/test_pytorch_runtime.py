"""PyTorch ランタイム設定の決定表と適用契約 (Issue #422)。

**決定は ``_decide()`` に閉じており、プロセスの環境変数も module-level の cache も
書き換えない。** だから表を汚染なしに網羅できる。適用 / 冪等 / drift だけが
``configure_pytorch_runtime()`` の責務であり、そちらは別クラスで見る。

実測の根拠 (#422 §2.3): ``USE_PYTORCH_KERNEL_CACHE`` は **``"0"`` だけが無効化**で、
``false`` / ``no`` / 空文字はすべて**有効化**する。無効化したつもりの利用者が
有効化している状態を作らないため、LiveCap は ``0`` / ``1`` 以外を fail loud にする。
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import pytest

from livecap_cli.runtime import pytorch as mod
from livecap_cli.runtime.pytorch import (
    BOUNDARY,
    ENV_KERNEL_CACHE_PATH,
    ENV_USE_KERNEL_CACHE,
    PyTorchRuntimeError,
    _decide,
    _reset_pytorch_runtime_for_tests,
    configure_pytorch_runtime,
    current_pytorch_runtime,
)

#: ACP (cp932 / cp1252) の**外側**。cjk_kana (``ユーザー``) は cp932 の内側なので
#: 再現せず、これを使わないと narrow path を見逃す (#422 §7)。
OUTSIDE_ACP = "한국어Ω"


@pytest.fixture(autouse=True)
def _isolated_decision():
    """テスト間で決定を持ち越さない。"""
    _reset_pytorch_runtime_for_tests()
    yield
    _reset_pytorch_runtime_for_tests()


def _win(**environ: str) -> dict:
    """Windows の環境として ``_decide`` に渡す最小の mapping。"""
    return dict(environ)


class TestDecisionTable:
    """#422 §3.1 / §3.2 の決定表 9 行。"""

    def test_non_windows_is_a_no_op(self) -> None:
        """**ACP が無いので境界が存在しない。** 何も設定しないし、何も約束しない。"""
        decision = _decide({ENV_USE_KERNEL_CACHE: "nonsense"}, "linux")

        assert decision.kernel_cache == "not_applicable"
        assert decision.source == "platform"
        # 未知の値でも落とさない — Linux では PyTorch がどう解釈しても壊れない。
        assert not decision.expected_env

    def test_explicit_disable_is_respected(self) -> None:
        decision = _decide(_win(**{ENV_USE_KERNEL_CACHE: "0"}), "win32")

        assert decision.kernel_cache == "disabled"
        assert decision.source == "explicit_disable"
        assert decision.expected_env[ENV_USE_KERNEL_CACHE] == "0"

    def test_disable_wins_over_path_and_says_so(self, tmp_path: Path) -> None:
        """**無効化が優先。** ただし path を無視したことを観測可能にする。"""
        decision = _decide(
            _win(**{ENV_USE_KERNEL_CACHE: "0", ENV_KERNEL_CACHE_PATH: str(tmp_path)}),
            "win32",
        )

        assert decision.kernel_cache == "disabled"
        ignored = dict(decision.ignored)
        assert ENV_KERNEL_CACHE_PATH in ignored, (
            "無視した設定をログに出せないと、運用者は自分の指定が効いたか分からない"
        )

    def test_explicit_enable_with_ascii_path_is_respected(self, tmp_path: Path) -> None:
        decision = _decide(
            _win(**{ENV_USE_KERNEL_CACHE: "1", ENV_KERNEL_CACHE_PATH: str(tmp_path)}),
            "win32",
        )

        assert decision.kernel_cache == "enabled"
        assert decision.source == "explicit_path"
        assert decision.kernel_cache_path == str(tmp_path)

    def test_explicit_path_alone_is_opt_in(self, tmp_path: Path) -> None:
        """``USE`` 未設定でも、**置き場所をわざわざ指定した利用者**を上書きしない。"""
        decision = _decide(_win(**{ENV_KERNEL_CACHE_PATH: str(tmp_path)}), "win32")

        assert decision.kernel_cache == "enabled"
        assert decision.source == "explicit_path"

    def test_enabled_paths_warn_about_the_windows_write_bug(self, tmp_path: Path) -> None:
        """**有効化しても populate されない**ことを伝える (#422 §2.1)。"""
        decision = _decide(_win(**{ENV_KERNEL_CACHE_PATH: str(tmp_path)}), "win32")

        assert decision.warnings, "有効化を尊重するなら、効かない理由も伝えること"
        assert "rename" in " ".join(decision.warnings)

    def test_explicit_enable_without_path_uses_temp(self, tmp_path: Path) -> None:
        decision = _decide(_win(TEMP=str(tmp_path), **{ENV_USE_KERNEL_CACHE: "1"}), "win32")

        assert decision.kernel_cache == "enabled"
        assert decision.source == "explicit_enable"
        # 既定の置き場所を使うので、我々は path を設定しない。
        assert decision.expected_env[ENV_KERNEL_CACHE_PATH] is None

    def test_default_disables(self, tmp_path: Path) -> None:
        """**明示が何も無いときだけ**既定の無効化が効く。"""
        decision = _decide(_win(TEMP=str(tmp_path)), "win32")

        assert decision.kernel_cache == "disabled"
        assert decision.source == "default"
        assert decision.expected_env[ENV_USE_KERNEL_CACHE] == "0"


class TestFailLoud:
    """**黙って上書きしない。** 意図が読めない / 確実に壊れる設定は送出する。"""

    def test_explicit_nonascii_path_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / OUTSIDE_ACP

        with pytest.raises(PyTorchRuntimeError) as excinfo:
            _decide(_win(**{ENV_KERNEL_CACHE_PATH: str(bad)}), "win32")

        message = str(excinfo.value)
        assert BOUNDARY in message, "境界名が無いと、どの契約が破れたのか分からない"
        assert ENV_KERNEL_CACHE_PATH in message, "直すべき変数名を名指しすること"
        assert ascii(str(bad)).strip("'") in message, "問題の path を出すこと"
        assert excinfo.value.boundary == BOUNDARY

    def test_explicit_enable_with_nonascii_temp_raises(self, tmp_path: Path) -> None:
        """**有効化の意図と、確実に壊れる事実が両立しない。**"""
        bad_temp = tmp_path / OUTSIDE_ACP

        with pytest.raises(PyTorchRuntimeError) as excinfo:
            _decide(_win(TEMP=str(bad_temp), **{ENV_USE_KERNEL_CACHE: "1"}), "win32")

        assert ENV_KERNEL_CACHE_PATH in str(excinfo.value), (
            "有効化を諦めずに済む逃げ道 (明示 ASCII path) を提示すること"
        )

    def test_unusable_ascii_path_raises(self, tmp_path: Path) -> None:
        """ASCII でも**作れない**なら使えない。ASCII 判定だけで通さない。"""
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")

        with pytest.raises(PyTorchRuntimeError) as excinfo:
            _decide(_win(**{ENV_KERNEL_CACHE_PATH: str(blocker / "cache")}), "win32")

        assert "cannot create" in str(excinfo.value)

    @pytest.mark.parametrize("value", ["false", "FALSE", "no", "", "2", "abc", "true"])
    def test_unknown_use_values_raise(self, value: str, tmp_path: Path) -> None:
        """**実測 (#422 §2.3): PyTorch は ``"0"`` 以外をすべて有効として扱う。**

        ``USE_PYTORCH_KERNEL_CACHE=false`` と書いた利用者は、無効化したつもりで
        有効化している。意図と実際が食い違うのに兆候がゼロなのは epic #380 が
        排除している形そのものなので、ここで断つ。
        """
        with pytest.raises(PyTorchRuntimeError) as excinfo:
            _decide(_win(TEMP=str(tmp_path), **{ENV_USE_KERNEL_CACHE: value}), "win32")

        message = str(excinfo.value)
        assert ENV_USE_KERNEL_CACHE in message
        assert "ENABLED" in message, (
            "「PyTorch はこれを有効として扱う」を書かないと、利用者は誤解したままになる"
        )


class TestApplyAndIdempotence:
    """``configure_pytorch_runtime()`` の副作用側。"""

    @pytest.fixture(autouse=True)
    def _as_windows(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setenv("TEMP", str(tmp_path))
        monkeypatch.delenv(ENV_USE_KERNEL_CACHE, raising=False)
        monkeypatch.delenv(ENV_KERNEL_CACHE_PATH, raising=False)

    def test_applies_the_decision_to_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import os

        decision = configure_pytorch_runtime()

        assert decision.kernel_cache == "disabled"
        assert os.environ[ENV_USE_KERNEL_CACHE] == "0"

    def test_is_idempotent(self) -> None:
        first = configure_pytorch_runtime()
        second = configure_pytorch_runtime()

        assert first is second, "2 回目以降は最初の決定を返すこと"

    def test_environment_drift_fails_loud(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """**黙って再適用しない** (#422 §4.1)。

        キャッシュ先が確定するのは最初の Jiterator 実行時で、**確定済みかを読む
        公開 API が PyTorch に無い**。再適用が効いた保証が無いので、「直したつもり」
        のログを残すより、誰が何を壊したかを見せる。
        """
        configure_pytorch_runtime()
        monkeypatch.setenv(ENV_USE_KERNEL_CACHE, "1")

        with pytest.raises(PyTorchRuntimeError) as excinfo:
            configure_pytorch_runtime()

        message = str(excinfo.value)
        assert ENV_USE_KERNEL_CACHE in message
        assert "'1'" in message, "実際に見つかった値を出すこと"

    def test_concurrent_calls_agree(self) -> None:
        """並行呼び出しでも設定が競合しないこと。"""
        seen: list = []
        errors: list = []
        barrier = threading.Barrier(8)

        def worker() -> None:
            try:
                barrier.wait()
                seen.append(configure_pytorch_runtime())
            except BaseException as exc:  # noqa: BLE001 - 何が来ても記録する
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors, f"並行呼び出しで例外: {errors}"
        assert len({id(d) for d in seen}) == 1, "スレッドごとに違う決定が配られている"


class TestObservability:
    """**決定が観測できること。** 黙って上書きしない、を成立させるための条件である。"""

    @pytest.fixture(autouse=True)
    def _as_windows(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setenv("TEMP", str(tmp_path))
        monkeypatch.delenv(ENV_USE_KERNEL_CACHE, raising=False)
        monkeypatch.delenv(ENV_KERNEL_CACHE_PATH, raising=False)

    def test_decision_is_readable_without_configuring(self) -> None:
        """``current_pytorch_runtime()`` は**読むだけ**である。

        設定してしまうと「まだ誰も初期化していない」という事実が消え、
        probe が production の配線を検証できなくなる。
        """
        assert current_pytorch_runtime() is None

        decision = configure_pytorch_runtime()

        assert current_pytorch_runtime() is decision

    def test_resolved_values_are_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """**解決値**をログに出す (raw args ではなく)。境界名も出す。"""
        with caplog.at_level("INFO", logger="livecap_cli.runtime.pytorch"):
            configure_pytorch_runtime()

        text = caplog.text
        assert BOUNDARY in text
        assert "kernel_cache=disabled" in text

    def test_ignored_settings_are_logged(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """無視した設定こそ観測できなければならない。"""
        monkeypatch.setenv(ENV_USE_KERNEL_CACHE, "0")
        monkeypatch.setenv(ENV_KERNEL_CACHE_PATH, str(tmp_path / "cache"))

        with caplog.at_level("WARNING", logger="livecap_cli.runtime.pytorch"):
            configure_pytorch_runtime()

        assert ENV_KERNEL_CACHE_PATH in caplog.text, (
            "path を無視したことがログに出ないと、運用者は自分の指定が効いたと誤解する"
        )


def test_configure_does_not_import_torch() -> None:
    """**契約: torch を import しない / CUDA を初期化しない。**

    ``BaseEngine.__init__`` から呼ぶので、ここで torch を引き込むと CPU-only 環境の
    構築コストと import 順序を変えてしまう。**同一プロセスでは他のテストが既に
    torch を import している可能性があるので、子プロセスで見る。**
    """
    code = (
        "import sys;"
        "from livecap_cli.runtime import configure_pytorch_runtime as c;"
        "c();"
        "print('torch' in sys.modules)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(mod.__file__).resolve().parents[2]),
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False", (
        f"configure_pytorch_runtime() が torch を import している: {proc.stdout!r}"
    )
