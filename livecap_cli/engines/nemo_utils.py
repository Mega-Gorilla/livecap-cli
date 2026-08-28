"""NeMo フレームワーク用の共通ユーティリティ

Canary, Parakeet エンジンで共有される NeMo 関連の機能を提供。

PyInstaller 互換性:
    PyInstaller (frozen) 環境では、NeMo をインポートすると以下のライブラリで
    循環インポートエラーが発生する:

    1. datasets: datasets/packaged_modules/arrow/arrow.py が datasets.utils.logging に
       アクセスする際に datasets モジュールがまだ完全に初期化されていない (#216)

    2. librosa: librosa.filters と librosa.core.spectrum の間で循環依存が発生 (#219)
       - librosa.filters → librosa.core.convert
       - librosa.core (lazy_loader) → librosa.core.spectrum
       - librosa.core.spectrum → librosa.filters.get_window (循環)

    対策:
    1. check_nemo_availability() では importlib.util.find_spec() で存在確認のみ行う
    2. prepare_nemo_environment() で datasets.utils, librosa サブモジュールを事前インポート
    3. 実際の NeMo インポートは各エンジンの関数内で行う

    通常の Python 環境では、実際にインポートを試行して依存関係の問題も検出する。
"""
from __future__ import annotations

import importlib.util
import os
import sys
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

logger = logging.getLogger(__name__)

# NeMo framework - 遅延インポート
NEMO_AVAILABLE = None  # 初期状態は未確認
_NEMO_ENVIRONMENT_PREPARED = False  # 環境準備済みフラグ

#: NeMo が実際に使う logger 名。``propagate=False`` + 独自 stream handler を持つため、
#: **windowed build では一次エラーが app log に届かない** (Issue #379)。
NEMO_LOGGER_NAME = "nemo_logger"

#: ``restore_nemo_model()`` が既定で黙らせる logger。engine ごとに増やせる
#: (Canary は ``lhotse`` / ``nemo.collections`` も出す)。
DEFAULT_QUIET_LOGGERS: tuple[str, ...] = (NEMO_LOGGER_NAME,)


def check_nemo_availability() -> bool:
    """NeMo の利用可能性をチェック

    PyInstaller (frozen) 環境では循環インポート問題を回避するため、
    importlib.util.find_spec() でパッケージの存在確認のみを行う。

    通常の Python 環境では実際にインポートを試行し、
    依存関係の問題も早期に検出する。

    Returns:
        bool: NeMo が利用可能な場合 True
    """
    global NEMO_AVAILABLE
    if NEMO_AVAILABLE is not None:
        return NEMO_AVAILABLE

    # PyInstaller 環境では find_spec のみ使用（循環インポート回避）
    if getattr(sys, 'frozen', False):
        try:
            NEMO_AVAILABLE = importlib.util.find_spec("nemo") is not None
            if NEMO_AVAILABLE:
                logger.debug("NeMo パッケージが検出されました (frozen環境)")
            else:
                logger.warning("NeMo パッケージがインストールされていません")
        except Exception as e:
            NEMO_AVAILABLE = False
            logger.warning(f"NeMo の可用性チェックに失敗: {e}")
        return NEMO_AVAILABLE

    # 通常環境では実際にインポートを試行（依存関係問題を早期検出）
    try:
        # matplotlib backend issue を回避（Parakeet 用）
        import matplotlib
        matplotlib.use('Agg')  # 非対話的バックエンドを使用

        # PyInstaller 互換性のための JIT パッチを適用
        from . import nemo_jit_patch

        import nemo.collections.asr
        NEMO_AVAILABLE = True
        logger.info("NVIDIA NeMo が正常にインポートされました")
    except (ImportError, AttributeError) as e:
        NEMO_AVAILABLE = False
        # NeMo が利用できない場合は、詳細エラーを記録
        logger.error(f"NVIDIA NeMo のインポートに失敗しました: {e}")
        logger.error(f"Import error details: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")

    return NEMO_AVAILABLE


def prepare_nemo_environment() -> None:
    """NeMo インポート前の環境準備

    NeMo を実際にインポートする前に呼び出す。以下の設定を行う:
    - matplotlib バックエンドを非対話的に設定
    - PyInstaller 互換性のための JIT パッチを適用
    - PyInstaller 環境での追加設定（torch._dynamo, TorchScript 無効化）
    - PyInstaller 環境での datasets サブモジュール事前インポート（循環インポート回避, #216）
    - PyInstaller 環境での librosa サブモジュール事前インポート（循環インポート回避, #219）

    この関数は複数回呼び出しても安全（冪等性あり）。
    """
    global _NEMO_ENVIRONMENT_PREPARED
    if _NEMO_ENVIRONMENT_PREPARED:
        return

    # matplotlib backend issue を回避（Parakeet 用）
    try:
        import matplotlib
        matplotlib.use('Agg')  # 非対話的バックエンドを使用
    except ImportError:
        pass  # matplotlib がない場合は無視

    # PyInstaller 互換性のための JIT パッチを適用
    try:
        from . import nemo_jit_patch
    except ImportError:
        logger.debug("nemo_jit_patch モジュールが見つかりません")

    # PyInstaller 環境での追加設定
    if getattr(sys, 'frozen', False):
        # torch._dynamo を無効化
        os.environ['TORCHDYNAMO_DISABLE'] = '1'
        # TorchScript を無効化
        os.environ['PYTORCH_JIT'] = '0'
        logger.debug("PyInstaller 環境用の設定を適用しました")

        # datasets サブモジュールを NeMo より先にインポート（循環インポート回避）
        # NeMo は内部で datasets をインポートするが、PyInstaller の frozen importer では
        # datasets/__init__.py が完全に初期化される前に datasets.utils にアクセスしようとして
        # AttributeError が発生する。事前に datasets.utils をインポートすることで回避。
        # See: https://github.com/Mega-Gorilla/livecap-cli/issues/216
        try:
            import datasets.utils
            import datasets.utils.logging
            logger.debug("datasets サブモジュールを事前インポートしました")
        except ImportError as e:
            logger.debug(f"datasets 事前インポートをスキップ: {e}")
        except Exception as e:
            # datasets が部分的にインストールされている場合など
            logger.debug(f"datasets 事前インポート中に予期しないエラー: {e}")

        # librosa サブモジュールを NeMo より先にインポート（循環インポート回避）
        # NeMo → lightning.pytorch → torchmetrics → librosa の依存チェーンで、
        # librosa.filters と librosa.core.spectrum の間で循環依存が発生する。
        # 依存関係の順序でインポートすることで回避。
        # See: https://github.com/Mega-Gorilla/livecap-cli/issues/219
        try:
            import librosa.util
            import librosa.core.convert
            import librosa.filters  # get_window を定義
            import librosa.core.spectrum  # get_window を使用
            logger.debug("librosa サブモジュールを事前インポートしました")
        except ImportError as e:
            logger.debug(f"librosa 事前インポートをスキップ: {e}")
        except Exception as e:
            # librosa が部分的にインストールされている場合など
            logger.debug(f"librosa 事前インポート中に予期しないエラー: {e}")

    _NEMO_ENVIRONMENT_PREPARED = True
    logger.debug("NeMo 環境準備が完了しました")


class _NemoErrorRelay(logging.Handler):
    """``nemo_logger`` の ERROR record を app logger へ転送しつつ retain する。

    NeMo は具象クラス生成中の SentencePiece 例外を捕捉して基底クラスへ fallback するので、
    **最終例外の ``__cause__`` を辿っても元例外に到達できない** (Issue #379)。一次エラーは
    ``nemo_logger`` にだけ出るため、そこを拾うのが唯一の経路である。

    パスは ``ascii()`` で包む — 日本語 Windows では stderr がリダイレクト時に
    cp932 + strict になり、素のパスを出すと**ログ自体が UnicodeEncodeError で落ちる**。
    """

    def __init__(self, *, boundary: str, model_path: Path):
        super().__init__(level=logging.ERROR)
        self._boundary = boundary
        self._path = ascii(str(model_path))
        self.messages: list[str] = []

    @property
    def first_error(self) -> str | None:
        return self.messages[0] if self.messages else None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - record 側の不整合
            return
        self.messages.append(message)
        # 自分自身 (livecap_cli.engines.nemo_utils) へ出すので再入しない。
        logger.error(
            "%s: NeMo reported an error while restoring %s: %s",
            self._boundary,
            self._path,
            ascii(message),
        )


@contextmanager
def _nemo_error_relay(
    *, boundary: str, model_path: Path, quiet_loggers: Sequence[str]
) -> Iterator[_NemoErrorRelay | None]:
    """NeMo の警告を黙らせつつ、**ERROR だけは app log へ通す**。

    **``propagate`` が ``False`` のときだけ relay を付ける。** ``True`` なら root の
    handler が既に受け取っているので、転送すると**同じ内容が二重に出る**
    (Issue #379 の「二重出力しない」)。

    level / handler / ``propagate`` は**成功・例外の双方で** exact restore する。
    呼び出し側が ``ascii_safe_temp_environment()`` の内側で使う前提なので、
    **専用のロックは持たない** — ``_TEMP_ENV_LOCK`` がスコープ全期間保持されており、
    そこから直列化を継承する (ロックを 2 つ持つと deadlock の余地を作るだけ)。
    """
    saved_levels = [(logging.getLogger(name), logging.getLogger(name).level)
                    for name in quiet_loggers]
    for target_logger, _level in saved_levels:
        target_logger.setLevel(logging.ERROR)

    nemo_logger = logging.getLogger(NEMO_LOGGER_NAME)
    relay: _NemoErrorRelay | None = None
    if not nemo_logger.propagate:
        relay = _NemoErrorRelay(boundary=boundary, model_path=model_path)
        nemo_logger.addHandler(relay)

    try:
        yield relay
    finally:
        if relay is not None:
            nemo_logger.removeHandler(relay)
        for target_logger, level in saved_levels:
            target_logger.setLevel(level)


def restore_nemo_model(
    model_class: Any,
    model_path: Path,
    *,
    boundary: str,
    map_location: str,
    quiet_loggers: Sequence[str] = DEFAULT_QUIET_LOGGERS,
) -> Any:
    """ローカル ``.nemo`` から NeMo モデルを復元する共通経路 (Issue #379)。

    **本関数は ``%TEMP%`` を移設しない。** 呼び出し側の engine が
    ``ascii_safe_temp_environment(boundary=..., purpose="nemo-restore")`` で包む。
    boundary を引数で受けて中で開くと **boundary が動的値になり、棚卸し registry との
    AST 突き合わせ (``test_every_staging_call_is_registered``) が成立しない** —
    境界を決めているのは helper ではなく engine である。

    Args:
        model_class: ``restore_from`` を持つ NeMo のモデルクラス (engine 側で import 済み)。
        model_path: ローカルの ``.nemo``。**staging / copy はしない** — 元パスから直接読む
            (``.nemo`` 自体は wide path で通ることが #378 の A/B で確定している)。
        boundary: 棚卸し registry の ``boundary_id`` と同一文字列。**ログ用**であり、
            ``ascii_safe_*`` へは渡さない。
        map_location: ``restore_from`` へそのまま渡す。
        quiet_loggers: ERROR まで黙らせる logger 名。
    """
    with _nemo_error_relay(
        boundary=boundary, model_path=model_path, quiet_loggers=quiet_loggers
    ) as relay:
        try:
            return model_class.restore_from(
                restore_path=str(model_path),
                map_location=map_location,
            )
        except Exception:
            # **元例外を置換しない。** ``raise ... from exc`` で包むと、「抽象クラスの
            # 二次例外にすり替わる」という #379 の症状を別の形で作り直すことになる。
            # 診断はログ側で足す。
            # **一次エラーの本文をここで再掲しない。** relay が既に 1 record 出して
            # いるので、繰り返すと同じ内容が app log に 2 回並ぶ。relay が何も掴めて
            # いないとき (NeMo がログを出す前に落ちた場合など) だけ、その事実を書く。
            if relay is not None and relay.first_error:
                primary = "logged above by this boundary"
            elif relay is None:
                primary = "not relayed (nemo_logger propagates; see the root log)"
            else:
                primary = "not captured; NeMo logged nothing before failing"
            logger.error(
                "%s: NeMo failed to restore the model at %s. "
                "If %%TEMP%% is non-ASCII this is usually the SentencePiece model inside "
                "NeMo's own untar directory, not the .nemo path itself. "
                "Primary NeMo error: %s",
                boundary,
                ascii(str(model_path)),
                primary,
            )
            raise
