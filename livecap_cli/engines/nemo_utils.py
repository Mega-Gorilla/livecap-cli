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
import importlib.util
import os
import sys
import logging

logger = logging.getLogger(__name__)

# NeMo framework - 遅延インポート
NEMO_AVAILABLE = None  # 初期状態は未確認
_NEMO_ENVIRONMENT_PREPARED = False  # 環境準備済みフラグ


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
