"""NeMo フレームワーク用の共通ユーティリティ

Canary, Parakeet エンジンで共有される NeMo 関連の機能を提供。

PyInstaller 互換性:
    PyInstaller (frozen) 環境では、NeMo をインポートすると datasets ライブラリの
    循環インポートエラーが発生するため、check_nemo_availability() では
    importlib.util.find_spec() で存在確認のみを行う。

    通常の Python 環境では、実際にインポートを試行して依存関係の問題も検出する。
    実際のインポートは prepare_nemo_environment() を呼び出した後、
    各エンジンの関数内で行う。
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

    _NEMO_ENVIRONMENT_PREPARED = True
    logger.debug("NeMo 環境準備が完了しました")
