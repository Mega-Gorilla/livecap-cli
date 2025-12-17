"""NeMo フレームワーク用の共通ユーティリティ

Canary, Parakeet エンジンで共有される NeMo 関連の機能を提供。
"""
import os
import sys
import logging

logger = logging.getLogger(__name__)

# NeMo framework - 遅延インポート
NEMO_AVAILABLE = None  # 初期状態は未確認


def check_nemo_availability() -> bool:
    """NeMo の利用可能性をチェック（遅延実行）

    Returns:
        bool: NeMo が利用可能な場合 True
    """
    global NEMO_AVAILABLE
    if NEMO_AVAILABLE is not None:
        return NEMO_AVAILABLE

    try:
        # matplotlib backend issue を回避（Parakeet 用）
        import matplotlib
        matplotlib.use('Agg')  # 非対話的バックエンドを使用

        # PyInstaller 互換性のための JIT パッチを適用
        from . import nemo_jit_patch

        # PyInstaller 環境での追加設定
        if getattr(sys, 'frozen', False):
            # torch._dynamo を無効化
            os.environ['TORCHDYNAMO_DISABLE'] = '1'
            # TorchScript を無効化
            os.environ['PYTORCH_JIT'] = '0'

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
