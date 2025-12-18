"""音声認識エンジンパッケージ"""
from .base_engine import BaseEngine
from .engine_factory import EngineFactory
from .metadata import EngineMetadata, EngineInfo

# 動的インポートされるモジュールを明示的にインポート
# これにより、PyInstallerなどのバンドラーがこれらのモジュールを含めるようになる
try:
    from . import reazonspeech_engine
    from . import parakeet_engine
    from . import canary_engine
    from . import whispers2t_engine
    from . import voxtral_engine
except ImportError:
    # 開発環境では一部のエンジンがインストールされていない可能性があるため、エラーを無視
    pass

__all__ = ['BaseEngine', 'EngineFactory', 'EngineMetadata', 'EngineInfo']