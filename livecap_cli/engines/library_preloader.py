"""重いライブラリの事前ロード管理"""
import threading
import logging
import time
import sys
from typing import Dict, Optional, Set, Any

logger = logging.getLogger(__name__)


class LibraryPreloader:
    """
    重いライブラリの事前ロード
    
    NeMo、Transformers、WhisperS2Tなどの重いライブラリを
    バックグラウンドで事前にインポートすることで、
    実際の使用時のロード時間を短縮する。
    """
    
    # クラス変数
    _preload_thread: Optional[threading.Thread] = None
    _preloaded: Dict[str, bool] = {
        'nemo': False,
        'transformers': False,
        'whisper_s2t': False,
        'sherpa_onnx': False,
        'matplotlib': False,
    }
    _preload_in_progress: Set[str] = set()
    _lock = threading.Lock()
    _enabled = True  # 事前ロード機能の有効/無効
    
    @classmethod
    def enable(cls, enabled: bool = True):
        """事前ロード機能の有効/無効を設定"""
        cls._enabled = enabled
        logger.debug(f"事前ロード機能: {'有効' if enabled else '無効'}")
    
    @classmethod
    def start_preloading(cls, engine_type: str, force: bool = False):
        """
        バックグラウンドで事前ロード開始
        
        Args:
            engine_type: エンジンタイプ
            force: 既にロード済みでも強制的に再ロード
        """
        if not cls._enabled:
            logger.debug("事前ロード機能が無効化されています")
            return
        
        # 既に実行中の場合はスキップ
        if cls._preload_thread and cls._preload_thread.is_alive():
            logger.debug("事前ロードスレッドは既に実行中です")
            return
        
        # エンジンタイプに応じて必要なライブラリを決定
        required_libs = cls._get_required_libraries(engine_type)
        
        # 全て既にロード済みの場合はスキップ（forceフラグが無い限り）
        if not force and all(cls._preloaded.get(lib, False) for lib in required_libs):
            logger.debug(f"必要なライブラリは全てロード済み: {required_libs}")
            return
        
        # バックグラウンドスレッドで事前ロード開始
        cls._preload_thread = threading.Thread(
            target=cls._preload_libraries,
            args=(engine_type, required_libs),
            daemon=True,
            name="LibraryPreloader"
        )
        cls._preload_thread.start()
        logger.debug(f"事前ロード開始: {engine_type} ({required_libs})")
    
    @classmethod
    def _get_required_libraries(cls, engine_type: str) -> Set[str]:
        """
        エンジンタイプに応じて必要なライブラリを取得
        
        Args:
            engine_type: エンジンタイプ
            
        Returns:
            必要なライブラリのセット
        """
        library_map = {
            'parakeet': {'matplotlib', 'nemo'},
            'parakeet_ja': {'matplotlib', 'nemo'},
            'canary': {'matplotlib', 'nemo'},
            'voxtral': {'transformers'},
            'whispers2t': {'whisper_s2t'},  # Unified engine ID
            'reazonspeech': {'sherpa_onnx'},
        }
        
        return library_map.get(engine_type, set())
    
    @classmethod
    def _preload_libraries(cls, engine_type: str, required_libs: Set[str]):
        """
        ライブラリを事前ロード（バックグラウンドスレッドで実行）
        
        Args:
            engine_type: エンジンタイプ
            required_libs: ロードすべきライブラリのセット
        """
        start_time = time.time()
        
        try:
            # matplotlib（NeMo依存）
            if 'matplotlib' in required_libs and not cls._preloaded['matplotlib']:
                cls._preload_matplotlib()
            
            # NeMo
            if 'nemo' in required_libs and not cls._preloaded['nemo']:
                cls._preload_nemo()
            
            # Transformers
            if 'transformers' in required_libs and not cls._preloaded['transformers']:
                cls._preload_transformers()
            
            # WhisperS2T
            if 'whisper_s2t' in required_libs and not cls._preloaded['whisper_s2t']:
                cls._preload_whispers2t()
            
            # Sherpa-ONNX
            if 'sherpa_onnx' in required_libs and not cls._preloaded['sherpa_onnx']:
                cls._preload_sherpa_onnx()
            
            elapsed = time.time() - start_time
            logger.debug(f"事前ロード完了: {engine_type} ({elapsed:.2f}秒)")
            
        except Exception as e:
            logger.debug(f"事前ロード中のエラー（無視）: {e}")
    
    @classmethod
    def _preload_matplotlib(cls):
        """matplotlibを事前ロード"""
        with cls._lock:
            if 'matplotlib' in cls._preload_in_progress:
                return
            cls._preload_in_progress.add('matplotlib')
        
        try:
            import matplotlib
            matplotlib.use('Agg')  # 非対話的バックエンド
            cls._preloaded['matplotlib'] = True
            logger.debug("matplotlib事前ロード完了")
        except Exception as e:
            logger.debug(f"matplotlib事前ロード失敗: {e}")
        finally:
            with cls._lock:
                cls._preload_in_progress.discard('matplotlib')
    
    @classmethod
    def _preload_nemo(cls):
        """NeMoを事前ロード"""
        with cls._lock:
            if 'nemo' in cls._preload_in_progress:
                return
            cls._preload_in_progress.add('nemo')
        
        try:
            # PyInstaller環境での設定
            if getattr(sys, 'frozen', False):
                import os
                os.environ['TORCHDYNAMO_DISABLE'] = '1'
                os.environ['PYTORCH_JIT'] = '0'
            
            # NeMoのインポート
            import nemo.collections.asr
            cls._preloaded['nemo'] = True
            logger.debug("NeMo事前ロード完了")
        except Exception as e:
            logger.debug(f"NeMo事前ロード失敗: {e}")
        finally:
            with cls._lock:
                cls._preload_in_progress.discard('nemo')
    
    @classmethod
    def _preload_transformers(cls):
        """Transformersを事前ロード"""
        with cls._lock:
            if 'transformers' in cls._preload_in_progress:
                return
            cls._preload_in_progress.add('transformers')
        
        try:
            # 基本的なインポート
            import transformers
            
            # Voxtral用の特定モデルをインポート（可能な場合）
            try:
                from transformers import VoxtralForConditionalGeneration, AutoProcessor
                logger.debug("Voxtralモデルクラス事前ロード完了")
            except ImportError:
                # 古いバージョンの場合は無視
                pass
            
            cls._preloaded['transformers'] = True
            logger.debug("Transformers事前ロード完了")
        except Exception as e:
            logger.debug(f"Transformers事前ロード失敗: {e}")
        finally:
            with cls._lock:
                cls._preload_in_progress.discard('transformers')
    
    @classmethod
    def _preload_whispers2t(cls):
        """WhisperS2Tを事前ロード"""
        with cls._lock:
            if 'whisper_s2t' in cls._preload_in_progress:
                return
            cls._preload_in_progress.add('whisper_s2t')
        
        try:
            import whisper_s2t
            cls._preloaded['whisper_s2t'] = True
            logger.debug("WhisperS2T事前ロード完了")
        except Exception as e:
            logger.debug(f"WhisperS2T事前ロード失敗: {e}")
        finally:
            with cls._lock:
                cls._preload_in_progress.discard('whisper_s2t')
    
    @classmethod
    def _preload_sherpa_onnx(cls):
        """Sherpa-ONNXを事前ロード"""
        with cls._lock:
            if 'sherpa_onnx' in cls._preload_in_progress:
                return
            cls._preload_in_progress.add('sherpa_onnx')
        
        try:
            import sherpa_onnx
            cls._preloaded['sherpa_onnx'] = True
            logger.debug("Sherpa-ONNX事前ロード完了")
        except Exception as e:
            logger.debug(f"Sherpa-ONNX事前ロード失敗: {e}")
        finally:
            with cls._lock:
                cls._preload_in_progress.discard('sherpa_onnx')
    
    @classmethod
    def is_preloaded(cls, library: str) -> bool:
        """
        指定されたライブラリがロード済みかチェック
        
        Args:
            library: ライブラリ名
            
        Returns:
            ロード済みの場合True
        """
        return cls._preloaded.get(library, False)
    
    @classmethod
    def wait_for_preload(cls, timeout: float = 10.0) -> bool:
        """
        事前ロードの完了を待機
        
        Args:
            timeout: タイムアウト秒数
            
        Returns:
            完了した場合True、タイムアウトした場合False
        """
        if not cls._preload_thread:
            return True
        
        cls._preload_thread.join(timeout=timeout)
        return not cls._preload_thread.is_alive()
    
    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """
        事前ロードの統計を取得
        
        Returns:
            統計情報の辞書
        """
        with cls._lock:
            return {
                'enabled': cls._enabled,
                'preloaded': dict(cls._preloaded),
                'in_progress': list(cls._preload_in_progress),
                'thread_alive': cls._preload_thread.is_alive() if cls._preload_thread else False
            }
    
    @classmethod
    def reset(cls):
        """事前ロード状態をリセット"""
        with cls._lock:
            cls._preloaded = {key: False for key in cls._preloaded}
            cls._preload_in_progress.clear()
            cls._preload_thread = None
            logger.debug("事前ロード状態をリセット")