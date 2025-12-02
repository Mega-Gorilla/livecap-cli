"""
改善版共有エンジンマネージャー

Template Methodパターンの進捗報告機能を統合し、
モデルロード進捗をリアルタイムで報告する。
"""

import threading
import logging
import queue
from typing import Dict, Any, Optional, Tuple, Callable, Protocol
import numpy as np
import time
from dataclasses import dataclass

from .model_memory_cache import ModelMemoryCache
from .library_preloader import LibraryPreloader
from livecap_core import create_transcription_event, validate_event_dict

logger = logging.getLogger(__name__)


class ProgressCallback(Protocol):
    """進捗報告コールバックのプロトコル定義"""
    def __call__(self, percent: int, message: str = "") -> None: ...


@dataclass
class TranscriptionRequest:
    """文字起こしリクエスト"""
    source_id: str
    audio: np.ndarray
    timestamp: float
    sample_rate: int = 16000
    priority: int = 0
    callback: Optional[Callable] = None
    is_final: bool = True  # Whether this is a final or intermediate segment
    
    def __lt__(self, other):
        """優先度比較（PriorityQueueのため）"""
        if not isinstance(other, TranscriptionRequest):
            return NotImplemented
        
        # 優先度で比較
        if self.priority != other.priority:
            return self.priority < other.priority
        
        # タイムスタンプで比較
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        
        # 同じ場合はsource_idで比較（numpy配列の比較を避ける）
        return self.source_id < other.source_id


class SharedEngineManager:
    """
    改善版：Template Method進捗報告統合
    
    主な改善点:
    - エンジンの進捗報告をリアルタイムで転送
    - ModelMemoryCacheの統計情報を活用
    - LibraryPreloaderとの連携
    """
    
    def __init__(
        self,
        engine_type: str,
        *,
        device: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
        **engine_options: Any
    ):
        """
        Args:
            engine_type: 使用するエンジンタイプ（必須、'auto'は非推奨）
            device: 使用するデバイス ('cpu', 'cuda', None=自動選択)
            progress_callback: 進捗報告用のコールバック関数 (percent: int, message: str) -> None
            **engine_options: エンジン固有のオプション
        """
        if engine_type == "auto":
            raise ValueError(
                "engine_type='auto' is deprecated. Use EngineMetadata.get_engines_for_language() "
                "to discover engines for a language, then specify the engine explicitly."
            )
        self.engine_type = engine_type
        self.device = device
        self.engine_options = engine_options
        self.progress_callback = progress_callback
        
        # エンジンインスタンス（単一）
        self.engine = None
        self.engine_lock = threading.Lock()
        
        # リクエストキュー（優先度付き）
        self.request_queue = queue.PriorityQueue()
        self.request_counter = 0  # ユニークなカウンター
        
        # 処理スレッド
        self.processing_thread = None
        self.is_running = False
        
        # モデルロードスレッド
        self.model_loader_thread = None
        self.model_loading = False
        self.model_load_complete = threading.Event()
        
        # 統計情報
        self.stats = {
            'total_requests': 0,
            'successful_transcriptions': 0,
            'failed_transcriptions': 0,
            'total_processing_time': 0.0,
            'source_stats': {},
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # ソース優先度管理
        self.source_priorities = {}
        
        logger.info(f"SharedEngineManager initialized with engine type: {engine_type}")
    
    def start(self) -> bool:
        """エンジンを初期化して処理を開始"""
        try:
            # ライブラリ事前ロードを開始
            self._start_library_preload()
            
            # モデルロードを非同期で開始
            self.model_loading = True
            self.model_load_complete.clear()
            self.model_loader_thread = threading.Thread(
                target=self._load_model_async,
                name="ModelLoader",
                daemon=True
            )
            self.model_loader_thread.start()
            
            # 処理スレッドを開始（モデルロード完了を待つ）
            if not self.is_running:
                self.is_running = True
                self.processing_thread = threading.Thread(
                    target=self._processing_loop,
                    name="SharedEngineProcessor",
                    daemon=True
                )
                self.processing_thread.start()
                logger.info("Processing thread started")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start SharedEngineManager: {e}")
            return False
    
    def stop(self):
        """エンジンを終了"""
        logger.info("Stopping SharedEngineManager...")
        
        # 処理スレッドを停止
        self.is_running = False
        self.model_loading = False
        
        # キューに終了シグナルを送信
        self.request_queue.put((-1, 0, None))
        
        # スレッドの終了を待機
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        if self.model_loader_thread and self.model_loader_thread.is_alive():
            self.model_loader_thread.join(timeout=5.0)
        
        # エンジンのクリーンアップ
        with self.engine_lock:
            if self.engine:
                try:
                    if hasattr(self.engine, 'cleanup'):
                        self.engine.cleanup()
                except Exception as e:
                    logger.error(f"Error during engine cleanup: {e}")
                finally:
                    self.engine = None
        
        logger.info("SharedEngineManager stopped")
    
    def transcribe_async(
        self,
        audio: np.ndarray,
        source_id: str,
        priority: int = 0,
        callback: Optional[Callable] = None,
        sample_rate: Optional[int] = None,
        is_final: bool = True
    ) -> bool:
        """非同期で文字起こしリクエストを送信"""
        if not self.is_running:
            logger.warning("SharedEngineManager is not running")
            return False
        
        # モデルロード完了を待つ（非ブロッキング）
        if not self.model_load_complete.is_set():
            logger.info("Model is still loading, request queued")
        
        try:
            # 優先度を取得
            if source_id in self.source_priorities:
                priority = self.source_priorities[source_id]
            
            # サンプルレートを決定（デフォルト 16kHz）
            if sample_rate is None:
                sample_rate = 16000
            
            # リクエストを作成
            request = TranscriptionRequest(
                source_id=source_id,
                audio=audio,
                timestamp=time.time(),
                sample_rate=sample_rate,
                priority=priority,
                callback=callback,
                is_final=is_final
            )
            
            # キューに追加（カウンターを使ってユニーク性を完全に保証）
            # 優先度が同じ場合でもカウンターでユニークになる
            self.request_counter += 1
            self.request_queue.put((priority, self.request_counter, request))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit transcription request: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得（キャッシュ情報付き）"""
        with self.engine_lock:
            avg_time = 0.0
            if self.stats['successful_transcriptions'] > 0:
                avg_time = self.stats['total_processing_time'] / self.stats['successful_transcriptions']
            
            # ModelMemoryCacheの統計を追加
            cache_stats = ModelMemoryCache.get_stats()
            
            return {
                'total_requests': self.stats['total_requests'],
                'successful': self.stats['successful_transcriptions'],
                'failed': self.stats['failed_transcriptions'],
                'average_processing_time': avg_time,
                'queue_size': self.request_queue.qsize(),
                'engine_type': self.engine_type,
                'is_running': self.is_running,
                'model_loading': self.model_loading,
                'model_loaded': self.model_load_complete.is_set(),
                'source_stats': dict(self.stats['source_stats']),
                'cache': {
                    'hits': self.stats['cache_hits'],
                    'misses': self.stats['cache_misses'],
                    'weak_refs': cache_stats['weak_refs'],
                    'strong_refs': cache_stats['strong_refs']
                }
            }
    
    def set_priority(self, source_id: str, priority: int):
        """ソースの優先度を設定"""
        self.source_priorities[source_id] = max(0, priority)
        logger.debug(f"Set priority for {source_id}: {priority}")
    
    def _start_library_preload(self):
        """ライブラリの事前ロードを開始"""
        # エンジンタイプから必要なライブラリを判定
        engine_base = self.engine_type.split('_')[0]  # whispers2t_base -> whispers2t
        
        logger.info(f"Starting library preload for {engine_base}")
        LibraryPreloader.start_preloading(engine_base)
    
    def _load_model_async(self):
        """非同期でモデルをロード"""
        try:
            with self.engine_lock:
                if self.engine is None:
                    # エンジンの作成
                    self.engine = self._create_engine_with_progress()
                    
                    if self.engine is None:
                        logger.error("Failed to create engine")
                        return
                    
                    logger.info(f"Engine initialized successfully: {self.engine_type}")
            
            # モデルロード完了を通知
            self.model_load_complete.set()
            self.model_loading = False
            
            # 最終進捗報告
            if self.progress_callback:
                self.progress_callback(100, "モデルロード完了")
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            self.model_loading = False
            
            # エラー時の進捗報告
            if self.progress_callback:
                self.progress_callback(-1, f"モデルロード失敗: {e}")
    
    def _create_engine_with_progress(self):
        """進捗報告付きでエンジンを作成"""
        try:
            logger.info(f"Creating engine: {self.engine_type}")

            # デバイス設定を決定
            device = self.device
            if device is None or device == 'null':
                # 自動選択
                try:
                    import torch
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                except ImportError:
                    device = 'cpu'

            # EngineFactoryを使用してエンジンを作成
            from engines.engine_factory import EngineFactory
            engine = EngineFactory.create_engine(
                engine_type=self.engine_type,
                device=device,
                **self.engine_options
            )
            
            # 進捗コールバックを設定（Template Method対応）
            if hasattr(engine, 'set_progress_callback'):
                def progress_wrapper(percent: int, message: str = "") -> None:
                    """エンジンからの進捗報告を転送"""
                    # SharedEngineManagerの進捗コールバックに転送
                    if self.progress_callback:
                        # エンジンタイプをプレフィックスとして追加
                        full_message = f"[{self.engine_type}] {message}" if message else f"[{self.engine_type}]"
                        self.progress_callback(percent, full_message)
                    
                    # キャッシュヒット/ミスの統計更新
                    if message and "キャッシュ" in message:
                        if "取得" in message or "ヒット" in message or "から" in message:
                            self.stats['cache_hits'] += 1
                        else:
                            self.stats['cache_misses'] += 1
                
                engine.set_progress_callback(progress_wrapper)
            
            # モデルをロード（Template Methodが進捗を報告）
            if hasattr(engine, 'load_model'):
                logger.info("Loading model with progress reporting...")
                engine.load_model()
                logger.info("Model loaded successfully")
            
            return engine
            
        except Exception as e:
            logger.error(f"Failed to create engine: {e}")
            return None
    
    # _get_engine_class メソッドは削除されました
    # EngineFactory.create_engine を使用するようにリファクタリング済み
    
    def _processing_loop(self):
        """処理ループ（モデルロード完了を待つ）"""
        logger.info("Processing loop started")
        
        # モデルロード完了を待つ
        logger.info("Waiting for model to load...")
        self.model_load_complete.wait()
        logger.info("Model loaded, starting request processing")
        
        while self.is_running:
            try:
                # キューからリクエストを取得
                item = self.request_queue.get(timeout=0.5)
                
                # タプルからリクエストを抽出
                if len(item) == 3:
                    priority, counter, request = item
                else:
                    # 互換性のための旧形式対応
                    priority, request = item
                
                # 終了シグナルチェック
                if request is None:
                    break
                
                # 統計情報更新
                self.stats['total_requests'] += 1
                
                # エンジンで文字起こし実行
                start_time = time.time()
                result = self._process_request(request)
                processing_time = time.time() - start_time
                
                # 統計情報更新
                if result:
                    self.stats['successful_transcriptions'] += 1
                    self.stats['total_processing_time'] += processing_time
                    
                    # ソース別統計
                    if request.source_id not in self.stats['source_stats']:
                        self.stats['source_stats'][request.source_id] = {
                            'count': 0,
                            'total_time': 0.0
                        }
                    self.stats['source_stats'][request.source_id]['count'] += 1
                    self.stats['source_stats'][request.source_id]['total_time'] += processing_time
                    
                    # コールバック実行
                    if request.callback:
                        try:
                            request.callback(result)
                        except Exception as e:
                            logger.error(f"Callback execution error: {e}")
                else:
                    self.stats['failed_transcriptions'] += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                self.stats['failed_transcriptions'] += 1
        
        logger.info("Processing loop ended")
    
    def _process_request(self, request: TranscriptionRequest) -> Optional[Dict[str, Any]]:
        """リクエストを処理"""
        with self.engine_lock:
            if not self.engine:
                logger.error("Engine not available")
                return None
            
            try:
                # エンジンで文字起こし実行
                result = self.engine.transcribe(request.audio, request.sample_rate)
                
                if result:
                    # タプル形式 (text, confidence) を処理
                    if isinstance(result, tuple) and len(result) >= 2:
                        text, confidence = result[0], result[1]
                        # 統一フォーマットでイベントを作成
                        event_dict = create_transcription_event(
                            text=text,
                            source_id=request.source_id,
                            is_final=request.is_final,  # Use is_final from request
                            timestamp=request.timestamp,
                            confidence=confidence
                        )
                        
                        # バリデーションによる早期検証（デバッグモード）
                        if logger.isEnabledFor(logging.DEBUG):
                            if validate_event_dict(event_dict):
                                logger.debug(f"[SEM] Valid transcription event created for {request.source_id}")
                            else:
                                logger.warning(f"[SEM] Invalid event dict created: {event_dict}")
                        
                        return event_dict
                    # 辞書形式の場合
                    elif isinstance(result, dict):
                        # 統一フォーマットでイベントを作成
                        event_dict = create_transcription_event(
                            text=result.get('text', ''),
                            source_id=request.source_id,
                            is_final=request.is_final,  # Use is_final from request
                            timestamp=request.timestamp,
                            confidence=result.get('confidence', 1.0),
                            language=result.get('language')  # 言語コードがあれば含める
                        )
                        
                        # バリデーションによる早期検証（デバッグモード）
                        if logger.isEnabledFor(logging.DEBUG):
                            if validate_event_dict(event_dict):
                                logger.debug(f"[SEM] Valid transcription event created for {request.source_id}")
                            else:
                                logger.warning(f"[SEM] Invalid event dict created: {event_dict}")
                        
                        return event_dict
                
                return None
                
            except Exception as e:
                logger.error(f"Transcription error for {request.source_id}: {e}")
                return None
    
