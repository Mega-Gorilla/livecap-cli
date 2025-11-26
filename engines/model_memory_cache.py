"""モデルインスタンスのメモリキャッシュ管理"""
from typing import Dict, Any, Optional
import weakref
import logging
import threading

logger = logging.getLogger(__name__)


class ModelMemoryCache:
    """
    モデルインスタンスのメモリキャッシュ
    
    弱参照と強参照の両方をサポートし、メモリ効率的なキャッシュを提供。
    エンジン切り替え時に前のモデルを保持することで、再ロードを回避。
    """
    
    # クラス変数（グローバルキャッシュ）
    _cache: Dict[str, weakref.ref] = {}
    _strong_refs: Dict[str, Any] = {}  # 強参照（オプション）
    _lock = threading.Lock()  # スレッドセーフ
    _access_count: Dict[str, int] = {}  # アクセス頻度追跡
    _cache_size_limit = 2  # 強参照の最大数
    _hit_count = 0  # キャッシュヒット数
    _miss_count = 0  # キャッシュミス数
    
    @classmethod
    def get(cls, cache_key: str) -> Optional[Any]:
        """
        キャッシュからモデルを取得
        
        Args:
            cache_key: キャッシュキー（通常は "engine_type_device" 形式）
            
        Returns:
            キャッシュされたモデルインスタンス、またはNone
        """
        with cls._lock:
            # アクセスカウントを更新
            cls._access_count[cache_key] = cls._access_count.get(cache_key, 0) + 1
            
            # 強参照チェック
            if cache_key in cls._strong_refs:
                logger.info(f"メモリキャッシュヒット（強参照）: {cache_key}")
                cls._hit_count += 1
                return cls._strong_refs[cache_key]
            
            # 弱参照チェック
            if cache_key in cls._cache:
                model_ref = cls._cache[cache_key]
                model = model_ref()
                if model is not None:
                    logger.info(f"メモリキャッシュヒット（弱参照）: {cache_key}")
                    cls._hit_count += 1
                    
                    # アクセス頻度が高い場合は強参照に昇格
                    if cls._access_count[cache_key] > 3:
                        cls._promote_to_strong_ref(cache_key, model)
                    
                    return model
                else:
                    # ガベージコレクトされた
                    logger.debug(f"弱参照がGCされました: {cache_key}")
                    del cls._cache[cache_key]
                    cls._access_count.pop(cache_key, None)
            
            # キャッシュミス
            cls._miss_count += 1
            return None
    
    @classmethod
    def set(cls, cache_key: str, model: Any, strong: bool = False):
        """
        モデルをキャッシュ

        Args:
            cache_key: キャッシュキー
            model: モデルインスタンス
            strong: 強参照で保持するか（デフォルト: False）
        """
        with cls._lock:
            if strong:
                cls._add_strong_ref(cache_key, model)
            else:
                # 弱参照として保存を試みる
                try:
                    cls._cache[cache_key] = weakref.ref(model)
                    logger.info(f"弱参照でキャッシュ: {cache_key}")
                except TypeError:
                    # tupleなど弱参照不可なオブジェクトは強参照で保存
                    logger.info(f"弱参照不可のため強参照でキャッシュ: {cache_key}")
                    cls._add_strong_ref(cache_key, model)

            # アクセスカウントを初期化
            cls._access_count[cache_key] = 1
    
    @classmethod
    def _add_strong_ref(cls, cache_key: str, model: Any):
        """強参照を追加（LRU方式）"""
        # サイズ制限チェック
        if len(cls._strong_refs) >= cls._cache_size_limit:
            # 最もアクセスが少ないものを削除（LRU）
            if cls._access_count:
                lru_key = min(cls._access_count.keys(), 
                            key=lambda k: cls._access_count.get(k, 0))
                if lru_key in cls._strong_refs:
                    logger.info(f"LRU削除: {lru_key}")
                    # 強参照から弱参照に降格
                    model_to_demote = cls._strong_refs.pop(lru_key)
                    cls._cache[lru_key] = weakref.ref(model_to_demote)
        
        cls._strong_refs[cache_key] = model
        logger.info(f"強参照でキャッシュ: {cache_key} (全{len(cls._strong_refs)}個)")
    
    @classmethod
    def _promote_to_strong_ref(cls, cache_key: str, model: Any):
        """弱参照を強参照に昇格"""
        if cache_key not in cls._strong_refs:
            logger.info(f"強参照に昇格: {cache_key}")
            cls._add_strong_ref(cache_key, model)
    
    @classmethod
    def clear(cls, cache_key: Optional[str] = None):
        """
        キャッシュをクリア
        
        Args:
            cache_key: 特定のキーのみクリア（Noneの場合は全クリア）
        """
        with cls._lock:
            if cache_key:
                cls._cache.pop(cache_key, None)
                cls._strong_refs.pop(cache_key, None)
                cls._access_count.pop(cache_key, None)
                logger.info(f"キャッシュクリア: {cache_key}")
            else:
                cls._cache.clear()
                cls._strong_refs.clear()
                cls._access_count.clear()
                logger.info("全キャッシュをクリア")
    
    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """
        キャッシュ統計を取得
        
        Returns:
            キャッシュの統計情報
        """
        with cls._lock:
            total_hits = getattr(cls, '_hit_count', 0)
            total_misses = getattr(cls, '_miss_count', 0)
            total_requests = total_hits + total_misses
            
            return {
                'hits': total_hits,
                'misses': total_misses,
                'hit_rate': total_hits / total_requests if total_requests > 0 else 0,
                'size': len(cls._cache) + len(cls._strong_refs),
                'weak_refs': len(cls._cache),
                'strong_refs': len(cls._strong_refs),
                'total_access': sum(cls._access_count.values()),
                'access_count': dict(cls._access_count),
                'cache_keys': list(cls._cache.keys()) + list(cls._strong_refs.keys())
            }
    
    @classmethod
    def set_size_limit(cls, limit: int):
        """
        強参照キャッシュのサイズ制限を設定
        
        Args:
            limit: 最大キャッシュ数
        """
        cls._cache_size_limit = max(1, limit)
        logger.info(f"キャッシュサイズ制限を{limit}に設定")
    
    @classmethod
    def exists(cls, cache_key: str) -> bool:
        """
        キャッシュが存在するかチェック
        
        Args:
            cache_key: キャッシュキー
            
        Returns:
            キャッシュが存在する場合True
        """
        with cls._lock:
            # 強参照チェック
            if cache_key in cls._strong_refs:
                return True
            
            # 弱参照チェック
            if cache_key in cls._cache:
                model_ref = cls._cache[cache_key]
                if model_ref() is not None:
                    return True
                else:
                    # GCされている場合は削除
                    del cls._cache[cache_key]
                    cls._access_count.pop(cache_key, None)
            
            return False