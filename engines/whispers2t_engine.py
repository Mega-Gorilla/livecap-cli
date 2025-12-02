"""WhisperS2Tエンジンの実装 (Template Method版)"""
import os
import logging
import tempfile
import time
import soundfile as sf
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np

from .base_engine import BaseEngine
from .model_memory_cache import ModelMemoryCache
from .library_preloader import LibraryPreloader
from livecap_core.languages import Languages

# リソースパス解決用のヘルパー関数とデバイス検出関数をインポート
from livecap_core.utils import detect_device, get_temp_dir

logger = logging.getLogger(__name__)

# パフォーマンス推定値（クラス変数）
CPU_SPEED_ESTIMATES = {
    'base': '3-5x real-time',
    'large-v3': '0.1-0.3x real-time (VERY SLOW)'
}


class WhisperS2TEngine(BaseEngine):
    """WhisperS2T音声認識エンジン (Template Method版)"""

    def __init__(
        self,
        device: Optional[str] = None,
        # カテゴリA: ユーザー向けパラメータ（EngineMetadata.default_params で定義）
        language: str = "ja",
        model_size: str = "base",
        batch_size: int = 24,
        use_vad: bool = True,
        **kwargs,
    ):
        """エンジンを初期化"""
        # モデルサイズを設定してengine_nameを設定（BaseEngine初期化前に必要）
        self.engine_name = f'whispers2t_{model_size}'
        self.model_size = model_size
        self.language = language
        self.batch_size = batch_size
        self.use_vad = use_vad

        # cuDNN設定（GPU使用時の安定性向上）
        os.environ['CUDNN_DETERMINISTIC'] = '1'
        os.environ['CUDNN_BENCHMARK'] = '0'

        # デバイスの自動検出と設定（共通関数を使用）
        self.device, self.compute_type = detect_device(device, "WhisperS2T")

        # large-v3モデル使用時の警告
        if self.model_size == 'large-v3' and self.device == 'cpu':
            logger.warning("⚠️ WhisperS2T Large-v3 on CPU will be VERY SLOW! Consider using GPU or smaller model.")

        # BaseEngine初期化（get_model_metadata()が呼ばれる）
        # detect_deviceで取得した正しいdevice値を渡す（Noneではなく）
        super().__init__(self.device, **kwargs)

        # 事前ロード開始
        LibraryPreloader.start_preloading('whispers2t')

        # 固定の一時ディレクトリを設定
        self._tmp_dir = get_temp_dir("whispers2t")

        # プロファイリング設定（kwargs から取得、デフォルト False）
        self._enable_profiling = kwargs.get('profile', False)

        # 初期化完了メッセージ
        if self.device == 'cuda':
            logger.info(f"✅ WhisperS2T {model_size} engine initialized (GPU mode: {self.compute_type})")
        else:
            logger.info(f"WhisperS2T {model_size} engine initialized (CPU mode)")
    
    def get_model_metadata(self) -> Dict[str, Any]:
        """モデルメタデータを取得"""
        descriptions = {
            'base': 'Whisper Base - Good balance',
            'large-v3': 'Whisper Large v3 - Best accuracy'
        }

        return {
            'name': f'openai/whisper-{self.model_size}',
            'version': 'v3' if 'v3' in self.model_size else 'v2',
            'format': 'ct2',
            'language': 'multilingual',
            'description': descriptions.get(self.model_size, descriptions['base']),
            'model_size': self.model_size
        }
    
    def _check_dependencies(self) -> None:
        """依存関係チェック (Step 1: 0-10%)"""
        self.report_progress(5, self.get_status_message("checking_availability", engine_name="WhisperS2T"))
        LibraryPreloader.wait_for_preload(timeout=2.0)

        try:
            import whisper_s2t
        except ImportError:
            raise ImportError("WhisperS2T is not installed. Please run: pip install whisper-s2t")

        if self.device == 'cuda':
            try:
                import torch
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            except ImportError:
                pass

        self.report_progress(10, self.get_status_message("dependencies_complete"))
    
    def _get_local_model_path(self, models_dir: Path) -> Path:
        """ローカルモデルパスを取得 (Step 2: 10-15%)"""
        model_path = models_dir / f"whisper-{self.model_size}"
        self.report_progress(15, self.get_status_message("model_info", model_name=f"whisper-{self.model_size}"))
        return model_path

    def _is_model_cached(self, model_path: Path) -> bool:
        """WhisperS2Tは内部でモデルを自動管理するため、常にTrueを返す"""
        return True

    def _verify_model_integrity(self, model_path) -> bool:
        """WhisperS2Tはローカル実体に依存しないため常にTrue"""
        return True
    
    def _download_model(self, target_path: Path, progress_callback, model_manager=None) -> None:
        """モデルダウンロード (Step 3: 15-70%)"""
        self.report_progress(70, self.get_status_message("model_ready", engine_name="WhisperS2T", model_name=self.model_size))
    
    def _load_model_from_path(self, model_path: Path) -> Any:
        """モデルをファイルからロード (Step 4: 70-90%)"""
        import whisper_s2t

        # キャッシュキーを生成
        cache_key = f"whispers2t_{self.model_size}_{self.device}_{self.compute_type}"
        cached_model = ModelMemoryCache.get(cache_key)

        if cached_model is not None:
            logger.info(f"メモリキャッシュからモデルを取得: {cache_key}")
            self.report_progress(90, self.get_status_message("loading_from_memory_cache"))
            return cached_model

        if self.model_size == 'large-v3' and self.device == 'cpu':
            logger.warning("📊 WhisperS2T Large-v3 requires ~10GB system memory on CPU")

        self.report_progress(75, self.get_status_message("initializing_model", engine_name="WhisperS2T", model_name=self.model_size))
        
        try:
            # WhisperS2Tモデルをロード
            model = whisper_s2t.load_model(
                model_identifier=self.model_size,
                backend='CTranslate2',
                device=self.device,
                compute_type=self.compute_type
            )
            
            self.report_progress(85, self.get_status_message("initialization_success", engine_name="WhisperS2T"))

            # キャッシュに保存
            ModelMemoryCache.set(cache_key, model, strong=True)

            if self.device == 'cuda':
                logger.info(f"✅ WhisperS2T {self.model_size} loaded on GPU")
            elif self.model_size == 'large-v3':
                logger.info(f"📊 WhisperS2T {self.model_size} on CPU: {CPU_SPEED_ESTIMATES.get(self.model_size)}")

            self.report_progress(90, self.get_status_message("model_ready_simple", engine_name="WhisperS2T"))
            return model
            
        except Exception as e:
            if "cuDNN" in str(e) and self.device == 'cuda':
                logger.warning(f"cuDNN error detected, falling back to CPU: {e}")
                self.device = 'cpu'
                self.compute_type = 'float32'

                model = whisper_s2t.load_model(
                    model_identifier=self.model_size,
                    backend='CTranslate2',
                    device='cpu',
                    compute_type='float32'
                )

                ModelMemoryCache.set(f"whispers2t_{self.model_size}_cpu_float32", model, strong=True)
                self.report_progress(90, self.get_status_message("model_ready_cpu_mode", engine_name="WhisperS2T"))
                return model
            else:
                logger.error(f"Failed to load WhisperS2T model: {e}")
                raise
    
    def _configure_model(self) -> None:
        """モデル設定 (Step 5: 90-100%)"""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        self.report_progress(95, self.get_status_message("final_configuration", engine_name="WhisperS2T"))

        logger.info(f"WhisperS2T {self.model_size} initialized")

        self.report_progress(100, self.get_status_message("initialization_complete", engine_name="WhisperS2T"))
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[str, float]:
        """
        音声データを文字起こしする

        Args:
            audio_data: 音声データ（numpy配列）
            sample_rate: サンプリングレート

        Returns:
            (transcription_text, confidence_score)のタプル
        """
        # WhisperS2Tは長時間音声も処理可能
        # 環境変数切替は不要（固定ディレクトリを使用）
        return self._transcribe_single_chunk(audio_data, sample_rate)
    
    def _transcribe_single_chunk(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[str, float]:
        """
        単一の音声チャンクを文字起こしする（内部使用）

        Args:
            audio_data: 音声データ（numpy配列）
            sample_rate: サンプリングレート

        Returns:
            (transcription_text, confidence_score)のタプル
        """
        if not self._initialized or self.model is None:
            raise RuntimeError("Engine not initialized. Call load_model() first.")

        # プロファイリング開始
        if self._enable_profiling:
            profile_times = {}
            total_start = time.perf_counter()
            
        # 16kHzに変換
        required_sr = 16000
        if sample_rate != required_sr:
            if self._enable_profiling:
                resample_start = time.perf_counter()

            from scipy.signal import resample_poly
            from math import gcd

            g = gcd(sample_rate, required_sr)
            audio_data = resample_poly(audio_data, required_sr // g, sample_rate // g).astype(np.float32)

            if self._enable_profiling:
                profile_times['resample'] = (time.perf_counter() - resample_start) * 1000
            
        # 正規化
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        if np.abs(audio_data).max() > 1.0:
            audio_data = audio_data / np.abs(audio_data).max()
            
        
        # 音声が短すぎる場合の処理
        min_samples = int(0.1 * 16000)  # 最小0.1秒
        if len(audio_data) < min_samples:
            return "", 1.0
            
        try:
            # 入力言語を取得（self.language を使用）
            input_language = self.language

            # WhisperS2T用の言語コードに変換
            # WhisperS2Tは'zh-CN'や'zh-TW'を受け付けず、'zh'のみをサポート
            lang_info = Languages.get_info(input_language)
            if lang_info:
                # Languages.pyのasr_codeを使用（zh-CN/zh-TW → zh への変換が定義済み）
                whisper_language = lang_info.asr_code
            else:
                # 言語情報が見つからない場合は元のコードを使用（後方互換性）
                whisper_language = input_language

            # 一時ファイルを作成
            if self._enable_profiling:
                io_start = time.perf_counter()

            with tempfile.NamedTemporaryFile(dir=self._tmp_dir, suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                sf.write(tmp_path, audio_data, 16000)

            if self._enable_profiling:
                profile_times['wav_write'] = (time.perf_counter() - io_start) * 1000

            try:
                if self._enable_profiling:
                    inference_start = time.perf_counter()

                # WhisperS2Tで文字起こし
                if self.use_vad:
                    outputs = self.model.transcribe_with_vad(
                        [tmp_path],
                        lang_codes=[whisper_language],
                        tasks=["transcribe"],
                        initial_prompts=[None],
                        batch_size=self.batch_size
                    )
                else:
                    outputs = self.model.transcribe(
                        [tmp_path],
                        lang_codes=[whisper_language],
                        tasks=["transcribe"],
                        initial_prompts=[None],
                        batch_size=self.batch_size
                    )

                if self._enable_profiling:
                    profile_times['inference'] = (time.perf_counter() - inference_start) * 1000
            finally:
                # 一時ファイルを削除
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            
            # 結果を取得
            if outputs and len(outputs) > 0:
                if isinstance(outputs[0], list) and len(outputs[0]) > 0:
                    result = outputs[0][0]
                else:
                    result = outputs[0]
                
                if isinstance(result, dict):
                    text = result.get('text', '').strip()
                    
                    # 信頼度スコアの計算
                    confidence = 1.0
                    if 'segments' in result and isinstance(result['segments'], list) and len(result['segments']) > 0:
                        total_logprob = 0
                        segment_count = 0
                        for segment in result['segments']:
                            if isinstance(segment, dict) and 'avg_logprob' in segment:
                                total_logprob += segment['avg_logprob']
                                segment_count += 1
                        
                        if segment_count > 0:
                            avg_logprob = total_logprob / segment_count
                            confidence = np.exp(avg_logprob)
                elif isinstance(result, str):
                    text = result.strip()
                    confidence = 1.0
                else:
                    text = str(result) if result else ""
                    confidence = 1.0
                
                # プロファイリング結果を出力
                if self._enable_profiling:
                    self._log_profiling_results(profile_times, total_start, audio_data)

                return text, confidence
            else:
                return "", 1.0
                
        except RuntimeError as e:
            if "cuDNN" in str(e) and self.device == 'cuda':
                logger.warning(f"cuDNN error, retrying with CPU: {e}")

                cpu_cache_key = f"whispers2t_{self.model_size}_cpu_float32"
                cpu_model = ModelMemoryCache.get(cpu_cache_key)

                if cpu_model is None:
                    import whisper_s2t
                    cpu_model = whisper_s2t.load_model(
                        model_identifier=self.model_size,
                        backend='CTranslate2',
                        device='cpu',
                        compute_type='float32'
                    )
                    ModelMemoryCache.set(cpu_cache_key, cpu_model, strong=True)

                original_model, original_device = self.model, self.device
                self.model, self.device = cpu_model, 'cpu'

                try:
                    result = self.transcribe(audio_data, sample_rate)
                finally:
                    self.model, self.device = original_model, original_device

                return result
            else:
                logger.error(f"Error during transcription: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise
            
    def _log_profiling_results(self, profile_times: Dict, start_time: float, audio_data: np.ndarray) -> None:
        """プロファイリング結果をログ出力"""
        total_time = (time.perf_counter() - start_time) * 1000
        profile_times['total'] = total_time
        audio_duration = len(audio_data) / 16000

        logger.info("=== WhisperS2T Performance Profile ===")
        for key, ms in profile_times.items():
            if key != 'total':
                percentage = (ms / total_time) * 100 if total_time > 0 else 0
                logger.info(f"  {key:12s}: {ms:6.1f}ms ({percentage:4.1f}%)")
        logger.info(f"  {'='*30}")
        logger.info(f"  {'Total':12s}: {total_time:6.1f}ms")
        logger.info(f"  Audio duration: {audio_duration:.2f}s")
        logger.info(f"  Real-time factor: {total_time / 1000 / audio_duration:.2f}x")

    def get_engine_name(self) -> str:
        """エンジン名を取得"""
        size_map = {
            'base': 'Base',
            'large-v3': 'Large-v3'
        }
        return f"WhisperS2T {size_map.get(self.model_size, self.model_size.title())}"
        
    def get_supported_languages(self) -> list:
        """サポートされる言語のリストを取得"""
        # WhisperS2Tは多言語対応
        return ["ja", "en", "zh", "ko", "es", "fr", "de", "ru", "ar", "pt", "it", "hi"]
        
    def get_required_sample_rate(self) -> int:
        """エンジンが要求するサンプリングレートを取得"""
        return 16000
        
    def cleanup(self) -> None:
        """リソースのクリーンアップ"""
        if self.model is not None:
            del self.model
            self.model = None

            if self.device == "cuda":
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass
        self._initialized = False
