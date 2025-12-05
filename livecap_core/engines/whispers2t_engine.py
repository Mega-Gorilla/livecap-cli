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
from .whisper_languages import WHISPER_LANGUAGES, WHISPER_LANGUAGES_SET
from .metadata import EngineMetadata

# リソースパス解決用のヘルパー関数とデバイス検出関数をインポート
from livecap_core.utils import detect_device, get_temp_dir

logger = logging.getLogger(__name__)

# モデル識別子マッピング（WhisperS2Tの_MODELSにないモデルはHuggingFaceパスで指定）
MODEL_MAPPING = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large-v1": "large-v1",
    "large-v2": "large-v2",
    "large-v3": "large-v3",
    "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
}

VALID_MODEL_SIZES = frozenset(MODEL_MAPPING.keys())
VALID_COMPUTE_TYPES = frozenset({"auto", "int8", "int8_float16", "float16", "float32"})

# 128メルバンクが必要なモデル（v3ベース）
MODELS_REQUIRING_128_MELS = frozenset({"large-v3", "large-v3-turbo", "distil-large-v3"})

# CPU速度推定値
CPU_SPEED_ESTIMATES = {
    'base': '3-5x real-time',
    'large-v3': '0.1-0.3x real-time (VERY SLOW)',
    'large-v3-turbo': '~0.5x real-time',
}


class WhisperS2TEngine(BaseEngine):
    """WhisperS2T音声認識エンジン (Template Method版)"""

    def __init__(
        self,
        device: Optional[str] = None,
        # カテゴリA: ユーザー向けパラメータ（EngineMetadata.default_params で定義）
        language: str = "ja",
        model_size: str = "large-v3",  # benchmark互換性維持（旧whispers2t_large_v3がデフォルト）
        compute_type: str = "auto",
        batch_size: int = 24,
        use_vad: bool = True,
        **kwargs,
    ):
        """エンジンを初期化

        Args:
            device: 使用するデバイス（cpu/cuda/auto）
            language: 言語コード（ISO 639-1の2文字コード、または地域コード）
            model_size: モデルサイズ（tiny/base/small/medium/large-v1/large-v2/large-v3/large-v3-turbo/distil-large-v3）
            compute_type: 量子化タイプ（auto/int8/int8_float16/float16/float32）
            batch_size: バッチサイズ
            use_vad: VADを使用するかどうか
        """
        # 入力バリデーション
        if model_size not in VALID_MODEL_SIZES:
            raise ValueError(
                f"Unsupported model_size: {model_size}. "
                f"Valid options: {', '.join(sorted(VALID_MODEL_SIZES))}"
            )
        if compute_type not in VALID_COMPUTE_TYPES:
            raise ValueError(
                f"Unsupported compute_type: {compute_type}. "
                f"Valid options: {', '.join(sorted(VALID_COMPUTE_TYPES))}"
            )

        # 言語コードの変換とバリデーション
        # BCP-47 コード（zh-CN等）→ ISO 639-1 コード（zh等）への変換
        asr_language = EngineMetadata.to_iso639_1(language)

        # WHISPER_LANGUAGES_SET でバリデーション（O(1) lookup）
        if asr_language not in WHISPER_LANGUAGES_SET:
            raise ValueError(
                f"Unsupported language: {language}. "
                f"WhisperS2T supports 100 languages. See: https://github.com/openai/whisper"
            )

        # engine_name を統一（旧: f'whispers2t_{model_size}'）
        self.engine_name = "whispers2t"
        self.model_size = model_size
        self.language = language  # 元のコードを保持（ログ/デバッグ用）
        self._asr_language = asr_language  # 変換後のコード（transcribe()で使用）
        self.batch_size = batch_size
        self.use_vad = use_vad

        # cuDNN設定（GPU使用時の安定性向上）
        os.environ['CUDNN_DETERMINISTIC'] = '1'
        os.environ['CUDNN_BENCHMARK'] = '0'

        # デバイスの自動検出と設定（共通関数を使用）
        # detect_device() は Tuple[str, str] を返すため、最初の要素のみ使用
        # 注: #166 完了後は戻り値が str になる
        device_result = detect_device(device, "WhisperS2T")
        self.device = device_result[0] if isinstance(device_result, tuple) else device_result

        # compute_type の解決（autoの場合はデバイスに応じて最適化）
        self.compute_type = self._resolve_compute_type(compute_type)

        # 大型モデル使用時の警告
        if self.model_size in ('large-v3', 'large-v3-turbo', 'distil-large-v3') and self.device == 'cpu':
            speed_estimate = CPU_SPEED_ESTIMATES.get(self.model_size, 'SLOW')
            logger.warning(f"⚠️ WhisperS2T {self.model_size} on CPU will be {speed_estimate}! Consider using GPU or smaller model.")

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
            logger.info(f"WhisperS2T {model_size} engine initialized (CPU mode: {self.compute_type})")

    def _resolve_compute_type(self, compute_type: str) -> str:
        """compute_typeを解決（autoの場合はデバイスに応じて最適化）"""
        if compute_type != "auto":
            return compute_type  # ユーザー指定を尊重
        # auto: CPU→int8（1.5倍高速）、GPU→float16
        return "int8" if self.device == "cpu" else "float16"

    def _get_n_mels(self) -> int:
        """モデルサイズに応じた n_mels 値を取得"""
        return 128 if self.model_size in MODELS_REQUIRING_128_MELS else 80

    def _get_model_identifier(self) -> str:
        """モデルサイズを WhisperS2T 用の識別子に変換"""
        return MODEL_MAPPING.get(self.model_size, self.model_size)
    
    def get_model_metadata(self) -> Dict[str, Any]:
        """モデルメタデータを取得"""
        descriptions = {
            'tiny': 'Whisper Tiny - Fastest, lowest accuracy',
            'base': 'Whisper Base - Good balance',
            'small': 'Whisper Small - Better accuracy',
            'medium': 'Whisper Medium - High accuracy',
            'large-v1': 'Whisper Large v1 - Original large model',
            'large-v2': 'Whisper Large v2 - Improved large model',
            'large-v3': 'Whisper Large v3 - Best accuracy',
            'large-v3-turbo': 'Whisper Large v3 Turbo - 8x faster than v3',
            'distil-large-v3': 'Distil Whisper Large v3 - 6x faster, ~1% WER increase',
        }

        # バージョン判定
        if 'v3' in self.model_size:
            version = 'v3'
        elif 'v2' in self.model_size:
            version = 'v2'
        elif 'v1' in self.model_size:
            version = 'v1'
        else:
            version = 'v2'  # tiny/base/small/medium はv2相当

        return {
            'name': f'openai/whisper-{self.model_size}',
            'version': version,
            'format': 'ct2',
            'language': 'multilingual',
            'description': descriptions.get(self.model_size, f'Whisper {self.model_size}'),
            'model_size': self.model_size,
            'n_mels': self._get_n_mels(),
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

        # 大型モデル使用時のメモリ警告
        if self.model_size in ('large-v3', 'large-v3-turbo', 'distil-large-v3') and self.device == 'cpu':
            logger.warning("📊 WhisperS2T large model requires ~10GB system memory on CPU")

        self.report_progress(75, self.get_status_message("initializing_model", engine_name="WhisperS2T", model_name=self.model_size))

        # モデル識別子を取得（HuggingFaceパスへの変換）
        model_identifier = self._get_model_identifier()
        n_mels = self._get_n_mels()

        try:
            # WhisperS2Tモデルをロード（n_mels を明示的に指定）
            model = whisper_s2t.load_model(
                model_identifier=model_identifier,
                backend='CTranslate2',
                device=self.device,
                compute_type=self.compute_type,
                n_mels=n_mels,  # v3ベースモデルには128を指定
            )

            self.report_progress(85, self.get_status_message("initialization_success", engine_name="WhisperS2T"))

            # キャッシュに保存
            ModelMemoryCache.set(cache_key, model, strong=True)

            if self.device == 'cuda':
                logger.info(f"✅ WhisperS2T {self.model_size} loaded on GPU (n_mels={n_mels})")
            else:
                speed_estimate = CPU_SPEED_ESTIMATES.get(self.model_size, '')
                if speed_estimate:
                    logger.info(f"📊 WhisperS2T {self.model_size} on CPU: {speed_estimate}")

            self.report_progress(90, self.get_status_message("model_ready_simple", engine_name="WhisperS2T"))
            return model

        except Exception as e:
            if "cuDNN" in str(e) and self.device == 'cuda':
                logger.warning(f"cuDNN error detected, falling back to CPU: {e}")
                self.device = 'cpu'
                self.compute_type = 'int8'  # CPU fallback でも int8 を使用

                model = whisper_s2t.load_model(
                    model_identifier=model_identifier,
                    backend='CTranslate2',
                    device='cpu',
                    compute_type='int8',
                    n_mels=n_mels,
                )

                ModelMemoryCache.set(f"whispers2t_{self.model_size}_cpu_int8", model, strong=True)
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
            # 言語コードは __init__ で変換済み（_asr_language を使用）
            whisper_language = self._asr_language

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
        """エンジン名を取得（ユーザー向け表示用）"""
        size_map = {
            'tiny': 'Tiny',
            'base': 'Base',
            'small': 'Small',
            'medium': 'Medium',
            'large-v1': 'Large-v1',
            'large-v2': 'Large-v2',
            'large-v3': 'Large-v3',
            'large-v3-turbo': 'Large-v3 Turbo',
            'distil-large-v3': 'Distil Large-v3',
        }
        return f"WhisperS2T {size_map.get(self.model_size, self.model_size.title())}"

    def get_supported_languages(self) -> list:
        """サポートされる言語のリストを取得"""
        # WhisperS2Tは100言語対応
        return list(WHISPER_LANGUAGES)
        
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
