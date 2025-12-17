"""MistralAI Voxtral Mini 3Bエンジンの実装 - Template Method版

Note: VoxtralはTransformersの最新版（git+https://github.com/huggingface/transformers）が必要です。
また、mistral-common[audio]>=1.8.1も必要です。
"""
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import tempfile
import soundfile as sf

from .base_engine import BaseEngine
from .model_memory_cache import ModelMemoryCache
from .library_preloader import LibraryPreloader

# リソースパス解決用のヘルパー関数をインポート
from livecap_core.utils import get_models_dir, detect_device, unicode_safe_temp_directory, get_temp_dir

# Transformersの遅延インポート
TRANSFORMERS_AVAILABLE = None

def check_transformers_availability():
    """Transformersの利用可能性をチェック（遅延実行）"""
    global TRANSFORMERS_AVAILABLE
    if TRANSFORMERS_AVAILABLE is not None:
        return TRANSFORMERS_AVAILABLE
        
    try:
        import transformers
        # Voxtral用のクラスをチェック
        try:
            from transformers import VoxtralForConditionalGeneration, AutoProcessor
        except ImportError:
            # VoxtralForConditionalGenerationが見つからない場合
            logger = logging.getLogger(__name__)
            logger.warning("VoxtralForConditionalGeneration not found in current transformers version")
        
        # バージョンチェック
        version = transformers.__version__
        logging.info(f"Transformers version: {version}")
        
        TRANSFORMERS_AVAILABLE = True
        logging.info("Transformersが正常にインポートされました")
    except ImportError as e:
        TRANSFORMERS_AVAILABLE = False
        logger = logging.getLogger(__name__)
        logger.error(f"Transformersのインポートに失敗しました: {str(e)}")
        
    return TRANSFORMERS_AVAILABLE


class VoxtralEngine(BaseEngine):
    """MistralAI Voxtral Mini 3Bを使用した音声認識エンジン - Template Method版"""

    def __init__(
        self,
        device: Optional[str] = None,
        language: str = "auto",
        model_name: str = "mistralai/Voxtral-Mini-3B-2507",
        temperature: float = 0.0,
        do_sample: bool = False,
        max_new_tokens: int = 448,
        **kwargs,
    ):
        """エンジンを初期化

        Args:
            device: 使用するデバイス ("cpu", "cuda", None=auto)
            language: 入力言語 (en, es, fr, pt, hi, de, nl, it, または "auto")
            model_name: モデル名
            temperature: 生成時の温度パラメータ
            do_sample: サンプリングを使用するか
            max_new_tokens: 最大生成トークン数
            **kwargs: 追加パラメータ
        """
        # エンジン名を設定
        self.engine_name = 'voxtral'

        # Category A パラメータ（明示的）
        self.language = language
        self.model_name = model_name
        self.temperature = temperature
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens

        super().__init__(device, **kwargs)
        self.model = None
        self.processor = None

        # デバイスの自動検出と設定（共通関数を使用）
        self.torch_device = detect_device(device, "Voxtral")

        # GPU RAM警告
        if self.torch_device == "cuda":
            logging.info("Voxtral requires ~9.5GB GPU RAM. Ensure sufficient memory is available.")

        # ライブラリ事前ロードを開始
        LibraryPreloader.start_preloading('voxtral')
    
    # ===============================
    # Template Method実装
    # ===============================
    
    def get_model_metadata(self) -> Dict[str, Any]:
        """モデルのメタデータを返す"""
        return {
            'name': self.model_name,
            'version': 'v2507',
            'format': 'safetensors',
            'description': 'MistralAI Voxtral Mini 3B - 8-language ASR with auto-detection'
        }
    
    def _check_dependencies(self) -> None:
        """
        Step 1: 依存関係のチェック（0-10%）
        """
        self.report_progress(3, "Checking Transformers availability...")

        # Transformersの利用可能性をチェック（初回のみインポートが試行される）
        if not check_transformers_availability():
            logging.error("TRANSFORMERS_AVAILABLEがFalseのため、Transformersのインポートエラーを発生させます")
            raise ImportError(
                "Transformers is not installed. Please run: pip install transformers>=4.40.0"
            )

        self.report_progress(6, "Checking Voxtral classes...")

        # VoxtralForConditionalGenerationのチェック
        try:
            from transformers import VoxtralForConditionalGeneration, AutoProcessor
        except ImportError:
            # 古いバージョンのtransformersの場合
            logging.error("VoxtralForConditionalGeneration not found. Please install transformers from source: pip install git+https://github.com/huggingface/transformers")
            raise ImportError("Please install transformers from source: pip install git+https://github.com/huggingface/transformers")

        self.report_progress(8, "Checking mistral-common...")

        # mistral-commonの依存関係チェック
        try:
            import mistral_common
            version = getattr(mistral_common, '__version__', 'unknown')
            logging.info(f"mistral-common version: {version}")
        except ImportError:
            logging.error("mistral-common is not installed. Please install: pip install mistral-common[audio]>=1.8.1")
            raise ImportError("mistral-common[audio]>=1.8.1 is required for Voxtral. Please install it.")

        self.report_progress(10, "Dependencies check complete")
    
    def _get_local_model_path(self, models_dir: Path) -> Path:
        """ローカルモデルパスを取得 (Step 2 override)"""
        # モデルファイルのパス
        local_model_path = models_dir / f"{self.model_name.replace('/', '--')}"

        self.report_progress(15, f"Model path: {local_model_path.name}")
        return local_model_path
    
    def _download_model(self, model_path: Path, progress_callback, model_manager=None) -> None:
        """
        Step 3: モデルのダウンロード（15-70%）
        """
        if model_path.exists():
            self.report_progress(70, "Model already downloaded")
            logging.info(f"ローカルファイルが存在: {model_path}")
            return

        self.report_progress(20, f"Downloading model from Hugging Face: {self.model_name}")
        manager = model_manager or getattr(self, "model_manager", None)
        if manager is None:
            from livecap_core.resources import get_model_manager

            manager = get_model_manager()

        # ここで初めてTransformersモジュールをインポート
        from transformers import VoxtralForConditionalGeneration, AutoProcessor
        import torch

        # dtype設定（GPU/CPU最適化）
        torch_dtype = torch.float16 if self.torch_device == "cuda" else torch.float32

        try:
            self.report_progress(30, "Starting model download...")

            logging.info(f"Hugging Faceからモデルをダウンロード: {self.model_name}")

            with manager.huggingface_cache() as hf_cache:
                transformers_cache = hf_cache / "transformers"
                transformers_cache.mkdir(parents=True, exist_ok=True)

                model = VoxtralForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                    cache_dir=str(transformers_cache)
                )

                self.report_progress(50, "Downloading processor...")

                processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    cache_dir=str(transformers_cache)
                )

            self.report_progress(60, "Saving model locally...")

            logging.info(f"モデルをローカルに保存: {model_path}")
            model.save_pretrained(str(model_path))
            processor.save_pretrained(str(model_path))

            del model
            del processor

            self.report_progress(70, "Model download complete")

        except Exception as e:
            logging.error(f"モデルダウンロードエラー: {e}")
            raise
    
    def _load_model_from_path(self, model_path: Path) -> Any:
        """
        Step 4: モデルファイルからロード（70-90%）
        """
        self.report_progress(75, f"Loading model file: {model_path.name}")

        # キャッシュキーを生成
        cache_key = f"voxtral_{self.model_name.replace('/', '_')}_{self.torch_device}"

        # キャッシュから取得を試みる
        cached_result = ModelMemoryCache.get(cache_key)
        if cached_result is not None:
            self.report_progress(90, "Loading from cache: Voxtral")
            logging.info(f"キャッシュからモデルを取得: {cache_key}")
            # タプルとして返す（model, processor）
            return cached_result

        # Transformersモジュールをインポート
        from transformers import VoxtralForConditionalGeneration, AutoProcessor
        import torch

        # dtype設定（GPU/CPU最適化）
        torch_dtype = torch.float16 if self.torch_device == "cuda" else torch.float32

        try:
            self.report_progress(80, "Restoring Voxtral model...")

            # ローカルファイルからロード
            logging.info(f"ローカルファイルからモデルをロード: {model_path}")
            model = VoxtralForConditionalGeneration.from_pretrained(
                str(model_path),
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            ).to(self.torch_device)

            self.report_progress(85, "Loading processor...")

            processor = AutoProcessor.from_pretrained(str(model_path))

            # タプルとしてキャッシュに保存
            result = (model, processor)
            ModelMemoryCache.set(cache_key, result)
            logging.info(f"モデルをキャッシュに保存: {cache_key}")

            self.report_progress(90, "Voxtral: Ready")
            return result

        except Exception as e:
            logging.error(f"モデルロードエラー: {e}")
            raise
    
    def _configure_model(self) -> None:
        """
        Step 5: モデルの設定（90-100%）
        """
        self.report_progress(92, "Setting model to evaluation mode...")

        # self.modelは_load_model_from_pathでタプルとして設定されている
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # モデルとプロセッサを分離（タプルの場合）
        if isinstance(self.model, tuple):
            self.model, self.processor = self.model
        else:
            # 互換性のため単一モデルの場合も処理
            # processorは別途ロードが必要
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.model_name)

        # 評価モードに設定
        self.model.eval()

        self._initialized = True
        self.report_progress(100, "Voxtral model configuration complete")
        logging.info("モデルの設定が完了しました。")
    
    # ===============================
    # 既存のインターフェース実装
    # ===============================
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[str, float]:
        """
        音声データを文字起こしする
        
        Args:
            audio_data: 音声データ（numpy配列）
            sample_rate: サンプリングレート
            
        Returns:
            (transcription_text, confidence_score)のタプル
        """
        # Voxtralは30分まで処理可能
        duration = len(audio_data) / sample_rate
        if duration > 1800:  # 30分
            logging.warning(f"Voxtral: Audio duration {duration:.1f}s exceeds 30 minutes limit")
        
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
            
        # モデルが要求するサンプリングレートに変換
        required_sr = self.get_required_sample_rate()
        if sample_rate != required_sr:
            import librosa
            audio_data = librosa.resample(
                audio_data, 
                orig_sr=sample_rate, 
                target_sr=required_sr
            )
            
        # float32に変換し、正規化
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            
        # 音声データの正規化（-1.0 から 1.0の範囲）
        if np.abs(audio_data).max() > 1.0:
            audio_data = audio_data / np.abs(audio_data).max()
            
        # デバッグ: 音声データの情報
        logging.debug(f"Audio data shape: {audio_data.shape}")
        logging.debug(f"Audio duration: {len(audio_data) / self.get_required_sample_rate():.2f} seconds")
        logging.debug(f"Audio max amplitude: {np.abs(audio_data).max():.4f}")
        
        # 音声が短すぎる場合の処理
        min_duration = 0.1  # 最小0.1秒
        min_samples = int(min_duration * self.get_required_sample_rate())
        if len(audio_data) < min_samples:
            logging.warning(f"Audio too short: {len(audio_data)} samples < {min_samples} samples")
            return "", 1.0
            
        try:
            import torch
            
            # Unicode対策: models/tempディレクトリを使用
            temp_path = get_temp_dir() / f"voxtral_temp_{os.getpid()}.wav"
            try:
                sf.write(str(temp_path), audio_data, sample_rate)

                # apply_transcription_request を使用
                inputs = self.processor.apply_transcription_request(
                    language=self.language,
                    audio=str(temp_path),
                    model_id=self.model_name
                ).to(self.torch_device)

                # 生成設定（転写用の設定）
                generation_config = {
                    "max_new_tokens": self.max_new_tokens,
                }

                # temperatureとdo_sampleは転写時には使用しない（do_sample=Falseの場合、temperatureは無視される）
                if self.do_sample:
                    generation_config["do_sample"] = True
                    generation_config["temperature"] = self.temperature
                
                # 自動言語検出を有効にして転写
                with torch.no_grad():
                    predicted_ids = self.model.generate(
                        **inputs,
                        **generation_config
                    )
                
                # デコード - 入力部分を除外して出力のみをデコード
                transcription = self.processor.batch_decode(
                    predicted_ids[:, inputs.input_ids.shape[1]:], 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )[0]
                
            finally:
                # 一時ファイルを削除
                if temp_path.exists():
                    temp_path.unlink()
            
            # 文字列のクリーンアップ
            transcription = transcription.strip()

            logging.debug(f"Voxtral transcription: '{transcription}'")
                    
            # 空の結果をチェック
            if not transcription:
                logging.debug("Voxtral returned empty transcription")
                    
            # 信頼度スコア（Voxtralでは利用不可のため固定値）
            confidence = 1.0
            
            return transcription, confidence
                
        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            raise
            
    def get_engine_name(self) -> str:
        """エンジン名を取得"""
        return "MistralAI Voxtral Mini 3B"
        
    def get_supported_languages(self) -> list:
        """サポートされる言語のリストを取得"""
        # Voxtralは8言語をサポート（自動言語検出付き）
        return ["en", "es", "fr", "pt", "hi", "de", "nl", "it"]
        
    def get_required_sample_rate(self) -> int:
        """エンジンが要求するサンプリングレートを取得"""
        # Voxtralは16kHzを使用
        return 16000
        
    def cleanup(self) -> None:
        """リソースのクリーンアップ"""
        if self.model is not None:
            # GPUメモリを解放
            del self.model
            self.model = None
            
        if self.processor is not None:
            del self.processor
            self.processor = None
            
        if self.torch_device == "cuda":
            # 遅延インポート: 必要な時のみtorchをインポート
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                # torchがインポートできない場合は何もしない
                pass
        self._initialized = False
