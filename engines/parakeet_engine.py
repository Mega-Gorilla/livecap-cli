"""NVIDIA Parakeet TDT 0.6B v3エンジンの実装"""
import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import platform
import warnings
import tempfile
import soundfile as sf

# Windows互換性のための設定
if platform.system() == 'Windows':
    # NeMoがSIGKILLを使用しようとするのを防ぐ
    import signal
    if not hasattr(signal, 'SIGKILL'):
        signal.SIGKILL = signal.SIGTERM
    
    # Windowsでの警告を抑制
    os.environ['NEMO_SUPPRESS_WARNINGS'] = '1'
    
# pydubのffmpeg警告を抑制
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub")

from .base_engine import BaseEngine
from .model_memory_cache import ModelMemoryCache

# リソースパス解決用のヘルパー関数をインポート
from livecap_core.utils import (
    get_models_dir,
    detect_device,
    unicode_safe_temp_directory,
    unicode_safe_download_directory,
)

# NeMo framework - 遅延インポートに変更
NEMO_AVAILABLE = None  # 初期状態は未確認

def check_nemo_availability():
    """NeMoの利用可能性をチェック（遅延実行）"""
    global NEMO_AVAILABLE
    if NEMO_AVAILABLE is not None:
        return NEMO_AVAILABLE
        
    try:
        # matplotlib backend issueを回避
        import matplotlib
        matplotlib.use('Agg')  # 非対話的バックエンドを使用
        
        # PyInstaller互換性のためのJITパッチを適用
        from . import nemo_jit_patch
        
        # PyInstaller環境での追加設定
        if getattr(sys, 'frozen', False):
            # torch._dynamoを無効化
            os.environ['TORCHDYNAMO_DISABLE'] = '1'
            # TorchScriptを無効化
            os.environ['PYTORCH_JIT'] = '0'
        
        import nemo.collections.asr
        NEMO_AVAILABLE = True
        logging.info("NVIDIA NeMoが正常にインポートされました")
    except (ImportError, AttributeError) as e:
        NEMO_AVAILABLE = False
        # NeMoが利用できない場合は、詳細エラーを記録
        logger = logging.getLogger(__name__)
        logger.error(f"NVIDIA NeMoのインポートに失敗しました: {str(e)}")
        logger.error(f"Import error details: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        
    return NEMO_AVAILABLE


class ParakeetEngine(BaseEngine):
    """NVIDIA Parakeet TDT/CTC モデルを使用した音声認識エンジン

    英語版: nvidia/parakeet-tdt-0.6b-v2 (TDT)
    日本語版: nvidia/parakeet-tdt_ctc-0.6b-ja (CTC)
    """

    # モデル名マッピング（定数）
    MODEL_MAPPING = {
        'parakeet': 'nvidia/parakeet-tdt-0.6b-v2',      # 英語モデル
        'parakeet_ja': 'nvidia/parakeet-tdt_ctc-0.6b-ja' # 日本語モデル
    }

    def __init__(
        self,
        device: Optional[str] = None,
        engine_name: str = "parakeet",
        model_name: Optional[str] = None,
        decoding_strategy: str = "greedy",
        **kwargs,
    ):
        """エンジンを初期化

        Args:
            device: 使用するデバイス ("cpu", "cuda", None=auto)
            engine_name: エンジン名 ("parakeet" or "parakeet_ja")
            model_name: モデル名 (None=engine_nameに応じたデフォルト)
            decoding_strategy: デコード戦略 ("greedy")
            **kwargs: 追加パラメータ
        """
        # エンジン名を設定
        self.engine_name = engine_name

        # Category A パラメータ（明示的）
        # model_nameがNoneの場合はMODEL_MAPPINGからデフォルト値を取得
        if model_name is None:
            self.model_name = self.MODEL_MAPPING.get(engine_name, self.MODEL_MAPPING['parakeet'])
        else:
            self.model_name = model_name
        self.decoding_strategy = decoding_strategy

        super().__init__(device, **kwargs)
        self.model = None

        # デバイスの自動検出と設定（共通関数を使用）
        self.torch_device, _ = detect_device(device, "Parakeet")
        
    def _check_dependencies(self) -> None:
        """
        Step 1: 依存関係の確認（0-10%）
        NeMoの利用可能性をチェック
        """
        # NeMoの利用可能性をチェック（初回のみインポートが試行される）
        if not check_nemo_availability():
            raise ImportError(
                "NVIDIA NeMo is not installed. Please run: pip install nemo_toolkit[asr]"
            )

    def _get_local_model_path(self, models_dir: Path) -> Path:
        """
        Step 2: ローカルモデルパスの決定（10-15%）
        """
        # モデル名は既にコンストラクタで決定済み
        model_name = self.model_name
        local_model_path = models_dir / f"{model_name.replace('/', '--')}.nemo"
        return local_model_path

    def _download_model(self, model_path: Path, progress_callback=None, model_manager=None) -> None:
        """
        Step 3: モデルのダウンロード（15-70%）
        Hugging Faceからモデルをダウンロード（必要な場合）
        """
        if model_path.exists():
            logging.info(f"ローカルファイルが存在: {model_path}")
            return

        # ここで初めてNeMoモジュールをインポート
        import nemo.collections.asr as nemo_asr
        from nemo.utils import logging as nemo_logging

        # NeMoの警告ログを抑制
        nemo_logger = logging.getLogger('nemo_logger')
        original_level = nemo_logger.level
        nemo_logger.setLevel(logging.ERROR)

        manager = model_manager or getattr(self, "model_manager", None)
        if manager is None:
            from livecap_core.resources import get_model_manager

            manager = get_model_manager()

        try:
            with unicode_safe_download_directory() as temp_dir:
                logging.info(f"Using download temporary directory: {temp_dir}")
                with manager.huggingface_cache():
                    model = nemo_asr.models.ASRModel.from_pretrained(
                        model_name=self.model_name,
                        map_location=self.torch_device
                    )

                    # 親ディレクトリを確実に作成
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    logging.info(f"Saving model to: {model_path.resolve()}")
                    model.save_to(str(model_path))
                    
                    # Windows Workaround: NeMoが親ディレクトリに保存してしまう場合の対策
                    if not model_path.exists():
                        logging.warning(f"Model file missing at expected path: {model_path}")
                        
                        # 想定: .../models/parakeet/file.nemo -> 実態: .../models/file.nemo
                        wrong_path = model_path.parent.parent / model_path.name
                        logging.info(f"Checking alternative path: {wrong_path.resolve()}")
                        
                        if wrong_path.exists():
                            logging.warning(f"Workaround: Found model at {wrong_path}, moving to {model_path}")
                            import shutil
                            shutil.move(str(wrong_path), str(model_path))
                        else:
                            logging.error(f"Model not found at {wrong_path} either.")
                            
                    del model
        finally:
            nemo_logger.setLevel(original_level)

    def _load_model_from_path(self, model_path: Path) -> Any:
        """
        Step 4: モデルファイルからロード（70-90%）
        """
        # キャッシュキーを生成
        cache_key = f"parakeet_{self.engine_name}_{self.model_name.replace('/', '_')}_{self.torch_device}"

        # キャッシュから取得を試みる
        cached_model = ModelMemoryCache.get(cache_key)
        if cached_model is not None:
            logging.info(f"キャッシュからモデルを取得: {cache_key}")
            return cached_model

        # NeMoモジュールをインポート
        import nemo.collections.asr as nemo_asr
        from nemo.utils import logging as nemo_logging

        # NeMoの警告ログを抑制
        nemo_logger = logging.getLogger('nemo_logger')
        original_level = nemo_logger.level
        nemo_logger.setLevel(logging.ERROR)

        try:
            # ローカルファイルからロード
            logging.info(f"ローカルファイルからモデルをロード: {model_path}")
            # ASRModelを使用（適切な具象クラスが自動的に選択される）
            model = nemo_asr.models.ASRModel.restore_from(
                restore_path=str(model_path),
                map_location=self.torch_device
            )

            # キャッシュに保存
            ModelMemoryCache.set(cache_key, model)

            return model

        finally:
            # NeMoのログレベルを元に戻す
            nemo_logger.setLevel(original_level)

    def _configure_model(self) -> None:
        """
        Step 5: モデルの設定（90-100%）
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # 評価モードに設定
        self.model.eval()

        # デコーディング戦略の設定
        if hasattr(self.model, 'change_decoding_strategy'):
            try:
                # NeMoの新しいAPIに対応
                decoding_config = {
                    'strategy': self.decoding_strategy,
                    'compute_confidence': True,
                    'preserve_alignments': False,
                    'preserve_frame_confidence': False
                }
                self.model.change_decoding_strategy(decoding_config)
            except TypeError:
                # 古いAPIの場合は引数なしで呼び出す
                try:
                    self.model.change_decoding_strategy()
                except Exception as e:
                    logging.warning(f"Could not set decoding strategy: {e}")
                    # デコーディング戦略の設定に失敗してもモデルは使用可能

        logging.info(f"{self.engine_name} model initialization complete")

    def load_model(self) -> None:
        """モデルをロードする（Windowsパス問題のワークアラウンド付き）"""
        # model_managerへのアクセス（遅延初期化）
        _ = self.model_manager
        
        models_dir = self.model_manager.get_models_dir(self.engine_name)
        model_path = self._get_local_model_path(models_dir)
        
        # Windows Workaround: 既存の古い場所のファイルを正しい場所に移動
        # ダウンロード済みだが場所が間違っている場合（CIキャッシュなど）の救済
        if not model_path.exists():
            # 想定: .../models/parakeet/file.nemo
            # 実態: .../models/file.nemo
            wrong_path = model_path.parent.parent / model_path.name
            
            if wrong_path.exists():
                logging.warning(f"Workaround: Found model at wrong location {wrong_path}, moving to {model_path}")
                try:
                    # 親ディレクトリを確実に作成
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.move(str(wrong_path), str(model_path))
                    logging.info("Model file moved successfully.")
                except Exception as e:
                    logging.error(f"Failed to move model file: {e}")
        
        # 親クラスの標準ロード処理を実行
        super().load_model()
            
    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[str, float]:
        """
        音声データを文字起こしする
        
        Args:
            audio_data: 音声データ（numpy配列）
            sample_rate: サンプリングレート
            
        Returns:
            (transcription_text, confidence_score)のタプル
        """
        # Parakeetは長時間音声も処理可能
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
            # NeMoのtranscribeメソッドはファイルパスのリストを期待するため、
            # 一時ファイルを作成して音声を保存
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_filename = tmp_file.name
                
                # 音声データを一時ファイルに保存
                # モデルが要求するサンプリングレートで保存
                sf.write(tmp_filename, audio_data, self.get_required_sample_rate())
                
            try:
                # プログレスバーを抑制
                import os
                import sys
                from io import StringIO
                
                old_tqdm = os.environ.get('TQDM_DISABLE')
                os.environ['TQDM_DISABLE'] = '1'
                
                # 標準出力を一時的にキャプチャ
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                
                try:
                    # NeMoのtranscribeメソッドを使用（ファイルパスのリストを渡す）
                    # TDTモデルでは'audio'パラメータを使用
                    transcriptions = self.model.transcribe(
                        audio=[tmp_filename],
                        batch_size=1
                    )
                finally:
                    # 標準出力を元に戻す
                    sys.stdout = old_stdout
                    
                    # 環境変数を元に戻す
                    if old_tqdm is None:
                        if 'TQDM_DISABLE' in os.environ:
                            del os.environ['TQDM_DISABLE']
                    else:
                        os.environ['TQDM_DISABLE'] = old_tqdm
                
                # 結果を取得
                # デバッグ: 結果の型と内容を確認
                logging.debug(f"Transcription result type: {type(transcriptions)}")
                logging.debug(f"Transcription result: {transcriptions}")
                
                # NeMo TDTモデルはタプルまたはリストを返すことがある
                if isinstance(transcriptions, tuple):
                    # タプルの場合、最初の要素が文字起こし結果
                    if len(transcriptions) > 0 and isinstance(transcriptions[0], list):
                        result = transcriptions[0][0] if len(transcriptions[0]) > 0 else ""
                    else:
                        result = transcriptions[0] if len(transcriptions) > 0 else ""
                elif isinstance(transcriptions, list) and len(transcriptions) > 0:
                    # 最初の結果を取得（単一ファイルなので1つだけ）
                    result = transcriptions[0] if transcriptions[0] else ""
                elif isinstance(transcriptions, str):
                    result = transcriptions
                else:
                    result = ""
                    logging.warning(f"Unexpected transcription result type: {type(transcriptions)}")
                
                # Hypothesisオブジェクトから文字列を取得
                if hasattr(result, 'text'):
                    # Hypothesisオブジェクトの場合、.textプロパティを使用
                    text = result.text if result.text else ""
                elif hasattr(result, 'pred_text'):
                    # 別のプロパティ名の可能性
                    text = result.pred_text if result.pred_text else ""
                elif isinstance(result, str):
                    # すでに文字列の場合
                    text = result
                else:
                    # その他の場合は文字列に変換
                    text = str(result) if result else ""
                
                # 文字列であることを確認してからstrip()を呼び出す
                if isinstance(text, str):
                    text = text.strip()
                else:
                    logging.warning(f"Unexpected text type: {type(text)}, converting to string")
                    text = str(text).strip() if text else ""
                
                logging.debug(f"Parakeet transcription: '{text}'")
                    
                # 空の結果をチェック
                if not text or text == "":
                    logging.debug("Parakeet returned empty transcription")
                    
                # 信頼度スコア（TDTでは利用不可）
                confidence = 1.0
                
                return text, confidence
                
            finally:
                # 一時ファイルを削除
                import os
                if os.path.exists(tmp_filename):
                    os.unlink(tmp_filename)
                
        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            raise
            
    def get_engine_name(self) -> str:
        """エンジン名を取得"""
        if self.engine_name == 'parakeet_ja':
            return "NVIDIA Parakeet TDT CTC 0.6B JA"
        return "NVIDIA Parakeet TDT 0.6B v3"
        
    def get_supported_languages(self) -> list:
        """サポートされる言語のリストを取得"""
        if self.engine_name == 'parakeet_ja':
            return ["ja"]
        return ["en"]
        
    def get_required_sample_rate(self) -> int:
        """エンジンが要求するサンプリングレートを取得"""
        # NeMoモデルは通常16kHzを使用
        return 16000
        
    def cleanup(self) -> None:
        """リソースのクリーンアップ"""
        if self.model is not None:
            # GPUメモリを解放
            del self.model
            self.model = None
            
            if self.torch_device == "cuda":
                # 遅延インポート: 必要な時のみtorchをインポート
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    # torchがインポートできない場合は何もしない
                    pass
