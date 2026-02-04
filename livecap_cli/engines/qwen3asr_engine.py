"""Qwen3-ASR 音声認識エンジンの実装 - Template Method版

Alibaba Cloud Qwen チームが開発した Qwen3-ASR を統合。
30言語以上をサポートし、Whisper-large-v3 を上回る精度を実現。

References:
    - https://github.com/QwenLM/Qwen3-ASR
    - https://huggingface.co/Qwen/Qwen3-ASR-0.6B
"""
import os
import sys
import logging
import tempfile
import importlib.util
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import soundfile as sf

from .base_engine import BaseEngine
from .model_memory_cache import ModelMemoryCache

from livecap_cli.utils import (
    get_models_dir,
    detect_device,
    unicode_safe_download_directory,
)
from livecap_cli.resources import get_model_manager

logger = logging.getLogger(__name__)

# Qwen-ASR availability check (lazy)
QWEN_ASR_AVAILABLE: Optional[bool] = None


def check_qwen_asr_availability() -> bool:
    """qwen-asr パッケージの利用可能性をチェック

    PyInstaller (frozen) 環境では importlib.util.find_spec() で
    パッケージの存在確認のみを行い、循環インポートを回避する。

    Returns:
        bool: qwen-asr が利用可能な場合 True
    """
    global QWEN_ASR_AVAILABLE
    if QWEN_ASR_AVAILABLE is not None:
        return QWEN_ASR_AVAILABLE

    # PyInstaller 環境では find_spec のみ使用
    if getattr(sys, 'frozen', False):
        try:
            QWEN_ASR_AVAILABLE = importlib.util.find_spec("qwen_asr") is not None
            if QWEN_ASR_AVAILABLE:
                logger.debug("qwen-asr パッケージが検出されました (frozen環境)")
            else:
                logger.warning("qwen-asr パッケージがインストールされていません")
        except Exception as e:
            QWEN_ASR_AVAILABLE = False
            logger.warning(f"qwen-asr の可用性チェックに失敗: {e}")
        return QWEN_ASR_AVAILABLE

    # 通常環境では実際にインポートを試行
    try:
        from qwen_asr import Qwen3ASRModel as Qwen3ASR
        QWEN_ASR_AVAILABLE = True
        logger.info("qwen-asr が正常にインポートされました")
    except ImportError as e:
        QWEN_ASR_AVAILABLE = False
        logger.error(f"qwen-asr のインポートに失敗しました: {e}")

    return QWEN_ASR_AVAILABLE


def prepare_qwen_asr_environment() -> None:
    """Qwen-ASR インポート前の環境準備

    PyInstaller 環境での循環インポート問題を回避するため、
    依存ライブラリのサブモジュールを事前にインポートする。
    """
    if not getattr(sys, 'frozen', False):
        return

    # librosa サブモジュールを事前インポート（#219 対策パターン）
    try:
        import librosa.util
        import librosa.core.convert
        import librosa.filters
        import librosa.core.spectrum
        logger.debug("librosa サブモジュールを事前インポートしました")
    except ImportError as e:
        logger.debug(f"librosa 事前インポートをスキップ: {e}")
    except Exception as e:
        logger.debug(f"librosa 事前インポート中に予期しないエラー: {e}")


class Qwen3ASREngine(BaseEngine):
    """Qwen3-ASR 音声認識エンジン - Template Method版

    Alibaba Cloud Qwen チームが開発した高精度 ASR エンジン。
    30言語以上をサポートし、言語自動検出機能を持つ。

    Attributes:
        engine_name: エンジン識別子 (qwen3asr / qwen3asr_large)
        language: 入力言語 (None = 自動検出)
        model_name: HuggingFace モデル名
    """

    # サポートする言語コード
    SUPPORTED_LANGUAGES = [
        "zh", "en", "yue", "ar", "de", "fr", "es", "pt", "id", "it",
        "ko", "ru", "th", "vi", "ja", "tr", "hi", "ms", "nl", "sv",
        "da", "fi", "pl", "cs", "fil", "fa", "el", "hu", "mk", "ro"
    ]

    def __init__(
        self,
        device: Optional[str] = None,
        language: Optional[str] = None,
        model_name: str = "Qwen/Qwen3-ASR-0.6B",
        engine_id: str = "qwen3asr",
        **kwargs,
    ):
        """エンジンを初期化

        Args:
            device: 使用するデバイス ("cpu", "cuda", None=auto)
            language: 入力言語 (None = 自動検出)
            model_name: HuggingFace モデル名
            engine_id: エンジン識別子 (metadata から渡される)
            **kwargs: 追加パラメータ
        """
        # エンジン名を設定（qwen3asr / qwen3asr_large を区別）
        self.engine_name = engine_id

        # パラメータ設定
        self.language = language
        self.model_name = model_name

        super().__init__(device, **kwargs)
        self.model = None
        self._initialized = False

        # デバイスの自動検出と設定
        self.torch_device = detect_device(device, "Qwen3-ASR")

    # ===============================
    # Template Method 実装
    # ===============================

    def get_model_metadata(self) -> Dict[str, Any]:
        """モデルのメタデータを返す"""
        return {
            'name': self.model_name,
            'version': '0.6B' if '0.6B' in self.model_name else '1.7B',
            'format': 'transformers',
            'description': 'Qwen3-ASR - High-accuracy multilingual ASR (30+ languages)'
        }

    def _check_dependencies(self) -> None:
        """Step 1: 依存関係のチェック（0-10%）"""
        self.report_progress(5, "Checking qwen-asr availability...")

        if not check_qwen_asr_availability():
            logger.error("qwen-asr がインストールされていません")
            raise ImportError(
                "qwen-asr is not installed. Please run: "
                "pip install 'livecap-cli[engines-qwen3asr]'"
            )

        self.report_progress(10, "Dependencies check complete")

    def _get_local_model_path(self, models_dir: Path) -> Path:
        """ローカルモデルパスを取得

        Qwen3-ASR は HuggingFace キャッシュを使用するため、
        モデルディレクトリへのマーカーファイルを返す。
        """
        # HuggingFace キャッシュを使用するため、マーカーファイルのみ
        return models_dir / f"{self.model_name.replace('/', '--')}.marker"

    def _prepare_model_directory(self) -> Path:
        """Step 2: モデルディレクトリの準備（10-15%）"""
        self.report_progress(12, "Preparing model directory...")

        models_dir = get_models_dir()
        models_dir.mkdir(exist_ok=True)

        self.report_progress(15, f"Model directory: {models_dir}")
        return models_dir

    def _is_model_cached(self, model_path: Path) -> bool:
        """モデルがキャッシュされているか確認

        Qwen3-ASR は HuggingFace キャッシュを使用するため、
        マーカーファイルの存在でキャッシュを判定する。
        """
        return model_path.exists()

    def _download_model(self, model_path: Path, progress_callback, model_manager=None) -> None:
        """Step 3: モデルのダウンロード（15-70%）

        Qwen3-ASR は初回ロード時に HuggingFace から自動ダウンロードされる。
        ここではマーカーファイルの作成のみ行う。
        """
        self.report_progress(20, f"Model will be downloaded on first load: {self.model_name}")

        # マーカーファイルを作成（実際のダウンロードは _load_model_from_path で行われる）
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text(f"model={self.model_name}\ndevice={self.torch_device}")

        self.report_progress(70, "Model marker created")

    def _load_model_from_path(self, model_path: Path) -> Any:
        """Step 4: モデルファイルからロード（70-90%）"""
        self.report_progress(75, f"Loading model: {self.model_name}")

        # キャッシュキーを生成
        cache_key = f"qwen3asr_{self.model_name.replace('/', '_')}_{self.torch_device}"

        # キャッシュから取得を試みる
        cached_model = ModelMemoryCache.get(cache_key)
        if cached_model is not None:
            self.report_progress(90, "Loading from cache: Qwen3-ASR")
            logger.info(f"キャッシュからモデルを取得: {cache_key}")
            return cached_model

        # 環境準備（PyInstaller 互換性）
        prepare_qwen_asr_environment()

        # qwen-asr モジュールをインポート
        from qwen_asr import Qwen3ASRModel as Qwen3ASR

        self.report_progress(80, "Initializing Qwen3-ASR model...")

        # ModelManager から HuggingFace キャッシュを使用（他エンジンと整合）
        manager = get_model_manager()

        with unicode_safe_download_directory():
            with manager.huggingface_cache() as hf_cache:
                # モデルをロード（from_pretrained API を使用）
                # device_map: "cpu" はそのまま、"cuda" は "auto" に変換
                # "auto" は利用可能な GPU を自動選択する
                device_map = "auto" if self.torch_device == "cuda" else self.torch_device
                model = Qwen3ASR.from_pretrained(
                    self.model_name,
                    device_map=device_map,
                )

        self.report_progress(85, "Model loaded successfully")

        # キャッシュに保存
        use_strong_cache = os.environ.get('LIVECAP_ENGINE_STRONG_CACHE', '').lower() in ('1', 'true', 'yes')
        ModelMemoryCache.set(cache_key, model, strong=use_strong_cache)
        logger.info(f"モデルをキャッシュに保存: {cache_key} (strong={use_strong_cache})")

        self.report_progress(90, "Qwen3-ASR: Ready")
        return model

    def _configure_model(self) -> None:
        """Step 5: モデルの設定（90-100%）"""
        self.report_progress(92, "Configuring model...")

        if self.model is None:
            raise RuntimeError("Model not loaded")

        self._initialized = True
        self.report_progress(100, "Qwen3-ASR model configuration complete")
        logger.info("モデルの設定が完了しました。")

    # ===============================
    # TranscriptionEngine プロトコル実装
    # ===============================

    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[str, float]:
        """音声データを文字起こしする

        Args:
            audio_data: 音声データ（numpy配列）
            sample_rate: サンプリングレート

        Returns:
            (transcription_text, confidence_score) のタプル
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

        # 音声データの正規化（-1.0 から 1.0 の範囲）
        if np.abs(audio_data).max() > 1.0:
            audio_data = audio_data / np.abs(audio_data).max()

        # デバッグログ
        logger.debug(f"Audio data shape: {audio_data.shape}")
        logger.debug(f"Audio duration: {len(audio_data) / required_sr:.2f} seconds")
        logger.debug(f"Audio max amplitude: {np.abs(audio_data).max():.4f}")

        # 音声が短すぎる場合の処理
        min_duration = 0.1  # 最小 0.1 秒
        min_samples = int(min_duration * required_sr)
        if len(audio_data) < min_samples:
            logger.warning(f"Audio too short: {len(audio_data)} samples < {min_samples} samples")
            return "", 1.0

        try:
            # Qwen3-ASR はファイルパスを期待するため、一時ファイルを作成
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_filename = tmp_file.name
                sf.write(tmp_filename, audio_data, required_sr)

            try:
                # 文字起こし実行
                # language パラメータを渡す（None = 自動検出）
                result = self.model.transcribe(
                    audio=tmp_filename,
                    language=self.language,
                )

                # 結果を取得
                # Qwen3-ASR は結果オブジェクトのリストを返す
                # 各オブジェクトには .text, .language, .time_stamps 属性がある
                if result and len(result) > 0:
                    # 最初の結果から text を抽出
                    first_result = result[0]
                    if hasattr(first_result, 'text'):
                        text = first_result.text
                    elif isinstance(first_result, str):
                        text = first_result
                    else:
                        text = str(first_result)
                    logger.debug(f"Qwen3-ASR transcription: '{text}'")
                else:
                    text = ""

                # 信頼度スコア（Qwen3-ASR では利用不可）
                confidence = 1.0

                return text, confidence

            finally:
                # 一時ファイルを削除
                if os.path.exists(tmp_filename):
                    os.unlink(tmp_filename)

        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise

    def get_engine_name(self) -> str:
        """エンジン名を取得"""
        if "1.7B" in self.model_name:
            return "Qwen3-ASR 1.7B"
        return "Qwen3-ASR 0.6B"

    def get_supported_languages(self) -> list:
        """サポートされる言語のリストを取得"""
        return self.SUPPORTED_LANGUAGES.copy()

    def get_required_sample_rate(self) -> int:
        """エンジンが要求するサンプリングレートを取得"""
        # Qwen3-ASR は 16kHz を使用
        return 16000

    def cleanup(self) -> None:
        """リソースのクリーンアップ"""
        if self.model is not None:
            del self.model
            self.model = None
            if self.torch_device == "cuda":
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass
        self._initialized = False
