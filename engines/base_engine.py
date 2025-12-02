"""音声認識エンジンの抽象基底クラス（Template Method実装）"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Callable
from typing import Protocol
import numpy as np
import logging

# 新しいフェーズ管理システムをインポート
from .model_loading_phases import (
    LoadPhase, PhaseInfo, ModelLoadingPhases, ProgressReport
)
from livecap_core.i18n import translate, register_fallbacks
from livecap_core.resources import get_model_manager
from livecap_core.utils import get_models_dir

logger = logging.getLogger(__name__)


register_fallbacks({
    "model_init_dialog.status_messages.checking_dependencies": "Checking dependencies...",
    "model_init_dialog.status_messages.dependencies_complete": "Dependencies check complete",
    "model_init_dialog.status_messages.preparing_directory": "Preparing model directory...",
    "model_init_dialog.status_messages.checking_files": "Checking model files...",
    "model_init_dialog.status_messages.loading_to_memory": "Loading model to memory...",
    "model_init_dialog.status_messages.applying_settings": "Applying model settings...",
    "model_init_dialog.status_messages.model_load_complete": "Model loading complete",
    "model_init_dialog.status_messages.loading_from_cache": "Loading from cache: {model_name}",
    "model_init_dialog.status_messages.downloading_model": "Downloading model: {model_name}",
    "model_init_dialog.status_messages.download_complete": "Model download complete",
    "model_init_dialog.status_messages.checking_availability": "Checking {engine_name} availability...",
    "model_init_dialog.status_messages.model_info": "Model: {model_name}",
    "model_init_dialog.status_messages.model_ready": "{engine_name}: {model_name} model ready",
    "model_init_dialog.status_messages.loading_from_memory_cache": "Loading from memory cache",
    "model_init_dialog.status_messages.initializing_model": "{engine_name}: Initializing {model_name} model...",
    "model_init_dialog.status_messages.initialization_success": "{engine_name}: Model initialization successful",
    "model_init_dialog.status_messages.initialization_complete": "{engine_name}: Initialization complete",
    "model_init_dialog.status_messages.final_configuration": "{engine_name}: Applying final settings...",
    "model_init_dialog.status_messages.model_ready_cpu_mode": "{engine_name}: Ready (CPU mode)",
    "model_init_dialog.status_messages.model_ready_simple": "{engine_name}: Ready",
    "model_init_dialog.status_messages.model_already_downloaded": "Model already downloaded",
    "model_init_dialog.status_messages.downloading_from_huggingface": "Downloading model from Hugging Face: {model_name}",
    "model_init_dialog.status_messages.download_starting": "Starting model download...",
    "model_init_dialog.status_messages.downloading_processor": "Downloading processor...",
    "model_init_dialog.status_messages.saving_model_locally": "Saving model locally...",
    "model_init_dialog.status_messages.restoring_model": "Restoring {engine_name} model...",
    "model_init_dialog.status_messages.loading_processor": "Loading processor...",
    "model_init_dialog.status_messages.setting_evaluation_mode": "Setting model to evaluation mode...",
    "model_init_dialog.status_messages.configuration_complete": "{engine_name} model configuration complete",
    "model_init_dialog.status_messages.checking_transformers": "Checking Transformers availability...",
    "model_init_dialog.status_messages.checking_class": "Checking {class_name} class...",
    "model_init_dialog.status_messages.checking_mistral_common": "Checking mistral-common...",
    "model_init_dialog.status_messages.checking_classes": "Checking {engine_name} classes...",
    "model_init_dialog.status_messages.checking_dependencies_detail": "Checking {dependency_name}...",
    "model_init_dialog.status_messages.model_path": "Model path: {path}",
    "model_init_dialog.status_messages.saving_to_local": "Saving model locally...",
    "model_init_dialog.status_messages.loading_model_file": "Loading model file: {path}",
    "model_init_dialog.status_messages.setting_eval_mode": "Setting model to evaluation mode...",
    "model_init_dialog.status_messages.updating_decoding_settings": "Updating decoding settings...",
    "model_init_dialog.status_messages.model_config_complete": "{model_name} model configuration complete",
    "model_init_dialog.status_messages.downloading_progress": "Downloading: {downloaded}/{total} bytes",
    "model_init_dialog.status_messages.extracting_archive": "Extracting: {filename}",
    "model_init_dialog.status_messages.copying_model_files": "Copying model files...",
    "model_init_dialog.status_messages.loading_model_type": "Loading {model_type} model...",
    "model_init_dialog.status_messages.model_load_success": "Model loaded successfully",
    "model_init_dialog.status_messages.configuring_model": "Configuring model...",
})


class ProgressCallback(Protocol):
    """進捗報告コールバックのプロトコル定義"""
    def __call__(self, percent: int, message: str = "") -> None: ...


class BaseEngine(ABC):
    """音声認識エンジンの抽象基底クラス（Template Methodパターン）"""

    def __init__(self, device: Optional[str] = None, **kwargs):
        """
        Args:
            device: 使用するデバイス（cuda/cpu/null）
            **kwargs: エンジン固有のパラメータ（各エンジンクラスで処理）
        """
        self.device = device
        self._initialized = False
        self.model = None
        self.progress_callback: Optional[ProgressCallback] = None
        self.model_manager = get_model_manager()

        # エンジン固有の設定を読み込み
        # 子クラスでengine_nameを設定する必要がある
        self.engine_name = getattr(self, 'engine_name', 'default')

        # モデルメタデータを取得（子クラスで定義）
        try:
            self.model_metadata = self.get_model_metadata()
        except NotImplementedError:
            # get_model_metadataが実装されていない場合はデフォルト値
            self.model_metadata = {}
    
    # =====================================
    # 進捗報告メソッド
    # =====================================
    def set_progress_callback(self, callback: ProgressCallback):
        """進捗報告用コールバックを設定
        
        Args:
            callback: 進捗報告コールバック (percent: int, message: str) -> None
        """
        self.progress_callback = callback
        
    def get_status_message(self, message_key: str, **kwargs) -> str:
        """多言語対応のステータスメッセージを取得

        Args:
            message_key: メッセージキー
            **kwargs: フォーマット用のパラメータ

        Returns:
            ローカライズされたメッセージ
        """
        return translate(
            f"model_init_dialog.status_messages.{message_key}",
            **kwargs,
        )

    def report_progress(self, percent: int, message: str = "", phase: Optional[LoadPhase] = None):
        """進捗を報告

        Args:
            percent: 進捗率（0-100）
            message: 進捗メッセージ
            phase: 現在のロードフェーズ（オプション）
        """
        # フェーズが指定されていない場合は進捗率から推定
        if phase is None:
            phase_info = ModelLoadingPhases.get_phase_by_progress(percent)
            phase = phase_info.phase if phase_info else None
        
        if self.progress_callback:
            self.progress_callback(percent, message)
        
        # ログにも記録
        if message:
            logger.info(f"[{self.engine_name}] [{percent}%] {message}")
            if phase:
                logger.debug(f"[{self.engine_name}] Phase: {phase.name}")
    
    # =====================================
    # Template Method（新実装）
    # =====================================
    def load_model(self) -> None:
        """
        モデルロードのテンプレートメソッド
        共通の流れを定義し、具体的な処理は子クラスに委譲
        """
        try:
            # Phase 1: 依存関係チェック
            phase_info = ModelLoadingPhases.get_phase_info(LoadPhase.CHECK_DEPENDENCIES)
            self.report_progress(phase_info.progress_start, self.get_status_message("checking_dependencies"), LoadPhase.CHECK_DEPENDENCIES)
            self._check_dependencies()
            self.report_progress(phase_info.progress_end, phase=LoadPhase.CHECK_DEPENDENCIES)
            
            # Phase 2: モデルディレクトリ準備
            phase_info = ModelLoadingPhases.get_phase_info(LoadPhase.PREPARE_DIRECTORY)
            self.report_progress(phase_info.progress_start, self.get_status_message("preparing_directory"), LoadPhase.PREPARE_DIRECTORY)
            models_dir = self._prepare_model_directory()
            self.report_progress(phase_info.progress_end, phase=LoadPhase.PREPARE_DIRECTORY)
            
            # Phase 3: ファイル確認
            phase_info = ModelLoadingPhases.get_phase_info(LoadPhase.CHECK_FILES)
            self.report_progress(phase_info.progress_start, self.get_status_message("checking_files"), LoadPhase.CHECK_FILES)
            # ファイル確認はダウンロード処理内で実施
            self.report_progress(phase_info.progress_end, phase=LoadPhase.CHECK_FILES)
            
            # Phase 4: モデル取得（ダウンロードまたはキャッシュ）
            model_path = self._get_or_download_model(models_dir)
            
            # Phase 5: メモリロード
            phase_info = ModelLoadingPhases.get_phase_info(LoadPhase.LOAD_TO_MEMORY)
            self.report_progress(phase_info.progress_start, self.get_status_message("loading_to_memory"), LoadPhase.LOAD_TO_MEMORY)
            self.model = self._load_model_from_path(model_path)
            self.report_progress(phase_info.progress_end, phase=LoadPhase.LOAD_TO_MEMORY)
            
            # Phase 6: モデル設定
            phase_info = ModelLoadingPhases.get_phase_info(LoadPhase.APPLY_SETTINGS)
            self.report_progress(phase_info.progress_start, self.get_status_message("applying_settings"), LoadPhase.APPLY_SETTINGS)
            self._configure_model()
            self.report_progress(phase_info.progress_end, self.get_status_message("model_load_complete"), LoadPhase.APPLY_SETTINGS)
            
            self._initialized = True
            logger.info(f"{self.engine_name} モデルロード成功")
            
        except Exception as e:
            logger.error(f"{self.engine_name} モデルロード失敗: {e}")
            raise
    
    # =====================================
    # 共通実装（全エンジンで使用）
    # =====================================
    def _prepare_model_directory(self) -> Path:
        """モデルディレクトリを準備"""
        models_dir = get_models_dir()
        models_dir.mkdir(exist_ok=True)
        return models_dir
    
    def _get_or_download_model(self, models_dir: Path) -> Path:
        """モデルファイルを取得（キャッシュまたはダウンロード）"""
        local_path = self._get_local_model_path(models_dir)
        
        # キャッシュチェック
        if self._is_model_cached(local_path):
            # キャッシュ済みの場合、ダウンロードフェーズをスキップ
            skip_progress = ModelLoadingPhases.skip_download_phase(
                ModelLoadingPhases.get_phase_info(LoadPhase.CHECK_FILES).progress_end
            )
            self.report_progress(skip_progress, self.get_status_message("loading_from_cache", model_name=local_path.name), LoadPhase.DOWNLOAD_MODEL)
            return local_path
        
        # ダウンロード開始
        phase_info = ModelLoadingPhases.get_phase_info(LoadPhase.DOWNLOAD_MODEL)
        model_name = self.model_metadata.get('name', 'model')
        self.report_progress(phase_info.progress_start, self.get_status_message("downloading_model", model_name=model_name), LoadPhase.DOWNLOAD_MODEL)
        self._download_model_with_progress(local_path)
        
        # 完全性チェック
        if not self._verify_model_integrity(local_path):
            local_path.unlink(missing_ok=True)
            raise ValueError(f"ダウンロードしたモデルが破損: {local_path}")
        
        self.report_progress(phase_info.progress_end, self.get_status_message("download_complete"), LoadPhase.DOWNLOAD_MODEL)
        return local_path
    
    def _is_model_cached(self, model_path: Path) -> bool:
        """モデルがキャッシュされているか確認"""
        if isinstance(model_path, Path):
            # ディレクトリまたは単一ファイルの場合
            if not model_path.exists():
                return False
            
            # ディレクトリの場合は存在チェックのみ
            if model_path.is_dir():
                # ディレクトリ内に少なくとも1つのファイルがあるか確認
                return any(model_path.iterdir())
        else:
            # 複数ファイルの場合（辞書形式）
            for file_path in model_path.values():
                if not Path(file_path).exists():
                    return False
        
        return self._verify_model_integrity(model_path)
    
    def _verify_model_integrity(self, model_path) -> bool:
        """モデル完全性チェック"""
        if isinstance(model_path, Path):
            if not model_path.exists():
                return False
            
            # ディレクトリの場合
            if model_path.is_dir():
                # ディレクトリ内に必要なファイルがあるか確認（エンジン固有のチェックを期待）
                # 最低限、ディレクトリ内にファイルがあることを確認
                try:
                    has_files = any(model_path.iterdir())
                    if not has_files:
                        logger.warning(f"モデルディレクトリが空: {model_path}")
                    return has_files
                except Exception as e:
                    logger.error(f"ディレクトリアクセスエラー: {e}")
                    return False
            
            # ファイルの場合
            try:
                with open(model_path, 'rb') as f:
                    header = f.read(4)
                    
                    # ファイル形式チェック
                    if model_path.suffix == '.nemo':
                        # .nemoファイルはTAR形式またはZIP形式
                        return header == b'PK\x03\x04' or header[:3] == b'./.'
                    elif model_path.suffix == '.onnx':
                        return len(header) >= 2 and header[:2] == b'\x08\x01'  # ONNX形式
                    elif model_path.suffix in ['.bin', '.pt', '.pth']:
                        return True  # PyTorchは多様なので基本チェックのみ
                        
                return True
            except Exception as e:
                logger.error(f"完全性チェック失敗: {e}")
                return False
        else:
            # 複数ファイルの場合
            return True  # 個別のチェックは子クラスに委譲
    
    def _download_model_with_progress(self, target_path):
        """進捗報告付きダウンロード（共通ラッパー）"""
        phase_info = ModelLoadingPhases.get_phase_info(LoadPhase.DOWNLOAD_MODEL)
        
        # ダウンロードフェーズの範囲内で進捗を報告
        def progress_wrapper(current: int, total: int):
            if total > 0:
                # フェーズの範囲内で進捗を計算
                progress_range = phase_info.progress_end - phase_info.progress_start
                percent = phase_info.progress_start + int((current / total) * progress_range)
                self.report_progress(percent, phase=LoadPhase.DOWNLOAD_MODEL)
        
        # エンジン固有のダウンロード処理を呼び出し
        self._download_model(target_path, progress_wrapper, self.model_manager)
    
    # =====================================
    # 新しい抽象メソッド（Template Method用）
    # =====================================
    def get_model_metadata(self) -> Dict[str, Any]:
        """
        モデルメタデータを返す（子クラスでオーバーライド）
        
        Returns:
            Dict containing model metadata:
                - name: モデル名
                - version: バージョン
                - size: ファイルサイズ（バイト）
                - download_url: ダウンロードURL
                - files: ファイルリスト
                - format: モデル形式
        """
        return {}
    
    def _check_dependencies(self) -> None:
        """依存ライブラリをチェック（子クラスでオーバーライド）"""
        pass
    
    def _get_local_model_path(self, models_dir: Path) -> Path:
        """ローカルモデルパスを取得（子クラスでオーバーライド）"""
        model_name = self.model_metadata.get('name', 'model').replace('/', '--')
        return models_dir / f"{model_name}.bin"
    
    def _download_model(self, target_path: Path, progress_callback: Callable) -> None:
        """モデルをダウンロード（子クラスで実装必須）"""
        raise NotImplementedError("_download_model must be implemented in subclass")
    
    def _load_model_from_path(self, model_path: Path) -> Any:
        """モデルファイルからロード（子クラスで実装必須）"""
        raise NotImplementedError("_load_model_from_path must be implemented in subclass")
    
    def _configure_model(self) -> None:
        """モデルを設定（子クラスでオーバーライド）"""
        pass
    
    # =====================================
    # 既存の抽象メソッド（変更なし）
    # =====================================
    @abstractmethod
    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[str, float]:
        """
        音声データを文字起こしする
        
        Args:
            audio_data: 音声データ（numpy配列）
            sample_rate: サンプリングレート
            
        Returns:
            (transcription_text, confidence_score)のタプル
        """
        pass
        
    @abstractmethod
    def get_engine_name(self) -> str:
        """エンジン名を取得"""
        pass
        
    @abstractmethod
    def get_supported_languages(self) -> list:
        """サポートされる言語のリストを取得"""
        pass
        
    @abstractmethod
    def get_required_sample_rate(self) -> int:
        """エンジンが要求するサンプリングレートを取得"""
        pass
        
    def is_initialized(self) -> bool:
        """エンジンが初期化されているか"""
        return self._initialized
        
    def cleanup(self) -> None:
        """リソースのクリーンアップ（オプション）"""
        pass
