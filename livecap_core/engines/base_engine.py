"""音声認識エンジンの抽象基底クラス（Template Method実装）"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Callable, Protocol
import numpy as np
import logging

from livecap_core.resources import get_model_manager
from livecap_core.utils import get_models_dir

logger = logging.getLogger(__name__)


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
        
    def report_progress(self, percent: int, message: str = ""):
        """進捗を報告

        Args:
            percent: 進捗率（0-100）
            message: 進捗メッセージ
        """
        if self.progress_callback:
            self.progress_callback(percent, message)

        # ログにも記録
        if message:
            logger.info(f"[{self.engine_name}] [{percent}%] {message}")
    
    # =====================================
    # Template Method
    # =====================================
    def load_model(self) -> None:
        """
        モデルロードのテンプレートメソッド
        共通の流れを定義し、具体的な処理は子クラスに委譲

        Progress ranges:
        - 0-10%: Check dependencies
        - 10-15%: Prepare directory
        - 15-20%: Check files
        - 20-70%: Download model (or skip if cached)
        - 70-90%: Load to memory
        - 90-100%: Apply settings
        """
        try:
            # Step 1: 依存関係チェック (0-10%)
            self.report_progress(0, "Checking dependencies...")
            self._check_dependencies()
            self.report_progress(10)

            # Step 2: モデルディレクトリ準備 (10-15%)
            self.report_progress(10, "Preparing model directory...")
            models_dir = self._prepare_model_directory()
            self.report_progress(15)

            # Step 3: ファイル確認 (15-20%)
            # Note: 進捗 15-20% は _get_local_model_path() と _get_or_download_model() で報告される
            self.report_progress(15, "Checking model files...")

            # Step 4: モデル取得（ダウンロードまたはキャッシュ）(20-70%)
            model_path = self._get_or_download_model(models_dir)

            # Step 5: メモリロード (70-90%)
            self.report_progress(70, "Loading model to memory...")
            self.model = self._load_model_from_path(model_path)
            self.report_progress(90)

            # Step 6: モデル設定 (90-100%)
            self.report_progress(90, "Applying model settings...")
            self._configure_model()
            self.report_progress(100, "Model loading complete")

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
        """モデルファイルを取得（キャッシュまたはダウンロード）

        Progress range: 20-70%
        """
        local_path = self._get_local_model_path(models_dir)

        # キャッシュチェック
        if self._is_model_cached(local_path):
            # キャッシュ済みの場合、ダウンロードフェーズをスキップして70%に
            self.report_progress(70, f"Loading from cache: {local_path.name}")
            return local_path

        # ダウンロード開始 (20-70%)
        model_name = self.model_metadata.get('name', 'model')
        self.report_progress(20, f"Downloading model: {model_name}")
        self._download_model_with_progress(local_path)

        # 完全性チェック
        if not self._verify_model_integrity(local_path):
            local_path.unlink(missing_ok=True)
            raise ValueError(f"ダウンロードしたモデルが破損: {local_path}")

        self.report_progress(70, "Model download complete")
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
        """進捗報告付きダウンロード（共通ラッパー）

        Progress range: 20-70%
        """
        # ダウンロードフェーズの範囲内で進捗を報告 (20-70%)
        def progress_wrapper(current: int, total: int):
            if total > 0:
                # 20-70% の範囲で進捗を計算
                percent = 20 + int((current / total) * 50)
                self.report_progress(percent)

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
    
    def _download_model(self, target_path: Path, progress_callback: Callable, model_manager=None) -> None:
        """モデルをダウンロード（子クラスで実装必須）

        Args:
            target_path: ダウンロード先のパス
            progress_callback: 進捗報告コールバック (current, total) -> None
            model_manager: モデルマネージャー（オプション）
        """
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
