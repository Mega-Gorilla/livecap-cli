"""
モデルロードフェーズの定義

BaseEngineとGUI間で共有される進捗フェーズの定義
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional


class LoadPhase(Enum):
    """モデルロードのフェーズ定義"""
    CHECK_DEPENDENCIES = auto()
    PREPARE_DIRECTORY = auto()
    CHECK_FILES = auto()
    DOWNLOAD_MODEL = auto()
    LOAD_TO_MEMORY = auto()
    APPLY_SETTINGS = auto()
    TRANSLATION_MODEL = auto()  # 翻訳モデルのロードフェーズ
    COMPLETED = auto()


@dataclass
class PhaseInfo:
    """フェーズ情報"""
    phase: LoadPhase
    progress_start: int  # このフェーズの開始進捗率
    progress_end: int    # このフェーズの終了進捗率
    display_name_key: str  # 多言語対応用のキー
    
    def contains_progress(self, progress: int) -> bool:
        """指定された進捗率がこのフェーズに含まれるか"""
        return self.progress_start <= progress <= self.progress_end


class ModelLoadingPhases:
    """モデルロードフェーズの管理"""
    
    # フェーズ定義（BaseEngineとGUIで共有）
    PHASE_DEFINITIONS = [
        PhaseInfo(
            phase=LoadPhase.CHECK_DEPENDENCIES,
            progress_start=0,
            progress_end=8,
            display_name_key="model_init_dialog.tasks.check_dependencies"
        ),
        PhaseInfo(
            phase=LoadPhase.PREPARE_DIRECTORY,
            progress_start=8,
            progress_end=12,
            display_name_key="model_init_dialog.tasks.prepare_directory"
        ),
        PhaseInfo(
            phase=LoadPhase.CHECK_FILES,
            progress_start=12,
            progress_end=15,
            display_name_key="model_init_dialog.tasks.check_files"
        ),
        PhaseInfo(
            phase=LoadPhase.DOWNLOAD_MODEL,
            progress_start=15,
            progress_end=50,
            display_name_key="model_init_dialog.tasks.download_model"
        ),
        PhaseInfo(
            phase=LoadPhase.LOAD_TO_MEMORY,
            progress_start=50,
            progress_end=70,
            display_name_key="model_init_dialog.tasks.load_to_memory"
        ),
        PhaseInfo(
            phase=LoadPhase.APPLY_SETTINGS,
            progress_start=70,
            progress_end=75,
            display_name_key="model_init_dialog.tasks.apply_settings"
        ),
        PhaseInfo(
            phase=LoadPhase.TRANSLATION_MODEL,
            progress_start=75,
            progress_end=100,
            display_name_key="model_init_dialog.tasks.load_translation"
        ),
    ]
    
    @classmethod
    def get_phase_by_progress(cls, progress: int) -> Optional[PhaseInfo]:
        """進捗率から現在のフェーズを取得"""
        for phase_info in cls.PHASE_DEFINITIONS:
            if phase_info.contains_progress(progress):
                return phase_info
        return None
    
    @classmethod
    def get_phase_info(cls, phase: LoadPhase) -> Optional[PhaseInfo]:
        """フェーズ列挙値からフェーズ情報を取得"""
        for phase_info in cls.PHASE_DEFINITIONS:
            if phase_info.phase == phase:
                return phase_info
        return None
    
    @classmethod
    def get_all_phases(cls) -> list[PhaseInfo]:
        """すべてのフェーズ情報を取得"""
        return cls.PHASE_DEFINITIONS.copy()
    
    @classmethod
    def skip_download_phase(cls, current_progress: int) -> int:
        """ダウンロードフェーズをスキップする際の進捗値を取得"""
        # キャッシュ済みの場合、ダウンロードフェーズの終了値へジャンプ
        download_phase = cls.get_phase_info(LoadPhase.DOWNLOAD_MODEL)
        if download_phase and current_progress < download_phase.progress_end:
            return download_phase.progress_end
        return current_progress


class ProgressReport:
    """進捗報告の構造化データ"""
    
    def __init__(self, progress: int, message: str = "", phase: Optional[LoadPhase] = None):
        """
        Args:
            progress: 進捗率（0-100）
            message: ステータスメッセージ
            phase: 明示的なフェーズ指定（オプション）
        """
        self.progress = progress
        self.message = message
        
        # フェーズを自動判定または明示的に指定
        if phase:
            self.phase = phase
        else:
            phase_info = ModelLoadingPhases.get_phase_by_progress(progress)
            self.phase = phase_info.phase if phase_info else None
    
    def __str__(self):
        return f"Progress: {self.progress}%, Phase: {self.phase}, Message: {self.message}"