"""NVIDIA Canary 1B v2エンジンの実装 - Template Method版"""
import os
import sys
import logging
import warnings
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from io import StringIO
import numpy as np
import soundfile as sf

from .base_engine import BaseEngine, EngineConfidence, TranscriptionResult
from .model_memory_cache import ModelMemoryCache
from .library_preloader import LibraryPreloader


def _extract_engine_confidence(hypothesis: Any) -> EngineConfidence:
    """NeMo Hypothesis から engine confidence signal を抽出 (Issue #311 PR-A.4.2)。

    Canary は EncDecMultiTaskModel (AED multitask)、Parakeet (TDT-CTC hybrid)
    と architecture は異なるが、両者とも NeMo の `confidence_method_cfg` 経由で
    生成される ``hypothesis.token_confidence`` を共用する。

    型差分 (Phase 1 probe で発覚):
    - Parakeet (CTC): ``token_confidence: List[float]`` (CPU、変換済)
    - Canary (AED multitask): ``token_confidence: torch.Tensor`` (GPU、要 ``.tolist()``)

    本 helper は両方を扱えるよう ``hasattr(token_conf, 'tolist')`` 防御で
    GPU tensor / numpy array / list / tuple を統一処理する。

    抽出ロジック:
    - ``hypothesis.token_confidence`` が populate されている場合、tolist 経由
      で list 化、mean を ``token_confidence_mean`` field に詰める。
    - それ以外の場合は全 None の ``EngineConfidence()`` を返す (fail-open)。

    pure-function として exposed しているのは unit test で実 model 不要に
    schema 抽出ロジックを pin するため (Parakeet pattern 流用)。
    """
    if hypothesis is None:
        return EngineConfidence()

    token_conf = getattr(hypothesis, 'token_confidence', None)
    if token_conf is None:
        return EngineConfidence()

    # torch.Tensor / numpy.ndarray は tolist() で list 化、既存 list/tuple
    # はそのまま (Parakeet も list で来るためカバー)
    if hasattr(token_conf, 'tolist') and not isinstance(token_conf, (list, tuple)):
        try:
            token_conf = token_conf.tolist()
        except Exception:
            return EngineConfidence()

    if isinstance(token_conf, (list, tuple)) and len(token_conf) > 0:
        numeric = []
        for value in token_conf:
            if value is None:
                continue
            try:
                numeric.append(float(value))
            except (TypeError, ValueError):
                continue
        if numeric:
            return EngineConfidence(
                token_confidence_mean=sum(numeric) / len(numeric),
            )

    return EngineConfidence()

# リソースパス解決用のヘルパー関数をインポート
from livecap_cli.utils import (
    get_models_dir,
    detect_device,
    unicode_safe_temp_directory,
    unicode_safe_download_directory,
)

# NeMo framework - 共通モジュールから遅延インポート
from .nemo_utils import check_nemo_availability, prepare_nemo_environment

logger = logging.getLogger(__name__)


class CanaryEngine(BaseEngine):
    """NVIDIA Canary 1B Flash音声認識エンジン - Template Method版"""

    def __init__(
        self,
        device: Optional[str] = None,
        language: str = "en",
        model_name: str = "nvidia/canary-1b-flash",
        beam_size: int = 1,
        **kwargs,
    ):
        """エンジンを初期化

        Args:
            device: 使用するデバイス ("cpu", "cuda", None=auto)
            language: 入力言語 (en, de, fr, es)
            model_name: モデル名
            beam_size: ビームサイズ (1=greedy)
            **kwargs: 追加パラメータ
        """
        # エンジン名を設定
        self.engine_name = 'canary'

        # Category A パラメータ（明示的）
        self.language = language
        self.model_name = model_name
        self.beam_size = beam_size

        super().__init__(device, **kwargs)
        self.model = None
        self._initialized = False

        # デバイスの自動検出と設定（共通関数を使用）
        self.torch_device = detect_device(device, "Canary")

        # ライブラリ事前ロードを開始
        LibraryPreloader.start_preloading('canary')
    
    # ===============================
    # Template Method実装
    # ===============================
    
    def get_model_metadata(self) -> Dict[str, Any]:
        """モデルのメタデータを返す"""
        return {
            'name': self.model_name,
            'version': 'v2',
            'format': 'nemo',
            'description': 'NVIDIA Canary 1B Flash - Multilingual ASR (en, de, fr, es)'
        }
    
    def _check_dependencies(self) -> None:
        """
        Step 1: 依存関係のチェック（0-10%）
        """
        self.report_progress(5, "Checking NeMo availability...")

        # NeMoの利用可能性をチェック（初回のみインポートが試行される）
        if not check_nemo_availability():
            logger.error("NEMO_AVAILABLEがFalseのため、NeMoのインポートエラーを発生させます")
            raise ImportError(
                "NVIDIA NeMo is not installed. Please run: pip install nemo_toolkit[asr]"
            )

        self.report_progress(10, "Dependencies check complete")

    def _get_local_model_path(self, models_dir: Path) -> Path:
        """ローカルモデルパスを取得 (base_engine override for .nemo extension)"""
        return models_dir / f"{self.model_name.replace('/', '--')}.nemo"

    def _prepare_model_directory(self) -> Path:
        """
        Step 2: モデルディレクトリの準備（10-15%）
        """
        self.report_progress(12, "Preparing model directory...")

        # ローカルモデルディレクトリの設定
        models_dir = get_models_dir()
        models_dir.mkdir(exist_ok=True)

        # モデルファイルのパス
        local_model_path = models_dir / f"{self.model_name.replace('/', '--')}.nemo"

        self.report_progress(15, f"Model path: {local_model_path.name}")
        return local_model_path
    
    def _download_model(self, model_path: Path, progress_callback, model_manager=None) -> None:
        """
        Step 3: モデルのダウンロード（15-70%）
        """
        if model_path.exists():
            self.report_progress(70, "Model already downloaded")
            logger.info(f"ローカルファイルが存在: {model_path}")
            return

        self.report_progress(20, f"Downloading model from Hugging Face: {self.model_name}")

        # NeMo 環境準備（PyInstaller 互換性のため）
        prepare_nemo_environment()

        # ここで初めてNeMoモジュールをインポート
        import nemo.collections.asr as nemo_asr
        from nemo.utils import logging as nemo_logging

        # NeMoの警告ログを抑制
        nemo_logger = logging.getLogger('nemo_logger')
        original_level = nemo_logger.level
        nemo_logger.setLevel(logging.ERROR)

        # 追加: Lhotseとデータローダーの警告を抑制
        lhotse_logger = logging.getLogger('lhotse')
        lhotse_original_level = lhotse_logger.level
        lhotse_logger.setLevel(logging.ERROR)

        # NeMo内部の特定警告を抑制
        nemo_collections_logger = logging.getLogger('nemo.collections')
        nemo_collections_original = nemo_collections_logger.level
        nemo_collections_logger.setLevel(logging.ERROR)

        manager = model_manager or getattr(self, "model_manager", None)
        if manager is None:
            from livecap_cli.resources import get_model_manager

            manager = get_model_manager()

        try:
            with unicode_safe_download_directory() as temp_dir:
                logger.info(f"Using download temporary directory: {temp_dir}")

                self.report_progress(30, "Starting model download...")

                with manager.huggingface_cache():
                    model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(
                        model_name=self.model_name,
                        map_location=self.torch_device
                    )

                self.report_progress(60, "Saving model locally...")

                logger.info(f"モデルをローカルに保存: {model_path}")
                model.save_to(str(model_path))

                del model

                self.report_progress(70, "Model download complete")
        finally:
            # すべてのログレベルを元に戻す
            nemo_logger.setLevel(original_level)
            lhotse_logger.setLevel(lhotse_original_level)
            nemo_collections_logger.setLevel(nemo_collections_original)
    
    def _load_model_from_path(self, model_path: Path) -> Any:
        """
        Step 4: モデルファイルからロード（70-90%）
        """
        self.report_progress(75, f"Loading model file: {model_path.name}")

        # キャッシュキーを生成
        cache_key = f"canary_{self.model_name.replace('/', '_')}_{self.torch_device}"

        # キャッシュから取得を試みる
        cached_model = ModelMemoryCache.get(cache_key)
        if cached_model is not None:
            self.report_progress(90, "Loading from cache: Canary")
            logger.info(f"キャッシュからモデルを取得: {cache_key}")
            return cached_model

        # NeMo 環境準備（PyInstaller 互換性のため）
        prepare_nemo_environment()

        # NeMoモジュールをインポート
        import nemo.collections.asr as nemo_asr
        from nemo.utils import logging as nemo_logging

        # NeMoの警告ログを抑制
        nemo_logger = logging.getLogger('nemo_logger')
        original_level = nemo_logger.level
        nemo_logger.setLevel(logging.ERROR)

        # 追加: Lhotseとデータローダーの警告を抑制
        lhotse_logger = logging.getLogger('lhotse')
        lhotse_original_level = lhotse_logger.level
        lhotse_logger.setLevel(logging.ERROR)

        # NeMo内部の特定警告を抑制
        nemo_collections_logger = logging.getLogger('nemo.collections')
        nemo_collections_original = nemo_collections_logger.level
        nemo_collections_logger.setLevel(logging.ERROR)

        try:
            self.report_progress(80, "Restoring NeMo model...")

            # ローカルファイルからロード
            logger.info(f"ローカルファイルからモデルをロード: {model_path}")
            model = nemo_asr.models.EncDecMultiTaskModel.restore_from(
                restore_path=str(model_path),
                map_location=self.torch_device
            )

            self.report_progress(85, "Model loaded successfully")

            # キャッシュに保存
            # 環境変数でstrong cacheが有効な場合は強参照でキャッシュ
            use_strong_cache = os.environ.get('LIVECAP_ENGINE_STRONG_CACHE', '').lower() in ('1', 'true', 'yes')
            ModelMemoryCache.set(cache_key, model, strong=use_strong_cache)
            logger.info(f"モデルをキャッシュに保存: {cache_key} (strong={use_strong_cache})")

            self.report_progress(90, "Canary: Ready")
            return model

        finally:
            # すべてのログレベルを元に戻す
            nemo_logger.setLevel(original_level)
            lhotse_logger.setLevel(lhotse_original_level)
            nemo_collections_logger.setLevel(nemo_collections_original)

    def _configure_model(self) -> None:
        """
        Step 5: モデルの設定（90-100%）
        """
        self.report_progress(92, "Setting model to evaluation mode...")

        # self.modelはload_model_from_pathで既に設定されている
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # 評価モードに設定
        self.model.eval()

        self.report_progress(95, "Updating decoding settings...")

        # PR-A.4.2 (Issue #311): greedy + confidence_cfg.preserve_token_confidence
        # で hypothesis.token_confidence を populate させる (filter signal)。
        # NeMo source 確認済 + 実機 verify (Phase 1 probe):
        #   multitask_greedy_decoding.py:44 (pack_hypotheses): hyp.token_confidence = hyp.frame_confidence
        #   multitask_decoding.py:187: preserve_token_confidence → TransformerAEDGreedyInfer に pass
        #   transformer_generators.py:218-224: confidence tensor 生成
        #   実機 verify: LibriSpeech 英語 → token_confidence_mean=0.0724 (>> threshold 0.005)
        self._configure_decoding_with_confidence()

        self._initialized = True
        self.report_progress(100, "Canary model configuration complete")
        logger.info("モデルの設定が完了しました。")

    def _configure_decoding_with_confidence(self) -> None:
        """Decoding strategy + confidence signal を設定 (3 段 fallback)。

        1. Greedy + confidence_cfg.preserve_token_confidence=True で
           token_confidence が populate される (本 PR の目的、Phase 1 probe 済)。
        2. 1 で TypeError/KeyError → Greedy のみ (confidence なし、fail-open)。
        3. 2 で失敗 → 引数なし呼出し (旧 NeMo API 互換)。
        いずれも raise しない (model 自体は動作させる)。

        Parakeet `parakeet_engine.py:296-365` の pattern 流用。
        """
        # Path 1: Greedy + confidence_cfg
        try:
            decode_cfg = {
                'strategy': 'greedy',
                'confidence_cfg': {
                    'preserve_token_confidence': True,
                    'preserve_frame_confidence': True,
                    'exclude_blank': True,
                    'aggregation': 'mean',
                },
                'greedy': {
                    'preserve_token_confidence': True,
                    'preserve_frame_confidence': True,
                },
            }
            self.model.change_decoding_strategy(decode_cfg)
            logger.info(
                "Canary greedy + token_confidence: enabled "
                "(filter active, threshold token_conf < 0.005)"
            )
            return
        except (TypeError, KeyError, ValueError, AttributeError) as e:
            logger.info(
                f"Canary confidence cfg rejected ({type(e).__name__}: {e}); "
                "falling back to greedy strategy only (fail-open)."
            )

        # Path 2: Greedy only (no confidence)
        try:
            self.model.change_decoding_strategy({'strategy': 'greedy'})
            logger.debug("Canary greedy strategy (no confidence, fail-open)")
            return
        except (TypeError, KeyError, ValueError) as e:
            logger.info(
                f"Canary greedy strategy rejected ({type(e).__name__}); "
                "falling back to argument-less call."
            )

        # Path 3: Argument-less (legacy)
        try:
            self.model.change_decoding_strategy()
        except Exception as inner:
            logger.warning(f"Could not set Canary decoding strategy: {inner}")
            # デコーディング設定失敗してもモデルは使用可能 (filter は fail-open)
    
    # ===============================
    # 既存のインターフェース実装
    # ===============================
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> TranscriptionResult:
        """
        音声データを文字起こしする

        Args:
            audio_data: 音声データ（numpy配列）
            sample_rate: サンプリングレート

        Returns:
            TranscriptionResult: Issue #311 PR-A.4.2 以降は
            ``engine_confidence.token_confidence_mean`` を populate する
            (greedy decoding + ``preserve_token_confidence`` 経由で NeMo
            Hypothesis の token_confidence を mean 化)。``confidence`` field も
            ``token_confidence_mean`` で意味化 (Parakeet と整合)。

        Notes:
            - decoding strategy を beam → greedy に切替済 (PR-A.4.2)。Beam
              decoding は ``preserve_token_confidence`` 未対応のため (NeMo
              ``multitask_beam_decoding.py``)、filter active のために greedy
              を採用。accuracy 重視 user は ``--confidence-filter off`` で
              opt-out 可能。
            - 全 token が空 / fail-open ケースは ``EngineConfidence()`` を
              返し、confidence_filter は pass-through する。
        """
        # Canaryは長時間音声も処理可能
        return self._transcribe_single_chunk(audio_data, sample_rate)

    def _transcribe_single_chunk(self, audio_data: np.ndarray, sample_rate: int) -> TranscriptionResult:
        """
        単一の音声チャンクを文字起こしする（内部使用）
        
        Args:
            audio_data: 音声データ（numpy配列）
            sample_rate: サンプリングレート

        Returns:
            TranscriptionResult (engine_confidence は default 全 None、
            PR-A.4.2 で beam→greedy 切替予定 — Issue [#311] v2.1 参照)
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
        logger.debug(f"Audio data shape: {audio_data.shape}")
        logger.debug(f"Audio duration: {len(audio_data) / self.get_required_sample_rate():.2f} seconds")
        logger.debug(f"Audio max amplitude: {np.abs(audio_data).max():.4f}")
        
        # 音声が短すぎる場合の処理
        min_duration = 0.1  # 最小0.1秒
        min_samples = int(min_duration * self.get_required_sample_rate())
        if len(audio_data) < min_samples:
            logger.warning(f"Audio too short: {len(audio_data)} samples < {min_samples} samples")
            return TranscriptionResult(text="", confidence=1.0)
            
        try:
            # Canaryのtranscribeメソッドはファイルパスを期待するため、
            # 一時ファイルを作成して音声を保存
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_filename = tmp_file.name
                
                # 音声データを一時ファイルに保存
                sf.write(tmp_filename, audio_data, self.get_required_sample_rate())
                
            try:
                # プログレスバーを抑制
                old_tqdm = os.environ.get('TQDM_DISABLE')
                os.environ['TQDM_DISABLE'] = '1'
                
                # 標準出力を一時的にキャプチャ
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                
                try:
                    # 警告を抑制するための環境変数設定
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="You are using a non-tarred dataset")
                        warnings.filterwarnings("ignore", message="Function `_transcribe_output_processing` is deprecated")
                        
                        # PR-A.4.2 probe: return_hypotheses=True で
                        # Hypothesis (token_confidence 含む) を取得
                        outputs = self.model.transcribe(
                            audio=[tmp_filename],
                            batch_size=1,
                            task='asr',  # Automatic Speech Recognition
                            source_lang=self.language,  # 入力音声の言語
                            target_lang=self.language,  # ASRの場合は同じ言語
                            pnc='yes',  # Punctuation and Capitalization
                            return_hypotheses=True,
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
                engine_confidence = EngineConfidence()
                if outputs and len(outputs) > 0:
                    first = outputs[0]
                    text = first.text if hasattr(first, 'text') else str(first)
                    logger.debug(f"Canary transcription: '{text}'")
                    # PR-A.4.2: hypothesis から engine_confidence を populate
                    # (NeMo Hypothesis の token_confidence: torch.Tensor を扱う)
                    engine_confidence = _extract_engine_confidence(first)
                else:
                    text = ""

                # PR-A.4.2: confidence semantics は Parakeet と整合。
                # token_confidence_mean が 0-1 range なので直接 confidence に。
                # populate されない (fail-open) 時は legacy 1.0 fallback。
                if engine_confidence.token_confidence_mean is not None:
                    confidence = float(engine_confidence.token_confidence_mean)
                else:
                    confidence = 1.0

                return TranscriptionResult(
                    text=text,
                    confidence=confidence,
                    engine_confidence=engine_confidence,
                )
                
            finally:
                # 一時ファイルを削除
                if os.path.exists(tmp_filename):
                    os.unlink(tmp_filename)
                
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise
            
    def get_engine_name(self) -> str:
        """エンジン名を取得"""
        return "NVIDIA Canary 1B Flash"
        
    def get_supported_languages(self) -> list:
        """サポートされる言語のリストを取得"""
        # Canary 1B Flashは英語、ドイツ語、フランス語、スペイン語をサポート
        return ["en", "de", "fr", "es"]
        
    def get_required_sample_rate(self) -> int:
        """エンジンが要求するサンプリングレートを取得"""
        # Canaryモデルは16kHzを使用
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
        self._initialized = False
