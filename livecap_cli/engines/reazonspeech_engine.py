"""ReazonSpeech K2エンジンの実装 (Template Method版)"""
import os
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np

from .base_engine import BaseEngine, EngineConfidence, TranscriptionResult
from .model_memory_cache import ModelMemoryCache
from .library_preloader import LibraryPreloader

# リソースパス解決用のヘルパー関数をインポート
from livecap_cli.utils import unicode_safe_temp_directory, unicode_safe_download_directory

logger = logging.getLogger(__name__)


def _extract_engine_confidence(result: Any) -> EngineConfidence:
    """sherpa-onnx ``OfflineRecognitionResult`` から engine confidence を抽出 (Issue #317 / PR-A.5.1)。

    ReazonSpeech は ``ys_log_probs`` (per-token log probability、負の値) を
    sherpa-onnx 1.12.39 で expose する (PR plan 段階で実機 verify 済)。
    本 helper は実 sherpa-onnx 不要に schema 抽出ロジックを unit test で
    pin するため module-level pure function として export している
    (Canary / Voxtral / Parakeet と同 pattern)。

    抽出ロジック:

    - ``result.ys_log_probs`` が non-empty iterable of float なら mean を
      ``EngineConfidence.avg_logprob`` に詰める (Voxtral と同 semantics、
      負の log probability、低いほど悪い)。
    - ``raw["ys_log_probs_mean"]`` + ``raw["ys_log_probs_n"]`` に metadata
      を保存 (debug / future calibration 用)。
    - それ以外は全 None の ``EngineConfidence()`` を返す (fail-open)。

    **設計判断 (Issue #317 codex-review Point 1)**: ``ys_log_probs`` は
    **負の log probability** (例: speech mean ≈ -0.07、non_speech mean
    ≈ -0.45) で、Parakeet/Canary の ``token_confidence_mean`` (0-1 range の
    probability) とは semantics が異なる。``token_confidence_mean`` field に
    詰めると ``token_conf_threshold = 0.005`` 比較で speech も全 reject
    される (-0.07 < 0.005)。Voxtral の ``avg_logprob`` field (負の log prob
    semantics) を流用する。

    populate される条件:

    - sherpa-onnx 1.12.39+ (現 livecap-cli 依存版で確認済)
    - ``decoding_method='greedy_search'`` (現 ``reazonspeech_engine.py`` default)
    - int8 / float32 model どちらでも (Phase 5 smoke で verify)
    """
    if result is None:
        return EngineConfidence()
    ys = getattr(result, 'ys_log_probs', None)
    if ys is None:
        return EngineConfidence()
    try:
        ys_list = list(ys) if not isinstance(ys, list) else ys
    except TypeError:
        return EngineConfidence()
    if not ys_list:
        return EngineConfidence()
    numeric = []
    for v in ys_list:
        if v is None:
            continue
        try:
            numeric.append(float(v))
        except (TypeError, ValueError):
            continue
    if not numeric:
        return EngineConfidence()
    mean_lp = sum(numeric) / len(numeric)
    return EngineConfidence(
        avg_logprob=mean_lp,
        raw={
            "ys_log_probs_mean": mean_lp,
            "ys_log_probs_n": len(numeric),
        },
    )

# 最適化された音声処理（存在する場合のみ使用）
try:
    from optimizations.audio_processing_optimized import resample_audio_optimized
    OPTIMIZED_AUDIO_AVAILABLE = True
except ImportError:
    OPTIMIZED_AUDIO_AVAILABLE = False
    logger.debug("Optimized audio processing not available for ReazonSpeech")


class ReazonSpeechEngine(BaseEngine):
    """ReazonSpeech K2を使用した音声認識エンジン（CPU専用） - Template Method版"""

    def __init__(
        self,
        device: Optional[str] = None,
        # カテゴリA: ユーザー向けパラメータ（EngineMetadata.default_params で定義）
        use_int8: bool = False,
        num_threads: int = 4,
        decoding_method: str = "greedy_search",
        # カテゴリB: 内部詳細パラメータ（**kwargs 経由で上書き可能）
        **kwargs,
    ):
        # エンジン名を設定
        self.engine_name = 'reazonspeech'
        self.device = "cpu"  # 常にCPUを使用

        # カテゴリA: ユーザー向けパラメータ
        self.use_int8 = use_int8
        self.num_threads = num_threads
        self.decoding_method = decoding_method

        # カテゴリB: 内部詳細パラメータ（kwargs から取得、デフォルト値はここでハードコード）
        self.auto_split_duration = kwargs.get('auto_split_duration', 30.0)
        self.padding_duration = kwargs.get('padding_duration', 0.9)
        self.padding_threshold = kwargs.get('padding_threshold', 5.0)
        self.min_audio_duration = kwargs.get('min_audio_duration', 0.3)
        self.short_audio_duration = kwargs.get('short_audio_duration', 1.0)
        self.extended_padding_duration = kwargs.get('extended_padding_duration', 2.0)
        self.decode_timeout = kwargs.get('decode_timeout', 5.0)

        # BaseEngine初期化（get_model_metadata()が呼ばれる）
        super().__init__(device, **kwargs)

        # 事前ロード開始
        LibraryPreloader.start_preloading('reazonspeech')

        model_type = "int8" if self.use_int8 else "float32"
        logger.info(f"ReazonSpeech K2 engine initialized for CPU ({model_type} precision, {self.num_threads} threads).")
        if self.auto_split_duration > 0:
            logger.info(f"Auto-splitting enabled for audio > {self.auto_split_duration}s")
    
    def get_model_metadata(self) -> Dict[str, Any]:
        """モデルメタデータを取得"""
        if self.use_int8:
            return {
                'name': 'sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01',
                'version': '2024-08-01',
                'format': 'onnx-int8',
                'language': 'ja',
                'description': 'ReazonSpeech K2 v2 Int8 Quantized Model'
            }
        else:
            return {
                'name': 'reazonspeech-k2-v2',
                'version': 'v2',
                'format': 'onnx',
                'language': 'ja',
                'description': 'ReazonSpeech K2 v2 Float32 Model'
            }
    
    def load_model(self) -> None:
        """モデルをロードする（Windowsパス問題のワークアラウンド付き）"""
        # model_managerへのアクセス（遅延初期化）
        if not hasattr(self, "model_manager"):
            from livecap_cli.resources import get_model_manager
            self.model_manager = get_model_manager()
        
        models_dir = self.model_manager.get_models_dir(self.engine_name)
        model_path = self._get_local_model_path(models_dir)
        
        # Windows Workaround: 既存の古い場所のファイルを正しい場所に移動
        # ダウンロード済みだが場所が間違っている場合（CIキャッシュなど）の救済
        if not model_path.exists():
            # 想定: .../models/reazonspeech/reazon-research--reazonspeech-k2-v2
            # 実態: .../models/reazon-research--reazonspeech-k2-v2
            wrong_path = model_path.parent.parent / model_path.name
            
            if wrong_path.exists() and wrong_path.is_dir():
                logger.warning(f"Workaround: Found ReazonSpeech model at wrong location {wrong_path}, moving to {model_path}")
                try:
                    # 親ディレクトリを確実に作成
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(wrong_path), str(model_path))
                    logger.info("ReazonSpeech model moved successfully.")
                except Exception as e:
                    logger.error(f"Failed to move ReazonSpeech model: {e}")
        
        # 親クラスの標準ロード処理を実行
        super().load_model()

    def _check_dependencies(self) -> None:
        """依存関係チェック (Step 1: 0-10%)"""
        self.report_progress(5, "Checking sherpa-onnx availability...")

        # ライブラリプリロードの完了を待つ（最大2秒）
        LibraryPreloader.wait_for_preload(timeout=2.0)

        # sherpa-onnxの利用可能性をチェック
        try:
            import huggingface_hub as hf
            import sherpa_onnx
            logger.debug("sherpa_onnx imported successfully")

            # バージョンチェック
            try:
                sherpa_version = sherpa_onnx.__version__
                logger.debug(f"sherpa-onnx version: {sherpa_version}")
                # バージョン比較（1.12.9以降を推奨）
                version_parts = sherpa_version.split('.')
                if len(version_parts) >= 3:
                    major, minor, patch = int(version_parts[0]), int(version_parts[1]), int(version_parts[2])
                    if (major < 1) or (major == 1 and minor < 12) or (major == 1 and minor == 12 and patch < 9):
                        logger.warning(f"sherpa-onnx {sherpa_version} is outdated. Please update to 1.12.9+ for better performance.")
            except:
                pass

        except ImportError as e:
            logger.error(f"Failed to import sherpa_onnx: {e}")
            raise ImportError("sherpa_onnx is not installed. Please check ReazonSpeech installation.")

        self.report_progress(10, "Dependencies check complete")
    
    def _get_local_model_path(self, models_dir: Path) -> Path:
        """ローカルモデルパスを取得 (Step 2: 10-15%)"""
        # モデルディレクトリ
        if self.use_int8:
            local_model_dir = models_dir / "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01"
        else:
            local_model_dir = models_dir / "reazon-research--reazonspeech-k2-v2"

        self.report_progress(15, f"Model path: {local_model_dir.name}")
        return local_model_dir
    
    def _verify_model_integrity(self, model_path: Path) -> bool:
        """ReazonSpeech用のモデル完全性チェック（ディレクトリ内のファイル確認）"""
        if not model_path.exists() or not model_path.is_dir():
            return False
        
        epochs = 99
        # 必要なファイルリスト
        if self.use_int8:
            required_files = [
                "tokens.txt",
                f"encoder-epoch-{epochs}-avg-1.int8.onnx",
                f"decoder-epoch-{epochs}-avg-1.onnx",
                f"joiner-epoch-{epochs}-avg-1.int8.onnx",
            ]
        else:
            required_files = [
                "tokens.txt",
                f"encoder-epoch-{epochs}-avg-1.onnx",
                f"decoder-epoch-{epochs}-avg-1.onnx",
                f"joiner-epoch-{epochs}-avg-1.onnx",
            ]
        
        # 全ての必要なファイルが存在するかチェック
        for file_name in required_files:
            file_path = model_path / file_name
            if not file_path.exists():
                logger.debug(f"必要なファイルが見つかりません: {file_path}")
                return False
        
        return True
    
    def _download_model(self, target_path: Path, progress_callback, model_manager=None) -> None:
        """モデルダウンロード (Step 3: 15-70%)"""
        import tarfile
        import huggingface_hub as hf

        epochs = 99
        manager = model_manager or getattr(self, "model_manager", None)
        if manager is None:
            from livecap_cli.resources import get_model_manager

            manager = get_model_manager()
        
        # 必要なファイル（int8またはfloat32モデル）
        if self.use_int8:
            # int8量子化モデル（サイズが小さく高速だが、わずかに精度が低い）
            required_files = {
                "tokens": "tokens.txt",
                "encoder": f"encoder-epoch-{epochs}-avg-1.int8.onnx",
                "decoder": f"decoder-epoch-{epochs}-avg-1.onnx",
                "joiner": f"joiner-epoch-{epochs}-avg-1.int8.onnx",
            }
            model_name = "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01"
            download_url = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{model_name}.tar.bz2"
            
            logger.info(f"sherpa-onnxからInt8モデルをダウンロード: {model_name}")
            self.report_progress(20, "Downloading model: Int8")

            try:
                with unicode_safe_download_directory():
                    archive_path = manager.download_file(
                        download_url,
                        filename=f"{model_name}.tar.bz2",
                        progress_callback=progress_callback,
                    )

                    self.report_progress(60, f"Extracting: {archive_path.name}")

                    with manager.temporary_directory("reazonspeech-extract") as temp_dir:
                        with tarfile.open(archive_path, 'r:bz2') as tar:
                            tar.extractall(temp_dir)

                        extracted_dir = temp_dir / model_name
                        target_path.mkdir(parents=True, exist_ok=True)

                        for file_name in required_files.values():
                            src = extracted_dir / file_name
                            dst = target_path / file_name
                            if src.exists():
                                shutil.copy2(src, dst)
                                logger.info(f"ファイルをコピー: {file_name}")
                            else:
                                logger.error(f"ファイルが見つかりません: {file_name}")

                logger.info(f"Int8モデルをローカルに保存: {target_path}")

            except Exception as e:
                logger.error(f"Int8モデルのダウンロードに失敗: {e}")
                raise
                
        else:
            # float32モデル（最高精度）
            required_files = {
                "tokens": "tokens.txt",
                "encoder": f"encoder-epoch-{epochs}-avg-1.onnx",
                "decoder": f"decoder-epoch-{epochs}-avg-1.onnx",
                "joiner": f"joiner-epoch-{epochs}-avg-1.onnx",
            }
            hf_repo_id = "reazon-research/reazonspeech-k2-v2"
            
            logger.info(f"Hugging FaceからFloat32モデルをダウンロード: {hf_repo_id}")
            self.report_progress(20, "Downloading model: Float32")

            # Unicode対策を適用してダウンロード
            with unicode_safe_download_directory():
                with manager.huggingface_cache() as hf_cache:
                    self.report_progress(30, "Downloading model from Hugging Face...")
                    downloaded_dir = hf.snapshot_download(hf_repo_id, cache_dir=str(hf_cache))

                # ローカルディレクトリにコピー
                self.report_progress(60, "Copying model files...")
                target_path.mkdir(parents=True, exist_ok=True)
                for file_name in required_files.values():
                    src = Path(downloaded_dir) / file_name
                    dst = target_path / file_name
                    if src.exists():
                        shutil.copy2(src, dst)
                        logger.info(f"ファイルをコピー: {file_name}")
                    else:
                        logger.error(f"ファイルが見つかりません: {file_name}")

                logger.info(f"Float32モデルをローカルに保存: {target_path}")

        self.report_progress(70, "Model download complete")
    
    def _load_model_from_path(self, model_path: Path) -> Any:
        """モデルをファイルからロード (Step 4: 70-90%)"""
        import sherpa_onnx
        
        # キャッシュキーを生成
        cache_key = f"reazonspeech_{self.use_int8}_{model_path.name}"
        cached_model = ModelMemoryCache.get(cache_key)
        
        if cached_model is not None:
            logger.info(f"キャッシュからモデルを取得: {cache_key}")
            self.report_progress(90, "Loading from cache: ReazonSpeech")
            return cached_model

        self.report_progress(75, f"Loading model file: {model_path.name}")
        
        epochs = 99
        # 必要なファイル
        if self.use_int8:
            required_files = {
                "tokens": "tokens.txt",
                "encoder": f"encoder-epoch-{epochs}-avg-1.int8.onnx",
                "decoder": f"decoder-epoch-{epochs}-avg-1.onnx",
                "joiner": f"joiner-epoch-{epochs}-avg-1.int8.onnx",
            }
        else:
            required_files = {
                "tokens": "tokens.txt",
                "encoder": f"encoder-epoch-{epochs}-avg-1.onnx",
                "decoder": f"decoder-epoch-{epochs}-avg-1.onnx",
                "joiner": f"joiner-epoch-{epochs}-avg-1.onnx",
            }
        
        basedir = str(model_path)
        
        # sherpa_onnxでモデルをロード（CPU専用、高精度設定）
        try:
            model_type = "Int8" if self.use_int8 else "Float32"
            logger.info(f"Loading {model_type} model with CPU provider ({self.num_threads} threads)...")
            self.report_progress(80, f"Loading {model_type} model...")
            
            # 高精度設定のためのパラメータ
            model = sherpa_onnx.OfflineRecognizer.from_transducer(
                tokens=os.path.join(basedir, required_files["tokens"]),
                encoder=os.path.join(basedir, required_files['encoder']),
                decoder=os.path.join(basedir, required_files['decoder']),
                joiner=os.path.join(basedir, required_files['joiner']),
                num_threads=self.num_threads,  # より多くのスレッドで高精度処理
                sample_rate=16000,
                feature_dim=80,
                decoding_method=self.decoding_method,  # デコーディング方法の設定
                provider="cpu",
                blank_penalty=0.0,  # ブランクペナルティ（デフォルト）
                debug=False  # デバッグモード
            )
            
            self.report_progress(85, "Model loaded successfully")

            # キャッシュに保存（強参照で保持）
            ModelMemoryCache.set(cache_key, model, strong=True)
            logger.debug(f"モデルをキャッシュに保存: {cache_key}")

            self.report_progress(90, "ReazonSpeech: Ready")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model with sherpa_onnx: {e}")
            logger.error(f"Model files directory: {basedir}")
            
            # より詳細なエラー情報を記録
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _configure_model(self) -> None:
        """モデル設定 (Step 5: 90-100%)"""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        self.report_progress(95, "Configuring model...")

        # ReazonSpeechは特別な設定は不要
        precision = "Int8" if self.use_int8 else "Float32"
        logger.info(f"モデルのロードが完了しました。(CPU, {precision}, {self.num_threads} threads)")

        self.report_progress(100, "ReazonSpeech model configuration complete")
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> TranscriptionResult:
        """
        音声データを文字起こしする

        Args:
            audio_data: 音声データ（numpy配列）
            sample_rate: サンプリングレート

        Returns:
            TranscriptionResult: text と confidence=1.0 を持つが、
            ``engine_confidence`` は **常に全 None** (Issue #308 / PR-A.0)。

        Note:
            sherpa-onnx Python bindings for transducer models do not expose
            per-token scores or lattice data — internal C++ scoring is not
            surfaced to Python. As a result, this engine cannot participate in
            the PR-A.1 engine-confidence filter (it will fail-open).
            Users who need engine-level hallucination defense should switch
            to Silero or TenVAD VAD backends (see ``docs/audio-filter-reference.md``).
        """
        duration = len(audio_data) / sample_rate

        # v2.0.6: シンプルな30秒分割（ReazonSpeech開発者推奨）
        if self.auto_split_duration > 0 and duration > self.auto_split_duration:
            return self._transcribe_with_split(audio_data, sample_rate)

        # 通常の処理
        return self._transcribe_single(audio_data, sample_rate)

    def _transcribe_with_split(self, audio_data: np.ndarray, sample_rate: int) -> TranscriptionResult:
        """
        30秒ごとに分割して文字起こし（ReazonSpeech公式推奨方式）

        Args:
            audio_data: 音声データ（numpy配列）
            sample_rate: サンプリングレート

        Returns:
            TranscriptionResult: 上記 transcribe() の docstring を参照。
        """
        duration = len(audio_data) / sample_rate
        logger.debug(f"ReazonSpeech: Splitting {duration:.1f}s audio into {self.auto_split_duration}s chunks")
        
        # 30秒ごとに単純分割
        max_samples = int(self.auto_split_duration * sample_rate)
        segments = []
        
        for i in range(0, len(audio_data), max_samples):
            segment = audio_data[i:i + max_samples]
            segments.append(segment)
        
        # 各セグメントを処理
        results = []
        # PR-A.5.1 (Issue #317): segment 別 engine_confidence を weighted-mean
        # で aggregate するため、(text, engine_confidence) を保持する。
        segment_results = []  # 各 segment の TranscriptionResult を保持
        for i, segment in enumerate(segments):
            seg_duration = len(segment) / sample_rate
            logger.debug(f"ReazonSpeech: Processing segment {i+1}/{len(segments)} ({seg_duration:.1f}s)")

            try:
                # PR-A.5.1 (Issue #317): segment_result は TranscriptionResult。
                # 旧 `text, confidence = ...` の tuple unpack は
                # `TranscriptionResult.__iter__` 削除 (PR #314) で TypeError。
                # `except Exception` が swallow するため長尺 (>30s) audio で
                # 全 segment が silently dropped していた production bug を修正。
                segment_result = self._transcribe_single(segment, sample_rate)
                if segment_result.text:
                    results.append(segment_result.text)
                segment_results.append(segment_result)
            except Exception as e:
                logger.error(f"ReazonSpeech: Error in segment {i+1}: {e}")
                continue

        # 結果を結合
        if not results:
            return TranscriptionResult(text="", confidence=0.0)

        combined_text = ''.join(results)

        # PR-A.5.1: 各 segment の avg_logprob を weighted mean で aggregate
        # (token 数 weight)。空 segment や engine_confidence 不在 segment は
        # skip。total_n == 0 → fail-open (EngineConfidence())。
        total_n = 0
        weighted_sum = 0.0
        for r in segment_results:
            ec = r.engine_confidence
            if ec.avg_logprob is None:
                continue
            n = ec.raw.get("ys_log_probs_n", 0)
            if n > 0:
                total_n += n
                weighted_sum += ec.avg_logprob * n
        if total_n > 0:
            combined_avg = weighted_sum / total_n
            combined_ec = EngineConfidence(
                avg_logprob=combined_avg,
                raw={
                    "ys_log_probs_mean": combined_avg,
                    "ys_log_probs_n": total_n,
                },
            )
        else:
            combined_ec = EngineConfidence()  # fail-open

        return TranscriptionResult(
            text=combined_text,
            confidence=1.0,
            engine_confidence=combined_ec,
        )

    def _transcribe_single(self, audio_data: np.ndarray, sample_rate: int) -> TranscriptionResult:
        """
        単一の音声を文字起こしする（内部使用）

        Args:
            audio_data: 音声データ（numpy配列）
            sample_rate: サンプリングレート

        Returns:
            TranscriptionResult: 上記 transcribe() の docstring を参照。
        """
        if not self._initialized or self.model is None:
            raise RuntimeError("Engine not initialized. Call load_model() first.")

        duration = len(audio_data) / sample_rate

        # 音声の前処理（長さチェックとパディング）
        processed_audio = self._preprocess_audio(audio_data, sample_rate)
        if processed_audio is None:
            return TranscriptionResult(text="", confidence=1.0)  # スキップされた音声

        audio_data = processed_audio

        # サンプルレート変換
        audio_data, sample_rate_to_save = self._ensure_sample_rate(audio_data, sample_rate)

        try:
            # 文字起こし実行 (PR-A.5.1: full sherpa-onnx result を取得)
            sherpa_result = self._execute_transcription(audio_data, sample_rate_to_save, duration)
            if sherpa_result is None:
                # decode timeout (fail-open、空 transcription)
                return TranscriptionResult(text="", confidence=1.0)

            result_text = (sherpa_result.text or "").strip()
            # PR-A.5.1 (Issue #317): ys_log_probs を avg_logprob に populate
            # (Voxtral と同 semantics、reviewer Point 1/2 で確定設計)
            engine_confidence = _extract_engine_confidence(sherpa_result)

            return TranscriptionResult(
                text=result_text,
                confidence=1.0,
                engine_confidence=engine_confidence,
            )

        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise
                
    def get_engine_name(self) -> str:
        """エンジン名を取得"""
        precision = "Int8" if self.use_int8 else "Float32"
        return f"ReazonSpeech K2 (CPU, {precision})"
        
    def get_supported_languages(self) -> list:
        """サポートされる言語のリストを取得"""
        return ["ja"]  # 日本語のみサポート
        
    def get_required_sample_rate(self) -> int:
        """エンジンが要求するサンプリングレートを取得"""
        return 16000

    def cleanup(self) -> None:
        """リソースのクリーンアップ"""
        if self.model is not None:
            del self.model
            self.model = None
        self._initialized = False
    
    # === ヘルパーメソッド（リファクタリング） ===
    
    def _preprocess_audio(self, audio_data: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
        """
        音声データの前処理（長さチェックとパディング）
        
        Returns:
            処理済み音声データ、またはNone（スキップの場合）
        """
        duration = len(audio_data) / sample_rate
        
        # 極短音声のスキップ
        if duration < self.min_audio_duration:
            logger.warning(f"ReazonSpeech: Extremely short audio detected "
                          f"({duration:.2f}s < {self.min_audio_duration}s), skipping")
            return None
        
        # パディング適用
        return self._apply_padding(audio_data, duration, sample_rate)
    
    def _apply_padding(self, audio_data: np.ndarray, duration: float, sample_rate: int) -> np.ndarray:
        """音声にパディングを適用"""
        if duration < self.short_audio_duration:
            # 短音声への拡張パディング
            padding_duration = self.extended_padding_duration
            logger.warning(f"ReazonSpeech: Very short audio detected ({duration:.2f}s), "
                          f"applying extended padding")
        elif duration < self.padding_threshold and self.padding_duration > 0:
            # 通常のパディング
            padding_duration = self.padding_duration
            logger.debug(f"ReazonSpeech: Adding standard padding to {duration:.2f}s audio")
        else:
            # パディング不要
            return audio_data
        
        # パディングを作成して適用
        padding_samples = int(sample_rate * padding_duration)
        padding = np.zeros(padding_samples, dtype=audio_data.dtype)
        padded_audio = np.concatenate([padding, audio_data, padding])
        
        total_padding = padding_duration * 2
        logger.debug(f"ReazonSpeech: Added {total_padding:.1f}s padding to {duration:.2f}s audio")
        
        return padded_audio
    
    def _ensure_sample_rate(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, int]:
        """サンプルレートを確認し、必要に応じて変換"""
        target_rate = 16000  # ReazonSpeechは16kHz固定
        if sample_rate != target_rate:
            if OPTIMIZED_AUDIO_AVAILABLE:
                # 最適化されたリサンプリング（キャッシュ付き）
                audio_data = resample_audio_optimized(
                    audio_data,
                    sample_rate,
                    target_rate
                )
            else:
                # 標準実装
                import librosa
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=sample_rate, 
                    target_sr=target_rate
                )
            return audio_data, target_rate
        return audio_data, sample_rate
    
    def _execute_transcription(self, audio_data: np.ndarray, sample_rate: int, duration: float) -> Any:
        """実際の文字起こし処理を実行（タイムアウト付き）。

        Returns:
            sherpa-onnx ``OfflineRecognitionResult`` (`.text` + `.ys_log_probs`
            等を持つ object) または timeout 時 ``None``。

            PR-A.5.1 (Issue #317) から caller が ``ys_log_probs`` を読めるよう
            full result object を返す (旧版は ``result.text`` だけ抽出していた)。
        """
        # ストリームを作成
        stream = self.model.create_stream()

        # 音声データを正規化してストリームに送信
        audio_float32 = self._normalize_audio(audio_data)
        stream.accept_waveform(sample_rate, audio_float32)

        # タイムアウト付きでデコード実行
        result = self._decode_with_timeout(stream, duration)

        return result  # PR-A.5.1: full sherpa-onnx OfflineRecognitionResult (or None)
    
    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """音声データをfloat32に変換し正規化"""
        audio_float32 = audio_data.astype(np.float32)
        max_val = np.abs(audio_float32).max()
        if max_val > 1.0:
            audio_float32 = audio_float32 / max_val
        return audio_float32
    
    def _decode_with_timeout(self, stream, duration: float):
        """タイムアウト付きでデコードを実行"""
        import threading
        
        decode_result = [None]
        decode_exception = [None]
        
        def decode_thread():
            try:
                self.model.decode_stream(stream)
                decode_result[0] = stream.result
            except Exception as e:
                decode_exception[0] = e
        
        # デコードスレッドを起動
        thread = threading.Thread(target=decode_thread)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.decode_timeout)
        
        # タイムアウトチェック
        if thread.is_alive():
            logger.error(f"ReazonSpeech: decode_stream timeout after {self.decode_timeout}s "
                         f"(duration={duration:.2f}s)")
            return None
        
        # 例外チェック
        if decode_exception[0]:
            raise decode_exception[0]
        
        return decode_result[0]
