"""ReazonSpeech K2エンジンの実装 (Template Method版)"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np

from .base_engine import BaseEngine, EngineConfidence, TranscriptionResult
from .metadata import EngineMetadata
from .legacy_model_layouts import remove_legacy_archives
from .model_memory_cache import ModelMemoryCache
from .repo_dir_engine import RepoDirModelMixin, RepoDirSpec
from .library_preloader import LibraryPreloader
from .reazonspeech_cache import (
    ModelIdentityChangedError,
    build_identity,
    required_files,
    resolve_model_files,
)

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
    詰めると ``token_conf_threshold = 0.001`` 比較で speech も全 reject
    される (-0.07 < 0.001)。Voxtral の ``avg_logprob`` field (負の log prob
    semantics) を流用しつつ、engine-specific threshold ``-0.40`` ([#334] PR-4
    で Phase 2 report §2.1 Pareto relaxed_B、旧 ``-0.2``) を
    ``FilterConfig.avg_logprob_thresholds["reazonspeech"]`` で適用する。

    populate される条件:

    - sherpa-onnx 1.12.39+ (expose された版。現 livecap-cli 依存版は 1.13.6 で、
      両版とも ``avg_logprob`` が一致することを #377 で実測済)
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


class ReazonSpeechEngine(RepoDirModelMixin, BaseEngine):
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
                'name': 'reazonspeech-k2-v2-int8',
                'version': 'v2',
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
    
    def _check_dependencies(self) -> None:
        """依存関係チェック (Step 1: 0-10%)"""
        self.report_progress(5, "Checking sherpa-onnx availability...")

        # ライブラリプリロードの完了を待つ（最大2秒）
        LibraryPreloader.wait_for_preload(timeout=2.0)

        # sherpa-onnxの利用可能性をチェック
        try:
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
    
    #: 両 precision とも同じ HF repo にある (int8 も `*.int8.onnx` として)。
    HF_REPO_ID = "reazon-research/reazonspeech-k2-v2"
    #: 旧 workaround が作っていた engine subdir (`<models_root>/reazonspeech/<name>`)。
    LEGACY_SUBDIRS = ("reazonspeech",)
    #: int8 の正本 dir 名。同じ repo の variant なので `<org>--<name>-int8` (float32 は `<org>--<name>`)。
    INT8_DIR_NAME = "reazon-research--reazonspeech-k2-v2-int8"
    #: #456 以前の int8 dir 名 (tarball 由来)。初回ロードで `INT8_DIR_NAME` へ取り込んで消す
    LEGACY_INT8_DIR_NAME = "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01"
    #: #456 以前の int8 経路が `<cache_root>/downloads/` に残したまま展開していた tarball (713 MB)。
    #: int8 の正本が validator を通った後に消す (#456 PR 1 手順)。
    LEGACY_INT8_ARCHIVE = "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2"

    @property
    def _variant(self) -> str:
        return "int8" if self.use_int8 else "float32"

    def _repo_dir_spec(self) -> RepoDirSpec:
        """int8 / float32 とも同じ HF repo から ``required_files()`` の 4 ファイルだけを取る (#456)。

        以前は float32 が repo 全体 (775 MB、int8 の encoder を含む) を ``<cache_root>`` へ落としてから
        copy し、int8 は GitHub の tarball (713 MB、float32 encoder + test_wavs を含む) を
        ``<cache_root>/downloads`` に**残したまま**展開していた。必要量は float32 615 MB / int8 160 MB。
        ファイル名の出所は ``required_files()`` だけ (Issue #409)。取り込み対象は、この dir 自身
        (manifest 無し = #456 以前の配置)、engine subdir の重複 (旧 ``load_model()`` override が作った)、
        ``<cache_root>/huggingface/*`` の旧 snapshot。
        """
        required = tuple(required_files(use_int8=self.use_int8).values())
        return RepoDirSpec(
            repo_id=self.HF_REPO_ID,
            required=required,
            variant=self._variant,
            allow_patterns=required,
            legacy_subdirs=self.LEGACY_SUBDIRS,
            dir_name_override=self.INT8_DIR_NAME if self.use_int8 else None,
            legacy_names=(self.LEGACY_INT8_DIR_NAME,) if self.use_int8 else (),
        )

    def _reconcile_legacy_layouts(self, model_path: Path) -> None:
        super()._reconcile_legacy_layouts(model_path)
        # 旧 int8 経路の tarball は、int8 の正本が validator を通った後にだけ消す (検証前には触らない)
        if self.use_int8 and self._is_model_cached(model_path):
            remove_legacy_archives(self.model_manager.cache_root, [self.LEGACY_INT8_ARCHIVE])

    def _download_model(self, target_path: Path, progress_callback) -> None:
        super()._download_model(target_path, progress_callback)
        if self.use_int8:
            # 取得直後 (validator を通った正本ができた後) にも旧 tarball を消す
            remove_legacy_archives(self.model_manager.cache_root, [self.LEGACY_INT8_ARCHIVE])

    def _load_model_from_path(self, model_path: Path) -> Any:
        """モデルをファイルからロード (Step 4: 70-90%)"""
        import sherpa_onnx
        
        # **identity は cache lookup より前に確定させる** (Issue #409)。
        # 旧 key は use_int8 と basename しか見ておらず、異なる models root の同名
        # ディレクトリが衝突し、モデルを差し替えても古い recognizer が返っていた。
        # 算出に失敗したら**そのまま落とす** — ここで fallback すると
        # 「identity を取れないときは簡易 key を使う」経路を作り込むことになる。
        identity_kwargs = dict(
            use_int8=self.use_int8,
            num_threads=self.num_threads,
            decoding_method=self.decoding_method,
        )
        model_files = resolve_model_files(model_path, use_int8=self.use_int8)
        identity_before = build_identity(model_path, **identity_kwargs)
        cache_key = identity_before.cache_key()

        cached_model = ModelMemoryCache.get(cache_key)
        if cached_model is not None:
            logger.info(f"キャッシュからモデルを取得: {cache_key}")
            self.report_progress(90, "Loading from cache: ReazonSpeech")
            return cached_model

        self.report_progress(75, f"Loading model file: {model_path.name}")

        basedir = str(model_path)
        
        # sherpa_onnxでモデルをロード（CPU専用、高精度設定）
        try:
            model_type = "Int8" if self.use_int8 else "Float32"
            logger.info(f"Loading {model_type} model with CPU provider ({self.num_threads} threads)...")
            self.report_progress(80, f"Loading {model_type} model...")
            
            # 高精度設定のためのパラメータ
            model = sherpa_onnx.OfflineRecognizer.from_transducer(
                # **identity が見たのと同じファイルを渡す** — 別々に組み立てると
                # hash 対象と実際に読むファイルがずれ得る (Issue #409)。
                tokens=str(model_files["tokens"]),
                encoder=str(model_files["encoder"]),
                decoder=str(model_files["decoder"]),
                joiner=str(model_files["joiner"]),
                num_threads=self.num_threads,  # より多くのスレッドで高精度処理
                sample_rate=16000,
                feature_dim=80,
                decoding_method=self.decoding_method,  # デコーディング方法の設定
                provider="cpu",
                blank_penalty=0.0,  # ブランクペナルティ（デフォルト）
                debug=False  # デバッグモード
            )
            
            self.report_progress(85, "Model loaded successfully")

            # **構築中にモデルが変わっていないか確かめる** (Issue #409)。
            # 変わっていた場合に保存すると、**古い identity のキーへ新しい内容の
            # recognizer が入る**。黙って保存するくらいなら落とす。
            identity_after = build_identity(model_path, **identity_kwargs)
            if identity_after != identity_before:
                raise ModelIdentityChangedError(
                    "ReazonSpeech model files changed while the recognizer was being "
                    f"built ({ascii(str(model_path))}). The recognizer is not cached "
                    "because its cache identity would be stale."
                )

            # キャッシュに保存（強参照で保持）
            # **健全性の判定は行わない** — post-load health check と保存ゲートは #392 の責務。
            ModelMemoryCache.set(cache_key, model, strong=True)
            logger.debug(f"モデルをキャッシュに保存: {cache_key}")

            self.report_progress(90, "ReazonSpeech: Ready")
            return model
            
        except ModelIdentityChangedError:
            raise
        except Exception as e:
            logger.error(f"Failed to load model with sherpa_onnx: {e}")
            logger.error(f"Model files directory: {basedir}")
            # **self-heal**: manifest に無い形で ONNX が壊れている場合、manifest を残すと
            # 以後ダウンロード phase を永久に skip して落ち続ける (#456)
            self._invalidate_model_dir(model_path, reason=f"ReazonSpeech from_transducer failed: {e}")
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
            TranscriptionResult: text + confidence=1.0 +
            ``engine_confidence.avg_logprob`` (sherpa-onnx 1.12.39+ の ``ys_log_probs``
            mean、PR-A.5.1 [#317] から populate)。負の log probability、低いほど
            engine confidence が低い。engine-specific threshold
            ``-0.40`` ([#334] PR-4 で Phase 2 report §2.1 Pareto relaxed_B
            に更新、旧 ``-0.2``、``FilterConfig.avg_logprob_thresholds["reazonspeech"]``)
            で reject 判定される。

        Note:
            sherpa-onnx 1.12.39 で ``OfflineRecognitionResult.ys_log_probs`` が
            expose されるようになり、本 engine も PR-A.1 confidence_filter の
            reject 対象となった (PR-A.5.1)。量子化 (int8/float32) と calibration
            data の整合性は Issue #334 Finding 8 で議論中。
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
                segment_result = self._transcribe_single(segment, sample_rate)
                # 空 text segment は weighted aggregate に含めない
                # (空 text + 低 avg_logprob が combined avg を下げて実テキスト
                # を reject するのを回避)。
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
        # 上記 if 段で除外済。total_n == 0 → fail-open (EngineConfidence())。
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
        """サポートされる言語のリストを取得 (正本は EngineMetadata、#230)"""
        return list(EngineMetadata.get(self.engine_name).supported_languages)
        
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
