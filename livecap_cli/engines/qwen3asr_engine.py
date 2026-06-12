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

from .base_engine import BaseEngine, EngineConfidence, TranscriptionResult
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


def _build_prompt(processor: Any, language: Optional[str]) -> str:
    """qwen-asr wrapper の ``_build_text_prompt`` と同 format で prompt を構築 (Issue #318 PR-A.5.2)。

    wrapper 内部の ``Qwen3ASRModel._build_text_prompt(context, force_language)``
    は private method のため、stable contract として local に replicate する
    (Phase 1 probe で確認した format がそのまま使える、~5 行の duplicate)。

    qwen-asr が将来 prompt format を変更した場合、本 helper も追従更新が
    必要 — engine が hallucination を出力するため smoke verify で検出可能。

    Args:
        processor: ``Qwen3ASRProcessor`` (wrapper 内部 attribute、tokenizer 共用)。
        language: qwen-asr API 用言語名 (``"Japanese"`` / ``"English"`` 等)。
            ``None`` の場合は ``language X<asr_text>`` suffix を付けず auto-detect mode に。

    Returns:
        chat template + (optional) ``language X<asr_text>`` suffix の文字列。
    """
    msgs = [
        {"role": "system", "content": "You are a speech recognition model."},
        {"role": "user", "content": [
            {"type": "audio", "audio": ""},
            {"type": "text", "text": ""},
        ]},
    ]
    base = processor.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False
    )
    if language:
        base = base + f"language {language}<asr_text>"
    return base


def _extract_engine_confidence(
    transition_scores: Any,
    gen_tokens: Any,
    special_ids: set,
) -> EngineConfidence:
    """qwen3asr transcribe 戻り値から ``EngineConfidence`` を構築する純粋関数 (Issue #318 PR-A.5.2)。

    Voxtral PR-A.4.1 (``voxtral_engine.py::_extract_engine_confidence``) と
    **schema 互換** で直接流用。``model.generate(output_scores=True,
    repetition_penalty=1.1, no_repeat_ngram_size=3)`` +
    ``model.compute_transition_scores(normalize_logits=True)`` で取得した
    per-token log-prob から ``avg_logprob`` field のみ populate する。

    特殊 token (EOS / PAD / BOS) は generated text の意味を表さず、しばしば
    高 logprob (確実性高) で全体平均を上方に歪ませるため、``special_ids`` で
    除外する。

    Args:
        transition_scores: 1D tensor-like (shape ``(T,)``) of per-token logprobs。
            ``compute_transition_scores(...)[0]`` (batch index 0) を渡す想定。
            ``None`` / 空 / 全要素 special の場合は ``EngineConfidence()`` を返す。
        gen_tokens: 1D tensor-like (shape ``(T,)``) of generated token ids。
            ``outputs.sequences[:, prompt_len:][0]`` を渡す想定。
        special_ids: 除外する token ID の集合 (EOS / PAD / BOS 等)。空でも OK。

    Returns:
        ``EngineConfidence(avg_logprob=...)`` (他 field は全 None)。
        計算不能 case は ``EngineConfidence()`` (= ``is_available=False``、
        confidence_filter の fail-open 規約に則り pass-through される)。

    Pure function として export するのは、PR-A.4.1 Voxtral / PR-A.5.1
    ReazonSpeech と同じ test pattern (実 model 不要、mock tensor で attribute
    pin) で semantics を恒久 pin するため。
    """
    if transition_scores is None or gen_tokens is None:
        return EngineConfidence()

    # tensor-like を list of float / int に変換 (torch.Tensor / np.ndarray / list 対応)
    if hasattr(transition_scores, "tolist"):
        score_list = transition_scores.tolist()
    else:
        score_list = list(transition_scores)
    if hasattr(gen_tokens, "tolist"):
        token_list = gen_tokens.tolist()
    else:
        token_list = list(gen_tokens)

    if len(score_list) == 0 or len(token_list) == 0:
        return EngineConfidence()

    # token と score を zip して special を除外
    paired = [
        (float(s), int(t))
        for s, t in zip(score_list, token_list)
    ]
    masked = [s for s, t in paired if t not in special_ids]

    if not masked:
        return EngineConfidence()

    avg_lp = sum(masked) / len(masked)
    return EngineConfidence(avg_logprob=avg_lp)


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

    # ISO 639-1/3 コード → qwen-asr API が期待する言語名（単一の正データ源）
    QWEN_ASR_LANGUAGE_NAMES: Dict[str, str] = {
        "zh": "Chinese", "en": "English", "yue": "Cantonese",
        "ar": "Arabic", "de": "German", "fr": "French",
        "es": "Spanish", "pt": "Portuguese", "id": "Indonesian",
        "it": "Italian", "ko": "Korean", "ru": "Russian",
        "th": "Thai", "vi": "Vietnamese", "ja": "Japanese",
        "tr": "Turkish", "hi": "Hindi", "ms": "Malay",
        "nl": "Dutch", "sv": "Swedish", "da": "Danish",
        "fi": "Finnish", "pl": "Polish", "cs": "Czech",
        "fil": "Filipino", "fa": "Persian", "el": "Greek",
        "hu": "Hungarian", "mk": "Macedonian", "ro": "Romanian",
    }

    # QWEN_ASR_LANGUAGE_NAMES から派生（単一データ源を維持）
    SUPPORTED_LANGUAGES = list(QWEN_ASR_LANGUAGE_NAMES.keys())
    _QWEN_ASR_LANGUAGE_NAME_LOOKUP: Dict[str, str] = {
        name.lower(): name for name in QWEN_ASR_LANGUAGE_NAMES.values()
    }

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
        self.language = language  # 元の入力を保持（ログ/デバッグ用）
        self._asr_language = self._resolve_language(language)  # qwen-asr API 用
        self.model_name = model_name

        super().__init__(device, **kwargs)
        self.model = None
        self._initialized = False

        # デバイスの自動検出と設定
        self.torch_device = detect_device(device, "Qwen3-ASR")

    # ===============================
    # 言語コード変換
    # ===============================

    @classmethod
    def _resolve_language(cls, language: Optional[str]) -> Optional[str]:
        """ISO 639-1/BCP-47 言語コードを qwen-asr API が期待する言語名に変換

        Args:
            language: 言語コード ("ja", "zh-CN", "Japanese", None など)

        Returns:
            qwen-asr API 用の言語名 ("Japanese" など)、None は自動検出

        Raises:
            ValueError: サポートされていない言語コードの場合
        """
        if language is None or language == "" or language.lower() == "auto":
            return None

        # 言語名そのものが渡された場合はパススルー（大文字小文字不問）
        resolved = cls._QWEN_ASR_LANGUAGE_NAME_LOOKUP.get(language.lower())
        if resolved is not None:
            return resolved

        # BCP-47 正規化 ("zh-CN" → "zh") via EngineMetadata.to_iso639_1()
        from .metadata import EngineMetadata
        try:
            iso_code = EngineMetadata.to_iso639_1(language)
        except Exception:
            iso_code = language.lower()

        # ISO コード → 言語名マッピング
        if iso_code in cls.QWEN_ASR_LANGUAGE_NAMES:
            return cls.QWEN_ASR_LANGUAGE_NAMES[iso_code]

        raise ValueError(
            f"Unsupported language: '{language}'. "
            f"Supported ISO codes: {cls.SUPPORTED_LANGUAGES}"
        )

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

    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> TranscriptionResult:
        """音声データを文字起こしする (PR-A.5.2 Issue #318 で wrapper bypass + confidence 抽出)。

        Returns:
            TranscriptionResult: ``_asr_language`` が指定されている場合は
            ``EngineConfidence.avg_logprob`` を populate する (Voxtral PR-A.4.1
            と同 semantics)。``_asr_language is None`` (auto-detect mode) の
            場合は旧 wrapper.transcribe() path で fail-open。

        Notes:
            PR-A.5.2 の wrapper bypass:
            - ``self.model.model`` (= ``Qwen3ASRForConditionalGeneration``、
              wrapper 内部 attribute) の ``generate()`` を直接呼び、
              ``output_scores=True / repetition_penalty=1.1 /
              no_repeat_ngram_size=3`` を pass する。
            - 内部で ``return_dict_in_generate=True`` が hardcoded のため
              外側で同 kwarg を渡すと TypeError。
            - ``repetition_penalty=1.1 + no_repeat_ngram_size=3`` の理由
              (Phase 1 probe で両言語確認):
                * Japanese desk_tap の 256 token repetition loop を解消
                * English applause の system prompt leak を avg_logprob で
                  filter 可能な水準に低下 (-0.036 → -1.080)
                * 両言語で margin > 0、threshold ``-0.3`` で 100% 分類可能

            WER 軽微退行 caveat (LLM typical 0.5-1%):
            - Voxtral PR-A.4.1 (greedy 切替) / Canary PR-A.4.2 (beam→greedy)
              precedent と同 framing で、filter benefit を優先。
            - WER 重視 user は ``--confidence-filter off`` で旧挙動
              (engine_confidence 全 None、filter 無効) に opt-out 可能。
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

        # 音声が短すぎる場合の処理
        min_duration = 0.1  # 最小 0.1 秒
        min_samples = int(min_duration * required_sr)
        if len(audio_data) < min_samples:
            logger.warning(f"Audio too short: {len(audio_data)} samples < {min_samples} samples")
            return TranscriptionResult(text="", confidence=1.0)

        # PR-A.5.2: language 指定なし (auto-detect mode) は wrapper bypass 不可
        # (prompt format が異なる)、旧 wrapper.transcribe() path で fail-open。
        if self._asr_language is None:
            return self._transcribe_via_wrapper_fallback(audio_data, required_sr)

        # PR-A.5.2: language 指定あり → wrapper bypass で avg_logprob 抽出
        return self._transcribe_with_scores(audio_data, required_sr)

    def _transcribe_via_wrapper_fallback(
        self, audio_data: np.ndarray, required_sr: int
    ) -> TranscriptionResult:
        """旧 wrapper.transcribe() path (auto-detect mode 用、fail-open、PR-A.5.2 Issue #318)。

        ``_asr_language is None`` 時に呼び出され、engine_confidence は default
        (全 None) で confidence_filter は pass-through する。
        """
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_filename = tmp_file.name
                sf.write(tmp_filename, audio_data, required_sr)
            try:
                result = self.model.transcribe(
                    audio=tmp_filename,
                    language=self._asr_language,
                )
                if result and len(result) > 0:
                    first_result = result[0]
                    if hasattr(first_result, 'text'):
                        text = first_result.text
                    elif isinstance(first_result, str):
                        text = first_result
                    else:
                        text = str(first_result)
                else:
                    text = ""
                return TranscriptionResult(text=text, confidence=1.0)
            finally:
                if os.path.exists(tmp_filename):
                    os.unlink(tmp_filename)
        except Exception as e:
            logger.error(f"Qwen3-ASR transcription failed (wrapper fallback): {e}")
            raise

    def _transcribe_with_scores(
        self, audio_data: np.ndarray, required_sr: int
    ) -> TranscriptionResult:
        """Wrapper bypass で `generate(output_scores=True)` 経由で confidence 抽出 (PR-A.5.2)。

        ``self.model.model`` (= 内部 ``Qwen3ASRForConditionalGeneration``) を
        直接呼ぶことで HuggingFace 標準の ``compute_transition_scores`` API
        に乗せる (Voxtral PR-A.4.1 pattern 完全同形)。
        """
        try:
            import torch

            # PR-A.5.2: wrapper 内部 attribute (.model / .processor) に access。
            # qwen-asr update で構造変化した場合 AttributeError → fail-open。
            try:
                inner_model = self.model.model      # Qwen3ASRForConditionalGeneration
                processor = self.model.processor    # Qwen3ASRProcessor
            except AttributeError as ae:
                logger.warning(
                    f"Qwen3-ASR wrapper internal attribute access failed "
                    f"(maybe qwen-asr update): {ae}. Falling back to wrapper.transcribe()."
                )
                return self._transcribe_via_wrapper_fallback(audio_data, required_sr)

            # prompt 構築 (PR-A.5.2: stable contract として local replicate)
            prompt = _build_prompt(processor, language=self._asr_language)

            # processor で audio + text を batch (sub) tensor 化
            inputs = processor(
                text=[prompt],
                audio=[audio_data],
                return_tensors="pt",
                padding=True,
            )
            inputs = inputs.to(inner_model.device).to(inner_model.dtype)

            # PR-A.5.2 主目的: generate に output_scores + repetition_penalty +
            # no_repeat_ngram_size を pass。return_dict_in_generate は wrapper 内部
            # で hardcoded (probe で確認、外側で渡すと TypeError)。
            with torch.no_grad():
                outputs = inner_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    output_scores=True,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3,
                )

            # text decode (wrapper の _infer_asr_transformers と同 logic)
            prompt_len = inputs["input_ids"].shape[1]
            gen_token_ids = outputs.sequences[:, prompt_len:]
            decoded = processor.batch_decode(
                gen_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            text = decoded[0] if decoded else ""

            # avg_logprob 抽出 (Voxtral pattern 完全同形)
            try:
                transition_scores = inner_model.compute_transition_scores(
                    outputs.sequences,
                    outputs.scores,
                    normalize_logits=True,
                )
                # batch idx 0
                ts = transition_scores[0]
                tokens = gen_token_ids[0]

                # special_ids 集合 (tokenizer attribute から)
                tokenizer = getattr(processor, 'tokenizer', None)
                special_ids: set = set()
                for attr in ('eos_token_id', 'pad_token_id', 'bos_token_id'):
                    if tokenizer is not None and hasattr(tokenizer, attr):
                        val = getattr(tokenizer, attr)
                        if val is not None:
                            special_ids.add(int(val))

                engine_confidence = _extract_engine_confidence(
                    transition_scores=ts,
                    gen_tokens=tokens,
                    special_ids=special_ids,
                )
            except Exception as score_err:
                logger.info(
                    f"Qwen3-ASR avg_logprob extraction failed "
                    f"({type(score_err).__name__}: {score_err}); fail-open."
                )
                engine_confidence = EngineConfidence()

            # confidence: Voxtral と同 semantics で exp(avg_logprob) を UI display 値に
            if engine_confidence.avg_logprob is not None:
                confidence = float(np.exp(engine_confidence.avg_logprob))
            else:
                confidence = 1.0

            return TranscriptionResult(
                text=text,
                confidence=confidence,
                engine_confidence=engine_confidence,
            )

        except Exception as e:
            logger.error(f"Qwen3-ASR transcription failed (bypass path): {e}")
            raise

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
