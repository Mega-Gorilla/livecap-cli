"""Qwen3-ASR 音声認識エンジンの実装 - Template Method版

Alibaba Cloud Qwen チームが開発した Qwen3-ASR を統合。
30言語以上をサポートし、Whisper-large-v3 を上回る精度を実現。

**transformers 本体の実装** (``Qwen3ASRForConditionalGeneration`` + ``AutoProcessor``) と
``-hf`` の repo (``Qwen/Qwen3-ASR-{0.6B,1.7B}-hf``) を使う (#471)。以前の ``qwen-asr`` package は
``transformers==4.57.6`` に固定していて transformers 5 系と両立しないので外した。重みは ``-hf`` 版と
非 ``-hf`` 版で同じ (名前の付け替えだけ、#471 で全 tensor を照合) で、同じ設定で推論した出力も一致する。

References:
    - https://github.com/QwenLM/Qwen3-ASR
    - https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from .base_engine import BaseEngine, EngineConfidence, TranscriptionResult
from .model_memory_cache import ModelMemoryCache
from .repo_dir_engine import RepoDirModelMixin, RepoDirSpec
from .qwen3asr_languages import QWEN_ASR_LANGUAGE_NAMES as _QWEN_ASR_LANGUAGE_NAMES

from livecap_cli.paths import ascii_safe_temp_environment
from livecap_cli.utils import detect_device, resolve_torch_dtype

logger = logging.getLogger(__name__)

# transformers の Qwen3-ASR が使えるか (lazy)
QWEN3ASR_AVAILABLE: Optional[bool] = None

#: 言語を指定したときの system prompt。PR-A.5.2 (Issue #318) で confidence 抽出の経路を作ったときの
#: 値で、avg_logprob の閾値 (``-0.42``、#334) はこの prompt で調整されている
_FORCED_LANGUAGE_SYSTEM_PROMPT = "You are a speech recognition model."

#: 言語を指定したときの生成の設定 (PR-A.5.2)。``repetition_penalty`` / ``no_repeat_ngram_size`` の理由は
#: :meth:`Qwen3ASREngine.transcribe` を参照
_FORCED_LANGUAGE_GENERATE_KWARGS: Mapping[str, Any] = {
    "max_new_tokens": 256,
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 3,
}

#: 自動検出のときの生成の設定。以前の ``qwen-asr`` の ``transcribe()`` と同じ (system prompt は空、
#: penalty なし、``max_new_tokens=512``)
_AUTO_LANGUAGE_GENERATE_KWARGS: Mapping[str, Any] = {"max_new_tokens": 512}


@dataclass(frozen=True)
class Qwen3ASRBundle:
    """ロード済みの model と processor (``ModelMemoryCache`` に 1 単位で載せる)。"""

    model: Any  # transformers.Qwen3ASRForConditionalGeneration
    processor: Any  # transformers.Qwen3ASRProcessor


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


def _special_token_ids(processor: Any, model: Any) -> set:
    """avg_logprob から除外する token ID (tokenizer の EOS / PAD / BOS と、生成を止める token 全部)。

    ``-hf`` の tokenizer は以前の版と ``pad_token_id`` が違う (151643 → 151645) ので、tokenizer の
    3 つだけだと、生成を止めた token (``generation_config.eos_token_id`` は 151643 と 151645 の 2 つ)
    の片方が平均に入りうる。止める token も明示的に除外し、どちらの版でも同じ集合にする (#471)。
    """
    tok = processor.tokenizer
    ids = {
        getattr(tok, "eos_token_id", None),
        getattr(tok, "pad_token_id", None),
        getattr(tok, "bos_token_id", None),
    }
    generation_config = getattr(model, "generation_config", None)
    for name in ("eos_token_id", "pad_token_id", "bos_token_id"):
        value = getattr(generation_config, name, None)
        if isinstance(value, (list, tuple, set)):
            ids.update(value)
        else:
            ids.add(value)
    ids.discard(None)
    return {int(i) for i in ids}


def _require_complete_weights(loading_info: Mapping[str, Any], model_dir: Path) -> None:
    """重みが model と過不足なく対応していることを確かめる (欠けた重みを乱数で初期化したまま使わない)。

    ``from_pretrained`` は対応の無い重みを**警告だけで**乱数初期化する。別形式の dir (非 ``-hf`` 版の
    ``thinker.*`` など) や壊れた dir を渡すと、ロードは成功してでたらめな文字を出し続けるので fail loud
    にする (呼び出し側が manifest を無効化し、次回の取得で取り直す)。
    """
    problems = {
        key: list(loading_info.get(key) or [])
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys")
    }
    problems = {key: value for key, value in problems.items() if value}
    if problems:
        summary = "; ".join(f"{key}={len(value)} (例: {value[:3]})" for key, value in problems.items())
        raise RuntimeError(f"Qwen3-ASR の重みが model と対応しない: {model_dir}: {summary}")


def check_qwen3asr_availability() -> bool:
    """transformers の Qwen3-ASR (``Qwen3ASRForConditionalGeneration``) が使えるかを確かめる。

    PyInstaller (frozen) 環境では importlib.util.find_spec() で
    パッケージの存在確認のみを行い、循環インポートを回避する。

    Returns:
        bool: 使える場合 True
    """
    global QWEN3ASR_AVAILABLE
    if QWEN3ASR_AVAILABLE is not None:
        return QWEN3ASR_AVAILABLE

    # PyInstaller 環境では find_spec のみ使用
    if getattr(sys, 'frozen', False):
        try:
            QWEN3ASR_AVAILABLE = importlib.util.find_spec("transformers") is not None
            if not QWEN3ASR_AVAILABLE:
                logger.warning("transformers がインストールされていません")
        except Exception as e:
            QWEN3ASR_AVAILABLE = False
            logger.warning(f"transformers の可用性チェックに失敗: {e}")
        return QWEN3ASR_AVAILABLE

    # 通常環境では実際にインポートを試行
    try:
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration  # noqa: F401
        QWEN3ASR_AVAILABLE = True
    except ImportError as e:
        QWEN3ASR_AVAILABLE = False
        logger.error(f"transformers の Qwen3-ASR をインポートできない: {e}")

    return QWEN3ASR_AVAILABLE


def prepare_qwen3asr_environment() -> None:
    """Qwen3-ASR のロード前の環境準備 (PyInstaller 環境だけ)。

    :meth:`Qwen3ASREngine.transcribe` は 16 kHz 以外の音声を librosa で resample する。PyInstaller の
    frozen importer では librosa のサブモジュール間で循環インポートが起きる (#219) ので、ロード時に
    先にインポートしておく。
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


class Qwen3ASREngine(RepoDirModelMixin, BaseEngine):
    """Qwen3-ASR 音声認識エンジン - Template Method版

    Alibaba Cloud Qwen チームが開発した高精度 ASR エンジン。
    30言語以上をサポートし、言語自動検出機能を持つ。

    Attributes:
        engine_name: エンジン識別子 (qwen3asr / qwen3asr_large)
        language: 入力言語 (None = 自動検出)
        model_name: HuggingFace モデル名
    """

    # ISO 639-1/3 コード → Qwen3-ASR が期待する言語名
    # 正本は qwen3asr_languages.py (#230) — EngineMetadata も同 module から派生。
    # class 属性 alias は後方互換 (_resolve_language / tests が参照) のため維持。
    # MappingProxyType のため item 代入不可 (metadata との split-brain 防止)。
    QWEN_ASR_LANGUAGE_NAMES: Mapping[str, str] = _QWEN_ASR_LANGUAGE_NAMES

    # QWEN_ASR_LANGUAGE_NAMES から派生（単一データ源を維持、tuple で不変化）
    SUPPORTED_LANGUAGES: Tuple[str, ...] = tuple(QWEN_ASR_LANGUAGE_NAMES)
    _QWEN_ASR_LANGUAGE_NAME_LOOKUP: Dict[str, str] = {
        name.lower(): name for name in QWEN_ASR_LANGUAGE_NAMES.values()
    }

    def __init__(
        self,
        device: Optional[str] = None,
        language: Optional[str] = None,
        model_name: str = "Qwen/Qwen3-ASR-0.6B-hf",
        engine_id: str = "qwen3asr",
        dtype: str = "auto",
        **kwargs,
    ):
        """エンジンを初期化

        Args:
            device: 使用するデバイス ("cpu", "cuda", None=auto)
            language: 入力言語 (None = 自動検出)
            model_name: HuggingFace モデル名 (transformers 本体の形式の ``-hf`` repo)
            engine_id: エンジン識別子 (metadata から渡される)
            dtype: 推論 dtype (``auto`` / ``bfloat16`` / ``float16`` / ``float32``)。``auto`` は
                bf16 対応 GPU で ``bfloat16``、それ以外で ``float32``
                (:func:`livecap_cli.utils.resolve_torch_dtype`)
            **kwargs: 追加パラメータ
        """
        # エンジン名を設定（qwen3asr / qwen3asr_large を区別）
        self.engine_name = engine_id

        # パラメータ設定
        self.language = language  # 元の入力を保持（ログ/デバッグ用）
        self._asr_language = self._resolve_language(language)  # Qwen3-ASR の言語名 (None = 自動検出)
        self.model_name = model_name

        super().__init__(device, **kwargs)
        self.model: Optional[Qwen3ASRBundle] = None
        self._initialized = False

        # デバイスの自動検出と設定
        self.torch_device = detect_device(device, "Qwen3-ASR")
        self.dtype = resolve_torch_dtype(dtype, self.torch_device, "Qwen3-ASR")

    # ===============================
    # 言語コード変換
    # ===============================

    @classmethod
    def _resolve_language(cls, language: Optional[str]) -> Optional[str]:
        """ISO 639-1/BCP-47 言語コードを Qwen3-ASR が期待する言語名に変換

        Args:
            language: 言語コード ("ja", "zh-CN", "Japanese", None など)

        Returns:
            Qwen3-ASR の言語名 ("Japanese" など)、None は自動検出

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
        self.report_progress(5, "Checking transformers (Qwen3-ASR) availability...")

        if not check_qwen3asr_availability():
            raise ImportError(
                "Qwen3-ASR requires transformers>=5.22 (Qwen3ASRForConditionalGeneration). "
                "Please run: pip install 'livecap-cli[engines-qwen3asr]'"
            )

        self.report_progress(10, "Dependencies check complete")

    #: repo から取らないファイル (重みでも tokenizer でもない)。
    IGNORE_PATTERNS = ("README.md", ".gitattributes")
    #: cache hit に最低限要るファイル (``-hf`` repo の config + 重み + tokenizer + processor + chat
    #: template + 生成の設定)。``generation_config.json`` は生成を止める token (151643 / 151645) と
    #: ``do_sample=False`` を持つので必須。``model.safetensors`` の要求は分割形式 (index + 全 shard) でも
    #: 満たされる (``model_store.missing_required``、#470)
    REQUIRED_FILES = (
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "chat_template.jinja",
    )

    def _repo_dir_spec(self) -> RepoDirSpec:
        """正本は ``<models_root>/<org>--<name>/`` (``Qwen--Qwen3-ASR-0.6B-hf`` / ``Qwen--Qwen3-ASR-1.7B-hf``、
        flattened dir + manifest、Issue #456 / #470 / #471)。

        model / processor には repo ID ではなくこのローカル dir を渡す (``from_pretrained(<repo id>)`` は
        既定 HF cache へ行く、#428)。#428〜#456 の間にあった marker (``*.marker``) と
        ``<cache_root>/huggingface/hub`` の snapshot は初回 cold load で取り込んで消す
        (:class:`RepoDirModelMixin`)。
        """
        return RepoDirSpec(
            repo_id=self.model_name,
            required=self.REQUIRED_FILES,
            ignore_patterns=self.IGNORE_PATTERNS,
        )

    def _load_model_from_path(self, model_path: Path) -> Qwen3ASRBundle:
        """Step 4: モデルファイルからロード（70-90%）"""
        self.report_progress(75, f"Loading model: {self.model_name}")

        # キャッシュキーを生成
        cache_key = f"qwen3asr_{self.model_name.replace('/', '_')}_{self.torch_device}_{self.dtype}"

        # キャッシュから取得を試みる
        cached_model = ModelMemoryCache.get(cache_key)
        if cached_model is not None:
            self.report_progress(90, "Loading from cache: Qwen3-ASR")
            logger.info(f"キャッシュからモデルを取得: {cache_key}")
            return cached_model

        # 環境準備（PyInstaller 互換性）
        prepare_qwen3asr_environment()

        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

        self.report_progress(80, "Initializing Qwen3-ASR model...")

        # **models_root 内のローカル dir** を渡す (Issue #428 / #456)。repo ID を渡すと
        # model / processor が既定の ~/.cache/huggingface から解決してしまう。
        snapshot = self._require_model_dir(model_path)

        # ローカル path なので通常はネットワークへ出ないが、from_pretrained の内部
        # (transformers) が temp を触る経路は残るため wrapper は維持する (#434 の範囲)。
        try:
            with ascii_safe_temp_environment(
                boundary="engine.qwen3asr.from_pretrained", purpose="download"
            ):
                processor = AutoProcessor.from_pretrained(str(snapshot))
                # device_map: "cpu" はそのまま、"cuda" は "auto" に変換
                # "auto" は利用可能な GPU を自動選択する
                device_map = "auto" if self.torch_device == "cuda" else self.torch_device
                # dtype は from_pretrained で渡す (後から .to() で変えると、読み込み中は
                # fp32 の重みが GPU に載る)。transformers は dtype を名前 ("bfloat16" 等) でも受ける
                model, loading_info = Qwen3ASRForConditionalGeneration.from_pretrained(
                    str(snapshot),
                    device_map=device_map,
                    dtype=self.dtype,
                    output_loading_info=True,
                )
            _require_complete_weights(loading_info, snapshot)
        except Exception:
            # **self-heal**: dir が壊れている (manifest には無い形の破損) 場合、manifest を
            # 残すと以後ダウンロード phase を永久に skip して落ち続ける。
            self._invalidate_model_dir(model_path, reason="Qwen3-ASR from_pretrained failed")
            raise
        model.eval()
        bundle = Qwen3ASRBundle(model=model, processor=processor)

        self.report_progress(85, "Model loaded successfully")

        # キャッシュに保存
        use_strong_cache = os.environ.get('LIVECAP_ENGINE_STRONG_CACHE', '').lower() in ('1', 'true', 'yes')
        ModelMemoryCache.set(cache_key, bundle, strong=use_strong_cache)
        logger.info(f"モデルをキャッシュに保存: {cache_key} (strong={use_strong_cache})")

        self.report_progress(90, "Qwen3-ASR: Ready")
        return bundle

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
        """音声データを文字起こしする。

        Returns:
            TranscriptionResult: ``_asr_language`` が指定されている場合は
            ``EngineConfidence.avg_logprob`` を populate する (Voxtral PR-A.4.1
            と同 semantics)。``_asr_language is None`` (auto-detect mode) の
            場合は confidence を出さない (fail-open、PR-A.5.2 から変えていない)。

        Notes:
            言語を指定したとき (PR-A.5.2 Issue #318):
            - ``generate()`` に ``output_scores=True / repetition_penalty=1.1 /
              no_repeat_ngram_size=3`` を pass し、``compute_transition_scores`` で
              avg_logprob を取る。
            - ``repetition_penalty=1.1 + no_repeat_ngram_size=3`` の理由
              (Phase 1 probe で両言語確認):
                * Japanese desk_tap の 256 token repetition loop を解消
                * English applause の system prompt leak を avg_logprob で
                  filter 可能な水準に低下 (-0.036 → -1.080)
                * 両言語で margin > 0、threshold ``-0.42`` ([#334] PR-4 で
                  Phase 2 report §2.2 Pareto relaxed_C、旧 ``-0.3``) で分類可能
                  (Layer 1 clean / Layer 2/3 SNR≥5 corpus 実測)

            WER 軽微退行 caveat (LLM typical 0.5-1%):
            - Voxtral PR-A.4.1 (greedy 切替) / Canary PR-A.4.2 (beam→greedy)
              precedent と同 framing で、filter benefit を優先。
            - ``--confidence-filter off`` は **post-ASR reject のみ**
              無効化し、本 method の ``repetition_penalty=1.1 +
              no_repeat_ngram_size=3`` の generation 側変更は固定で残る
              (Voxtral greedy / Canary greedy と同 design)。

            自動検出のとき: 以前の ``qwen-asr`` の ``transcribe()`` と同じ設定 (system prompt は空、
            penalty なし、``max_new_tokens=512``) で生成し、``language X<asr_text>本文`` を processor の
            ``transcription_only`` で解析する (繰り返しの修正も同じ)。
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

        if self._asr_language is None:
            return self._transcribe_auto_language(audio_data)
        return self._transcribe_with_scores(audio_data)

    def _generate(self, audio_data: np.ndarray, *, language: Optional[str], prompt: Optional[str], **kwargs) -> Tuple[Any, Any]:
        """``apply_transcription_request`` → ``generate``。(``outputs``, 生成した token ID) を返す。"""
        import torch

        model = self.model.model
        processor = self.model.processor
        # language を渡すと prompt の末尾に ``language X<asr_text>`` が付き、本文だけが生成される
        inputs = processor.apply_transcription_request(audio=audio_data, language=language, prompt=prompt)
        inputs = inputs.to(model.device).to(model.dtype)
        with torch.no_grad():
            outputs = model.generate(**inputs, do_sample=False, return_dict_in_generate=True, **kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        return outputs, outputs.sequences[:, prompt_len:]

    def _transcribe_auto_language(self, audio_data: np.ndarray) -> TranscriptionResult:
        """自動検出 (``language=None``)。confidence は出さない (fail-open)。"""
        try:
            _outputs, gen_token_ids = self._generate(
                audio_data, language=None, prompt=None, **_AUTO_LANGUAGE_GENERATE_KWARGS
            )
            text = self.model.processor.decode(gen_token_ids[0], return_format="transcription_only")
            return TranscriptionResult(text=text, confidence=1.0)
        except Exception as e:
            logger.error(f"Qwen3-ASR transcription failed (auto language): {e}")
            raise

    def _transcribe_with_scores(self, audio_data: np.ndarray) -> TranscriptionResult:
        """言語を指定して転写し、``generate(output_scores=True)`` から confidence を取る (PR-A.5.2)。"""
        try:
            model = self.model.model
            processor = self.model.processor
            outputs, gen_token_ids = self._generate(
                audio_data,
                language=self._asr_language,
                prompt=_FORCED_LANGUAGE_SYSTEM_PROMPT,
                output_scores=True,
                **_FORCED_LANGUAGE_GENERATE_KWARGS,
            )

            decoded = processor.batch_decode(
                gen_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            text = decoded[0] if decoded else ""

            # avg_logprob 抽出 (Voxtral pattern 完全同形、framework contract を trust)
            transition_scores = model.compute_transition_scores(
                outputs.sequences,
                outputs.scores,
                normalize_logits=True,
            )
            engine_confidence = _extract_engine_confidence(
                transition_scores=transition_scores[0],
                gen_tokens=gen_token_ids[0],
                special_ids=_special_token_ids(processor, model),
            )

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
            logger.error(f"Qwen3-ASR transcription failed: {e}")
            raise

    def get_engine_name(self) -> str:
        """エンジン名を取得"""
        if "1.7B" in self.model_name:
            return "Qwen3-ASR 1.7B"
        return "Qwen3-ASR 0.6B"

    def get_supported_languages(self) -> list:
        """サポートされる言語のリストを取得 (正本は qwen3asr_languages.py、#230)"""
        return list(self.SUPPORTED_LANGUAGES)

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
