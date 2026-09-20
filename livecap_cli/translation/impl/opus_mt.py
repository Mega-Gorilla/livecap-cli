"""
OPUS-MT 翻訳エンジン実装

Helsinki-NLP の OPUS-MT モデルを CTranslate2 で高速推論する翻訳エンジン。
ローカルで動作し、CPU でも十分高速。
"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

# Import dependencies at module level to enable conditional import in __init__.py
# This allows `impl/__init__.py` to catch ImportError when deps are missing
import ctranslate2
import transformers

from livecap_cli.engines.hf_cache import RepoContentError, fetch_repo_dir
from livecap_cli.engines.model_store import (
    MANIFEST_NAME,
    adopt_dir,
    build_manifest_from_dir,
    invalidate_manifest,
    model_lock,
    publish_dir,
    validate_repo_dir,
)

from ..base import BaseTranslator
from ..exceptions import TranslationModelError, UnsupportedLanguagePairError
from ..lang_codes import get_opus_mt_model_name, to_iso639_1
from ..result import TranslationResult

if TYPE_CHECKING:
    pass  # Type-only imports go here if needed

logger = logging.getLogger(__name__)


class OpusMTTranslator(BaseTranslator):
    """
    OPUS-MT via CTranslate2

    Helsinki-NLP の OPUS-MT モデルを使用したローカル翻訳エンジン。
    CTranslate2 による INT8 量子化で高速・省メモリ推論。

    Note:
        デフォルトで文脈機能は無効（default_context_sentences=0）。
        OPUS-MT は改行を保持せず、文境界の検出が不安定なため、
        文脈連結による翻訳品質向上より誤抽出リスクが高い。
        文脈を活用したい場合は RivaInstructTranslator を推奨。

        See: https://github.com/Mega-Gorilla/livecap-cli/issues/190

    Examples:
        >>> translator = OpusMTTranslator(source_lang="ja", target_lang="en")
        >>> translator.load_model()
        >>> result = translator.translate("こんにちは", "ja", "en")
        >>> print(result.text)
        "Hello"
    """

    def __init__(
        self,
        source_lang: str = "ja",
        target_lang: str = "en",
        model_name: Optional[str] = None,
        device: str = "cpu",
        compute_type: str = "int8",
        default_context_sentences: int = 0,
        **kwargs,
    ):
        """
        OpusMTTranslator を初期化

        Args:
            source_lang: ソース言語コード（デフォルト: "ja"）
            target_lang: ターゲット言語コード（デフォルト: "en"）
            model_name: HuggingFace モデル名（省略時は言語ペアから自動生成）
            device: 推論デバイス（"cpu" or "cuda"、デフォルト: "cpu"）
            compute_type: 量子化タイプ（"int8", "float16" 等、デフォルト: "int8"）
            default_context_sentences: 文脈として使用する文数（デフォルト: 0）。
                OPUS-MT は文脈抽出が不安定なため、デフォルトは無効。
            **kwargs: BaseTranslator に渡すパラメータ
        """
        super().__init__(default_context_sentences=default_context_sentences, **kwargs)
        self.source_lang = source_lang
        self.target_lang = target_lang

        # model_name が指定されていない場合は言語ペアから生成
        if model_name is None:
            model_name = get_opus_mt_model_name(source_lang, target_lang)
        self.model_name = model_name

        self.device = device
        self.compute_type = compute_type
        self._model: Optional[ctranslate2.Translator] = None
        self._tokenizer: Optional[transformers.PreTrainedTokenizer] = None

    #: 正本 dir (``<models_root>/opus-mt/<org>--<name>/``) に必ず要るファイル:
    #: CTranslate2 model (``model.bin`` + CT2 の ``config.json``) と、変換元と同じ tokenizer。
    #: **tokenizer を同梱する**のは、以前 ``load_model()`` のたびに ``AutoTokenizer.from_pretrained(<repo id>)``
    #: が既定 HF cache (root の外、582 MB の変換元 snapshot ごと) へ行っていたため (#455)。
    REQUIRED_FILES = ("model.bin", "config.json", "tokenizer_config.json", "vocab.json", "source.spm", "target.spm")
    #: 変換に要る変換元ファイル。重みは ``model.safetensors`` を優先し、無い repo (古い revision) は
    #: ``pytorch_model.bin`` へ fallback する (両方取ると 300 MB × 2 になる)
    SOURCE_COMMON_FILES = ("config.json", "generation_config.json", "tokenizer_config.json", "vocab.json", "source.spm", "target.spm")
    SOURCE_WEIGHT_CANDIDATES = ("model.safetensors", "pytorch_model.bin")
    #: tokenizer だけを足すとき (旧配置の adopt) に取るファイル。HF の ``config.json`` (``model_type``) も要る —
    #: repo の ``tokenizer_config.json`` には ``tokenizer_class`` が無く、``AutoTokenizer`` は config から
    #: model_type を引く (実測)。``save_pretrained`` 後の tokenizer_config には tokenizer_class が入るので、
    #: 正本 dir から読むときは CT2 の ``config.json`` と衝突しない
    TOKENIZER_SOURCE_FILES = ("config.json", "tokenizer_config.json", "vocab.json", "source.spm", "target.spm")
    #: 正本 dir に無ければ「tokenizer 未同梱の旧配置」と見なすファイル
    TOKENIZER_FILES = ("tokenizer_config.json", "vocab.json", "source.spm", "target.spm")

    @property
    def model_dir(self) -> Path:
        """正本 dir ``<models_root>/opus-mt/<org>--<name>/`` (dir 名は #456 以前から変えない)。"""
        return self.model_manager.get_models_dir() / "opus-mt" / self.model_name.replace("/", "--")

    def _validate(self, directory: Path) -> bool:
        return validate_repo_dir(directory, repo_id=self.model_name, required=self.REQUIRED_FILES) is not None

    def load_model(self) -> None:
        """
        モデルをロード

        正本 (``<models_root>/opus-mt/<org>--<name>/``: CTranslate2 model + tokenizer + manifest) が
        無ければ用意してから (:meth:`_ensure_model_dir`)、**ローカル dir** を ``ctranslate2.Translator``
        と ``AutoTokenizer.from_pretrained`` に渡す。repo id は渡さない (既定 HF cache へ行く、#455)。

        Raises:
            TranslationModelError: モデルの用意またはロードに失敗した場合
        """
        model_dir = self._ensure_model_dir()

        try:
            self._model = ctranslate2.Translator(
                str(model_dir),
                device=self.device,
                compute_type=self.compute_type,
            )
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_dir))
            self._initialized = True
            logger.info(
                "Loaded OPUS-MT model: %s from %s (device=%s, compute_type=%s)",
                self.model_name,
                model_dir,
                self.device,
                self.compute_type,
            )
        except Exception as e:
            # **self-heal**: manifest に無い形で dir が壊れている場合、manifest を残すと以後
            # 永久に「用意済み」と判定して落ち続ける。無効化して次回作り直す
            invalidate_manifest(model_dir, reason=f"OPUS-MT load failed: {e}")
            raise TranslationModelError(f"Failed to load model: {e}") from e

    def _ensure_model_dir(self) -> Path:
        """正本 dir を返す。無ければ **変換元 → 変換 → tokenizer 同梱 → manifest → 原子的 publish**。

        順に試す (destination 単位の lock 内、ASR engine と同じ規則):

        1. manifest 込みで valid → そのまま
        2. #456 以前の変換済み dir (manifest 無し、CT2 model はあるが tokenizer が無い) → 変換元 repo から
           tokenizer だけを取って同梱し、その場で採用 (adopt)。300 MB の再変換はしない
        3. 変換元を ``fetch_repo_dir`` で staging (``<cache_root>/downloads/``) へ取り、
           ``TransformersConverter`` で payload へ変換、tokenizer を ``save_pretrained`` で同梱、
           manifest を書いて ``publish_dir``。成功後に staging を消す

        Raises:
            TranslationModelError: 取得 / 変換に失敗した場合
        """
        manager = self.model_manager
        destination = self.model_dir
        staging_root = manager.get_temp_dir("downloads")
        try:
            with model_lock(staging_root, destination):
                if self._validate(destination):
                    return destination
                if self._adopt_converted_dir(destination, staging_root):
                    return destination
                self._convert_model(destination, staging_root)
                return destination
        except TranslationModelError:
            raise
        except Exception as e:
            raise TranslationModelError(f"Failed to prepare model: {e}") from e

    def _fetch_source(self, staging_root: Path, *, files: Tuple[str, ...], with_weights: bool) -> Path:
        """変換元 repo の必要ファイルを staging 内の dir (manifest 付き、transient) へ取る。"""
        # dir 名を正本 (`<org>--<name>`) と変える: fetch_repo_dir の lock は destination 名で切られるので、
        # 同名だと _ensure_model_dir が持つ lock と同じファイルを同一 process で二重に取ることになる
        source_dir = staging_root / "opus-mt-source" / f"{self.model_name.replace('/', '--')}.source"
        hub_root = self.model_manager.get_huggingface_cache_dir()
        if not with_weights:
            return fetch_repo_dir(
                self.model_name,
                hub_root=hub_root,
                staging_root=staging_root,
                destination=source_dir,
                allow_patterns=files,
                required=files,
            )
        last_error: Optional[Exception] = None
        for weight in self.SOURCE_WEIGHT_CANDIDATES:
            try:
                return fetch_repo_dir(
                    self.model_name,
                    hub_root=hub_root,
                    staging_root=staging_root,
                    destination=source_dir,
                    allow_patterns=files + (weight,),
                    required=files + (weight,),
                )
            except RepoContentError as e:
                # 必要ファイルが無い = この revision にはその形式の重みが無い → 次の候補
                # (ネットワーク / offline のエラーはここで握らず、そのまま fail loud)
                last_error = e
                logger.info("OPUS-MT source has no %s (trying next): %s", weight, e)
        raise TranslationModelError(f"No convertible weights in {self.model_name}: {last_error}") from last_error

    def _adopt_converted_dir(self, destination: Path, staging_root: Path) -> bool:
        """#456 以前に変換した dir (tokenizer 無し) に tokenizer を足してその場で採用する。"""
        if not destination.is_dir() or (destination / MANIFEST_NAME).exists():
            return False
        if not all((destination / name).is_file() for name in ("model.bin", "config.json")):
            return False
        missing = [name for name in self.TOKENIZER_FILES if not (destination / name).is_file()]
        if missing:
            logger.info("OPUS-MT: adding tokenizer files %s to pre-existing converted dir %s", missing, destination)
            source_dir = self._fetch_source(staging_root, files=self.TOKENIZER_SOURCE_FILES, with_weights=False)
            try:
                tokenizer = transformers.AutoTokenizer.from_pretrained(str(source_dir))
                tokenizer.save_pretrained(str(destination))
            finally:
                shutil.rmtree(source_dir, ignore_errors=True)
        manifest = adopt_dir(destination, repo_id=self.model_name, required=self.REQUIRED_FILES)
        if manifest is None:
            return False
        logger.info("OPUS-MT: adopted pre-existing converted model dir: %s", destination)
        return True

    def _convert_model(self, destination: Path, staging_root: Path) -> None:
        """
        HuggingFace モデルを CTranslate2 形式に変換し、tokenizer と manifest を付けて正本へ publish する。

        変換元は staging (``<cache_root>/downloads/opus-mt-source/…``) に取り、変換後に消す。
        以前は ``TransformersConverter(<repo id>)`` が変換元を既定 HF cache (root の外) へ落としていた (#455)。

        Args:
            destination: 正本 dir ``<models_root>/opus-mt/<org>--<name>/``
            staging_root: ``<cache_root>/downloads``

        Raises:
            TranslationModelError: 取得 / 変換に失敗した場合
        """
        from ctranslate2.converters import TransformersConverter

        logger.info("Converting %s to CTranslate2 format...", self.model_name)
        source_dir = self._fetch_source(staging_root, files=self.SOURCE_COMMON_FILES, with_weights=True)
        payload = staging_root / "opus-mt-source" / f".{destination.name}.convert-{uuid.uuid4().hex[:8]}"
        try:
            converter = TransformersConverter(str(source_dir))
            converter.convert(str(payload), quantization=self.compute_type)
            tokenizer = transformers.AutoTokenizer.from_pretrained(str(source_dir))
            tokenizer.save_pretrained(str(payload))
            source_manifest = validate_repo_dir(source_dir, repo_id=self.model_name)
            manifest = build_manifest_from_dir(
                payload,
                repo_id=self.model_name,
                revision=f"ct2:{self.compute_type}",
                commit_sha=source_manifest.commit_sha if source_manifest else None,
                source="download",
            )
            (payload / MANIFEST_NAME).write_text(manifest.to_json(), encoding="utf-8")
            publish_dir(payload, destination, validate=self._validate)
            logger.info("Model conversion completed: %s", destination)
        except TranslationModelError:
            raise
        except Exception as e:
            raise TranslationModelError(f"Failed to convert model: {e}") from e
        finally:
            shutil.rmtree(payload, ignore_errors=True)
            shutil.rmtree(source_dir, ignore_errors=True)

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Optional[List[str]] = None,
    ) -> TranslationResult:
        """
        テキストを翻訳

        Args:
            text: 翻訳対象テキスト
            source_lang: ソース言語コード (BCP-47)
            target_lang: ターゲット言語コード (BCP-47)
            context: 過去の文脈（直近N文）。デフォルトでは無効（default_context_sentences=0）。
                OPUS-MT は改行を保持せず文境界検出が不安定なため、文脈使用時は
                誤抽出のリスクがある。文脈を活用したい場合は RivaInstructTranslator を推奨。

        Returns:
            TranslationResult

        Raises:
            TranslationModelError: モデル未ロード、または推論エラー
            UnsupportedLanguagePairError: 同一言語が指定された場合
        """
        # モデルロードチェック
        if not self._initialized or self._model is None or self._tokenizer is None:
            raise TranslationModelError("Model not loaded. Call load_model() first.")

        # 入力バリデーション: 空文字列
        if not text or not text.strip():
            return TranslationResult(
                text="",
                original_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
            )

        # 入力バリデーション: 同一言語
        if to_iso639_1(source_lang) == to_iso639_1(target_lang):
            raise UnsupportedLanguagePairError(
                source_lang, target_lang, self.get_translator_name()
            )

        # 文脈連結（改行区切りで段落として認識させる）
        # default_context_sentences=0 の場合は context を無視（文脈無効化）
        num_context_sentences = 0
        if context and self._default_context_sentences > 0:
            ctx = context[-self._default_context_sentences :]
            num_context_sentences = len(ctx)
            full_text = "\n".join(ctx) + "\n" + text
        else:
            full_text = text

        try:
            # トークナイズ
            source_tokens = self._tokenizer.convert_ids_to_tokens(
                self._tokenizer.encode(full_text)
            )

            # 翻訳
            results = self._model.translate_batch([source_tokens])
            target_tokens = results[0].hypotheses[0]

            # デコード
            result = self._tokenizer.decode(
                self._tokenizer.convert_tokens_to_ids(target_tokens),
                skip_special_tokens=True,
            )
        except Exception as e:
            raise TranslationModelError(f"Translation failed: {e}") from e

        # 文脈を含めた場合、最後の文を抽出
        if context and self._default_context_sentences > 0:
            result = self._extract_relevant_part(result, num_context_sentences)

        return TranslationResult(
            text=result,
            original_text=text,
            source_lang=source_lang,
            target_lang=target_lang,
        )

    def _extract_relevant_part(self, translated: str, num_context_sentences: int) -> str:
        """
        翻訳結果から対象部分（最後の文）を抽出

        OPUS-MT は改行を保持しないため、文末記号で分割して最後の文を抽出する。

        Warning:
            この抽出ロジックは不安定で、誤抽出のリスクがある。
            デフォルトでは default_context_sentences=0 のため、このメソッドは呼ばれない。
            文脈を有効にした場合のみ使用される。

        Args:
            translated: 翻訳結果テキスト
            num_context_sentences: 文脈として連結した文の数

        Returns:
            最後の文（対象テキストの翻訳結果）
        """
        import re

        # まず改行で分割を試みる（改行が保持されている場合）
        lines = translated.strip().split("\n")
        if len(lines) > 1:
            return lines[-1]

        # 改行がない場合、文末記号で分割
        # 英語の文末: . ! ? と、それに続く空白または文末
        sentences = re.split(r"(?<=[.!?])\s+", translated.strip())

        if len(sentences) <= 1:
            return translated

        # 文脈文の数を超える文がある場合、最後の文を返す
        if len(sentences) > num_context_sentences:
            return sentences[-1]

        return translated

    def get_translator_name(self) -> str:
        """翻訳エンジン名を取得"""
        return "opus_mt"

    def get_supported_pairs(self) -> List[Tuple[str, str]]:
        """
        サポートする言語ペアを取得

        Returns:
            初期化時に指定された言語ペア
        """
        return [(self.source_lang, self.target_lang)]

    def cleanup(self) -> None:
        """
        リソースのクリーンアップ

        モデルとトークナイザーを解放。
        """
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._initialized = False
        logger.debug("OpusMTTranslator cleanup completed")
