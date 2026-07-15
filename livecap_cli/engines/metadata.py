"""
エンジンメタデータの定義

移動元：gui/dialogs/settings/constants.py: ENGINE_METADATA
このモジュールは、ASRエンジンのメタデータを一元管理します。
"""

from typing import Dict, Any, List, NamedTuple, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import langcodes

from .qwen3asr_languages import QWEN_ASR_LANGUAGE_NAMES
from .whisper_languages import WHISPER_LANGUAGES


class LanguageQuality(NamedTuple):
    """ある言語における engine の品質階層と、その根拠レベル。

    tier:     "best" | "good" | "fallback"
    evidence: "measured"   … 本リポジトリのベンチ実測由来
              "model_card" … モデルカード / 公称 WER 由来
              "heuristic"  … 推定 (根拠が弱い、要レビュー)
    """
    tier: str
    evidence: str


class ReasonCode(str, Enum):
    """recommend() が返す推奨/非推奨の根拠コード (i18n 対応)。

    自由文でなく closed enum にすることで GUI 側が
    ``translate("engines.reason.{value}")`` パターンで localize できる。
    """
    LANGUAGE_SPECIALIZED = "language_specialized"  # 当該言語に特化
    MULTILINGUAL = "multilingual"                  # 多言語対応
    ONLY_CANDIDATE = "only_candidate"              # 他に選択肢がない
    FITS_VRAM = "fits_vram"                         # VRAM に収まる
    EXCEEDS_VRAM = "exceeds_vram"                   # VRAM 超過 (OOM 懸念)
    STREAMING_SUPPORTED = "streaming_supported"     # streaming 対応
    OFFLINE_ONLY = "offline_only"                   # 非 streaming (オフライン専用)
    LOW_COMPUTE = "low_compute"                      # 低負荷
    GPU_RECOMMENDED = "gpu_recommended"              # GPU 推奨
    REALTIME_ON_CPU = "realtime_on_cpu"              # CPU でリアルタイム可
    FALLBACK = "fallback"                            # 汎用 fallback


class EngineRecommendation(NamedTuple):
    """recommend() の戻り値要素。

    ``params`` は ``EngineFactory.create_engine(engine_id, device=..., **params)``
    にそのまま展開できる形。whispers2t のみ ``{"model_size": ...}`` を持ち、
    他 engine は空 dict (model が default_params で一意)。
    """
    engine_id: str
    params: Dict[str, Any]
    rank: int
    quality: str
    reason_codes: List[ReasonCode]
    scores: Dict[str, float]


@dataclass
class EngineInfo:
    """エンジン情報"""
    id: str
    display_name: str
    description: str
    supported_languages: Tuple[str, ...]
    # 構築時は list/tuple どちらも受理し __post_init__ で tuple 化する (#230)。
    # resolve_language() の受理判定の正本のため、外部からの mutation を封鎖。
    requires_download: bool = False
    model_size: Optional[str] = None
    device_support: List[str] = field(default_factory=lambda: ["cpu"])
    streaming: bool = False
    default_params: Dict[str, Any] = field(default_factory=dict)
    module: Optional[str] = None  # エンジンモジュールのパス
    class_name: Optional[str] = None  # エンジンクラス名
    available_model_sizes: Optional[List[str]] = None  # 選択可能なモデルサイズ一覧

    # === recommend() 用 metadata (Issue #286) ===
    quality_tier: Dict[str, LanguageQuality] = field(default_factory=dict)
    # 言語(ISO639-1) → 品質階層 + 根拠。未登録言語は recommend 内で "fallback" 扱い

    vram_required_mb: Optional[int] = None
    # GPU 推論の代表 VRAM 要件 (MB)。VRAM 要件の正本はここ。
    # None = 軽量 or 未計測 → filter で除外しない。実測値は issue-73 由来。

    # capability flags (device_support は全 engine 同値で選別力が無いため補完)
    cpu_supported: bool = True       # CPU で動作可能か
    cpu_recommended: bool = False    # CPU で実用的な速度か
    gpu_recommended: bool = True     # GPU 推奨か
    realtime_on_cpu: bool = False    # CPU でリアルタイム(RTF<=1)達成可能か

    # === 言語解決 metadata (Issue #365 / #230 で改名) ===
    cli_default_language: str = ""
    # CLI ``--language`` 未指定時に resolve_language() が返す実効値。
    # 全登録 engine で明示設定必須 ("" は未設定を意味する保険 default)。
    # **CLI policy であり、engine constructor の default との一致は要求しない**
    # — qwen3asr は意図的に constructor=None (auto) / CLI=ja (PR-A.5.2:
    # confidence filter の avg_logprob 経路を CLI 既定で有効にするため)。

    supports_language_auto: bool = False
    # engine が native 自動言語検出に対応し "auto" 指定を受理できるか。

    def __post_init__(self) -> None:
        # mutable 流出封鎖 (#230): get() が内部 instance を返しても
        # 外部から supported_languages を書き換えられないよう tuple 化する。
        self.supported_languages = tuple(self.supported_languages)


class EngineMetadata:
    """
    エンジンメタデータの中央管理

    移動元：gui/dialogs/settings/constants.py: ENGINE_METADATA
    参照元：12箇所（engine_factory, lazy_loader, basic_tab等）
    """

    _ENGINES: Dict[str, EngineInfo] = {
        "reazonspeech": EngineInfo(
            id="reazonspeech",
            display_name="ReazonSpeech K2 v2",
            description="Japanese-specialized high-accuracy ASR engine optimized for real-time transcription",
            supported_languages=["ja"],
            requires_download=True,
            model_size="159.34MB",
            device_support=["cpu", "cuda"],
            streaming=True,
            module=".reazonspeech_engine",
            class_name="ReazonSpeechEngine",
            default_params={
                # カテゴリA パラメータ（エンジンの__init__で使用）
                "use_int8": False,
                "num_threads": 4,
                "decoding_method": "greedy_search",
            },
            # Issue #286: 軽量(159MB)・日本語特化・CPU 実用/realtime 可
            quality_tier={"ja": LanguageQuality("best", "model_card")},
            vram_required_mb=None,  # 軽量・未計測 → filter で除外しない
            cpu_recommended=True,
            realtime_on_cpu=True,
            gpu_recommended=False,
            cli_default_language="ja",
        ),
        "parakeet": EngineInfo(
            id="parakeet",
            display_name="NVIDIA Parakeet TDT 0.6B v2",
            description="English-optimized high-accuracy ASR with WER 6.05%",
            supported_languages=["en"],
            requires_download=True,
            model_size="1.2GB",
            device_support=["cpu", "cuda"],
            streaming=True,
            module=".parakeet_engine",
            class_name="ParakeetEngine",
            default_params={
                "model_name": "nvidia/parakeet-tdt-0.6b-v2",
                "decoding_strategy": "greedy",
            },
            # Issue #286: 英語特化 (WER 6.05%)、VRAM 2417MB は issue-73 実測
            quality_tier={"en": LanguageQuality("best", "model_card")},
            vram_required_mb=2417,  # measured (docs/planning/issue-73)
            cli_default_language="en",
        ),
        "parakeet_ja": EngineInfo(
            id="parakeet_ja",
            display_name="NVIDIA Parakeet TDT CTC 0.6B JA",
            description="Japanese-specialized high-accuracy streaming ASR model",
            supported_languages=["ja"],
            requires_download=True,
            model_size="600MB",
            device_support=["cpu", "cuda"],
            streaming=True,
            module=".parakeet_engine",
            class_name="ParakeetEngine",
            default_params={
                "model_name": "nvidia/parakeet-tdt_ctc-0.6b-ja",
                # parakeet_ja は EncDecHybridRNNTCTCBPEModel (TDT-CTC hybrid)。
                # adapter は default で CTC decoder + greedy_batch に切替
                # (PR #309: token_confidence_mean を populate するため必須、
                # かつ RNNT greedy より 1.83x 高速)。docs/research/
                # parakeet-ja-confidence-spec-2026-06-10.md を参照。
                "decoding_strategy": "greedy_batch",
            },
            # Issue #286: 日本語特化 streaming。VRAM は parakeet 同アーキ由来の推定
            quality_tier={"ja": LanguageQuality("best", "model_card")},
            vram_required_mb=2500,  # heuristic (~parakeet 0.6B arch, 未実測)
            cli_default_language="ja",
        ),
        "canary": EngineInfo(
            id="canary",
            display_name="NVIDIA Canary 1B Flash",
            description="Fast multilingual ASR supporting EN, DE, FR, ES",
            supported_languages=["en", "de", "fr", "es"],
            requires_download=True,
            model_size="1.5GB",
            device_support=["cpu", "cuda"],
            streaming=True,
            module=".canary_engine",
            class_name="CanaryEngine",
            default_params={
                "model_name": "nvidia/canary-1b-flash",
            },
            # Issue #286: multilingual specialist。de/fr/es は特化、en は good。VRAM 実測
            quality_tier={
                "en": LanguageQuality("good", "model_card"),
                "de": LanguageQuality("best", "model_card"),
                "fr": LanguageQuality("best", "model_card"),
                "es": LanguageQuality("best", "model_card"),
            },
            vram_required_mb=6830,  # measured (docs/planning/issue-73)
            cli_default_language="en",
        ),
        "voxtral": EngineInfo(
            id="voxtral",
            display_name="MistralAI Voxtral Mini 3B",
            description="Advanced multilingual ASR with auto language detection",
            supported_languages=["en", "es", "fr", "pt", "hi", "de", "nl", "it"],
            requires_download=True,
            model_size="3GB",
            device_support=["cpu", "cuda"],
            streaming=True,
            module=".voxtral_engine",
            class_name="VoxtralEngine",
            default_params={
                "temperature": 0.0,
                "do_sample": False,
                "max_new_tokens": 448,
                "model_name": "mistralai/Voxtral-Mini-3B-2507",
            },
            # Issue #286: advanced multilingual (8 lang, 全て good)。VRAM 実測 (load 8923MB,
            # 推論 peak はさらに上、12GB+ 推奨)
            quality_tier={
                lang: LanguageQuality("good", "model_card")
                for lang in ["en", "es", "fr", "pt", "hi", "de", "nl", "it"]
            },
            vram_required_mb=8923,  # measured load (docs/planning/issue-73), 推論 peak↑
            cli_default_language="auto",       # 未指定時は native 自動検出 (現行挙動維持)
            supports_language_auto=True,
        ),
        # WhisperS2T - Unified multilingual ASR engine
        "whispers2t": EngineInfo(
            id="whispers2t",
            display_name="WhisperS2T",
            description="Multilingual ASR model with selectable model sizes (tiny to large-v3-turbo)",
            supported_languages=list(WHISPER_LANGUAGES),  # 99 languages
            requires_download=True,
            model_size=None,  # Multiple sizes available
            device_support=["cpu", "cuda"],
            streaming=True,
            module=".whispers2t_engine",
            class_name="WhisperS2TEngine",
            available_model_sizes=[
                # Standard models
                "tiny", "base", "small", "medium",
                # Large models
                "large-v1", "large-v2", "large-v3",
                # High-speed models
                "large-v3-turbo", "distil-large-v3",
            ],
            default_params={
                "model_size": "large-v3",  # Benchmark compatibility (was whispers2t_large_v3)
                "compute_type": "auto",
                "batch_size": 24,
                "use_vad": True,
            },
            # Issue #286: 汎用 fallback (99 lang、特化 engine に劣るが全言語で動く)。
            # VRAM は size 依存 (CTranslate2 は torch 計測外) のため None、
            # 推奨 size は recommend() が params["model_size"] で hardware に応じ返す。
            quality_tier={
                lang: LanguageQuality("fallback", "heuristic")
                for lang in WHISPER_LANGUAGES
            },
            vram_required_mb=None,  # size 依存 / CTranslate2 計測外
            cpu_recommended=True,   # tiny/base は CPU 実用
            realtime_on_cpu=True,   # base で 3-5x realtime (CPU_SPEED_ESTIMATES)
            cli_default_language="ja",  # 旧 CLI parser default を維持 (#365)
        ),
        # Qwen3-ASR - High-accuracy multilingual ASR
        "qwen3asr": EngineInfo(
            id="qwen3asr",
            display_name="Qwen3-ASR 0.6B",
            description="High-accuracy multilingual ASR supporting 30+ languages",
            # 正本は qwen3asr_languages.py (#230) — adapter の言語 map と同源
            supported_languages=list(QWEN_ASR_LANGUAGE_NAMES),
            requires_download=True,
            model_size="1.2GB",
            device_support=["cpu", "cuda"],
            streaming=False,  # MVP: オフラインのみ
            module=".qwen3asr_engine",
            class_name="Qwen3ASREngine",
            default_params={
                "model_name": "Qwen/Qwen3-ASR-0.6B",
                "engine_id": "qwen3asr",
            },
            # Issue #286: 高精度 multilingual だが non-streaming(offline MVP)。
            # v1 draft の「30言語 best」は過剰主張のため good に是正。専用 engine 不在
            # 言語(zh/ko 等)では自動的に最上位に来る。
            quality_tier={
                lang: LanguageQuality("good", "model_card")
                for lang in QWEN_ASR_LANGUAGE_NAMES
            },
            vram_required_mb=None,  # 未計測 (~0.6B)
            # PR-A.5.2: CLI 未指定時に ja を渡し avg_logprob filter 経路を維持
            # (auto-detect は confidence fail-open のため既定にしない)
            cli_default_language="ja",
            supports_language_auto=True,  # literal "auto" は engine 内で None に解決
        ),
    }

    @classmethod
    def get(cls, engine_id: str) -> Optional[EngineInfo]:
        """
        エンジン情報を取得

        Args:
            engine_id: エンジンID

        Returns:
            EngineInfo オブジェクト、またはNone
        """
        return cls._ENGINES.get(engine_id)

    @classmethod
    def get_all(cls) -> Dict[str, EngineInfo]:
        """
        全エンジン情報を取得

        Returns:
            エンジンID -> EngineInfo の辞書
        """
        return cls._ENGINES.copy()

    @classmethod
    def get_display_name(cls, engine_id: str) -> str:
        """
        表示名を取得

        Args:
            engine_id: エンジンID

        Returns:
            表示名（見つからない場合はエンジンIDを返す）
        """
        info = cls.get(engine_id)
        return info.display_name if info else engine_id

    @classmethod
    def get_engines_for_language(cls, lang_code: str) -> List[str]:
        """
        指定言語をサポートするエンジンリストを取得

        Args:
            lang_code: 言語コード（"ja", "zh-CN", "en" など）

        Returns:
            エンジンIDのリスト

        Note:
            BCP-47 形式（zh-CN, zh-TW, pt-BR など）は ISO 639-1（zh, pt）に
            変換してから比較する。これにより WhisperS2T の100言語サポートが
            正しく機能する。
        """
        # BCP-47 → ISO 639-1 変換（自己完結）
        iso_code = cls.to_iso639_1(lang_code)

        result = []
        for engine_id, info in cls._ENGINES.items():
            if iso_code in info.supported_languages:
                result.append(engine_id)
        return result

    @classmethod
    def recommend(
        cls,
        language: str,
        gpu_available: bool = False,
        vram_gb: Optional[float] = None,
    ) -> List["EngineRecommendation"]:
        """言語 × ハードウェアに基づく推奨 engine を rank 昇順で返す (Issue #286)。

        Args:
            language: 認識する言語コード (BCP-47 / ISO 639-1)。内部で ISO 639-1 に正規化。
            gpu_available: GPU (CUDA) が利用可能か。``torch.cuda.is_available()`` の結果を渡す。
            vram_gb: GPU の VRAM (GB)。``gpu_available=True`` 時のみ意味を持つ。
                ``None`` なら VRAM 容量による絞り込みは行わない (fit 仮定)。

        Returns:
            ``rank`` 昇順の ``EngineRecommendation`` リスト。

            - hard 除外は「言語非対応」のみ。VRAM 超過等は除外せず ``EXCEEDS_VRAM`` code +
              低 score で沈める (呼出側 wizard が全選択肢を提示 / gray-out できるように)。
            - sort は分解 score の辞書式多段 (quality → hardware_fit → latency → streaming)。
              同 score の tie は登録順で安定 (「重いモデル優先」は採らない)。
            - whispers2t は hardware に応じた ``params["model_size"]`` を返す。他 engine は空 dict。
              ``params`` は ``EngineFactory.create_engine(engine_id, device=..., **params)`` にそのまま渡せる。

        Example:
            >>> recs = EngineMetadata.recommend("ja", gpu_available=True, vram_gb=8.0)
            >>> [(r.rank, r.engine_id, r.quality) for r in recs]  # doctest: +SKIP
            [(1, 'parakeet_ja', 'best'), (2, 'reazonspeech', 'best'),
             (3, 'qwen3asr', 'good'), (4, 'whispers2t', 'fallback')]
            >>> # 上位2 (rank<=2) は日本語 best の parakeet_ja / reazonspeech。
            >>> # 同 quality の順序はハードウェア適合に依存 (GPU では VRAM 既知の
            >>> # parakeet_ja が、CPU では realtime_on_cpu の reazonspeech が上位)。
        """
        iso_code = cls.to_iso639_1(language)

        candidates = []  # (info, quality)
        for info in cls._ENGINES.values():
            if iso_code not in info.supported_languages:
                continue
            lq = info.quality_tier.get(iso_code, LanguageQuality("fallback", "heuristic"))
            candidates.append((info, lq.tier))

        n_candidates = len(candidates)

        scored = []  # (sort_key, insertion_index, info, quality, scores)
        for idx, (info, quality) in enumerate(candidates):
            scores = _compute_scores(info, quality, gpu_available, vram_gb)
            sort_key = (
                -scores["quality_score"],
                -scores["hardware_fit_score"],
                -scores["latency_score"],
                -scores["streaming_score"],
                idx,  # 登録順で安定 tiebreak
            )
            scored.append((sort_key, idx, info, quality, scores))

        scored.sort(key=lambda t: t[0])

        recommendations = []
        for rank, (_, _, info, quality, scores) in enumerate(scored, start=1):
            params: Dict[str, Any] = {}
            if info.id == "whispers2t":
                params = {"model_size": _recommend_whisper_size(gpu_available, vram_gb)}
            recommendations.append(
                EngineRecommendation(
                    engine_id=info.id,
                    params=params,
                    rank=rank,
                    quality=quality,
                    reason_codes=_build_reason_codes(
                        info, quality, gpu_available, vram_gb, n_candidates
                    ),
                    scores=scores,
                )
            )
        return recommendations

    @classmethod
    def get_module_info(cls, engine_id: str) -> tuple[Optional[str], Optional[str]]:
        """
        エンジンのモジュール情報を取得

        Args:
            engine_id: エンジンID

        Returns:
            (module_path, class_name) のタプル
        """
        info = cls.get(engine_id)
        if info:
            return info.module, info.class_name
        return None, None

    @classmethod
    def to_iso639_1(cls, code: str) -> str:
        """
        BCP-47 言語コードを ISO 639-1 に変換

        Args:
            code: 言語コード（"ja", "zh-CN", "ZH-TW" など）

        Returns:
            ISO 639-1 言語コード（"ja", "zh" など）

        Raises:
            langcodes.LanguageTagError: 無効な言語コード形式の場合

        Examples:
            >>> EngineMetadata.to_iso639_1("zh-CN")
            'zh'
            >>> EngineMetadata.to_iso639_1("pt-BR")
            'pt'
            >>> EngineMetadata.to_iso639_1("ja")
            'ja'
            >>> EngineMetadata.to_iso639_1("ZH-CN")  # 大文字も自動正規化
            'zh'
            >>> EngineMetadata.to_iso639_1("yue")  # ISO 639-3 はパススルー
            'yue'
            >>> EngineMetadata.to_iso639_1("auto")  # パススルー
            'auto'
        """
        return langcodes.Language.get(code).language

    @classmethod
    def resolve_language(cls, engine_id: str, requested: Optional[str]) -> str:
        """CLI ``--language`` を engine 別の実効言語に解決する (Issue #365)。

        単一解決点: 未指定 (None/空) は ``cli_default_language``、明示指定は
        BCP-47 → primary language subtag へ正規化 (ISO 639-1 のほか ``yue``
        等の 3 文字 code も含む) + ``supported_languages`` 検証、``auto`` は
        ``supports_language_auto`` の engine のみ許可。全ての拒否は
        モデルロード前に ``ValueError`` で fail-fast する (silent fallback 禁止)。

        Args:
            engine_id: engine ID (``_ENGINES`` の key)
            requested: ユーザー指定の言語コード。None/空文字は未指定扱い。

        Returns:
            解決済み言語コード (BCP-47 primary language subtag、
            または auto 対応 engine の "auto")

        Raises:
            ValueError: unknown engine / 不正形式コード / 非対応言語 /
                auto 非対応 engine への "auto" 指定
        """
        info = cls.get(engine_id)
        if info is None:
            raise ValueError(
                f"Unknown engine type: {engine_id}. "
                f"Available engines: {sorted(cls._ENGINES)}"
            )
        if not requested:
            return info.cli_default_language
        try:
            normalized = cls.to_iso639_1(requested)
        except langcodes.LanguageTagError as exc:
            raise ValueError(f"Invalid language code {requested!r}: {exc}") from exc
        if normalized == "auto":
            if info.supports_language_auto:
                return "auto"
            raise ValueError(
                f"Engine '{engine_id}' does not support automatic language "
                f"detection (--language auto). Supported languages: "
                f"{cls._format_supported(info.supported_languages)}"
            )
        if normalized not in info.supported_languages:
            raise ValueError(
                f"Language '{normalized}' is not supported by engine "
                f"'{engine_id}'. Supported: "
                f"{cls._format_supported(info.supported_languages)}"
            )
        return normalized

    @staticmethod
    def _format_supported(languages: List[str], limit: int = 10) -> str:
        """エラーメッセージ用の対応言語一覧 (whispers2t の 99 言語対策で切り詰め)。"""
        if len(languages) <= limit:
            return ", ".join(languages)
        return f"{', '.join(languages[:limit])}, ... ({len(languages)} languages total)"


# ===== recommend() 用の内部 helper (Issue #286) =====

_QUALITY_SCORE: Dict[str, float] = {"best": 2.0, "good": 1.0, "fallback": 0.0}

# VRAM 適合判定の安全マージン (can_fit_on_gpu の precedent に準拠)
_VRAM_SAFETY_MARGIN = 0.9


def _compute_scores(
    info: EngineInfo,
    quality: str,
    gpu_available: bool,
    vram_gb: Optional[float],
) -> Dict[str, float]:
    """recommend() の分解スコアを算出する。

    単一の重み付き合計は使わず、各観点を個別に露出して sort key に使う
    (透明性 + magic weight 回避)。
    """
    quality_score = _QUALITY_SCORE.get(quality, 0.0)

    if gpu_available:
        req = info.vram_required_mb
        if req is None or vram_gb is None:
            hardware_fit = 0.8  # 未知/軽量は fit を仮定
        elif req <= vram_gb * 1024 * _VRAM_SAFETY_MARGIN:
            hardware_fit = 1.0
        else:
            hardware_fit = 0.0  # 超過: 除外せず沈める
        latency = 1.0 if info.streaming else 0.5
    else:
        # CPU シナリオ: 軽量/realtime 可の engine を上位に、重い engine を沈める
        if info.realtime_on_cpu:
            hardware_fit = 1.0
        elif info.cpu_recommended:
            hardware_fit = 0.5
        else:
            hardware_fit = 0.1
        latency = hardware_fit

    streaming_score = 1.0 if info.streaming else 0.0

    return {
        "quality_score": quality_score,
        "hardware_fit_score": hardware_fit,
        "latency_score": latency,
        "streaming_score": streaming_score,
    }


def _recommend_whisper_size(gpu_available: bool, vram_gb: Optional[float]) -> str:
    """whispers2t の推奨 model_size を hardware から選ぶ (heuristic)。

    - CPU: "base" (CPU_SPEED_ESTIMATES で 3-5x realtime、実用)
    - GPU (VRAM 不明 or 6GB+): "large-v3" (最高精度)
    - GPU 2-6GB: "small"
    - GPU <2GB: "base"
    いずれも WhisperS2T の VALID_MODEL_SIZES 内。
    """
    if not gpu_available:
        return "base"
    if vram_gb is None or vram_gb >= 6:
        return "large-v3"
    if vram_gb >= 2:
        return "small"
    return "base"


def _build_reason_codes(
    info: EngineInfo,
    quality: str,
    gpu_available: bool,
    vram_gb: Optional[float],
    n_candidates: int,
) -> List[ReasonCode]:
    """推奨理由を i18n 可能な code 列で組み立てる。"""
    codes: List[ReasonCode] = []

    # 単一言語 engine のみ「特化」。複数言語対応は multilingual
    # (canary 等が best tier でも、その品質は quality field で表現される)。
    if len(info.supported_languages) == 1:
        codes.append(ReasonCode.LANGUAGE_SPECIALIZED)
    else:
        codes.append(ReasonCode.MULTILINGUAL)

    if n_candidates == 1:
        codes.append(ReasonCode.ONLY_CANDIDATE)

    if quality == "fallback":
        codes.append(ReasonCode.FALLBACK)

    # VRAM 適合 (GPU + 要件既知 + 容量既知 のときのみ判定)
    if gpu_available and info.vram_required_mb is not None and vram_gb is not None:
        if info.vram_required_mb <= vram_gb * 1024 * _VRAM_SAFETY_MARGIN:
            codes.append(ReasonCode.FITS_VRAM)
        else:
            codes.append(ReasonCode.EXCEEDS_VRAM)

    if info.streaming:
        codes.append(ReasonCode.STREAMING_SUPPORTED)
    else:
        codes.append(ReasonCode.OFFLINE_ONLY)

    if not gpu_available and info.realtime_on_cpu:
        codes.append(ReasonCode.REALTIME_ON_CPU)

    if info.vram_required_mb is None and info.cpu_recommended:
        codes.append(ReasonCode.LOW_COMPUTE)

    if info.gpu_recommended:
        codes.append(ReasonCode.GPU_RECOMMENDED)

    return codes
