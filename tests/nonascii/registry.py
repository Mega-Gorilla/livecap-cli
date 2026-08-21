"""棚卸し表の source of truth (Issue #378)。

**1 BoundarySpec = 棚卸し表の 1 行。** ``report.py`` がここと
``results.json`` を突き合わせて docs の表をレンダリングするので、
表を手で書き換えてはならない。

行番号は保持しない (数日で腐る)。``callsite_symbol`` を rendering 時に
ファイル内検索して行番号を解決する。``test_registry.py`` が全 callsite の
生存を検査するので、#375 / #379 / #377 がコードを動かしても黙って腐らない。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class Method(str, Enum):
    """Epic #380 が定めた解決方式。優先順位はこの順。

    ``ascii_safe_path()`` (= STAGING) は **第 3 の fallback であり共通解ではない**。
    """

    BUFFER = "①buffer"          # bytes / serialized proto / file-object API がある
    WIDE_PATH = "②wide-path"    # 既に wide path 対応。現状維持
    STAGING = "③staging"        # narrow path。ASCII staging が必要
    FAIL_FAST = "④fail-fast"    # ③も成立しない
    NOT_APPLICABLE = "非該当"    # そもそもパス境界でない


class Section(str, Enum):
    ENGINE_LOAD = "3.1 エンジンモデルロード"
    RUNTIME_TEMP = "3.2 ランタイム temp wav"
    DOWNLOAD = "3.3 ダウンロード / アーカイブ展開"
    AUDIO_IO = "3.4 音声 I/O・ffmpeg"
    OUTPUT_CLI = "3.5 出力・CLI・リソース解決"
    NOT_APPLICABLE = "3.6 非該当"


SECTION_ORDER: tuple[Section, ...] = (
    Section.ENGINE_LOAD,
    Section.RUNTIME_TEMP,
    Section.DOWNLOAD,
    Section.AUDIO_IO,
    Section.OUTPUT_CLI,
    Section.NOT_APPLICABLE,
)


@dataclass(frozen=True)
class BoundarySpec:
    boundary_id: str
    section: Section
    callsite_file: str          # repo 相対
    callsite_symbol: str        # ファイル内に必ず存在する文字列 (行番号解決に使う)
    path_desc: str              # 渡すパスは何か
    receiver: str               # 受け側ライブラリ
    wide_path_support: str      # source-check の見立て
    candidate_method: Method
    rationale: str              # なぜその方式か (非該当 / source_check 行では必須)
    # **実測で確定した方式。** 未実測 / skip / プローブが境界を覆っていない行では
    # None のままにする。issue #378 の ② は「実測で非 ASCII が通る」が採用条件なので、
    # 未実測を ② として数えると「未分類ゼロ」が実態より強い保証に見えてしまう
    # (レビュー指摘 2)。candidate_method は「決定」、verified_method は「証拠」。
    verified_method: Method | None = None
    # プローブが境界の一部しか覆っていない場合にその範囲を明記する。
    measurement_caveat: str | None = None
    # プローブが**境界そのもの**を通しているか。False の行は実測レコードが
    # あっても verified_method を名乗れない — 「実測した」と「境界を実測した」は
    # 別である (レビュー指摘 5)。
    covers_boundary: bool = True
    evidence_kind: str = "runtime"       # runtime | source_check | not_applicable
    probe_id: str | None = None
    tier: str = "cheap"                  # cheap | real_model | heavy | network | none
    granularity: str = "-"               # file | dir | %TEMP% | -
    expected_verdict: str | None = None  # 既知の実測結果 (回帰ゲート)
    # expected_verdict がある特定 variant でのみ成立する場合にその id を書く。
    # 例: cp932 の内側にある「ユーザー」では落ちないが、cp932 の外側にある
    # 「한국어Ω」では落ちる、という encoding 依存の行。None なら全 variant に適用。
    expected_verdict_variant: str | None = None
    # expected_verdict が特定プラットフォームでのみ成立する場合に sys.platform の値を書く。
    # 例: stdout のエンコーディングは Windows (ACP=cp932) では落ちるが、
    # Linux CI (stdout=UTF-8) では落ちない。None なら全プラットフォームに適用。
    expected_verdict_platform: str | None = None
    failure_visibility: str = ""         # 失敗が「黙る」か「落ちる」か
    followup_issue: str | None = None
    unmeasured_reason: str | None = None
    # True の行は control との差分判定を行わない (照会系。ASCII と非 ASCII で
    # 答えが違うこと自体が観測目的であり、差分は「失敗」を意味しない)。
    informational: bool = False


# --- 3.1 エンジンモデルロード -------------------------------------------------

_ENGINE_LOAD: tuple[BoundarySpec, ...] = (
    BoundarySpec(
        boundary_id="engine.reazonspeech.sherpa_from_transducer",
        section=Section.ENGINE_LOAD,
        callsite_file="livecap_cli/engines/reazonspeech_engine.py",
        callsite_symbol="sherpa_onnx.OfflineRecognizer.from_transducer(",
        path_desc="モデルディレクトリ (basedir) に tokens.txt / encoder / decoder / joiner を os.path.join",
        receiver="sherpa-onnx (native, narrow path)",
        wide_path_support="非対応",
        candidate_method=Method.STAGING,
        verified_method=Method.STAGING,
        rationale=(
            "sherpa-onnx 1.12.39 の OfflineModelConfig は tokens を持つが tokens_buf を持たず、"
            "OfflineTransducerModelConfig は encoder_filename / decoder_filename / joiner_filename "
            "のみ → 方式①は利用不可。消費側が basedir に既知の名前を join するので粒度は dir。"
        ),
        probe_id="sherpa.from_transducer.real",
        tier="real_model",
        granularity="dir",
        expected_verdict="fail_silent",
        failure_visibility=(
            "**黙る**。ロードは成功し decode が全件 IndexError。さらに壊れた recognizer が "
            "ModelMemoryCache.set(..., strong=True) でプロセス寿命の間キャッシュされる。"
        ),
        followup_issue="#377",
    ),
    BoundarySpec(
        boundary_id="engine.reazonspeech.hotwords_file",
        section=Section.ENGINE_LOAD,
        callsite_file="livecap_cli/engines/reazonspeech_engine.py",
        callsite_symbol="decoding_method=self.decoding_method",
        path_desc="hotwords ファイル (#361 で追加予定。現時点では未実装)",
        receiver="sherpa-onnx (native, narrow path)",
        wide_path_support="非対応",
        candidate_method=Method.STAGING,
        rationale=(
            "OfflineRecognizerConfig に hotwords_file はあるが hotwords_buf は無い "
            "(1.12.39 で実測) → 方式①不可。#361 実装時に同じ narrow path を踏むため、"
            "先に分類を確定させておく。"
        ),
        evidence_kind="source_check",
        probe_id=None,
        tier="none",
        granularity="file",
        failure_visibility="未実装。#361 実装時に本行を runtime 実測へ格上げすること。",
        unmeasured_reason="#361 未実装のため呼び出し箇所がまだ存在しない",
        followup_issue="#361",
    ),
    BoundarySpec(
        boundary_id="engine.parakeet.nemo_restore_from",
        section=Section.ENGINE_LOAD,
        callsite_file="livecap_cli/engines/parakeet_engine.py",
        callsite_symbol="nemo_asr.models.ASRModel.restore_from(",
        path_desc=(
            "``restore_from`` 呼び出し全体 (実運用条件)。**③ の適用先は ``restore_path`` ではなく "
            "NeMo 内部の %TEMP% 展開先**である"
        ),
        receiver="NeMo (tar 展開) → sentencepiece (native, narrow path)",
        wide_path_support="``restore_path`` は**対応** (実測) / NeMo 内部の %TEMP% 展開先が**非対応**",
        candidate_method=Method.STAGING,
        verified_method=Method.STAGING,
        measurement_caveat=(
            "実運用条件の計測 — .nemo のパスと NeMo 内部の %TEMP% 展開先が"
            "**同時に**非 ASCII になる。どちらが主因かは "
            "engine.nemo.restore_path_only / engine.nemo.untar_temp の 2 行で分離している。"
        ),
        rationale=(
            "base_engine の _load_model_from_path から呼ばれ、unicode-safe な context は "
            "一切効いていない。**当初は .nemo のパスそのものが narrow path 境界だと"
            "見込んでいたが、実測が否定した** — 壊れる原因は NeMo 内部の %TEMP% 展開先"
            "だけである (engine.nemo.untar_temp / engine.nemo.restore_path_only 参照)。"
            "したがってこの呼び出しを直すレバーは **%TEMP% の移設であり、"
            ".nemo の staging ではない**。"
        ),
        probe_id="nemo.restore_from",
        tier="heavy",
        granularity="%TEMP%",
        expected_verdict="fail_silent",
        failure_visibility=(
            "**黙る / すり替わる**。元例外が抽象クラスの二次例外に置換される。加えて "
            "nemo_utils.check_nemo_availability() が NEMO_AVAILABLE=False をプロセス全体に "
            "キャッシュし、呼び出し側は汎用 ImportError('NeMo is not installed') を raise する。"
        ),
        followup_issue="#379",
    ),
    BoundarySpec(
        boundary_id="engine.canary.nemo_restore_from",
        section=Section.ENGINE_LOAD,
        callsite_file="livecap_cli/engines/canary_engine.py",
        callsite_symbol="nemo_asr.models.EncDecMultiTaskModel.restore_from(",
        path_desc=(
            "``restore_from`` 呼び出し全体 (実運用条件)。**③ の適用先は ``restore_path`` ではなく "
            "NeMo 内部の %TEMP% 展開先**である"
        ),
        receiver="NeMo (tar 展開) → sentencepiece (native, narrow path)",
        wide_path_support="``restore_path`` は**対応** (実測) / NeMo 内部の %TEMP% 展開先が**非対応**",
        candidate_method=Method.STAGING,
        verified_method=Method.STAGING,
        measurement_caveat=(
            "実運用条件の計測 — .nemo のパスと NeMo 内部の %TEMP% 展開先が"
            "**同時に**非 ASCII になる。どちらが主因かは "
            "engine.nemo.restore_path_only / engine.nemo.untar_temp の 2 行で分離している。"
        ),
        rationale=(
            "parakeet と同一機構。実 canary-1b .nemo で実測し、同じく fail_silent。"
            "レバーも同じく %TEMP% の移設である。"
        ),
        probe_id="nemo.restore_from",
        tier="heavy",
        granularity="%TEMP%",
        expected_verdict="fail_silent",
        failure_visibility="**黙る / すり替わる** (parakeet と同一)。",
        followup_issue="#379",
    ),
    BoundarySpec(
        boundary_id="engine.nemo.untar_temp",
        section=Section.ENGINE_LOAD,
        callsite_file="livecap_cli/engines/parakeet_engine.py",
        callsite_symbol="restore_path=str(model_path)",
        path_desc="NeMo が内部で選ぶ %TEMP% 展開先 (我々からは名前が見えない)",
        receiver="NeMo internal untar → sentencepiece (narrow path)",
        wide_path_support="非対応",
        candidate_method=Method.STAGING,
        rationale=(
            "**restore_from の副境界として独立した行。** sentencepiece には "
            "LoadFromSerializedProto(bytes) があり sentencepiece 層では方式①が存在するが、"
            "restore_from は自前で untar 先を決めるため NeMo API 越しには到達不能 → "
            "%TEMP% 移設 (ascii_safe_temp_environment) が唯一の手段。粒度は dir。"
            "**実測で確定**: .nemo を ASCII 側に置き %TEMP% だけを非 ASCII にすると"
            "それだけで壊れる。逆に .nemo だけを非 ASCII にしても通る "
            "(engine.nemo.restore_path_only)。つまり**これが唯一の主因**である。"
        ),
        verified_method=Method.STAGING,
        probe_id="nemo.restore_from.ascii_model_nonascii_temp",
        tier="heavy",
        granularity="%TEMP%",
        expected_verdict="fail_silent",
        measurement_caveat=(
            "NeMo 内部の展開先は外から観測できないため間接測定である — "
            "``.nemo`` を ASCII 側に置き ``%TEMP%`` だけを非 ASCII にして、"
            "それだけで壊れるかを見る。"
        ),
        failure_visibility="**黙る**。展開先が非 ASCII だと sentencepiece が読めず二次例外にすり替わる。",
        followup_issue="#379",
    ),
    BoundarySpec(
        boundary_id="engine.nemo.restore_path_only",
        section=Section.ENGINE_LOAD,
        callsite_file="livecap_cli/engines/parakeet_engine.py",
        callsite_symbol="map_location=self.torch_device",
        path_desc="``restore_path`` に渡す .nemo のパスだけを非 ASCII にする (%TEMP% は ASCII 固定)",
        receiver="NeMo (tar 展開) → sentencepiece",
        wide_path_support="**対応 (実測)**",
        candidate_method=Method.WIDE_PATH,
        verified_method=Method.WIDE_PATH,
        rationale=(
            "**因果の切り分け専用の行。** 実運用条件では .nemo のパスと NeMo 内部の "
            "%TEMP% 展開先が**同時に**非 ASCII になるため、どちらが主因か分からない。"
            "この行は %TEMP% を ASCII に固定して .nemo のパスだけを変える。"
            "**結果は pass — restore_path そのものは非 ASCII を正しく扱える。** "
            "当初は ③ を見込んでいたが実測が否定したので ② へ変更した。"
            "#379 にとって決定的で、**.nemo の staging は不要**である。"
        ),
        probe_id="nemo.restore_from.nonascii_model_ascii_temp",
        tier="heavy",
        granularity="file",
        expected_verdict="pass",
    ),
    BoundarySpec(
        boundary_id="engine.voxtral.from_pretrained",
        section=Section.ENGINE_LOAD,
        callsite_file="livecap_cli/engines/voxtral_engine.py",
        callsite_symbol="VoxtralForConditionalGeneration.from_pretrained(",
        path_desc="ローカルモデルディレクトリ (str(model_path))",
        receiver="transformers → safetensors / torch.load",
        wide_path_support="対応 (実測)",
        candidate_method=Method.WIDE_PATH,
        verified_method=Method.WIDE_PATH,
        rationale=(
            "**重み (safetensors 2 shard / 8.8 GB) を含めて実体化し、実際に "
            "VoxtralForConditionalGeneration を構築して確認した実測。** "
            "safetensors.torch.load(data: bytes) と torch.load(f: IO[bytes]) は"
            "いずれも buffer API を持つため、仮に NG なら①へ退避できる。"
        ),
        probe_id="voxtral.from_pretrained",
        tier="real_model",
        granularity="dir",
    ),
    BoundarySpec(
        boundary_id="lib.transformers.autoconfig",
        section=Section.ENGINE_LOAD,
        callsite_file="livecap_cli/engines/voxtral_engine.py",
        callsite_symbol="low_cpu_mem_usage=True",
        path_desc="ローカルモデルディレクトリからの config / safetensors index の解決",
        receiver="transformers (pure Python)",
        wide_path_support="対応 (実測)",
        candidate_method=Method.WIDE_PATH,
        verified_method=Method.WIDE_PATH,
        rationale=(
            "モデルローダ境界の**手前**の層を独立した行として分離する。"
            "これを分けないと「config が読めた」ことをもって「モデルローダが通った」と"
            "誤って主張してしまう (レビュー指摘 5)。実ロード側は "
            "engine.voxtral.from_pretrained が測る。"
        ),
        probe_id="transformers.autoconfig.local_dir",
        tier="real_model",
        granularity="dir",
    ),
    BoundarySpec(
        boundary_id="engine.voxtral.autoprocessor",
        section=Section.ENGINE_LOAD,
        callsite_file="livecap_cli/engines/voxtral_engine.py",
        callsite_symbol="AutoProcessor.from_pretrained(",
        path_desc="ローカルモデルディレクトリ (str(model_path))",
        receiver="transformers → tokenizer / config (mistral-common tekken)",
        wide_path_support="要実測 (tokenizers は Rust native)",
        candidate_method=Method.WIDE_PATH,
        rationale="実測で判定。tokenizers は Rust native なので narrow path の可能性がある。",
        evidence_kind="source_check",
        probe_id="voxtral.autoprocessor",
        tier="real_model",
        granularity="dir",
        unmeasured_reason=(
            "processor の optional 依存 mistral-common が未導入のため skip された。"
            "`uv sync --extra engines-voxtral` を入れた環境で再測定すること。"
        ),
        followup_issue="#387",
    ),
    BoundarySpec(
        boundary_id="engine.whispers2t.load_model",
        section=Section.ENGINE_LOAD,
        callsite_file="livecap_cli/engines/whispers2t_engine.py",
        callsite_symbol="whisper_s2t.load_model(",
        path_desc="HF repo id (パスではない) + 既定 HF cache ディレクトリ",
        receiver="whisper_s2t → huggingface_hub → CTranslate2 (native) + tokenizers",
        wide_path_support="要実測 (CTranslate2 は native)",
        candidate_method=Method.WIDE_PATH,
        rationale=(
            "この engine だけ manager.huggingface_cache() で包まれていないため、"
            "既定の HF cache (= ユーザープロファイル配下) が実世界の経路になる。実測で判定。"
        ),
        evidence_kind="source_check",
        probe_id="whispers2t.load_model",
        tier="real_model",
        granularity="dir",
        unmeasured_reason=(
            "既定 HF cache 配下のモデルを非 ASCII HF_HOME へ再配置する実装が未了。"
            "CTranslate2 は native なので narrow path の可能性があり、real_model tier の"
            "別 PR で実測すること。"
        ),
        followup_issue="#387",
    ),
    BoundarySpec(
        boundary_id="engine.qwen3asr.from_pretrained",
        section=Section.ENGINE_LOAD,
        callsite_file="livecap_cli/engines/qwen3asr_engine.py",
        callsite_symbol="Qwen3ASR.from_pretrained(",
        path_desc="HF repo id + HF_HOME (unicode_safe_download_directory + huggingface_cache 内)",
        receiver="qwen_asr → transformers → HF snapshot + safetensors + tokenizer",
        wide_path_support="要実測",
        candidate_method=Method.WIDE_PATH,
        rationale=(
            "**重要**: 唯一 unicode_safe_download_directory() で包まれた engine だが、"
            "同ヘルパは %TEMP% を cache_root へ移すだけで、その cache_root 自体が "
            "appdirs 既定では**ユーザー名を含む**ため、**包んでも ASCII 安全にはならない**。"
        ),
        evidence_kind="source_check",
        probe_id="qwen3asr.from_pretrained",
        tier="real_model",
        granularity="dir",
        unmeasured_reason=(
            "qwen_asr パッケージ未導入 (engines-qwen3asr extra)。HF snapshot はローカルにある。"
        ),
        followup_issue="#387",
    ),
    BoundarySpec(
        boundary_id="engine.reazonspeech.sherpa_narrow_path_signature",
        section=Section.ENGINE_LOAD,
        callsite_file="livecap_cli/engines/reazonspeech_engine.py",
        callsite_symbol="tokens=os.path.join(basedir",
        path_desc="不正な ONNX + tokens.txt を ASCII / 非 ASCII に置き、エラー署名を比較",
        receiver="sherpa-onnx (native, narrow path)",
        wide_path_support="非対応",
        candidate_method=Method.STAGING,
        covers_boundary=False,
        measurement_caveat=(
            "不正 ONNX が tokens.txt より先に検証されるため ONNX 層までしか到達しない。既知 NG の本体は real_model tier でのみ観測できる。"
        ),
        rationale=(
            "**実モデル無しで narrow path を判定する軽量プローブ。** ASCII で「protobuf の"
            "解析に失敗」、非 ASCII で「ファイルを開けない」となれば、740 MB のモデルを"
            "使わずに narrow path が確定する。実モデル行 "
            "(engine.reazonspeech.sherpa_from_transducer) の cheap tier 裏付け。"
        ),
        probe_id="sherpa.from_transducer.diff",
        tier="cheap",
        granularity="dir",
        failure_visibility=(
            "**この行の pass は「sherpa-onnx が安全」を意味しない。** 不正な ONNX は "
            "tokens.txt より先に検証されるため、本プローブが到達できるのは ONNX 層までで "
            "(ASCII / 非 ASCII のどちらも同じ parse 失敗署名になった)、既知 NG の本体である "
            "tokens.txt の SymbolTable 誤読には届かない。そちらは real_model tier で "
            "fail_silent を再現している。"
        ),
        followup_issue="#377",
    ),
    BoundarySpec(
        boundary_id="lib.onnxruntime.inference_session",
        section=Section.ENGINE_LOAD,
        callsite_file="livecap_cli/engines/reazonspeech_engine.py",
        callsite_symbol="encoder=os.path.join(basedir",
        path_desc="encoder / decoder / joiner の .onnx パス (sherpa-onnx 内部で ORT へ渡る)",
        receiver="onnxruntime (native)",
        wide_path_support="対応 (実測済み)",
        candidate_method=Method.WIDE_PATH,
        verified_method=Method.WIDE_PATH,
        rationale=(
            "issue #378 の初期リストで「実測済み (OK)」とされていた層。**同一プロセス内で"
            "ライブラリごとに対応状況がバラバラ**という主張の片側の実証であり、"
            "sherpa-onnx が NG でも ORT は OK であることを固定する。"
            "``InferenceSession`` は bytes も受けるので方式①への退避路もある。"
        ),
        probe_id="onnxruntime.InferenceSession.str_path",
        tier="cheap",
        granularity="file",
    ),
    BoundarySpec(
        boundary_id="lib.torch.load",
        section=Section.ENGINE_LOAD,
        callsite_file="livecap_cli/engines/voxtral_engine.py",
        callsite_symbol="torch_dtype=",
        path_desc="重みファイルのパス (transformers 内部で torch.load へ渡る)",
        receiver="torch (native)",
        wide_path_support="対応の見込み。方式①も可 (IO[bytes] を受ける)",
        candidate_method=Method.WIDE_PATH,
        verified_method=Method.WIDE_PATH,
        rationale="``torch.load(f: str | PathLike | IO[bytes])`` なので、NG でも①へ退避できる。",
        probe_id="torch.load.path",
        tier="cheap",
        granularity="file",
    ),
    BoundarySpec(
        boundary_id="lib.safetensors.load_file",
        section=Section.ENGINE_LOAD,
        callsite_file="livecap_cli/engines/voxtral_engine.py",
        callsite_symbol="use_safetensors=True",
        path_desc="safetensors 重みファイルのパス",
        receiver="safetensors (Rust native)",
        wide_path_support="対応の見込み。方式①も可 (load(data: bytes) がある)",
        candidate_method=Method.WIDE_PATH,
        verified_method=Method.WIDE_PATH,
        rationale="``safetensors.torch.load(data: bytes)`` があるため、NG でも①へ退避できる。",
        probe_id="safetensors.load_file.path",
        tier="cheap",
        granularity="file",
    ),
    BoundarySpec(
        boundary_id="lib.tokenizers.from_file",
        section=Section.ENGINE_LOAD,
        callsite_file="livecap_cli/engines/whispers2t_engine.py",
        callsite_symbol="backend=",
        path_desc="tokenizer.json のパス (whispers2t / transformers が共有する層)",
        receiver="tokenizers (Rust native)",
        wide_path_support="要実測",
        candidate_method=Method.WIDE_PATH,
        verified_method=Method.WIDE_PATH,
        rationale="Rust native なので narrow path の可能性がある。実測で確定させる。",
        probe_id="tokenizers.from_file",
        tier="cheap",
        granularity="file",
    ),
    BoundarySpec(
        boundary_id="engine.base.verify_model_integrity",
        section=Section.ENGINE_LOAD,
        callsite_file="livecap_cli/engines/base_engine.py",
        callsite_symbol="def _verify_model_integrity",
        path_desc="ダウンロード済みモデルファイル (open(model_path, 'rb'))",
        receiver="CPython builtin open",
        wide_path_support="対応 (CPython は *W API)",
        candidate_method=Method.WIDE_PATH,
        verified_method=Method.WIDE_PATH,
        rationale="CPython のみを経由するので wide path。ただし失敗の可視性に別途問題がある。",
        probe_id="stdlib.open_read",
        tier="cheap",
        granularity="file",
        failure_visibility=(
            "**黙る**。except Exception: return False で呼び出し側がファイルを削除し "
            "ValueError('ダウンロードしたモデルが破損') を raise するため、真因 "
            "(権限・エンコーディング等) が消える。"
        ),
    ),
)


# --- 3.2 ランタイム temp wav ---------------------------------------------------


def _utterance_wav_row(
    engine: str, file: str, symbol: str, anchored: str
) -> BoundarySpec:
    return BoundarySpec(
        boundary_id=f"engine.{engine}.utterance_wav",
        section=Section.RUNTIME_TEMP,
        callsite_file=file,
        callsite_symbol=symbol,
        path_desc=f"発話ごとの一時 wav ({anchored})",
        receiver="soundfile (書き込み) → ネイティブ ASR (読み込み)",
        wide_path_support="書き込みは対応 (sf_wchar_open) / 読み込み側は engine 依存",
        candidate_method=Method.STAGING,
        rationale=(
            "**書き込みはバグではない** — soundfile は Windows で sf_wchar_open を使う。"
            "バグは書いた path をネイティブ ASR に渡す側。正解は ascii_safe_workspace() で"
            "**最初から ASCII 空間に ASCII 名で作る**こと (非 ASCII %TEMP% に作ってから "
            "staging するのではない)。ascii_safe_temp_environment は発話ごとにプロセス"
            "グローバル状態を書き換えるので**使ってはならない**。"
        ),
        probe_id="tempfile.named_temporary_wav",
        tier="cheap",
        granularity="dir",
        covers_boundary=False,
        # プローブが覆うのは producer 側 (%TEMP% への sf.write と読み戻し) だけで、
        # 本当の境界である「その path をネイティブ ASR に渡す側」には届かない。
        # したがって verified_method は None のままにする (レビュー指摘 2 / 5 と同じ規律)。
        measurement_caveat=(
            "プローブが覆うのは producer 側 (注入した %TEMP% への sf.write と読み戻し) のみ。"
            "本当の境界である consumer (model.transcribe([tmp]) = ネイティブ ASR) は "
            "real_model / heavy tier でしか測れないため未確定。"
        ),
        followup_issue="#375",
    )


_RUNTIME_TEMP: tuple[BoundarySpec, ...] = (
    _utterance_wav_row(
        "parakeet",
        "livecap_cli/engines/parakeet_engine.py",
        "tempfile.NamedTemporaryFile(suffix='.wav', delete=False)",
        "dir= 指定なし → 素の %TEMP%",
    ),
    _utterance_wav_row(
        "canary",
        "livecap_cli/engines/canary_engine.py",
        "tempfile.NamedTemporaryFile(suffix='.wav', delete=False)",
        "dir= 指定なし → 素の %TEMP%",
    ),
    _utterance_wav_row(
        "qwen3asr",
        "livecap_cli/engines/qwen3asr_engine.py",
        "tempfile.NamedTemporaryFile(suffix='.wav', delete=False)",
        "dir= 指定なし → 素の %TEMP% (auto-detect 経路のみ)",
    ),
    _utterance_wav_row(
        "whispers2t",
        "livecap_cli/engines/whispers2t_engine.py",
        "tempfile.NamedTemporaryFile(",
        "dir=self._tmp_dir → cache_root/whispers2t (唯一 %TEMP% を避けている)",
    ),
    _utterance_wav_row(
        "voxtral",
        "livecap_cli/engines/voxtral_engine.py",
        "voxtral_temp_",
        "get_temp_dir() → cache_root/runtime",
    ),
    BoundarySpec(
        boundary_id="lib.soundfile.write",
        section=Section.RUNTIME_TEMP,
        callsite_file="livecap_cli/engines/voxtral_engine.py",
        callsite_symbol="sf.write(",
        path_desc="発話 wav の書き込み先パス",
        receiver="soundfile / libsndfile",
        wide_path_support="対応 (soundfile.py が sf_wchar_open を使う)",
        candidate_method=Method.WIDE_PATH,
        verified_method=Method.WIDE_PATH,
        rationale=(
            "**発話 wav 問題の「書き込み側」を独立した行として分離する。** "
            "soundfile は wide path 対応なので**書き込みはバグではない** — "
            "バグは書いた path をネイティブ ASR に渡す側 (上記 utterance_wav 各行) にある。"
            "この区別を表の上で明示しないと、#375 が直す場所を取り違える。"
        ),
        probe_id="soundfile.write.path",
        tier="cheap",
        granularity="file",
    ),
)


# --- 3.3 ダウンロード / アーカイブ展開 -----------------------------------------

_DOWNLOAD: tuple[BoundarySpec, ...] = (
    BoundarySpec(
        boundary_id="resources.model_manager.urlretrieve",
        section=Section.DOWNLOAD,
        callsite_file="livecap_cli/resources/model_manager.py",
        callsite_symbol="urllib.request.urlretrieve(",
        path_desc="cache_root/downloads 配下のダウンロード先",
        receiver="CPython urllib",
        wide_path_support="対応 (CPython)",
        candidate_method=Method.WIDE_PATH,
        verified_method=Method.WIDE_PATH,
        measurement_caveat=(
            "file:// を source にした計測。ネットワーク経路は未計測 (保存先パスの扱いは同一)。"
        ),
        rationale="CPython のみ。file:// URL で実測する (ネットワーク不要)。",
        probe_id="urllib.urlretrieve.file_url",
        tier="cheap",
        granularity="file",
    ),
    BoundarySpec(
        boundary_id="resources.model_manager.hf_home",
        section=Section.DOWNLOAD,
        callsite_file="livecap_cli/resources/model_manager.py",
        callsite_symbol='os.environ["HF_HOME"]',
        path_desc="HF_HOME 環境変数経由で huggingface_hub に渡る cache ディレクトリ",
        receiver="huggingface_hub / transformers",
        wide_path_support="対応の見込み (pure Python)",
        candidate_method=Method.WIDE_PATH,
        verified_method=Method.WIDE_PATH,
        rationale="huggingface_hub は pure Python。実測で確定。",
        probe_id="huggingface_hub.local_files_only",
        tier="cheap",
        granularity="dir",
    ),
    BoundarySpec(
        boundary_id="engine.reazonspeech.snapshot_download",
        section=Section.DOWNLOAD,
        callsite_file="livecap_cli/engines/reazonspeech_engine.py",
        callsite_symbol="snapshot_download(",
        path_desc="cache_dir=str(hf_cache)",
        receiver="huggingface_hub",
        wide_path_support="対応の見込み (pure Python)",
        candidate_method=Method.WIDE_PATH,
        verified_method=Method.WIDE_PATH,
        measurement_caveat=(
            "local_files_only での計測。実ダウンロード時の一時ファイル / ロック処理は未計測。"
        ),
        rationale="pure Python。cheap tier の local_files_only プローブが同一コード経路を通る。",
        probe_id="huggingface_hub.local_files_only",
        tier="cheap",
        granularity="dir",
    ),
    BoundarySpec(
        boundary_id="engine.reazonspeech.tarfile_extract",
        section=Section.DOWNLOAD,
        callsite_file="livecap_cli/engines/reazonspeech_engine.py",
        callsite_symbol="tarfile.open(",
        path_desc="アーカイブパス + 展開先ディレクトリ (+ メンバ名)",
        receiver="CPython tarfile",
        wide_path_support="対応 (CPython)",
        candidate_method=Method.WIDE_PATH,
        verified_method=Method.WIDE_PATH,
        rationale="CPython のみ。展開先ディレクトリ名 × メンバ名の 2 軸で実測する。",
        probe_id="tarfile.extractall",
        tier="cheap",
        granularity="dir",
    ),
    BoundarySpec(
        boundary_id="resources.ffmpeg_manager.zipfile_extract",
        section=Section.DOWNLOAD,
        callsite_file="livecap_cli/resources/ffmpeg_manager.py",
        callsite_symbol="zipfile.ZipFile(",
        path_desc="アーカイブパス + 展開先ディレクトリ (+ メンバ名)",
        receiver="CPython zipfile",
        wide_path_support="対応 (CPython)",
        candidate_method=Method.WIDE_PATH,
        verified_method=Method.WIDE_PATH,
        rationale="CPython のみ。",
        probe_id="zipfile.extractall",
        tier="cheap",
        granularity="dir",
    ),
    BoundarySpec(
        boundary_id="utils.unicode_safe_download_directory",
        section=Section.DOWNLOAD,
        callsite_file="livecap_cli/utils/__init__.py",
        callsite_symbol="def unicode_safe_download_directory",
        path_desc="TEMP / TMP / TMPDIR / tempfile.tempdir を cache_root/downloads へ移設",
        receiver="プロセス全体 (os.environ + tempfile.tempdir)",
        wide_path_support="**移設先自体が ASCII 保証でない**",
        candidate_method=Method.STAGING,
        covers_boundary=False,
        measurement_caveat=(
            "プローブが測るのは共有 rmtree によるデータ消失であり、ASCII 保証の有無ではない。非 ASCII 軸では control と同挙動 (pass)。"
        ),
        rationale=(
            "cache_root は appdirs 既定では**ユーザー名を含む**ため、本ヘルパは TEMP 移設"
            "ヘルパであって ASCII 安全ヘルパではない。加えてロック無し・refcount 無し・"
            "ネスト深度カウンタ無しで、cleanup が**共有**ディレクトリを rmtree する。"
            "#375 で ascii_safe_temp_environment へ転送して修理する。"
        ),
        probe_id="utils.download_dir_data_loss",
        tier="cheap",
        granularity="%TEMP%",
        # 非 ASCII 軸では control と同じ挙動なので verdict は pass になる。
        # データ消失そのものは非 ASCII 依存ではないため、
        # test_probes.py::test_download_directory_data_loss_is_recorded が
        # 観測値に対して直接 assert する。
        failure_visibility=(
            "**黙ってデータを消す**。download スコープが開いている間、プロセス内のあらゆる "
            "NamedTemporaryFile が downloads/ に飛ばされ、スコープ退出時の共有 rmtree で"
            "削除される (発話 wav を含む)。"
        ),
        followup_issue="#386",
    ),
    BoundarySpec(
        boundary_id="utils.unicode_safe_temp_directory",
        section=Section.DOWNLOAD,
        callsite_file="livecap_cli/utils/__init__.py",
        callsite_symbol="def unicode_safe_temp_directory",
        path_desc="TEMP を cache_root/runtime へ移設 (**デッドコード**)",
        receiver="プロセス全体 (os.environ + tempfile.tempdir)",
        wide_path_support="**移設先自体が ASCII 保証でない**",
        candidate_method=Method.STAGING,
        verified_method=Method.STAGING,
        rationale=(
            "4 engine が import しているが**呼び出しはゼロ**。移設先が cache_root/runtime "
            "(appdirs 既定ではユーザー名を含む) なので ASCII 保証がない。"
            "#375 で deprecate → 次マイナーで削除。"
        ),
        evidence_kind="runtime",
        probe_id="utils.temp_helper_is_not_ascii_safe",
        tier="cheap",
        granularity="%TEMP%",
        expected_verdict="fail_silent",
        # ASCII-only の space_paren variant では移設先も ASCII のままなので pass する。
        # 「移設先が ASCII 保証でない」という主張は非 ASCII variant で成立する。
        expected_verdict_variant="cjk_kana",
        failure_visibility="デッドコードのため実害は無いが、ASCII 安全策と誤解される危険がある。",
        followup_issue="#375",
    ),
)


# --- 3.4 音声 I/O・ffmpeg -------------------------------------------------------

_AUDIO_IO: tuple[BoundarySpec, ...] = (
    BoundarySpec(
        boundary_id="audio_sources.file.sf_read",
        section=Section.AUDIO_IO,
        callsite_file="livecap_cli/audio_sources/file.py",
        callsite_symbol="sf.read(self.file_path",
        path_desc="ユーザー指定の入力音声パス (Path オブジェクトをそのまま渡す)",
        receiver="soundfile / libsndfile",
        wide_path_support="対応の見込み (soundfile.py が sf_wchar_open を使う)",
        candidate_method=Method.WIDE_PATH,
        verified_method=Method.WIDE_PATH,
        rationale="soundfile.py に sf_wchar_open の使用を実物確認済み。実測で確定させる。",
        probe_id="soundfile.read.path",
        tier="cheap",
        granularity="file",
    ),
    BoundarySpec(
        boundary_id="transcription.file_pipeline.temp_root",
        section=Section.AUDIO_IO,
        callsite_file="livecap_cli/transcription/file_pipeline.py",
        callsite_symbol='tempfile.mkdtemp(prefix="livecap-file-pipeline-")',
        path_desc="pipeline の作業ディレクトリ (**cache_root ではなくシステム %TEMP%**)",
        receiver="CPython tempfile → 後段の ffmpeg / soundfile",
        wide_path_support="対応 (実測)",
        candidate_method=Method.WIDE_PATH,
        verified_method=Method.WIDE_PATH,
        rationale=(
            "unicode_safe_* を一切通らずシステム %TEMP% を使い、さらにユーザーのファイル名 "
            "stem がそのまま temp ファイル名になるため、当初は ③ を見込んでいた。"
            "**しかし実測で否定された** — 非 ASCII の %TEMP% × 非 ASCII stem で "
            "抽出〜ロードまで通る。後段の消費者 (ffmpeg-python の argv / soundfile / "
            "librosa) がいずれも wide path 対応であるため。証拠に従って ② へ変更した。"
        ),
        probe_id="pipeline.extract_audio.nonascii_stem",
        tier="cheap",
        granularity="dir",
    ),
    BoundarySpec(
        boundary_id="transcription.file_pipeline.ffmpeg_input",
        section=Section.AUDIO_IO,
        callsite_file="livecap_cli/transcription/file_pipeline.py",
        callsite_symbol="ffmpeg.input(str(source))",
        path_desc="ユーザー指定の入力ファイルパス",
        receiver="ffmpeg-python → subprocess argv (シェル文字列ではない)",
        wide_path_support="要実測 (CreateProcessW 経由の list-argv)",
        candidate_method=Method.WIDE_PATH,
        verified_method=Method.WIDE_PATH,
        rationale="ffmpeg-python は argv list を組んで subprocess.Popen する。実測で確定。",
        probe_id="ffmpeg.input_path",
        tier="cheap",
        granularity="file",
    ),
    BoundarySpec(
        boundary_id="transcription.file_pipeline.ffmpeg_output",
        section=Section.AUDIO_IO,
        callsite_file="livecap_cli/transcription/file_pipeline.py",
        callsite_symbol='f"{source.stem}_audio.wav"',
        path_desc="**ユーザーのファイル名 stem から組み立てた** temp wav の出力先",
        receiver="ffmpeg-python → subprocess argv",
        wide_path_support="要実測",
        candidate_method=Method.WIDE_PATH,
        verified_method=Method.WIDE_PATH,
        rationale=(
            "cache root が ASCII でも、**ユーザーのファイル名**が非 ASCII なら temp パスが"
            "非 ASCII になる。cache root の行とは独立した行として扱う。"
        ),
        probe_id="ffmpeg.output_path",
        tier="cheap",
        granularity="file",
    ),
    BoundarySpec(
        boundary_id="transcription.file_pipeline.ffmpeg_binary",
        section=Section.AUDIO_IO,
        callsite_file="livecap_cli/transcription/file_pipeline.py",
        callsite_symbol="cmd=self._ffmpeg_path",
        path_desc="ffmpeg 実行ファイルのパス",
        receiver="subprocess (CreateProcessW)",
        wide_path_support="要実測",
        candidate_method=Method.WIDE_PATH,
        verified_method=Method.WIDE_PATH,
        rationale="実測で確定。",
        probe_id="ffmpeg.binary_path",
        tier="cheap",
        granularity="file",
    ),
    BoundarySpec(
        boundary_id="transcription.file_pipeline.ffmpeg_env_export",
        section=Section.AUDIO_IO,
        callsite_file="livecap_cli/transcription/file_pipeline.py",
        callsite_symbol='os.environ.setdefault("FFMPEG_BINARY"',
        path_desc="解決済み ffmpeg / ffprobe パスをプロセス env に流す",
        receiver="pydub / moviepy 系の第三者コンシューマ",
        wide_path_support="対応 (env は str)",
        candidate_method=Method.WIDE_PATH,
        rationale=(
            "env に str を置くだけで、実際に消費するのは第三者ライブラリ。"
            "本リポジトリが制御できる境界ではないため runtime 実測の対象外。"
        ),
        evidence_kind="source_check",
        probe_id=None,
        tier="none",
        granularity="-",
        unmeasured_reason=(
            "実際の消費者は pydub / moviepy 系の第三者ライブラリであり、本リポジトリからは"
            "観測できない。source-check で ② と判定する。"
        ),
        followup_issue="#387",
    ),
    BoundarySpec(
        boundary_id="engine.librosa_resample",
        section=Section.NOT_APPLICABLE,
        callsite_file="livecap_cli/engines/parakeet_engine.py",
        callsite_symbol="librosa.resample(",
        path_desc="なし (ndarray in / ndarray out)",
        receiver="librosa",
        wide_path_support="n/a",
        candidate_method=Method.NOT_APPLICABLE,
        rationale=(
            "**パス境界ではない。** ndarray を受け渡すだけ。issue #378 の調査対象初期リストに "
            "「librosa の内部リーダ」として挙がっていたが、``librosa.resample`` はファイルを"
            "開かない。実際にパスを扱うのは ``librosa.load`` 側 (別行)。"
            "「未分類ゼロ」を証明するために明示的に列挙する。"
        ),
        evidence_kind="not_applicable",
        probe_id=None,
        tier="none",
    ),
    BoundarySpec(
        boundary_id="transcription.file_pipeline.librosa_load",
        section=Section.AUDIO_IO,
        callsite_file="livecap_cli/transcription/file_pipeline.py",
        callsite_symbol="def _load_audio",
        path_desc="音声ファイルパス (librosa の内部リーダ経路)",
        receiver="librosa → soundfile / audioread",
        wide_path_support="対応の見込み。方式①も可 (BinaryIO を受ける)",
        candidate_method=Method.WIDE_PATH,
        verified_method=Method.WIDE_PATH,
        rationale="librosa.load のシグネチャは BinaryIO も受けるため、必要なら①へ退避できる。",
        probe_id="librosa.load.path",
        tier="cheap",
        granularity="file",
    ),
)


# --- 3.5 出力・CLI・リソース解決 -----------------------------------------------

_OUTPUT_CLI: tuple[BoundarySpec, ...] = (
    BoundarySpec(
        boundary_id="transcription.srt.write_srt",
        section=Section.OUTPUT_CLI,
        callsite_file="livecap_cli/transcription/srt.py",
        callsite_symbol="def write_srt",
        path_desc="SRT 出力先パス",
        receiver="CPython open(..., encoding='utf-8')",
        wide_path_support="対応 (CPython)",
        candidate_method=Method.WIDE_PATH,
        verified_method=Method.WIDE_PATH,
        rationale="CPython のみ。ネイティブライブラリを経由しない。",
        probe_id="srt.write_srt",
        tier="cheap",
        granularity="file",
    ),
    BoundarySpec(
        boundary_id="cli.path_arguments",
        section=Section.OUTPUT_CLI,
        callsite_file="livecap_cli/cli.py",
        callsite_symbol='"--output"',
        path_desc="input_file (positional) と -o/--output。いずれも素の str",
        receiver="argparse → Path()",
        wide_path_support="対応 (str→Path は無損失)",
        candidate_method=Method.WIDE_PATH,
        rationale="CPython のみ。ただしここから ③ の行へパスが流入する入口である。",
        evidence_kind="source_check",
        probe_id=None,
        tier="none",
        granularity="file",
        unmeasured_reason=(
            "argparse は CPython のみを経由し情報を失わない。ここは ③ の境界へパスが流入する"
            "入口であって、それ自体が壊れる箇所ではないため runtime 実測の対象外とする。"
        ),
    ),
    BoundarySpec(
        boundary_id="cli.stderr_path_print",
        section=Section.OUTPUT_CLI,
        callsite_file="livecap_cli/cli.py",
        callsite_symbol="Transcribing:",
        path_desc="非 ASCII パスを stderr へ出力する",
        receiver="コンソール / リダイレクト先のエンコーダ",
        wide_path_support="n/a (エンコーディングの話であってパスの話ではない)",
        candidate_method=Method.WIDE_PATH,
        verified_method=Method.WIDE_PATH,
        rationale=(
            "**実測で安全と判明**。CPython は ``sys.stderr`` に ``backslashreplace`` を"
            "既定で使う (パイプ接続時: cp932 / backslashreplace を実測)。ACP に無い文字も"
            "エスケープされるだけで例外にならない。当初 ④fail-fast と見込んでいたが、"
            "実測により ② へ変更した。**リスクがあるのは stdout 側** (下記)。"
        ),
        probe_id="stdio.stderr_path",
        tier="cheap",
        granularity="-",
        failure_visibility="落ちない (エスケープされる)。",
    ),
    BoundarySpec(
        boundary_id="cli.stdout_srt_write",
        section=Section.OUTPUT_CLI,
        callsite_file="livecap_cli/cli.py",
        callsite_symbol="sys.stdout.write",
        path_desc="SRT 本文 (認識結果テキスト) と パス文字列を stdout へ出力する",
        receiver="コンソール / リダイレクト先のエンコーダ",
        wide_path_support="n/a (エンコーディングの話)",
        candidate_method=Method.FAIL_FAST,
        verified_method=Method.FAIL_FAST,
        measurement_caveat=(
            "Windows (ACP != UTF-8) でのみ落ちる。Linux CI では stdout が UTF-8 のため pass。"
        ),
        rationale=(
            "**本調査で新規に発見した経路。** ``sys.stdout`` は ``surrogateescape`` で "
            "``backslashreplace`` ではないため、ACP (cp932) に無い文字を書くと "
            "``UnicodeEncodeError`` で落ちる (実測: 韓国語 + Ω を stdout へ書いて exit 1)。"
            "SRT 本文には任意言語の認識結果が乗るので実害がある。対処は staging ではなく "
            "出力ストリームの明示的な UTF-8 化 / errors 指定。**別 issue を起票すること。**"
            "既存の回帰テスト precedent: tests/core/cli/test_transcribe_file.py の "
            "cp932 コンソールでの --help ガード。"
            "**プラットフォーム依存**: Linux CI では stdout が UTF-8 のため pass する "
            "(実測)。落ちるのは ACP が UTF-8 でない Windows のみ。"
        ),
        probe_id="stdio.stdout_path",
        tier="cheap",
        granularity="-",
        expected_verdict="fail_loud",
        expected_verdict_variant="outside_acp",
        expected_verdict_platform="win32",
        failure_visibility="**落ちる**。ただし真因と無関係な UnicodeEncodeError として現れる。",
        followup_issue="#385",
    ),
    BoundarySpec(
        boundary_id="resources.model_manager.roots",
        section=Section.OUTPUT_CLI,
        callsite_file="livecap_cli/resources/model_manager.py",
        callsite_symbol="ENV_MODELS_DIR",
        path_desc="models_root / cache_root (env var または appdirs 既定)",
        receiver="CPython pathlib → 後段の全境界",
        wide_path_support="対応 (CPython)",
        candidate_method=Method.WIDE_PATH,
        verified_method=Method.WIDE_PATH,
        rationale=(
            "根の注入機構そのもの。既定値 appdirs.user_cache_dir('LiveCap','PineLab') は"
            "**ユーザー名を含む**ため ASCII 保証がない。#375 の「既定 data root の ASCII 保証」"
            "が対象とする箇所。ハーネスの前提条件テストも兼ねる。"
        ),
        probe_id="model_manager.roots",
        tier="cheap",
        granularity="dir",
        followup_issue="#375",
    ),
    BoundarySpec(
        boundary_id="resources.resource_locator.env_root",
        section=Section.OUTPUT_CLI,
        callsite_file="livecap_cli/resources/resource_locator.py",
        callsite_symbol="ENV_RESOURCE_ROOT",
        path_desc="LIVECAP_RESOURCE_ROOT からの同梱リソース解決",
        receiver="CPython pathlib / importlib.resources",
        wide_path_support="対応 (CPython)",
        candidate_method=Method.WIDE_PATH,
        verified_method=Method.WIDE_PATH,
        rationale="CPython のみ。",
        probe_id="resource_locator.env_root",
        tier="cheap",
        granularity="dir",
    ),
    BoundarySpec(
        boundary_id="resources.resource_locator.source_root",
        section=Section.OUTPUT_CLI,
        callsite_file="livecap_cli/resources/resource_locator.py",
        callsite_symbol="Path(__file__).resolve()",
        path_desc="**インストール先ディレクトリ**から導出される探索 root",
        receiver="CPython pathlib / importlib.resources",
        wide_path_support="対応 (CPython) だが後段の消費者に依存",
        candidate_method=Method.WIDE_PATH,
        rationale=(
            "非 ASCII なディレクトリへインストールした場合にここから非 ASCII が流入する。"
            "CPython 側は wide path だが、そこから ③ の境界へ渡ると問題になる。"
        ),
        evidence_kind="source_check",
        probe_id=None,
        tier="none",
        granularity="dir",
        unmeasured_reason=(
            "非 ASCII パス配下への第二 install tree が必要 (site-packages を丸ごと複製する)。"
            "本 issue のコストに見合わないため未実測。#375 着手時に判断する。"
        ),
        followup_issue="#387",
    ),
    BoundarySpec(
        boundary_id="logging.file_handler",
        section=Section.NOT_APPLICABLE,
        callsite_file="livecap_cli/utils/__init__.py",
        callsite_symbol="import logging",
        path_desc="ログファイルの出力先パス",
        receiver="CPython logging (FileHandler)",
        wide_path_support="n/a",
        candidate_method=Method.NOT_APPLICABLE,
        rationale=(
            "**本リポジトリには存在しない境界。** issue #378 の調査対象初期リストに"
            "「出力側: ログファイルパス」として挙がっていたが、実際に検索すると "
            "``livecap_cli/`` 配下に ``logging.FileHandler`` も "
            "``basicConfig(filename=...)`` も無い — ログは stream のみで、"
            "**ファイル出力先を構成するのは host アプリの責務**である。"
            "したがって非該当だが、初期リストの項目を落とさないため明示的に列挙する。"
            "host 側の可観測性は livecap-gui#405 (起動ログに解決済みリソースパスを"
            "出力する) が扱う。"
        ),
        evidence_kind="not_applicable",
        probe_id=None,
        tier="none",
        followup_issue="livecap-gui#405",
    ),
    BoundarySpec(
        boundary_id="win32.short_path_name",
        section=Section.NOT_APPLICABLE,
        callsite_file="tests/nonascii/paths.py",
        callsite_symbol="def short_path_name",
        path_desc="非 ASCII ディレクトリの 8.3 短縮名を照会する",
        receiver="kernel32.GetShortPathNameW",
        wide_path_support="n/a",
        candidate_method=Method.NOT_APPLICABLE,
        covers_boundary=False,
        measurement_caveat=(
            "却下理由の照会プローブであり、境界の合否を測るものではない。"
        ),
        rationale=(
            "**却下した代替案の機械記録。** 8.3 短縮名は ASCII staging の代替にならない — "
            "(1)『ユーザー』は 8.3 に収まるので別名が生成されない、(2) 現代の Windows では "
            "8.3 生成が既定無効、(3) 別名が無いとき GetShortPathNameW は**エラーも出さず**"
            "長い名前を返す。散文ではなく実測として残す。"
        ),
        probe_id="win32.short_path_name",
        tier="cheap",
        granularity="-",
        informational=True,
    ),
)


BOUNDARIES: tuple[BoundarySpec, ...] = (
    _ENGINE_LOAD + _RUNTIME_TEMP + _DOWNLOAD + _AUDIO_IO + _OUTPUT_CLI
)

BOUNDARIES_BY_ID: dict[str, BoundarySpec] = {b.boundary_id: b for b in BOUNDARIES}


def resolve_callsite_line(spec: BoundarySpec) -> int | None:
    """``callsite_symbol`` をファイル内検索して行番号を解決する。

    registry に行番号を書かないための仕組み。見つからなければ ``None``
    (``test_registry.py::test_callsites_exist`` が失敗する)。
    """
    path = REPO_ROOT / spec.callsite_file
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    for lineno, line in enumerate(text.splitlines(), 1):
        if spec.callsite_symbol in line:
            return lineno
    return None


def callsite_label(spec: BoundarySpec) -> str:
    """``path/to/file.py:123`` 形式のラベル (行番号は動的解決)。"""
    line = resolve_callsite_line(spec)
    return f"{spec.callsite_file}:{line}" if line else f"{spec.callsite_file}:?"


__all__ = [
    "BOUNDARIES",
    "BOUNDARIES_BY_ID",
    "REPO_ROOT",
    "SECTION_ORDER",
    "BoundarySpec",
    "Method",
    "Section",
    "callsite_label",
    "resolve_callsite_line",
]
