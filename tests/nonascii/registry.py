"""棚卸し表の source of truth (Issue #378)。

**1 BoundarySpec = 棚卸し表の 1 行。** ``report.py`` がここと
``results.json`` を突き合わせて docs の表をレンダリングするので、
表を手で書き換えてはならない。

行番号は保持しない (数日で腐る)。``callsite_symbol`` を rendering 時に
ファイル内検索して行番号を解決する。``test_registry.py`` が全 callsite の
生存を検査するので、#375 / #379 / #377 がコードを動かしても黙って腐らない。
"""

from __future__ import annotations

import ast
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
    #: **variant root 配下のまま残さず、ASCII 側へ固定する env root。**
    #:
    #: worker は models / cache / resources / %TEMP% / HF_HOME を**すべて** variant
    #: root へ向ける (`runner.py`)。そのままだと**複数の境界を同時に非 ASCII にする**
    #: ことになり、失敗したときにどれが原因か分からない。**この行が測りたい 1 つ**
    #: 以外をここに列挙して ASCII へ固定する。
    #:
    #: 実例 (#413): whispers2t の一時 wav は `cache_root` にあり `%TEMP%` ではない。
    #: 両方を非 ASCII にしていたため、**PyTorch の CUDA Jiterator kernel cache**
    #: (`%TEMP%` が既定の置き場所) の破綻を utterance_wav の失敗として記録しかけた
    #: (-> **#422**)。
    #:
    #: 値は env 変数名: ``"TEMP"`` (TMP / TMPDIR も連動) / ``"LIVECAP_CORE_CACHE_DIR"``
    #: / ``"LIVECAP_RESOURCE_ROOT"`` / ``"HF_HOME"``。
    ascii_pinned_roots: tuple[str, ...] = ()
    #: **この行だけは代表 variant 1 件で済ませない。** slow tier は既定で
    #: cjk_kana しか回さないが、``ユーザー`` は cp932 の内側なので、consumer が
    #: narrow path でも日本語 Windows なら通ってしまう。ACP の外側 (outside_acp)
    #: まで要求する行はここに列挙する。**足りなければ test は fail する** (skip しない)。
    required_variants: tuple[str, ...] = ()
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
    # production code がこの境界を包んでいる ``livecap_cli.paths`` の API 名
    # (``ascii_safe_temp_environment`` / ``ascii_safe_workspace``)。包んでいなければ None。
    # **実行時ログの ``boundary=`` は、この行の ``boundary_id`` と同一文字列である。**
    # 境界一覧の SSOT を registry に一本化するための field で、
    # ``tests/core/paths/test_download_migration.py`` はここから期待値を導出する
    # (#375 PR 3 のレビュー指摘 2 — 一覧が registry とテストの 2 箇所に分裂していた)。
    staging_api: str | None = None
    # 上記 API へ渡している ``purpose``。**両 API とも purpose を取る**
    # (``ascii_safe_workspace`` の既定は ``"runtime"``)。テスト側に値を持たせると
    # SSOT が再び分裂するので registry に置く — #379 が既定以外の purpose を使っても、
    # registry を更新するだけで検査が追随する (#375 PR 3 の再レビュー指摘 1)。
    staging_purpose: str | None = None


# --- 3.1 エンジンモデルロード -------------------------------------------------

_ENGINE_LOAD: tuple[BoundarySpec, ...] = (
    BoundarySpec(
        boundary_id="engine.reazonspeech.sherpa_from_transducer",
        section=Section.ENGINE_LOAD,
        callsite_file="livecap_cli/engines/reazonspeech_engine.py",
        callsite_symbol="sherpa_onnx.OfflineRecognizer.from_transducer(",
        path_desc="tokens.txt / encoder / decoder / joiner の絶対 path (#409 以降は resolve_model_files() が解決する)",
        receiver="sherpa-onnx (native, 1.13.6+ は wide path)",
        wide_path_support="対応 (1.13.6+)",
        candidate_method=Method.WIDE_PATH,
        verified_method=Method.WIDE_PATH,
        rationale=(
            "**上流が修正済み (#377)。** 1.12.39 では SymbolTable が narrow path の "
            "std::ifstream で tokens.txt を開いており、Windows の非 ASCII パスで**空のまま"
            "黙って構築**されていた。上流 PR #3255 で SymbolTable が OpenInputFile() を"
            "使い、Windows では ToWideString() 経由で開くようになった。1.12.39 / 1.13.6 の "
            "A/B を tokens-only と全 dir 非 ASCII の両方で実測し、後者で解消を確認している。"
            "OfflineModelConfig に tokens_buf が無い (方式①不可) のは変わらないが、②が"
            "成立するので staging は不要。"
        ),
        probe_id="sherpa.from_transducer.real",
        tier="real_model",
        expected_verdict="pass",
        failure_visibility=(
            "**1.12.39 では黙っていた** — ロードは成功し decode が全件 IndexError、さらに"
            "壊れた recognizer が ModelMemoryCache.set(..., strong=True) でプロセス寿命の間"
            "キャッシュされた。1.13.6 で解消。**壊れた recognizer を保存させない責務は #392** "
            "(post-load health check と保存ゲート) が持つ — sherpa-onnx のバージョンに"
            "依存しないため。#409 (cache key v2) は identity だけを扱い、健全性は判定しない。"
        ),
        followup_issue="#392",
    ),
    BoundarySpec(
        boundary_id="engine.reazonspeech.hotwords_file",
        section=Section.ENGINE_LOAD,
        callsite_file="livecap_cli/engines/reazonspeech_engine.py",
        callsite_symbol="decoding_method=self.decoding_method",
        path_desc="hotwords ファイル (#361 で追加予定。現時点では未実装)",
        receiver="sherpa-onnx (native, 1.13.6+ は wide path の見込み)",
        wide_path_support="対応の見込み (source-level のみ)",
        candidate_method=Method.WIDE_PATH,
        rationale=(
            "OfflineRecognizerConfig に hotwords_file はあるが hotwords_buf は無い "
            "(1.12.39 で実測) → 方式①不可。**上流実装では hotwords 経路も tokens と同じ "
            "OpenInputFile() を通る**ため、#377 で確認した wide-path 修正が及ぶ見込み。"
            "ただし #361 が未実装のため**呼び出し箇所が存在せず runtime 未確認**であり、"
            "source-level の見立てに留まる。"
        ),
        evidence_kind="source_check",
        probe_id=None,
        tier="none",
        granularity="file",
        failure_visibility="未実装。#361 実装時に本行を runtime 実測へ格上げすること。",
        unmeasured_reason=(
            "#361 未実装のため呼び出し箇所がまだ存在しない。**runtime 確認は #361 で実施する** "
            "— #377 の wide-path 修正が hotwords にも及ぶかは source-level でしか見ていない。"
        ),
        followup_issue="#361",
    ),
    BoundarySpec(
        boundary_id="engine.parakeet.nemo_restore_from",
        section=Section.ENGINE_LOAD,
        callsite_file="livecap_cli/engines/parakeet_engine.py",
        # **boundary リテラルそのものを symbol にする。** #379 で restore_from() 自体は
        # nemo_utils.restore_nemo_model() へ移ったが、**境界を決めているのは engine** であり、
        # ascii_safe_temp_environment(boundary=...) はここに残る。
        callsite_symbol='boundary="engine.parakeet.nemo_restore_from"',
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
            "**#379 で ASCII 保証済み** — ascii_safe_temp_environment("
            "boundary=\"engine.parakeet.nemo_restore_from\", purpose=\"nemo-restore\") で包み、"
            "NeMo の一次エラーを app log へ転送するようにした。"
            "対策前は**黙る / すり替わる** — 元例外 (SentencePiece が展開先の tokenizer.model を"
            "開けない) が捕捉され、抽象クラスの二次例外 "
            "TypeError('Can't instantiate abstract class ASRModel ...') に置換されていた。"
            "**#379 で実モデル再現済み**。"
            "なお `check_nemo_availability()` の `NEMO_AVAILABLE=False` キャッシュは"
            "**別事象**である — 同関数は `restore_from` より前の import 成功時点で `True` を"
            "キャッシュするので、本行の失敗経路では触られない。False になるのは import 自体が"
            "失敗したとき (実例: lightning 2.6 が NeptuneLogger を削除して NeMo が import "
            "できなくなったケース。#379 の CI で観測) であり、非 ASCII %TEMP% とは無関係。"
        ),
        followup_issue="#379",
        staging_api="ascii_safe_temp_environment",
        staging_purpose="nemo-restore",
    ),
    BoundarySpec(
        boundary_id="engine.canary.nemo_restore_from",
        section=Section.ENGINE_LOAD,
        callsite_file="livecap_cli/engines/canary_engine.py",
        callsite_symbol='boundary="engine.canary.nemo_restore_from"',
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
        failure_visibility=(
            "**#379 で ASCII 保証済み** (parakeet と同一機構)。対策前は**黙る / すり替わる**。"
        ),
        followup_issue="#379",
        staging_api="ascii_safe_temp_environment",
        staging_purpose="nemo-restore",
    ),
    BoundarySpec(
        boundary_id="engine.parakeet.from_pretrained",
        section=Section.ENGINE_LOAD,
        callsite_file="livecap_cli/engines/parakeet_engine.py",
        callsite_symbol="nemo_asr.models.ASRModel.from_pretrained(",
        path_desc=(
            "初回ダウンロード時の ``from_pretrained`` — **NeMo が内部で "
            "``restore_from`` を呼び、.nemo を自前で %TEMP% へ展開する**"
        ),
        receiver="NeMo (download → tar 展開) → sentencepiece (native, narrow path)",
        wide_path_support="NeMo 内部の %TEMP% 展開先が**非対応**",
        candidate_method=Method.STAGING,
        evidence_kind="source_check",
        rationale=(
            "``engine.nemo.untar_temp`` と**同一機構の 2 つめの callsite**である。"
            "``nemo_restore_from`` (ロード経路、#379) と違い、こちらは**ダウンロード経路**で、"
            "旧 unicode_safe_download_directory() が包んでいた 5 箇所のうちの 1 つだった。"
            "旧 helper は %TEMP% を cache_root へ移すだけで ASCII 保証が無かったため、"
            "**#375 PR 3 で ascii_safe_temp_environment へ移して初めて保証が付いた**。"
        ),
        measurement_caveat=(
            "本 callsite 単体は未実測。ただし**機構そのもの** (NeMo 内部の %TEMP% 展開) は "
            "engine.nemo.untar_temp が heavy tier で fail_silent を実測済みで、"
            "from_pretrained はその restore_from を内部で呼ぶ。"
        ),
        tier="heavy",
        granularity="%TEMP%",
        failure_visibility=(
            "**#375 PR 3 で ASCII 保証済み** — ascii_safe_temp_environment("
            "boundary=\"engine.parakeet.from_pretrained\", purpose=\"download\") で包んでいる。"
            "ASCII root を確保できなければ AsciiStagingUnavailableError で**落ちる** "
            "(黙って非 ASCII へ移設しない)。"
        ),
        unmeasured_reason=(
            "実ダウンロードを伴う heavy tier。機構は engine.nemo.untar_temp で実測済みのため、"
            "本 callsite の再実測は費用に見合わないと判断した。"
        ),
        staging_api="ascii_safe_temp_environment",
        staging_purpose="download",
    ),
    BoundarySpec(
        boundary_id="engine.canary.from_pretrained",
        section=Section.ENGINE_LOAD,
        callsite_file="livecap_cli/engines/canary_engine.py",
        callsite_symbol="nemo_asr.models.EncDecMultiTaskModel.from_pretrained(",
        path_desc=(
            "初回ダウンロード時の ``from_pretrained`` — **NeMo が内部で "
            "``restore_from`` を呼び、.nemo を自前で %TEMP% へ展開する**"
        ),
        receiver="NeMo (download → tar 展開) → sentencepiece (native, narrow path)",
        wide_path_support="NeMo 内部の %TEMP% 展開先が**非対応**",
        candidate_method=Method.STAGING,
        evidence_kind="source_check",
        rationale="parakeet と同一機構・同一経路 (engine.parakeet.from_pretrained 参照)。",
        measurement_caveat=(
            "本 callsite 単体は未実測。機構は engine.nemo.untar_temp が実測済み。"
        ),
        tier="heavy",
        granularity="%TEMP%",
        failure_visibility=(
            "**#375 PR 3 で ASCII 保証済み** — ascii_safe_temp_environment("
            "boundary=\"engine.canary.from_pretrained\", purpose=\"download\") で包んでいる。"
        ),
        unmeasured_reason=(
            "実ダウンロードを伴う heavy tier。機構は engine.nemo.untar_temp で実測済み。"
        ),
        staging_api="ascii_safe_temp_environment",
        staging_purpose="download",
    ),
    BoundarySpec(
        boundary_id="engine.nemo.untar_temp",
        section=Section.ENGINE_LOAD,
        # #379 で restore_from() の呼び出しが共通 helper へ移った。本行は
        # **NeMo 内部の機構**を指すので、engine ではなく helper を callsite にする。
        callsite_file="livecap_cli/engines/nemo_utils.py",
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
        # **.nemo の path だけを非 ASCII にする。** %TEMP% も同時に非 ASCII だと
        # engine.nemo.untar_temp と主因を切り分けられない。
        ascii_pinned_roots=("TEMP",),
        section=Section.ENGINE_LOAD,
        callsite_file="livecap_cli/engines/nemo_utils.py",
        callsite_symbol="map_location=map_location,",
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
        path_desc="HF repo id + HF_HOME (ascii_safe_temp_environment + huggingface_cache 内)",
        receiver="qwen_asr → transformers → HF snapshot + safetensors + tokenizer",
        wide_path_support="要実測",
        candidate_method=Method.WIDE_PATH,
        rationale=(
            "**初回ロード時に HF から落ちてくる**ので、ここが download 境界そのものである "
            "(_download_model はマーカーを置くだけ)。#375 PR 3 で "
            "ascii_safe_temp_environment(boundary=\"engine.qwen3asr.from_pretrained\") へ移した — "
            "旧 unicode_safe_download_directory() は %TEMP% を cache_root へ移すだけで、"
            "その cache_root 自体が appdirs 既定では**ユーザー名を含む**ため、"
            "**包んでも ASCII 安全にはならなかった**。"
        ),
        evidence_kind="source_check",
        probe_id="qwen3asr.from_pretrained",
        tier="real_model",
        granularity="dir",
        failure_visibility=(
            "**#375 PR 3 で ASCII 保証済み** — ascii_safe_temp_environment("
            "boundary=\"engine.qwen3asr.from_pretrained\", purpose=\"download\") で包んでいる。"
            "**本行を包んでいるのは「② が実測で確定していない」からである** — ReazonSpeech の "
            "download 経路は ② が確定しているので #375 PR 3 では包み直さなかった。"
            "**#387 で ② が実測で確定したら、本行の wrapper も外すこと** "
            "(§6.10「② で足りる境界に ③ を持ち込まない」)。"
        ),
        unmeasured_reason=(
            "**`qwen_asr` は導入済みである** — #413 PR C で `engines-qwen3asr` extra を"
            "入れ、CI の GPU job にも追加した (NeMo と競合しないことを実測済み)。"
            "残っているのは測定側であり、(1) `qwen3asr.from_pretrained` probe が import "
            "可否を見るだけの stub であること、(2) `_REAL_MODEL_SOURCES` に source 定義が"
            "無く tier 側で先に skip されること、の 2 点である。"
            "**この行は初回ダウンロード境界なので**、real_model tier の「ネットワークを"
            "使わない」契約とどう両立させるかを #387 で決める必要がある。"
        ),
        followup_issue="#387",
        staging_api="ascii_safe_temp_environment",
        staging_purpose="download",
    ),
    BoundarySpec(
        boundary_id="engine.reazonspeech.sherpa_narrow_path_signature",
        section=Section.ENGINE_LOAD,
        callsite_file="livecap_cli/engines/reazonspeech_engine.py",
        # #409 で path の組み立てが reazonspeech_cache.resolve_model_files() へ移った。
        # 渡す path の意味 (tokens.txt の絶対 path) は変わらない。
        callsite_symbol='tokens=str(model_files[',
        path_desc="不正な ONNX + tokens.txt を ASCII / 非 ASCII に置き、エラー署名を比較",
        receiver="sherpa-onnx (native, 1.13.6+ は wide path)",
        wide_path_support="対応 (1.13.6+)",
        candidate_method=Method.WIDE_PATH,
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
        failure_visibility=(
            "**この行の pass は「sherpa-onnx が安全」を意味しない。** 不正な ONNX は "
            "tokens.txt より先に検証されるため、本プローブが到達できるのは ONNX 層までで "
            "(ASCII / 非 ASCII のどちらも同じ parse 失敗署名になった)、既知 NG の本体である "
            "tokens.txt の SymbolTable 誤読には届かない。そちらは real_model tier で "
            "fail_silent を再現している。"
        ),
        # **#377 は version bump で close したが、本行の計測ギャップは解消していない。**
        # closed issue を指したままだと孤児化する (test_unverified_rows_have_a_tracking_home は
        # followup_issue が空でなければ通るので検出できない) — #387 へ付け替えた。
        followup_issue="#387",
    ),
    BoundarySpec(
        boundary_id="lib.onnxruntime.inference_session",
        section=Section.ENGINE_LOAD,
        callsite_file="livecap_cli/engines/reazonspeech_engine.py",
        callsite_symbol='encoder=str(model_files[',
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


#: consumer を実モデルで測る 5 engine の tier。**qwen3asr も #413 PR C で加わった** —
#: `qwen_asr` は NeMo と競合せず (25 パッケージの純粋な追加)、隔離環境は要らなかった。
_UTTERANCE_WAV_TIERS: dict[str, str] = {
    "parakeet": "heavy",      # NeMo。_HEAVY_SOURCES が .nemo を boundary_id 単位で引く
    "canary": "heavy",
    "whispers2t": "real_model",
    "voxtral": "real_model",
    # #413 PR C: consumer probe を得たので producer-only の分岐から移した。
    "qwen3asr": "real_model",
}


def _utterance_wav_row(
    engine: str, file: str, symbol: str, anchored: str
) -> BoundarySpec:
    tier = _UTTERANCE_WAV_TIERS[engine]
    return BoundarySpec(
        boundary_id=f"engine.{engine}.utterance_wav",
        section=Section.RUNTIME_TEMP,
        callsite_file=file,
        callsite_symbol=symbol,
        path_desc=f"発話ごとの一時 wav ({anchored})",
        receiver="soundfile (書き込み) → ネイティブ ASR (読み込み)",
        wide_path_support="書き込みは対応 (sf_wchar_open) / 読み込み側は engine 依存",
        # **実測で ③staging から ②wide-path へ変わった行である** (#413 PR B)。
        candidate_method=Method.WIDE_PATH,
        rationale=(
            "**書き込みはバグではない** — soundfile は Windows で sf_wchar_open を使う。"
            "問題があるとすれば書いた path をネイティブ ASR に渡す側だが、**実測では "
            "consumer も非 ASCII path を正しく扱えた** — `cjk_kana` と `outside_acp` の"
            "両方で ASCII control と転写が一致する。\n"
            "**したがって staging を追加してはならない。** #378 §6.10 の「② で足りる境界に "
            "③ を持ち込まない」に該当する。当初 (#375 PR 4) は 5 consumer すべてを "
            "ascii_safe_workspace() へ移す計画だったが、実測が方針を覆した。"
            "なお ascii_safe_temp_environment は発話ごとにプロセスグローバル状態を"
            "書き換えるので、仮に staging が要る場合でも**使ってはならない** (#386)。"
        ),
        probe_id=f"asr.utterance_wav.{engine}",
        tier=tier,
        # **この境界が測りたい 1 つ以外を ASCII へ固定する。**
        #   parakeet / canary : 一時 wav は %TEMP% -> TEMP だけを変数にする
        #   whispers2t / voxtral : 一時 wav は cache_root -> それだけを変数にする
        ascii_pinned_roots=(
            ("TEMP", "LIVECAP_RESOURCE_ROOT", "HF_HOME")
            if engine in {"whispers2t", "voxtral"}
            else ("LIVECAP_CORE_CACHE_DIR", "LIVECAP_RESOURCE_ROOT", "HF_HOME")
        ),
        granularity="dir",
        # **consumer を実モデルで通す probe になった** (#413 PR A)。
        covers_boundary=True,
        # **cjk_kana だけでは足りない。** `ユーザー` は cp932 の内側なので、consumer が
        # narrow path (ACP 変換) でも日本語 Windows なら通ってしまう。ACP の外側まで要求する。
        required_variants=("cjk_kana", "outside_acp"),
        expected_verdict="pass",
        measurement_caveat=(
            "**モデルは ASCII 側に固定し、一時 wav の置き場所だけを非 ASCII にした** "
            "計測である。両方を同時に非 ASCII にすると、失敗したとき「モデルの path が"
            "原因」か「一時 wav の path が原因」かを切り分けられない "
            "(engine.nemo.restore_path_only / engine.nemo.untar_temp と同じ分け方)。"
            + (
                " whispers2t / voxtral は一時 wav が cache_root にあるため "
                "**%TEMP% も ASCII へ固定**している — 固定しないと**無関係な"
                "ライブラリの %TEMP% 利用**が原因でも同じ verdict になる。"
                "実際 **PyTorch の CUDA Jiterator kernel cache** が %TEMP% を既定の"
                "置き場所にしており、ACP 外だと CUDA 上の複素数演算が "
                "UnicodeDecodeError で落ちる (**#422**)。whispers2t の前処理が "
                "torch.fft.rfft(...).abs() を通るため最初に踏んだが、**utterance_wav "
                "とは別の境界**である。"
                if engine in {"whispers2t", "voxtral"}
                else ""
            )
            + (
                " **qwen3asr は auto-detect 経路でのみこの境界に到達する** — 一時 wav を"
                "書くのは `_transcribe_via_wrapper_fallback()` だけで、そこへ入るのは "
                "`_asr_language is None` のときに限られる。言語を指定する呼び出しは "
                "`_transcribe_with_scores()` へ行き**一時 wav を書かない**。probe が"
                "言語を渡さないのはそのためである (他の 4 engine とは逆)。"
                "また重みは models root ではなく `huggingface_hub` が実際に使う "
                "**HF hub cache** (`huggingface_hub.constants.HF_HUB_CACHE`) にあり、"
                "models root にあるのは 38 バイトの marker だけ"
                "なので、probe は snapshot の実在まで確かめたうえで `HF_HUB_OFFLINE=1` を"
                "課す。**場所を当てるのではなくネットワークへ出たら落ちるようにする** — "
                "`ModelManager.huggingface_cache()` は実行時に `HF_HOME` を書き換えるが、"
                "`huggingface_hub` は import 時に cache path を確定するので効かない (実測)。"
                if engine == "qwen3asr"
                else ""
            )
        ),
        # **実測で確定** (#413 PR B / PR C)。証拠は
        # benchmark_results/nonascii/2026-08-31/results.json — clean tree から全 tier を
        # 1 セッションで生成し、**5 engine とも** `cjk_kana` / `outside_acp` の両方で
        # pass した。
        #
        # **この行の probe は production 経路 (EngineFactory -> load_model ->
        # transcribe) を通る。** raw 境界を直接叩く nemo.restore_from 系とは測って
        # いるものが違い、`pass` = 「境界そのものが健全」を意味する (緩和が効いて
        # いることではない)。だから ②wide-path と整合する。
        #
        # **5 engine すべてが実測で確定した** (#413 PR C で qwen3asr が加わった)。
        # 当初 (#375 PR 4) は 5 consumer すべてを ascii_safe_workspace() へ移す計画
        # だったが、**実測が 5/5 で ②wide-path を示した**ため 1 つも staging しない。
        verified_method=Method.WIDE_PATH,
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
    BoundarySpec(
        boundary_id="framework.pytorch.cuda_jiterator_kernel_cache",
        section=Section.RUNTIME_TEMP,
        callsite_file="livecap_cli/runtime/pytorch.py",
        callsite_symbol='ENV_KERNEL_CACHE_PATH = "PYTORCH_KERNEL_CACHE_PATH"',
        path_desc=(
            "PyTorch が nvrtc 生成カーネルを置く先 "
            "(PYTORCH_KERNEL_CACHE_PATH → 既定は %TEMP%\\torch\\kernels)"
        ),
        receiver="PyTorch (native, aten/src/ATen/native/cuda/jit_utils.cpp)",
        wide_path_support="非対応 (std::string + narrow CRT/file API。上流 main も同じ)",
        candidate_method=Method.FAIL_FAST,
        rationale=(
            "**engine 固有ではなく framework 単位の境界である。** #413 の作業中に "
            "WhisperS2T で最初に踏んだが、最小再現の結果 `torch` だけで再現し、"
            "**モデルを一切ロードせずに** CUDA 上の複素数 `abs()` が "
            "`UnicodeDecodeError` になる。`%TEMP%` を ASCII にしても "
            "`PYTORCH_KERNEL_CACHE_PATH` を非 ASCII にすれば同じ失敗が出るので、"
            "**`%TEMP%` は原因ではなく既定値**であり境界はキャッシュ先の path である。\n"
            "**方式が ④ になる理由**: ①buffer に相当する API が無く (置き場所は path で"
            "しか指定できない)、②wide-path は上流が narrow のまま成立せず、③staging も"
            "成立しない — PyTorch はキャッシュ先を関数内 static として保持するので、"
            "`ascii_safe_temp_environment()` のようにスコープを抜けて戻す機構と組むと"
            "**握っている path と実体の寿命が一致しなくなる** (#386 と同型)。"
            "したがって**既定では境界そのものを通さず** "
            "(`USE_PYTORCH_KERNEL_CACHE=0`)、明示された非 ASCII path と未知の値は "
            "fail fast にする。\n"
            "**無効化の代償が無い**ことは実測で確かめた: PyTorch 2.9.1 の Windows "
            "書き込み経路は `<name>_tmp_<pid>` から最終名への rename を行わず、"
            "ルックアップは最終名で行われるため **cache が populate されない** "
            "(`%TEMP%\\torch\\kernels` に 75 ファイル / 最終名 0 / 実カーネル 2 種)。"
            "外部で pre-populate された cache はヒットする (98.4 ms → 20.2 ms) ので、"
            "**明示指定は尊重する**。"
        ),
        measurement_caveat=(
            "probe が変えるのは `%TEMP%` だけで、cache / resources / HF_HOME は ASCII へ"
            "固定する。**再評価 trigger**: PyTorch を bump したら 2 プロセス判定 "
            "(空 cache に最終名が書かれ、次プロセスが新しい `_tmp_` を作らない) を"
            "やり直すこと。成立したら永続 ASCII cache root の是非を再検討する — "
            "`tests/integration/runtime/test_pytorch_kernel_cache.py` が固定している。"
        ),
        probe_id="framework.pytorch.jiterator_cache",
        tier="gpu",
        granularity="dir",
        # **`cjk_kana` だけでは再現しない。** `ユーザー` は cp932 の内側なので、
        # 日本語 Windows では narrow path でも通ってしまう。ACP の外側まで要求する。
        required_variants=("cjk_kana", "outside_acp"),
        # **変数にするのは `%TEMP%` だけ。** 他を非 ASCII のままにすると、失敗した
        # ときにどの境界が原因か分からない (#413 で実際に誤帰属しかけた形)。
        ascii_pinned_roots=("LIVECAP_CORE_CACHE_DIR", "LIVECAP_RESOURCE_ROOT", "HF_HOME"),
        expected_verdict="pass",
        failure_visibility=(
            "**診断上 fail_silent。** 例外は送出されるが `error_mentions_path=False` で、"
            "テンソル演算が `UnicodeDecodeError` を投げるという因果も読めない "
            "(C++ 側のメッセージが ANSI で返り UTF-8 復号に失敗している形)。"
            "`cjk_kana` では再現せず ACP の外側でのみ壊れるため、日本語 Windows での"
            "素朴な確認では見逃す。"
        ),
        followup_issue="#425",
        unmeasured_reason=(
            "**現行の証拠モデルでは、この行の緩和策を表現できない** (#425)。実測自体は"
            "済んでいる (2026-08-31 の証拠 JSON: `cjk_kana` / `outside_acp` とも pass)。\n"
            "問題は `Method` が「境界の能力」(①②) と「production の緩和」(③④) を"
            "混在させている点にある。#422 の実装は**複合戦略** — 既定では境界を通さず "
            "(`USE_PYTORCH_KERNEL_CACHE=0`)、明示 opt-in 時のみ path を検証して pin し、"
            "非 ASCII と未知の値は fail-fast する。どの値も正しくならない:\n"
            "- ④fail-fast: probe は**緩和後の production 経路**を測るので verdict は "
            "  `pass` であり、`test_verified_rows_match_committed_evidence` が"
            "  「fail-fast と主張しているが実測は全て pass」として弾く\n"
            "- ②wide-path: **嘘になる**。上流 PyTorch は narrow のままで、我々は境界を"
            "  回避しただけである\n"
            "raw 側の証拠 (ACP 外で壊れること) は "
            "`tests/integration/runtime/test_pytorch_kernel_cache.py` が CI ゲート付きで"
            "固定しているが、**証拠 JSON の外**にある。両者を関連付ける表現を #425 で決める。"
        ),
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
            "ASCII staging を一切通らずシステム %TEMP% を使い、さらにユーザーのファイル名 "
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
        callsite_file="livecap_cli/resources/configuration.py",
        callsite_symbol="ENV_MODELS_DIR",
        path_desc="models_root / cache_root (env var または appdirs 既定)",
        receiver="CPython pathlib → 後段の全境界",
        wide_path_support="対応 (CPython)",
        candidate_method=Method.WIDE_PATH,
        verified_method=Method.WIDE_PATH,
        rationale=(
            "根の注入機構そのもの。既定値 appdirs.user_cache_dir('LiveCap','PineLab') は"
            "**ユーザー名を含む**ため ASCII 保証がない。ただし **#380 の確定方針は "
            "『canonical root を黙って ASCII 領域へ置換しない』** — 既存ユーザーに"
            "モデル再ダウンロードを強いるうえ、%TEMP% など置換しきれない経路が残るため。"
            "#375 が提供するのは (1) 公開 configuration / readback API と "
            "(2) narrow-path consumer を**境界で** staging する基盤であって、"
            "canonical root の ASCII 化ではない。CPython 経由なので本行自体は ② で、"
            "ハーネスの前提条件テスト (env 注入が効いているか) も兼ねる。"
        ),
        probe_id="model_manager.roots",
        tier="cheap",
        granularity="dir",
        followup_issue="#375",
    ),
    BoundarySpec(
        boundary_id="resources.resource_locator.env_root",
        section=Section.OUTPUT_CLI,
        callsite_file="livecap_cli/resources/configuration.py",
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
        callsite_file="livecap_cli/resources/configuration.py",
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


# --- ASCII staging の実使用スキャン --------------------------------------------

#: production code が呼びうる ``livecap_cli.paths`` の境界 API。
STAGING_APIS: tuple[str, ...] = ("ascii_safe_temp_environment", "ascii_safe_workspace")

#: スキャン対象から外す module。**API の定義側**なので、ここの ``def`` や docstring は
#: 呼び出しではない (AST 上も ``Call`` にならないが、意図を明示するために除外する)。
_SCAN_EXCLUDE = ("livecap_cli/paths/",)


@dataclass(frozen=True)
class StagingCall:
    """production code に実在する境界 API の呼び出し 1 件。"""

    callsite_file: str          # repo 相対
    api: str
    boundary: str | None        # 定数でなければ None
    purpose: str | None         # 定数でなければ / 省略されていれば None
    lineno: int

    def key(self) -> tuple[str, str, str | None, str | None]:
        return (self.callsite_file, self.api, self.boundary, self.purpose)


def _constant_kwarg(call: ast.Call, name: str) -> str | None:
    for keyword in call.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            return keyword.value.value
    return None


def evidence_rows_for(spec: BoundarySpec, results: list[dict]) -> list[dict]:
    """``spec`` の証拠として数えてよい実測レコードだけを返す。

    **``boundary_id`` だけで引いてはならない。** 同じ境界を別の probe で測り直すと、
    **古い probe の結果が新しい主張の証拠として通ってしまう**。実例 (#413):
    ``engine.*.utterance_wav`` は producer 側だけを測る ``tempfile.named_temporary_wav``
    の ``pass`` を持っており、consumer を一度も通していないのに
    ``verified_method=WIDE_PATH`` を名乗れる状態だった。

    **検査 (`test_registry.py`) と棚卸し表 (`report.py`) が同じ規則を使う**ように、
    照合はここ 1 箇所に置く。片方だけ直すと表が嘘をつく。
    """
    rows = [r for r in results if r.get("boundary_id") == spec.boundary_id]
    if spec.probe_id:
        rows = [r for r in rows if r.get("probe_id") == spec.probe_id]
    if spec.tier:
        rows = [r for r in rows if r.get("tier") == spec.tier]
    return rows


def scan_staging_calls(package_root: Path | None = None) -> list[StagingCall]:
    """``livecap_cli`` を AST で走査し、境界 API の**実使用**を列挙する。

    registry → code の一方向検査だけでは、**registry に無いファイルへ新しい
    ``ascii_safe_*`` 呼び出しを足しても検査対象にならず緑のまま**になる
    (#375 PR 3 の再レビュー指摘 1)。``test_registry.py`` はこの結果と
    ``staging_api`` を持つ行を**双方向で突き合わせる**。

    ``boundary`` / ``purpose`` が定数でない呼び出しは ``None`` として返す
    (registry と突き合わせられないので、検査側が失敗として扱う)。
    """
    root = package_root or (REPO_ROOT / "livecap_cli")
    found: list[StagingCall] = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if any(rel.startswith(prefix) for prefix in _SCAN_EXCLUDE):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            api = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if api not in STAGING_APIS:
                continue
            found.append(
                StagingCall(
                    callsite_file=rel,
                    api=api,
                    boundary=_constant_kwarg(node, "boundary"),
                    purpose=_constant_kwarg(node, "purpose"),
                    lineno=node.lineno,
                )
            )
    return found


def registered_staging_calls() -> list[StagingCall]:
    """``staging_api`` を持つ registry 行を :class:`StagingCall` として返す。"""
    return [
        StagingCall(
            callsite_file=spec.callsite_file,
            api=spec.staging_api,
            boundary=spec.boundary_id,
            purpose=spec.staging_purpose,
            lineno=0,
        )
        for spec in BOUNDARIES
        if spec.staging_api
    ]


__all__ = [
    "BOUNDARIES",
    "evidence_rows_for",
    "BOUNDARIES_BY_ID",
    "REPO_ROOT",
    "SECTION_ORDER",
    "STAGING_APIS",
    "BoundarySpec",
    "Method",
    "Section",
    "StagingCall",
    "callsite_label",
    "registered_staging_calls",
    "resolve_callsite_line",
    "scan_staging_calls",
]
