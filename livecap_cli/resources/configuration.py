"""Resource root の解決と、ホストへ返す immutable な snapshot (Issue #375)。

なぜ manager から切り離すのか
----------------------------
以前は 3 つの manager が**それぞれ constructor で env を読み**、独立した
singleton として生成されていた。その結果 ``FFmpegManager`` が使う cache root と
``get_model_manager()`` の cache root が**別物になり得る**状態で、ホストが
「設定したのに効かない」を観測する手段も無かった。

解決を 1 箇所に集め、結果を frozen な :class:`ResourceConfiguration` として
公開する。manager は解決済みの値を**注入される**だけになる。

filesystem に触れる境界
----------------------
``resolve_configuration(..., enforce=False)`` は **filesystem を一切変更しない**。
``get_resource_configuration()`` の preview がこれを使うため、**参照しただけで
ディレクトリができる**ことがあってはならない (Issue #375 AC)。

検証と作成を行うのは ``enforce=True`` (= freeze する経路) のときだけで、しかも
**明示指定された root に限る**。既定 root は graph 構築時に作られる。
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Optional, Sequence, Tuple

from .errors import AsciiStagingUnavailableError, ResourceConfigurationError

try:
    from appdirs import user_cache_dir
except ImportError:  # pragma: no cover - appdirs is a dependency
    user_cache_dir = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

__all__ = [
    "ENV_MODELS_DIR",
    "ENV_CACHE_DIR",
    "ENV_RESOURCE_ROOT",
    "ENV_ASCII_STAGING_DIR",
    "PACKAGE_FALLBACK_KEYS",
    "STAGING_ROOT_MAX_LEN",
    "Source",
    "ConfiguredPath",
    "OverriddenEnv",
    "RootResolution",
    "ResourceSearchResolution",
    "StagingPolicy",
    "StagingRootStatus",
    "ResourceConfiguration",
    "ResourceRequest",
    "normalize_path",
    "resolve_configuration",
]

ENV_MODELS_DIR = "LIVECAP_CORE_MODELS_DIR"
ENV_CACHE_DIR = "LIVECAP_CORE_CACHE_DIR"
ENV_RESOURCE_ROOT = "LIVECAP_RESOURCE_ROOT"
ENV_ASCII_STAGING_DIR = "LIVECAP_CORE_ASCII_STAGING_DIR"

#: ``ResourceLocator`` が filesystem で見つからなかったときに参照する package。
PACKAGE_FALLBACK_KEYS: Tuple[str, ...] = ("src", "config", "languages", "html", "fonts")

#: staging root に許す長さ (Windows のみ)。
#:
#: MAX_PATH 260 から ``<boundary>/<lease-id>/<filename>`` 用に 100 を予約した値。
#: **PR 2 が lease-id の形式を確定したら締め直すこと** — 現時点では lease-id の
#: 長さが決まっていないため、予約を多めに取っている。
STAGING_ROOT_MAX_LEN = 160

Source = Literal["api", "env", "default", "fallback"]


def normalize_path(value: str | Path) -> Path:
    """``expanduser`` -> ``abspath`` -> ``normpath``。

    **``Path.resolve()`` は使わない。** symlink を追跡してホストが渡した path と
    別の場所を指し始めるうえ、存在しない path に対する挙動が platform ごとに
    違う。ここで欲しいのは「同じ場所を指す正準表記」であって実体の追跡ではない。
    """
    expanded = os.path.expanduser(str(value))
    return Path(os.path.normpath(os.path.abspath(expanded)))


@dataclass(frozen=True, slots=True)
class ConfiguredPath:
    """ホストが渡した生値と、正規化後の値の対。

    ``raw`` を ``compare=False`` にしているのが要点である。再設定の同一性判定は
    **正規化後**で行いたい (``~/models`` と展開済みの絶対 path を別物として弾く
    のは過剰) 一方、readback の ``configured`` はホストが実際に書いた文字列を
    返す必要があるため (Issue #375)。
    """

    normalized: Path
    raw: str = field(compare=False, default="")

    @classmethod
    def of(cls, value: str | Path) -> "ConfiguredPath":
        return cls(normalized=normalize_path(value), raw=str(value))


@dataclass(frozen=True, slots=True)
class OverriddenEnv:
    """API が上書きした env var とその値 (R3)。

    「非 ASCII パス問題を ``LIVECAP_CORE_MODELS_DIR`` で回避しているユーザーの
    ホストが ``data_root`` を渡すと、env が無視されて数 GB の再ダウンロードが
    起きる」を観測可能にするための記録。
    """

    name: str
    value: str


@dataclass(frozen=True, slots=True)
class RootResolution:
    """1 つの書き込み用 root がどう決まったか。

    Attributes:
        configured: **正規化前**の値。API 指定でも、採用された env の生値でも、
            ホストが実際に書いた文字列がそのまま入る。未指定なら ``None``。
        resolved: 正規化後の値。
        source: どこから来たか。
        is_ascii: ``resolved`` が ASCII のみか。ネイティブ境界の可否判断に使う。
        fallback_reason: **root ごと**に保持する。models だけ fallback して cache
            はしない、という状態を表現できる必要があるため。
        overridden_env: R3。
    """

    configured: Optional[Path]
    resolved: Path
    source: Source
    is_ascii: bool
    fallback_reason: Optional[str] = None
    overridden_env: Tuple[OverriddenEnv, ...] = ()


@dataclass(frozen=True, slots=True)
class ResourceSearchResolution:
    """静的 resource の検索順。

    単一の ``resource_root`` は返さない — 実際の解決は**順序付きの複数 root**で
    行われるため、1 つだけ見せると嘘になる。

    ``LIVECAP_RESOURCE_ROOT`` の :class:`OverriddenEnv` はここに載る。書き込み用
    root と違い対応する :class:`RootResolution` が無いので、記録先が他に無い。
    """

    effective_roots: Tuple[Path, ...]
    configured_root: Optional[Path]
    source: Source
    package_fallback_keys: Tuple[str, ...] = PACKAGE_FALLBACK_KEYS
    overridden_env: Tuple[OverriddenEnv, ...] = ()


@dataclass(frozen=True, slots=True)
class StagingPolicy:
    """ASCII staging root の**明示指定**の状態。

    Note:
        ``source is None`` は「明示指定が無い」という意味であって「staging が
        できない」ではない。候補 ladder (ソースと同一ボリューム ->
        ``%ProgramData%`` -> ...) は staging core (PR 2) の責務。
    """

    configured_root: Optional[Path] = None
    source: Optional[Source] = None
    overridden_env: Tuple[OverriddenEnv, ...] = ()


@dataclass(frozen=True, slots=True)
class StagingRootStatus:
    """実際に選ばれた staging root (PR 2 が埋める)。

    staging root は source volume 等によって**遅延・複数**決定され得るため、
    単一の ``get_staging_root()`` は設けず tuple で公開する。
    """

    path: Path
    source_volume: Optional[str]
    mechanism: str
    selected_at: float


@dataclass(frozen=True, slots=True)
class ResourceConfiguration:
    """ホストへ返す immutable な snapshot。

    各インスタンスは immutable で、``get_resource_configuration()`` は呼び出し
    時点の**新しい** snapshot を返してよい (同じインスタンスを返し続ける必要は
    ない)。

    Note:
        ``is_frozen=False`` の preview では **root の利用可能性が未検証**である。
        preview は directory 作成も書き込み probe も行わないため、``resolved``
        が実際に使えるかは freeze 時まで確定しない。
    """

    models: RootResolution
    cache: RootResolution
    resource_search: ResourceSearchResolution
    staging_policy: StagingPolicy = field(default_factory=StagingPolicy)
    staging_roots: Tuple[StagingRootStatus, ...] = ()
    is_frozen: bool = False

    @property
    def models_root(self) -> Path:
        return self.models.resolved

    @property
    def cache_root(self) -> Path:
        return self.cache.resolved


@dataclass(frozen=True, slots=True)
class ResourceRequest:
    """``configure_resources()`` への入力を正規化して保持したもの。

    再設定の可否は**この値の一致**で判定する。resolved path だけを比べると、
    「``data_root`` を渡した」と「``models_dir`` / ``cache_dir`` を個別に渡した」
    が同じ結果になったときに区別できず、ホストの意図が違うのに no-op 成功して
    しまう。
    """

    data_root: Optional[ConfiguredPath] = None
    models_dir: Optional[ConfiguredPath] = None
    cache_dir: Optional[ConfiguredPath] = None
    resource_root: Optional[ConfiguredPath] = None
    extra_resource_roots: Tuple[ConfiguredPath, ...] = ()
    staging_root: Optional[ConfiguredPath] = None

    @classmethod
    def from_arguments(
        cls,
        *,
        data_root: Optional[str | Path] = None,
        models_dir: Optional[str | Path] = None,
        cache_dir: Optional[str | Path] = None,
        resource_root: Optional[str | Path] = None,
        extra_resource_roots: Optional[Sequence[str | Path]] = None,
        staging_root: Optional[str | Path] = None,
    ) -> "ResourceRequest":
        def norm(value: Optional[str | Path]) -> Optional[ConfiguredPath]:
            return None if value is None else ConfiguredPath.of(value)

        return cls(
            data_root=norm(data_root),
            models_dir=norm(models_dir),
            cache_dir=norm(cache_dir),
            resource_root=norm(resource_root),
            extra_resource_roots=tuple(
                ConfiguredPath.of(root) for root in (extra_resource_roots or ())
            ),
            staging_root=norm(staging_root),
        )

    @property
    def is_empty(self) -> bool:
        return self == ResourceRequest()


# ---------------------------------------------------------------------------
# 既定 root
# ---------------------------------------------------------------------------


def _default_root(leaf: str) -> Tuple[Path, Source, Optional[str]]:
    """(path, source, fallback_reason) を返す。

    ``appdirs`` が無い環境では ``~/.livecap`` へ落ちる。**これは fallback なので
    そう記録する** — 「既定どおり」と「既定が使えず落ちた」をホストが区別できる
    必要がある。
    """
    if user_cache_dir is None:  # pragma: no cover - appdirs is a dependency
        return (
            normalize_path(Path.home() / ".livecap" / leaf),
            "fallback",
            "appdirs is unavailable; using ~/.livecap",
        )
    return normalize_path(Path(user_cache_dir("LiveCap", "PineLab")) / leaf), "default", None


def _static_roots() -> Tuple[Path, Path]:
    """(project_root, source_root)。

    ``source_root`` は本 package を含むリポジトリ root、``project_root`` はその
    親。検索順は project -> source で、これは #375 以前からの挙動をそのまま
    引き継ぐ (変えると同梱 resource の解決先が変わる)。
    """
    source_root = Path(__file__).resolve().parents[2]
    return source_root.parent, source_root


# ---------------------------------------------------------------------------
# 検証 (R2)
# ---------------------------------------------------------------------------


def _validate_writable_root(path: Path, label: str) -> None:
    """作成できて書き込める root か。models / cache / data 用。"""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ResourceConfigurationError(
            f"{label} '{path}' cannot be created: {error}"
        ) from error

    # **固定名にしない。** 同名のファイルがあると内容を truncate したうえで削除
    # することになり (symlink ならリンク先まで)、複数プロセスの同時 configure も
    # 同じ probe を奪い合う。mkstemp は本呼び出しが**原子的に所有した**一意な
    # ファイルだけを返すので、消してよいのはそれだけだと保証できる。
    try:
        handle, probe_name = tempfile.mkstemp(dir=path, prefix=".livecap-write-probe-")
    except OSError as error:
        raise ResourceConfigurationError(
            f"{label} '{path}' is not writable: {error}"
        ) from error
    os.close(handle)
    try:
        os.unlink(probe_name)
    except OSError:
        pass


def _validate_readable_dir(path: Path, label: str) -> None:
    """存在する読み取り可能な directory か。resource / extra 用。

    **書き込みは要求しない** — 静的 resource は読むだけであり、read-only な
    インストール先を指すのは正当な使い方だから。
    """
    if not path.is_dir():
        raise ResourceConfigurationError(
            f"{label} '{path}' is not an existing directory"
        )
    try:
        # ``with`` が要る。``next()`` だけだと**中身のあるディレクトリでは
        # イテレータが枯渇せず**、OS のディレクトリハンドルが開いたまま残る
        # (Windows ではそのディレクトリの削除・rename を妨げ得る)。空の
        # ディレクトリでは即枯渇して自動 close されるため、テストからは
        # 見えなかった。
        with os.scandir(path) as entries:
            next(iter(entries), None)
    except OSError as error:
        raise ResourceConfigurationError(
            f"{label} '{path}' is not readable: {error}"
        ) from error


def validate_staging_root(path: Path, origin: str) -> None:
    """staging root の述語 — **ASCII / 長さ / 作成・書き込み可能**。

    Raises:
        AsciiStagingUnavailableError: いずれかを満たさないとき。R2 により候補
            ladder へは落ちない。明示指定が使えないことは「別の場所を勝手に
            使ってよい」という意味ではないため。
    """
    text = str(path)
    if not text.isascii():
        raise AsciiStagingUnavailableError(
            f"staging root from {origin} is not ASCII: '{text}'. "
            "Pick a path without non-ASCII characters."
        )
    if sys.platform == "win32" and len(text) > STAGING_ROOT_MAX_LEN:
        raise AsciiStagingUnavailableError(
            f"staging root from {origin} is too long "
            f"({len(text)} > {STAGING_ROOT_MAX_LEN} characters): '{text}'. "
            "Staged paths must stay inside the Windows MAX_PATH budget."
        )
    try:
        _validate_writable_root(path, f"staging root from {origin}")
    except ResourceConfigurationError as error:
        raise AsciiStagingUnavailableError(str(error)) from error


# ---------------------------------------------------------------------------
# 解決
# ---------------------------------------------------------------------------


def _overridden(env: Mapping[str, str], name: str) -> Tuple[OverriddenEnv, ...]:
    value = env.get(name)
    if not value:
        return ()
    return (OverriddenEnv(name=name, value=value),)


def _warn_override(name: str, env_value: str, api_value: Path) -> None:
    """R3 — API が「実際に設定されている」env を上書きするときは黙って行わない。

    メッセージには **env var 名 / env の値 / 採用された API の値 / 上書きした事実**
    を含める。readback を見ないホストでも起動ログで気づけるようにするため。
    """
    logger.warning(
        "%s is set to '%s' but the host passed an explicit value; "
        "using '%s' and ignoring the environment variable.",
        name,
        env_value,
        api_value,
    )


def _resolve_writable_root(
    *,
    label: str,
    leaf: str,
    api_value: Optional[ConfiguredPath],
    data_root: Optional[ConfiguredPath],
    env_name: str,
    env: Mapping[str, str],
    enforce: bool,
) -> RootResolution:
    """API 個別 > API ``data_root`` 派生 > env > default (R1)。"""
    env_value = env.get(env_name) or None

    configured: Optional[Path]
    resolved: Path
    source: Source
    fallback_reason: Optional[str] = None
    overridden: Tuple[OverriddenEnv, ...] = ()

    if api_value is not None:
        configured, resolved, source = Path(api_value.raw), api_value.normalized, "api"
    elif data_root is not None:
        # data_root から派生するのは models と cache **だけ**。resource 検索 root
        # は派生させない (静的 resource と書き込み用 root は別物であるため)。
        configured = Path(data_root.raw)
        resolved, source = data_root.normalized / leaf, "api"
    elif env_value is not None:
        configured, resolved, source = Path(env_value), normalize_path(env_value), "env"
    else:
        resolved, source, fallback_reason = _default_root(leaf)
        configured = None

    if source == "api" and env_value:
        if enforce:
            _warn_override(env_name, env_value, resolved)
        overridden = _overridden(env, env_name)

    if enforce and source in ("api", "env"):
        # **明示指定のみ**検証する。既定 root は graph 構築時に作られる。
        _validate_writable_root(resolved, label)

    return RootResolution(
        configured=configured,
        resolved=resolved,
        source=source,
        is_ascii=str(resolved).isascii(),
        fallback_reason=fallback_reason,
        overridden_env=overridden,
    )


def _resolve_resource_search(
    *,
    api_root: Optional[ConfiguredPath],
    extra_roots: Sequence[ConfiguredPath],
    env: Mapping[str, str],
    enforce: bool,
) -> ResourceSearchResolution:
    """API 指定の有無で 2 分岐する。**API と env は混在しない。**

    API を指定したのに env root も検索候補に残すと、それは「上書き」ではなく
    「優先 fallback」であり R3 と矛盾する。除外して ``overridden_env`` に記録する。
    """
    env_value = env.get(ENV_RESOURCE_ROOT) or None
    project_root, source_root = _static_roots()
    extras = tuple(root.normalized for root in extra_roots)

    head: Tuple[Path, ...]
    configured: Optional[Path]
    source: Source
    overridden: Tuple[OverriddenEnv, ...] = ()

    if api_root is not None:
        head, configured, source = (api_root.normalized,), Path(api_root.raw), "api"
        if env_value:
            if enforce:
                _warn_override(ENV_RESOURCE_ROOT, env_value, api_root.normalized)
            overridden = _overridden(env, ENV_RESOURCE_ROOT)
    elif env_value is not None:
        head, configured, source = (normalize_path(env_value),), Path(env_value), "env"
    else:
        head, configured, source = (), None, "default"

    if enforce:
        # **採用された先頭 root は、API 由来でも env 由来でも検証する。**
        # env だけ素通しにすると、存在しない LIVECAP_RESOURCE_ROOT を設定しても
        # configure は成功し、resolve() が project/source root へ黙って落ちる —
        # 本 PR が防ごうとしている silent degradation そのものになる (R2)。
        if head:
            _validate_readable_dir(
                head[0],
                "resource root" if api_root is not None else ENV_RESOURCE_ROOT,
            )
        for extra in extras:
            _validate_readable_dir(extra, "extra resource root")

    return ResourceSearchResolution(
        effective_roots=head + (project_root, source_root) + extras,
        configured_root=configured,
        source=source,
        overridden_env=overridden,
    )


def _resolve_staging(
    *,
    api_root: Optional[ConfiguredPath],
    env: Mapping[str, str],
    enforce: bool,
) -> StagingPolicy:
    """API > env。どちらも無ければ未解決 (候補 ladder は PR 2 の責務)。"""
    env_value = env.get(ENV_ASCII_STAGING_DIR) or None

    if api_root is not None:
        overridden: Tuple[OverriddenEnv, ...] = ()
        if env_value:
            if enforce:
                _warn_override(ENV_ASCII_STAGING_DIR, env_value, api_root.normalized)
            overridden = _overridden(env, ENV_ASCII_STAGING_DIR)
        if enforce:
            validate_staging_root(
                api_root.normalized, "configure_resources(staging_root=...)"
            )
        return StagingPolicy(
            configured_root=Path(api_root.raw), source="api", overridden_env=overridden
        )

    if env_value is not None:
        if enforce:
            validate_staging_root(normalize_path(env_value), ENV_ASCII_STAGING_DIR)
        return StagingPolicy(configured_root=Path(env_value), source="env")

    return StagingPolicy()


def resolve_configuration(
    request: ResourceRequest,
    env: Mapping[str, str],
    *,
    enforce: bool,
    frozen: bool,
) -> ResourceConfiguration:
    """入力と env から snapshot を組み立てる。

    Args:
        request: 正規化済みの API 入力。
        env: 使用する環境変数。**freeze 時に固定した写し**を渡すこと。以後の
            ``os.environ`` の変更が resolved 値を動かさないための引数である。
        enforce: freeze 経路かどうか。``True`` のとき**明示指定された root だけ**
            を検証し、使えなければ送出し (R2)、env 上書きの ``WARNING`` を出す
            (R3)。``False`` は preview 用で **filesystem を一切変更せず**、警告も
            出さない (readback のたびに同じ警告が積み上がるため)。
        frozen: 返す snapshot の ``is_frozen``。
    """
    models = _resolve_writable_root(
        label="models root",
        leaf="models",
        api_value=request.models_dir,
        data_root=request.data_root,
        env_name=ENV_MODELS_DIR,
        env=env,
        enforce=enforce,
    )
    cache = _resolve_writable_root(
        label="cache root",
        leaf="cache",
        api_value=request.cache_dir,
        data_root=request.data_root,
        env_name=ENV_CACHE_DIR,
        env=env,
        enforce=enforce,
    )
    return ResourceConfiguration(
        models=models,
        cache=cache,
        resource_search=_resolve_resource_search(
            api_root=request.resource_root,
            extra_roots=request.extra_resource_roots,
            env=env,
            enforce=enforce,
        ),
        staging_policy=_resolve_staging(
            api_root=request.staging_root, env=env, enforce=enforce
        ),
        staging_roots=(),  # PR 2 が runtime status として埋める
        is_frozen=frozen,
    )
