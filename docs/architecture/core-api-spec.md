# LiveCap Core API 仕様書

> **バージョン:** 1.0.0
> **最終更新:** 2025-11-25

## 1. 概要

LiveCap Core は音声認識（ASR）エンジンとファイル文字起こしパイプラインを提供するPythonライブラリです。

### 主な機能

- 複数のASRエンジン（ReazonSpeech, Whisper, Parakeet, Canary）のサポート
- ファイルベースの文字起こしパイプライン
- FFmpeg/モデルキャッシュの自動管理
- 多言語対応（日本語、英語、中国語など15言語以上）

## 2. パッケージ構成

```
livecap_cli/
├── __init__.py              # 公開APIの再エクスポート
├── cli.py                   # CLIエントリーポイント・診断機能
├── i18n.py                  # 国際化ヘルパー
├── languages.py             # 言語定義（Languagesクラス）
├── transcription_types.py   # イベントTypedDict定義
├── config/
│   ├── __init__.py          # 設定エクスポート
│   ├── defaults.py          # デフォルト設定
│   ├── schema.py            # TypedDictスキーマ
│   └── validator.py         # 設定バリデーション
├── resources/
│   ├── __init__.py          # リソースマネージャーエクスポート
│   ├── model_manager.py     # モデルキャッシュ管理
│   ├── ffmpeg_manager.py    # FFmpegバイナリ管理
│   └── resource_locator.py  # リソースパス解決
├── transcription/
│   ├── __init__.py          # 文字起こしエクスポート
│   └── file_pipeline.py     # FileTranscriptionPipeline
└── utils/
    └── __init__.py          # ユーティリティ関数

engines/                     # ASRエンジン実装（別パッケージ）
├── __init__.py              # BaseEngine, EngineFactoryエクスポート
├── base_engine.py           # 抽象基底クラス
├── engine_factory.py        # エンジンファクトリ
├── metadata.py              # エンジンメタデータ定義
├── reazonspeech_engine.py   # ReazonSpeech（日本語）
├── whispers2t_engine.py     # WhisperS2T（多言語）
├── parakeet_engine.py       # NVIDIA Parakeet（英語）
├── canary_engine.py         # NVIDIA Canary（多言語）
└── voxtral_engine.py        # Voxtral（多言語）
```

## 3. 公開API

### 3.1 トップレベルエクスポート (`livecap_cli`)

```python
from livecap_cli import (
    # 言語ユーティリティ
    Languages,

    # ファイル文字起こしパイプライン
    FileTranscriptionPipeline,
    FileTranscriptionProgress,
    FileProcessingResult,
    FileSubtitleSegment,
    FileTranscriptionCancelled,

    # イベントヘルパー
    create_transcription_event,
    create_status_event,
    create_error_event,
    create_translation_request_event,
    create_translation_result_event,
    create_subtitle_event,
    validate_event_dict,
    get_event_type_name,
    normalize_to_event_dict,
    format_event_summary,

    # バリデーション
    ValidationError,
)
```

### 3.2 エンジン設定 (`livecap_cli.engines.metadata`)

> **Note**: Phase 2 で `livecap_cli.config` モジュールは廃止されました。エンジン設定は `EngineMetadata.default_params` で管理されます。

```python
from livecap_cli import EngineMetadata

# 利用可能なエンジンを取得
engines = EngineMetadata.get_all()

# 特定言語に対応するエンジンを検索
ja_engines = EngineMetadata.get_engines_for_language("ja")
# → ["reazonspeech", "parakeet_ja", "qwen3asr", "whispers2t_base", ...]

# エンジンのデフォルトパラメータを確認
info = EngineMetadata.get("reazonspeech")
print(info.default_params)
# → {"temperature": 0.0, "beam_size": 10, "use_int8": False, ...}
```

### 3.3 リソース (`livecap_cli.resources`)

```python
from livecap_cli.resources import (
    # 設定 API (Issue #375)
    configure_resources,        # root を設定して configuration を freeze する
    get_resource_configuration, # 解決結果を読む (freeze しない)
    reset_resource_graph,       # graph を作り直す (configuration は維持)

    # snapshot 型
    ResourceConfiguration, RootResolution, ResourceSearchResolution,
    StagingPolicy, StagingRootStatus, ConfiguredPath, OverriddenEnv,

    # 例外
    ResourceConfigurationError, AsciiStagingUnavailableError,
    FFmpegNotFoundError, FFmpegUpstreamUnavailable,

    # 共有 graph のアクセサ
    get_model_manager, get_ffmpeg_manager, get_resource_locator,
)
```

#### configure_resources()

```python
configure_resources(
    *,
    data_root: str | None = None,
    models_dir: str | None = None,
    cache_dir: str | None = None,
    resource_root: str | None = None,
    extra_resource_roots: Sequence[str] | None = None,
    staging_root: str | None = None,
) -> ResourceConfiguration
```

**優先順位は API > env > built-in default。** `data_root` から派生するのは
`data_root/"models"` と `data_root/"cache"` **だけ**で、静的 resource の検索 root
は派生しない。個別指定と `data_root` の併用はエラーにせず、個別指定が勝つ。

**明示された入力が使えないときは候補へ黙って落ちず送出する。** root 種別ごとの
判定:

| root 種別 | 判定 |
|---|---|
| models / cache / data | 作成可能 + 書き込み probe 成功 |
| resource / extra | 存在する読み取り可能な directory (書き込みは要求しない) |
| staging | ASCII + 長さ + 作成・書き込み可能 → 不可なら `AsciiStagingUnavailableError` |

**API が設定済みの env を上書きするときは `WARNING` を出し、readback の
`overridden_env` にも載せる。** 非 ASCII パス問題を `LIVECAP_CORE_MODELS_DIR` で
回避しているユーザーのホストが `data_root` を渡すと、env が無視されて数 GB の
再ダウンロードが起きるため。

#### 静的 resource の検索順

API 指定の有無で 2 分岐し、**混在しない**。

```
API resource_root あり:  API → project → source → extra → package fallback
                         LIVECAP_RESOURCE_ROOT は検索順から除外し overridden_env へ記録
API resource_root なし:  env  → project → source → extra → package fallback
```

API を指定したのに env root も検索候補に残すと、それは「上書き」ではなく
「優先 fallback」になり、`overridden_env` の意味と食い違う。

#### freeze / reset

| 操作 | freeze | filesystem |
|---|---|---|
| `configure_resources()` 成功 | **する** | **明示指定 root のみ**検証 (作成 + probe) |
| manager getter による graph 初期化 | **する** | 既定 root を作成 |
| `get_resource_configuration()` | **しない** | **一切触らない** (preview) |

すべて単一 lock 下で行い、**部分生成された graph を公開しない**。env は freeze
時点で写しを取り、以後の変更は無視する — manager は env を読まない。

再設定は**静的 configuration 全体が一致するときのみ** no-op として成功する。
path だけで判定しない (`data_root` を渡すのと `models_dir`/`cache_dir` を個別に
渡すのは、結果の path が同じでも意図が違う)。

`reset_resource_graph()` は graph 全体を作り直すが **frozen configuration は
維持する**。個別 manager だけを差し替える手段は用意しない — graph の一部が古い
configuration を参照する状態を作れてしまうため。

#### get_resource_configuration()

| フィールド | 型 / 内容 |
|---|---|
| `models` / `cache` | `RootResolution` |
| `resource_search` | `ResourceSearchResolution` (`effective_roots` は順序付き tuple) |
| `staging_policy` | `StagingPolicy` (明示指定の有無) |
| `staging_roots` | `tuple[StagingRootStatus, ...]`。root が選ばれた時点で埋まる |
| `is_frozen` | freeze 済みか |
| `models_root` / `cache_root` | `models.resolved` / `cache.resolved` への property |

`RootResolution` は `configured` (**正規化前**の値) / `resolved` / `source`
(`api`/`env`/`default`/`fallback`) / `is_ascii` / `fallback_reason` (**root ごと**) /
`overridden_env` を持つ。

**`is_frozen=False` の preview では root の利用可能性が未検証である。** preview は
directory 作成も書き込み probe も行わないため — 起動ログに readback を出すホストが
意図せず root を実体化しないようにするため。

#### path 正規化

`expanduser()` → `abspath()` → `normpath()`。**`Path.resolve()` は使わない** —
symlink を追跡するとホストが渡した path と別の場所を指し始め、readback が
「渡していない path」を返すことになる。

#### 構築の唯一点

`ModelManager` / `FFmpegManager` / `ResourceLocator` を構築するのは
`resources/graph.py` の `build_resource_graph()` **だけ**である。`FFmpegManager` は
locator と model manager を**必須注入**で受け取り、無引数構築を拒否する
(`tests/core/resources/test_resource_graph.py` が AST で検査)。他所で構築すると
その instance だけが frozen configuration の外側に立ち、「設定したのに効かない」
が再発する。

#### ModelManager API


| プロパティ/メソッド | 説明 |
|-------------------|------|
| `models_root` | モデル保存のルートディレクトリ |
| `cache_root` | キャッシュのルートディレクトリ |
| `get_models_dir(engine_name)` | エンジン固有のモデルディレクトリを取得 |
| `get_temp_dir(purpose)` | 目的別の一時ディレクトリを取得 |
| `download_file(url, ...)` | ファイルをキャッシュにダウンロード |
| `download_file_async(url, ...)` | download_fileの非同期版 |
| `temporary_directory(purpose)` | 一時ディレクトリのコンテキストマネージャ |
| `huggingface_cache()` | HF_HOMEを設定するコンテキストマネージャ |

#### FFmpegManager API

| メソッド | 説明 |
|---------|------|
| `resolve_executable()` | FFmpegバイナリを検索 |
| `resolve_probe()` | FFprobeバイナリを検索（見つからなければ `None`） |
| `ensure_executable()` | FFmpegの存在を確認（必要なら **ffmpeg/ffprobe を対で**自動ダウンロード） |
| `ensure_executable_async()` | 非同期版 |
| `configure_environment()` | PATHを設定して実行パスを返す |
| `configure_environment_async()` | 非同期版 |

解決順は `LIVECAP_FFMPEG_BIN` → managed cache → 同梱 `ffmpeg-bin` → system PATH。

**managed cache と host 管理の区別** (Issue #398)。managed cache
(`<cache_root>/ffmpeg`) は自動ダウンロードが置いた領域なので、固定した SHA-256
と一致するかを検証し、一致しなければ**対で再取得**する。managed cache は
PATH より優先されるため、破損時に黙って PATH へ落ちることはしない (実行される
FFmpeg が不可視に変わってしまうため)。再取得が**上流へ到達できずに失敗した場合のみ** fall through する (`FFmpegUpstreamUnavailable`)。checksum 不一致・permanent 4xx・実行不能などは fail loud。
配置は staging + `os.replace()` で、**staging 中の失敗は cache を変更しない**。
rename 2 回の間で失敗した場合は stamp が更新されないため次回検証で修復される。`LIVECAP_FFMPEG_BIN` /
同梱 `ffmpeg-bin` / PATH はユーザーが用意したものとして**検証も置換もしない**。

検証コストを避けるため `<cache_root>/ffmpeg/.livecap-ffmpeg.json` に
`(sha256, size, mtime_ns)` を記録し、一致する間はハッシュを再計算しない。

`resolve_probe()` が `None` を返し得るのは従来どおり（host 管理の FFmpeg には
ffprobe が無いことがある）。一方 **managed install は片方だけの状態を残さない** —
2 本とも検証を通ってから配置する。

自動ダウンロードの対象は `win-64` / `linux-64` / `macos-64` (Intel) のみ。それ以外
(macOS arm64、Linux ARM、32bit) は**明示的なエラー**で導入方法を案内する。

### 3.4 文字起こし (`livecap_cli.transcription`)

```python
from livecap_cli.transcription import (
    # メインパイプライン
    FileTranscriptionPipeline,

    # データクラス
    FileTranscriptionProgress,
    FileProcessingResult,
    FileSubtitleSegment,

    # 例外
    FileTranscriptionCancelled,

    # コールバック型
    ProgressCallback,
    StatusCallback,
    FileResultCallback,
    ErrorCallback,
    SegmentTranscriber,
    Segmenter,
)
```

#### FileTranscriptionPipeline

```python
class FileTranscriptionPipeline:
    def __init__(
        self,
        config: Dict[str, Any],
        ffmpeg_manager: Optional[FFmpegManager] = None,
        segmenter: Optional[Segmenter] = None,
    ): ...

    def process_file(
        self,
        file_path: Path,
        segment_transcriber: SegmentTranscriber,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> FileProcessingResult: ...

    def process_files(
        self,
        file_paths: List[Path],
        segment_transcriber: SegmentTranscriber,
        progress_callback: Optional[ProgressCallback] = None,
        status_callback: Optional[StatusCallback] = None,
        result_callback: Optional[FileResultCallback] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> None: ...

    def close(self) -> None: ...
```

### 3.5 言語コード変換 (`livecap_cli.engines.metadata`)

```python
from livecap_cli.engines import EngineMetadata

# BCP-47 → ISO 639-1 変換（ASRエンジン用）
iso_code = EngineMetadata.to_iso639_1("zh-CN")  # -> "zh"
iso_code = EngineMetadata.to_iso639_1("pt-BR")  # -> "pt"
iso_code = EngineMetadata.to_iso639_1("ja")     # -> "ja"

# 言語に対応したエンジンを取得
engines = EngineMetadata.get_engines_for_language("zh-CN")
# -> ["whispers2t"]
```

> **Note**: `livecap_cli.languages` モジュールは Issue #168 で廃止されました。
> 言語コード変換には `EngineMetadata.to_iso639_1()` を使用してください。

### 3.6 CLI (`livecap_cli.cli`)

```python
from livecap_cli.cli import (
    main,              # CLIエントリーポイント
    diagnose,          # プログラム的な診断実行
    DiagnosticReport,  # 診断結果データクラス
)
```

CLI使用法:
```bash
# 診断を実行（FFmpeg, CUDA, VAD backends, ASR engines を表示）
python -m livecap_cli --info

# JSON形式で出力
python -m livecap_cli --as-json

# FFmpegを確保
python -m livecap_cli --ensure-ffmpeg
```


### 3.4 ASCII path 保証 (`livecap_cli.paths`)

```python
# 公開面はこの 2 つと例外だけ。root 選定 (roots) と回収 (reaper) は内部実装で、
# 選ばれた root は get_resource_configuration().staging_roots から読む
# (selector を直接呼ぶと configuration を freeze する副作用がある)。
from livecap_cli.paths import (
    ascii_safe_temp_environment,  # ネイティブが自前で %TEMP% へ展開する境界
    ascii_safe_workspace,         # 我々がファイルを作る境界
    AsciiPathError, AsciiStagingUnavailableError, TempEnvironmentConflictError,
)
```

**ネイティブライブラリが narrow path で path を扱う境界だけ**に使う。次の場合は使わない:

- `*_buf` / `*_bytes` / serialized-proto / file-object 版の API がある (= 方式①)
- CPython 経由のみで到達する (`open` / `pathlib` / `shutil` / `tarfile` / `json`)。
  実測で `tarfile.extractall` / `urlretrieve` / `huggingface_hub` はすべて非 ASCII でも通る (= 方式②)

**② で足りる境界に ③ を持ち込まないこと。**

#### 2 つの API と、その非対称

| | env を変える | 退出時に自分の dir を消す |
|---|---|---|
| `ascii_safe_temp_environment()` | **する** | **しない** |
| `ascii_safe_workspace()` | **しない** | **する** |

同じ 1 つの事実から出る — **プロセス全体の TEMP を向けている間は、無関係なスレッドの
`NamedTemporaryFile()` もそこへ落ちる**。向けていなければ自分のファイルしか無いので消して安全。
前者で消すと Issue #386 のデータ消失が再発する。残骸は TTL reaper が回収する。

発話ごとの一時 wav の正解は **`ascii_safe_workspace()`** — 非 ASCII な `%TEMP%` に作ってから
staging するのではなく、**最初から ASCII 空間に ASCII 名で作る**。ここで
`ascii_safe_temp_environment()` を使うと、発話ごとにプロセスグローバル状態を書き換えることになる。

#### staging root の選定

`configure_resources(staging_root=...)` / `LIVECAP_CORE_ASCII_STAGING_DIR` が最優先で、
**不正なら freeze 時に `AsciiStagingUnavailableError`** (候補へ降りない)。明示指定が無ければ
候補 ladder を降りる: `%ProgramData%` → `%SystemDrive%` → `%PUBLIC%` → cache root → system temp。
述語は **ASCII → 長さ → 作成 → 書き込み probe**。全滅すれば送出する — **元の非 ASCII path へ
黙って fallback しない**。

選ばれた root は `get_resource_configuration().staging_roots` に出る。
`StagingRootStatus` は `path` / `source_volume` / **`root_source`** (どの候補が採用されたか) /
**`fallbacks`** (拒否された候補と理由) / `selected_at` を持つ。

`source_volume` は **staging 元**のボリューム — 呼び出し側が渡した入力そのものであり、
**採用された root の drive ではない**。`D:` から staging しようとして同一ボリューム候補が
拒否され `C:\ProgramData\...` へ降りた場合もここは `"D:"` のまま残る。そうでないと
fallback の関係が説明できない。採用先の drive が要るなら `path` から求められる。
source を持たない境界 (現行 2 API) では `None`。

重複判定は **`(path, source_volume)`** で行う — 同じ root でも staging 元が違えば別の関係で、
`D:` と `E:` が同じ fallback 先へ降りたことは**どちらも観測できるべき**である。

`fallbacks` を持つのは、**拒否理由が後続候補の成功と同時に失われる**情報だからである。
運用者にとって重要なのは「cache root が選ばれた」ことではなく「`%ProgramData%` が
長すぎたので cache root へ降りた」ことである。

`root_source` を **`mechanism` と呼ばない**。本 repo では "mechanism" を hardlink / copy の
materialization の意味で使っており (`tests/nonascii/artifacts.py`)、root の選択元をそこへ
入れると読み手が誤解する。どの staging API を通ったかは root ではなく**呼び出しごと**の
属性なので、この型ではなく下記のログに出る。

#### staging 発生ログ

staging のたびに 1 行の構造化ログを出す:

```text
ASCII staging: boundary=parakeet.nemo.restore_from.untar mechanism=temp-environment
  resolved_root='C:\\LiveCap\\staging' root_source=%SystemDrive%
  fallbacks=[%ProgramData%: 'C:\\ProgramData\\...' -> too long (139 > 120)]
```

- `mechanism` は **`temp-environment` / `workspace`** — どの staging API を通ったか
- **root が cache hit でも出す。** 「なぜこの root か」は 2 回目以降こそ分からなくなる
- **`(boundary, mechanism, root)` ごとに初回だけ INFO**、以降は DEBUG。
  `ascii_safe_workspace()` は**発話ごと**に呼ばれるため毎回 INFO にすると realtime
  転写でログが埋まる。一方 DEBUG だけでは通常の CLI / GUI ログで観測できない

#### 所有権マーカー兼 lease と孤児回収

各 entry の中に `.livecap-entry` を置く。**1 つのファイルが 2 つの役割を持つ**:

| 役割 | 意味 | 寿命 |
|---|---|---|
| **所有権** (存在) | この entry は LiveCap が作った | entry と同じ (個別に消さない) |
| **lease** (開いている) | いま使っている | スコープの間だけ |

- **reaper は印のある entry にしか触らない。** 明示 staging root には運用者が**既存の
  ディレクトリ**を指定できるので、TTL だけで回収するとその配下の無関係なデータを消す
  (Issue #386 と同種)
- **lease は開いていることが実体。** 「TTL 超過かつ `rmtree` が通る」は生存判定ではない。
  Windows では `rmtree` の `PermissionError` が、POSIX では `flock` が判定になる
- **entry の中に置く**のが Windows の保護そのもの (外に置くと `rmtree(entry)` を妨げない)。
  消費側には entry の子を渡すので「空のディレクトリを返す」契約と両立する
- **退出時に unlink しない** — 所有権の印が失われるうえ、POSIX では他者が lock を保持する
  path を消してしまう
- **確立できなければ `AsciiPathError`。** 保護なしのディレクトリを渡さない

#### 明示的な非保証

- **`ascii_safe_temp_environment()` が支えるのはスコープ内で完了する同期境界だけ。**
  Python のハンドルは既定で非継承 (PEP 446) なので、**親のスコープより長生きする子プロセスは
  lease で保護されない**。この context の中で spawn した子は、抜ける前に終了 / join すること
- **`fork()` は支えない。復旧手段も用意しない。** 子が引き継ぐ壊れた状態は 1 つではない
  — temp-environment の `RLock` (別スレッドが保持したまま fork するとデッドロック) と
  深度カウンタ、lease の file descriptor (親子が同じ open file description を共有するので
  子が閉じると**親の lease が外れる**)、root 選定キャッシュ、reaper の once-state、
  freeze 済み configuration。一括で戻す API は**使う consumer が居ない**ので作らない。
  マルチプロセスが要るなら `spawn` を使うか、本 API を親でだけ使うこと
- **ブロッキング**する。event loop スレッドから呼ばない。async から使うときは
  **context の enter・境界処理・exit を同じ同期関数にまとめ、その関数全体を 1 回の
  `asyncio.to_thread()` で実行する**。**enter / exit を別々の呼び出しへ分割しない** —
  別の worker スレッドで走り得るが、`RLock` はスレッド所有権を持つので取得した
  スレッド以外からは解放できない (`docs/reference/api.md` の「async から使う」)
- `ascii_safe_temp_environment()` は**単一スレッド上の複数 async task から使わない** —
  排他が `threading.RLock` なので、`await` を跨いだ交差利用は字句的なネストと区別できない
- **`ascii_safe_temp_environment()` はプロセス内で 1 つずつ。** `TEMP` がプロセス全体の
  状態なので排他をスコープの全期間保持する — **別スレッドの呼び出しは boundary / purpose に
  関係なく直列化される** (待たされるのであって `TempEnvironmentConflictError` にはならない。
  同エラーは同一スレッドで別 purpose をネストしたときに出る)。`ascii_safe_workspace()` は
  直列化されない。**スコープ外のモデルロードや推論は直列化しない**

#### まだ実装していないもの

既存のツリーを ASCII 領域へ staging する `ascii_safe_path()` は**実装していない**。
設計は #378 §6 に確定しているが、**現時点で必要とする境界が 0 件**である
(唯一の候補だった sherpa-onnx は 1.13.6 への version bump で ②wide-path になった)。
消費者が現れた時点で実装する。

## 4. Engines パッケージ

### 4.1 エンジンファクトリ

```python
from livecap_cli import EngineFactory, EngineMetadata

# エンジンを作成（EngineMetadata.default_params が自動適用）
engine = EngineFactory.create_engine(
    engine_type="whispers2t_base",
    device="cuda",  # または "cpu"
)

# パラメータを上書きする場合
engine = EngineFactory.create_engine(
    engine_type="reazonspeech",
    device="cpu",
    use_int8=True,  # default_params を上書き
)

# モデルをロード
engine.load_model()

# 音声を文字起こし
result = engine.transcribe(audio_data, sample_rate)
text = result.text
confidence = result.confidence
```

### 4.2 利用可能なエンジン

#### ReazonSpeech

| エンジンID | モデル名 | モデルサイズ | 対応言語 |
|-----------|---------|-------------|---------|
| `reazonspeech` | ReazonSpeech K2 v2 | 159MB | ja |

#### NVIDIA Parakeet

| エンジンID | モデル名 | モデルサイズ | 対応言語 |
|-----------|---------|-------------|---------|
| `parakeet` | Parakeet TDT 0.6B v2 | 1.2GB | en |
| `parakeet_ja` | Parakeet TDT CTC 0.6B JA | 600MB | ja |

#### NVIDIA Canary

| エンジンID | モデル名 | モデルサイズ | 対応言語 |
|-----------|---------|-------------|---------|
| `canary` | Canary 1B Flash | 1.5GB | en, de, fr, es |

#### MistralAI Voxtral

| エンジンID | モデル名 | モデルサイズ | 対応言語 |
|-----------|---------|-------------|---------|
| `voxtral` | Voxtral Mini 3B | 3GB | en, es, fr, pt, hi, de, nl, it |

#### WhisperS2T (OpenAI Whisper)

| エンジンID | モデル名 | モデルサイズ | 対応言語 |
|-----------|---------|-------------|---------|
| `whispers2t_tiny` | Whisper Tiny | 39MB | 多言語（13言語） |
| `whispers2t_base` | Whisper Base | 74MB | 多言語（13言語） |
| `whispers2t_small` | Whisper Small | 244MB | 多言語（13言語） |
| `whispers2t_medium` | Whisper Medium | 769MB | 多言語（13言語） |
| `whispers2t_large_v3` | Whisper Large-v3 | 1.55GB | 多言語（13言語） |

> **WhisperS2T対応言語**: ja, en, zh-CN, zh-TW, ko, de, fr, es, ru, ar, pt, it, hi

### 4.3 BaseEngine インターフェース

```python
class BaseEngine(ABC):
    def __init__(self, device: Optional[str], config: Optional[Dict]): ...

    def load_model(self) -> None: ...
    def set_progress_callback(self, callback: ProgressCallback) -> None: ...

    @abstractmethod
    def transcribe(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> TranscriptionResult: ...
    # ``TranscriptionResult`` は text / confidence / engine_confidence を
    # 持つ frozen dataclass。attribute access で値取得 (Issue #308 / PR-A.0)。

    @abstractmethod
    def get_engine_name(self) -> str: ...

    @abstractmethod
    def get_supported_languages(self) -> List[str]: ...

    @abstractmethod
    def get_required_sample_rate(self) -> int: ...

    def is_initialized(self) -> bool: ...
    def cleanup(self) -> None: ...
```

## 5. 環境変数

| 変数名 | 説明 | デフォルト |
|--------|------|-----------|
| `LIVECAP_CORE_MODELS_DIR` | モデル保存ディレクトリ | `~/.livecap/models`（またはappdirs） |
| `LIVECAP_CORE_CACHE_DIR` | キャッシュディレクトリ | `~/.livecap/cache`（またはappdirs） |
| `LIVECAP_FFMPEG_BIN` | FFmpegバイナリディレクトリ | 自動検出 |
| `LIVECAP_CALIBRATION_CORPUS_DIR` | Confidence filter calibration corpus (`benchmarks/confidence_calibration/`) dir | `appdirs.user_data_dir("LiveCap", "PineLab") / "calibration_corpus"` (Windows: `%LOCALAPPDATA%\PineLab\LiveCap\calibration_corpus`、 Linux: `~/.local/share/LiveCap/calibration_corpus`、 appauthor は `appdirs` 仕様上 Windows 専用) |

## 6. 使用例

### 6.1 ファイル文字起こしパイプライン

```python
from pathlib import Path
from livecap_cli import FileTranscriptionPipeline, FileTranscriptionProgress

def transcribe_segment(audio_chunk, sample_rate):
    # ここでASR推論を実行
    return "transcribed text"

def on_progress(progress: FileTranscriptionProgress):
    print(f"[{progress.current}/{progress.total}] {progress.status}")

pipeline = FileTranscriptionPipeline()

result = pipeline.process_file(
    file_path=Path("audio.wav"),
    segment_transcriber=transcribe_segment,
    progress_callback=on_progress,
)

for segment in result.subtitles:
    print(f"{segment.start:.2f}-{segment.end:.2f}: {segment.text}")

pipeline.close()
```

### 6.2 エンジンを直接使用

```python
from livecap_cli import EngineFactory
import numpy as np

# 英語音声を文字起こし
engine = EngineFactory.create_engine(
    engine_type="whispers2t_base",
    device="cuda",
    language="en",  # 言語を明示指定
)

engine.load_model()

# audio_dataはfloat32サンプルのnumpy配列を想定
audio_data = np.zeros(16000, dtype=np.float32)  # 1秒の無音
sample_rate = 16000

result = engine.transcribe(audio_data, sample_rate)
text = result.text
confidence = result.confidence
print(f"文字起こし結果: {text} (確信度: {confidence:.2f})")
```

### 6.3 リソース管理

```python
from livecap_cli.resources import get_model_manager, get_ffmpeg_manager

# モデル管理
model_manager = get_model_manager()
models_dir = model_manager.get_models_dir("whispers2t")
print(f"モデル保存先: {models_dir}")

# FFmpeg管理
ffmpeg_manager = get_ffmpeg_manager()
ffmpeg_path = ffmpeg_manager.ensure_executable()
print(f"FFmpegパス: {ffmpeg_path}")
```

### 6.4 言語コード変換

```python
from livecap_cli.engines import EngineMetadata

# BCP-47 → ISO 639-1 変換（ASRエンジン用）
print(EngineMetadata.to_iso639_1("zh-CN"))  # "zh"
print(EngineMetadata.to_iso639_1("pt-BR"))  # "pt"
print(EngineMetadata.to_iso639_1("ZH-TW"))  # "zh" (大文字も自動正規化)

# 言語に対応するエンジンを取得
engines = EngineMetadata.get_engines_for_language("ja")
print(engines)  # ["reazonspeech", "parakeet_ja", "qwen3asr", "whispers2t"]

engines = EngineMetadata.get_engines_for_language("zh-CN")
print(engines)  # ["qwen3asr", "whispers2t"]
```

## 7. インストール

```bash
# 基本インストール
pip install livecap-core

# 翻訳サポート付き
# Issue #402 以降、Google 翻訳は追加の依存を必要としない (core の requests と
# 標準ライブラリだけで動く) ため、この extra は空。互換のため名前は維持している。
# ローカル翻訳は [translation-local] (OPUS-MT) / [translation-riva] を使う。
pip install livecap-core[translation]

# 開発ツール付き
pip install livecap-core[dev]

# PyTorchエンジン付き（ReazonSpeech, Whisper）
pip install livecap-core[engines-torch]

# NeMoエンジン付き（Parakeet, Canary）
pip install livecap-core[engines-nemo]
```

## 7.9 翻訳エンジン (Translator)

`TranslatorFactory.create_translator(translator_id, **options)` で生成する。

**所有権 (Issue #402 D9)**: **生成した者が所有する。**

| 対象 | 所有者 |
|---|---|
| adapter が自分で生成した `requests.Session` | adapter (`cleanup()` で close) |
| `transport=` で**注入した** Session | **注入元** (adapter は close しない) |
| translator 本体 | **生成元 (CLI / GUI)** — `StreamTranscriber` は所有しない |

**translator インスタンスを複数の `StreamTranscriber` 間で共有しない。** 共有すると
同一 `requests.Session` が並行利用され、安全性が保証されない。source ごとに生成する
(Google adapter はモデルロード不要なので生成コストは低い)。

**Google adapter は文脈 (context) を使わない。** 改行連結した文脈は Google では
行単位に訳され、VAD で分割された 1 文が壊れるため。`context` 引数は Protocol 互換の
ために受け取るだけで無視される。

**翻訳の失敗は通知される。** `set_callbacks(on_translation_status=...)` が
`TranslationStatusEvent` を受け取る (Issue #402 D1)。**segment ごとには発火せず**、
`healthy→failed` と `failed→healthy` のときだけ 1 回ずつ。個々の字幕が原文のままで
ある理由は `TranscriptionResult.translation_state` (`not_requested` / `translated` /
`failed` / `skipped_busy` / `empty`) を見る。

**翻訳は ASR とは別の worker で走る。** 以前は `max_workers=1` の executor を共用して
おり、居座った翻訳が文字起こし自体を止めていた。翻訳の in-flight は常に 1 件で、前が
終わっていなければ後続 segment は `skipped_busy` として飛ばす — 順番を守って遅れて
全部出すより、落とす方が字幕としては良いため。

**リトライは呼び出し側が決める。** adapter は HTTP 1 試行のみで、失敗を型で分類する
(`TranslationNetworkError` = 再試行の価値あり / `TranslationError` = 恒久的)。
リアルタイムは fail fast、ファイル処理は `FILE_RETRY_POLICY` で再試行する。

---

## 8. Phase 1: リアルタイム文字起こし API

Phase 1 で追加されたリアルタイム文字起こし機能の API です。

### 8.1 トップレベルエクスポート（Phase 1 追加分）

```python
from livecap_cli import (
    # 結果型
    TranscriptionResult,
    InterimResult,

    # ストリーミング
    StreamTranscriber,
    TranscriptionEngine,  # Protocol
    TranscriptionError,
    EngineError,

    # 音声ソース
    AudioSource,
    DeviceInfo,
    FileSource,
    MicrophoneSource,  # 遅延インポート（PortAudio依存）

    # VAD
    VADConfig,
    VADProcessor,
    VADSegment,
    VADState,
)
```

### 8.2 結果型

#### TranscriptionResult

```python
@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """文字起こし結果（確定）"""
    text: str
    start_time: float
    end_time: float
    is_final: bool = True
    confidence: float = 1.0
    language: str = ""
    source_id: str = "default"

    @property
    def duration(self) -> float: ...
    def to_srt_entry(self, index: int) -> str: ...
```

#### InterimResult

```python
@dataclass(frozen=True, slots=True)
class InterimResult:
    """中間結果（確定前の途中経過）"""
    text: str
    accumulated_time: float
    source_id: str = "default"
```

### 8.3 StreamTranscriber

VAD プロセッサと ASR エンジンを組み合わせてリアルタイム文字起こしを行うクラス。

```python
class StreamTranscriber:
    def __init__(
        self,
        engine: TranscriptionEngine,
        vad_config: Optional[VADConfig] = None,
        vad_processor: Optional[VADProcessor] = None,
        source_id: str = "default",
        max_workers: int = 1,
    ): ...

    # 低レベル API
    def feed_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> None: ...
    def get_result(self, timeout: Optional[float] = None) -> Optional[TranscriptionResult]: ...
    def get_interim(self) -> Optional[InterimResult]: ...

    # コールバック API
    def set_callbacks(
        self,
        on_result: Optional[Callable[[TranscriptionResult], None]] = None,
        on_interim: Optional[Callable[[InterimResult], None]] = None,
    ) -> None: ...

    # 高レベル API
    def transcribe_sync(self, audio_source: AudioSource) -> Iterator[TranscriptionResult]: ...
    async def transcribe_async(self, audio_source: AudioSource) -> AsyncIterator[TranscriptionResult]: ...

    # 制御
    def finalize(self) -> list[TranscriptionResult]: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...
```

| メソッド | 説明 |
|---------|------|
| `feed_audio()` | 音声チャンクを入力。VADでセグメント検出時は文字起こし実行のためブロッキング |
| `get_result()` | 確定結果を取得（ブロッキング） |
| `get_interim()` | 中間結果を取得（ノンブロッキング） |
| `set_callbacks()` | 結果受信時のコールバックを設定 |
| `transcribe_sync()` | AudioSource から同期的に文字起こし |
| `transcribe_async()` | AudioSource から非同期的に文字起こし |
| `finalize()` | 残っているセグメントを処理して結果リストを返す |
| `reset()` | 内部状態をリセット |
| `close()` | リソースを解放 |

### 8.4 VAD モジュール

#### VADConfig

```python
@dataclass(frozen=True, slots=True)
class VADConfig:
    """VAD設定（すべてミリ秒単位で統一）"""
    threshold: float = 0.5              # 音声検出閾値
    neg_threshold: Optional[float] = None  # ノイズ閾値
    min_speech_ms: int = 250            # 最小音声継続時間
    min_silence_ms: int = 100           # 音声終了判定の無音時間
    speech_pad_ms: int = 100            # 発話前後のパディング
    max_speech_ms: int = 0              # 最大発話時間（0=無制限）
    interim_min_duration_ms: int = 2000 # 中間結果送信の最小時間
    interim_interval_ms: int = 1000     # 中間結果送信間隔

    @classmethod
    def from_dict(cls, config: dict) -> VADConfig: ...
    def to_dict(self) -> dict: ...
```

#### VADProcessor

```python
class VADProcessor:
    """VADプロセッサ（Silero VAD + ステートマシン）"""
    SAMPLE_RATE: int = 16000
    FRAME_SAMPLES: int = 512  # 32ms @ 16kHz

    def __init__(
        self,
        config: Optional[VADConfig] = None,
        backend: Optional[VADBackend] = None,
    ): ...

    def process_chunk(self, audio: np.ndarray, sample_rate: int = 16000) -> list[VADSegment]: ...
    def finalize(self) -> Optional[VADSegment]: ...
    def reset(self) -> None: ...

    @property
    def state(self) -> VADState: ...
    @property
    def config(self) -> VADConfig: ...
```

#### VADSegment / VADState

```python
@dataclass(slots=True)
class VADSegment:
    """検出された音声セグメント"""
    audio: np.ndarray
    start_time: float
    end_time: float
    is_final: bool

class VADState(Enum):
    """VAD状態"""
    SILENCE = 1           # 無音
    POTENTIAL_SPEECH = 2  # 音声の可能性（検証中）
    SPEECH = 3            # 確定した音声
    ENDING = 4            # 音声終了処理中
```

### 8.5 AudioSource モジュール

#### AudioSource (ABC)

```python
class AudioSource(ABC):
    """音声ソースの抽象基底クラス"""

    def __init__(self, sample_rate: int = 16000, chunk_ms: int = 100): ...

    @property
    def sample_rate(self) -> int: ...
    @property
    def is_active(self) -> bool: ...

    @abstractmethod
    def start(self) -> None: ...
    @abstractmethod
    def stop(self) -> None: ...
    @abstractmethod
    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]: ...

    # 同期/非同期イテレータ
    def __iter__(self) -> Iterator[np.ndarray]: ...
    async def __aiter__(self) -> AsyncIterator[np.ndarray]: ...

    # コンテキストマネージャ
    def __enter__(self) -> AudioSource: ...
    def __exit__(self, *args) -> None: ...
    async def __aenter__(self) -> AudioSource: ...
    async def __aexit__(self, *args) -> None: ...
```

#### FileSource

```python
class FileSource(AudioSource):
    """ファイルからの音声ストリーム（テスト・デバッグ用）"""

    def __init__(
        self,
        file_path: Path | str,
        sample_rate: int = 16000,
        chunk_ms: int = 100,
        realtime: bool = False,  # リアルタイムシミュレーション
    ): ...
```

#### MicrophoneSource

```python
class MicrophoneSource(AudioSource):
    """sounddevice ベースのマイク入力"""

    def __init__(
        self,
        device_id: Optional[int] = None,
        sample_rate: int = 16000,
        chunk_ms: int = 100,
    ): ...

    @classmethod
    def list_devices(cls) -> list[DeviceInfo]: ...
```

> **注意**: MicrophoneSource は遅延インポートされます（PortAudio 依存）。CI 環境など PortAudio がインストールされていない環境では、明示的にインポートするまでエラーは発生しません。

#### DeviceInfo

```python
@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """オーディオデバイス情報"""
    index: int
    name: str
    channels: int
    sample_rate: int
    is_default: bool = False
```

### 8.6 例外型

```python
class TranscriptionError(Exception):
    """文字起こしエラーの基底クラス"""
    pass

class EngineError(TranscriptionError):
    """エンジン関連のエラー"""
    pass
```

### 8.7 使用例

#### 同期ストリーム処理

```python
from livecap_cli import StreamTranscriber, FileSource, EngineFactory

engine = EngineFactory.create_engine("whispers2t_base", "cuda")
engine.load_model()

with StreamTranscriber(engine=engine) as transcriber:
    with FileSource("audio.wav") as source:
        for result in transcriber.transcribe_sync(source):
            print(f"[{result.start_time:.2f}s] {result.text}")
```

#### 非同期ストリーム処理

```python
import asyncio
from livecap_cli import StreamTranscriber, MicrophoneSource

async def main():
    engine = EngineFactory.create_engine("whispers2t_base", "cuda")
    engine.load_model()

    transcriber = StreamTranscriber(engine=engine)

    async with MicrophoneSource() as mic:
        async for result in transcriber.transcribe_async(mic):
            print(f"{result.text}")

asyncio.run(main())
```

#### コールバック方式

```python
transcriber = StreamTranscriber(engine=engine)

transcriber.set_callbacks(
    on_result=lambda r: print(f"[確定] {r.text}"),
    on_interim=lambda r: print(f"[途中] {r.text}"),
)

with FileSource("audio.wav") as source:
    for chunk in source:
        transcriber.feed_audio(chunk, source.sample_rate)

for final in transcriber.finalize():
    print(f"[最終] {final.text}")

transcriber.close()
```

## 9. 互換性ポリシー

- **安定API**: `__all__` に記載された全シンボルは安定版とみなされる
- **破壊的変更**: メジャーバージョン更新時のみ
- **非推奨化**: 削除前に最低1マイナーバージョンの警告期間
- **TypedDictフィールド**: 既存フィールドは維持される。追加フィールドはオプショナル

> **1.0.0 未満での優先関係**: 本プロジェクトは現在 `0.1.0` である。上記の
> 「破壊的変更はメジャー更新時のみ」「削除前に最低1マイナーの警告期間」は
> **1.0.0 をタグ付けして以降の規定**であり、それまでは `AGENTS.md` の
> [Backward Compatibility Policy (pre-1.0)](../../AGENTS.md) が優先する。
> 唯一の既知 consumer である `livecap-gui` は lockstep で開発されているため、
> 1.0.0 までは正しさのために内部挙動を壊すことが許容される
> (バグのある既定値を「後方互換」として温存することは許容されない)。
> この方針は最初の `1.0.0` タグ前に再評価する。
