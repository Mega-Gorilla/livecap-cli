# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Epic #64 (livecap-cli refactoring) - completion of all 6 phases.

This represents the completion of a major refactoring effort spanning 6 phases.
Package renamed from `livecap-core` to `livecap-cli`.

### Added

#### ホスト設定可能な resource API と共有 resource graph (Issue [#375] PR 1、epic [#380])

**ホストアプリが root を設定しても黙って効かない**状態の解消。3 つの manager が独立した singleton として生成され、**それぞれが構築時に env を直読み**していたため、`FFmpegManager` が使う cache root と `get_model_manager()` の cache root が**別物になり得た**。ホストにはそれを観測する手段も無かった。

- **`configure_resources()` / `get_resource_configuration()`** を追加。優先順位は **API > env > built-in default**。`data_root` から派生するのは `data_root/"models"` と `data_root/"cache"` **だけ**で、静的 resource の検索 root は派生しない (書き込み用 root と読み取り用 root は別物であるため)
- **明示された入力が使えないときは候補へ黙って落ちず送出する。** 「ホストが渡した root が使えない」ことは「別の場所を勝手に使ってよい」という意味ではない。判定は root 種別ごとに定義した — models/cache/data は **作成 + 書き込み probe**、resource/extra は**存在する読み取り可能な directory** (書き込みは要求しない — read-only なインストール先を指すのは正当な使い方)、staging は **ASCII + 長さ + 書き込み可能**
- **API が設定済みの env を上書きするときは `WARNING` を出し、readback の `overridden_env` にも載せる。** 非 ASCII パス問題を `LIVECAP_CORE_MODELS_DIR` で回避しているユーザーのホストが `data_root` を渡すと、env が無視されて**数 GB の再ダウンロード**が起きる。優先順位を決めるだけでは readback を見ない限り観測できない
- **静的 resource の検索順は API 指定の有無で 2 分岐し、混在しない**: API 指定時は `API → project → source → extra → package fallback` とし、**`LIVECAP_RESOURCE_ROOT` を検索順から除外**して `overridden_env` に記録する。env root を fallback として残すとそれは「上書き」ではなく「優先 fallback」であり、上記の記録と食い違う
- **central factory** `resources/graph.py` が manager 一式を組み立て、`FFmpegManager` へ locator と model manager を**注入する**。以前は `FFmpegManager.__init__` が private に `ResourceLocator()` / `ModelManager()` を作っていた。**`livecap_cli/` の他の場所で構築していないことを AST で検査する test** を追加 — 直接構築するとその instance だけが frozen configuration の外側に立ち、本 issue の不具合が再発するため
- **`get_resource_configuration()` は freeze せず、filesystem も一切触らない。** 起動ログに readback を出すホストが、参照しただけで root を実体化してしまうのを避ける。したがって `is_frozen=False` の preview は**利用可能性が未検証**である
- **env は freeze 時点で固定**し、以後の変更を無視する。manager は env を読まない
- **再設定は静的 configuration 全体が一致するときのみ no-op 成功**。resolved path だけを比べると「`data_root` を渡した」と「`models_dir`/`cache_dir` を個別に渡した」を区別できず、意図が違うのに成功してしまう
- **path 正規化は `expanduser` → `abspath` → `normpath`。`Path.resolve()` は使わない** — symlink を追跡するとホストが渡した path と別の場所を指し始め、readback が「渡していない path」を返すことになる
- **Added**: `ResourceConfiguration` / `RootResolution` / `ResourceSearchResolution` / `StagingPolicy` / `StagingRootStatus` / `ConfiguredPath` / `OverriddenEnv` / `ResourceConfigurationError` / `AsciiStagingUnavailableError`、環境変数名の定数 (`ENV_MODELS_DIR` 等)
- **Changed**: **`ModelManager.__init__` / `ResourceLocator.__init__` が解決済みの値を要求する**
  - **Before**: `ModelManager()` / `ResourceLocator()` が env と既定値を自分で解決していた
  - **After**: `ModelManager(models_root=..., cache_root=...)` / `ResourceLocator(search_roots=...)`。解決は `resources/configuration.py` の責務
  - **Migration**: 直接構築せず `get_model_manager()` / `get_resource_locator()` を使う。**構築は `build_resource_graph()` のみが行う**
- **Changed**: **manager getter から `force_reset` を削除**
  - **Before**: `get_model_manager(force_reset=True)` で 1 つだけ作り直せた
  - **After**: 引数なし。graph 全体を作り直すには `reset_resource_graph()`
  - **Migration**: 一部だけ差し替えられると **graph の一部が古い configuration を参照する**状態が作れてしまうため撤去した。テストで env を読み直したい場合は `_reset_resources_for_tests()`
- **Removed**: **`reset_resource_managers()`**。`reset_resource_graph()` (frozen configuration を維持して graph を再生成) と `_reset_resources_for_tests()` (configuration も消して env 再読込) に分かれた。前者は「graph だけ作り直すのか configuration も消すのか」が未定義だった。shim は残さない (pre-1.0)
- **Tests**: 新規 60 件。優先順位の全組み合わせ / 検索順の 3 ケース (**API あり + env あり で env が除外され記録されること**を含む) / root 種別ごとの fail loud / 上書きの WARNING と記録 / **preview が directory を 1 つも作らないこと** / freeze 境界と env 固定 / 再設定の同一性判定 / 正規化 (**symlink を追跡しないこと**) / **AST による直接構築の検査** / configure と初期アクセスの競合

**本 PR は #375 の PR 1** であり、`ascii_safe_path()` 等の staging core (PR 2)、旧 `unicode_safe_*` の削除 (PR 3)、`utterance_wav` の移行 (PR 4) は含まない。`staging_policy` は**明示指定の有無**だけを保持し、候補 ladder は PR 2 が実装する。

#### 非 ASCII パス境界の棚卸しと検証ハーネス (Issue [#378]、epic [#380] Phase 0)

非 ASCII パス起因の不具合 (sherpa-onnx / NeMo で確認済み 2 件) の**露出範囲を確定**するための調査成果。実際の修正は #375 / #379 / #377 が担当し、本変更は `livecap_cli/` の production code を変更しない。

- **棚卸し表** `docs/research/nonascii-path-boundary-inventory-2026-08.md` — ネイティブ / 第三者ライブラリへパスを渡す **47 境界**を列挙し、境界ごとに ①buffer / ②wide-path / ③ASCII staging / ④fail-fast / 非該当 のいずれかへ分類 (**未分類ゼロ**)。内訳は applicable 44 行 (runtime 実測で確定 30 / 未確定 14) + 非該当 3 行、実測レコード 132 件。**「決定」と「実測で確定」を別の列に分離**しており、未実測を ② として数えない。`ascii_safe_path()` の契約 (シグネチャ / 生存期間 / 機構 ladder / staging root / 並行ロード契約 / fail-loud / 却下した代替案) もここに確定
- **検証ハーネス** `tests/nonascii/` — 実装 PR がそのまま回帰テストとして再利用できる。全プローブを子プロセスで実行 (ネイティブ `abort()` 耐性 + 親プロセスの env を汚さない)、ASCII control との **differential 判定**で `fail_silent` を機械的に検出。仕込み欠陥による自己検証付き
- **受け入れ条件の機械化** `tests/nonascii/test_registry.py` — 未分類ゼロ / silent-failure 行への「現状維持」割り当て禁止 / callsite の生存 / 証拠の有無を CI で強制する
- pytest marker `nonascii_paths` を追加 (cheap tier は既定スイートで実行、実モデル tier は `slow` + `LIVECAP_NONASCII_REAL_MODELS=1`)
- `docs/contributor/adding-an-engine.md` に §10 パス境界チェックリストと AP-6 を追加 (新規 engine 追加時の必須確認項目化)

**新規に判明した事実**:

- Windows で stdio がパイプの場合、`sys.stdout` は `surrogateescape`、`sys.stderr` は `backslashreplace` で、**stdout だけが `UnicodeEncodeError` で落ちる**。CLI の SRT stdout 出力が該当し、別 issue で対応する
- `unicode_safe_temp_directory` / `unicode_safe_download_directory` は `%TEMP%` を `cache_root` へ移設するだけで、その `cache_root` はユーザー名を含むため **ASCII 安全ではない** (全 variant で `fail_silent` を実測)。加えて後者は**共有ディレクトリを rmtree** し、スコープ中に他スレッドが作った一時ファイルを消す (実測で確認)
- sherpa-onnx 1.12.39 は `tokens_buf` / `hotwords_buf` を持たない → 方式①は利用不可 (#361 の hotwords も同じ narrow path を踏む)
- **NeMo が壊れる主因は `.nemo` のパスではなく NeMo 内部の `%TEMP%` 展開先だった** — 片側ずつ固定して実測した結果 (`.nemo` だけ非 ASCII → pass / `%TEMP%` だけ非 ASCII → fail_silent)、`restore_from(restore_path=...)` は非 ASCII を正しく扱えると確定。#379 のレバーは `ascii_safe_temp_environment()` だけで、`.nemo` の staging は不要

#### `VADFileSegmenter` — offline 音声の VAD 分割 adapter (Issue [#366] Phase 1)

`livecap_cli.vad.VADFileSegmenter`: streaming 用 `VADProcessor` を `FileTranscriptionPipeline` の `Segmenter` 契約 (`Callable[[np.ndarray, int], list[tuple[float, float]]]`) に適合させる公開 adapter。呼び出し毎に `reset()` (ファイル間・例外後の状態非持越)、interim segment 除外、EOF で `finalize()` 回収。CLI file mode の `--vad` 接続の基盤で、GUI 等の offline 一括処理からも利用可能。あわせて `FileProcessingResult.metadata` に `detected_segment_count` (検出区間数) / `segmentation_empty` (注入 segmenter がセグメントなしと判定) を追加。

#### EngineInfo 言語解決 metadata + `EngineMetadata.resolve_language()` (Issue [#365])

- `EngineInfo` に末尾 default 付き field を追加 (後方互換、#286 と同パターン): `cli_default_language: str` (`--language` 未指定時の実効値。導入時の名称 `default_language` は [#230] で改名) / `supports_language_auto: bool` (native 自動検出対応、voxtral / qwen3asr のみ True)
- `EngineMetadata.resolve_language(engine_id, requested) -> str`: CLI `--language` の単一解決点。未指定 → `cli_default_language`、明示 → 正規化 + 検証、`auto` → 対応 engine のみ許可。全拒否は `ValueError` (不正形式コードの `langcodes.LanguageTagError` も friendly な `ValueError` に変換)

#### Public SRT serializer `build_srt` / `write_srt` (Issue [#363])

`livecap_cli.transcription.srt` に公開 SRT serializer を追加 (top-level `livecap_cli` からも import 可)。`FileTranscriptionPipeline` の private serializer (`_build_srt` / `_build_translated_srt` / `_format_timestamp`) を module 関数として抽出したもので、pipeline の SRT 書き出しはこれらへ委譲 (出力はバイト単位で同一、`tests/core/transcription/test_srt_serializer.py` で固定)。

- `build_srt(subtitles, *, translated=False) -> str` / `write_srt(path, subtitles, *, translated=False) -> Path`
- `translated=True` は `translated_text` を持つ segment のみ出力 (index は renumber しない — 従来の翻訳 SRT と同一挙動)
- caller が `process_file(write_subtitles=False)` と組み合わせて出力先を制御できる (CLI `-o` / stdout 出力の基盤、入力横への不要な sidecar 生成を回避)

#### EngineMetadata: capability/quality metadata + `recommend()` API (Issue [#286])

`livecap_cli.engines.EngineMetadata` に **言語 × ハードウェアに最適な ASR engine を優先度順で推奨** する `recommend()` API を追加。動機は livecap-gui setup wizard ([gui#326](https://github.com/Mega-Gorilla/livecap-gui/issues/326))。従来 `get_engines_for_language()` は宣言順の list を返すのみで順位付けが無かった。

- **`EngineInfo` 拡張 (後方互換、末尾 default 付き field)**:
  - `quality_tier: Dict[str, LanguageQuality]` — 言語別の品質階層 (`best`/`good`/`fallback`) + 根拠レベル (`measured`/`model_card`/`heuristic`)
  - `vram_required_mb: Optional[int]` — GPU 推論の代表 VRAM 要件 (**VRAM 要件の正本**)。実測値は `docs/planning/issue-73` 由来 (parakeet 2417 / canary 6830 / voxtral 8923)。軽量/未計測/size 依存は `None` (filter で除外しない)
  - capability flags: `cpu_supported` / `cpu_recommended` / `gpu_recommended` / `realtime_on_cpu`
- **新規 public 型**: `LanguageQuality` (NamedTuple)、`ReasonCode` (str Enum、i18n 対応の根拠コード)、`EngineRecommendation` (NamedTuple: `engine_id`/`params`/`rank`/`quality`/`reason_codes`/`scores`)。`livecap_cli` と `livecap_cli.engines` から export
- **`EngineMetadata.recommend(language, gpu_available=False, vram_gb=None)`**:
  - `to_iso639_1()` で言語正規化 (既存 `get_engines_for_language` と parity)
  - hard 除外は「言語非対応」のみ。VRAM 超過は除外せず `EXCEEDS_VRAM` code + 低 score で沈める (wizard が全選択肢を提示 / gray-out できる)
  - sort は分解 score (quality → hardware_fit → latency → streaming) の辞書式多段。同 score の tie は登録順で安定 (「重いモデル優先」は不採用)
  - whispers2t は hardware に応じた `params["model_size"]` を返す (`create_engine(id, device, **params)` で一発起動)。他 engine は空 dict
- **VRAM 要件の正本を `EngineInfo` に一元化** — 専用 `model_vram.py` は作らず、Issue [#96] (VRAM 事前 check) を supersede
- **注意 (値の精度)**: `vram_required_mb` の一部 (parakeet_ja 2500 / qwen3asr None) と `quality_tier` の tier は `heuristic`/`model_card` タグ止まりの推定値。純追加 API で既定挙動は不変、精緻化は #86 benchmark / 別 PR で予定。qwen3asr は v1 draft の「30 言語すべて best」を過剰主張として **good** に是正
- **Tests**: `tests/core/engines/test_metadata.py` 新規 19 件 (recommend の言語/CPU-GPU/VRAM filter/rank/reason_codes/whispers2t params/後方互換)、torch 非依存

#### EnergyGate 限界寄与 ablation harness (Issue [#357])

`benchmarks/confidence_calibration/energygate_ablation.py` を追加。EnergyGate(#292)/ ConfidenceFilter(#334)/ 空text guard の 3 層を独立判定し、4 config (baseline/+energy/+confidence/both) で非音声抑制率・speech FRR・EnergyGate の marginal 寄与を測る。simulate ロジックは engine 非依存で単体 test 可能 (`tests/benchmark_tests/confidence_calibration/test_energygate_ablation.py` 13 件)。

結果 (`docs/research/energygate-effectiveness-2026-07.md`、星の王子さま JA corpus 1375 sample、3 engine): EnergyGate の必要性は **signal 依存だが Whisper の `no_speech_prob` だけが例外**。実際の信頼度量を出す engine (avg_logprob=reazonspeech unique=0 / token_confidence=parakeet_ja unique=1) では ConfidenceFilter が無音幻聴を捕捉し EnergyGate は品質冗長。一方 no_speech_prob engine (whispers2t) では ConfidenceFilter が無音幻聴を見逃し、**EnergyGate だけが 76/676 を捕捉 → 相補的で必要**。speech への false-drop は全 engine で 0 件。→ 既定 ON 維持が妥当。**production コード変更なし (調査のみ)**。

#### Confidence filter observe log: `is_interim` field 追加 (Issue [#351] PR 1)

`livecap_cli/transcription/confidence_filter.py:apply_filter()` に `is_interim: bool = False` kwarg を追加、 `FilterDecision` dataclass と `_decision_to_dict()` の JSON schema に `"is_interim": bool` field を追加。 observe mode log entry の interim path 由来 (`_transcribe_interim`) と final segment path 由来 (`_transcribe_segment` / async 版) を区別可能にする ([Issue #351](https://github.com/Mega-Gorilla/livecap-cli/issues/351))。

**Backward compat 完全維持** — default `False` で既存 caller は kwarg 未指定で動作、 legacy log (`is_interim` field なし) は parser 側で final 相当扱いとする compromise (caveat: legacy log の interim/final は本質的に区別不能)。

- **`apply_filter()` signature 拡張**: `is_interim: bool = False` kwarg 追加、 `FilterDecision` 構築時に反映
- **`FilterDecision` dataclass**: `is_interim: bool = False` field を末尾に追加 (frozen dataclass、 default 値で backward compat)
- **`_decision_to_dict()` JSON schema**: `"is_interim": bool` field を `"reason"` の後に固定位置で追加、 docstring example も更新
- **`stream.py` 3 call site から適切な値を明示 pass**:
  - `_transcribe_segment` (sync final): `is_interim=False`
  - `_transcribe_segment_async` (async final): `is_interim=False`
  - `_transcribe_interim` (interim): `is_interim=True`
- **Docs**: `apply_filter()` docstring に `is_interim` の意味論と backward compat / legacy log caveat を明記、 `benchmarks/confidence_calibration/README.md` の observe log JSON example に `"is_interim": false` 追加
- **Tests**: 11 新 test (`test_confidence_filter.py` に 8 = `TestFilterDecisionDataclass` +2 + `TestIsInterim` +6、 `test_stream.py` integration に 3 = sync final / async final / interim path 各 log field 検証)、 backward compat + interim/final 各 path の log field 検証、 全 268 pass in transcription/cli suite (退行ゼロ)

**下流の 別 PR (Issue #351 PR 2、 CLI merge 後)**: `benchmarks/confidence_calibration/parse_observe.py` の consumer 側対応 — default で `is_interim=True` entry を occurrence counter 前に除外、 `--include-interim` flag で opt-in。 これにより Layer 4 replay pipeline ([Task #393]) で production observe log を calibration に安全に使用可能に。

**関連**: Issue [#334] Finding F6 (Qwen3-ASR auto-detect fail-open) の calibration accuracy 向上、 [Issue #338](https://github.com/Mega-Gorilla/livecap-cli/issues/338) Layer 4 前提整備の 1 つ。

#### Confidence calibration corpus: OS 標準 data dir に永続化 default 化 (Issue [#338] follow-up)

`benchmarks/confidence_calibration/` の corpus directory を **`.tmp/calibration_corpus_full/` (session-local temp) から OS 標準 `user_data_dir` に移行**。 Phase 1/2 で `.tmp/` 配下に手動 mkdir が必要だった導入摩擦を解消、 コーパス構築後も次 sweep で自動的に再利用できるようにする。 dev-only、 production runtime (`livecap_cli/`) には影響なし。

- **`benchmarks/confidence_calibration/pipeline.py:resolve_corpus_dir()` の semantics 変更**:
  - Before: env var 未 set 時 `None` return、 呼出側が `--corpus-dir` 必須 or `return 1` で fail
  - After: env var 未 set 時 **`_default_corpus_dir()` fallback** — `appdirs.user_data_dir("LiveCap", "PineLab") / "calibration_corpus"`
- **OS 別 default path**:
  - Windows: `%LOCALAPPDATA%\PineLab\LiveCap\calibration_corpus`
  - Linux: `~/.local/share/LiveCap/calibration_corpus` (or `$XDG_DATA_HOME/LiveCap/calibration_corpus`、 appauthor は `appdirs` 仕様上 Windows 専用)
  - macOS: `~/Library/Application Support/LiveCap/calibration_corpus`
  - appdirs 欠損 fallback: `~/.livecap/calibration_corpus` (`ModelManager` precedent)
- **`user_data_dir` (persistent) を採用、 `user_cache_dir` (`ModelManager` precedent) ではない理由**: corpus は user が build した label + Layer 2/3 augmented data + reports の集合で、 model cache (再 download 可) と異なり **再生成に時間がかかる persistent data**。 OS の cache 自動削除で消えるリスクを回避。
- **CLI default 変更 (backward compat)**:
  - `sweep.py --corpus-dir`: 未指定時 fail → OS default fallback + directory 存在確認 (存在しない場合は明確な error message + `return 1`)
  - `gen_esc50_non_speech.py --output-dir` / `gen_musan_noise.py --output-dir` / `gen_mixed_noisy_speech.py --output-dir`: `required=True` → `default=None` (env var + OS default fallback)
  - `build_corpus.py --output-dir`: sub directory (e.g. `<root>/ja_clean`) を指定する用途のため `required=True` のまま、 help に typical usage 例を追記
- **docs 更新** (canonical のみ、 research/* Phase 1/2 report は historical artifact として不変):
  - `benchmarks/confidence_calibration/README.md`: Stage 2 quickstart + Corpus 準備方針で OS default fallback を明示
  - `docs/architecture/core-api-spec.md`: 環境変数 table に `LIVECAP_CALIBRATION_CORPUS_DIR` 追加
  - `docs/reference/cli.md`: 環境変数 table に追加
  - `docs/testing/README.md`: env var table に追加
- **Tests**: `tests/benchmark_tests/confidence_calibration/test_pipeline.py:TestResolveCorpusDir` 4 test を新 semantics (fallback) 用に更新 (env unset → OS default、 空文字 → OS default、 tilde expand 既存挙動)、 新規 sentinel assertion (`resolved.name == "calibration_corpus"`) 追加。 `test_sweep.py:test_main_returns_error_when_no_corpus_dir` を `test_main_returns_error_when_corpus_dir_not_exists` にリネーム + semantics 変更 (env unset → directory 存在しない場合の fail-close verify)。
- **Migration**: 旧挙動を維持したい場合は明示的に `--corpus-dir <path>` を指定するか、 `LIVECAP_CALIBRATION_CORPUS_DIR=<path>` で override。 既存 `.tmp/calibration_corpus_full/` を使い続けたい user は env var を設定すれば従来通り。

#### Confidence threshold calibration harness — `sweep.py --breakdown-by` per-metadata-key 混同行列 (Issue [#338] Phase 6a)

Issue #338 Phase 5 の sweep 実行 **前に必須の infrastructure**。 現行 `sweep.py:measure_signals()` は `item.metadata` から `language` の 1 field のみ cherry-pick して `LabeledSample.metadata` に copy しており、 Layer 3 で追加した `snr_db` / `subtype` / `noise_source_dataset` 等が sweep report まで到達しなかった。 このため Phase 5 で 5 engine × ~1 hour GPU sweep を実行しても SNR 別 FRR / noise category 別 pass rate が計算不能で、 Issue #334 PR-4 の Pareto gate 「`clean_frr ≤ 1%` かつ `noisy_frr(SNR 別) ≤ 5%` かつ `known_probe を却下`」を validate できない状態だった。 本 PR は additive schema 拡張で 1 sweep から SNR 別 / subtype 別 / dataset 別の混同行列を全部取り出せるようにする。

- **`benchmarks/confidence_calibration/_core.py`** (拡張、 +100 行): `BreakdownReport` dataclass 追加 (`key` / `value_counts: dict[str, int]` / `sweep_by_value: dict[str, list[ThresholdMetrics]]`)。 `SweepReport.breakdown: dict[str, BreakdownReport]` field を default_factory=dict で **additive 追加** (未指定時は空 dict、 Phase 1 report / `parse_observe.py` output と backward compat 完全維持)。 `_breakdown_key(value) -> str` helper で `None` → `"__none__"` sentinel + float/int/str/bool の `str()` 変換を deterministic に処理。 `compute_breakdowns(samples, key, thresholds, direction) -> BreakdownReport` 関数で per-value サンプル分類 + 既存 `_confusion_matrix()` を各 bucket で reuse (stateless で安全)。 `sweep_threshold()` に `breakdown_by: Optional[list[str]] = None` param 追加、 `None` 時は 100% 現行挙動 fallback。 `report_to_dict()` に breakdown シリアライズ追加、 空時も `"breakdown": {}` として schema 上位互換維持。
- **`benchmarks/confidence_calibration/sweep.py`** (拡張、 +50 行): `measure_signals()` の `LabeledSample.metadata` 構築を **`item.metadata` full merge + 既存 3 key (`text` / `is_available`) override** に変更 (main path + error path 両方、 error 時も `snr_db` 等が保持されて後追い分類可能)。 `breakdown_list(value)` argparse type 新規追加 (comma-separated key + 空文字 / duplicate reject、 `positive_int` / `snr_list` の pattern 踏襲)。 `--breakdown-by` CLI flag 追加、 未指定時は空 list → `sweep_threshold(breakdown_by=None)` として backward compat。 対象 key が全 sample に存在しない場合は warning log で継続 (typo detection、 fail-close ではない)。 `SweepReport.metadata` に `"breakdown_by": [...]` を record して sweep report で実施 key を追跡可能。
- **設計 highlight** (Phase 6a plan D1-D10): **完全 additive schema** (Phase 1 report / PR #342 / `parse_observe.py` output を壊さない)、 `_confusion_matrix()` 再利用で計算誤り抑制、 `None` → `"__none__"` bucket で clean speech (`snr_db` field なし) を SNR 別 breakdown に含めても分類可能、 float 値の str 変換は Layer 3 output の全 integer float 保証で precision 問題回避、 O(N_samples × N_thresholds × N_breakdown_keys) の計算量で GPU 不要 (679 sample × 100 threshold × 3 key で < 1 sec)。
- **タイミング**: 本 PR **は Phase 5 sweep 実行前に merge 必須**。 逆順だと 5 engine × ~1 hour GPU sweep を再走する必要が発生する。 本 PR merge 後、 user は `--breakdown-by snr_db,subtype,noise_source_dataset` 付き 1 回の sweep 実行で Phase 2 report / PR-4 Pareto gate の全 evidence を取得可能。
- **Production runtime 影響**: なし (`livecap_cli/` に `breakdown` / `BreakdownReport` / `compute_breakdowns` / `breakdown_by` / `breakdown_list` 一切 import されない、 grep verify 済)。 dev 依存追加なし (numpy 既存 dep のみ、 stdlib で完結)。
- **テスト**: 新規 33 test (`test_core.py` +21 = `TestBreakdownKey` 5 / `TestComputeBreakdowns` 11 / `TestSweepThresholdWithBreakdown` 4 / serialization empty breakdown 1、 `test_sweep.py` +12 = `TestBreakdownList` 8 / `TestMeasureSignals` full metadata pass-through + error path 2 / `TestMainE2E` breakdown E2E + backward compat 2)。 既存 374 test 全 retain、 zero regression、 full calibration suite total 407 pass。 Phase 1 report / #342 の JSON schema 互換 test で pin。

#### Confidence threshold calibration harness — Layer 3 SNR-mixed noisy_speech corpus CLI (Issue [#338])

PR #343 codex-review 2nd round で確認された **Layered evaluation** framework の Layer 3 (Layer 1 clean baseline + Layer 2 ESC-50/MUSAN hard negative の下流) として、 clean speech に Layer 2 noise を目標 SNR で混合した `label=noisy_speech` corpus を生成する CLI を追加。 Issue #334 PR-4 の Pareto gate `noisy_speech_frr by SNR ≤ 5%` に対する direct evidence 収集用で、 default threshold を下げた際に production の実 mic で背景 noise 混在会話が過剰 reject される trade-off を SNR 別に定量化する土台。

- **`benchmarks/confidence_calibration/_mix_snr.py`** (新規、 ~130 行): RMS-based SNR mixing helper。 `mix_at_snr(speech, noise, snr_db)` は closed-form scale = `sqrt(P_speech / (P_noise * 10^(snr_db/10)))` で合成し、 `match_length` は tile / truncate で noise 長を speech に合わせ、 `check_and_renorm` は `|mix|.max() > 1.0` 時に 0.95 peak へ scale down (全 sample 均一 scaling で SNR ratio 保持)。 numpy-only、 `lhotse` / `torchaudio` 依存追加なし、 `compute_snr_db` は back-compute で test 内実測 verify に使用。
- **`benchmarks/confidence_calibration/gen_mixed_noisy_speech.py`** (新規、 ~360 行、 CLI): default 50 speech × 5 SNR = 250 mixed entry 生成。 `--snr-db-list "-5,0,5,10,20"` (default、 comma-separated float + NaN/inf reject + **raw / formatted duplicate reject** の `snr_list` argparse type) / `--samples` (`positive_int` 再利用) / `--noise-datasets esc50,musan` (default) / `--speech-language ja` (default、 **output manifest entry の `language` は speech-language を継承**、 別引数だと mismatch で `sweep.py --filter-by-language` を汚染するため独立 `--language` は排除) / `--force` で `source_dataset="layer3_mix"` の safe re-augment / `--dry-run`。 speech は `label=speech + language=<speech-language>` filter → filename sort 先頭 N 件、 noise は Layer 2 output (`source_dataset in {esc50, musan}`) から deterministic rotation (`noise_pool[i % len]`)、 同一 speech を全 SNR で mix (paired within-subject 比較で SNR effect 純粋化)。 CLI 起動時に prerequisite check (`manifest.jsonl` 存在 / speech 数十分 / Layer 2 output 非空) で loud fail、 error message で `build_corpus` / `gen_esc50_non_speech` / `gen_musan_noise` を明示誘導。
- **Manifest schema additive**: Phase 2 の 17 field を保持し、 Layer 3 で `snr_db: float` / `noise_source_dataset: "esc50"|"musan"` / `noise_source_file: str` / `noise_source_path: str` の 4 field を additive 追加。 既存 entry (field なし) は `pipeline.load_calibration_corpus()` が `CalibrationCorpusItem.metadata: dict` にそのまま格納するため backward compat 完全維持 (verify 済)。 `label="noisy_speech"` は `_core.py:_normalize_label()` で `"speech"` 扱い → filter reject 時に FRR contribution となり confusion matrix に統合 (既存 test で pin 済)。
- **Output layout (multi-language 対応)**: filename pattern `{speech_stem}_snr{db_str}dB_{noise_subtype}.wav` (例: `segment_0000_snr10dB_clapping.wav`、 負 SNR は `snr-5dB`)、 `{corpus_root}/{speech_language}_noisy_speech/` に配置 (JA なら `ja_noisy_speech/`、 EN なら `en_noisy_speech/`)、 JA と EN を同 corpus で augment しても path 衝突なし。 既存 `.tmp/` gitignore rule で保護。 `--speech-language` は language の **single source of truth** (clean speech filter + output entry `language` field + output subdir 全て一元制御)。
- **Sweep report metadata gap** (Phase 6a で解消済): Layer 3 (本 entry) 時点では `sweep.py:measure_signals()` は `item.metadata` から 3 field (`text` / `language` / `is_available`) のみ cherry-pick しており、 `snr_db` は sweep report まで到達しない gap が残った状態で merge した。 **Phase 6a (PR #345) で `measure_signals()` を `item.metadata` full pass-through + `sweep.py --breakdown-by` CLI flag 追加として解消** され、 1 sweep から SNR 別 / subtype 別 / noise_source_dataset 別の混同行列を report で取得可能に。
- **License**: 出力音声は derivative (clean speech 朗読 + Layer 2 noise)、 raw audio は `.tmp/` 配下のみ、 production runtime 依存追加なし (numpy 既存 dep のみ)。
- **テスト**: 新規 74 test (`test_mix_snr.py` 37 / `test_gen_mixed_noisy_speech.py` 37) で SNR 精度 ±0.5 dB (5 SNR × 3 signal pair pin) / length matching / clip renormalization / determinism / edge case (zero-power / empty) / CLI 引数 validation (`snr_list` NaN/inf reject、 `positive_int` 0 reject) / prerequisite fail case / E2E augment + real SNR back-computation verify を pin。 実 dataset なしで synthetic sine + white noise で covered。 既存 288 test 全 retain、 zero regression。

#### Confidence threshold calibration harness — ESC-50 / MUSAN augmentation CLIs (Issue [#338] Phase 2)

Phase 1 report ([`docs/research/calibration-japan-engines-2026-07.md`](docs/research/calibration-japan-engines-2026-07.md)) の最重要 caveat (synthetic silence + low-level noise では production の applause / laughing / engine 等より easier で、 data-driven threshold が probe を pass してしまう) を解消するため、 **ESC-50** (CC BY-NC 4.0 / dev-only) と **MUSAN noise** (CC BY 4.0 / dev-only) を calibration corpus に augment する CLI 2 本を追加。 Issue #334 PR-4 (default threshold 変更) の直接入力となる Phase 2 report 執筆に load-bearing。 なお本 CLI は Layered evaluation の **Layer 2** (production-realistic non_speech hard negative augmentation) に位置し、 PR-4 default threshold 確定には Layer 3 (SNR-mixed noisy_speech corpus、 別 PR) + Layer 4 (production observe replay) + Layer 5 (VAD + noise gate + confidence_filter + ASR realtime E2E) の follow-up が必要 (PR #343 codex-review 方針 review 反映)。

- **`benchmarks/confidence_calibration/_augment_common.py`** (新規、 ~250 行): ESC-50 / MUSAN 共通の resample (16 kHz mono、 `pipeline._resample_to_16k_mono` 再利用) + deterministic uniform-stride chunking (default 1.5 sec × up to 3-5 chunks) + manifest upsert (`build_corpus._load_manifest_entries` / `_write_manifest` を import 再利用、 `source_dataset` filter で選択的削除 support) + optional dataset download (SHA-256 検証 support)。
- **`benchmarks/confidence_calibration/gen_esc50_non_speech.py`** (新規、 ~280 行、 CLI): ESC-50 の 15 production-realistic category (`laughing` / `sneezing` / `coughing` / `breathing` / `clapping` / `footsteps` / `rain` / `door_wood_knock` / `mouse_click` / `keyboard_typing` / `clock_tick` / `glass_breaking` / `engine` / `car_horn` / `siren`) から default 10 file/category × 3 chunk = ~450 sample augment。 `--categories` で override、 `--samples-per-category` で件数調整、 `--force` で safe re-augment、 `--dry-run` で preview、 `--download` (~600 MB) 自動 fetch。 `meta/esc50.csv` deterministic 選択 (filename sort、 先頭 N 件)。 音楽 (BGM) は controversial として除外。
- **`benchmarks/confidence_calibration/gen_musan_noise.py`** (新規、 ~250 行、 CLI): MUSAN の `noise/{free-sound,sound-bible}/` sub-directory から default 50 file × up to 5 chunk = ~150-250 sample augment。 `music/` と `speech/` は意図的に除外 (music は BGM 判断が別問題、 speech は false positive)。 `--samples` で file 選択総数 (uniform stride、 deterministic)、 `--max-chunks-per-file` で file 当たり chunk 数、 `--force` で safe re-augment、 `--dry-run` で preview、 `--download` (~11 GB) 自動 fetch。
- **Manifest schema additive fields**: Phase 2 augmented entry は Phase 1 の 14 field に加えて `source_dataset` (`"esc50"` / `"musan"`)、 `source_file` (元 filename、 attribution)、 `source_license` (`"CC BY-NC 4.0"` / `"CC BY 4.0"`) の 3 field を持つ。 既存 entry (field なし) は `pipeline.load_calibration_corpus()` が `CalibrationCorpusItem.metadata: dict` にそのまま格納するため forward-compat 完全維持。
- **License safety** (Plan D1): 両 dataset とも raw audio data のため production code から import しようがなく、 `.tmp/` 配下は既存 `.gitignore` rule で git push 事故を物理的に防止。 CI では dataset unavailable のため実行 skip、 Phase 4 (user 環境で dataset download + augment) と Phase 5 (5 engine 再 sweep) の後 Phase 2 report を生成し PR-4 threshold candidate を最終確定する運用。
- **テスト**: 新規 56 test (`test_augment_common.py` 24 / `test_gen_esc50_non_speech.py` 16 / `test_gen_musan_noise.py` 16) で resample / chunking / manifest upsert / force filter / dry-run / CLI 引数 / license attribution / determinism / forward-compat を pin。 実 dataset download なしで synthetic mini-corpus fixture で覆う。

#### Confidence threshold calibration harness — kana-level alignment metric (Issue [#338] PR-γ)

PR-β (Stage 2) で発覚した **表記揺れ起因の偽 low coverage** 問題への対応として、
calibration alignment metric に **kana-level coverage** を並列追加。`pykakasi`
(`GPL-3.0-or-later`、dev 限定) で kanji / katakana を hiragana に正規化し、
acoustic confidence (音素列が正しいか) と lexical surface form (kanji 変換差) を
分離。Phase 4 smoke verify で観測された「1人で vs 一人で」 (text 0.95) / 「サハラ砂漠
vs さはらさばく」 (text 0.21) 等の表記差を kana 化で吸収しつつ、「真っ先 vs さっき」
のような真の音響誤認識は低 score として保持。`sweep.py` は **不変** (PR #340 codex-
review 3rd round の scope minimization 訂正反映、kana ベース sweep は Phase 4 で
必要性確認後の別 PR に分離)。

- **`benchmarks/confidence_calibration/_normalize_jp.py`** (新規): `pykakasi`
  (GPL-3.0-or-later) と `kanjize` (MIT) の lazy import + ``to_hiragana()`` +
  ``normalize_for_alignment()`` (NFKC → CJK 隣接の Arabic 数字 run → kanjize
  で 漢数字化 (`1200 → 千二百`、 `1人 → 一人`) → hiragana → 句読点 strip)。
  両 lib とも **dev 限定 import**、 production runtime は一切 import しない
  (`tests/test_production_no_pykakasi.py` で parametrize した grep guard)。
  正規化は PR #341 codex-review で 4 段階を経た: v1 blanket mask (`一人` と
  `二人` を同一視する false-high) → v2 per-char canonical substitution
  (compound `千二百` を `10002100` と誤変換) → v3 kanji→arabic via
  `kanji2number` (compound numeral は OK だが `一緒` / `十分` / `一番` /
  `一人` 等の idiom で pykakasi の自然な読みを壊す) → **v4 arabic→kanji via
  `number2kanji`** で全方位対応 (idiom は無変更、 EN の `Chapter 1` 等も
  CJK 非隣接なので無変更、 cross-form `1人 ↔ 一人` は kanjize で kanji 化
  後 pykakasi の compound rules で `ひとり` に統一)。
- **`benchmarks/confidence_calibration/build_corpus.py`**: 新規
  ``compute_alignment_score_kana()`` を ``compute_alignment_score()`` と並列に
  追加 (既存関数の signature / 挙動は **不変**)。build_corpus main loop で text +
  kana を並列計算、manifest entry に 3 つの kana field を additive で追記:
  ``alignment_score_kana`` / ``reference_text_matched_kana`` /
  ``transcribed_text_kana``。
- **`benchmarks/confidence_calibration/recompute_alignment.py`** (新規): 既存
  Phase 4 manifest を **audio 再 transcribe なしで** kana metric に migrate する
  CLI。``--manifest`` + ``--reference-text-ja`` / ``--reference-text-en`` +
  ``--force``、idempotent (既に kana field がある entry は skip)、forensic safe
  (text-level field を一切変更しない)。
- **`tests/test_production_no_pykakasi.py`** (新規): livecap_cli/ の .py に
  pykakasi 文字列が現れないことを static grep で assert。本 repo の
  AGPL-3.0-only と pykakasi GPL-3.0-or-later の dev 限定整合性を CI で常時
  guard。
- **`pyproject.toml`**: ``[project.optional-dependencies] dev`` に
  ``pykakasi>=2.3.0`` を追加 (`yt-dlp` と同 group、`uv sync --extra dev` で
  インストール)。runtime ``dependencies`` には一切影響なし。
- **`benchmarks/confidence_calibration/README.md`**: §4.5 (任意) section
  追加、kana metric の動機 / recompute_alignment quickstart / license note。
- 新規 test: ``test_normalize_jp.py`` (28) + kana 専用 test in ``test_build_corpus.py``
  (7) + ``test_recompute_alignment.py`` (13) + production guard (1) = 49 件追加。

#### Confidence threshold calibration harness — Stage 2 (Issue [#338] PR-β)

PR-α (Stage 1) で landed した signal-agnostic sweep core (`_core.py`) を base に、
**user 提供 audio corpus から直接 calibration する Stage 2 CLI** を追加。Stage 1
は observe log 経由、Stage 2 は audio + engine.transcribe() を直接呼ぶ active
calibration。Issue #334 PR-2 / PR-3 / PR-4 の 1-2 月待ちを **~2-3 週** に短縮可能。

- **`benchmarks/confidence_calibration/sweep.py`**: Stage 2 CLI、corpus loader
  (`pipeline.load_calibration_corpus()`) → 各 sample で `engine.transcribe()` →
  `engine_confidence` の signal 抽出 → `_core.sweep_threshold()` で sweep。
  `--engine-kwargs key=value` で engine 個別 kwargs (例: `use_int8=true`) 対応、
  `--quantization` / `--filter-by-language` を report metadata に embed。
- **`benchmarks/confidence_calibration/build_corpus.py`**: corpus build CLI、
  yt-dlp で audio download (URL / local file 両対応) + ffmpeg で 0:06 trim +
  16kHz mono 変換 + Silero VAD で speech segment 切り出し + 各 segment で
  Whisper transcribe + `difflib.SequenceMatcher.find_longest_match()` で 原稿
  fuzzy match (**coverage** = matched substring 長 / transcribed 長、0.0-1.0)
  → `manifest.jsonl` 生成。**idempotent + upsert** (path 単位で同 entry を
  rewrite、`--force` で再生成しても重複なし、他 source の entry は保持)。CLI
  に `--engine-kwargs` 追加 (alignment 用 WhisperS2T の重い default
  `model_size=large-v3` を CPU 環境向けに `model_size=base` 等に override 可能、
  PR #340 review 反映)。`SequenceMatcher` の **autojunk=False を明示**
  (Phase 4 smoke verify で発覚、default `autojunk=True` は長文 reference +
  頻出 char で partial match に縮小される bug、両言語 coverage 値を大幅
  改善: JA 高一致 6/15 → 9/15、EN 1/24 → 12/24)。
- **`benchmarks/confidence_calibration/_vad_chunker.py`**: Silero VAD probability
  stream → speech segment list の pure logic (threshold + hysteresis + min/max
  duration)。`SileroVAD.process()` の 32ms frame probability を sliding 評価、
  `min_silence_sec` 連続 silence で boundary 確定、`max_segment_sec` 超過は
  均等 split。
- **`docs/research/calibration-corpus-sources.md`**: JA / EN リトル・プリンス
  Chapter 1 corpus (user 提供) + PD alternative (Common Voice ja / JSUT /
  LibriSpeech / ESC-50) URL 一覧、著作権 caveat (raw audio は repo 外、private
  利用に限定)。
- **`docs/contributor/adding-an-engine.md` §5** 更新: threshold calibration template
  から本 harness の invoke 例を link (Issue #334 PR-6 で landed した template を
  operational 化、AP-3 量子化 verify の手動 procedure を automated harness で代替)。
- **`benchmarks/confidence_calibration/README.md`** 拡張: Stage 2 quickstart
  (JA / EN Chapter 1 corpus build → 5 engine sweep の流れ)、`--engine-kwargs`
  + metadata embed の説明。
- **`pyproject.toml`**: `[project.optional-dependencies] dev` に `yt-dlp>=2024.0.0`
  追加 (runtime 不要、build_corpus 用)。

Design (Plan D1, D2, D3, D6, D7, D8, D9):
- VAD chunking は Silero probability + simple boundary detection (forced
  alignment lib 追加せず、`difflib` で代替)
- Audio resample は PR-α `pipeline._resample_to_16k_mono` を import 再利用
- Engine.transcribe() は 1 sample 1 回のみ呼ぶ (cache、sweep は cached value)
- Calibration target: ReazonSpeech (P0、int8/float32 両方) / Qwen3-ASR (P1、
  ja+en) / Parakeet_ja (P2) / WhisperS2T (P2)

Tests (`tests/benchmark_tests/confidence_calibration/`): 73 (Stage 1) + 56 new
(Stage 2) = **129 passed** (vad_chunker: 16、build_corpus: 27、sweep: 13)。実
model + 実 yt-dlp / ffmpeg 不要 (MockEngine + subprocess.run mock + 合成 audio
で全 unit test)。

#### Confidence threshold calibration harness — Stage 1 (Issue [#338] PR-α)

新規 `benchmarks/confidence_calibration/` sub-package を追加。observe mode で
蓄積した JSON log + user 提供 label から threshold sweep を実行する CLI
tooling (Stage 1)。Issue [#334](https://github.com/Mega-Gorilla/livecap-cli/issues/334)
PR-2 / PR-3 / PR-4 (observe mode 1-2 月運用に依存) を ~1-2 週に短縮する path。

- **`benchmarks/confidence_calibration/_core.py`**: signal-agnostic な sweep
  logic。confusion matrix (TP/FP/TN/FN)、F1 / precision / recall / Youden's J、
  false_reject_rate を計算。direction (`reject_if_less` / `reject_if_greater`)
  と criterion (`f1` / `youden_j` / `precision` / `recall`) を arg 化、
  `LabeledSample` で input 一般化、`SweepReport` で output 標準化。
- **`benchmarks/confidence_calibration/parse_observe.py`**: Stage 1 CLI。
  ``confidence_filter[observe]: <JSON>`` の jsonl + user 提供 `labels.jsonl`
  (source_id → label mapping) を input、`_core.sweep_threshold()` 経由で
  sweep report を生成。
- **`benchmarks/confidence_calibration/pipeline.py`**: `manifest.jsonl`
  corpus loader、`LIVECAP_CALIBRATION_CORPUS_DIR` env var pattern (既存
  `LIVECAP_NON_SPEECH_CORPUS_DIR` を踏襲)、16 kHz mono float32 への
  自動 resample。
- **`benchmarks/confidence_calibration/README.md`**: Quickstart + signal
  direction / confusion matrix の解釈 / criterion 選択指針 / corpus 準備方針。

**Stage 2 (PR-β、未実装)**: user 提供 audio corpus + engine.transcribe() で
直接 active calibration、yt-dlp + Silero VAD + 原稿 fuzzy match による
corpus build helper を提供予定。

**design 判断**:
- 既存 `benchmarks/non_speech_filter/sweep.py` の argparse + grid sweep
  canonical pattern を踏襲、sibling として位置付け
- `_core` を共通化、Stage 1 / Stage 2 は input 経路のみ異なる (DRY)
- code 挙動の変更ゼロ — 既存 `FilterConfig` / engine 実装は touch しない、
  新 harness のみ追加 (Issue #334 PR-4 で本 output を活用して default を update)

**Tests**: 40 passed (`tests/benchmark_tests/confidence_calibration/`)、
unit test 中心、実 model 不要、Mock + synthetic data で confusion matrix /
edge case / log parse / corpus loader を pin。

#### Qwen3-ASR auto-detect fail-open warning (Issue [#334] Finding 6)

`StreamTranscriber.__init__` で、`filter_config.mode != "off"` かつ engine が
**Qwen3-ASR + `language=None` (auto-detect)** の組合せの時に `logger.warning(...)`
で 1 回通知する。auto-detect path (`Qwen3ASREngine._transcribe_via_wrapper_fallback`)
は ``engine_confidence`` 全 None で fail-open するため filter mode "on" でも
実質無効になる現象を、**programmatic API 利用者** が早期に気付けるようにする。

- **検出 logic**: duck typing (`engine.engine_name == "qwen3asr"` + `engine._asr_language
  is None`) で識別、`isinstance` は循環 import / Mock false negative 回避のため不採用。
- **発火条件 matrix**:
  - `filter=on` + qwen3asr + `language=None` → **warn 1 回**
  - `filter=off` + 同上 → warn なし (filter 不要のため)
  - `filter=on` + qwen3asr + `language="Japanese"` → warn なし (filter active)
  - `filter=on` + 非 qwen3asr engine → warn なし
  - `filter=observe` + qwen3asr + `language=None` → warn 1 回 (filter active)
- **実装場所** (reviewer 2nd round 指摘): `Qwen3ASREngine.__init__` は `FilterConfig`
  を受けないため、両方を知る `StreamTranscriber.__init__` で警告するのが
  architectural separation 上正しい。
- **CLI default は `--language ja`** のため CLI 利用者は通常通り保護される。`language`
  引数を明示すれば warn は出ない (actionable message)。
- **Migration**: 既存 caller は変更不要。`language=None` で auto-detect mode を
  programmatic に利用していた user は warn を 1 回受け取る (silent fail-open の
  解消、行動は変更されない)。
- **Tests** (`tests/transcription/test_qwen3_warn.py`、5 test):
  warn 発火条件 matrix を pin、`MockQwen3LikeEngine` で `Qwen3ASREngine` の
  identifying attribute を模擬。

#### Utterance lifecycle observation hook (Issue [#332])

`StreamTranscriber` の post-processing 経路には 5 種類の silent drop
(filter reject / energy_gate / engine error / engine 空 text / empty audio)
があり、interim 字幕を出した後 final が drop されると consumer 側 state
が clear されず残置する問題があった (livecap-gui#362、ReazonSpeech
`avg_logprob ≈ -0.2` 境界で正常音声断片の false reject が頻発)。本機能で
1 論理 utterance の処理確定 (emit / drop どちらでも) を観測する callback を
追加し、consumer が `emitted=False` 時に interim state を clear できるよう
にした。

- **`on_utterance_settled` callback** を `StreamTranscriber.set_callbacks`
  に追加 (3 番目の kwarg、optional)。`**kwargs` swallow なし、未知 kwarg は
  `TypeError` で即時 fail (policy「不要な後方互換は廃する」、pre-1.0
  cleanup 系列と整合)。
- **`UtteranceSettledEvent` dataclass** (`livecap_cli.transcription.utterance`、
  top-level `livecap_cli` から re-export): 5 field
  (`emitted` / `reason` / `source_id` / `utterance_start_time` /
  `utterance_end_time`)、`frozen=True`。
- **`REASON_*` 静的 reason 定数** (`Final[str]`、public re-export):
  - `REASON_EMPTY_AUDIO = "segment:empty_audio"`
  - `REASON_ENERGY_GATE = "energy_gate:low_rms"`
  - `REASON_FILTER_REJECT = "confidence_filter:reject"` ← GUI #362 主因
  - `REASON_ENGINE_EMPTY = "engine:empty_text"`
- **動的 reason**: `engine_error:<ExceptionType>` (例:
  `"engine_error:RuntimeError"`)。`raise EngineError(...) from e` で chain
  された場合は `__cause__` の型名、chain なし (`__cause__ is None`) の
  場合は `EngineError` 自身の型名 (`"NoneType"` 出力を回避)。
- **Tier 1 の 7 hook point** が settled event を発火: empty_audio /
  energy_gate / filter reject / engine_empty / engine_error /
  coalescer push emission (per output、0-2 件) / coalescer flush emission
  (periodic / force / finalize)。
- **Delivery ordering**:
  - `feed_audio` (callback path): `on_result` 完了 **後** に
    `on_utterance_settled` 発火 (同期実行、stack frame 内で順序保証)
  - `transcribe_async` (async generator): `yield` **直前** に発火
    (yield 後の code は caller が次の `__anext__()` を呼ぶまで実行されない
    ため、break で永久未発火を回避)
  - `finalize` (list return): list append **直前** に発火 (generator path
    と整合)

Consumer example:

```python
from livecap_cli import (
    StreamTranscriber, UtteranceSettledEvent, REASON_FILTER_REJECT,
)

def on_settled(event: UtteranceSettledEvent) -> None:
    if not event.emitted and event.reason == REASON_FILTER_REJECT:
        gui.clear_interim()  # consumer 側 state を即時 clear

transcriber.set_callbacks(
    on_result=on_result,
    on_interim=on_interim,
    on_utterance_settled=on_settled,
)
```

Migration: 既存 caller は不変 (default `on_utterance_settled=None`、
発火コストゼロ)。新 consumer は `set_callbacks` で opt-in。

Out of scope (別 issue で defer): interim path informational reject signal、
coalescer periodic flush で utterance なし時の event、multi-source 内部統合、
`coalescer:discarded` reason (現行実装に該当 branch なし)。

#### NoiseGate `PEAK_SAFETY_MARGIN_DB` user-tunable (Issue [#327])

`analyze_noise_samples` の `suggested_threshold_db` 計算に
`peak_safety_margin_db` keyword 引数を追加、CLI `levels` コマンドに
`--noise-gate-margin <dB>` flag を追加。`engine_min_rms_margin_db` (#292)
と並列の API 対称性を回復。

- **`analyze_noise_samples(peak_safety_margin_db=...)`** (keyword-only):
  `suggested_threshold_db = peak_p95_db + peak_safety_margin_db`。
  default = `PEAK_SAFETY_MARGIN_DB = 6.0`、既存呼び出しは bit-identical。
- **CLI `levels --noise-gate-margin <dB>`**: user が任意の margin を渡せる。
  - 高 SNR studio コンデンサーマイク (AT4040、SM7B 等、self-noise <15 dBA):
    `2` 〜 **負値** (例: `-5` で `suggested = peak_p95 - 5`、`peak_p95 ≈
    -60 dB` の AT4040 で `-65 dB` が得られる)
  - 高ノイズ環境 / 低品質 USB マイク: `10` 程度
- **CLI 出力**: 旧 hardcoded `(= peak_p95 + 6.0)` を user value 反映
  `(= peak_p95 + {margin})` に変更、user が `--noise-gate-margin -5` を
  渡した時に正確な値を表示。

scope: `transcribe` には flag 追加しない (現状 auto-calibration なし、
parse-only effectless flag は anti-pattern。auto-calibration mode は
別 issue 候補)。

Workflow (既存 `levels → --noise-gate-threshold` 手動 pass を維持):

```pwsh
# AT4040 等 studio mic
livecap-cli levels --mic 0 --duration 10 --noise-gate-margin -5 --json
# → {"suggested_threshold_db": -64.6, ...}
livecap-cli transcribe --realtime --mic 0 --noise-gate --noise-gate-threshold -64.6
```

#### Engine confidence signal schema (Issue [#308] PR-A.0 / Phase 1 Layer 3)

- **`EngineConfidence` / `TranscriptionResult` dataclasses** added to
  `livecap_cli/engines/base_engine.py`:
  - `EngineConfidence`: `Optional[float]` fields for `no_speech_prob`,
    `avg_logprob`, `compression_ratio`, `token_confidence_mean`, plus a
    `raw: dict[str, float]` overflow bucket for engine-specific signals.
    `is_available` property returns `True` when at least one signal field
    is non-`None` (PR-A.1 filter precondition).
  - `TranscriptionResult`: `text`, `confidence`, `engine_confidence`
    (default = all-None `EngineConfidence()`). `__iter__` yields
    `(text, confidence)` so the legacy `text, confidence = result` tuple
    unpacking pattern continues to work — no caller migration required
    for existing engine adapters.
  - Both dataclasses are `frozen=True` (immutable) and re-exported via
    `livecap_cli.engines.__all__`.
- **Engine adapter expose paths** (engine-by-engine breakdown):
  - `whispers2t_engine.py`: extracts `no_speech_prob`, `avg_logprob`, and
    `compression_ratio` from the CTranslate2 backend result dict via the
    new pure-function `_extract_engine_confidence()`. Handles both
    top-level signals (current backend shape) and per-segment lists
    (legacy / future shape). Real-machine smoke verify (`normal_speech_neko`
    vs `desk_tap` vs `applause_5_claps`): `no_speech_prob` = 0.036 (speech)
    vs 0.63-0.66 (non-speech) — clean separation usable by the PR-A.1
    filter. Existing `confidence: float` calculation untouched.
  - `parakeet_engine.py`: hybrid `EncDecHybridRNNTCTCBPEModel` is now
    explicitly switched to the CTC decoder via the new
    `_configure_decoding_with_confidence()` helper (see "Changed"
    section below). Real-machine `token_confidence_mean` separation:
    0.01-0.10 (speech) vs 0.0000029-0.0003 (non-speech) — 3-4 orders
    of magnitude, usable by the PR-A.1 filter with a `> 0.005` threshold.
  - `reazonspeech_engine.py`: returns `EngineConfidence()` (all `None`)
    with a docstring `Note` explaining that sherpa-onnx Python bindings
    for transducer models do not expose per-token scores. Users who
    require engine-level hallucination defense are pointed to Silero /
    TenVAD backends. Real-machine smoke verify confirmed
    `is_available is False` on all corpus clips.
  - `qwen3asr_engine.py`, `voxtral_engine.py`, `canary_engine.py`,
    `benchmarks/non_speech_filter/mock_engine.py` (`MockEngine`): no-op
    migration — return `TranscriptionResult(text=..., confidence=...)`
    with default empty `EngineConfidence`. PR-A.1 filter will treat
    these engines as fail-open (`is_available is False` → pass-through).
- **Caller migration** (defensive `hasattr`-based dispatch retained for
  legacy `Tuple[str, float]` mocks):
  - `shared_engine_manager.py` `_process_request` uses `hasattr(result,
    'text')` primary branch, keeps tuple/dict legacy branches.
  - `benchmarks/non_speech_filter/mock_engine.py` `InstrumentedEngine`
    accepts both `TranscriptionResult` and the historical tuple shape so
    benchmark harnesses keep working unmodified.
  - `livecap_cli/transcription/stream.py` `TranscriptionEngine` Protocol
    return type updated to `EngineTranscriptionResult` (runtime import
    alias from `engines.base_engine` — kept out of `TYPE_CHECKING` so
    `typing.get_type_hints()` resolves it). Stream call sites at lines
    546 / 618 / 767 use `text, confidence = ...` unpacking and continue
    to work via the dataclass `__iter__`.
- **New unit tests** (do not require ASR models — pure-function pins
  the extraction logic):
  - `tests/core/engines/test_engine_confidence_schema.py` (17 cases):
    default values, `is_available` semantics across all four signal
    fields, frozen-mutation rejection, `__iter__` yields exactly two
    items (engine_confidence excluded from tuple unpacking), public
    re-export coverage.
  - `tests/core/engines/test_whispers2t_confidence_extraction.py`
    (17 cases): mock CTranslate2 result dicts covering top-level + per-
    segment mean aggregation, missing fields, non-numeric / `None`
    values, and non-dict segment entries.
  - `tests/core/engines/test_parakeet_confidence_extraction.py`
    (12 cases): `FakeHypothesis` mock pinning the token-confidence
    primary path and edge cases (string input, empty list, non-numeric
    values, completely empty hypothesis).
  - `tests/core/engines/test_parakeet_return_hypotheses.py` (5 cases):
    pins that `return_hypotheses=True` is passed to NeMo and that the
    legacy `score / len(y_sequence)` fallback is no longer populated.
  - `tests/core/engines/test_parakeet_decoding_strategy.py` (5 cases):
    pins hybrid-model detection, CTC switch via `decoder_type='ctc'`,
    fallback to strategy-only on `TypeError`, and exception resilience.
  - `tests/transcription/test_transcription_engine_protocol.py`
    (2 cases): pins that `typing.get_type_hints()` resolves
    `EngineTranscriptionResult` to `engines.base_engine.TranscriptionResult`
    rather than the `transcription.result` dataclass of the same name.

The new signals feed into PR-A.1 (`--confidence-filter {off,observe,on}`
post-filter) and PR-A.3 (calibration + production default). Together with
PR-B calibration (PR [#304]) and the PR #307 audio-filter-reference
rewrite, this lands the Phase 1 Layer 3 schema required to close Issue
[#295].

### Documentation

#### 新規 ASR engine 実装 contributor guide 追加 (Issue [#334] PR-6)

Issue [#334](https://github.com/Mega-Gorilla/livecap-cli/issues/334) audit で
発見した「engine 追加時の docstring stale 化」「signal scale 誤認」「silent
fail-open」、および「量子化 calibration 観点の明文化」を構造的に行うため、
新規 ASR engine 追加 contributor 向けの **single source of truth doc** を
新設。本 audit の findings (F2 / F5 / F6 / F8) を anti-pattern として codify
(F8 は既存 ReazonSpeech では PR-A.5.1 で int8 / float32 両方 verify 済、本
codify は新 engine への一般則として位置付け)。

- **`docs/contributor/adding-an-engine.md` 新規**: 9 section (Quickstart 10-step
  checklist / Engine 契約 / 登録 flow / Confidence signal extraction / Threshold
  calibration / 既存 7 engine の reference table / Anti-patterns AP-1 ~ AP-5 /
  Testing 慣用 pattern / CHANGELOG・docs update checklist) を 1 doc で完結
  (~444 行)。
- **`livecap_cli/engines/base_engine.py` `BaseEngine` class docstring 拡張**:
  必須 attribute (`engine_name` / `device`) / Abstract method 4 個 / Hook method
  6 個 / Optional contract (`engine_confidence` populate) を明文化、
  `docs/contributor/adding-an-engine.md` への link。
- **`CLAUDE.md` / `AGENTS.md` cross-reference**: engine adapter section に
  「新規 engine 追加時は `docs/contributor/adding-an-engine.md` 参照」を 1 行
  ずつ追加。
- **Codified anti-patterns** (Issue #334 audit 由来):
  - **AP-1** (F2): 「engine_confidence は常に全 None」 docstring → 後で populate
    追加時に stale 化、新規 consumer が誤読
  - **AP-2** (F2): `token_confidence_mean` threshold を直感で 0.5 等に変更
    → engine 別 scale (Parakeet ja 0.0504 / en 0.2452 / Canary 0.0724) を
    知らないと全 speech false reject regression
  - **AP-3** (F8、一般則): 新 engine 追加時に量子化 (int8 / float32) を smoke
    verify せず threshold 採用 → 量子化で signal 分布が変わる可能性。
    既存 ReazonSpeech は PR-A.5.1 で両量子化 verified (margin +0.13 / +0.10)、
    本 codify は新 engine への一般則。
  - **AP-4** (F6): auto-detect / fail-open path を user 通知なしで残す
    → 「filter on にしたのに reject 0 件」silent failure。
    `StreamTranscriber._maybe_warn_qwen3_auto_detect_fail_open` (PR #336) が
    参考実装
  - **AP-5** (本 doc 自身に対する meta-rule): 新 engine 追加時に本 doc の
    reference table を update しない → doc が stale 化

#### Engine confidence signal semantics clarified (Issue [#334] Findings 1 / 2 / 5)

Issue [#334](https://github.com/Mega-Gorilla/livecap-cli/issues/334) audit
で発見した既存 docstring と実装の乖離 + signal semantics の誤認 risk を
docstring/comment レベルで解消。code 挙動の変更なし、low-risk な
documentation cleanup。

- **`EngineConfidence` の各 field 説明を `Attributes:` section に拡充**
  (`livecap_cli/engines/base_engine.py:22-44`):
  - 各 field の **scale / populate engine / filter 取扱**を明記
  - `token_confidence_mean` の **低 scale (Parakeet ja ≈ 0.0504、
    Parakeet en ≈ 0.2452、Canary en ≈ 0.0724、典型 NeMo confidence 0.85+
    ではない)** を明示 (Issue #334 Finding 2)
  - 「ReazonSpeech / qwen3asr は未対応で全 None」という冒頭の stale 記述を
    削除 (PR-A.5.1 / PR-A.5.2 で対応済)
- **`ReazonSpeechEngine.transcribe()` docstring を PR-A.5.1 反映**
  (`livecap_cli/engines/reazonspeech_engine.py:443-454`):
  - 以前は「`engine_confidence` は **常に全 None**、filter fail-open」と
    読めたが、現在は `avg_logprob` populate 済 (sherpa-onnx 1.12.39+ の
    `ys_log_probs` mean、engine-specific threshold `-0.2`)
- **`FilterConfig.no_speech_threshold` の公式 Whisper 0.6 との差を明記**
  (`livecap_cli/transcription/confidence_filter.py:86-101`):
  - livecap-cli は ``0.5`` (公式より ``0.1`` strict)、PR-A.0 data-calibrated
  - Speech margin / non-speech margin の数値も明記 (Issue #334 Finding 1)
- **`FilterConfig.token_conf_threshold` の docstring に engine 別 scale 追加**
  (`livecap_cli/transcription/confidence_filter.py:102-120`):
  - 「threshold を高い値に変更すると全 speech が false reject される深刻
    regression」を明示 (Issue #334 Finding 2)
- **`FilterConfig.compression_ratio_threshold` の「未使用予約 field」を実態
  に書き換え** (`livecap_cli/transcription/confidence_filter.py:121-128`):
  - extract logic は実装済だが、**現 CTranslate2 backend (WhisperS2T base)
    では `compression_ratio` は常に `None`** (`whispers2t_engine.py:31-33`
    smoke verify 済)
  - forward-compatibility 用、enable には populate verify + calibration の
    2 段階が必要 (Issue #334 Finding 5)

### Removed

#### `SharedEngineManager` orphan module 削除 (Issue [#326])

[Issue #321 PR #3](https://github.com/Mega-Gorilla/livecap-cli/pull/325) の
API contract cleanup 中に発見した orphan code (`livecap_cli/engines/shared_engine_manager.py`、
**467 行**) を完全削除。pre-1.0 cleanup。

**削除対象** (3 symbols すべて zero caller、`__all__` 非 export、production / tests
からの参照ゼロを grep で確認):

- `ProgressCallback` Protocol
- `TranscriptionRequest` dataclass (`__lt__` 比較含む)
- `SharedEngineManager` class (threading + queue + 進捗 callback)

**Migration**: production / tests から未参照のため影響なし。仮に第三者
plugin が import していた場合は git history (`git log -- livecap_cli/engines/shared_engine_manager.py`)
から復元可能。

**reviewer feedback で追加 scope** (本 PR で実施):

- `livecap_cli/transcription/stream.py` の `TranscriptionEngine` Protocol
  docstring 2 箇所 (line 118 / 153) から `SharedEngineManager._process_request`
  の挙動説明を削除、`apply_filter` 単一 consumer 記述に整理
- `AGENTS.md:5` の repo guidance を更新、共有 tooling 説明を
  `shared_engine_manager.py` → `model_memory_cache.py` / `library_preloader.py` /
  `nemo_utils.py` (actually active な shared utility) に置換

**Verification** (本 PR merge 後):

```pwsh
git grep -n "SharedEngineManager\|shared_engine_manager" -- `
  livecap_cli tests AGENTS.md docs/reference docs/guides
# → 0 件 (CHANGELOG.md と docs/planning/archive/* の歴史的言及は許容)

uv run python -c "from livecap_cli.engines import EngineFactory, BaseEngine; print('OK')"
# → OK
```

### Fixed

#### 翻訳の失敗が黙って原文になっていた問題を修正 (Issue [#402])

上記 (Google 翻訳の復旧) と同じ issue の後半。**翻訳エンジンが何であれ、失敗が原文として出る構造そのもの**を直した。これを直さないと、次に上流が変わったとき同じ「日本語→日本語」報告が再発する。

- **Before**: 翻訳が失敗しても `logger.warning` を出すだけで `translated_text=None` を返し、表示側はそれを「翻訳なし」として原文を出していた。**ユーザには何も起きていないように見える** — 実際「モデルを変えても再起動しても直らない」という報告になった (原因は Google 側にあり、こちらは何も変わっていなかった)。swallow は `stream.py` の 3 箇所に散っていた
- **After**: **`TranslationStatusEvent`** を新設し、`set_callbacks(on_translation_status=...)` で受け取れるようにした。**segment ごとには発火しない** — `healthy→failed` と `failed→healthy` のときだけ 1 回ずつで、失敗が続く間は沈黙する。復旧も通知するので「いつまで壊れているか」が分かる
- **個々の字幕が原文のままである理由が分かる**: `TranscriptionResult.translation_state` を追加 (`not_requested` / `translated` / `failed` / `skipped_busy` / `empty`)。原文が出る状態は 1 つではなく、**障害と輻輳時の正常な方針を区別できないと今回の不具合と見分けがつかない**。独立イベントにすると `(source_id, start_time, end_time)` での突き合わせが要るため、**結果そのものの属性**にした
- **失敗理由を失わない内部型**: 3 箇所を個別に直すのではなく `_TranslationOutcome` に統一し、sync / async 双方が同じ funnel (`_settle_translation`) を通る。`_do_translate_direct` は **worker スレッドの中で動く**ため callback を呼ばず、outcome を返すだけにした
- **翻訳が文字起こしを止めない**: 翻訳を ASR とは別の executor へ分離した。以前は `max_workers=1` の executor を共用しており、**居座った翻訳が ASR 自体をブロック**していた
- **輻輳しても backlog を積まない**: 翻訳の in-flight は常に 1 件で、前が終わっていなければ後続 segment は `skipped_busy` として飛ばす。timeout した future は誰も読まないので、**古い翻訳が後から字幕に混ざることはない**。順番を守って遅れて全部出すより落とす方が字幕としては良い
- **callback の例外がパイプラインを壊さない**: 通知の失敗で文字起こしまで止まるのは本末転倒。捕捉して警告し、転写は継続する
- **`close()` は翻訳が translator を使い終わるまで待つ**: translator は呼び出し側が所有しており、`close()` の直後に `cleanup()` される。待たずに返すと**借りている `requests.Session` を使っている最中に閉じられる**。`cancel_futures=True` は実行中の future を止めないため、in-flight を明示的に待つ。**上限は設けない** — 打ち切ると、まさに待つ理由だったケースで借用中の Session を閉じさせることになる。`ThreadPoolExecutor` の worker は non-daemon で interpreter 終了時に join されるため、打ち切ってもハングから逃げられるわけでもない (1 試行を打ち切るのは translator 自身の timeout の役目)。長引いたら 1 度だけ警告する。**デストラクタからは待たない** — GC 中のブロックは危険なため。同じ契約を file pipeline にも適用し、CLI の後始末順を `pipeline.close()` → `translator.cleanup()` へ直した。file pipeline は **completeness を優先して queue する** — realtime と違い、走っている翻訳がすぐ終われば後続は自分の予算内に成功できるため。ただし timeout した queued future を `cancel()` できた場合は in-flight の参照を**実行中の previous へ戻す** — cancel された future は `done()` を返すので、そのままだと `close()` が実行中の翻訳を drain せずに返ってしまう
- **`reset()` が翻訳状態も初期化する**: 持ち越すと前セッションの `failed` のせいで次の障害が通知されず、逆に最初の成功が前セッションに対する `recovered` として出る。**in-flight は捨てない** (捨てると走っている worker と新しい翻訳が同じ translator を並行利用する) 代わりに世代番号で分離し、reset を跨いで完了した翻訳が新セッションの文脈へ書き戻さないようにした
- **`recoverable` は `error_type` から導出する**: 独立フィールドだった頃は `error_type="fatal"` かつ `recoverable=True` のような矛盾が constructor から作れた。導出にすれば矛盾が構築不能になる。`failed` は `message` も必須 (理由の分からない通知では受け手が説明できない)
- **Changed**: **`LIVECAP_TRANSLATION_TIMEOUT` の既定を `10.0` → `5.0` 秒に変更**
  - **Before**: 1 segment の翻訳を最大 10 秒待っていた
  - **After**: 5 秒。リアルタイム字幕では**遅れて届いた翻訳は今話している内容と重なるだけで価値が無い**ため 10 秒は明確に長すぎる。一方で実測 (Session 再利用時の中央値 155-191ms、観測した最悪 1331ms) に対し 5 秒は 4 倍近い余裕があり、回線の遅い環境や重いローカルモデルでも正常な翻訳を切らない
  - **Migration**: 従来どおり 10 秒待ちたい場合は `LIVECAP_TRANSLATION_TIMEOUT=10` を明示する。回線が遅い環境では上げる。超過した segment は原文のまま出て `translation_state="failed"` になる
  - あわせて PR 1 で追加した `LIVECAP_TRANSLATION_REALTIME_DEADLINE` を**削除**した (未リリース)。リアルタイムはリトライしないため実効的な上限は「待つ時間」そのもので、**同じ関心事に 2 つの knob があると片方だけ設定して効かない事故になる**
- **診断**: init 時に resolved な待ち時間と translator の見積 (`estimated_attempt_seconds`) をログする。見積が待ち時間を超える設定では毎回 timeout してしまうため、食い違いを警告する。`close()` で失敗数と skip 数を出す (障害と方針を分けて集計)
- **Added**: `TranslationStatusEvent` (`livecap_cli` から re-export)、`TranslationStatus` / `TranslationErrorType` (`livecap_cli.transcription` から export — 型 alias を submodule に留めるのは `StaticSettledReason` と同じ扱い)、`TranscriptionResult.translation_state`、`set_callbacks(on_translation_status=...)`
- **Removed**: `with_retry` デコレータ。リトライを呼び出し側へ移した結果 production から使われなくなり、`RetryPolicy` と同じことを別の形でするコードが 1 つの module に 2 つ残っていた
- **Removed**: `REALTIME_RETRY_POLICY` / `resolve_realtime_deadline()` と関連定数。**リアルタイムは retry しない**ので `max_attempts=1` の policy は direct call と等価で、実際の待機上限は `StreamTranscriber` が持っている。残しておくと `LIVECAP_TRANSLATION_TIMEOUT` を 2 箇所で parse することになり、不正値で**警告が二重に出ていた**
- **Tests**: 新規 60 件。**通知が 1 回だけ発火し連打しない** / **復旧が通知される** / sync・async 双方が同じ funnel を通る / timeout も失敗として通知される / **callback が例外を投げても転写が継続する** / **発話がイベントに載らない** / `translation_state` の 5 状態 / **並行翻訳が起きない (`peak=1`)** / **翻訳が詰まっていても ASR executor が空いている** / `close()` が両方の executor を畳む / `TranslationStatusEvent` の不正状態を `__post_init__` が弾く

#### Google 翻訳が User-Agent 起因で失敗し、原文がそのまま出ていた問題を修正 (Issue [#402])

ユーザ報告「日本語→英語が日本語→日本語になった。モデルを変えても再起動しても直らない」への対応。原因は当リポジトリ外にあり、**Google が `python-requests/2.x` の User-Agent を絞り、HTTP 200 のまま本文に "Error 500" ページを返す**ようになったこと。実測では 10 回中 5 回失敗していた (ブラウザ UA では 10/10 成功)。

- **Before**: `deep-translator` 経由で `translate.google.com/m` をスクレイピング。同ライブラリは `requests.get()` を**ヘッダ無しで**呼ぶため UA を変更できず、`headers` も `session` も渡す口が無い。最新 1.11.4 でも該当コードは同一で、最終リリースは 2023-06-28。**アップグレードでは直らない**
- **After**: Google 経路の HTTP 呼び出しのみを自前 adapter に置き換え、**ブラウザ UA・timeout・`requests.Session`・transport 注入**を渡せるようにした。エンドポイントと解析対象は従来と同じ。実測で **20 連続 ok=20/20、中央値 155ms**
- **connection を再利用**: 従来は字幕 1 本ごとに TLS ハンドシェイクが走っていた。Session 再利用で **403ms → 191ms (53% 改善)**。リアルタイム字幕では体感品質そのもの
- **リトライを adapter から呼び出し側へ移動**: adapter は**自分がリアルタイムかファイル処理か判断できない**ため予算を分けられなかった。**分類は adapter、方針は呼び出し側**に分離し、adapter は HTTP 1 試行のみ + 型による分類 (`TranslationNetworkError` = 再試行の価値あり / `TranslationError` = 恒久的)。リアルタイムは **fail fast** (遅れて出す方が字幕としては邪魔)、ファイル処理は `FILE_RETRY_POLICY` (3 試行 / 10 秒)。**1 試行あたりの所要時間は translator 自身が見積もる** (`estimated_attempt_seconds`) — 同じ policy が任意の `BaseTranslator` に適用され、ローカルモデルは見積もれないため。deadline は **soft** で、「次の試行を始めてよいか」の判断 (admission control) に使う。**上限の保証ではない** — HTTP client の read timeout はバイト間の待ち時間であって総 wall-clock ではなく、実行中の 1 試行を外から止める手段も無いため
- **リアルタイム deadline は設定可能** (`LIVECAP_TRANSLATION_TIMEOUT`)。実測の最悪が 1331ms だったことから逆算。回線・地域差で一律失敗させないため固定値にしていない。不正値は警告のうえ既定へフォールバック
- **HTTP 200 に埋め込まれたエラーページを検出する**: 従来はステータスしか見ておらず、200 を成功とみなして解析に進み、要素が無いことによる例外になっていた。判定は**成功要素が取れなかった場合にのみ**行う (翻訳結果に "Error 500" が含まれ得るため)
- **翻訳対象テキストがログ・例外へ漏れないようにした**: 翻訳対象は GET query の `q=` に入るため、`requests` 由来の例外文字列には**発話内容が percent-encode された URL ごと**含まれる。`deep-translator` は `TranslationNotFound(text)` と発話そのものを例外にしており、**失敗のたびにユーザのログへ発話が書き込まれていた**。`from None` で cause chain を切り、診断情報は `provider` / `reason` / `status_code` の構造化フィールドで持ち越す。`from error` では呼び出し側が `exc_info=True` にした瞬間に `__cause__` 経由で漏れる。あわせて `file_pipeline.py` が timeout 時に `text[:50]` を直接ログしていた箇所も削除
- **Google では文脈 (context) を使わない**: 改行連結した文脈は Google では**行単位に訳され**、VAD で分割された 1 文が壊れる (`'昨日は
雨が
降りました'` → `'Yesterday
rain
I got off'`)。さらに `context[-0:]` が `context[:]` = 全履歴になる潜在バグがあり、単に 0 にすると悪化した。adapter が context を**無視する**ことで構造的に解消。`opus_mt` が [#190] で 0 にしたのと同じ理由
- **`beautifulsoup4` を使わず標準ライブラリの `html.parser` で解析**: bs4 は `deep-translator` の推移的依存でしかなく、同ライブラリを外すと存在が保証されない。新たな依存を足さずに済ませた
- **URL 長を送信前に検証**: 実測で ~16.3KB を超えると HTTP 400。**文字数ではなく percent-encode 後のバイト長**で測る (同じ 1500 文字でも ASCII 1.5KB / 日本語 13.5KB / 絵文字 18KB)
- **User-Agent はリクエスト単位で付与する**: Session の headers に設定すると、**注入された Session では設定されず #402 の障害が再発する**。こちらが所有していないオブジェクトを恒久的に変更しない意味でも正しい
- **Google の HTTP timeout を `(connect 1.5s, read 2.5s)` にした** — 実測の中央値 155-191ms、観測した最悪 1331ms に対し倍近い余裕がある。1 試行の最悪 4.0 秒が申告値になり、ファイル処理の 10 秒予算にリトライが収まる
- **ファイル処理の翻訳は pipeline が単一 worker を所有する** — 従来は呼び出しごとに executor を作っており、timeout した翻訳が走ったまま次の segment が**同じ translator を並行利用**していた (同一 `requests.Session` の並行利用)。単一 worker により後続はキューへ回り、`close()` で回収される
- **Session の所有権を明確化**: adapter が自分で生成した Session のみ `cleanup()` が close する。注入された transport は注入元が所有する。**translator インスタンスを複数の `StreamTranscriber` 間で共有しない** (`requests.Session` の並行利用は保証されない)
- **Migration**: `deep-translator` 依存を削除した。`translation` extra は**空になったが名前は維持**している (CI 5 箇所と install ドキュメントが `--extra translation` を使うため)。Google 翻訳は core の `requests` と標準ライブラリだけで動く。`--translate google` の使い方に変更は無い
- **既知の限界**: これはスクレイピングであり、**Google 側の変更で再び壊る**。過去にも結果要素の class が `t0` → `result-container` へ変わっている。壊れたときの調査手順を `docs/troubleshooting/translation.md` に用意した。自前化で変わるのは壊れる頻度ではなく、**直るまでの時間** (上流待ち = 無限 → 数時間)
- **Tests**: 新規 87 件。**UA が実際に送信されること** (根本原因そのもの) / 埋め込みエラーページ・空結果・レイアウト変更・permanent 4xx の分類 / **adapter が 1 回しか HTTP を投げないこと** / 明示 context が送信されないこと / **機密文字列がログ・例外・`exc_info=True` のいずれにも現れないこと** (6 失敗形 × 3 観点) / URL 長超過を ASCII・日本語・絵文字それぞれで送信前に弾くこと / Session 所有権 / `RetryPolicy` の deadline が試行回数より優先されること / `@pytest.mark.network` による実エンドポイント疎通 (訳文は変わり得るので「非空・原文と異なる・ASCII 英字を含む」の緩い検証)

#### runtime の FFmpeg 自動ダウンロードが全プラットフォームで 404 していた問題を修正 (Issue [#398])

**ユーザー環境で走る経路**の修復。`_resolve_binary()` の解決順 (`LIVECAP_FFMPEG_BIN` → managed cache → 同梱 `ffmpeg-bin` → system PATH) が全て外れたとき、つまり **FFmpeg を持っていない新規ユーザーの初回起動**がこの経路に乗る。原因は 1 つではなく 6 つあった。

- **Before**: `ffmpeg-windows-64.zip` のような**実在しない資産名**を `releases/latest/download/` から取得しようとして**全プラットフォームで 404**。ffbinaries の資産名は `<tool>-<version>-<platform>.zip` 形式でバージョンが名前の一部であり、platform token も `win-64` / `linux-64` / `macos-64` である (`windows-64` / `osx-64` ではない)。アーキテクチャ判定は `"64" in platform.machine()` だったため **aarch64 に x86-64 ビルド**、`armv7l` に **x86 32bit ビルド**を渡す。ffmpeg と ffprobe は別アーカイブなのに 1 本しか取得せず、`_place_binaries()` は見つからないバイナリを `continue` で黙って飛ばしていた。checksum 検証もリトライも無し
- **After**: 固定した **ffbinaries v6.1** から **ffmpeg / ffprobe の 2 アーカイブ**を取得し、archive と展開後 binary の **SHA-256 を両方検証**する。固定値は `livecap_cli/resources/ffmpeg_manifest.json` (CI の setup action と**共有する単一の正本**)。リトライは **timeout / DNS・transport error / 408 / 429 / 5xx のみ**で指数バックオフ、permanent 4xx とチェックサム不一致は fail loud。失敗時の `FFmpegNotFoundError` は **URL・試行回数・最後のエラー・`LIVECAP_FFMPEG_BIN` による回避策**を含む
- **managed cache と host 管理を分離**: managed cache (`<cache_root>/ffmpeg`) は自動ダウンロードが置いた領域なので固定 SHA-256 と照合し、一致しなければ**対で再取得**する。`LIVECAP_FFMPEG_BIN` / 同梱 `ffmpeg-bin` / PATH のバイナリは**検証も置換もしない** — ユーザーが選んだものを勝手に差し替えない
- **pair は不可分**: ffmpeg だけ正常で ffprobe が欠けた managed cache は、`ensure_executable()` が ffmpeg を見つけた時点で戻るため**従来は永久に修復されなかった**。両方が固定値と一致したときだけ managed cache を候補にすることで、片方の破損が対全体の再取得を発火させる
- **インストールは準原子的**: 一時 workspace で 2 本ともダウンロード・SHA-256 検証・**実行確認**してから `os.replace()` で配置する。**staging 中の失敗は managed cache を一切変更しない**。配置自体は rename 2 回なので完全な原子操作ではないが、**stamp を最後に書く**ため、その間で失敗すると次回の検証が必ず不一致を検出して対ごと再取得する。同期・非同期の `ensure` は同じ lock で直列化する (#386 と同じ所有権問題)
- **壊れた managed cache は下位ソースへ fall through せず修復する**: managed cache は PATH より優先されるため、破損時に黙って PATH へ落ちると**ファイルが壊れたという理由でアプリが実行する FFmpeg が変わる**。`absent` (何も無い) と `invalid` (あるが一致しない) を区別し、invalid のときだけ修復する。修復に失敗した場合の扱いは**失敗の種類で分ける** — 上流へ到達できなかったとき (`FFmpegUpstreamUnavailable`) だけ fall through して可用性を優先し、**checksum 不一致 / permanent 4xx / archive 不正 / 実行不能 / ローカル書き込み失敗は fail loud** する。これらは「インストールしようとしたものがおかしい」という主張であり、代わりに別のビルドを黙って使うのは本 issue が排除した silent degradation そのものだから。`LIVECAP_FFMPEG_BIN` はもともと managed cache より優先なので対象外
- **対応プラットフォーム**: `win-64` / `linux-64` / `macos-64` (Intel) のみ。**macOS arm64・Linux ARM・32bit は明示的なエラー**で `brew install ffmpeg` 等の導入方法を案内する。とくに macOS arm64 で Intel ビルドを黙って Rosetta 2 実行することはしない
- **Migration**: **既存ユーザーへの影響は無い。** この経路は最初のコミット (`5824265`, 2025-11-05) から一度も成功したことがなく、release tag も存在しないため、managed cache は誰の環境でも空である。`<cache_root>/ffmpeg` へ手動でバイナリを置いていた場合のみ、固定版に置き換わる — それを避けるには `LIVECAP_FFMPEG_BIN` でそのディレクトリを指定する
- **性能**: 起動ごとに 268 MB を再ハッシュしないよう、`<cache_root>/ffmpeg/.livecap-ffmpeg.json` に `(sha256, size, mtime_ns)` を記録する。一致する間はハッシュを省略する (実測 190 ms → 0.7 ms)。これは**陳腐化と破損の検出**であり、cache ディレクトリへ書ける相手に対する防御ではない
- **ライセンス**: 取得するバイナリは **GPLv3** (`--enable-gpl --enable-version3`、`--enable-nonfree` 無し)。第三者からユーザーの環境へ取得され別プロセスとして起動されるもので、当プロジェクト (AGPL-3.0) は再配布もリンクもしない。ffbinaries-prebuilt は SPDX 未設定かつ 2023-12-28 以降更新が無く、**v6.1 は「CI と同じだから」ではなく上流の最新であるため**の選択。バージョンを上げる際は供給元の再評価とセットで CHANGELOG に記載すること
- **実装**: `livecap_cli/resources/ffmpeg_pins.py` (manifest + platform 表引き) と `livecap_cli/resources/downloader.py` (分類表・バックオフ・リトライ) を新設。`ModelManager.download_file()` は**変更しない** — 全モデル取得の挙動を巻き込むため、共通化は別途
- **Tests**: 新規 88 件 (うち 1 件は `network` マーカーで既定除外)。platform 写像 parametrize (`x86_64` / `AMD64` / `aarch64` / `arm64` / `armv7l` / `i686` × 3 OS → token または明示エラー) / manifest 整合 (資産名にバージョン・`-windows-`/`-osx-` 不在・ffprobe 名を ffmpeg から導出しない) / 分類表と partial 削除 / **pair 契約** (ffprobe 欠損・ffmpeg 改竄で対ごと再取得) / **stamp** (一致時にハッシュしない) / **原子性** (2 本目の失敗で cache 不変) / **並行 ensure が 1 回だけ install する** / `@pytest.mark.network` で固定 6 URL に HEAD

#### `unicode_safe_download_directory()` が並行処理の一時ファイルを削除していた問題を修正 (Issue [#386])

このヘルパは**プロセス全体**の `TEMP` / `TMP` / `TMPDIR` / `tempfile.tempdir` を `cache_root/downloads` へ向ける。したがってスコープが開いている間は、**プロセス内のあらゆるスレッドの `NamedTemporaryFile()` がそこへ落ちる**。それにもかかわらずスコープ退出時にその**共有**ディレクトリを `shutil.rmtree` していたため、**別処理が使用中の一時ファイルまで削除していた** — `dir=` を指定しない parakeet / canary / qwen3asr の発話ごとの wav が該当する。非 ASCII とは独立した実在の data-loss bug で、Issue [#378] の検証ハーネスで実測されたもの。

- **Before**: スコープ退出時に `cache_root/downloads` を丸ごと `shutil.rmtree` (`_cleanup_directory`)。ロック・refcount・ネスト深度カウンタのいずれも無く、並行/ネストしたスコープが互いの環境スナップショットを壊し得た
- **After**: **最外周スコープごと**に `cache_root/downloads/<uuid>` を作り、**退出時に再帰削除しない**。module level の `threading.RLock` をスコープの全期間保持して直列化し、深度カウンタにより **0→1 でのみ**環境を変更、**1→0 でのみ**復元する。ネストしたスコープは**外側のディレクトリを再利用し同じパスを返す** (環境は外側を指したままなので、別パスを返すと呼び出し側に嘘をつくことになる)。purpose が異なるネストは新設の `TempEnvironmentConflictError` で失敗する
- **Migration**: 呼び出し側の変更は不要 (シグネチャと yield 値の意味は互換)。ただし挙動が 3 点変わる — ①**スコープを抜けてもディレクトリが残る** (回収は Issue [#375] PR 2 の lease / reaper が担当。「別 pid かつ N 時間経過」は生存判定にならないため、安全に回収できる仕組みが揃うまで意図的にリークを受け入れる)、②**別スレッドのダウンロードスコープは直列化される** (プロセス全体の状態を書き換える以上、並行実行は一貫させられない)、③ASCII 保証は**依然として無い** (`cache_root` はユーザー名を含む)。無関係な一時ファイルの**置き場所がずれる問題も未解消**で、消えなくなるだけである。①〜③の恒久対応は Issue [#375] PR 2 / PR 3 が担当する
- **Tests**: `tests/core/utils/test_temp_environment.py` 新規 10 件 (別スレッド / 子プロセスのファイル残存、直列化、ネストのパス一致、例外時の復元、再帰削除しないこと、purpose 衝突)。`tests/nonascii/` のプローブ期待値を `victim_survived_scope_exit=True` へ反転し、**再発したら落ちる**向きで固定

#### Voxtral が `language="auto"` 文字列を mistral-common へ渡していた契約違反を修正 (Issue [#365])

`VoxtralEngine` は `__init__` の生値 (default `"auto"` を含む) をそのまま `processor.apply_transcription_request(language=...)` へ渡していたが、mistral-common の `TranscriptionRequest.language` は **`LanguageAlpha2 | None`** (自動検出 = `None`、具体言語 = ISO 639-1) であり `"auto"` は契約外だった。

- qwen3asr `_resolve_language` と同型の `_asr_language` を導入: `auto`/`None`/空 → `None`、具体言語 → `to_iso639_1` で ISO 639-1 正規化
- 推論時は `language=self._asr_language` を渡す (`self.language` は生値のままログ用に保持)
- **Tests**: `tests/core/engines/test_voxtral_language.py` 新規 — mock processor で上流引数をモデルロードなしに固定 (auto → `None` / `en` → `"en"` / constructor 経由の wiring)

#### CLI `transcribe <file>` が構築時 TypeError で全滅していた問題を修正 (Issue [#363])

`livecap-cli transcribe <file>` (CLI ファイル文字起こし経路) は Phase 6B (ee4ffdc) の初回統合時から `FileTranscriptionPipeline` の**実在しない API** (`FileTranscriptionPipeline(engine=)` / `pipeline.transcribe()` / `result.to_srt()` / `result.segments`) を前提に実装されており、pipeline 構築時点で `TypeError` となり一切機能しなかった (CLI file 成功経路のテストが無く未検出)。`--translate` 経路も実在しない `translator.initialize()` を呼んでいた (実 API は `load_model()`)。

- **`livecap_cli/cli.py:_transcribe_file` を実 pipeline 契約へ全面書き換え**:
  - engine から `segment_transcriber` closure を構築 (`engine.transcribe(audio, sr).text` — `TranscriptionResult` は #314 以降 tuple unpack 不可) → `pipeline.process_file(path, segment_transcriber=…, write_subtitles=False, …)`
  - translator lifecycle 修正: `load_model()` 呼び出し + 言語ペア (`--language` / `--target-lang`) を `create_translator()` へ渡す (OPUS-MT が constructor default `ja→en` に固定される問題も同時解消) + finally で `cleanup()`
  - `result.success=False` (Issue [#362] の全滅検出) を exit 1 + stderr `error` 表示に反映
  - 入力ファイルの存在検証をモデルロード前に移動
  - engine / translator / pipeline の cleanup を finally で保証
- **`FileTranscriptionPipeline.close()` の `getattr` 防御**: 構築時 TypeError で `__init__` 本体が未実行のままインスタンスが GC されると `__del__` → `close()` が未初期化 `_temp_root` を参照して二次 `AttributeError` を出していた
- **`transcribe --help` が日本語 Windows console (cp932) で crash する既存バグを修正**: `--confidence-filter` help に cp932 非対応の em-dash (U+2014) が混入しており `UnicodeEncodeError` で help 表示自体が失敗していた (テストは StringIO 捕捉のため CI で未検出)。help 文字列を ASCII 安全化し、cp932 encode の regression テストで固定
- **help 表記**: realtime 専用オプション 19 件の argparse help に `[realtime only]` を明記、`-o` help に「省略時 stdout」を明記
- **refactor**: engine 生成 boilerplate (device 解決 + kwargs routing + `create_engine` + `load_model`) を `_load_engine()` に一元化し realtime / file 両経路で共有 (経路間 contract drift — #363 の原因パターン — の再発防止)
- **Tests**: `tests/core/cli/test_transcribe_file.py` 新規 13 件 — mock engine が実 `TranscriptionResult` を返し **pipeline は mock せず** temp WAV から実 `process_file()` を通す CLI E2E (engine/torch/FFmpeg/network 不要)。今回のような API drift を CI で検出可能にする

**関連**: [Issue #363](https://github.com/Mega-Gorilla/livecap-cli/issues/363) / [#362] (pipeline 側全滅検出 — 本修正の土台) / [#365] (`--language` routing、別 issue) / [#366] (file mode filter parity、別 issue)

#### Voxtral cache が `LIVECAP_ENGINE_STRONG_CACHE` に従わない問題を修正 (Issue [#198])

Voxtral engine は `(model, processor)` **tuple** を cache していたが、 tuple は `weakref.ref()` を作れないため `ModelMemoryCache.set()` の weak-cache path で `TypeError` になり、 **`LIVECAP_ENGINE_STRONG_CACHE` の設定に関わらず常に強参照** で cache されていた (= env var が Voxtral に対して no-op、 一度 load すると同一プロセス内で VRAM を永続保持)。

- **`livecap_cli/engines/voxtral_engine.py`**: `(model, processor)` tuple を **`VoxtralModelContainer` dataclass** (weakref 可能) に置換
  - `_load_model_from_path()`: `VoxtralModelContainer(model, processor)` を cache に保存、 env var 未 opt-in 時は `allow_promotion=False` も渡す
  - `_configure_model()`: container から model / processor を分離しつつ、 container への strong ref を `self._model_container` で保持 (weak-cache された container が engine 生存中に GC されないように)。 旧 tuple コード由来の未到達 `else` fallback (「単一モデルが直接渡された場合」の compat path、 `_load_model_from_path` は必ず container を返すため dead) を削除し、 contract-trust の bare unpacking に整理 (Issue [#321] 方針)。 未使用の `Tuple` import も削除
  - `cleanup()`: `self._model_container = None` で解放 → 最後の engine が消えれば container も GC され VRAM 解放
- **`livecap_cli/engines/model_memory_cache.py`**: `set()` に `allow_promotion: bool = True` param 追加 (codex-review)
  - `get()` の weak-hit auto-promotion (`_access_count > 3` で `_promote_to_strong_ref`) は、 `allow_promotion=False` の key では走らない
  - **これがないと env var 未設定でも hot-access (4 回目〜) で weak→strong 昇格し VRAM を永続保持してしまう** — Issue #198 の目的 (env var で制御) と衝突するため fix
- **挙動 (env var 別)**:
  - `LIVECAP_ENGINE_STRONG_CACHE=1/true/yes`: 強参照 cache (VRAM 保持、 cross-instance 高速再利用) — 従来と同じ
  - **未設定 (default)**: **weak-cache + auto-promotion 無効** — engine 生存中は再利用、 hot-access でも昇格せず、 最後の参照が消えれば GC で VRAM 解放 (**修正前は env var 無視で常に強参照 or hot-access で昇格していた**)
- **Tests**: 新規 `tests/core/engines/test_voxtral_cache.py` 9 test — tuple が weakref 不可 (root cause) / container が weakref 可能 / weak-cache が holder drop 後に GC / strong-cache が生存 / engine holder が weak-cache を生かす / **default auto-promotion (regression guard)** / **`allow_promotion=False` で >3 access でも weak 維持** / `_no_promote_keys` bookkeeping。 実 Voxtral model は load せず参照挙動のみ検証 (GPU 不要)。
- Option A (weakref-able container + auto-promotion opt-out) 採用 — env var 未設定でも weak-cache として engine 生存中の再利用を維持しつつ、 hot-access 昇格も含め VRAM 永続保持を回避。

#### Layer 3 noise rotation bias fix — 646 pool 全体を uniform stride で span (Issue [#338] Phase 2b)

[Phase 2 report](docs/research/calibration-japan-engines-phase2-2026-07.md) §5.4 / §5.7 で判明した Layer 3 noise diversity bug の恒久修正。 `benchmarks/confidence_calibration/gen_mixed_noisy_speech.py` の `noise_pool[i % len(noise_pool)]` rotation は、 `select_noise_pool` の path sort と組み合わさって **noise pool の先頭 N=n_samples entries のみ選択** する意図しない挙動になっていた。 ESC-50 default 15 categories (alphabetically: breathing → car_horn → ...) では typical n_samples=50 で **breathing (30 files × 5 SNR = 150 sample) + car_horn (20 files × 5 SNR = 100 sample) の 2 subtypes 250 sample だけ** が Layer 3 に含まれ、 remaining 13 ESC-50 categories + MUSAN 全 pool 未使用だった。 dev-only、 production runtime (`livecap_cli/`) には影響なし。

**Before / After distribution** (n_samples=50、 646 pool = 450 ESC-50 + 196 MUSAN):

| Subtype coverage | Before (`i % len`) | After (`_uniform_stride_indices`) |
|---|---|---|
| ESC-50 categories touched | **2 / 15** (breathing、 car_horn のみ) | **15 / 15** (全 category) |
| MUSAN subtypes touched | **0 / 2** (未評価) | **2 / 2** (free-sound、 sound-bible) |
| breathing sample 比率 | **60%** (150/250) | **~7%** (~1/15) |
| car_horn sample 比率 | **40%** (100/250) | **~7%** (~1/15) |
| Layer 3 total variety | 2 subtypes | 17 subtypes (15 ESC-50 + 2 MUSAN) |

**主要変更** (`benchmarks/confidence_calibration/gen_mixed_noisy_speech.py`):

- 新規 `_uniform_stride_indices(pool_size, n_samples) -> list[int]` helper 追加
- `augment()` 内の rotation loop を stride-based に変更:
  ```python
  # Before (biased to first N sorted entries)
  noise_entry = noise_pool[i % len(noise_pool)]
  # After (uniform stride via np.linspace)
  noise_indices = _uniform_stride_indices(len(noise_pool), len(speech_samples))
  noise_entry = noise_pool[noise_indices[i]]
  ```
- Module docstring "Design (Plan D3)" を revised
- 挙動: `n_samples ≤ pool_size` 時は `np.linspace(0, pool_size-1, n_samples)` round で均等分布、 `n_samples > pool_size` 時は各 index を floor/ceil 回数使用 (grouped、 max diff = 1)

**Tests**: 12 新 test 追加 (`TestUniformStrideIndices` × 10 [初回 7 + codex-review 2nd round で pool=3/n=8, pool=4/n=7 の regression + property test の 3 追加] + `TestNoiseSubtypeDiversity` × 2)、 既存 `test_noise_rotation` を invariant-based に update (rename も含む、 rotation pattern を pin しない設計)、 `_fake_corpus_multi_subtype` helper 追加。 全 63 test pass in `test_gen_mixed_noisy_speech.py` + 419 pass in calibration suite、 退行ゼロ。

**Migration (既存 Layer 3 corpus を持つ user)**:

```bash
# Layer 3 entries を削除して再生成 (Layer 1/2 は保持)
uv run python -m benchmarks.confidence_calibration.gen_mixed_noisy_speech \
    --samples 50 --snr-db-list="-5,0,5,10,20" --force --speech-language ja
# --force は source_dataset=layer3_mix の既存 entries を消して再生成

# ja_noisy_speech/ 内の旧 wav が残る場合は手動削除推奨:
rm -rf "$LIVECAP_CALIBRATION_CORPUS_DIR/ja_noisy_speech"/*.wav
```

**フォローアップ (別 PR で対応予定)**:

- Phase 2 report §5.7 addendum: user GPU 環境で corpus 再生成 + Phase 5 sweep 再実行 (RTX 4090 で ~20 min) 後、 Before/After 実測比較を report に追記
- 必要に応じて Issue #334 PR-4 default 閾値 (`avg_logprob_thresholds["qwen3-asr"] = -0.42` 等) の再算定 (`qwen3-asr: -0.42` の根拠が §2.2 で breathing + car_horn only を base としていたため、 diverse subtype 下での SNR 10 dB borderline 6% が悪化 / 改善する可能性を Layer 4 replay と合わせて判断)

**関連**: Parent Issue [#338](https://github.com/Mega-Gorilla/livecap-cli/issues/338)、 上流依存 PR #347 (Issue #334 PR-4、 merged) + PR #348 (calibration corpus persistent dir、 merged)

### Changed

#### CI の FFmpeg 取得をバージョン固定 + チェックサム検証へ (Issue [#395])

PR #394 の Windows CI が **gyan.dev の 503 だけで 2 job 落ち**、既定の PR ゲートがブロックされた件への対応。調査の結果、単発の障害ではなく `.github/actions/setup-livecap-ffmpeg` の構造的な弱点だった。**`livecap_cli/` の production code は変更しない** (runtime 側の同種問題は [#398])。

- **Before**: Windows は `ffmpeg-release-essentials.zip` (ローリング URL) を `Invoke-WebRequest` 1 回。Linux は `curl -L` 1 回で `--fail` 無し。チェックサム検証もリトライも無く、**両 OS で違う FFmpeg バージョンをテスト**していた。キャッシュは 5 workflow 中 1 つだけ、しかも広い `restore-keys` 付きで、**バージョンを上げても古いキャッシュが復元され続ける**状態だった
- **After**: 両 OS とも **ffbinaries v6.1 に固定**し、アーカイブ 4 資産と展開後バイナリ 4 つの **SHA-256 を manifest に固定して検証**する (この manifest は後に [#398] で `livecap_cli/resources/ffmpeg_manifest.json` へ移り、runtime と共有する単一の正本になった)。リトライ・分類・検証は `setup_ffmpeg.py` の 1 実装を両 OS が共有し、**再試行するのは timeout / DNS・transport error / 408 / 429 / 5xx のみ** (指数バックオフ)。permanent 4xx とチェックサム不一致は fail loud。キャッシュ (`restore-keys` 無しの exact key) の所有者を action へ移し、**既定の PR ゲートも初めてキャッシュの恩恵を受ける**
- **検証はバイトの出所に依存しない** — cache restore / self-hosted の永続ディレクトリ / 新規ダウンロードのいずれでも、存在・SHA-256・実行可能性・期待バージョンを同じ経路で検査する。これにより **self-hosted に残った古いバイナリも検出され再取得される** (従来は workflow 側の「`ffmpeg.exe` があれば action をスキップ」ゲート 5 箇所により、一度も検証されなかった)
- **取得元と期待ハッシュは毎回記録される** — 固定 URL・期待 archive SHA-256・期待 binary SHA-256 は manifest 由来なので、ダウンロードが発生しない cache hit でもログと job summary に出る。取得経路は `source` (`download` / `cache` / `existing directory`)、キャッシュ状態は `cache-state` (`hit` / `miss` / `disabled`) として別々に報告する (`cache: 'false'` は miss ではない)
- **Migration**: workflow から `Check FFmpeg existence` ゲートと `Cache ffmpeg-bin` step を削除済み。新たに action を使う場合、ワークスペース外の永続ディレクトリを使う job では `cache: 'false'` を渡す (検証は引き続き実行される)
- **同一バージョン ≠ 同一挙動**: Linux と Windows のビルドは `configuration:` が異なる。固定はあくまで**比較可能性**を上げるものである

あわせて `core-tests-windows.yml` の `LIVECAP_FFMPEG_BIN: "${{ github.workspace }}\ffmpeg-bin"` を修正。YAML の二重引用符では `\f` が**フォームフィード (U+000C)** になるため、action が配置した直後にテストへ存在しないパスを渡していた。

#### CLI file mode に NoiseGate を接続 + `AudioPreprocessor` 注入点 (Issue [#366] Phase 3)

#366 の最終実装 Phase。NoiseGate parity には「VAD 判定と ASR 入力が同じ処理後音声を見る」ことが必要 (file pipeline は時刻範囲で元音声を slice するため、segmenter 前処理だけでは VAD 判定のみ処理後になる)。

- **Before**: file mode では `--noise-gate` 系 6 option を警告して無視
- **After**: `--noise-gate` 指定時、NoiseGate が **VAD 前処理として音声全体へ 1 回**適用され、VAD 判定・EnergyGate・ASR 入力の**すべてが処理後音声**を見る (realtime と同一の意味論)。有効化時は resolved 値 (open/close threshold・floor・attack/release) を stderr 表示。**既定 (off) のため未指定ユーザーの挙動は不変**
- **Migration**: 従来どおり無加工で処理する場合は `--noise-gate` を指定しない
- **実装**: `FileTranscriptionPipeline(audio_preprocessor=...)` 注入点を新設 (`AudioPreprocessor = Callable[[np.ndarray, int], np.ndarray]`、公開 export)。戻り値契約は fail-fast (`np.ndarray` / 1 次元 / shape・dtype とも入力と同一 = float32)、preprocessor の例外・契約違反は **file-level failure** (#362 経路で `process_files` が failed result へ変換)。CLI は **per-file factory** (ファイル毎に新 NoiseGate 生成) で `process_files` のファイル間・例外後の状態非共有を構造的に保証 — pipeline は `reset()` の暗黙契約を持たない
- **Tests**: 注入点契約 11 件 (厳密 1 回 / segmenter・ASR 同一配列 / identity 等価 / fail-fast 5 種 / file-level failure) + NoiseGate の全配列一括処理と chunk 逐次処理の bit-identical 等価性 + CLI 8 件 (per-file 生成 / 状態隔離 / **層の合成**: 静音 + hard-mute + EnergyGate で ASR 未呼出 / resolved 値ログ)。realtime NoiseGate suite は無改修 green (realtime 経路のコード変更なし)

#### CLI file mode で EnergyGate・confidence filter を既定有効化 + `SegmentTranscriber` 契約拡張 (Issue [#366] Phase 2)

Phase 1 (VAD 分割) に続き、EnergyGate と confidence filter を file mode に接続。realtime と同一の判定式を共有 module (`should_drop_low_energy` / `apply_filter`) で使う。

**1. `SegmentTranscriber` 契約拡張 (公開契約)**

- **Before**: `Callable[[np.ndarray, int], str]` — str のみ
- **After**: `str | SegmentOutcome` を受理。**str 返却は意味不変で継続受理** (既存 caller は無改修で動作)。`SegmentOutcome(text, drop_reason, asr_called)` は caller 側 filter の判定結果を運び、reason 別統計が `metadata["drop_counts"]` (新設、常時格納) に集計される。`asr_calls` は「engine を実際に呼んだ数」の意味を厳密化 — EnergyGate drop (`asr_called=False`) は数えず、confidence reject / engine empty は数える。drop は `empty_results` / `asr_errors` に混ざらない。全 segment drop でも `success=True` (全滅判定 #362 は不変 — 「gate drop + 残り全例外」は正しく `success=False`)
- **Migration**: 統計が必要な caller のみ `SegmentOutcome` へ移行。`drop_reason` の語彙は realtime と共通の `REASON_*` 定数

**2. file mode で filter 群が既定実効 (CLI)**

- **Before**: `--engine-min-rms` 系 / `--confidence-filter` / `LIVECAP_CONFIDENCE_FILTER` は file mode で warning 付き無視
- **After**:
  - **EnergyGate** (既定 `-45.0` dBFS) が VAD 分割された**各 segment 単位**で実効 — threshold 未満は **`engine.transcribe()` を呼ばずに** drop (`--engine-min-rms off` で無効化)。判定式は realtime と単一の `should_drop_low_energy` (strict `<`、equality pass、`-inf` は energy 計算ごと skip)
  - **confidence filter** (既定 `on`) が ASR 後に実効 — mode 3 種 + `LIVECAP_CONFIDENCE_FILTER` env の precedence を realtime と共有。observe log の `source_id` は入力 file path (複数ファイルの区別が可能)
  - drop があれば stderr に `Dropped segments: <reason>=<count>, ...` を表示
  - `LIVECAP_CONFIDENCE_FILTER` の file mode warning (#363) を撤去 (env が実効になったため)
- **Migration**: 従来の file mode 挙動 (filter なし) が必要な場合は `--engine-min-rms off --confidence-filter off`。無音区間が drop され字幕が減る場合は `Dropped segments:` の stderr 表示と `metadata["drop_counts"]` で診断可能。signal を出さない engine では confidence filter は fail-open (realtime と同一)

**3. Added**: `SegmentOutcome` (`livecap_cli` / `livecap_cli.transcription` から export、不変条件を `__post_init__` で強制) / `should_drop_low_energy` (`livecap_cli.audio` — realtime `StreamTranscriber._should_skip_low_energy` も同 helper へ委譲、挙動不変) / `metadata["drop_counts"]`

- **Tests**: helper 単体 7 件 (equality pass / `-inf` skip 未計算 / validation) / pipeline 正規化集計 12 件 (混在ケースの期待値固定、gate drop + 全例外 → `success=False`、legacy 空と structured 空の別勘定) / CLI 7 件 (無音 file で `transcribe` 未呼出 / per-segment 評価 / observe caplog + source_id / env precedence) / realtime `TestEnergyGate` は無改修 green (委譲の受け入れ条件)

#### CLI file mode の VAD 分割を既定有効化 + 注入 segmenter の空 fallback 廃止 (Issue [#366] Phase 1)

file mode は音声全体を 1 segment として ASR に渡しており長尺 file で品質・メモリに不利だった。`--vad` を file mode に接続する (#366 Phase 1)。

**1. file mode の VAD 分割 (CLI)**

- **Before**: `--vad` は realtime 専用 (file mode では warning 付き無視)。file transcription は常に音声全体を 1 segment として処理
- **After**: file mode は**既定 (`--vad auto`) で VAD により音声区間を分割**してから ASR へ渡す (resolved language (#365) の preset を適用)。音声セグメント 0 件の場合は **exit 0・出力は空** (`-o` 時は空 SRT) + stderr に `No speech segments detected.`。新設の **`--vad off`** で従来の全音声 1 segment 処理へ opt-out。`--realtime --vad off` は**モデルロード前に明確なエラー** (realtime は VAD 必須 — VAD なし realtime は別機能)
- **Migration**: 従来の file mode 挙動が必要な場合は `--vad off` を指定。無音 file が「空 SRT + exit 0」になる点に依存する script は `segmentation_empty` metadata / stderr 表示で判別可能

**2. `FileTranscriptionPipeline` の注入 segmenter 空 fallback 廃止 (公開契約)**

- **Before**: 注入 segmenter が `[]` を返すと**全音声 1 segment へ fallback** — VAD が正しく無音判定した場合ほど全音声が ASR へ流れ hallucination を招く逆転があった
- **After**: 注入 segmenter の `[]` は「音声セグメントなし」= **ASR 呼び出しゼロ** (`success=True` / `subtitles=[]` / metadata `segmentation_empty=True`)。`segmenter=None` (未注入) の全音声 fallback は従来どおり
- **Migration**: 全音声 1 segment 処理が必要な caller は `segmenter=None` で構築する

- **Tests**: `tests/vad/test_file_segmenter.py` 新規 7 件 (MockVADBackend で torch-free: final のみ採用 / 全無音 `[]` / finalize 回収 / reset lifecycle / 例外後回復 / 複数ファイル非持越)、`TestSegmentationEmpty` (pipeline 契約 3 件)、`TestVadFileMode` (CLI 7 件: off / no-speech semantics / realtime off 拒否 / preset 連携)、実 Silero integration 3 件
- **Docs**: `docs/reference/cli.md` (--vad 行 / file mode note / realtime-only 一覧から --vad 除去)、`docs/reference/api.md` (segmenter 意味論 + `VADFileSegmenter`)

#### 言語データ正本の一元化 + `supported_languages` の不変化 + `cli_default_language` 改名 (Issue [#230])

engine の対応言語データが metadata と adapter に二重定義されており、[#365] で `EngineMetadata.resolve_language()` が CLI の受理判定の正本になったことで、乖離が「対応言語の誤拒否 / 非対応言語の誤受理」という correctness バグに昇格していた。また `EngineMetadata.get()` が内部 `EngineInfo` を直接返すため、外部の list 変更が受理判定を書き換え可能だった (実証済み)。

- **Before**:
  - canary / voxtral / reazonspeech / parakeet の `get_supported_languages()` と metadata が独立 hardcode (一致は偶然維持)。qwen3asr は engine 内こそ #229 で単一化済みだが metadata の 30 言語は独立 hardcode
  - `EngineInfo.supported_languages` は mutable `List[str]` — `EngineMetadata.get("canary").supported_languages.append("ja")` の後 `resolve_language("canary", "ja")` が誤受理
  - field 名 `default_language` (constructor default と混同しやすい)
- **After**:
  - **言語データの正本を engine 特性別に一元化**: canary / voxtral / reazonspeech / parakeet(_ja) は `EngineMetadata` 正本 (adapter は defensive copy を返す) / whispers2t は `whisper_languages.py` 正本 (従来どおり) / qwen3asr は新設 `qwen3asr_languages.py` (data-only) 正本で metadata と adapter の両方が派生。全 engine の対応言語の**集合・順序は不変**
  - `EngineInfo.supported_languages` を `__post_init__` で **tuple 化** (構築時は list 可) + **`EngineInfo` 自体を frozen dataclass 化** (field 再代入 `info.supported_languages += (...)` の経路も封鎖)。qwen3asr の正本 map は **`MappingProxyType`** (item 代入で adapter だけが新言語を受理する split-brain を防止)、`SUPPORTED_LANGUAGES` class 属性も tuple。`EngineFactory.get_engine_info()` の `supported_languages` は毎回独立した list copy
  - `EngineInfo.default_language` → **`cli_default_language`** に改名 (未リリース field のため単純 rename)。CLI の未指定時ポリシーであり **constructor default との一致は要求しない**ことを明文化 (qwen3asr は意図的に constructor=auto / CLI=ja — PR-A.5.2)
- **Migration**: `supported_languages` を list として mutate していた外部 code は copy を取ること (`list(info.supported_languages)`)。`get_engine_info()` の戻り値は従来どおり `List[str]` (ただし独立 copy)。`default_language` 参照は `cli_default_language` へ (同一 Unreleased 内の rename)
- **Tests**: `tests/core/engines/test_language_authority.py` 新規 (adapter↔metadata 一致 ×5 / data module 同源 ×2 / qwen3asr 意図的差 pin / auto adapter のモデルロードなし変換)、`TestLanguageDataAuthority` (正本派生・golden 順序・tuple 不変・factory copy 独立性)
- **Docs**: `adding-an-engine.md` に「言語データの正本規約」(hardcode 禁止・パターン表・新 engine 追加時のテスト行追加) を新設

#### CLI `--language` を全 engine に適用 + 言語解決の一元化 (Issue [#365])

従来 `--language` は **qwen3asr にのみ** routing され、whispers2t は常に constructor default (`ja`) で動作 (英語音声を ja 設定で認識する silent 精度劣化)、canary は `en` 固定、voxtral は明示指定不能、単一言語 engine (reazonspeech/parakeet/parakeet_ja) は不一致を silent 無視していた。#363 で file 経路が実働化したため実害化していた。

- **Before**:
  - parser default `--language ja`。qwen3asr 以外の engine へは渡らない
  - 非対応言語・不正コード・単一言語 engine の不一致は silent 無視 (誤設定のまま認識)
  - VAD preset は常に args 生値 (実質 ja) で選択
- **After**:
  - parser default を **未指定 sentinel (None)** に変更し、`EngineMetadata.resolve_language(engine_id, requested)` が起動時に一度だけ engine 別へ解決。**未指定時の実効言語は全 engine 現状維持** (whispers2t/qwen3asr/reazonspeech/parakeet_ja: ja、canary/parakeet: en、voxtral: auto)
  - 明示指定は BCP-47 → primary language subtag 正規化 (`ja-JP` → `ja`、`yue-HK` → `yue`) + `supported_languages` 検証。非対応言語・不正形式コード・auto 非対応 engine への `auto`・単一言語 engine の不一致は**モデルロード前に exit 1** (silent fallback しない)
  - resolved 値を engine kwargs (whispers2t/canary/voxtral/qwen3asr へ routing)・VAD preset・翻訳 `source_lang`・起動ログ (`Language: requested=..., resolved=...`) へ一貫配布
  - `--translate` 併用時は resolved が具体言語であることを必須化 (voxtral の未指定/auto は拒否。default whispers2t + 翻訳は resolved=ja のため従来どおり動作)
- **Migration**: 言語未指定の利用は無影響。従来 silent 無視されていた不正組合せ (`--language ja --engine canary` / `--language en --engine reazonspeech` / `--language auto --engine whispers2t` 等) はエラーになるため、正しい言語または対応 engine を指定すること。realtime の VAD preset が engine 別 resolved 値で選択される (canary 未指定時は en preset 等。preset 不在言語は default VAD へ fallback)
- **Tests**: `TestLanguageResolutionMetadata`/`TestResolveLanguage` (metadata、engine 別解決表を固定)、`TestLanguageResolutionE2E` (CLI file 経路 9 件: routing / fail-fast 4 種 / 翻訳併用 2 種 / 起動ログ)、pin テスト 2 件を新契約へ書き換え (誤った「whisper は VAD 経由」docstring を是正)、VAD auto-fallback (integration)

#### CLI `transcribe <file>`: 出力・翻訳失敗の意味論を仕様化 (Issue [#363])

CLI file 経路の復旧 (上記 Fixed) に伴い、出力先と失敗時の観測可能な挙動を確定。旧実装は全滅していたため実利用上の互換影響はないが、pre-1.0 policy に従い Before/After を記録する。

- **Before** (旧コードの意図した挙動、実際には未動作):
  - `-o` はファイルモード時必須扱い (docs 記載)。未指定時は `[start - end] text` 形式の独自行を stdout 出力
  - 翻訳失敗時の仕様なし
  - realtime 専用オプションをファイルモードで指定しても silent no-op
- **After**:
  - `-o` は任意。**未指定時は SRT content を stdout へ出力** (進捗・警告は stderr のみ — pipe/リダイレクトで安全)
  - `--translate` 指定で**全 segment の翻訳が失敗** → exit 1 + 出力ファイル非生成 (**原文 SRT への silent fallback はしない**)。**一部失敗** → 翻訳成功 segment のみ出力 + 件数付き stderr warning (segment index は元の値を維持)
  - realtime 専用オプション (`--vad` / noise-gate 系 / EnergyGate 系 / transient 系 / `--confidence-filter` / `--mic`) を**既定値から変更して**指定した場合、および `LIVECAP_CONFIDENCE_FILTER` 設定時に stderr warning (「既定値と同値の明示指定」の検出は [#366])
  - file mode は VAD 分割なしで音声全体を 1 segment として処理 (pipeline の従来契約どおり。segmenter 接続は [#366])
- **Migration**: `-o` 必須を前提にした script はそのまま動作。stdout を parse していた script は SRT 形式に更新すること (旧形式はそもそも全滅バグで出力されなかったため実影響なし)

#### FileTranscriptionPipeline: 全 ASR segment 例外を `success=False` に格上げ (Issue [#362])

livecap-gui v3.1.0 の「常に空 SRT」障害 ([gui#392](https://github.com/Mega-Gorilla/livecap-gui/issues/392)) の増幅要因を修正。`FileTranscriptionPipeline._transcribe_segments` は segment 単位の transcriber 例外を fail-soft で握り潰すため、**全 segment が例外** (契約不整合・モデル破損等の異常状態) でも `FileProcessingResult(success=True)` + 0 byte SRT を正常出力し、障害が caller (GUI) から不可視だった。segment 単位 fail-soft は維持しつつ、「全滅」と「部分失敗」を区別する集計を導入。

- **Before**:
  - 全 ASR 呼び出しが例外でも `success=True` で完走し、空 SRT を書き出す (既存 SRT があれば空で上書き)
  - `metadata` に ASR 呼び出し件数の内訳なし
  - `process_files` の `error_callback` は `process_file` が例外を**送出**した場合のみ発火
- **After**:
  - ASR 呼び出しが 1 件以上かつ全件例外 → **`success=False`**、`error` に `"All N ASR segment calls failed; first error: <Type>: <msg>"`、**`output_path=None` で SRT を新規作成も上書きもしない**
  - `metadata` に `asr_calls` / `asr_errors` / `empty_results` を**成功・失敗を問わず常時**格納 (caller の警告表示・診断用)
  - `process_files` は `success=False` **返却**時も `error_callback(error_msg, None)` を発火 (送出経路と契約を一貫)
  - 部分失敗 (例外 < 全件) と正常な空 (無音・全件正常空認識・ASR 呼び出し 0 件) は従来どおり `success=True` (gui#392 レビュー合意: 件数ヒューリスティック誤発火の回避)
  - `FileTranscriptionCancelled` は従来どおり集計せず再送出
- **Migration**: `result.success` のみを見る caller は挙動改善のみ (全滅が正しく失敗として見える)。全滅時の 0 byte SRT 生成に依存する caller は想定なし。`error_callback` を「送出例外専用」として使っていた caller は `exc=None` のケース (返却された failure) を許容すること。
- **Tests**: `tests/core/transcription/test_file_pipeline_outcome.py` 新規 12 件 (全滅/既存 SRT 保護/部分失敗/3 種混在/全件正常空/ASR 0 件/Cancelled 再送出/metadata 常時格納/process_files callback 契約)、engine/torch/FFmpeg 不要
- **Docs**: `docs/reference/api.md` — FileTranscriptionPipeline 使用例の `engine.transcribe(audio, sr)[0]` を `.text` に修正 (#314 以降 `[0]` は TypeError、gui#392 と同種の誤実装を誘発するサンプルだった) + 「失敗の意味論」節を新設。`docs/reference/feature-inventory.md` — `error_callback` の `Exception | None` 契約・metadata 件数内訳を反映、実在しない `config=` 引数をサンプルから除去

**関連**: [Issue #362](https://github.com/Mega-Gorilla/livecap-cli/issues/362) / gui#392 (GUI 側 hotfix v3.1.1) / [#314] (`TranscriptionResult.__iter__` 削除 — 障害トリガー) / [#363] (CLI file 経路の API 乖離 — 別 issue、本修正の scope 外)

#### Calibration harness: `parse_observe.py` が interim entry を default 除外 (Issue [#351] PR 2)

[Issue #351](https://github.com/Mega-Gorilla/livecap-cli/issues/351) の PR 2。 PR 1 ([#352] で observe log entry に `is_interim` field 追加) の **consumer 側対応**。 `benchmarks/confidence_calibration/parse_observe.py` が `is_interim` field を認識し、 threshold sweep から **interim path 由来 entry を default で除外** (final のみで calibration) するよう変更。 dev-only、 production runtime (`livecap_cli/`) には影響なし。

**背景**: `parse_observe_log()` は従来、 同一発話の interim n 回 + final 1 回を独立 sample として counting していた。 interim は蓄積途中の temporary UI feedback で threshold tuning の「正解」ではないため、 final のみを使うのが正しい。 Layer 4 replay pipeline ([Issue #338] Task) で production observe log を calibration に使う前提整備。

- **`parse_observe.py:parse_observe_log()` の default 挙動変更**:
  - `include_interim: bool = False` param 追加、 default で `is_interim=True` entry を除外
  - **occurrence 採番の critical fix**: interim 除外は occurrence counter increment の**前**に行う。 counter を先に進めてから除外すると final の occurrence が `1, 3, 5...` と skip 交じりになり、 user の `occurrence_index` label (final 前提) が match しなくなる
  - `skipped_interim` counter + `logger.info` 報告 (observability、 silent drop 回避)
  - sample metadata に `is_interim` field 追加 (`--include-interim` 時の下流分析用)
- **`LogEntry` / `parse_log_line()`**: `is_interim` field 追加、 他 optional field と同じ `.get(..., default)` pattern で読む (key 欠落は final 扱い)
- **`main()` CLI**: `--include-interim` flag で opt-in (interim hallucination filter tuning 用 advanced analysis)

**Migration (advanced analysis 用途)**:

```bash
# default: interim 除外、 final のみで sweep (推奨)
uv run python -m benchmarks.confidence_calibration.parse_observe \
    --log observe.log --labels labels.jsonl --engine reazonspeech --signal avg_logprob

# opt-in: interim も含める (occurrence_index label は final-only 前提だと mismatch、 caveat あり)
uv run python -m benchmarks.confidence_calibration.parse_observe \
    --log observe.log --labels labels.jsonl --engine reazonspeech --signal avg_logprob \
    --include-interim
```

**Tests**: 8 新 test (`TestIsInterimConsumer`) 追加 — `parse_log_line` の is_interim 読取 (True/False/key 欠落→False の parse robustness)、 default final-only 除外、 `--include-interim` opt-in で全 entry 保持、 **occurrence 採番が final entries 内で連続 (interim skip 後 0, 1)** の critical verify、 skipped_interim logging、 sample metadata、 CLI flag E2E。 全 53 pass in `test_parse_observe.py` (元 45 + 新規 8)、 退行ゼロ。 `_make_log_line` helper は PR 1 schema に合わせ `is_interim` を常時 emit。

**関連**: 上流 PR 1 ([#352]、 CLI schema 追加)、 [Issue #338](https://github.com/Mega-Gorilla/livecap-cli/issues/338) Layer 4 replay pipeline の前提、 [Issue #334](https://github.com/Mega-Gorilla/livecap-cli/issues/334) Finding F6 (Qwen3-ASR auto-detect) の calibration accuracy 向上。

#### Confidence filter 既定閾値を Phase 2 report 反映で更新 (Issue [#334] PR-4)

[Phase 2 report](docs/research/calibration-japan-engines-phase2-2026-07.md) で Layer 2 (ESC-50/MUSAN hard negative) + Layer 3 (SNR-mixed noisy_speech) 込みの augmented corpus 1375 sample を用いて 5 engine を再 calibration した結果、 **Pareto gate 「`clean_frr ≤ 3%` かつ `noisy_frr(SNR≥5) ≤ 5%` かつ known probe reject」適用値** を新 default として採用。 [Issue #334](https://github.com/Mega-Gorilla/livecap-cli/issues/334) audit の **main deliverable**。 主要効果: **ReazonSpeech で現 default `-0.2` の FRR 42.5% 実害を新 default `-0.40` で 5.4% に改善** (Phase 2 report §2.1 実測)。 全 engine で Pareto gate を実測 evidence に基づき採用。

- **`livecap_cli/transcription/confidence_filter.py:FilterConfig` の default 4 値を Phase 2 recommended に変更**:
  - `no_speech_threshold`: `0.5 → 0.71` (WhisperS2T、 Pareto relaxed_B、 Whisper 公式 0.6 近傍、 Phase 2 report §2.3)
  - `token_conf_threshold`: `0.005 → 0.001` (Parakeet_ja Pareto strict pass、 F1=0.961、 false reject 39→11 = 72% 削減 with non_speech recall 97.9%→94.1% の trade-off、 Phase 2 report §2.4、 collateral: Parakeet_en / Canary にも適用、 いずれも speech margin 十分)
  - `avg_logprob_thresholds["reazonspeech"]`: `-0.2 → -0.40` (Pareto relaxed_B、 現 default -0.20 の FRR 42.5% 実害を 5.4% に改善、 Phase 2 report §2.1)
  - `avg_logprob_thresholds["qwen3-asr"]`: `-0.3 → -0.42` (Pareto relaxed_C、 JA/EN 両方に適用、 EN は Phase 1 probe で safety verify 済、 Phase 2 report §2.2、 SNR 10 borderline は Layer 4 で再確認予定)
  - `avg_logprob_threshold`: **不変** (`-1.0`、 Voxtral 用 global fallback として維持)

**Migration**:

```python
# 旧 default (PR-A.0/A.5.X で verify された値)
FilterConfig(
    no_speech_threshold=0.5,
    token_conf_threshold=0.005,
    avg_logprob_threshold=-1.0,  # 変更なし
    avg_logprob_thresholds={"reazonspeech": -0.2, "qwen3-asr": -0.3},
)

# 新 default (Unreleased / Issue #334 PR-4 以降、 Phase 2 report Pareto gate 適用値)
FilterConfig(
    no_speech_threshold=0.71,
    token_conf_threshold=0.001,
    avg_logprob_threshold=-1.0,  # 変更なし
    avg_logprob_thresholds={"reazonspeech": -0.40, "qwen3-asr": -0.42},
)

# 旧挙動を維持したい場合 (明示 override)
FilterConfig(
    no_speech_threshold=0.5,
    token_conf_threshold=0.005,
    avg_logprob_thresholds={"reazonspeech": -0.2, "qwen3-asr": -0.3},
)
```

**Trade-off** (Phase 2 report §2.4 / §4.1):

- **Parakeet_ja** (`0.005 → 0.001`): false reject **39 → 11 (72% 削減)** の引き換えに non_speech recall 97.9% → 94.1% (-3.9pt)。 clean speech user 体感で顕著な改善、 non_speech 側の recall 低下は許容範囲判定。
- **Parakeet_en / Canary** (collateral、 Phase 2 未 calibrate): scalar `token_conf_threshold` 変更で共用のため collateral 適用、 speech `token_confidence_mean` 実測 mean が Parakeet_en 0.2452 / Canary 0.0724 と新 threshold `0.001` から margin 十分 (245× / 72×)、 実害なし想定。
- **Qwen3-ASR EN** (collateral、 単一 dict key で JA/EN 共用): EN Phase 1 probe (speech `-0.05` / applause `-1.08`) で `-0.42` が safety verify 済 (speech pass 維持 / applause reject 維持)。 engine+language subkey 化は [Finding F9] の対応、 別 PR で。

**Test 更新**:

- `test_default_thresholds_from_pr_a0_verify` を **削除**、 新規 `test_default_thresholds_from_phase2_report` を追加 (Phase 2 report §4.1 の Pareto gate 適用値を pin)。
- `TestRegressionPrA0Values.test_pr_b_corpus_classification` の WhisperS2T 2 rows (`0.635` desk_tap / `0.662` applause) で `expected_reject: True → False` に flip + Pareto trade-off コメント (Phase 2 report §2.3 relaxed_B の直接 evidence として code 内に記録)。
- 12+ test で threshold literal (`0.5` / `0.005` / `-0.2` / `-0.3`) を新値に update。
- 既存 65 test 全 retain、 全 pass。

**Documentation**:

- `docs/reference/api.md` / `cli.md` の閾値表 + code sample 更新
- `docs/audio-filter-reference.md` の signal-family table + banner example + prose margin 説明更新
- `livecap_cli/cli.py` argparse help text 更新
- `livecap_cli/engines/base_engine.py` / `reazonspeech_engine.py` / `qwen3asr_engine.py` docstring 更新

**関連**:

- Historical smoke report (`reazonspeech-confidence-smoke-2026-06-11.md` / `qwen3asr-confidence-smoke-2026-06-12.md` / Phase 1 report) は frozen artifact として **不変**、 最新値は Phase 2 report で記録。
- livecap-gui 側 release note で閾値変更を告知 (本 PR merge 後の別 timeline)。
- Voxtral の `avg_logprob_threshold = -1.0` global fallback は Phase 2 未 calibrate のため不変。 dict 化 schema change は [Finding F9] の対応で別 PR。

#### Engine API contract — fallback adapter cleanup (Issue [#321] PR #3、3-PR 系列完成)

[Issue #321](https://github.com/Mega-Gorilla/livecap-cli/issues/321) の
**3-PR 系列 (PR #1 wording + Canary `beam_size` / PR #2 NeMo fallback chain /
PR #3 本) を完成**させる最終 PR。PR #320 (qwen3asr) / PR #322 / PR #323 で
確立した「framework contract を trust、silent degradation より hard fail」
方針を `TranscriptionEngine` Protocol contract に最終適用。

##### Engine I/O 契約の明文化

`TranscriptionEngine` Protocol (`livecap_cli/transcription/stream.py`) の
docstring を厳格化:

- 実装者は `transcribe()` から **必ず `TranscriptionResult` を返すこと**
- tuple / dict / str / None は契約違反、`apply_filter` (StreamTranscriber
  経路) 側で `AttributeError` が caller に propagate して fail-fast
- 別 path の `SharedEngineManager._process_request` も bare attribute
  access に整理したが、module-level の `except Exception` で contract
  violation も "request failure" として log + `None` 返却 (orphan code、
  Issue [#326] で本 file 削除予定のため fail-fast 化は scope 外)
- pre-1.0 cleanup の方針を明示、precedent (PR #320/#322/#323) を docstring
  で reference

##### `apply_filter()` — `hasattr` legacy guard 削除

`confidence_filter.py:386-390` の `hasattr(result, "engine_confidence")`
guard を削除、bare attribute access に統一:

- **Before**: `if not hasattr(result, "engine_confidence"): return result`
  (旧 mock の tuple 返却互換)
- **After**: bare `result.engine_confidence` access、契約違反時は
  `AttributeError` propagate
- **Audit verify**: 全 test MockEngines (6 件) + 全 `apply_filter()` test
  caller が既に `TranscriptionResult` 返却済を grep + read で確認、guard は
  dead code

##### `SharedEngineManager._process_request` — tuple/dict adapter 削除

`shared_engine_manager.py:437-490` の `hasattr` tuple branch + `isinstance(result, dict)` branch を削除、direct attribute access only に rewrite:

- **Before**: `hasattr(result, 'text')` 主、tuple `(text, conf)` fallback、
  dict `{"text": ..., "confidence": ...}` fallback の 3 path
- **After**: `result.text` / `result.confidence` の bare access。
  ただし method-level の `except Exception as e` は維持されているため、
  契約違反 (`AttributeError`) は **caller に propagate せず** "request
  failure" として log + `None` 返却。`apply_filter` 側 (fail-fast) と
  挙動が異なる点に注意
- **Caveat**: `SharedEngineManager` 自体は production / tests から完全に
  未参照の **orphan code** (`__all__` にも非 export)。本 PR では契約
  整合のみ実施、`except Exception` を狭めて fail-fast 化することは
  scope 外。**ファイル削除自体は [Issue #326]** で対応予定

##### Stale docstring 整理

- `tests/transcription/test_stream.py::FilteringMockEngine` docstring の
  「`MockEngine` は legacy tuple を返す」記述を削除 (実態は `TranscriptionResult`
  返却、stale comment)
- `CLAUDE.md:78` の TranscriptionEngine Protocol 例を `Tuple[str, float]` →
  `TranscriptionResult` に修正 (AI agent guidance と code 契約の乖離を解消)

##### Audit findings (本 PR scope を絞った根拠)

| Audit item | 結果 |
|---|---|
| 全 test MockEngines (6 件) | ✅ 既に `TranscriptionResult` 返却済 |
| `apply_filter()` 全 test caller | ✅ 既に `TranscriptionResult` 渡し済 |
| `SharedEngineManager` の production/test caller | ✅ **0 件** (orphan code) |
| `CLAUDE.md:78` 旧 `Tuple[str, float]` 型 | ⚠ stale、本 PR で修正 |
| `FilteringMockEngine` docstring | ⚠ stale、本 PR で修正 |

→ test fixture 統一 phase は不要、PR scope は contract tightening +
stale comment 整理に絞れた。

##### Migration

- **既存 production engine (WhisperS2T/Parakeet/Voxtral/Canary/ReazonSpeech/
  qwen3asr) は影響なし**: 既に `TranscriptionResult` を返却済
- **既存 test mocks も影響なし**: 既に `TranscriptionResult` 返却 (audit verify)
- **第三者 plugin / custom engine 実装者** (もしいれば): `transcribe()` の
  戻り値を `TranscriptionResult` に統一する必要あり。tuple/dict/str/None
  返却は `AttributeError` で fail-fast

##### Tests (退行ゼロ、712 baseline 維持)

- `tests/transcription/test_confidence_filter.py`: 全 pass (`apply_filter`
  caller 全て `TranscriptionResult`)
- `tests/transcription/test_stream.py`: 全 pass (MockEngine / FilteringMockEngine
  共に `TranscriptionResult`)
- Full local regression: 712 passed

##### Out of scope (本 PR では行わない)

- `livecap_cli/engines/shared_engine_manager.py` orphan file 自体の削除 →
  別 issue "audit unused engine infrastructure" (`SharedEngineManager` +
  `TranscriptionRequest` + `ProgressCallback` の orphan 確認 + 削除提案)
- `BaseEngine.__init__` の `**kwargs` swallowing 削除 → 別 issue
- 他 engine の `__init__` `**kwargs` 削除 → 別 issue
- `docs/planning/archive/*.md` の旧型 reference → archive 性質上 触らない

##### Issue #321 完成宣言

| PR | scope | 状況 |
|---|---|---|
| **PR #1 ([#322])** | wording cleanup + Canary `beam_size` fail-fast | ✅ merged |
| **PR #2 ([#323])** | Canary/Parakeet NeMo fallback chain | ✅ merged |
| **PR #3 (本)** | API contract cleanup | ✅ |

3-PR 系列完成、本 PR merge 後に Issue #321 を close。

#### Engine confidence — Canary / Parakeet NeMo fallback chain cleanup (Issue [#321] PR #2)

`CanaryEngine._configure_decoding_with_confidence` と
`ParakeetEngine._configure_decoding_with_confidence` から、`token_confidence`
取得 path (Path 1) が失敗した場合に **token_confidence なしで継続する silent
degradation を生む fallback chain** を削除して fail-fast 化。加えて
`ParakeetEngine._transcribe` の `return_hypotheses=True` TypeError silent
fallback も削除。

PR #320 (qwen3asr) / PR #322 (Canary `**kwargs`) の precedent と整合、
「framework contract を trust、silent degradation より hard fail」方針を
NeMo 系 engine にも適用。

##### Before / After

- **Before**:
  - Canary: Path 1 (greedy + confidence_cfg) → Path 2 (greedy のみ、confidence
    なし) → Path 3 (argument-less)。Path 1 失敗時に silent fallback、
    `token_confidence_mean` が None になり confidence filter は pass-through に degrade
  - Parakeet: Path 1 (Hybrid CTC) → Path 1.5 (Pure RNNT/TDT) → Path 2
    (strategy-only) → Path 3 (argument-less)。同様に Path 1/1.5 失敗時に
    silent fallback
  - `ParakeetEngine._transcribe`: `transcribe(return_hypotheses=True)` が
    TypeError なら kwarg なしで再 transcribe、`engine_confidence` 全 None
    で filter が pass-through に degrade
- **After**:
  - Canary: Path 1 のみ、bare 呼出 (try/except 削除)。失敗時は NeMo native
    `TypeError` / `ValueError` 等が propagate
  - Parakeet: Path 1 (Hybrid model-family dispatch、temporary fallback で
    Path 1.5 へ) + Path 1.5 (Pure RNNT/TDT primary、bare 呼出)。Path 1.5
    失敗時は NeMo error が propagate
  - `ParakeetEngine._transcribe`: bare 呼出、`return_hypotheses=True` 失敗時
    は TypeError が propagate (`nemo-toolkit>=2.3,<2.5` の supported range で
    公式安定 API)

##### Migration

- `nemo-toolkit>=2.3,<2.5` (lockfile `2.3.0`) の supported range では既存
  挙動と完全に同じ (Path 1 / Path 1.5 / `return_hypotheses=True` は常に成功)
- supported range 外の旧 nemo build を使う user は、Path 1 が拒否された時点で
  従来の silent fallback ではなく `TypeError`/`KeyError`/`ValueError` 等が
  直接 raise されるため、具体的な NeMo error message から nemo version を
  確認する actionable hint を得る
- Parakeet **Path 1 (Hybrid CTC)** と **Path 1.5 (Pure RNNT/TDT)** は
  **model-family dispatch** (hybrid vs pure decoder の正規 dispatch) として
  温存。legacy fallback ではないため reviewer 承認の上で温存

##### Verification (merge gate)

`tests/integration/engines/test_smoke_engines.py::test_token_confidence_populated`
で実機 GPU verify (RTX 4090 self-hosted runner):

| Case | Expected token_confidence_mean (probe baseline) |
|---|---|
| `canary_gpu_en` (LibriSpeech 英語) | > 0.05 (PR-A.4.2 で 0.0724) |
| `parakeet_gpu_en` (LibriSpeech 英語) | > 0.10 (PR-A.4.3 で 0.2452) |
| `parakeet_ja_gpu_ja` (jsut 日本語) | > 0.02 (PR-A.0 で 0.0504) |

新 test は `@pytest.mark.engine_smoke` で hosted CI から除外、self-hosted
GPU runner でのみ実行。失敗時は merge を blocking する design。

##### Out of scope (本 PR では行わない)

- `confidence_filter.py:386` `hasattr(result, "engine_confidence")` guard
  削除 — **Issue #321 PR #3** で `shared_engine_manager.py` tuple fallback
  + `TranscriptionEngine` Protocol cleanup とセットで扱う
- `BaseEngine.__init__` の `**kwargs` swallowing 削除 — 別 issue
- nemo-toolkit version の pin 化 / 上限拡大 — 別 issue (本 PR では現状の
  `>=2.3,<2.5` range を contract として扱う)

#### Engine confidence filter — qwen3asr support via wrapper bypass (Issue [#318] PR-A.5.2)

PR-A 系列の **7 engine 対応** を達成、Confidence Filter (Phase 1 Layer 5) を
最終形に到達させる PR。Phase 1 probe (Issue #318 で User 意向「EN/JP 両言語
対応出来なければ close」に対する go condition 達成) を受けて実装、両言語
verified で qwen3asr を追加対応。

##### Engine integration — wrapper bypass で avg_logprob 抽出

qwen3asr は qwen-asr wrapper が ``output_scores=True`` を渡さない設計の
ため confidence filter 非対応だったが、**`Qwen3ASRModel.model` (= 内部
``Qwen3ASRForConditionalGeneration``) を直接呼ぶ wrapper bypass** で対応:

- ``transcribe()`` を rewrite、``self.model.model.generate(output_scores=True,
  repetition_penalty=1.1, no_repeat_ngram_size=3)`` を直接呼出
- ``compute_transition_scores(normalize_logits=True)`` 経由で per-token
  logprob を取得、Voxtral PR-A.4.1 と完全同形の ``_extract_engine_confidence``
  helper で mean → ``EngineConfidence.avg_logprob`` populate
- ``_asr_language is None`` (auto-detect mode) は旧 wrapper.transcribe()
  path に fail-open (engine_confidence 全 None、filter pass-through)

##### `repetition_penalty=1.1 + no_repeat_ngram_size=3` で両言語 failure mode 解消

Phase 1 probe で確認した critical finding:

- **Japanese**: desk_tap の 256-token repetition loop ("うんうんうん...") を
  4-token "うん。" に短縮、avg_logprob -0.13 → -0.65、**margin -0.02 (逆転)
  → +0.27 (filter 可能)**
- **English**: applause の system prompt leak ("You are a speech
  recognition model.") を avg_logprob -0.04 → -1.08 に低下 + "You are an AI."
  に短縮、**margin -0.03 (逆転) → +0.21 (filter 可能)**

→ **両言語で同じ generation parameter** で対応可能、言語別実装不要。

##### Section 1 smoke (両言語 6 clip)、Phase 1 probe 値を上回る margin 確認

- **English**: speech -0.05、non-speech -0.71、**margin +0.65** (Phase 1
  probe +0.21 を大幅に上回る) → Case A
- **Japanese**: speech -0.12、non-speech -0.63、**margin +0.42** (Phase 1
  probe +0.27 を上回る) → Case A
- 両言語で threshold ``-0.3`` で 100% 分類成功

##### Section 2 (12 cell stream pipeline benchmark)

- **Hall.(pre) = 0% 全 cell** — qwen3asr は `repetition_penalty` 適用後
  本 corpus で hallucinate しない (Canary PR-A.4.2 と同 engine 固有
  fail-safe pattern)
- SR(post) = 100% real corpus 全 cell (legit speech は 1 件も drop されず)
- Latency 影響なし

##### Engine-specific threshold = `-0.3` (両言語 safe)

- `FilterConfig.avg_logprob_thresholds["qwen3-asr"] = -0.3` を default load
- dict key を ``"qwen3-asr"`` (ハイフン含む) にしているのは、
  ``_engine_id_from_name("Qwen3-ASR 0.6B")`` が ``"qwen3-asr"`` に normalize
  するため (PR-A.5.1 codex Point 1 の learning を pre-empt)
- Phase 4 unit test で **production display string** での threshold lookup
  を pin (6 件 + helper mapping 拡張で qwen3-asr 追加)

##### Migration

| Engine | Before | After |
|---|---|---|
| qwen3asr (language 指定あり) | fail-open (engine_confidence 全 None) | ``avg_logprob < -0.3`` で reject、`repetition_penalty=1.1 + no_repeat_ngram_size=3` で failure mode 解消 |
| qwen3asr (auto-detect mode) | (不変) | (不変、wrapper fallback で fail-open) |
| WhisperS2T / Parakeet (ja/en) / Voxtral / Canary / ReazonSpeech | (不変) | 退行ゼロ |

##### Caveats (production user 向け)

1. **WER 軽微退行 (LLM typical 0.5-1%)**: ``repetition_penalty=1.1 +
   no_repeat_ngram_size=3`` で稀に正常 token も抑制可能性。Voxtral
   PR-A.4.1 / Canary PR-A.4.2 と同 framing で filter benefit を優先。
   ``--confidence-filter off`` は **post-ASR reject のみ**無効化し、
   generation 側変更 (``repetition_penalty=1.1`` / ``no_repeat_ngram_size=3``)
   は固定で残る (Voxtral greedy / Canary greedy と同 design)
2. **多言語 verify (28+ 言語) は本 PR scope 外**: en/ja のみ verified、
   他言語は user feedback ベース (Voxtral / Canary と同 framing)
3. **`_asr_language is None`** (auto-detect mode) は fail-open、production
   user は ``--language en/ja/...`` を明示推奨
4. **wrapper internal attribute 依存**: ``self.model.model`` (=
   ``Qwen3ASRForConditionalGeneration``) の private structure に依存。
   qwen-asr が future update でこの構造を変更すると AttributeError で hard
   fail する (Voxtral PR-A.4.1 と同じく framework contract を trust する design)

##### Tests (新規 +20 件、合計 703 passed)

- `tests/core/engines/test_qwen3asr_confidence_extraction.py` (新規 14 件) —
  Voxtral pattern 流用、masking ロジック + tensor/numpy 互換 + Phase 1
  probe 値再現を pin
- `tests/transcription/test_confidence_filter.py` (+6 件) —
  `TestQwen3AsrEngineSpecificThreshold` + `TestEngineIdNormalization` に
  `"Qwen3-ASR 0.6B"` / `"Qwen3-ASR 1.7B"` display string pin
- `tests/transcription/test_stream.py` (+1 件) — banner test に
  ``qwen3-asr`` + ``-0.3`` assertion

##### Docs

- ``docs/research/qwen3asr-confidence-smoke-2026-06-12.md`` (新規 decision doc)
- ``docs/audio-filter-reference.md``: Engine support table を **7 engine
  対応**に拡大、Property table / Decision section / 完成サマリ / Comparison
  table の全 section で 6 → 7 engine 整合
- ``docs/reference/cli.md`` + ``docs/reference/api.md``: qwen3asr 行追加 +
  `avg_logprob_thresholds` 例に qwen3-asr 追記
- ``base_engine.py`` ``EngineConfidence`` docstring の populate status table
  に qwen3asr 追加

##### Out of scope (本 PR では行わない)

- 多言語 verify (en/ja 以外の 28+ 言語) — user feedback ベース
- ``Qwen3ASRModel.transcribe()`` の auto-detect mode (``force_language=None``)
  — scope 外、必要なら follow-up
- 長尺音声 (>50s) の auto-chunking — stream pipeline では不要 (VAD segments
  典型 <30s)
- CLI flag ``--qwen3asr-repetition-penalty`` 等の generation param 制御 —
  不要 (hardcoded、Voxtral / Canary と整合)

#### Engine confidence filter — ReazonSpeech support + engine-specific threshold (Issue [#317] PR-A.5.1)

PR-A 系列の **6 engine 対応** を達成、Confidence Filter (Phase 1 Layer 5) を
完成形に到達させる PR。reviewer feedback (Issue #317) で 7 件の critical
指摘を受領、本 PR で全反映。

##### Production bug fix (reviewer Point 6、CRITICAL)

`reazonspeech_engine.py:430` の ``text, confidence = self._transcribe_single(...)``
unpack が PR #314 で削除済 ``TranscriptionResult.__iter__`` で TypeError を
投げるが、外側の ``except Exception`` で silent swallow + ``continue`` して
いたため、**長尺音声 (>30s、``auto_split_duration`` 経路) で全 segment が
silently dropped されていた production critical bug** を Phase 1 で独立
commit で修正。3 件 regression test (mock-based) で pin。

旧挙動: 30s 超え audio → 空 transcription (production reach 中)
新挙動: 各 segment の text が正しく combined_text に積まれる

これは breaking change ではなく **production bug 修正**。

##### ReazonSpeech confidence filter integration (Issue [#317] core)

旧 docs ([Issue #308 close 時点]) では「sherpa-onnx Python bindings に
per-token score API なし、PR #2897 closed/not-merged、Python 未対応」を
理由に **PR-A.5 candidate (heavy refactor)** としていたが、本 PR plan
段階の実機 probe で **sherpa-onnx 1.12.39 で
``OfflineRecognitionResult.ys_log_probs`` が既に exposed されている**こと
が判明、standard integration work で対応:

- **Before**: ReazonSpeech の ``transcribe()`` は ``engine_confidence =
  EngineConfidence()`` (全 None) で fail-open
- **After**:
  - ``reazonspeech_engine.py`` に module-level ``_extract_engine_confidence(result)``
    helper 追加 (Canary / Voxtral pattern 流用、実 sherpa-onnx 不要に unit
    test で schema pin)
  - ``_transcribe_single()`` で sherpa-onnx ``result.ys_log_probs`` を抽出、
    mean を **``EngineConfidence.avg_logprob`` field** に populate
    (Voxtral と同 semantics、reviewer Point 1/2 で確定設計)
  - ``raw["ys_log_probs_mean"]`` + ``raw["ys_log_probs_n"]`` に metadata 保存
  - ``_transcribe_with_split()`` で segment 別 engine_confidence を
    **weighted mean** で aggregate (token 数 weight)

reviewer Point 1 (CRITICAL): ``token_confidence_mean`` field 再利用は
**probability (0-1 range) vs log probability (負の値) semantics 不整合**で
全 reject になる critical bug。``avg_logprob`` field 使用が正解。

##### Engine-specific threshold (reviewer Point 3、HIGH)

ReazonSpeech (margin +0.10-0.13) と Voxtral (margin +1.0) は同 ``avg_logprob``
field を共用するが分布が桁違い。global ``avg_logprob_threshold = -1.0`` は
ReazonSpeech に機能しないため、**``FilterConfig.avg_logprob_thresholds:
Dict[str, float]``** を追加:

- ReazonSpeech default ``-0.2``
- Voxtral は dict に load しない → ``avg_logprob_threshold`` (global) fallback
  で ``-1.0`` 維持 (**backward compat ゼロ regression**)
- ``should_reject()`` の signature を ``(result, config, engine_name=None)``
  に refactor、``apply_filter()`` から engine_name pass-through
- engine-specific lookup → global fallback の 2 段 fallback で scalable

##### Findings (詳細は ``docs/research/reazonspeech-confidence-smoke-2026-06-11.md``)

- **Phase 1 bug fix** ✅ — 3 件 regression test で long-audio silent drop bug pin
- **Section 1 (engine smoke、5 clip × int8/float32)** ✅ —
  - int8: speech mean -0.14、non_speech mean -0.30、margin +0.13、Case A
  - float32: speech mean -0.16、non_speech mean -0.45、margin +0.10、Case A
  - 両 model で threshold -0.2 が 100% 分類成功 (reviewer Point: int8
    availability も確認済)
- **Section 2 (12 cell stream pipeline benchmark)** ✅ —
  - **``webrtc × reazonspeech × real × on``: Hall.(post) 50% → 0%**
    (Issue #295 元 motivation の最後の cell 完了)
  - ``webrtc × synthetic × on``: 62.5% → **0%** (完全解消、codex-review
    #319 1st round の engine_name normalize fix 後の 2nd run で確認、初版
    は 25% 残)
  - silero / tenvad × real: 0% → 0% (VAD で除去済、filter は冗長安全網)
  - Latency 影響ゼロ
- **Section 3 (language coverage)** — ReazonSpeech は日本語 native only、
  Canary PR-A.4.2 と同 framing

##### Migration

- WhisperS2T / Parakeet (ja/en) / Voxtral / Canary 退行ゼロ
- ReazonSpeech user は ``--confidence-filter on`` (default) で hallucination
  が自動 drop されるようになる
- 長尺音声 (>30s) user は production bug 修正で正しい transcription を
  受け取れるようになる

##### Out of scope (qwen3asr は Issue [#318] で research-phase)

reviewer Point 5 (HIGH): qwen3asr の avg_logprob 単独 filter は危険
(confidence filter ≠ hallucination content guard、英語 mode で system
prompt leak、日本語 mode で repetition loop)。Issue [#318] で probe +
hallucination guard 設計を別 PR で扱う (本 PR scope 外)。

##### docs update (6 engine 対応に整合)

- ``docs/research/reazonspeech-confidence-smoke-2026-06-11.md`` (新規)
- ``docs/audio-filter-reference.md`` Engine support table を 6 engine 対応、
  Property table / Decision section / 完成サマリ / Comparison table 全
  section で 5 → 6 engine 整合
- ``docs/reference/cli.md`` / ``docs/reference/api.md``: ReazonSpeech 追加、
  ``avg_logprob_thresholds`` field 明記
- ``base_engine.py`` EngineConfidence docstring の populate status table に
  ReazonSpeech 追加
- ``confidence_filter.py`` module docstring を engine-specific threshold
  framing に整合

#### Engine confidence filter — Parakeet 英語 support (Issue [#311] PR-A.4.3)

PR-A.0 ([#309]) / PR-A.4.1 ([#313]) / PR-A.4.2 ([#315]) で whispers2t /
parakeet_ja / Voxtral / Canary に対応した confidence filter を **Parakeet
英語** (`nvidia/parakeet-tdt-0.6b-v2`) にも拡張。本 PR で **Parakeet 英語が
構造的限界ではなく PR #309 時点の設定漏れだった**ことが判明、新規 **Path
1.5** で対応:

- **Before**: Parakeet 英語の `transcribe()` は `engine_confidence =
  EngineConfidence()` (全 None) で fail-open。decoding は strategy-only
  (旧 Path 2)。
- **After**:
  - `parakeet_engine.py::_configure_decoding_with_confidence` に **Path 1.5**
    追加 (Path 1 Hybrid CTC と Path 2 strategy-only の間に挿入):
    - pure RNNT/TDT model 用、`preserve_alignments=True` + `confidence_cfg`
      + `greedy.preserve_frame_confidence=True`
    - NeMo の制約「preserve_frame_confidence は preserve_alignments と同時
      設定必須」(`rnnt_decoding.py:280-282`) を満たす形で構成
    - Path 1.5 が rejected された場合は Path 2 (strategy-only) に fail-open
      fallback
  - `_extract_engine_confidence()` helper は Parakeet_ja と同じものを共用
    (Canary PR-A.4.2 で Tensor / List / numpy 全部扱えるよう拡張済)
  - `_log_filter_banner()` の表現を `parakeet_ja / canary` → **`parakeet (ja/en) / canary`**
    に整合
- **Migration**:
  - WhisperS2T / Parakeet_ja / Voxtral / Canary 退行ゼロ
  - Parakeet 英語 user は `--confidence-filter on` (default) で英語 audio の
    hallucination が自動 drop される
  - **非英語入力時の false reject リスク**: Parakeet 英語は English-only model、
    非英語音声には低 confidence で false reject の可能性。`--confidence-filter off`
    で opt-out 可能
- **Findings (詳細は `docs/research/parakeet-english-confidence-smoke-2026-06-11.md`)**:
  - **Phase 1 probe** ✅ — `hypothesis.token_confidence` は **List[float]** で
    populate。LibriSpeech 英語 → token_confidence_mean = **0.2452**
  - **Section 1 (engine smoke、3 clip)** ✅ — speech 0.2452 vs threshold 0.005
    で **49× margin** (Case A、3 engine 中で最大)。非音声は engine 自体が
    empty text を返す fail-safe
  - **Section 2 (stream pipeline、12 cell)** ✅ — `webrtc × synthetic × on`
    で **Hall.(post) 75% → 12.5%** を実証 (filter で hallucination の 5/6 を drop)
  - **Section 3 (language coverage)** — English native validate、非英語入力時
    の language mismatch も実機確認 (production user 注意点として docs 記載)
- **Tests** (新規 +3 件、合計 655 passed):
  - `tests/core/engines/test_parakeet_decoding_strategy.py` の pin を新挙動
    (Path 1.5 で confidence_cfg 試行) に整合、CTC failure fallback も Path 1.5
    に整合
- **Docs update**:
  - `docs/research/parakeet-english-confidence-smoke-2026-06-11.md` (新規)
  - `docs/audio-filter-reference.md`: Engine support table を Parakeet 英語
    ✅ Production に更新、PR-A 系列完成サマリ 5 engine 対応に拡大
  - `docs/reference/cli.md` / `docs/reference/api.md`: Parakeet 英語追加
  - `livecap_cli/cli.py` / `confidence_filter.py` / `base_engine.py` /
    `stream.py`: 全 layer で Parakeet (ja/en) 一貫表示

#### PR-A 系列完成 docs 整合 (Issue [#311] PR-A.4.docs)

Issue #311 v2.1 plan の最終 PR。PR-A.4.1 ([#313 MERGED]) で Voxtral、PR-A.4.2
([#315 MERGED]) で Canary の filter 対応を完了した後の **docs 整合 sweep**。

- **Before**: 一部 docs に stale な「voxtral / canary は fail-open」記述が
  残存:
  - `docs/benchmarks/pr-a-calibration-2026-06-10.md:177` (PR-B calibration
    時の残作業 list、voxtral/canary がまだ fail-open とされていた)
  - `docs/research/voxtral-confidence-smoke-2026-06-11.md:108` (他 engine
    の挙動 section に Canary が fail-open list で残存)
- **After**:
  - `pr-a-calibration-2026-06-10.md`: PR-A.4.1/A.4.2 完了状況を反映、qwen3asr
    のみ PR-A.5 candidate として残存する旨を明示
  - `voxtral-confidence-smoke-2026-06-11.md`: Canary を populate engine list
    に移動 (PR-A.4.2 整合)
  - `docs/audio-filter-reference.md`:
    - Property table の Production-ready statement を 4 engine (WhisperS2T /
      Parakeet_ja / Voxtral / Canary) に拡張
    - Comparison table の Confidence Filter 行を「4 engine 対応」+ 50%→0%
      実測実証を反映
    - **新 section: PR-A 系列 完成サマリ** (2026-06-11 時点) を追加 — Engine
      support table の最終状態 / production user 選択ガイド / Phase 1 多段
      防御 5 layer 到達点を 1 section に集約
- **Side effects**:
  - 全 docs 層 (audio-filter-reference / cli.md / api.md / feature-inventory
    / decision doc × 2 / CHANGELOG / Engine support table / source docstring)
    で **Canary が `token_confidence_mean` populate engine** として一貫表示
    完了
  - Issue #311 v2.1 plan の Core scope (PR-A.4.1 + PR-A.4.2 + PR-A.4.docs)
    が完了、close 候補に
- **Out of scope (PR-A.5 candidate に申し送り)**:
  - qwen3asr: qwen-asr wrapper が内部で ``output_scores=True`` を渡さず、
    ``text_ids = model.generate(...)`` のみ実行 ([source 確認済](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py))。
    wrapper bypass or vLLM logprobs 移行が必要 (heavy)。
  - ReazonSpeech: sherpa-onnx Python bindings に per-token score API
    なし。upstream [PR #2897](https://github.com/k2-fsa/sherpa-onnx/pull/2897)
    が C/Dart で `getVocabLogProbs()` 追加したが closed (not merged)、
    Python 未対応。upstream PR or PyTorch native 実装切替が必要。
- **🆕 PR-A.4.3 candidate (Issue #311 v2.2 へ申し送り)**:
  - **Parakeet 英語** (`parakeet-tdt-0.6b-v2`): 旧 docs では「NeMo RNNT
    path に token_confidence 未実装」を根拠に PR-A.5 candidate としていた
    が、本 PR 作業中の調査で **「構造的限界ではなく PR #309 時点の設定漏れ」**
    と判明。NeMo source (`rnnt_decoding.py:95-106`, `tdt_loop_labels_computer.py`)
    で `preserve_token_confidence` documented + 実装あり、`preserve_alignments
    =True` 同時設定で populate される。**実機 probe (本 PR docs 作業中)**
    で `token_confidence_mean = 0.2452` を確認 (LibriSpeech 英語、threshold
    0.005 の 49x)。実装は別 PR (PR-A.4.3) で対応 — 本 PR は docs scope の
    ため probe 修正は revert 済、PR-A.4.3 で改めて Path 1.5 実装 + smoke
    verify + 完全 docs 整合を実施予定。
  - **PR-A.4.3 acceptance criteria** (codex-review PR #316 3rd round 提示):
    1. ``parakeet_engine.py::_configure_decoding_with_confidence`` の pure
       RNNT/TDT path に ``preserve_alignments=True`` と
       ``confidence_cfg.preserve_token_confidence=True`` を含む dedicated
       path (Path 1.5) を追加
    2. その path が失敗した場合は現行の strategy-only path に fail-open
       fallback (Path 2 既存)
    3. ``tests/core/engines/test_parakeet_decoding_strategy.py:113-138``
       を更新し、Parakeet English で confidence cfg を試行することを pin
       (現状は pure RNNT で confidence_cfg を含めないことを pin している
       ため、PR-A.4.3 で挙動変更に合わせ更新必須)
    4. ``tests/core/engines/test_parakeet_confidence_extraction.py`` は
       既存 helper が list/tuple の ``token_confidence`` を扱えているため
       大枠流用可能。Parakeet English の hypothesis shape が異なる場合
       (Tensor / numpy 等) は fixture 追加
    5. 実機 smoke で ``token_confidence_mean`` populate + speech が
       threshold ``0.005`` を十分上回ること確認 (本 PR probe で 0.2452 =
       49× を確認済、smoke verify で再現性確保)

#### Engine confidence filter — Canary support (Issue [#311] PR-A.4.2)

PR-A.0 ([#309]) / PR-A.4.1 ([#313]) で whispers2t / parakeet_ja / Voxtral に
対応した confidence filter を **Canary 1B Flash** にも拡張。NeMo
``EncDecMultiTaskModel`` の **beam → greedy decoding 切替** +
``confidence_cfg.preserve_token_confidence=True`` で
``hypothesis.token_confidence`` (torch.Tensor) を取得、Parakeet 同様の
``token_confidence_mean`` を ``EngineConfidence`` に populate。

- **Before**: Canary の ``transcribe()`` は ``engine_confidence = EngineConfidence()``
  (全 None) で fail-open、``confidence = 1.0`` ハードコード。
- **After**:
  - ``canary_engine.py::_configure_model()`` で ``_configure_decoding_with_confidence()``
    呼出 (3-fallback path、Parakeet pattern 流用): Greedy + confidence_cfg →
    Greedy only → argument-less。いずれも raise しない。
  - ``_transcribe_single_chunk()`` に ``return_hypotheses=True`` 追加、
    Hypothesis から ``_extract_engine_confidence()`` 経由で
    ``token_confidence_mean`` 取得。
  - **新 helper**: ``_extract_engine_confidence(hypothesis)`` — Canary は
    ``token_confidence: torch.Tensor`` (Parakeet は ``List[float]``) で
    返すため ``hasattr(token_conf, 'tolist')`` 防御で GPU tensor / numpy /
    list を統一処理。
  - ``confidence = float(token_confidence_mean)`` で UI display 意味化。
  - **filter logic 変更なし**: ``confidence_filter.py::should_reject()``
    の ``token_conf_threshold = 0.005`` path を Parakeet_ja と共用。
  - **新 doc**: ``docs/research/canary-confidence-smoke-2026-06-11.md`` に
    Phase 1 probe + Section 1/2/3 を永続化。
- **Migration**:
  - **WhisperS2T / Parakeet_ja / Voxtral 退行ゼロ** (filter logic 不変)。
  - **Canary user** は ``--confidence-filter on`` (default) で対応言語
    (en/de/fr/es) の hallucination が自動 drop される。
  - decoding strategy が beam → greedy に切替 (NeMo AED の confidence 取得
    のため)。Parakeet_ja TDT→CTC 同様、軽微 WER 退行可能性あるが filter
    benefit を優先。**``--confidence-filter off`` は post-ASR の reject を
    止めるだけで、decoding は常に greedy** (filter logic と decoding strategy
    を独立に管理)。旧 beam decoding に戻す option は本 PR では非提供。
  - **``beam_size`` parameter 削除 (silent no-op cleanup、PR-A.4.2 →
    Issue #321 PR #1 で完成)**: 旧 ``CanaryEngine`` constructor の
    ``beam_size: int = 1`` および ``metadata.default_params`` の
    ``beam_size`` は ``_configure_decoding_with_confidence()`` で常に
    greedy 切替されるため silent no-op だった。pre-1.0 cleanup 方針に従い
    削除。
    - **Before**: ``CanaryEngine(beam_size=N)`` は warn + silent ignore
    - **After** (Issue #321 PR #1): ``CanaryEngine.__init__`` から
      ``**kwargs`` を完全削除、``CanaryEngine(beam_size=N)`` は
      ``TypeError: unexpected keyword argument 'beam_size'`` で fail-fast
    - **Migration**: caller は ``beam_size`` 指定を削除する必要がある
      (greedy が常に有効、beam search に戻す option は本 PR では非提供)
- **Findings**:
  - **Phase 1 probe** ✅ — ``hypothesis.token_confidence`` は **torch.Tensor**
    で populate (Parakeet と型差分、helper で吸収)。LibriSpeech 英語 →
    token_confidence_mean = **0.0724**
  - **Section 1 (engine smoke、3 clip)** ✅ — speech 0.0724 vs threshold
    0.005 で **14.5x margin** (Case A)。非音声は engine 自体が empty text を
    返す **fail-safe 挙動** (Voxtral/Parakeet と異なり Canary は元々
    hallucinate しない)
  - **Section 2 (stream pipeline benchmark、12 cell)**:
    - Hall.(pre) = 0% 全 cell (Canary 固有の robustness)
    - SR(post) = 0% all real cells — PR-B corpus は日本語、Canary 非対応
      で engine が empty text を返す fail-safe
    - filter on/off で同一結果 (pre-filter hallucination が 0% のため filter
      効果が観察不可、Section 1 で margin 検証済)
    - Latency 影響なし
  - **Section 3 (language coverage)** ✅ — English native で margin 確認済、
    Japanese は Canary 非対応で empty text 返却 (Voxtral の translation
    regime と異なり誤訳混入なし)
- **Out of scope**: qwen3asr / reazonspeech / parakeet_en は **PR-A.5**
  (heavy refactor)。Canary 他言語 verify は user feedback ベース。
  > _Superseded by PR-A.4.docs_ ([#316]): `parakeet_en` は本 PR の probe で
  > 「実は populate 可能 (PR #309 時点の `preserve_alignments` 併設漏れ)」と
  > 判明、**PR-A.4.3 candidate に格上げ**。PR-A.5 は qwen3asr / reazonspeech
  > の 2 engine に縮減 (上の PR-A.4.docs entry 参照)。

#### ``TranscriptionResult.__iter__`` 削除 (pre-1.0 cleanup)

PR-A.0 ([#309]) で導入した ``TranscriptionResult.__iter__`` (旧
``Tuple[str, float]`` 戻り値との後方互換 shim) を削除。pre-1.0 (
`1.0.0.dev0`) では legacy compat shim は不要との CLAUDE.md / AGENTS.md
方針に従う。

- **Before**: ``text, confidence = engine.transcribe(audio, sr)`` の
  tuple unpacking 形が動作 (``TranscriptionResult.__iter__`` 経由)。
- **After**: ``result = engine.transcribe(audio, sr); text =
  result.text; confidence = result.confidence`` の attribute access に
  統一。``__iter__`` 削除により tuple unpacking は ``TypeError`` で fail。
- **Migration**:
  - `livecap_cli/transcription/stream.py`: 3 path (sync / async /
    interim) を attribute access に migration
  - `benchmarks/asr/runner.py` / `benchmarks/common/engines.py` /
    `benchmarks/optimization/objective.py`: 同 migration
  - `tests/`: 4 mock engine fixture (test_stream.py /
    test_stream_translation.py / test_mock_realtime_flow.py /
    test_from_language_integration.py) を ``Tuple[str, float]`` 返却 →
    ``TranscriptionResult`` 返却に統一
  - `tests/core/engines/test_engine_confidence_schema.py`:
    ``__iter__`` 互換を pin する 3 件を削除、代わりに
    ``test_tuple_unpacking_no_longer_supported`` で ``TypeError`` 化を
    pin
  - `livecap_cli/engines/{base,whispers2t,parakeet,canary,qwen3asr}_engine.py`
    docstring から「tuple unpacking 互換」記述削除
  - `livecap_cli/transcription/stream.py` / `confidence_filter.py`
    docstring も同様に整合
- **Side effects**:
  - `livecap_cli/engines/shared_engine_manager.py:443-449` の defensive
    3-branch (attribute / ``__getitem__`` / fallback) は不変。
    ``__iter__`` に依存しない別 path のため独立 cleanup PR で対応可。
  - `livecap_cli/engines/reazonspeech_engine.py:430` の internal helper
    ``_transcribe_single()`` も不変 (``TranscriptionResult`` ではなく
    raw tuple を返す内部関数、cleanup scope 外)。
- **Tests**: 508 → 506 passed (削除 3 - 新 1 = -2)。

#### Engine confidence filter — Voxtral support (Issue [#311] PR-A.4.1)

PR-A.0 ([#309]) / PR-A.1 ([#310]) で whispers2t / parakeet_ja に対応した
confidence filter を **Voxtral** にも拡張。`transformers.GenerationMixin.compute_transition_scores(normalize_logits=True)` 経由で per-token logprob
を復元、special token (EOS/PAD/BOS) 除外平均を `EngineConfidence.avg_logprob`
field に populate する。

- **Before**: Voxtral の `transcribe()` は `engine_confidence = EngineConfidence()`
  (全 None) で返却 → `is_available=False` → filter 常に fail-open。`confidence`
  field は `1.0` ハードコード。
- **After**:
  - `livecap_cli/engines/voxtral_engine.py` の `_transcribe_single_chunk()`
    を `model.generate(output_scores=True, return_dict_in_generate=True)` 化、
    `compute_transition_scores` で score 復元、`_extract_engine_confidence()`
    helper で special token 除外平均を計算。
  - `confidence = exp(avg_logprob)` で UI confidence display を意味化
    (PR-A.0 の WhisperS2T と整合)。
  - **新 helper**: `_extract_engine_confidence(transition_scores, gen_tokens, special_ids) -> EngineConfidence`
    — pure function として export、PR-A.0 の whispers2t / parakeet 同パターン。
  - **filter 拡張**: `confidence_filter.py::should_reject()` に **strict-gated**
    `avg_logprob` 分岐追加。`no_speech_prob is None` AND
    `token_confidence_mean is None` の時のみ評価 (WhisperS2T 退行回避)。
  - **新 default**: `FilterConfig.avg_logprob_threshold = -1.0` (PR-A.4.1
    smoke verify 結果に基づく決定)。Voxtral smoke (2026-06-11, RTX 4090) で
    speech 4 clip mean=-0.420 vs non-speech mean=-1.525、margin +1.002、
    midpoint -1.024 → `-1.0` で 100% 分類可能。
  - **新 doc**: `docs/research/voxtral-confidence-smoke-2026-06-11.md` に
    decision (Setup / 4 Hypotheses / Results / Decision / Implications /
    Reproducibility) を永続化。
  - **docs update**: `docs/audio-filter-reference.md` の Engine support
    table を Voxtral ✅ avg_logprob (gated) に更新。
- **Migration**:
  - **WhisperS2T / Parakeet_ja 退行ゼロ** (strict gate により avg_logprob
    経路に到達しない、unit test で pin)。
  - **Voxtral user** は `--confidence-filter on` (default) で hallucination
    が自動 drop されるようになる (production behavior 変化、PR-A.4.1 smoke で
    実証)。
  - Python API で `FilterConfig(avg_logprob_threshold=None)` を明示すれば
    avg_logprob 判定経路を opt-out 可能 (e.g., Voxtral debugging 時)。
  - ReazonSpeech / qwen3asr / Canary / mock は `engine_confidence` 全 None
    のまま (fail-open 不変)。
- **Findings (詳細は `docs/research/voxtral-confidence-smoke-2026-06-11.md`)**:
  - **Section 1 (engine-level smoke、6 clip)**:
    - H1 ✅ — Voxtral speech 4 clip max=-0.354、min=-0.523、worst でも -1.0 上
    - H2 ✅ — Voxtral non-speech `applause_5_claps`: -1.525 << -1.0、
      `desk_tap`: empty (全 EOS) → fail-open
    - H3 ✅ — clear margin +1.002 (Case A 確定)
    - H4 ✅ — special token 除外 logic で `desk_tap` (全 EOS) は
      `EngineConfidence()` を返す (filter fail-open)
  - **Section 2 (stream pipeline benchmark、12 cell sweep)**:
    - F2.1 ✅ — **`webrtc × voxtral × real × filter on`: post-filter
      hallucination 50% → 0%**、post-filter speech recall 100% 維持。
      PR-A.4.1 核心 claim を stream pipeline 経由で実機実証。
    - F2.2 ✅ — silero / tenvad × voxtral × real は filter on/off 関係なく
      0% 維持 (副作用ゼロ、VAD 段階で既に non-speech 除去)
    - F2.3 — synthetic positive (formant proxy) は filter on で SR(post)
      40-60% drop。PR-A.3 H3.b と同じ意図通り挙動 (real speech ではない)
    - F2.4 — synthetic Hall.(post) は partial drop (75% → 25% on webrtc)、
      残存は threshold -1.0 と real corpus 100% 維持の trade-off
    - F2.5 ✅ — latency 影響なし (p50/p95 は filter off と同等)
  - **Section 3 (language-stratified follow-up)**: 旧 Section 1 は
    **日本語音声 × language="en"** で実行されていたが Voxtral は en/es/fr/
    pt/hi/de/nl/it の 8 言語のみサポート (ja は対象外)、結果として旧 smoke
    は **translation regime** (ja→en) を測定していた。native English
    transcription (LibriSpeech) で再検証:
    - F3.1 ✅ — Native transcription: avg_logprob -0.115 (translation
      mean -0.420 より 0.305 高信頼度)
    - F3.2 ✅ — Threshold -1.0 は translation regime の lower bound に
      calibrate されていた → native regime では margin +1.410 (translation
      +1.002 から拡大)、両 regime で validate 完了
    - F3.3 — 言語 coverage: en (native + translation) のみ、他 7 言語は
      merge 後 user feedback で順次検証 (false reject 報告時は
      `FilterConfig(avg_logprob_threshold=None)` opt-out 可能)
- **Out of scope (次の handle)**:
  - **Canary** filter 対応: **PR-A.4.2** (NeMo `EncDecMultiTaskModel`、
    beam→greedy decoding 切替が gate)
  - **qwen3asr / reazonspeech / parakeet_en**: **PR-A.5** (wrapper bypass /
    vLLM 移行 / sherpa-onnx 構造的限界などの heavy refactor 系)
  - **Voxtral non-English language** での smoke verify: user feedback で
    順次対応 (本 PR は `language="en"` で実施)

  > _Superseded by PR-A.4.docs_ ([#316]): `parakeet_en` は PR-A.5 から外され
  > **PR-A.4.3 candidate** に格上げ済 (probe で `token_confidence_mean = 0.2452`
  > 確認、threshold 0.005 の 49×)。PR-A.5 は qwen3asr / reazonspeech の 2
  > engine に縮減。詳細は最上段 PR-A.4.docs entry を参照。

#### Confidence filter calibration sweep + new `post_filter_hallucination_rate` metric (Issue [#308] PR-A.3)

PR-A.1 ([#310]) で実装した confidence filter を 54 cell sweep
(1 preset × 3 backend × 3 engine × 2 corpus × 3 filter_mode) で validate
し、Phase 1 epic ([#295]) closure を数値証拠付きで achievable な状態に。

- **Before**: PR-A.1 で filter 本体は実装済だが、production 全 cell での効果は
  smoke verify (6 clip × 2 engine = 12 ケース) のみで実証。`benchmarks/
  non_speech_filter/` の既存 metric (`false_asr_trigger_rate` / `non_empty_
  hallucination_rate`) は **engine の生出力** を測定しており、filter 適用後
  の user の subtitle stream に届く text は計測できなかった。
- **After**:
  - `benchmarks/non_speech_filter/runner.py` の `NonSpeechFilterBenchmark
    Config` に `filter_config: Optional[FilterConfig] = None` を追加、
    `_make_pipeline_factory()` で `build_pipeline()` に pass-through。
  - `benchmarks/non_speech_filter/sweep.py` の `run_sweep()` に
    `filter_mode ∈ {off, observe, on}` の 3 段 nested loop を追加。
    `SweepCellResult` に `filter_mode` field 追加、CSV / Markdown 出力に
    column 追加。
  - **新 metric `post_filter_hallucination_rate`** を `evaluate_pipeline()`
    に追加。`transcriber.finalize()` 戻り値 + `_result_queue` の直接 drain
    (`InterimResult` を明示的に skip し `TranscriptionResult` のみ収集)
    を合算することで、user の subtitle stream に実際に届く text を計測する。
    `non_empty_hallucination_rate` (pre-filter engine 直出力) と並列で出力。
    旧版 (initial commit) は `finalize()` 戻り値を取りこぼし、queue drain も
    interim 先頭で停止する 2 件の bug があったため、codex-review on #312
    1st + 2nd round で修正済。
  - **新 metric `post_filter_speech_recall` / `post_filter_short_utterance_recall`**
    を追加 (codex-review on #312 3rd round Item 1 HIGH)。旧 `speech_recall`
    は engine call の counter で計測されており、filter が legit speech を
    drop しても 1.0 のままだった。新 metric は user の subtitle stream に
    届く speech 比率を直接測定。`measure_hallucination=True` 時のみ意味あり。
  - `_collect_post_filter_texts` helper を追加 (codex-review #312 3rd round
    Item 2 MED)。`_result_queue` 直接 access を helper に閉じ込め、将来の
    StreamTranscriber queue 実装変更時の修正箇所を 1 箇所に集約。
  - `docs/benchmarks/pr-a-calibration-2026-06-10.md` 新規 — PR-A 系列
    (A.0/A.1/A.3) の calibration 総括 doc を PR-B (2026-06-07) と同じ
    Setup / Hypotheses / Findings / Decision / Implications / Reproducibility
    構造で執筆。
- **Migration**: 既存 sweep を本 PR の harness で再実行すると、新 column
  `post_filter_hallucination_rate` が CSV / Markdown に追加されている。
  既存の `non_empty_hallucination_rate` semantics は不変 (pre-filter
  engine 出力を測定)、解釈時には 2 列を比較して filter 効果を測る。
- **Findings (詳細は `docs/benchmarks/pr-a-calibration-2026-06-10.md`)**:
  - H1 ✅ — `webrtc × parakeet_ja × real desk_tap` filter `on` で
    50% → 0%、`synthetic` でも 75% → 0% を実証。Issue #295 の元 motivation を
    実機で完全解決。
  - H1.b ✅ — synthetic corpus で WhisperS2T 内部 `no_speech_prob` filter
    を bypass する edge case (25%) も filter `on` で 0% に。重複防御として
    実効的に機能。
  - H2 ✅ — `silero / tenvad × all engines` で filter mode に関係なく
    0% 維持 (production user の副作用ゼロ)。
  - H3 ✅ (v3 refined) — **real corpus** で post-filter SR = 100% 維持
    (filter は legit speech を 1 件も drop していない)。synthetic positive
    の SR(post) drop は filter が formant proxy を正しく低信頼度として
    drop している = 期待挙動。production user は real speech を扱うため
    real corpus の結果が production 挙動。
  - H4 — `BASELINE_INVARIANTS` は不変判断。CI test は synthetic + Mock
    Engine で filter は fail-open のため tighten 不要。
  - ReazonSpeech — `engine_confidence` 全 None で filter fail-open。
    `post_filter = pre_filter` で filter 効果なし (sherpa-onnx 構造的限界、
    PR-A.5 で長期対応)。
- **Side effects**:
  - `benchmarks/non_speech_filter/report.py::NonSpeechFilterRunRecord`
    に `post_filter_hallucination_rate: float | None = None` field 追加。
    既存 caller は default None で動作するため後方互換。
  - 既存 sweep test 全 pass 維持。新規 sweep axis test 5 件追加で
    filter_mode 軸の挙動を pin。
  - `CHANGELOG.md` の本 entry で PR-B との比較を可能に。
  - Phase 1 epic ([#295]) close 候補状態に到達 — 残作業は PR-A.4
    ([#311] qwen3asr/voxtral/canary の filter 拡張、別 track)。

#### Engine confidence filter — default ON (Issue [#308] PR-A.1)

Adds the post-ASR `livecap_cli.transcription.confidence_filter` module
that watches `engine_confidence` (PR-A.0) and silently drops outputs the
engine itself judged as non-speech, before they reach the subtitle stream.
Default is **on** for all realtime sessions.

- **Before** (PR-A.0): every engine output reached the subtitle stream,
  even when `no_speech_prob` was high (WhisperS2T) or
  `token_confidence_mean` was near zero (Parakeet_ja). The PR-B 144-cell
  matrix showed `webrtc × parakeet_ja × desk_tap` hallucinated 50 % of
  the time.
- **After** (PR-A.1): `StreamTranscriber` calls `apply_filter()` on the
  3 call sites (sync L566 / async L638 / interim L787). For
  `--confidence-filter on` (default), rejected outputs become `None`
  drops with a structured INFO log; the subtitle stream sees nothing.
  Real-machine smoke verify on the 6-clip PR-B corpus produced 100 %
  classification (all speech clips passed, both non-speech clips dropped
  on both whispers2t and parakeet_ja).
- **Migration**: existing scripts keep the same flags. To restore the
  previous behavior, pass `--confidence-filter off` or export
  `LIVECAP_CONFIDENCE_FILTER=off`. Engines that do not expose confidence
  signals (`reazonspeech`, `qwen3asr`, `voxtral`, `canary`) are
  pass-through regardless of the flag (fail-open by design).
- **Side effects**:
  - Per-engine thresholds (`whispers2t no_speech_prob > 0.5`,
    `parakeet_ja token_confidence_mean < 0.005`) are baked in from the
    PR-A.0 verify values; PR-A.3 will revisit them after a full 144-cell
    sweep.
  - The `--confidence-filter observe` mode emits the same structured
    decision log **for both pass and reject decisions** (codex-review
    #310 Item 4) — PR-A.3 calibration needs the speech-side
    `engine_confidence` distribution as well, not just the reject side,
    to evaluate threshold margins and speech-recall safety. The `on`
    mode keeps logging reject only to avoid production log spam. Log
    payload is JSON (stable schema documented in `_decision_to_dict()`
    of `confidence_filter.py`) so PR-A.3 parsers can read it as JSONL.
  - The `StreamTranscriber.__init__` gained a `filter_config:
    Optional[FilterConfig] = None` parameter; `None` constructs the
    default (on) at instantiation time. Direct API users who want the
    old behavior should pass `FilterConfig(mode="off")` explicitly.
  - `benchmarks/non_speech_filter/pipeline.py::build_pipeline` defaults
    to `FilterConfig(mode="off")` so existing sweep baselines remain
    bit-identical; PR-A.3 will pass `FilterConfig(mode="on")` to
    measure filter impact on the cell matrix.
  - **Scope clarification (codex-review #310 Item 3)**: this PR exposes
    `filter_config` on `build_pipeline()` only. Adding a
    `confidence_filter` axis to `benchmarks/non_speech_filter/sweep.py`
    (so that the existing preset/backend/engine matrix is multiplied by
    `{off, observe, on}`) is **deferred to PR-A.3**, together with the
    full 144-cell sweep run. The pipeline-level hook here is sufficient
    for PR-A.3 to construct sweep cells programmatically.
  - A startup INFO log line (`"Confidence filter: ON (...)"`) makes the
    active mode visible at every session start.

#### Parakeet_ja decoder strategy: RNNT greedy → CTC greedy_batch (Issue [#308] PR-A.0)

Investigation during PR #309 smoke verify uncovered that the
`nvidia/parakeet-tdt_ctc-0.6b-ja` checkpoint is an
`EncDecHybridRNNTCTCBPEModel` whose RNNT decoder (NeMo default) does not
implement `token_confidence`. The CTC decoder does. To make
`engine_confidence` actually populated for `parakeet_ja`, the adapter now
switches to the CTC decoder on `load_model`.

- **Before**: `parakeet_ja` used the RNNT decoder with `strategy=greedy`
  and the old `confidence_cfg` block — which NeMo silently rejected on
  the current version (`preserve_frame_confidence=True` requires
  `preserve_alignments=True`), so `token_confidence` was always `None`
  and the old `score / len(y_sequence)` fallback returned an
  empirically-inverted signal (speech `-71.5` vs applause `-47.3`).
- **After**: `_configure_decoding_with_confidence()` detects the hybrid
  model via `hasattr(self.model, 'cur_decoder')` and switches to
  `decoder_type='ctc'` with `strategy=greedy_batch`,
  `greedy.preserve_frame_confidence=True`, and a full `confidence_cfg`.
  `token_confidence_mean` is now populated with clean speech-vs-noise
  separation (0.01-0.10 vs 0.0000029-0.0003). A 3-stage fallback path
  protects against older NeMo versions and the non-hybrid English
  `parakeet` model.
- **Migration**: `EngineMetadata.default_params["parakeet_ja"]
  ["decoding_strategy"]` updated from `"greedy"` to `"greedy_batch"`
  (single source of truth, surfaced by GUI / diagnostics / docs). The
  English `parakeet` (pure RNNT) default remains `"greedy"`. Users who
  hard-coded `decoding_strategy="greedy"` on `parakeet_ja` will keep
  working (CTC greedy is slower but functional, NeMo emits a
  `greedy_batch` recommendation warning).
- **Side effects** measured on RTX 4090 with the
  `.tmp/non_speech_corpus/` 6-clip set:
  - **Latency** improves: CTC + `greedy_batch` runs 1.83× faster on the
    speech clip than the old RNNT `greedy` path (p50 81.4 ms vs
    149.8 ms).
  - **Transcription text** is preserved on 4/6 clips, slightly improved
    on 1/6 (`applause_5_claps` hallucinates fewer characters), and
    differs by 1 hiragana on 2/6 (e.g. 「とんと」 → 「とんど」). Not
    a regression for production usage; documented in
    `docs/research/parakeet-ja-confidence-spec-2026-06-10.md`.
- **Score fallback removed**: the previous
  `score / len(y_sequence) → avg_logprob` path inside
  `_extract_engine_confidence` is gone. When `token_confidence` is not
  available (older NeMo, non-hybrid model, or fallback path), the
  function returns `EngineConfidence()` honestly. The smoke-verified
  signal inversion made the old fallback actively harmful for the
  PR-A.1 filter.

This is an **engine behavior change** for `parakeet_ja`, but it
strengthens (not regresses) production behavior on every measured axis:
faster, comparable text, and a now-usable confidence signal.

#### Phase 2 SED model evaluation harness (Issue [#305] PR-D0)

- **New `benchmarks/sed/` package (research-only off-line evaluation;
  does not touch `livecap_cli/`)**:
  - `class_mapping.py` — pins the AudioSet 527-class taxonomy mapping for
    livecap reject signals. Defines `TARGET_CLASSES` (Hands / Finger
    snapping / Clapping / Applause / Door / Sliding door / Slam / Knock
    / Tap / Thump, thud — 10 classes) and `SPEECH_LIKE_CLASSES` (Speech
    family + Singing — 7 classes). Implements the three Issue #305 v3
    threshold policies (`max`, `sum`, `target_minus_speech`).
  - `inference.py` — loads EfficientAT pretrained models
    (`mn04_as` / `dymn04_as` / `dymn10_as`), resamples 16 kHz audio to
    32 kHz, slices into 1-second windows (Issue #305 v3 primary metric
    unit), and returns per-window 527-dim sigmoid probability matrices.
  - `metrics.py` — clip-level confusion-matrix metrics with hand-pinned
    semantics: class-level + reject-signal-level (Issue #305 v3 two-axis
    report), provisional-gate verdict (`precision ≥ 0.70` AND
    `recall ≥ 0.50` AND target clip flagged at the chosen threshold).
  - `latency.py` — 5-axis runtime measurement (checkpoint size, installed
    dep delta vs `engines-torch` baseline, runtime peak memory via
    `tracemalloc`, CPU + GPU p50/p95 latency, cold-start) per Issue #305
    v3 Dimension 3 refinement.
  - `orchestrator.py` — full evaluation pipeline (corpus → inference →
    CSVs + NPZ + JSON metadata).
  - `analyze.py` — post-hoc analysis: threshold sweep, class-level
    summary tables, provisional-gate verdict, `analysis.{json,md}` for
    decision-doc paste-in.
  - `cli.py` + `__main__.py` — `python -m benchmarks.sed` entry point.
  - `README.md` — manual EfficientAT setup, env vars, command reference.
- **New `tests/integration/sed/` (23 tests, `sed_evaluation` marker)**:
  - `test_class_mapping.py` (12 tests) — AudioSet index integrity, three
    policy semantics, validation; the
    `test_indices_match_efficientat_csv` test cross-checks the pinned
    indices against the canonical AudioSet CSV when EfficientAT is
    cloned.
  - `test_metrics.py` (10 tests) — synthetic 4-clip corpus with
    hand-derived precision/recall, gate truth-table (pass / precision
    fail / recall fail / target-not-flagged).
  - `test_inference_smoke.py` (1 test) — env-gated 1-clip smoke
    verifying `(n_windows, 527)` output shape; skipped automatically
    when the EfficientAT clone is absent.
  - New `sed_evaluation` pytest marker declared in `pyproject.toml`.
- **New `docs/research/phase2-sed-evaluation-2026-06-10.md` (~430
  lines)** — 4-dimension decision document covering Accuracy / Safety /
  Runtime / License (verdicts honest after codex-review on #306):
  - Accuracy: PASS (provisional). `target_minus_speech` policy at
    threshold ~0.10 yields precision=1.0, recall=1.0 on the 6-clip
    corpus; the critical `overlapping_applause_speech` case is correctly
    retained (`max(target)=0.16` would over-fire, but
    `target − speech_like = -0.66` correctly suppresses).
  - Safety: PASS. speech_recall = 1.00, short_utterance_recall = 1.00.
  - Runtime: **Conditional PASS** (CPU production-device path). CPU p95
    = 29.0 ms (3.4× under the 100 ms budget), checkpoint 4.07 MB
    (12× under 50 MB), runtime peak 6.68 MB (30× under 200 MB).
    **GPU p95 = 32.8 ms misses the original 30 ms ceiling by 9 %** —
    documented honestly rather than papered over; CPU runs faster than
    GPU at this 3.9 M-parameter scale, so production device = CPU and
    the CPU budget is satisfied.
  - License: PASS at the **Auto-download OK tier** (not Bundle OK —
    corrected after codex-review). The upstream EfficientAT release
    does not explicitly grant a license on the model weights, so the
    integration ships the checkpoint via `torch.hub` auto-download
    rather than packaging the `.pt` file; this matches both the legal
    evidence we have and the implementation already in use.
    Attribution stub recorded for PR-D1's `THIRD_PARTY_NOTICES.md`.
- **New `benchmark_results/sed/2026-06-10/` (committed per Issue #305 v3
  artifact policy)**: `probabilities.csv`, `probabilities_full.npz`,
  `latency.csv`, `metadata.json`, `analysis.json`, `analysis.md`.
- **`.gitignore` update**: changed `benchmark_results/` to
  `benchmark_results/*` with `!benchmark_results/sed/` exception so the
  PR-D0 evidence is committed while other benchmark outputs remain
  ignored. Added `.tmp/` to ignore the EfficientAT clone and any
  research scratch.
- **Issue #305 v2 → v3 body update** with six clarifications: metric
  calculation unit (window primary / clip-level max decision unit),
  license outcome 4-classification (Bundle OK / Auto-download OK /
  Manual user-provided only / NG), artifact commit policy, accuracy
  provisional-gate disclaimer, runtime constraint detailing
  (checkpoint / installed dep / runtime peak memory split), class-level
  + reject-signal-level two-axis metric report.
- **Scope discipline**: this PR does **not** modify any file under
  `livecap_cli/`. SED pipeline integration is PR-D1; default-decision
  is PR-D2; DSP detector disposition is PR-D2.

Verification: `pytest tests/integration/sed/` → 23 passed, 0 failed.
PR-relevant regression
(`tests/audio tests/integration/non_speech_filter tests/integration/vad
tests/transcription tests/core/cli tests/audio_sources`) → 307 passed,
5 skipped (env-gated), 0 failed — identical to the pre-PR baseline.

[#305]: https://github.com/Mega-Gorilla/livecap-cli/issues/305

#### Calibration follow-up: real-engine sweep + threshold tuning (Issue [#295] PR-B follow-up)

- **3 new hypothesis-driven candidate presets** appended to
  `benchmarks/non_speech_filter/sweep.py::default_named_presets()`:
  - `on_relaxed_rms` — drop `rms_min_db` floor from -35 to -45 to admit
    quieter real-corpus frames (real recordings sit at -41 to -46 dBFS
    overall, so the default floor was rejecting > 95 % of frames before
    the AND combination could fire).
  - `on_low_freq_aware` — widen the spectral centroid window
    (`centroid_min_hz` 2500 → 500) and tighten `voiced_max` (0.25 →
    0.15) to test whether `desk_tap`-style low-frequency thumps can be
    caught without dropping low-pitched speech.
  - `on_speech_safe` — tightest preset (`flatness_min` 0.45,
    `centroid_min_hz` 3000, `onset_ratio` 5.0) as a safety ceiling that
    confirms short-utterance recall stays at 100 % under aggressive
    filtering.
- **New `benchmarks/non_speech_filter/calibration.py` (~430 lines)**:
  reads the CSV emitted by `sweep.py` and produces a structured Markdown
  report containing (1) per-engine hallucination delta vs `baseline_off`
  (segmented by backend and corpus), (2) recall-regression flags for any
  preset/cell pair that dropped recall below the baseline, (3) a Pareto
  summary across presets with explicit dominance markers, and (4) a
  structured recommendation driven by Issue #295 PR-B follow-up plan
  rule D4 (≥30 % hallucination drop on `webrtc × parakeet_ja × real`
  with no recall regression → promote that preset; otherwise default
  off, document gap, propose Phase 2 SED).
- **Calibration findings** (full record in
  `docs/benchmarks/calibration-results-2026-06-07.md`):
  - 144 cells (8 presets × 3 backends × 3 engines × 2 corpora) ran in
    ~16.5 min on a single RTX 4090 with engine-load amortisation.
  - **`parakeet_ja × WebRTC × real desk_tap` hallucination unchanged
    at 50 % across all 8 presets** (the PR-B v4 AC target).
  - Same on `reazonspeech × WebRTC × real desk_tap` (50 % → 50 %).
  - `parakeet_ja × WebRTC × synthetic burst` hallucination drops
    75 % → 62.5 % (one item out of eight) on the 4 Pareto-dominant
    presets (`on_moderate`, `on_aggressive`, `on_relaxed_rms`,
    `on_low_freq_aware`).
  - **Zero recall regressions** in any of the 144 cells.
- **Default mode decision: `--transient-filter=off` is maintained.**
  Rule D4's headline criterion (≥30 % hallucination drop on the AC
  target cell) is unmet, so no preset earns a promotion to default.
- **`on_moderate` is documented as the best observed DSP preset for
  synthetic rapid-burst tests only** — explicitly **not** a production
  hallucination mitigation recommendation. Calibration showed zero
  improvement on the real-corpus target cell.
- **The DSP transient detector layer is positioned as `experimental`
  going forward**: not deprecated (no replacement exists yet) but not a
  production-hallucination-mitigation candidate. CLI invocations of
  `--transient-filter observe/on` now emit a one-line experimental
  notice to make the status visible at the moment of opt-in. Phase 2
  SED (sound-event detection) is the planned successor for
  `desk_tap`-style low-frequency transients.
- **New `docs/audio-filter-reference.md`**: user-facing reference for
  every audio filter in the pipeline (NoiseGate / TransientDetector /
  VAD / EnergyGate) — purpose, pipeline position, CLI surface, default
  state, measured effectiveness with citations, recommendation, known
  limitations. Single doc users can scan to decide which filter to
  enable.
- **No detector code change**. The sweep + analysis is pure data
  collection; this PR does not modify
  `livecap_cli/audio/transient_detector.py` or any production pipeline.
- **Issue #295 v6** reframes the PR-B AC line `WebRTC × desk_tap (real)
  false_trigger 50 % → 0 %` with the empirically demonstrated bound
  ("0.0 pp achievable with 6-feature AND DSP detector — Phase 2 SED is
  the correct route").
- **BASELINE_INVARIANTS bounds remain unchanged** (default unchanged
  → no tightening warranted).
- **Out of scope** (separate follow-ups): Phase 2 SED epic for low-
  frequency / non-broadband transient detection; detector architecture
  changes (AND → OR, weighted-sum, new features); `#302` lookahead
  delay (still gated on a future reject default ON decision that this
  calibration data argues against).
- Verification: full PR-relevant suite (`pytest tests/audio/
  tests/integration/non_speech_filter/ tests/integration/vad/
  tests/transcription/ tests/core/cli/ tests/audio_sources/`) → **300
  passed, 5 skipped (env-gated), 0 failed**.

#### Fixed: PR-B follow-up — async path bypass + causal best-effort spec (Issue [#295] PR-B follow-up)

- **`StreamTranscriber.transcribe_async()` が transient detector を bypass**
  していた問題を修正。`feed_audio()` と `transcribe_async()` の pre-VAD
  処理を `_apply_pre_vad_processing()` 共通 helper に集約し、両 path が
  必ず NoiseGate → Layer 1 detector → VAD の順で走るよう pin。
- **`tests/transcription/test_stream.py::TestTransientDetectorWiring`**
  を追加 (3 cases): sync/async 両 path の detector 起動、両 path の
  telemetry 完全一致を assert。再発防止層。
- **`TransientDetector.process()` docstring** に causal / no-lookahead
  仕様を明文化。`on` mode の chunked output は best-effort upper bound で
  あり、bit-exact reconstruction ではないことを記載。32 ms lookahead-
  delay 拡張は別 issue で track。
- **`tests/audio/test_transient_detector.py::TestStreamingEquivalence::
  test_on_mode_chunked_is_causal_best_effort`** を追加: telemetry が
  full / chunked で一致することと、`on` mode で flagged frame があれば
  energy が入力より低下することを assert (full vs chunked output の
  bit-exact equality は意図的に検証しない)。
- **default mode を `off` のまま維持** + ドキュメント整合: Issue #295
  / docs / CHANGELOG では旧 v3/v4 で「default observe」と記載していた
  が、PR-B 実装は安全側の `default off` を採用。calibration は
  `--transient-filter observe` で明示的に opt-in する運用を明記。
- 既存挙動への影響: なし (default `off` で detector 構築されないため)。
- Verification: `pytest tests/audio/test_transient_detector.py
  tests/transcription/test_stream.py tests/integration/non_speech_filter/`
  → 86 passed, 8 skipped (env-gated).

#### Layer 1: DSP Transient/Applause Detector (Issue [#295] PR-B)

- **新規 `livecap_cli/audio/transient_detector.py`**: 6 DSP feature
  (`spectral_flatness` / `spectral_centroid_hz` / `zero_crossing_rate` /
  `onset_strength` / `voiced_ratio` / `rms_db`) を AND 結合して
  applause-like フレームを検出する frame-based stateful detector。
  3 mode: `off` (構築されない) / `observe` (telemetry のみ、audio 不変) /
  `on` (applause-flag frame を zero-out)。
- **`StreamTranscriber` 統合**: 新引数
  `transient_detector: Optional[TransientDetector] = None`、`feed_audio`
  の NoiseGate 後 / VAD 前で起動。`reset()` / `close()` テレメトリにも
  対応 (EnergyGate と同じ pattern)。
- **CLI flags** (`transcribe` サブコマンド): `--transient-filter`
  (`off`/`observe`/`on`、default `off`) + 6 threshold flag。
- **Benchmark CLI flags** 同名で揃え、`build_pipeline()` に
  `transient_config` kwarg を追加。`None` を渡せば baseline pipeline は
  bit-identical に保たれる (PR-0 BASELINE_INVARIANTS regression なし)。
- **新規 `benchmarks/non_speech_filter/sweep.py`**: 5 named preset
  (`baseline_off` / `observe_defaults` / `on_conservative` /
  `on_moderate` / `on_aggressive`) を回す threshold sweep harness。
  CSV + Markdown 出力。
- **テスト**:
  - `tests/audio/test_transient_detector.py`: 26 unit test
    (feature 算出 / AND 決定 / streaming 等価性 / mode semantics /
    config validation)。
  - `tests/integration/non_speech_filter/test_transient_detector_integration.py`:
    7 integration test (observe = no-op on metrics, on-mode positive
    preservation, WebRTC burst no-regression)。
- **検証結果 (private real corpus + synthetic、mock engine)**:
  - observe mode は BASELINE_INVARIANTS と完全一致 (silero 0/0/0 %,
    tenvad 25/100/100 %, webrtc 75/100/100 %)。
  - on mode (moderate/aggressive) で **WebRTC × synthetic burst の
    false_trigger 75 % → 62.5 %**。
  - WebRTC × real desk_tap は default 閾値で 50 % のまま (per-clip 観測で
    `centroid_min_hz=2500` が desk_tap の低域成分を弾いていることを確認、
    calibration follow-up で対応)。
- **限界 (docs に明示)**:
  - 既定閾値は **synthetic rapid burst 想定**で、private real corpus の
    個別 clip (desk_tap、scattered 拍手) は未較正。
  - reject default ON 化は PR-B のスコープ外 — calibration sweep の
    結果が出てから別 PR で実施する設計。
- **Out of scope**:
  - PR-C で予定する Layer 2 (VADStateMachine cooldown) との signaling は
    実装しない (検出器は event を emit するが消費側は別 PR)。
  - PR-A 系の confidence filter / prompt reset は対象外。

#### Non-speech filter evaluation harness (Issue [#295] PR-0)

- **新規 `tests/integration/non_speech_filter/`**: Phase 1 多段防御
  (DSP transient detector / VADStateMachine cooldown 拡張 / Confidence filter /
  Prompt reset) の **baseline + regression 検出基盤** を導入。3 VAD backend
  (Silero / TenVAD / WebRTC) × 13 件の synthetic corpus (negative 8 + positive 5、
  うち短発話 2) で現状 pipeline (NoiseGate + VAD + EnergyGate) を計測し、
  `baselines/{backend}.json` に schema v1 で永続化。後続 PR-B/C/A はこの JSON を
  比較基準にする。
- **新規 `benchmarks/non_speech_filter/`**:
  `python -m benchmarks.non_speech_filter` で ad-hoc 評価可能な runner +
  Markdown/JSON レポート。`--engine whispers2t` 等を指定すれば
  `non_empty_hallucination_rate` (engine が非空 text を返した負例の割合) も計測。
  実音源は `LIVECAP_NON_SPEECH_CORPUS_DIR` で manifest+WAV を渡すと自動 load。
- **指標**: `false_asr_trigger_rate` / `speech_recall` /
  **`short_utterance_recall`** (最重要) / `non_empty_hallucination_rate` (opt-in) /
  `added_latency_p50_ms` / `_p95_ms`。
- **新 marker**: `evaluation_harness` (pyproject.toml に登録)。`-m evaluation_harness`
  で opt-in 実行、CI baseline tests のみ拾う。
- **既存コード touch ゼロ**: `livecap_cli/` 配下は無変更。`benchmarks/common/`
  (`DatasetManager` / `BenchmarkEngineManager`) は real-engine 経路でのみ利用。
- **動機**: Issue #295 v2 のレビュー指摘
  「**評価ハーネス先行整備** + pre-engine 優先 + DSP detector default off-by-default
  + 実装前 corpus 整備」を満たすため、Phase 1 PR-B/C/A の前提として独立着地させる。
- **限界 (docs/benchmarks/non-speech-filter.md に明記)**:
  - 合成 speech proxy は Silero VAD (実音声学習) で recall=0 になる構造的限界。
    Silero baseline を意味のある形で測るには実音源 fixture が必要。
  - WebRTC backend は binary 出力 (0.0/1.0) のため、Phase 1 PR-C で導入予定の
    hysteresis は no-op (duration-based cooldown のみ機能)。
- **検証**:
  - `uv run pytest tests/integration/non_speech_filter/ -m evaluation_harness`
    → 6 passed (3 backend × 2 tests), 6 skipped (env-var gated)。
  - `uv run python -m benchmarks.non_speech_filter --mode quick --backend silero,tenvad,webrtc`
    → JSON + Markdown 出力、Silero / TenVAD / WebRTC の baseline 差を可視化。
  - 既存 `tests/integration/vad/` + `tests/audio/` の 74 test に regression なし。

### Changed

#### **BREAKING** `StreamTranscriber` に engine-input low-energy gate (EnergyGate) を追加 (Issue [#292])

- **Before**: VAD segment は energy 不問で全て `engine.transcribe()` に渡され、低 RMS / 純ノイズ segment で hallucination ("うん"/"ピッ"/"え?"/"どうぞ" 等) が発生。
- **After**: `StreamTranscriber` の 3 callsites (`_transcribe_segment` / `_transcribe_segment_async` / `_transcribe_interim`) で共通 helper `_should_skip_low_energy(audio, kind)` を呼び、per-segment energy が threshold 未満なら `engine.transcribe()` を skip。
- **動機**: `#291` (NoiseGate 単位ミスマッチ) は NoiseGate 有効時の primary fix だが、NoiseGate は opt-in (default off) で大半のユーザーは VAD のみで防御。VAD false-positive segment が engine に渡って hallucination する経路が残っていた (Mega-Gorilla/livecap-gui#331 の root-cause の一つ)。実音源 pre-evaluation で parakeet_ja は **silent audio に対して 100% hallucination** を確認、EnergyGate を経由すると -45 dBFS threshold で 26% 削減 (stress test)。production 条件 (VAD default threshold) では VAD が一次防御として効き、本機能は副次防御として機能する。
- **API 変更点 (`StreamTranscriber.__init__`)**:
  - 新引数: `engine_min_rms_dbfs: float = -45.0` — threshold (dBFS)。`float("-inf")` で完全 opt-out。
  - 新引数: `engine_energy_metric: str = "max_frame_rms"` — 4 metric から選択 (`max_frame_rms` / `whole_rms` / `p95_frame_rms` / `top3_frame_rms`)。default は VAD padding 希釈に耐性の `max_frame_rms`。
  - 新引数: `engine_energy_frame_ms: float = 32.0` — frame-based metric の窓長 (ms)。
- **CLI 変更**:
  - `transcribe` に `--engine-min-rms` / `--engine-energy-metric` / `--engine-energy-frame-ms` の 3 flag を追加。`--engine-min-rms` には custom type を実装し `off` / `disabled` / `none` 文字列を `float("-inf")` に map (argparse の leading-`-` value 制約のため bare `-inf` は不可、`=-inf` か `off` を使う)。
  - `levels` に `--engine-min-rms-margin` flag を追加 (default `+6 dB`)。`suggested_engine_min_rms_dbfs` の margin を user 任意に調整可能。
- **`NoiseAnalysis` 変更**:
  - 新 field: `suggested_engine_min_rms_dbfs: float` (= `noise_rms_p95_db + engine_min_rms_margin_db`)。CLI `levels` で 1 回の calibration から peak-unit / RMS-unit 両方の suggested 値が得られる。
  - `analyze_noise_samples()` に optional `engine_min_rms_margin_db` キーワード引数を追加 (default `ENGINE_MIN_RMS_SAFETY_MARGIN_DB = 6.0`)。
- **新公開 API (`livecap_cli.audio`)**:
  - `ENGINE_MIN_RMS_SAFETY_MARGIN_DB = 6.0` (定数)
  - `ENERGY_METRICS = ("max_frame_rms", "whole_rms", "p95_frame_rms", "top3_frame_rms")`
  - `_segment_energy_dbfs(audio, sample_rate, metric, frame_ms) -> float` (helper、user-configurable metric/frame で per-segment energy を測定)
- **Telemetry**: `StreamTranscriber.close()` 時に drop counter の内訳 (`final_sync` / `final_async` / `interim`) を `logger.info` で 1 行サマリ。silent failure 防止。
- **Migration**:
  - 既存挙動を完全に維持したい場合: `StreamTranscriber(engine=..., engine_min_rms_dbfs=float("-inf"))` または CLI `--engine-min-rms off` で opt-out。
  - 通常はデフォルト (`-45.0`) で問題なく動作 (synthetic regression + 実音源プローブで通常会話・小声・ささやきレベル speech は pass を確認)。whisper 録音など特殊用途は閾値を下げる (`--engine-min-rms -50` 等) か opt-out。
- **検証**:
  - `tests/audio/test_analysis.py::TestSegmentEnergyDbfs` (10 cases) — 4 metric ごとの動作 / fallback / 物理的妥当性。
  - `tests/audio/test_energy_gate_regression.py` (新規 7 cases) — synthetic fixture (silent noise / speech-like burst / padded short utterance) で default threshold の drop/pass を assert。VAD padding 希釈に max_frame が耐性を持ち、whole_rms は希釈で false-drop することを documentation。
  - `tests/transcription/test_stream.py::TestEnergyGate` (10 cases) — 3 callsites (sync / async / interim) で `engine.transcribe()` が呼ばれない / 呼ばれることを mock の `call_count` で検証 + opt-out + invalid arg validation + close() log。
  - `tests/core/cli/test_cli.py::TestEnergyGateFlags` (12 cases) — `--engine-min-rms` の 4 parse パターン (numeric / off / disabled / =-inf) + invalid raises + metric choices + frame-ms parse + help text 可視性。
- **限界**:
  - EnergyGate は **silver bullet ではない**。実音源プローブで parakeet_ja は 73 silent windows で 100% hallucinate し、-45 dBFS threshold (max_frame_rms) で削減できるのは 26% のみ (transient を含む noisy silence は max frame で pass する)。完全防御には `--noise-gate` との併用 + VAD threshold 適切化が必要。
  - Engine choice が hallucination 耐性に大きく影響 (Parakeet 100% vs ReazonSpeech は known 低い)。
- **将来 follow-up**:
  - `BaseEngine` 側へのガード追加 (`StreamTranscriber` を経由しない advanced user 向け)
  - top-k metric の k=3 から user-configurable に拡張

#### **BREAKING** `NoiseAnalysis` / `analyze_noise_samples()` を peak-based calibration に置換 (Issue [#291])

- **Before**: `analyze_noise_samples(samples_db, sample_rate_hz)` →
  `suggested_threshold_db = noise_peak (chunk RMS p95) + 10 dB`
- **After**: `analyze_noise_samples(samples_db, peak_samples_db, sample_rate_hz)` →
  `suggested_threshold_db = peak_p95 (per-chunk |x|.max() p95) + 6 dB`
- **動機**: `NoiseGate` (`livecap_cli/audio/noise_gate.py`) の envelope follower
  は per-sample peak を追跡するが、calibration は chunk RMS を計測していた → unit
  mismatch により impulsive noise (キーボード/呼吸/breath bursts) で threshold が
  peak の下に潜り、無音時 hallucination ("あ"/"うん"/"ピッ") を引き起こしていた
  (Mega-Gorilla/livecap-gui#331 root-cause)。White noise の crest factor ≈ 11 dB
  が偶然 `+10` で吸収されていたが、impulsive noise では crest factor がより大きく
  破綻する。
- **API 変更点**:
  - `NoiseAnalysis` 新 field: `peak_p95_db: float` (per-chunk `|x|.max()` の 95%ile)
  - `NoiseAnalysis` 改名: `noise_peak_db` → `noise_rms_p95_db` (unit を field 名に明示)
  - `NoiseAnalysis` 削除: `safe_zone_min_db` (新 `suggested_threshold_db` と 1 dB 差で意味崩壊)
  - `analyze_noise_samples()` 新 required 引数: `peak_samples_db` (Optional=None の
    legacy default は無し: pre-1.0 backward-compat policy で旧バグ温存の flag は不可)
  - 新 module-level 定数: `livecap_cli.audio.PEAK_SAFETY_MARGIN_DB = 6.0`
  - `danger_zone` は据え置き (RMS-unit diagnostic として docstring で明記)
- **Migration**:
  ```python
  # 旧
  rms_db_list = [20*log10(rms(chunk)) for chunk in chunks]
  a = analyze_noise_samples(rms_db_list)

  # 新
  rms_db_list  = [20*log10(rms(chunk))    for chunk in chunks]
  peak_db_list = [20*log10(|chunk|.max()) for chunk in chunks]
  a = analyze_noise_samples(rms_db_list, peak_db_list)
  # a.noise_peak_db    -> a.noise_rms_p95_db
  # a.safe_zone_min_db -> (削除; suggested_threshold_db を直接使用)
  # a.peak_p95_db      -> 新 field; threshold の基準
  ```
  CLI `levels` は内部で per-chunk peak を収集するように移行済みのため、
  外部から CLI を呼ぶ場合の変更は不要 (JSON schema のみ変更)。
- **検証**: `tests/audio/test_noise_gate_calibration.py` (新規) で synthetic
  impulsive noise を旧 / 新 threshold それぞれで NoiseGate に通し、旧で gate
  が開く / 新で閉じ続けることを assert する end-to-end 回帰テストを追加。
- **GUI ペア PR**: livecap-gui 側は `core/noise_statistics.py` を削除し本 API
  に委譲する PR を別 issue (Mega-Gorilla/livecap-gui#335 の対) で受ける。
- **将来 follow-up**: NoiseGate の envelope follower を calibration 入力に対して
  simulate し envelope の 95%ile を取れば margin を 1-2 dB に縮められる ([#283]
  と組で別 issue 化予定)。

#### **BREAKING** `NoiseGate` デフォルト `release_ms` 変更 (Issue [#283] PR C)

- **Before** (PR #279 / PR #281 / PR #282): `release_ms=30`
- **After**: `release_ms=100`
- **動機**: PR #282 で導入された hard-mute と短い release の組み合わせが、aggressive な閾値 (-25/-17 dB) で whisper 系エンジンの fragmentation ハルシネーション (「んんん...」loop) を引き起こす。A/B 実測で `release_ms=100` により完全解消を確認 (316→102, 299→96 chars)
- **Migration**: 旧挙動を明示的に再現するには `release_ms=30` を直接渡す:
  ```python
  NoiseGate(release_ms=30)  # pre-PR-C default
  ```
  CLI の場合:
  ```bash
  livecap-cli transcribe --noise-gate --noise-gate-release 30 ...
  ```
- **検証結果**: `docs/benchmarks/noise-gate-ab.md` に更新後のテーブル掲載

#### **BREAKING** `NoiseGate` 既定挙動変更 (Issue [#280] PR B)

- **Before** (PR [#279] / PR [#281]): 単一閾値 + `-60 dB` soft-mute
- **After** (PR B): 自動ヒステリシス (`threshold_db - 6 dB`) + hard-mute (出力ゼロ)
- **動機**: PR #281 の A/B 検証で、PR #281 までの挙動が whisper 系エンジンで flicker によるハルシネーション暴走 ("どうもどうも..." 等) を引き起こすことが実証された
- **Migration**: 過去挙動を明示的に再現するには以下を指定:
  ```python
  NoiseGate(
      threshold_db=-35,
      close_threshold_db=-35,  # single-threshold (ヒステリシス無効)
      noise_floor_db=-60,      # soft-mute
  )
  ```
  CLI の場合:
  ```bash
  livecap-cli transcribe --noise-gate \
      --noise-gate-threshold -35 \
      --noise-gate-close-threshold -35 \
      --noise-gate-floor -60 \
      ...
  ```

新規オプション (既存呼び出しは無変更で動作、挙動のみ変化):

- `NoiseGate` / `transcribe` CLI に `close_threshold_db` / `--noise-gate-close-threshold` を追加 (ヒステリシス制御)
- `NoiseGate` / `transcribe` CLI に `noise_floor_db` / `--noise-gate-floor` を追加 (ゲート閉鎖時の減衰量制御)
- 初期化ログが resolved 値 (open/close/noise_floor) を出力するように改善 (ポリシー準拠)

既知の follow-up:

- `release_ms=30` は PR B の新しい gate 挙動 (hard-mute による clean silence) に対して短すぎるため、攻撃的な閾値で fragmentation hallucination が発生することがあります。`--noise-gate-release 100` または `200` で回避可能。デフォルト値の変更は別 issue で対応予定。

### Added

#### Noise Gate & Calibration ([#278], [#279], [#280], [#281])

- `livecap_cli.audio.NoiseGate` — 音量ベースのリアルタイムノイズゲート（サンプル単位エンベロープフォロワー、numba JIT で < 0.1 ms / 100 ms chunk）。VAD 前段に挿入してハルシネーションを抑制。
- `transcribe` サブコマンドに `--noise-gate` / `--noise-gate-threshold` / `--noise-gate-attack` / `--noise-gate-release` オプションを追加。
- `livecap-cli levels` サブコマンド — マイク入力レベルを dB 単位でリアルタイム表示し、環境ノイズから推奨閾値を算出。
  - `--duration N` — N 秒後に自動停止（非対話モード）。
  - `--json` — `NoiseAnalysis` を JSON で stdout に出力（GUI / スクリプト連携向け）。
- `livecap_cli.audio.analysis` モジュール — `NoiseAnalysis` dataclass と `analyze_noise_samples()` 関数（CLI / GUI 共通キャリブレーション API）。
- 推奨閾値アルゴリズム: `noise_peak (95%ile) + 10 dB`（[livecap-gui PR #294](https://github.com/Mega-Gorilla/livecap-gui/pull/294) の実測に基づく保守的マージン）。「死のゾーン」(`noise_floor ± 5 dB`) を回避する設計。

**段階導入について**: PR #281 は **キャリブレーション API 基盤の先行導入** (Issue #280 C-3 + C-4) です。NoiseGate 本体の安定化 ([Issue #280](https://github.com/Mega-Gorilla/livecap-cli/issues/280) の C-1 ヒステリシス + C-2 hard-mute) は follow-up PR で提供予定。現行実装 (単一閾値 + `-60 dB` soft-mute) では、閾値が speech peak 付近の場合に flicker で逆にハルシネーションを誘発することがあります。特に `whispers2t` エンジンで影響が大きく、`reazonspeech` / `parakeet_ja` / `qwen3asr` は影響を受けにくいことが A/B テストで確認されています (PR #281 comments 参照)。暫定対応として、低 SNR 環境では `levels` の推奨値より保守的な値の使用、または別エンジンの利用を推奨します。

#### Phase 6: CLI Subcommand Structure ([#74], [#201])

New CLI with subcommand architecture:

| Command | Description |
|---------|-------------|
| `livecap-cli info` | Display installation diagnostics |
| `livecap-cli devices` | List audio input devices |
| `livecap-cli engines` | List available ASR engines |
| `livecap-cli translators` | List available translators |
| `livecap-cli transcribe` | Transcribe audio (file or realtime) |

**transcribe options:**
- `<file> -o <output.srt>` - File transcription to SRT
- `--realtime --mic <id>` - Realtime microphone transcription
- `--translate <id> --target-lang <lang>` - Translation support
- `--vad <auto|silero|tenvad|webrtc>` - VAD backend selection
- `--engine <id>` - ASR engine selection
- `--device <auto|gpu|cpu>` - Device selection

**Package extras:**
- `recommended`: Google translation (deep-translator)
- `all`: All optional dependencies

#### Phase 5: Engine Optimizations ([#73], [#194], [#196], [#197])

- Template Method pattern for `BaseEngine` with standardized lifecycle
- Progress reporting during model loading (0-100%)
- Model memory caching for faster subsequent loads
- Library preloading for reduced import time
- Standardized cleanup and resource management

#### Phase 4: Translation Support ([#72], [#180], [#181], [#182], [#184], [#186])

**Translators:**
- `google` - Google Translate ([#180])
- `opus_mt` - Helsinki-NLP Opus-MT local models ([#181])
- `riva_instruct` - NVIDIA Riva Translate 4B Instruct ([#182])

**Features:**
- Context-aware translation with sentence buffering
- `StreamTranscriber` translation integration ([#184])
- `FileTranscriptionPipeline` translation integration ([#186])
- Configurable timeout via `LIVECAP_TRANSLATION_TIMEOUT`
- Async translation deadlock prevention ([#189])

#### Phase 3: Package Structure ([#71])

- Reorganized module structure under `livecap_cli/`
- Clear separation: `engines/`, `vad/`, `transcription/`, `translation/`
- Unified public API exports in `__init__.py`

#### Phase 2: API Unification ([#70])

- `TranscriptionResult` dataclass replacing `TranscriptionEventDict`
- `VADConfig` dataclass for VAD parameters
- `EngineFactory.create_engine(engine_type, device, **options)` API
- Consistent error handling with `TranscriptionError`, `EngineError`

#### Phase 1: Realtime Transcription ([#69], [#65], [#66], [#67], [#68])

**Core components:**
- `StreamTranscriber` - VAD + ASR streaming orchestration ([#65])
- `VADProcessor` - Pluggable VAD with state machine ([#66])
- `TranscriptionResult` / `InterimResult` - Unified result types ([#67])
- `AudioSource` / `FileSource` / `MicrophoneSource` - Audio abstraction ([#68])

**VAD backends:**
- Silero VAD (default, neural network-based)
- WebRTC VAD (fast, low memory)
- TenVAD (optimized for Japanese)

**Language optimization:**
- `VADProcessor.from_language("ja")` - Auto-select optimal VAD
- Benchmark-based presets for Japanese and English

#### File Transcription

- `FileTranscriptionPipeline` for batch processing
- SRT subtitle output format
- FFmpeg integration for audio extraction
- Translation integration for bilingual subtitles

### Changed

#### Breaking Changes

| Before | After |
|--------|-------|
| Package: `livecap-core` | Package: `livecap-cli` |
| Module: `livecap_core` | Module: `livecap_cli` |
| CLI: `livecap-core --info` | CLI: `livecap-cli info` |
| CLI: `livecap-core --as-json` | CLI: `livecap-cli info --as-json` |

#### API Changes

- `TranscriptionEventDict` → `TranscriptionResult` dataclass
- Engine creation unified to `EngineFactory.create_engine()`
- VAD configuration via `VADConfig` instead of dict
- `detect_device()` returns `str` instead of `Tuple` ([#175])

### Deprecated

- `TranscriptionEventDict` (use `TranscriptionResult`)
- `languages.py` module (use `langcodes` for BCP-47) ([#173])

### Removed

- Old flag-based CLI interface (`--info`, `--ensure-ffmpeg`, `--as-json`)
- `livecap-core` entry point
- `livecap_core` module name
- `Languages.get_engines_for_language()` (use `EngineMetadata`) ([#171])

### Fixed

- GitHub Actions workflows updated for module rename ([#201])
- Integration test path filters updated
- Async translation deadlock in concurrent scenarios ([#189])
- Translation timeout handling improvements ([#187])
- OPUS-MT context disabled by default for stability ([#191])

### Security

- No security issues in this release

---

## Migration Guide

### From `livecap-core` to `livecap-cli`

#### 1. Update package installation

```bash
# Before
pip install livecap-core[engines-torch]

# After
pip install livecap-cli[engines-torch]

# Or use the recommended bundle:
pip install livecap-cli[recommended]
```

#### 2. Update imports

```python
# Before
from livecap_core import StreamTranscriber, EngineFactory
from livecap_core.vad import VADProcessor, VADConfig

# After
from livecap_cli import StreamTranscriber, EngineFactory
from livecap_cli.vad import VADProcessor, VADConfig
```

#### 3. Update CLI commands

```bash
# Before
livecap-core --info
livecap-core --as-json

# After
livecap-cli info
livecap-cli info --as-json

# New commands
livecap-cli devices
livecap-cli engines
livecap-cli translators
livecap-cli transcribe input.mp4 -o output.srt
livecap-cli transcribe --realtime --mic 0
```

#### 4. Update result handling (if using old dict API)

```python
# Before (TranscriptionEventDict)
result = {"text": "...", "start": 0.0, "end": 1.0}
print(result["text"])

# After (TranscriptionResult dataclass)
# result is now a dataclass with attributes
print(result.text)
print(result.start_time)
print(result.end_time)
print(result.to_srt_entry(index=1))
```

---

## Issue References

- Epic: [#64] - livecap-cli リファクタリング
- Phase 1: [#69] - リアルタイム文字起こし実装
- Phase 2: [#70] - API 統一と Config 簡素化
- Phase 3: [#71] - パッケージ構造整理
- Phase 4: [#72] - 翻訳機能実装
- Phase 5: [#73] - エンジン最適化
- Phase 6: [#74] - 依存関係整理・CLI・パッケージ名変更
- Docs: [#75] - ドキュメント更新

---

[Unreleased]: https://github.com/Mega-Gorilla/livecap-cli/compare/main...HEAD

[#64]: https://github.com/Mega-Gorilla/livecap-cli/issues/64
[#65]: https://github.com/Mega-Gorilla/livecap-cli/issues/65
[#66]: https://github.com/Mega-Gorilla/livecap-cli/issues/66
[#67]: https://github.com/Mega-Gorilla/livecap-cli/issues/67
[#68]: https://github.com/Mega-Gorilla/livecap-cli/issues/68
[#69]: https://github.com/Mega-Gorilla/livecap-cli/issues/69
[#70]: https://github.com/Mega-Gorilla/livecap-cli/issues/70
[#71]: https://github.com/Mega-Gorilla/livecap-cli/issues/71
[#72]: https://github.com/Mega-Gorilla/livecap-cli/issues/72
[#73]: https://github.com/Mega-Gorilla/livecap-cli/issues/73
[#74]: https://github.com/Mega-Gorilla/livecap-cli/issues/74
[#75]: https://github.com/Mega-Gorilla/livecap-cli/issues/75
[#171]: https://github.com/Mega-Gorilla/livecap-cli/pull/171
[#173]: https://github.com/Mega-Gorilla/livecap-cli/pull/173
[#175]: https://github.com/Mega-Gorilla/livecap-cli/pull/175
[#180]: https://github.com/Mega-Gorilla/livecap-cli/pull/180
[#181]: https://github.com/Mega-Gorilla/livecap-cli/pull/181
[#182]: https://github.com/Mega-Gorilla/livecap-cli/pull/182
[#184]: https://github.com/Mega-Gorilla/livecap-cli/pull/184
[#186]: https://github.com/Mega-Gorilla/livecap-cli/pull/186
[#187]: https://github.com/Mega-Gorilla/livecap-cli/pull/187
[#189]: https://github.com/Mega-Gorilla/livecap-cli/pull/189
[#191]: https://github.com/Mega-Gorilla/livecap-cli/pull/191
[#194]: https://github.com/Mega-Gorilla/livecap-cli/pull/194
[#196]: https://github.com/Mega-Gorilla/livecap-cli/pull/196
[#197]: https://github.com/Mega-Gorilla/livecap-cli/pull/197
[#201]: https://github.com/Mega-Gorilla/livecap-cli/pull/201
[#278]: https://github.com/Mega-Gorilla/livecap-cli/issues/278
[#279]: https://github.com/Mega-Gorilla/livecap-cli/pull/279
[#280]: https://github.com/Mega-Gorilla/livecap-cli/issues/280
[#281]: https://github.com/Mega-Gorilla/livecap-cli/pull/281
[#283]: https://github.com/Mega-Gorilla/livecap-cli/issues/283
[#291]: https://github.com/Mega-Gorilla/livecap-cli/issues/291
[#292]: https://github.com/Mega-Gorilla/livecap-cli/issues/292
[#230]: https://github.com/Mega-Gorilla/livecap-cli/issues/230
[#295]: https://github.com/Mega-Gorilla/livecap-cli/issues/295
[#314]: https://github.com/Mega-Gorilla/livecap-cli/issues/314
[#362]: https://github.com/Mega-Gorilla/livecap-cli/issues/362
[#363]: https://github.com/Mega-Gorilla/livecap-cli/issues/363
[#365]: https://github.com/Mega-Gorilla/livecap-cli/issues/365
[#366]: https://github.com/Mega-Gorilla/livecap-cli/issues/366
[#375]: https://github.com/Mega-Gorilla/livecap-cli/issues/375
[#378]: https://github.com/Mega-Gorilla/livecap-cli/issues/378
[#380]: https://github.com/Mega-Gorilla/livecap-cli/issues/380
[#386]: https://github.com/Mega-Gorilla/livecap-cli/issues/386
[#395]: https://github.com/Mega-Gorilla/livecap-cli/issues/395
[#398]: https://github.com/Mega-Gorilla/livecap-cli/issues/398
[#190]: https://github.com/Mega-Gorilla/livecap-cli/issues/190
[#402]: https://github.com/Mega-Gorilla/livecap-cli/issues/402
