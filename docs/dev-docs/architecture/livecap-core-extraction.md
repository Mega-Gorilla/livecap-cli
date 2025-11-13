# LiveCap Core Extraction Plan / LiveCap Core 分離計画

> **Context / 背景**  
> Issue #148 で定義された Phase 3 の準備タスクとして、`livecap_core` を新規リポジトリへ切り出すための棚卸しと移行順序を整理する。

## 1. Scope Inventory / 対象インベントリ

### Code & Tests / コードとテスト
- `livecap_core/**` – Core ランタイム一式（`config/`, `resources/`, `transcription/`, `i18n.py` などを含む）
- `tests/core/**` – Core API を直接検証する pytest 群
- Cross-package fixtures & tests  
  - `tests/transcription/test_file_transcriber_worker.py`  
  - `tests/transcription/test_transcription_event_normalization.py`  
  - `tests/transcription/test_live_transcriber_callbacks.py`  
  - GUI 依存フィクスチャ（`tests/conftest.py` 経由）で Core を利用している箇所は、分離後のテスト配置を再検討する
- CLI エントリーポイント: `pyproject.toml` の `[project.scripts] livecap-core = "livecap_core.cli:main"`
- Runtime assets: `licenses/`, `THIRD_PARTY_LICENSES.md` に列挙済み OSS 情報（デュアルライセンス運用に連動）

### Documentation / ドキュメント
- `docs/dev-docs/architecture/core-api-spec.md`
- `docs/dev-docs/architecture/core-config-boundaries.md`
- `docs/dev-docs/architecture/core-separation-proposal.md`
- `docs/dev-docs/architecture/phase0-prerequisites.md` / `phase1-architecture-plan.md` – 歴史的経緯の整理が必要
- `docs/dev-docs/licensing-archive/**` – 新リポジトリ側で参照できるよう複製、または Submodule/リンクで共有
- 公開向け metadata: `docs/update-notes/`, `docs/dev-docs/technical/refactoring/unify_transcription_dict_format.md` など Core の内部仕様を説明しているファイル

### Tooling & Pipelines / ツール・パイプライン
- `uv.lock`, `pyproject.toml`, `requirements.txt` – Core 専用に再構成が必要
- GitHub Actions 連携: `.github/workflows/` は新設リポジトリに最低限の pytest smoke を実装
- Build 手順: `python -m build`, `uv sync --extra ...` のフローを新リポジトリで再現

## 2. Coupling Audit / 依存関係の棚卸し

### GUI ↔ Core Imports
- `src/engines/**`（`base_engine.py`, `engine_factory.py`, `shared_engine_manager.py`, 各エンジン実装）
- `src/gui/widgets/transcriber/**`（`core_managers.py`, `transcription_data.py`, `display_manager.py`）
- `src/gui/core/managers/**`（`subtitle_manager.py`, `translation_manager.py`, `utils/subtitle_flow.py`）
- `src/config/core_config_builder.py`, `src/config/config_loader.py`
- Translators & localization: `src/translation/translator.py`, `src/translation/backends/*.py`, `src/localization/translator.py`
- Application entrypoints: `src/main.py`, `src/gui_main.py`, `src/file_transcriber.py`, `src/audio/transcription/multi_source_transcriber.py`

### Non-Python Assets / 非Python資産
- `config.yaml` の Core 関連セクション（`audio`, `transcription`, `engines`, `translation` 等）は GUI 向けプリセットも混在している
- `html/`, `fonts/`, `models/` など GUI 同梱アセットは Core 切り出し対象外。Core が依存するファイルのみ新リポジトリで扱う想定

### Known Risk Items / 既知の論点
- VCS 依存 (`ten-vad`, `reazonspeech-k2-asr`) – PyPI 公開方針として optional extras か、別配布パッケージ準備が必要
- `livecap_core/resources/ffmpeg_manager.py` で参照する FFmpeg バイナリの扱い（環境変数での切り替えドキュメント化が必要）
- GUI 固有の設定スキーマと Core TypedDict (`livecap_core/config/schema.py`) の同期方法

### Versioning & Lockfile Strategy / バージョン・ロック方針

**PEP 440 / Git タグ整合**
- `pyproject.toml` の `version` は Core リポジトリの `main` で常に `1.0.0.dev0`（開発版）を指し、ブランチ切替時にのみ更新する。GUI リポジトリは `livecap-core` の公開バージョンに追従。
- プレリリースは PEP 440 形式を採用し、段階ごとに `1.0.0aN`（alpha）、`1.0.0bN`（beta）、`1.0.0rcN`（release candidate）へバージョンを引き上げる。タグは `core-1.0.0a1` など `core-<semver>` で揃える。
- 安定版リリース時に `1.0.0` へ昇格し、タグは `core-1.0.0`。GUI 側は同タグに合わせた依存バンプ PR を起票。

**ブランチ運用**
- `main`: 常に `1.0.0.dev0`。日常開発と依存更新を受け入れる。
- `release/<major.minor>` ブランチ: リリース準備時に作成し、alpha → beta → rc → final の順でバージョンを進める。各ステップでタグを付与し、TestPyPI 検証を実施。

**`uv lock` 再生成ポリシー**
- `main` では月次（毎月第1週）に `uv lock --upgrade` を実行し、依存アップデートをまとめて取り込む。結果は単独 PR とし、`uv pip check` / `uv sync --extra translation` で smoke 確認。
- セキュリティアドバイザリや重大バグに関しては ad-hoc で `uv lock --upgrade package==<patched>` を実行し、リリースノートに追記。
- リリースブランチではバージョンバンプ直後に `uv lock --locked` で差分確認、必要であれば `uv lock --upgrade --prerelease allow` を使用。生成後に `uv sync --extra transcription --extra translation` → `pytest tests/core` → `python -m build` → `twine upload --repository testpypi` を順にドライラン。

**CI / ドキュメント反映**
- 新 `livecap-core` リポジトリでは `uv lock` 更新用のワークフローを用意し、PR にロックファイル差分と `uv pip check` ログを添付する。
- GUI リポジトリでは `main` マージ時に `dependabot` / `renovate` を使わず、`livecap-core` のタグ更新をトリガーに依存バンプ PR を自動生成する GitHub Actions を検討。
- コントリビュータ向けガイド（`docs/dev-docs/architecture/core-separation-proposal.md` など）へ上記のバージョニング・ロック運用を転記し、Issue #148 完了時に cross-link する。

### Interface Focus: EngineFactory & SubtitleManager / インターフェース重点レビュー

**Current coupling / 現状の結合点**
- `src/engines/engine_factory.py` は GUI サイドに配置されているが、`config.core_config_builder.build_core_config` を経由して Core スキーマへ正規化し、`livecap_core.i18n.translate` を利用してメタデータの多言語化も担っている。
- エンジンの実体は GUI 側の `engines/*.py` に存在し、`EngineMetadata` (`src/engines/metadata.py`) も同一ツリー内で管理。Core から CLI 経由で利用する場合も GUI モジュールに依存する構造になっている。
- `SubtitleManager` (`src/gui/core/managers/subtitle_manager.py`) は PySide 依存の GUI コンポーネントだが、`livecap_core.transcription_types.create_subtitle_event` を呼び出してイベント辞書を生成し、Core 側が定義する `SubtitleEventDict` を前提に送信経路を構築している。

**Abstraction requirements / 抽象化要件**
- Config 正規化: `build_core_config` を Core リポジトリへ移す、もしくは `livecap_core.config.normalize(raw_config)` のような公開 API を用意し、GUI からは「正規化済みの CoreConfig を渡す」だけにする。
- Engine レジストリ: `EngineMetadata` とエンジンファクトリ（`create_engine`, `get_engine_info` 等）を Core パッケージへ移し、GUI 側では `livecap_core.engines` を参照する薄いラッパーに留める。表示名翻訳は Core の `translate()` に任せれば両リポジトリで統一運用できる。
- インポート経路: `_get_engine_class` は `importlib.import_module(..., package="engines")` に依存しているため、レジストリ移行後は `livecap_core.engines` パッケージを参照するよう更新し、GUI 側の遅延ロードが壊れないようにする。
- GUI 専用拡張（OBS/VRChat 中間設定）は GUI 層で注入する。Core の `create_engine` は純粋にエンジン初期化と設定適用に集中させる。
- Subtitle イベント: Core は `SubtitleEventDict` のスキーマと生成ヘルパーを提供し続け、GUI 側では `SubtitleManager` が「イベント辞書を受け取り送信サービスへ橋渡しする」役割に徹する。今後は `SubtitleSink` などの抽象インターフェースを用意し、Core からのイベント配信（WebSocket / CLI など）でも再利用可能にする。

**Action hooks / 今後の検討フック**
- Pre-split TODO で要求している `engine_factory` / `subtitle_manager` の抽象化レビューは、上記の API 再配置案をベースに具体的なモジュール構成図を作成する。
- SharedEngineManager 等、EngineFactory 経由で Core エンジンを取得している箇所を洗い出し、Core 側 API に切り替えた際の差分を設計（`src/engines/shared_engine_manager.py` ほか）。
- Subtitle 周りはイベントバス経由での分離を想定し、Core から GUI への通知を「イベント辞書のみ」に限定する方針を docs 化する。

### Test Relocation Plan / テスト移管方針

| Test file | Current dependency highlights | Target repo | Notes |
|-----------|------------------------------|-------------|-------|
| `tests/transcription/test_file_transcriber_worker.py` | Exercises `file_transcriber.TranscriptionWorker`, patches `EngineFactory`, pulls GUI translation strings, VAD adapters, and OBS-friendly messaging. | **GUI repo（現行リポジトリ）** | 残存。Core を外部依存に切り替えた後も pytestfixture で engine/pipeline をモックして動作確認する。 |
| `tests/transcription/test_live_transcriber_callbacks.py` | Stubs `LiveTranscriber` (`src/live_transcribe.py`) to assert CLI/GUI callback wiring. Uses the GUI shim `transcription.TranscriptionProgress/Result` re-export (facade over core types). | **GUI repo（現行リポジトリ）** | 残存。Core 分離後も shim の再輸出を維持しつつ `livecap_core` のイベント型と整合するよう確認が必要。 |
| `tests/transcription/test_transcription_event_normalization.py` | Targets `livecap_core.transcription_types.normalize_to_event_dict` exclusively. | **新 `livecap-core` リポジトリ** | コアテストへ移管（例: `tests/events/test_normalize_transcription.py`）。GUI 側では重複テスト不要。 |

移管時の留意点:
- Core リポジトリでは pytest の最小構成を用意し `tests/core/**` と同様に GUI 依存のフィクスチャを排除する。
- GUI リポジトリに残るテストは `pip install -e ../livecap-core` を前提にしつつ、`EngineFactory` モックで GUI 特有のハンドラを確認する。
- CI フロー整備時に双方のテストスイートが独立して通ること（Core: TestPyPI 前提、GUI: 外部依存としてインストール）をチェックリストに組み込む。

## 3. Migration Order / 移行順序

1. **Repository Split Prep**  
   - `split/livecap-core` ブランチで対象ファイル・依存の棚卸し（本ドキュメント）  
   - Core 関連のテスト/ドキュメント範囲を最小閉包となるよう確認
2. **New Repository Bootstrap**  
   - `livecap-core` リポジトリを作成  
   - 初期コミットに AGPLv3 LICENSE、README、`pyproject.toml (1.0.0-dev)`、`uv.` 系設定、`docs/roadmap.md`、`tests/` 雛形、`.github/workflows/test.yml` を含める
3. **Code Migration**  
   - 本リポジトリから `livecap_core/`, `tests/core/`, Core 関連 docs をコピー  
   - VCS 依存の扱いを整理し、`pyproject.toml` の extras から GUI 向け項目を排除
   - `uv lock --upgrade` → `uv sync --extra ...` → `pytest` smoke を新リポジトリで実行
4. **GUI Side Refactor**  
   - 本リポジトリで `livecap-core` を外部依存として扱うブランチを作成  
   - `pip install -e ../livecap-core` 前提の開発手順を docs 更新  
   - `src/` 側のインポートパスや設定受け渡しを確認し、必要があればラッパー層を導入
5. **Documentation & Communication**  
   - `docs/dev-docs/architecture` に進捗を反映（Issue #91/#135 と連携）  
   - Discord/Steam 告知テンプレート、FAQ、デュアルライセンス説明を整備  
   - Release checklist に TestPyPI ドライラン手順とロールバックポリシーを追記

## 4. Immediate Action Items / 直近アクション

1. `livecap_core` と GUI を接続している低レベル API（特に `engine_factory`, `subtitle_manager`）ごとに必要な抽象化を洗い出す  
2. `tests/transcription/` 内の Core 依存テストをどちらのリポジトリで維持するか決定し、テスト重複を避ける設計をまとめる  
3. `uv.lock` 再生成戦略とリリースバージョニング（1.0 系移行）のドラフトを Issue #148 コメントへ共有する

---

Prepared on branch `split/livecap-core` (2025-11-05).
