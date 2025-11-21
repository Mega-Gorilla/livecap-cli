# LiveCap Core テストガイド

リポジトリのテスト構成、必要な依存オプション、変更内容に応じた実行パターン、CI ワークフローとの対応関係をまとめています。

## ディレクトリ構成

| Path | Scope |
| --- | --- |
| `tests/core/cli` | CLI エントリーポイント、設定ダンプ、対話 I/O |
| `tests/core/config` | 設定ビルダーとデフォルト |
| `tests/core/engines` | EngineFactory 配線とアダプター登録 |
| `tests/core/i18n` | 翻訳テーブルとロケールフォールバック |
| `tests/core/resources` | FFmpeg・モデルキャッシュ・uv プロファイル管理 |
| `tests/transcription` | 変換ヘルパーのユニットテスト（Live_Cap_v3 互換のため残置） |
| `tests/integration/transcription` | 音声・ディスク・モデル DL を含むエンドツーエンド |
| `tests/integration/engines` | 実音源を使ったエンジンスモーク（CPU/GPU 切替可） |

実体のあるバイナリやモデルを使うシナリオは `tests/integration/` に置きます。これらも `pytest tests` で走るため、極力決定的に保ち、フラグで明示的に制御します。

## 依存プロファイル（`pyproject.toml` の extras）

| Extra | 説明 |
| --- | --- |
| `translation` | 言語パックとテキスト処理依存 |
| `dev` | pytest・型・lint 用ツール |
| `engines-torch` | Whisper / ReazonSpeech など Torch 系エンジン |
| `engines-nemo` | Parakeet / Canary など NVIDIA NeMo 系 |

日常開発は `translation` + `dev` が基本。特定エンジンを動かすときに対応する extra を追加します。

## テスト実行の基本

依存インストール（uv 推奨）:

```bash
uv sync --extra translation --extra dev
```

CI と同じデフォルトスイート:

```bash
uv run python -m pytest tests
```

## 変更に応じたテスト実行ガイド

| 変更内容 | 手元での推奨コマンド | 推奨 CI ワークフロー / ジョブ | 備考 |
| --- | --- | --- | --- |
| CLI / 設定のみ | `uv run python -m pytest tests/core/cli tests/core/config` | `Core Tests`（Linux/Windows） | FFmpeg 依存なし |
| パイプライン・リソース（FFmpeg 等） | `uv run python -m pytest tests/integration -m "not engine_smoke"` | `Integration Tests` の `transcription-pipeline` ジョブ | 実 FFmpeg が必要（下記参照） |
| エンジン（CPU） | `uv sync --extra translation --extra dev --extra engines-torch`<br>`uv run python -m pytest tests/integration/engines -m "engine_smoke and not gpu"` | `Integration Tests` の `engine-smoke-cpu` ジョブ | Whisper / ReazonSpeech の実音源スモーク |
| エンジン（GPU） | `LIVECAP_ENABLE_GPU_SMOKE=1 uv sync --extra translation --extra dev --extra engines-torch --extra engines-nemo`<br>`uv run python -m pytest tests/integration/engines -m "engine_smoke and gpu"` | `Integration Tests` の `engine-smoke-gpu` ジョブ（self-hosted Linux/Windows） | レポジトリ変数 `LIVECAP_ENABLE_GPU_SMOKE=1` でジョブ起動。強制 fail は `LIVECAP_REQUIRE_ENGINE_SMOKE=1` |
| 自社ランナーの環境確認 | ― | `Verify Self-Hosted Linux Runner` / `Verify Self-Hosted Windows Runner` | FFmpeg / Python / uv の前提チェック |

`Integration Tests` は `workflow_dispatch` で手動起動できます。GPU スモークを含める場合は、事前にレポジトリ変数で `LIVECAP_ENABLE_GPU_SMOKE=1` を設定してください。フェイルファストしたい場合は `LIVECAP_REQUIRE_ENGINE_SMOKE=1` も併用します。

## ターゲット別実行例

```bash
# CLI / 設定のみ
uv run python -m pytest tests/core/cli tests/core/config

# エンジンの配線（Torch 系依存込み）
uv sync --extra translation --extra dev --extra engines-torch
uv run python -m pytest tests/core/engines
```

## Integration Tests と FFmpeg セットアップ

`tests/integration/` はデフォルトの `pytest tests` に含まれます。多くはスタブ FFmpeg を使いますが、Issue #21 由来の MKV 抽出リグレッションは実 FFmpeg を要求します。

実 FFmpeg を使う場合:

1. [ffbinaries-prebuilt](https://github.com/ffbinaries/ffbinaries-prebuilt/releases) などから FFmpeg/FFprobe を取得
2. `./ffmpeg-bin/` に `ffmpeg` / `ffprobe` を配置
3. 環境変数 `LIVECAP_FFMPEG_BIN="$PWD/ffmpeg-bin"` を設定（PowerShell: `$env:LIVECAP_FFMPEG_BIN="$(Get-Location)\ffmpeg-bin"`）

`ffmpeg-bin/` は git 無視対象なので、各自のバイナリをコミットせずに置けます。

### エンジンスモーク（実音源）と GPU 切替

- すべての実音源スモークには `engine_smoke` マークが付き、GPU 必須ケースには追加で `gpu` マークが付きます。
- GPU ケースは `LIVECAP_ENABLE_GPU_SMOKE=1` かつ CUDA 利用可のときのみ実行し、それ以外は skip。
- 依存不足・モデル未キャッシュ・CUDA なしでも失敗扱いにしたい場合は `LIVECAP_REQUIRE_ENGINE_SMOKE=1` を指定します。
- CI では `Integration Tests` ワークフロー内で CPU/GPU スモークを分割実行します（GPU ジョブは self-hosted かつ環境変数が有効なときのみ起動）。

## CI 対応表

- `Core Tests`: Python 3.10/3.11/3.12 で `pytest tests`（integration 含む）。`LIVECAP_FFMPEG_BIN` を用意して MKV 回りもオフライン解決。
- `Optional Extras` ジョブ: `engines-torch` / `engines-nemo` のインストール検証（存在する場合）。
- `Integration Tests`: 手動/週次で実行。`ffmpeg-bin` を用意し、パイプライン統合テストとエンジンスモークを CPU/GPU に分離。GPU ジョブはレポジトリ変数 `LIVECAP_ENABLE_GPU_SMOKE=1` かつ self-hosted GPU ランナーがあるときのみ起動。
- `Windows Core Tests`: `windows-latest` で `pytest tests`。`translation`+`dev` 依存を使用。
- `Verify Self-Hosted Linux Runner` / `Verify Self-Hosted Windows Runner`: self-hosted ランナーの Python/uv/FFmpeg 前提を確認する手動ワークフロー。

### ランナー在庫（現状）

| Workflow | Runner | FFmpeg 準備 |
| --- | --- | --- |
| `Core Tests` / `Integration Tests`（transcription-pipeline, engine-smoke-cpu） | GitHub-hosted `ubuntu-latest` | `apt-get install ffmpeg` 後に `./ffmpeg-bin/` へ配置（Integration Tests では OS ごとにキャッシュ） |
| `Windows Core Tests` | GitHub-hosted `windows-latest` | gyan.dev のポータブル版を composite action で `.\ffmpeg-bin\` へ展開 |
| `Integration Tests` engine-smoke-gpu | Self-hosted Linux/Windows (GPU) | `setup-livecap-ffmpeg` で `./ffmpeg-bin/` に展開（キャッシュ有効）。Python/torch/ドライバはランナー側で事前用意 |
| `Verify Self-Hosted Linux Runner` | self-hosted Linux | ffbinaries 由来のポータブル FFmpeg を配置 |
| `Verify Self-Hosted Windows Runner` | self-hosted Windows | gyan.dev 由来のポータブル FFmpeg を配置 |

ローカル Windows デバッグ時は、同様に [ffbinaries-prebuilt](https://github.com/ffbinaries/ffbinaries-prebuilt/releases) の 6.1 ビルドを `ffmpeg-bin/` に展開し、`LIVECAP_FFMPEG_BIN` をそのパスに向けてください。self-hosted GPU ランナー（Linux/Windows）を使う場合は、CUDA 対応ハードウェア、事前インストール済み Python（Windows は `python3.12` をピン）、torch・NVIDIA ドライバ、そして `LIVECAP_ENABLE_GPU_SMOKE=1`（必要に応じて `LIVECAP_REQUIRE_ENGINE_SMOKE=1`）が必要です。

ワークフローや依存プロファイルを変更した場合は、本ドキュメントも併せて更新してください。
