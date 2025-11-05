# LiveCap Core Release & Dependency Workflow

この文書では `livecap-core` パッケージを維持する際の依存管理とリリース手順をまとめます。Phase 3 で合意した運用を CI/ドキュメントに反映するためのリファレンスです。

> **Python 対応バージョン**: 現状は `>=3.10,<3.13` をサポート対象としています。`engines-nemo` extra が `ml-dtypes` に依存しており、Python 3.13+ では numpy 2 系が必須となるため、PyPI 公開までは 3.13 を除外しています。

## 1. `uv lock` の更新ポリシー

| 頻度 | コマンド | 補足 |
|------|-----------|------|
| 毎月第 1 週 | `uv lock --upgrade` | 通常の依存アップデート。PR では `uv pip check` と `uv run pytest ...` を実行してマージ。 |
| 緊急パッチ | `uv lock --upgrade "<package>==<patched>"` | セキュリティ／重大バグ対応。PR で変更点と影響範囲を明記する。 |

更新後は `uv sync --extra translation` で仮想環境を再構成し、「Core テスト」ワークフロー（後述）が通ることを確認します。

## 2. TestPyPI ドライラン手順

安定版リリース候補 (RC) の前に以下を実施します。

```bash
# 1. 環境構築
uv sync --extra translation --extra engines-torch

# 2. 単体テスト
uv run pytest tests/core tests/transcription/test_transcription_event_normalization.py

# 3. パッケージ生成
uv run python -m build

# 4. TestPyPI へアップロード
uv run python -m twine upload --repository testpypi dist/*

# 5. インストール確認
uv pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple livecap-core
```

> **Note:** `engines-nemo` extra は CUDA 依存のため、Linux GPU 環境で別途検証する。現状は `translation` / `engines-torch` を最優先で確認します。

## 3. GitHub Actions

`.github/workflows/core-tests.yml` は pull request および main への push 時に以下を実行します。

1. `astral-sh/setup-uv` で `uv` をセットアップ
2. `uv sync --extra translation --extra dev`
3. `./.venv/bin/python -m pytest tests/core tests/transcription/test_transcription_event_normalization.py`

必要に応じて `workflow_dispatch` で手動トリガーし、追加の extras (`engines-torch`) を試験することもできます。

## 4. リリースブランチ運用

1. `main` から `release/1.0` ブランチを作成。
2. バージョンを `1.0.0aN → 1.0.0bN → 1.0.0rcN → 1.0.0` の順で更新し、それぞれタグ `core-<version>` を付与。
3. 各段階で上記 TestPyPI ドライランを実行し、結果をリリースノートに記録。
4. 安定版 (`1.0.0`) の公開を判断したら PyPI へ `uv publish` もしくは `twine upload` を行う。

## 5. GUI リポジトリとの連携

- `Live_Cap_v3` 側で `livecap-core` を取り込む CI を整備し、タグ公開時に自動バンプする予定（Issue #148 TODO）。
- リリースチェックリストにはこのドキュメントのリンクを追加し、両リポジトリで同じ手順を参照するようにします。
