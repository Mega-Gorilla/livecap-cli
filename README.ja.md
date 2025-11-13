# LiveCap Core (日本語版)

LiveCap Core は LiveCap GUI やヘッドレスデプロイで利用されるランタイムです。ストリーミング文字起こしパイプライン、エンジンアダプター（Whisper / ReazonSpeech / NeMo など）、リソースマネージャ、設定ヘルパーを単体パッケージとして提供します。

- `livecap_core/config`: CLI/GUI で共有するデフォルト設定やバリデータ
- `livecap_core/transcription`: ストリーミング/ファイル変換パイプライン、イベント正規化ユーティリティ
- `livecap_core/engines`: Whisper / ReazonSpeech / Parakeet などのアダプタ（optional extras で依存を追加）

> ℹ️ **プロジェクト状況** – 現在も Live_Cap_v3 からモジュールを切り出し中です。1.0.0 RC までは API が変更される可能性があります。

## 必要環境

- Python **3.10 – 3.12**
- POSIX 系 OS（Linux/macOS）。Windows サポートは Live_Cap_v3 リポジトリで議論中です。
- 依存管理は [uv](https://github.com/astral-sh/uv) 推奨（pip/venv でも可）
- ReazonSpeech エンジン向けに `sherpa-onnx>=1.12.15` を同梱（Live_Cap_v3 と同じバージョン下限）

## インストール手順

```bash
git clone https://github.com/Mega-Gorilla/livecap-core
cd livecap-core

# 推奨: uv
uv sync --extra translation --extra dev

# 代替: pip/venv
python -m venv .venv && source .venv/bin/activate
pip install -e .[translation,dev]
```

### Optional extras

| Extra | 追加される依存 | 用途 |
| --- | --- | --- |
| `engines-torch` | `reazonspeech-k2-asr`, `torch`, `torchaudio`, `torchvision`（ReazonSpeech スタック） | ReazonSpeech / Torch 系エンジン |
| `engines-nemo` | `nemo-toolkit`, `hydra-core` など | NVIDIA NeMo エンジン |
| `translation` | `deep-translator` | 翻訳パイプライン |
| `dev` | `pytest` | テスト/開発ツール |

```
uv sync --extra engines-torch
# または
pip install "livecap-core[engines-torch]"
```

## 使い方

### CLI

```bash
uv run livecap-core --dump-config > default-config.json
```

### Python サンプル

```python
from livecap_core import FileTranscriptionPipeline, normalize_to_event_dict
from livecap_core.config.defaults import get_default_config

config = get_default_config()
pipeline = FileTranscriptionPipeline(config=config)

def stub_transcriber(audio_data, sample_rate):
    return [(0.0, 1.2, "Hello world")]

result = pipeline.process_file(
    file_path="sample.wav",
    segment_transcriber=stub_transcriber,
    write_subtitles=False,
)

event = normalize_to_event_dict({"text": "Hello", "offset": 0.0, "duration": 1.2})
print("success:", result.success, "sample text:", event["text"])
```

## テスト

```bash
# CI と同じテスト
uv sync --extra translation --extra dev
uv run python -m pytest tests

# エンジン系 (必要な extra を追加)
uv sync --extra translation --extra dev --extra engines-torch
uv run python -m pytest tests/core/engines

# 統合テスト (FFmpeg/モデルのダウンロードを伴う)
LIVECAP_ENABLE_INTEGRATION=true uv run python -m pytest tests/integration
```

## 関連ドキュメント

- プレリリース手順:
  [`docs/dev-docs/releases/pre-release-tag-workflow.md`](docs/dev-docs/releases/pre-release-tag-workflow.md)
- テストガイド:
  [`docs/dev-docs/testing/README.md`](docs/dev-docs/testing/README.md)
- アーキテクチャ/ロードマップ:
  [`docs/dev-docs/architecture/livecap-core-extraction.md`](https://github.com/Mega-Gorilla/Live_Cap_v3/blob/main/docs/dev-docs/architecture/livecap-core-extraction.md)
- ライセンス/コンプライアンス:
  [`docs/dev-docs/licensing/README.md`](docs/dev-docs/licensing/README.md)

## ライセンス・連絡先

LiveCap Core は AGPL-3.0 で提供しています (全文は `LICENSE`)。利用条件やコラボの相談は
[LiveCap Discord コミュニティ](https://discord.gg/hdSV4hJR8Y) へ、商用/セキュリティ関連は
`LICENSE-COMMERCIAL.md` 記載の連絡先をご利用ください。
