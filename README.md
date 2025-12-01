# LiveCap CLI

LiveCap CLI は高性能な音声文字起こしライブラリです。リアルタイム文字起こし、ファイル処理、複数の ASR エンジン対応を提供します。

## 主な機能

- **リアルタイム文字起こし** - VAD（音声活動検出）ベースのストリーミング処理
- **ファイル文字起こし** - 音声/動画ファイルから SRT 字幕生成
- **マルチエンジン対応** - Whisper, ReazonSpeech, Parakeet, Canary など
- **16言語サポート** - 日本語、英語、中国語、韓国語など

## クイックスタート

### インストール

```bash
git clone https://github.com/Mega-Gorilla/livecap-cli
cd livecap-cli

# uv を使用（推奨）
uv sync --extra vad --extra engines-torch

# pip を使用
pip install -e ".[vad,engines-torch]"
```

### リアルタイム文字起こし

```python
from livecap_core import StreamTranscriber, MicrophoneSource
from engines import EngineFactory

# エンジン初期化
engine = EngineFactory.create_engine("whispers2t_base", device="cuda")
engine.load_model()

# マイクから文字起こし
with StreamTranscriber(engine=engine) as transcriber:
    with MicrophoneSource() as mic:
        for result in transcriber.transcribe_sync(mic):
            print(f"[{result.start_time:.2f}s] {result.text}")
```

### ファイル文字起こし

```python
from livecap_core import FileTranscriptionPipeline
from livecap_core.config import get_default_config
from engines import EngineFactory

config = get_default_config()
engine = EngineFactory.create_engine("whispers2t_base", device="cuda", config=config)
engine.load_model()

pipeline = FileTranscriptionPipeline(config=config)
result = pipeline.process_file(
    file_path="audio.wav",
    segment_transcriber=lambda audio, sr: engine.transcribe(audio, sr)[0],
)
print(f"字幕出力: {result.output_path}")
```

## オプション依存

| Extra | 内容 | 用途 |
|-------|------|------|
| `vad` | `silero-vad`, `webrtcvad`, `ten-vad` | リアルタイム文字起こし（VAD） |
| `engines-torch` | `torch`, `reazonspeech-k2-asr` | PyTorch 系エンジン |
| `engines-nemo` | `nemo-toolkit` | NVIDIA NeMo エンジン |
| `translation` | `deep-translator` | 翻訳機能 |
| `dev` | `pytest` | 開発・テスト |

```bash
# 例: VAD + PyTorch エンジン
uv sync --extra vad --extra engines-torch
```

**注意**: Linux で `[vad]` を使用する場合、TenVAD には libc++ が必要です：

```bash
# Ubuntu/Debian
sudo apt-get install libc++1
```

## 対応エンジン

| ID | モデル | サイズ | 言語 |
|----|--------|--------|------|
| `reazonspeech` | ReazonSpeech K2 v2 | 159MB | ja |
| `whispers2t_base` | Whisper Base | 74MB | 多言語 |
| `whispers2t_large_v3` | Whisper Large-v3 | 1.55GB | 多言語 |
| `parakeet` | Parakeet TDT 0.6B | 1.2GB | en |
| `parakeet_ja` | Parakeet TDT CTC JA | 600MB | ja |
| `canary` | Canary 1B Flash | 1.5GB | en, de, fr, es |

## サンプルスクリプト

`examples/` ディレクトリにサンプルスクリプトがあります：

```bash
# ファイル文字起こし
python examples/realtime/basic_file_transcription.py

# マイク入力（Ctrl+C で停止）
python examples/realtime/async_microphone.py

# デバイス一覧
python examples/realtime/async_microphone.py --list-devices

# VAD プロファイル一覧
python examples/realtime/custom_vad_config.py --list-profiles
```

環境変数で設定を変更できます：

```bash
LIVECAP_DEVICE=cpu      # デバイス（cuda/cpu）
LIVECAP_ENGINE=whispers2t_base  # エンジン
LIVECAP_LANGUAGE=ja     # 言語
```

## ドキュメント

- [リアルタイム文字起こしガイド](docs/guides/realtime-transcription.md)
- [API 仕様書](docs/architecture/core-api-spec.md)
- [機能一覧](docs/reference/feature-inventory.md)
- [テストガイド](docs/testing/README.md)
- [サンプル README](examples/README.md)

## 必要環境

- Python 3.10 - 3.12
- Linux / macOS / Windows
- [uv](https://github.com/astral-sh/uv)（推奨）または pip

マイク入力を使用する場合は PortAudio が必要です：

```bash
# Ubuntu/Debian
sudo apt-get install libportaudio2

# macOS
brew install portaudio
```

## ライセンス

AGPL-3.0 - 詳細は [LICENSE](LICENSE) を参照

質問やコラボレーションは [LiveCap Discord](https://discord.gg/hdSV4hJR8Y) へ
