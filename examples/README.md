# LiveCap Core Examples

livecap-core の使用例を示すサンプルスクリプト集です。

## 前提条件

### インストール

```bash
# 基本インストール（PyTorch エンジン）
pip install livecap-core[engines-torch]

# または uv を使用
uv sync --extra engines-torch
```

### テスト用音声ファイル

サンプルスクリプトの一部は `tests/assets/audio/` 内の音声ファイルを使用します：

- `ja/jsut_basic5000_0001.wav` - 日本語音声（約3秒）
- `en/librispeech_1089-134686-0001.wav` - 英語音声（約3秒）

## サンプル一覧

### リアルタイム文字起こし (`examples/realtime/`)

| ファイル | 説明 | 難易度 |
|---------|------|--------|
| [basic_file_transcription.py](realtime/basic_file_transcription.py) | FileSource を使った最小構成の文字起こし | 初級 |
| [callback_api.py](realtime/callback_api.py) | コールバック方式での文字起こし | 初級 |
| [async_microphone.py](realtime/async_microphone.py) | マイク入力からの非同期文字起こし | 中級 |
| [custom_vad_config.py](realtime/custom_vad_config.py) | カスタム VAD 設定の使用例 | 上級 |

## 実行方法

### 基本的な使い方

```bash
# リポジトリルートから実行
cd /path/to/livecap-core

# 基本的なファイル文字起こし
python examples/realtime/basic_file_transcription.py

# 特定の音声ファイルを指定
python examples/realtime/basic_file_transcription.py path/to/audio.wav

# マイク入力（Ctrl+C で停止）
python examples/realtime/async_microphone.py
```

### 環境変数

| 変数名 | 説明 | デフォルト |
|--------|------|-----------|
| `LIVECAP_DEVICE` | 使用するデバイス（`cuda` / `cpu`） | `cuda`（利用可能な場合） |
| `LIVECAP_ENGINE` | 使用するエンジン | `whispers2t_base` |
| `LIVECAP_LANGUAGE` | 入力言語 | `ja` |

```bash
# CPU で実行
LIVECAP_DEVICE=cpu python examples/realtime/basic_file_transcription.py

# 英語音声を文字起こし
LIVECAP_LANGUAGE=en python examples/realtime/basic_file_transcription.py tests/assets/audio/en/librispeech_1089-134686-0001.wav
```

## トラブルシューティング

### Q: `OSError: PortAudio library not found`

マイク入力には PortAudio が必要です：

```bash
# Ubuntu/Debian
sudo apt-get install libportaudio2

# macOS
brew install portaudio
```

### Q: CUDA out of memory

CPU モードで実行するか、小さいモデルを使用してください：

```bash
LIVECAP_DEVICE=cpu python examples/realtime/basic_file_transcription.py

# または小さいモデル
LIVECAP_ENGINE=whispers2t_tiny python examples/realtime/basic_file_transcription.py
```

### Q: cuDNN version mismatch warning

```
WARNING: cuDNN error, retrying with CPU: cuDNN failed with status CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
```

この警告は cuDNN バージョンの不一致を示しています。エンジンは自動的に CPU にフォールバックして処理を継続します。警告を回避するには：

```bash
# 最初から CPU を使用
LIVECAP_DEVICE=cpu python examples/realtime/async_microphone.py

# または cuDNN を PyTorch 対応バージョンに更新
# https://pytorch.org/get-started/locally/ を参照
```

## 関連ドキュメント

- [リアルタイム文字起こしガイド](../docs/guides/realtime-transcription.md)
- [API 仕様書](../docs/architecture/core-api-spec.md)
- [テストガイド](../docs/testing/README.md)
