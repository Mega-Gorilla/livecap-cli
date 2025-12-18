# Issue #74 実装計画: 依存関係整理・CLI・パッケージ名変更

## 概要

Issue #74 は livecap-core リファクタリングの Phase 6 として、CLI コマンドの実装とパッケージ名変更を行う。

**親 Issue:** #64 [Epic] livecap-cli リファクタリング

## 現状分析

### 1. pyproject.toml (既存構造)

```toml
# 現在の optional-dependencies
"engines-nemo"       # NeMo系エンジン (canary, parakeet, voxtral)
"engines-torch"      # PyTorch系エンジン (reazonspeech)
"translation"        # Google翻訳 (deep-translator)
"translation-local"  # OPUS-MT ローカル翻訳
"translation-riva"   # Riva-4B 翻訳
"dev"                # 開発ツール (pytest)
"benchmark"          # ベンチマーク (jiwer, tabulate)
"optimization"       # 最適化 (optuna, plotly)
```

**Issue #74 の目標構造との差分:**

| 目標 | 現状 | 対応 |
|------|------|------|
| `engine-whisper` | コア依存に含む | ✅ 維持 |
| `engine-sherpa` | コア依存に含む | ✅ 維持 |
| `engine-torch` | `engines-torch` | ✅ 既存 |
| `engine-nemo` | `engines-nemo` | ✅ 既存 |
| `translation-google` | `translation` | 🔄 リネーム検討 |
| `translation-riva` | `translation-riva` | ✅ 既存 |
| `recommended` | なし | ➕ 追加 |
| `all` | なし | ➕ 追加 |

### 2. CLI (既存機能)

```bash
# 現在のコマンド（Phase 6B で廃止）
livecap-core --info           # インストール診断
livecap-core --ensure-ffmpeg  # FFmpeg確保
livecap-core --as-json        # JSON出力
```

> **注意:** 既存の CLI フラグは Phase 6B で完全に廃止し、サブコマンド構造に移行する。
> Epic #64 の方針「互換性維持は不要」に従い、deprecation warning は設けない。

**Issue #74 の目標コマンド:**

| コマンド | 現状 | 実装難易度 |
|----------|------|-----------|
| `livecap-cli transcribe --realtime --mic <id>` | なし | 中 (ロジックは examples/ に存在) |
| `livecap-cli transcribe --realtime --system` | なし | 高 (システム音声キャプチャ未実装) |
| `livecap-cli transcribe <file> -o <output>` | なし | 低 (FileTranscriptionPipeline 利用) |
| `livecap-cli devices` | なし | 低 (MicrophoneSource.list_devices()) |
| `livecap-cli engines` | なし | 低 (EngineMetadata.get_all()) |

### 3. 利用可能なエンジン・翻訳器

**ASR エンジン (6種):**
- `whispers2t` - WhisperS2T (CTranslate2)
- `reazonspeech` - ReazonSpeech K2 (CPU専用)
- `canary` - NVIDIA Canary 1B Flash
- `parakeet` / `parakeet_ja` - NVIDIA Parakeet TDT
- `voxtral` - Mistral Voxtral Mini 3B

**翻訳器 (3種):**
- `google` - Google Translate (deep-translator)
- `opus_mt` - OPUS-MT (CTranslate2)
- `riva_instruct` - Riva-Translate-4B-Instruct

### 4. システム音声キャプチャの課題

`--system` オプションにはシステム音声のキャプチャが必要。これはプラットフォーム依存:

| プラットフォーム | 方法 | 難易度 |
|-----------------|------|--------|
| Windows | WASAPI loopback | 中 |
| macOS | BlackHole/Soundflower 経由 | 外部依存 |
| Linux | PulseAudio monitor | 中 |

**推奨:** Phase 6 では `--system` をスコープ外とし、将来課題とする。

---

## 実装計画

### Phase 6A: pyproject.toml 整理 (低リスク)

**目的:** extras の整理と `recommended` / `all` メタエクストラの追加

**変更内容:**

```toml
[project.optional-dependencies]
# 既存（変更なし）
"engines-nemo" = [...]
"engines-torch" = [...]
"translation" = [...]
"translation-local" = [...]
"translation-riva" = [...]
"dev" = [...]
"benchmark" = [...]
"optimization" = [...]

# 新規追加
"recommended" = [
  "livecap-core[translation]",  # Google翻訳
]
"all" = [
  "livecap-core[engines-nemo,engines-torch,translation,translation-local,translation-riva,benchmark,optimization]",
]
```

> **注意:** 自己参照形式 (`livecap-core[...]`) は pip/setuptools で動作するが、
> PyPI 公開前は `.[translation]` のようなローカル参照形式でテストすること。
> 実装時に循環依存や意図しない PyPI 参照が起きないか検証が必要。

> **Phase 6B との連携:** Phase 6B で `name = "livecap-cli"` に変更するため、
> 自己参照も `livecap-cli[...]` に更新する必要がある。Phase 6A と 6B を
> 同一 PR で行うか、6A では自己参照を避けて依存を直接列挙することを推奨。

**完了条件:**
- [ ] `pip install livecap-core[recommended]` が動作 (または 6B と同時なら `livecap-cli[recommended]`)
- [ ] `pip install livecap-core[all]` が動作 (または 6B と同時なら `livecap-cli[all]`)
- [ ] 既存の extras が引き続き動作

### Phase 6B: CLI コマンド実装 + エントリポイント変更 (中リスク)

**目的:** `transcribe`, `devices`, `engines` コマンドの実装と `livecap-cli` エントリポイント導入

> **Note:** 当初 Phase 6C で予定していたエントリポイント変更を Phase 6B に統合。
> 理由: 新しいサブコマンド CLI を古い `livecap-core` で実装してからすぐに `livecap-cli` に
> 変更するのは二度手間であり、最初から `livecap-cli` として提供する方が効率的。

**互換性方針:** Epic #64 に従い、以下を**完全に廃止**する。deprecation warning は設けない。

- 既存フラグ (`--info`, `--ensure-ffmpeg`, `--as-json`)
- 旧エントリポイント (`livecap-core`)
- 旧モジュール名 (`livecap_cli`) → `livecap_cli` に変更

> **Python API について:** パッケージ名 (`livecap-cli`) とモジュール名 (`livecap_cli`) を
> 一致させることで、ユーザー体験を向上させる。`pip install livecap-cli` したら
> `from livecap_cli import ...` でインポートできるのが自然。
> 利用者がほぼいないプレリリース段階の今が変更の最適なタイミング。

#### 6B-0: モジュール名変更

`livecap_cli/` ディレクトリを `livecap_cli/` にリネーム:

```bash
# ディレクトリリネーム
mv livecap_cli/ livecap_cli/

# 影響範囲
- livecap_cli/ → livecap_cli/  # モジュールディレクトリ
- tests/         # import 文を更新
- examples/      # import 文を更新
- docs/          # 参照を更新
- CLAUDE.md      # 参照を更新
```

> **Note:** 相対インポート (`from .engines import ...`) は影響を受けない。
> 変更が必要なのは絶対インポート (`from livecap_cli import ...`) のみ。

#### 6B-1: エントリポイント変更 + サブコマンド構造の導入

`pyproject.toml` を更新し、新しいエントリポイントとサブコマンド構造を同時に導入:

```toml
[project]
name = "livecap-cli"  # パッケージ名変更

[project.scripts]
livecap-cli = "livecap_cli.cli:main"  # 新規（唯一のエントリポイント）
# livecap-core は廃止（Epic #64 方針）
```

```bash
livecap-cli info               # 診断情報（--as-json オプション付き）
livecap-cli devices            # オーディオデバイス一覧
livecap-cli engines            # ASRエンジン一覧
livecap-cli translators        # 翻訳器一覧
livecap-cli transcribe [args]  # 文字起こし
```

**廃止されるコマンド:**
```bash
# これらは動作しなくなる
livecap-core --info            # → livecap-cli info
livecap-core --ensure-ffmpeg   # → livecap-cli info --ensure-ffmpeg
livecap-core --as-json         # → livecap-cli info --as-json
```

#### 6B-2: devices コマンド

```python
def cmd_devices(args):
    from livecap_cli import MicrophoneSource
    devices = MicrophoneSource.list_devices()
    for dev in devices:
        default = " (default)" if dev.is_default else ""
        print(f"[{dev.index}] {dev.name}{default}")
```

#### 6B-3: engines コマンド

```python
def cmd_engines(args):
    from livecap_cli.engines.metadata import EngineMetadata
    for engine_id, meta in EngineMetadata.get_all().items():
        print(f"{engine_id}: {meta.display_name}")
```

#### 6B-4: translators コマンド (追加提案)

```python
def cmd_translators(args):
    from livecap_cli.translation.metadata import TranslatorMetadata
    for tid, info in TranslatorMetadata.get_all().items():
        gpu = " (GPU)" if info.requires_gpu else ""
        print(f"{tid}: {info.display_name}{gpu}")
```

#### 6B-5: transcribe コマンド

**マイク入力 (リアルタイム):**

```bash
livecap-cli transcribe --realtime --mic 0 \
  --engine whispers2t --device gpu --language ja
```

実装: `examples/realtime/async_microphone.py` のロジックを CLI に統合

**ファイル入力:**

```bash
livecap-cli transcribe input.mp4 -o output.srt \
  --engine whispers2t --device gpu --language ja
```

実装: `FileTranscriptionPipeline` を利用

**オプション:**

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--engine` | ASRエンジンID | `whispers2t` |
| `--device` | デバイス (auto/gpu/cpu) | `auto` |
| `--language` | 入力言語 | `ja` |
| `--model-size` | WhisperS2Tモデルサイズ | `base` |
| `--vad` | VADバックエンド (auto/silero/tenvad/webrtc) | `auto` |
| `--translate` | 翻訳器ID | なし |
| `--target-lang` | 翻訳先言語 | `en` |

> **デバイス表記について:** CLI では `gpu` を使用し、内部で `cuda` にマッピングする。
> これは Issue #74 の仕様 (`auto/gpu/cpu`) に準拠し、ユーザーフレンドリーな表記を優先する。

> **モデルサイズについて:** CLI のデフォルトは `base` だが、エンジン API のデフォルトは
> `large-v3` (`livecap_cli/engines/metadata.py:147`)。CLI では起動速度とリソース効率を
> 優先し、ユーザーが明示的に `--model-size large-v3` を指定した場合のみ高精度モードを使用する。

> **VAD バックエンド選択について:** VAD バックエンド選択は既に API レベルで実装済み
> (`VADProcessor.from_language()`)。CLI では `--vad auto` がデフォルトで、
> `--language` に基づき最適な VAD を自動選択する。未対応言語の場合は警告を表示し、
> Silero VAD にフォールバックする（`VADProcessor()` のデフォルト動作）。

**完了条件:**
- [ ] `livecap-cli info` が動作
- [ ] `livecap-cli devices` が動作
- [ ] `livecap-cli engines` が動作
- [ ] `livecap-cli translators` が動作
- [ ] `livecap-cli transcribe --realtime --mic 0` が動作
- [ ] `livecap-cli transcribe input.mp4 -o output.srt` が動作
- [ ] 旧フラグ (`--info` 等) が廃止されていることを確認

---

## スコープ外 (将来課題)

### システム音声キャプチャ (`--system`)

プラットフォーム依存性が高く、Phase 6 では実装しない。

**将来的な実装方針:**
1. WASAPI loopback (Windows)
2. PulseAudio monitor (Linux)
3. 外部ツール連携 (macOS: BlackHole)

> **Note:** Issue #74 本文には `--realtime --system` が要件として記載されているが、
> プラットフォーム依存性が高いためスコープ外とする。必要に応じてフォローアップ Issue を作成。

---

## 実装順序とリスク評価

```
Phase 6A (pyproject.toml整理)           [低リスク, 0.5日]
    ↓
Phase 6B (CLI実装 + エントリポイント変更)  [中リスク, 1-2日]
```

> **Note:** 当初の Phase 6C（パッケージ名変更）は Phase 6B に統合。
> CLI 実装とエントリポイント変更を同時に行うことで効率化。

**推奨:** 6A → 6B の順で、各 Phase を別 PR として作成

---

## 完了条件チェックリスト

### Phase 6A
- [ ] `recommended` extras 追加
- [ ] `all` extras 追加
- [ ] 既存テスト通過

### Phase 6B (CLI 実装 + エントリポイント変更)
- [ ] `livecap_cli/` → `livecap_cli/` にリネーム
- [ ] tests/examples/docs の import 文を更新
- [ ] pyproject.toml 更新 (`name = "livecap-cli"`, エントリポイント変更)
- [ ] サブコマンド構造導入（既存フラグは完全廃止）
- [ ] `info` コマンド実装
- [ ] `devices` コマンド実装
- [ ] `engines` コマンド実装
- [ ] `translators` コマンド実装
- [ ] `transcribe --realtime --mic` 実装
- [ ] `transcribe <file> -o <output>` 実装
- [ ] `--vad` オプション実装
- [ ] ドキュメント更新
- [ ] ユニットテスト追加
- [ ] 既存テスト通過

---

## 関連ファイル

| ファイル | 役割 |
|---------|------|
| `pyproject.toml` | パッケージ設定 |
| `livecap_cli/cli.py` | CLI エントリポイント |
| `livecap_cli/audio_sources/microphone.py` | マイク入力 |
| `livecap_cli/transcription/file_pipeline.py` | ファイル文字起こし |
| `livecap_cli/transcription/stream.py` | リアルタイム文字起こし |
| `livecap_cli/engines/metadata.py` | エンジンメタデータ |
| `livecap_cli/translation/metadata.py` | 翻訳器メタデータ |
| `examples/realtime/async_microphone.py` | マイク入力サンプル |

> **Note:** `livecap_cli/` は `livecap_cli/` にリネームされる（Phase 6B-0）。

---

## 参考: CLI 使用例 (目標)

```bash
# インストール
pip install livecap-cli[recommended]

# デバイス一覧
livecap-cli devices

# エンジン一覧
livecap-cli engines

# 翻訳器一覧
livecap-cli translators

# リアルタイム文字起こし (マイク)
livecap-cli transcribe --realtime --mic 0 --engine whispers2t --language ja

# ファイル文字起こし
livecap-cli transcribe input.mp4 -o output.srt --engine whispers2t

# 翻訳付き文字起こし
livecap-cli transcribe input.mp4 -o output.srt --translate google --target-lang en
```
