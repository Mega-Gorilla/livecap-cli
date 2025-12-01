# VAD Benchmark Guide

VAD（Voice Activity Detection）+ ASRパイプラインの精度・性能を評価するベンチマークツールです。

## クイックスタート

```bash
# 依存関係のインストール
uv sync --extra benchmark --extra vad --extra engines-torch

# クイックベンチマーク実行
uv run python -m benchmarks.vad --mode quick

# 最適化済みパラメータで実行
uv run python -m benchmarks.vad --mode standard --param-source preset

# 結果確認
ls benchmark_results/
```

## ASRベンチマークとの違い

| 項目 | ASRベンチマーク | VADベンチマーク |
|------|----------------|-----------------|
| 評価対象 | ASRエンジン単体 | VAD + ASRパイプライン |
| 入力 | 音声ファイル全体 | VADで分割されたセグメント |
| 組み合わせ | エンジン × 言語 | エンジン × VAD × 言語 |
| 追加指標 | - | VAD RTF, セグメント数, 音声比率 |

## データセット準備

ASRベンチマークと同じデータセットを使用します。詳細は [ASR Benchmark Guide](./asr-benchmark.md#データセット準備) を参照してください。

```bash
# データ準備（standard/full モード用）
uv run python scripts/download_benchmark_assets.py
uv run python scripts/prepare_benchmark_data.py --mode standard
```

## CLI オプション

```bash
python -m benchmarks.vad [OPTIONS]
```

### 基本オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--mode` | 実行モード（debug/quick/standard/full） | `quick` |
| `--language`, `-l` | 評価言語（複数指定可） | `ja en` |
| `--engine`, `-e` | ASRエンジン（複数指定可） | モード依存 |
| `--vad` | VADバックエンド（複数指定可） | モード依存 |
| `--param-source` | パラメータソース（default/preset） | `default` |
| `--runs`, `-r` | RTF測定の繰り返し回数 | `1` |
| `--device` | 使用デバイス（cuda/cpu） | `cuda` |
| `--output-dir`, `-o` | 結果出力ディレクトリ | `benchmark_results/` |
| `--list-vads` | 利用可能なVAD一覧を表示 | - |
| `--verbose`, `-v` | 詳細ログ出力 | `false` |

### パラメータソース（`--param-source`）

| 値 | 説明 | 利用可能VAD |
|----|------|------------|
| `default` | ハードコードされたデフォルト値 | 全VAD |
| `preset` | Bayesian最適化済みパラメータ | silero, tenvad, webrtc |

**preset モード**: `livecap_core/vad/presets.py` から最適化済みパラメータを読み込みます。Phase D (#126) で最適化された値を使用。

### 使用例

```bash
# VAD一覧表示
uv run python -m benchmarks.vad --list-vads

# 特定VAD・エンジンの組み合わせ
uv run python -m benchmarks.vad --engine parakeet_ja --vad silero webrtc --language ja

# 最適化パラメータで日本語評価
uv run python -m benchmarks.vad --mode standard --param-source preset --language ja

# 全VAD比較（fullモード）
uv run python -m benchmarks.vad --mode full --language ja en

# CPUで実行
uv run python -m benchmarks.vad --mode quick --device cpu
```

## 利用可能なVAD

```bash
uv run python -m benchmarks.vad --list-vads
```

| VAD ID | 説明 | 最適化対応 |
|--------|------|-----------|
| `silero` | Silero VAD v5/v6（ONNX） | Yes |
| `tenvad` | TenVAD（軽量VAD） | Yes |
| `webrtc` | WebRTC VAD | Yes |
| `javad_tiny` | JaVAD Tiny（日本語特化） | No |
| `javad_balanced` | JaVAD Balanced | No |
| `javad_precise` | JaVAD Precise | No |

## 評価指標

### 精度指標（ASRと共通）

| 指標 | 説明 | 対象言語 |
|------|------|---------|
| **CER** | Character Error Rate | 日本語 |
| **WER** | Word Error Rate | 英語 |

### VAD固有指標

| 指標 | 説明 | 目安 |
|------|------|------|
| **VAD RTF** | VAD処理のReal-Time Factor | < 0.01 が理想 |
| **Seg** | 検出セグメント数 | ファイル依存 |
| **Speech%** | 音声区間の割合 | 30-70% が典型 |

### 総合RTF

総合RTF = VAD RTF + ASR RTF

VADは非常に軽量なため、総合RTFはほぼASR RTFに等しくなります。

## デフォルト設定

### モード別デフォルトVAD

| モード | VAD |
|--------|-----|
| debug/quick/standard | silero, webrtc |
| full | 全VAD |

### モード別デフォルトエンジン

| 言語 | debug/quick/standard | full |
|------|---------------------|------|
| 日本語 | parakeet_ja, whispers2t_large_v3 | 全対応エンジン |
| 英語 | parakeet, whispers2t_large_v3 | 全対応エンジン |

## 結果の解釈

### 出力ファイル

```
benchmark_results/vad_YYYYMMDD_HHMMSS_MODE/
├── summary.md          # Markdown形式のサマリー
└── raw/
    ├── engine1_vad1_ja.csv
    └── engine1_vad2_en.csv
```

### summary.md の読み方

```markdown
## Results by Language

### JA
| Engine              | VAD    | CER   | WER    | RTF   | VAD RTF | Seg | Speech% | Files |
|---------------------|--------|-------|--------|-------|---------|-----|---------|-------|
| parakeet_ja         | silero | 5.2%  | 12.3%  | 0.089 | 0.003   | 12  | 45%     | 100   |
| parakeet_ja         | tenvad | 4.8%  | 11.5%  | 0.091 | 0.001   | 14  | 48%     | 100   |
| parakeet_ja         | webrtc | 5.5%  | 13.1%  | 0.087 | 0.002   | 10  | 42%     | 100   |

**Best CER:** parakeet_ja+tenvad (4.8%)
```

- **VAD RTF**: VAD処理速度（低いほど高速）
- **Seg**: 検出セグメント数（VADの分割粒度を反映）
- **Speech%**: 音声区間の割合

## Phase D 最適化結果

#126 で実施したBayesian最適化の結果:

| 言語 | 最適VAD | CER/WER | 改善率 |
|------|---------|---------|--------|
| 日本語 | **tenvad** | 7.2% CER | -1.9% (vs silero) |
| 英語 | **webrtc** | 3.3% WER | -2.6% (vs silero) |

**推奨**: `--param-source preset` を使用して最適化済みパラメータを適用

## CI/GitHub Actions での実行

### ワークフローディスパッチ

1. Actions タブ → "VAD Benchmark" 選択
2. "Run workflow" クリック
3. パラメータを設定

### 入力パラメータ

| パラメータ | 説明 | デフォルト |
|-----------|------|-----------|
| `mode` | 実行モード | `quick` |
| `languages` | 評価言語（カンマ区切り） | `ja,en` |
| `engine_preset` | エンジン選択方式 | `default` |
| `engines` | カスタムエンジン | - |
| `vad_preset` | VAD選択方式 | `default` |
| `vads` | カスタムVAD | - |
| `param_source` | パラメータソース | `default` |
| `runs` | RTF測定回数 | `1` |

### preset モードでの実行例

最適化パラメータを使用する場合:
- `param_source`: `preset`
- `vad_preset`: `default` または `custom` (silero, tenvad, webrtc のみ)

## トラブルシューティング

### よくあるエラー

#### VADが見つからない

```
Unknown VAD: unknown_vad
```

**解決策**: `--list-vads` で利用可能なVADを確認
```bash
uv run python -m benchmarks.vad --list-vads
```

#### preset モードでJaVADを指定した

```
VAD 'javad_balanced' has no optimized preset.
```

**解決策**: preset モードでは silero, tenvad, webrtc のみ使用可能。JaVADは `--param-source default` で使用してください。

#### VAD依存関係が不足

```
ImportError: silero_vad is required
```

**解決策**: VAD依存関係をインストール
```bash
uv sync --extra vad
```

#### TenVADのONNX警告

```
UserWarning: TenVAD requires onnxruntime
```

**解決策**: 動作には影響ありませんが、警告を消すには:
```bash
uv pip install onnxruntime
```

### パフォーマンスチューニング

#### VAD処理が遅い場合

- TenVADは最も軽量（RTF < 0.001）
- WebRTCも軽量（RTF < 0.002）
- SileroはGPU使用時に高速化

#### メモリ使用量が高い場合

- JaVADは比較的軽量
- Sileroはモデルサイズ中程度
- 複数VADを同時評価する場合はメモリに注意

## 関連ドキュメント

### VAD リファレンス

- [VAD バックエンドリファレンス](../../reference/vad/backends.md) - 各バックエンドの詳細仕様
- [VADConfig リファレンス](../../reference/vad/config.md) - 共通パラメータの詳細
- [VAD バックエンド比較](../../reference/vad/comparison.md) - ベンチマーク結果

### ガイド

- [ASR Benchmark Guide](./asr-benchmark.md) - ASRベンチマークガイド
- [VAD Optimization Guide](./vad-optimization.md) - VADパラメータ最適化ガイド
- [VAD Benchmark Plan](../../planning/archive/vad-benchmark-plan.md) - 計画ドキュメント（アーカイブ）
