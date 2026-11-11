# Qwen3-ASR 1.7B の ASR ベンチマーク (#470 Phase 3)

Qwen3-ASR 1.7B (`qwen3asr_large`) を 0.6B (`qwen3asr`) と参考の engine と比べた結果。`benchmarks/asr` の出力 (`summary.md`) を 4 本置いた。値は出力のままで、`gpu_ja.md` の VRAM 列にだけ、比べられない値の注記を後から加えた (下の「VRAM 列」)。

| ファイル | 内容 |
|---|---|
| [`gpu_ja.md`](gpu_ja.md) | GPU・日本語: `qwen3asr` / `qwen3asr_large` / `reazonspeech` / `parakeet_ja` / `whispers2t` (large-v3) |
| [`gpu_en.md`](gpu_en.md) | GPU・英語: `qwen3asr` / `qwen3asr_large` / `parakeet` / `canary` |
| [`cpu_ja.md`](cpu_ja.md) | CPU・日本語: `qwen3asr` / `qwen3asr_large` |
| [`cpu_en.md`](cpu_en.md) | CPU・英語: `qwen3asr` / `qwen3asr_large` |

## 条件

| 項目 | 値 |
|---|---|
| コード | `main` の `8844be0` (#483 の bf16 推論を含む) |
| 実行 | `python -m benchmarks.asr --mode standard --language <ja/en> --engine ... --device <cuda/cpu>` (GPU は `--runs 3`) |
| GPU | RTX 4070 Ti 12 GB (driver 580.178.04)、Ubuntu-Desktop |
| CPU | 12 スレッド (同じ機体) |
| torch | 2.9.1+cu128 |
| dtype | `auto` → GPU は bfloat16 (Qwen3-ASR / Parakeet / Canary)、whispers2t は float16、CPU は float32 |
| データ | standard: ja は JSUT basic5000 の 99 files (`_0002`〜`_0100`。`_0001` は test fixture と重複するので準備 script が除外)、en は LibriSpeech test-clean の 100 files (話者 61 の 2 章) |

## 読み方の注意

- **VAD と後段 (EnergyGate / confidence filter) を通さない、ASR 単体の値**。1 file ずつ丸ごと engine に入れている
- **CER / WER / RTF は file ごとの値の単純平均** (`benchmarks/asr/runner.py`)。全体の誤り数 ÷ 全体の正解の長さ (合算) ではない
- **ja の WER は参考にならない** (単語の区切りが無い)。ja は CER で比べる
- データはどちらも**話者 1 人の静かな読み上げで、1 本 4 秒前後**。配信に近い長尺の音声での評価は #488 で行い、そこで全 engine を測り直す
- **RTX 4090 では測っていない**。4090 と 4070 Ti の結果が一致するかは確かめていない (同じ世代・同じ dtype でも、数値の一致や allocator の確保量は実行条件に依存する)
- file ごとの結果 (正解・認識結果を含む CSV) は、コーパスのテキストを含むので置いていない

## VRAM 列

`benchmarks/common/metrics.py` の `GPUMemoryTracker.get_peak()` = `torch.cuda.max_memory_allocated()` で、**torch から見える割り当ての peak** だけを数える。torch 以外の allocator が確保した領域は含まない。

| engine | VRAM 列の扱い |
|---|---|
| `qwen3asr` / `qwen3asr_large` / `parakeet` / `parakeet_ja` / `canary` | torch で動くので、モデルの領域を含む。この 5 つの間では比べられる |
| `whispers2t` | **未計測 (比べられない)**。本体は CTranslate2 で独自の CUDA allocator を使うので、1237MB はモデルの領域を含まない。metadata の `vram_required_mb` も同じ理由で `None` |
| `reazonspeech` | **N/A**。CPU provider 固定で GPU を使わない (9MB は同じ process の torch 側の割り当てで、ReazonSpeech のモデルとは関係しない) |

この列は peak **allocated** で、`vram_required_mb` に入れた値 (0.6B 3,690 / 1.7B 9,076 MB) は #483 の調査で測った fp32 の peak **reserved** (`docs/research/torch-engines-bf16-2026-11.md`)。測り方も dtype も違うので、この表の値をそのまま `vram_required_mb` と比べない

## 結果からの判断 (#470)

- `vram_required_mb` を fp32 の実測 (0.6B 3,690 MB / 1.7B 9,076 MB、#483 と同じ方針) にした
- `quality_tier` と既定は変えない (en では 1.7B が最良だが、#488 で全 engine を測り直してから決める)
- CPU では 1.7B は実時間に届かない (RTF 約 2.0)。0.6B は 0.85
