# ASR Benchmark Report

**Date:** 2026-11-10T19:00:40.883692Z
**Mode:** standard
**Device:** cuda

## Results by Language

### JA

| Engine         | CER   | WER   |   RTF | VRAM   |   Files |
|:---------------|:------|:------|------:|:-------|--------:|
| parakeet_ja    | 7.2%  | 61.6% | 0.011 | 1231MB |      99 |
| reazonspeech   | 7.8%  | 60.6% | 0.041 | 9MB †  |      99 |
| whispers2t     | 8.1%  | 55.6% | 0.061 | 1237MB * |      99 |
| qwen3asr_large | 8.4%  | 59.6% | 0.111 | 4042MB |      99 |
| qwen3asr       | 11.3% | 68.7% | 0.107 | 1640MB |      99 |

> 注記 (出力の後から追記。値は出力のまま):
> VRAM 列は `torch.cuda.max_memory_allocated()` (torch から見える割り当ての peak) で、torch 以外の割り当ては含まない。
> **\*** `whispers2t`: 本体は CTranslate2 で独自の CUDA allocator を使うので、モデルの領域は含まれない。**必要 VRAM として使えず、他の engine と比べられない (未計測)**。
> **†** `reazonspeech`: CPU provider 固定で GPU を使わない (**N/A**)。

**Best CER:** parakeet_ja (7.2%)

## Summary

- **Total files:** 495
- **Total duration:** 1966.3 sec
