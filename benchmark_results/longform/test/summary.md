# 長尺の評価: YODAS2 日本語 `test` (31 本、8 人、5.7 時間)

主の値は `fillers-notation_excluded` (片方にしか無い口ぐせを数えない、表記だけ違う区間を除く) の CER (%)。全体は誤りの総数 ÷ 正解の総文字数、話者平均は話者ごとの CER の平均。

| ASR | CER | 置換 | 脱落 | 挿入 | 話者平均 | そのまま | 数字をそろえる | 読みで比べる | RTF | 読み込み (秒) | GPU (torch, MB) | 区間の長さ (中央 / 最長, 秒) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `whispers2t-large-v3` | **11.52** | 4.86 | 3.88 | 2.78 | 11.09 | 12.02 | 12.07 | 8.85 | 0.065 | 3.35 | 46 | 2.0 / 15 (VAD) |
| `qwen3asr_large` | **12.60** | 6.65 | 2.55 | 3.40 | 11.91 | 13.69 | 12.65 | 9.17 | 0.101 | 4.0 | 4610 | 2.1 / 15 (VAD) |
| `qwen3asr` | **14.90** | 8.13 | 3.02 | 3.75 | 14.18 | 15.91 | 14.98 | 11.02 | 0.094 | 3.54 | 2200 | 3.6 / 15 (VAD) |
| `reazonspeech` | **20.40** | 5.51 | 12.82 | 2.07 | 20.78 | 20.73 | 19.64 | 16.03 | 0.072 | 1.76 | 0 | 1.7 / 15 (VAD) |
| `parakeet_ja` | **20.72** | 4.17 | 13.98 | 2.57 | 20.58 | 21.06 | 21.10 | 18.13 | 0.010 | 16.72 | 1254 | 15.0 / 15 (VAD) |
| `whispers2t-base` | **38.30** | 19.95 | 4.74 | 13.61 | 35.83 | 38.63 | 38.61 | 29.70 | 0.015 | 1.17 | 46 | 2.0 / 15 (VAD) |

「そのまま」「数字をそろえる」「読みで比べる」は、表記だけ違う区間も含めた値。GPU は torch が確保した量の最大で、CTranslate2 (whispers2t) や ONNX Runtime の分は数えない。区間の長さは、全部の動画に VAD の区間ごとの記録 (`vad_segments`) があれば落ちた区間も含めた VAD の区間 (VAD)、1 本でも無ければ ASR の出力の残った区間 (出力) の長さ。

## 話者ごとの CER (`fillers-notation_excluded`、%)

| ASR | 2000 | 2012 | 2014 | 2016 | 2018 | 2031 | 2034 | 2035 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `whispers2t-large-v3` | 7.35 | 9.09 | 19.21 | 17.10 | 16.71 | 4.56 | 2.91 | 11.75 |
| `qwen3asr_large` | 10.71 | 10.77 | 18.01 | 18.95 | 13.63 | 6.39 | 6.01 | 10.82 |
| `qwen3asr` | 12.90 | 11.71 | 20.39 | 21.24 | 16.70 | 7.99 | 8.21 | 14.32 |
| `reazonspeech` | 9.91 | 17.28 | 22.86 | 36.32 | 25.05 | 17.42 | 5.54 | 31.90 |
| `parakeet_ja` | 13.86 | 23.41 | 22.09 | 27.27 | 23.80 | 12.12 | 22.42 | 19.69 |
| `whispers2t-base` | 30.38 | 32.69 | 50.99 | 54.41 | 48.84 | 16.91 | 23.33 | 29.12 |

## drop の内訳 (EnergyGate / confidence filter で落ちた区間の数と、その長さの合計)

| ASR | 区間の数 | 長さの合計 (秒) |
|---|---|---|
| `whispers2t-large-v3` | confidence_filter:reject=62, energy_gate:low_rms=3 | confidence_filter:reject=133.2, energy_gate:low_rms=2.9 |
| `qwen3asr_large` | confidence_filter:reject=55, energy_gate:low_rms=4 | confidence_filter:reject=73.3, energy_gate:low_rms=3.2 |
| `qwen3asr` | confidence_filter:reject=300, energy_gate:low_rms=127 | confidence_filter:reject=497.0, energy_gate:low_rms=98.3 |
| `reazonspeech` | confidence_filter:reject=880, energy_gate:low_rms=592 | confidence_filter:reject=680.6, energy_gate:low_rms=205.7 |
| `parakeet_ja` | confidence_filter:reject=199, energy_gate:low_rms=78 | confidence_filter:reject=555.2, energy_gate:low_rms=90.6 |
| `whispers2t-base` | confidence_filter:reject=47, energy_gate:low_rms=3 | confidence_filter:reject=26.9, energy_gate:low_rms=2.9 |
