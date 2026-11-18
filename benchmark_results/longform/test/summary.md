# 長尺の評価: YODAS2 日本語 `test` (31 本、8 人、5.7 時間)

主の値は `fillers-notation_excluded` (片方にしか無い口ぐせを数えない、表記だけ違う区間を除く) の CER (%)。全体は誤りの総数 ÷ 正解の総文字数、話者平均は話者ごとの CER の平均。

| ASR | CER | 置換 | 脱落 | 挿入 | 話者平均 | そのまま | 数字をそろえる | 読みで比べる | RTF | 読み込み (秒) | GPU (torch, MB) | 区間の長さ (中央 / 最長, 秒) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `whispers2t-large-v3` | **12.38** | 4.78 | 4.92 | 2.68 | 12.16 | 12.66 | 12.72 | 9.48 | 0.064 | 7.94 | 104 | 2.0 / 55 |
| `qwen3asr_large` | **12.72** | 6.62 | 2.73 | 3.37 | 12.04 | 13.61 | 12.58 | 9.12 | 0.100 | 11.82 | 5576 | 2.1 / 51 |
| `qwen3asr` | **18.49** | 7.87 | 6.99 | 3.63 | 17.95 | 19.10 | 18.21 | 14.26 | 0.088 | 7.01 | 6592 | 3.8 / 193 |
| `reazonspeech` | **20.05** | 5.37 | 12.58 | 2.10 | 20.45 | 20.18 | 19.07 | 15.47 | 0.049 | 2.83 | 0 | 2.3 / 73 |
| `parakeet_ja` | **32.46** | 3.58 | 26.72 | 2.16 | 31.63 | 32.15 | 32.23 | 29.81 | 0.009 | 22.15 | 10210 | 14.0 / 540 |
| `whispers2t-base` | **37.78** | 19.65 | 5.82 | 12.31 | 35.64 | 38.02 | 38.00 | 28.85 | 0.015 | 1.57 | 104 | 2.0 / 55 |

「そのまま」「数字をそろえる」「読みで比べる」は、表記だけ違う区間も含めた値。GPU は torch が確保した量の最大で、CTranslate2 (whispers2t) や ONNX Runtime の分は数えない。区間の長さは、VAD の区間ごとの記録 (`vad_segments`) があれば落ちた区間も含めた VAD の区間、無ければ ASR の出力の残った区間の長さ (どちらかは summary.json の `segment_basis`)。

## 話者ごとの CER (`fillers-notation_excluded`、%)

| ASR | 2000 | 2012 | 2014 | 2016 | 2018 | 2031 | 2034 | 2035 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `whispers2t-large-v3` | 7.35 | 9.23 | 19.44 | 18.27 | 18.07 | 4.56 | 2.99 | 17.37 |
| `qwen3asr_large` | 10.72 | 10.92 | 18.24 | 19.38 | 13.60 | 6.39 | 6.09 | 10.96 |
| `qwen3asr` | 13.05 | 12.57 | 20.37 | 25.93 | 26.38 | 11.52 | 7.96 | 25.84 |
| `reazonspeech` | 9.60 | 17.82 | 22.47 | 36.77 | 22.86 | 15.59 | 6.02 | 32.47 |
| `parakeet_ja` | 7.72 | 26.34 | 32.09 | 45.34 | 56.37 | 5.24 | 68.96 | 10.99 |
| `whispers2t-base` | 30.39 | 32.86 | 51.05 | 50.56 | 48.90 | 16.91 | 23.41 | 31.01 |

## drop の内訳 (EnergyGate / confidence filter で落ちた区間の数と、その長さの合計)

| ASR | 区間の数 | 長さの合計 (秒) |
|---|---|---|
| `whispers2t-large-v3` | confidence_filter:reject=45, energy_gate:low_rms=2 | 記録なし |
| `qwen3asr_large` | confidence_filter:reject=55, energy_gate:low_rms=2 | 記録なし |
| `qwen3asr` | confidence_filter:reject=283, energy_gate:low_rms=119 | 記録なし |
| `reazonspeech` | confidence_filter:reject=1001, energy_gate:low_rms=585 | 記録なし |
| `parakeet_ja` | confidence_filter:reject=153, energy_gate:low_rms=57 | 記録なし |
| `whispers2t-base` | confidence_filter:reject=43, energy_gate:low_rms=2 | 記録なし |
