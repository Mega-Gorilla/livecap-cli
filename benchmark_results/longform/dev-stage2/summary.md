# 長尺の評価: YODAS2 日本語 `dev` (25 本、4 人、5.9 時間)

主の値は `fillers-notation_excluded` (片方にしか無い口ぐせを数えない、表記だけ違う区間を除く) の CER (%)。全体は誤りの総数 ÷ 正解の総文字数、話者平均は話者ごとの CER の平均。

| ASR | CER | 置換 | 脱落 | 挿入 | 話者平均 | そのまま | 数字をそろえる | 読みで比べる | RTF | 読み込み (秒) | GPU (torch, MB) | 区間の長さ (中央 / 最長, 秒) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `whispers2t-large-v3-gateoff` | **17.03** | 5.96 | 3.65 | 7.42 | 16.27 | 17.23 | 17.43 | 14.76 | 0.058 | 8.2 | 46 | 2.1 / 15 (VAD) |
| `qwen3asr_large-gateoff` | **17.75** | 9.07 | 4.05 | 4.63 | 16.47 | 18.42 | 17.34 | 14.28 | 0.088 | 11.9 | 4288 | 2.2 / 15 (VAD) |
| `qwen3asr-gateoff` | **19.76** | 10.32 | 4.21 | 5.24 | 18.50 | 20.43 | 19.24 | 15.83 | 0.083 | 7.08 | 1748 | 4.1 / 15 (VAD) |
| `qwen3asr-vadsilero-gateoff` | **20.27** | 10.24 | 5.48 | 4.55 | 19.05 | 20.93 | 19.70 | 16.37 | 0.083 | 3.57 | 1846 | 2.6 / 15 (VAD) |
| `parakeet_ja-gateoff` | **25.24** | 7.06 | 14.58 | 3.60 | 25.67 | 25.43 | 25.66 | 22.17 | 0.011 | 22.24 | 1250 | 14.0 / 15 (VAD) |
| `reazonspeech-gateoff` | **28.31** | 8.76 | 16.54 | 3.00 | 27.94 | 28.52 | 27.25 | 23.57 | 0.071 | 2.74 | 0 | 1.8 / 15 (VAD) |

「そのまま」「数字をそろえる」「読みで比べる」は、表記だけ違う区間も含めた値。GPU は torch が確保した量の最大で、CTranslate2 (whispers2t) や ONNX Runtime の分は数えない。区間の長さは、全部の動画に VAD の区間ごとの記録 (`vad_segments`) があれば落ちた区間も含めた VAD の区間 (VAD)、1 本でも無ければ ASR の出力の残った区間 (出力) の長さ。

## 話者ごとの CER (`fillers-notation_excluded`、%)

| ASR | 2004 | 2006 | 2013 | 2025 |
|---|---:|---:|---:|---:|
| `whispers2t-large-v3-gateoff` | 20.02 | 11.73 | 19.59 | 13.73 |
| `qwen3asr_large-gateoff` | 23.11 | 16.86 | 15.21 | 10.68 |
| `qwen3asr-gateoff` | 25.45 | 17.52 | 17.64 | 13.38 |
| `qwen3asr-vadsilero-gateoff` | 25.81 | 17.97 | 18.21 | 14.19 |
| `parakeet_ja-gateoff` | 25.83 | 29.65 | 18.56 | 28.63 |
| `reazonspeech-gateoff` | 33.83 | 23.10 | 24.39 | 30.44 |

## drop の内訳 (EnergyGate / confidence filter で落ちた区間の数と、その長さの合計)

| ASR | 区間の数 | 長さの合計 (秒) |
|---|---|---|
| `whispers2t-large-v3-gateoff` | confidence_filter:reject=84 | confidence_filter:reject=45.9 |
| `qwen3asr_large-gateoff` | confidence_filter:reject=84 | confidence_filter:reject=91.1 |
| `qwen3asr-gateoff` | confidence_filter:reject=728 | confidence_filter:reject=1348.8 |
| `qwen3asr-vadsilero-gateoff` | confidence_filter:reject=175 | confidence_filter:reject=287.4 |
| `parakeet_ja-gateoff` | confidence_filter:reject=569 | confidence_filter:reject=1876.4 |
| `reazonspeech-gateoff` | confidence_filter:reject=1693 | confidence_filter:reject=1801.8 |
