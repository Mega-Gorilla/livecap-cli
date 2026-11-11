# torch で推論する engine の bf16 化 (2026-11-10、#470)

Qwen3-ASR (`qwen3asr` / `qwen3asr_large`)・Parakeet (`parakeet` / `parakeet_ja`)・Canary (`canary`) は fp32 で読み込んでいた。bf16 にしたときの精度・VRAM・速度・confidence filter の判定を、fp32 と同じ条件で比べた記録。

## 結論

- **精度は同等**: CER / WER の差は 7 通りすべてで ±0.0013 以内 (bf16 が良い場合もある)
- **VRAM は約半分**: peak reserved で 49〜50% 減 (1.7B は 9,076 → 4,564 MB)
- **速度は同等か少し速い**: Qwen3-ASR で RTF が 5〜8% 改善。NeMo の 3 つは同じ
- **confidence filter の判定は実音声で 100% 一致**: 区間ごとの採否 (計 777 区間) が fp32 と同じ
- 以上から、`dtype="auto"` (既定) は **bf16 をネイティブに計算できる GPU (compute capability 8.x 以降) で `bfloat16`**、CPU とそれより前の GPU で `float32` とした

## 条件

| 項目 | 値 |
|---|---|
| 機材 | Ubuntu 26.04、RTX 4070 Ti 12 GB (Ada、compute capability 8.9)、driver 580 |
| ソフト | torch 2.9.1+cu128、nemo-toolkit 2.3.0、transformers 4.57.6 |
| データ | ベンチマークの standard mode — ja: JSUT 99 files (393 s)、en: LibriSpeech test-clean 100 files (457 s) |
| VAD | 既存 preset `silero_{lang}_{engine}.json` で固定 (`qwen3asr_large` は `qwen3asr` の preset) |
| 評価 | `VADObjective(objective="production")` を `optuna.trial.FixedTrial` で 1 回 — CLI と同じ後段 (EnergyGate → transcribe → confidence filter `on`) を通した CER (ja) / WER (en) |
| dtype | `EngineFactory.create_engine(engine_id, device="cuda", dtype=...)` で明示 (`float32` → `bfloat16` の順) |
| VRAM | `torch.cuda.max_memory_reserved()` (読み込み + 評価 + 合成音の転写の全体) |

## 精度・VRAM・速度

| engine | 言語 | CER/WER fp32 | CER/WER bf16 | VRAM fp32 (MB) | VRAM bf16 (MB) | RTF fp32 | RTF bf16 |
|---|---|---:|---:|---:|---:|---:|---:|
| `parakeet` | en | 0.0308 | **0.0295** | 2,500 | **1,260** | 0.026 | 0.026 |
| `parakeet_ja` | ja | 0.0746 | 0.0750 | 2,492 | **1,250** | 0.018 | 0.019 |
| `canary` | en | 0.0651 | 0.0651 | 3,480 | **1,750** | 0.047 | 0.046 |
| `qwen3asr` | ja | 0.1107 | 0.1107 | 3,690 | **1,866** | 0.119 | 0.113 |
| `qwen3asr` | en | 0.0412 | 0.0422 | 3,690 | **1,862** | 0.117 | 0.111 |
| `qwen3asr_large` | ja | 0.0797 | 0.0802 | 9,076 | **4,564** | 0.127 | 0.118 |
| `qwen3asr_large` | en | 0.0216 | 0.0216 | 9,070 | **4,562** | 0.125 | 0.115 |

- 区間ごとの転写文は 92〜100% が完全一致 (parakeet 97/105、parakeet_ja 109/111、canary 112/119、qwen3asr ja 102/110・en 109/111、qwen3asr_large ja 107/110・en 111/111)
- 転写の例外は 0 件 (bf16 で落ちる経路は無い)
- 読み込み直後の peak allocated は fp32 / bf16 で parakeet 2,417 / 1,209、parakeet_ja 2,427 / 1,218、canary 3,424 / 1,716、qwen3asr 3,587 / 1,798、qwen3asr_large 8,971 / 4,490 MB
- 読み込み時間は比べていない (NeMo の fp32 は初回ダウンロードを含む)

## confidence filter の判定

engine が返す信頼度 (`token_confidence_mean` / `avg_logprob`) と、それによる採否を区間ごとに突き合わせた。閾値は本番と同じ engine 名 (`get_engine_name()`) で引いた (Qwen3-ASR は `"qwen3-asr": -0.42`、NeMo の 3 つは `token_conf_threshold`)。

### 実音声 (上の評価の VAD 区間)

| engine | 言語 | 採否の一致 | 棄却 fp32 / bf16 | 信頼度の差 (平均 / 最大) |
|---|---|---:|---:|---|
| `parakeet` | en | 105/105 | 1 / 1 | 0.0029 / 0.0203 |
| `parakeet_ja` | ja | 111/111 | 0 / 0 | 0.0012 / 0.0065 |
| `canary` | en | 119/119 | 1 / 1 | 0.0025 / 0.0560 |
| `qwen3asr` | ja | 110/110 | 0 / 0 | 0.0038 / 0.0501 |
| `qwen3asr` | en | 111/111 | 1 / 1 | 0.0019 / 0.0214 |
| `qwen3asr_large` | ja | 110/110 | 0 / 0 | 0.0021 / 0.0125 |
| `qwen3asr_large` | en | 111/111 | 1 / 1 | 0.0020 / 0.0399 |

### 合成音 (`benchmarks.non_speech_filter.corpus`: 非音声 8 + 音声の代わり 5)

本番では ASR の前に EnergyGate (-45 dBFS) が入るので、そこを通る 12 件 (非音声 7 + 音声の代わり 5、`silence_amplified` が落ちる) で比べた。

| engine | 言語 | 非音声を止めた数 fp32 / bf16 (/7) | 音声の代わりを残した数 fp32 / bf16 (/5) |
|---|---|---:|---:|
| `parakeet` | en | 7 / 7 | 5 / 5 |
| `parakeet_ja` | ja | 7 / 7 | 0 / 0 |
| `canary` | en | 7 / 7 | 1 / 0 |
| `qwen3asr` | ja | 7 / 7 | 1 / 1 |
| `qwen3asr` | en | 7 / 7 | 0 / 0 |
| `qwen3asr_large` | ja | 6 / 6 | 2 / 2 |
| `qwen3asr_large` | en | 7 / 7 | 0 / 1 |

- **非音声を止めた数はすべて同じ**
- 違いは `short_utterance_ok` (合成した短い発話の代わり) の 1 件だけ。canary は fp32 で「uh huh.」(信頼度 0.037) を残し、bf16 では「mm hmm」(1.8e-5) で棄却した。qwen3asr_large (en) は逆に、bf16 で残した。どちらも転写文自体が変わった閾値近くの例で、実音声には出ていない
- 合成の「音声の代わり」は実際の発話ではなく、parakeet_ja / qwen3asr (en) は fp32 でも 0 件 — この表は判定のずれを見るためのもので、短い発話の再現率の評価ではない

## 実装メモ

- **判定**: `livecap_cli.utils.resolve_torch_dtype()`。`auto` は CUDA かつ `torch.cuda.get_device_capability()` の major が 8 以上で `bfloat16`。Turing 以前は bf16 を模擬実行するだけで遅いので fp32 に留める。fp16 は桁あふれの検証をしていないので自動では選ばない (明示はできる)
- **Qwen3-ASR**: `from_pretrained(dtype="bfloat16")` で読み込み時から bf16 (transformers は dtype を名前で受ける)
- **NeMo (Parakeet / Canary)**: `restore_from(map_location="cpu")` で復元してから `model.to(device=..., dtype=...)` で 1 回で移す。`map_location="cuda"` で復元すると fp32 の重みが GPU に載り、bf16 にしても読み込み中の peak が fp32 の大きさに残る。NeMo の `AudioPreprocessor` は model の dtype によらず特徴量を fp32 で計算し、出力だけを model の dtype に合わせる (`dtype_sentinel_tensor`) ので、model 全体を `.to(bf16)` してよい
- **canary の fp32 の VRAM**: issue-73 の 6,830 MB は `map_location="cuda"` の復元で重みが GPU に 2 重に載っていた値。CPU で復元してから移すと fp32 でも 3,480 MB
- `ModelMemoryCache` の key に dtype を入れ、fp32 の model を bf16 の要求に返さない

## 対象外

- **Voxtral**: すでに CUDA で fp16 (重みは半分)。bf16 にしてもメモリーは減らない
- **WhisperS2T**: CTranslate2 で GPU は float16、CPU は int8 (`compute_type="auto"`)
- **ReazonSpeech**: sherpa-onnx (CPU 専用、int8 の選択肢あり)

## 残り

- `qwen3asr_large` の confidence filter の閾値 (いまは 0.6B と同じ `"qwen3-asr": -0.42`。表示名の先頭語で引くので 1.7B にも当たる) は、#470 では決めず、**雑音・非音声を含む長尺の評価 (#488) で bf16 の状態で**検証する (静かな読み上げでは誤字幕がほとんど出ず、閾値を合わせられないため)
- **`vram_required_mb` は fp32 の実測値にしてある** (parakeet 2,500 / parakeet_ja 2,492 / canary 3,480 MB、#470 Phase 3 で qwen3asr 3,690 / qwen3asr_large 9,076 MB も同じ方針で入れた)。`EngineMetadata.recommend()` は GPU の世代も解決後の dtype も受け取らないので、bf16 の値を入れると、`auto` が fp32 になる Ampere より前の GPU で「収まる」と誤判定する (PR #483 レビュー F1)。bf16 対応 GPU では実際の使用量の約 2 倍で判定することになる。dtype ごとの値で判定するには `recommend()` に GPU の世代を渡す API の変更が要る (未対応)
