# EnergyGate(#292) 有効性・限界寄与 ablation (2026-07)

Issue: [#357](https://github.com/Mega-Gorilla/livecap-cli/issues/357)
対象コーパス: 星の王子さま JA full corpus (1375 sample、`.tmp/calibration_corpus_full/`、**git 管理外**)
harness: `benchmarks/confidence_calibration/energygate_ablation.py`

> ⚠️ 生音源は git に push しない。本 doc は集計数値のみを記録し、音源はローカル `.tmp/` を参照する。

## 1. 問い

realtime path (`StreamTranscriber`) は無音/非音声を **4 段**で落とす:

```
VAD → EnergyGate(#292, RMS gate, 既定 ON -45dBFS)
    → engine 空text guard(engine:empty_text, 常時 ON)
    → ConfidenceFilter(#334, ASR 信頼度, 既定 ON)
```

ConfidenceFilter 単体の有効性は Phase 2 で実測済み(`calibration-japan-engines-phase2-2026-07.md`、ReazonSpeech F1=0.934)。本調査は **EnergyGate が他 guard がある上で追加で必要か(marginal necessity)** に絞る。

## 2. 手法

corpus 全 sample に engine を1回実行し、3 guard の判定を **独立に記録**してから 4 config を simulate する。EnergyGate は本来 pre-engine だが、marginal を測るため engine は全件走らせる(EnergyGate が落とす sample を「engine に通したら他 guard が捕捉したか」を知るため)。

- `energy_drop` = `_segment_energy_dbfs(audio, max_frame_rms, 32ms) < -45 dBFS`(EnergyGate 実使用ロジック)
- `empty_text` = engine 出力が空(空text guard)
- `conf_reject` = `should_reject(result, FilterConfig())`(現行 main の production 既定閾値)

| config | drop(=suppression)条件 |
|---|---|
| baseline | `empty_text` |
| +energy | `energy_drop or empty_text` |
| +confidence | `empty_text or conf_reject` |
| **both**(production 既定) | `energy_drop or empty_text or conf_reject` |

confusion の positive class = `non_speech`(落とすべき)。`noisy_speech` は `speech` 扱い(落とすと false reject)。

engine: **reazonspeech**(avg_logprob, JA primary)/ **whispers2t**(no_speech_prob — 非音声検出特化 → EnergyGate 冗長性テストが最も厳しい)。

## 3. Preliminary(engine 不要 / EnergyGate 単層)

全 1375 サンプルの energy(dBFS, max_frame_rms/32ms)分布:

| label | n | p5 | median | p95 |
|---|---|---|---|---|
| speech | 449 | -34.8 | -24.4 | -16.2 |
| noisy_speech | 250 | -32.6 | -24.5 | -17.0 |
| non_speech | 676 | **-200(無音)** | -19.0 | -4.7 |

既定 -45 dBFS での drop: **speech false-drop 0/699 (0.00%)**、**non_speech drop 101/676 (14.9%)**(内訳 esc50 81 / phase1 13 / musan 7、うち 83件は -200dBFS のデジタル無音)。
→ EnergyGate は無音を落とし発話を守る。ただし大音量 non_speech 85% には無力(ConfidenceFilter の領域)。**この corpus の non_speech は ESC-50/MUSAN の大音量環境音に偏るため、EnergyGate には不利な(= 実価値を過小評価する)条件下の下限評価**。実 mic の VAD 誤検出はより低音量寄りで、EnergyGate の実効果はこれより高い可能性がある(realtime 分布検証は Layer 5 = 内部 Task #394)。

## 4. Ablation 結果

各 config の非音声抑制率 / speech false-drop 率(FRR)。positive class = non_speech(676件)、speech(+noisy)= 699件。

### 4.1 ReazonSpeech K2 (CPU Float32, signal=avg_logprob)

| config | non_speech 抑制 | speech FRR |
|---|---|---|
| baseline(空text のみ) | 0/676 (0.0%) | 0/699 (0.0%) |
| +energy | 101/676 (14.9%) | 0 (0.0%) |
| +confidence | 610/676 (90.2%) | 27 (3.9%) |
| **both**(production) | 610/676 (90.2%) | 27 (3.9%) |

EnergyGate marginal: energy が落とす 101件 → **unique=0 / overlap=101**。無音 101件で engine は全件幻聴(non-empty=101)だが **ConfidenceFilter が全件捕捉(conf PASS=0)**。
→ **avg_logprob engine では EnergyGate は品質面 100% 冗長**(both == confidence)。speech 害ゼロ。

### 4.2 WhisperS2T Large-v3 (signal=no_speech_prob)

| config | non_speech 抑制 | speech FRR |
|---|---|---|
| baseline(空text のみ) | 0/676 (0.0%) | 0/699 (0.0%) |
| +energy | 101/676 (14.9%) | 0 (0.0%) |
| +confidence | 295/676 (43.6%) | 6 (0.9%) |
| **both**(production) | 371/676 (54.9%) | 6 (0.9%) |

EnergyGate marginal: energy が落とす 101件 → **unique=76 / overlap=25**。無音 101件で engine は全件幻聴(non-empty=101)、そのうち **ConfidenceFilter が見逃す 76件を EnergyGate だけが捕捉(conf PASS=76)**。
→ **no_speech_prob engine では EnergyGate は相補的で必要**(both 54.9% ≫ confidence 43.6%、+76件 / +11.2pt)。これは Whisper の無音幻聴(silence hallucination)を `no_speech_prob` が検出できない既知の failure mode を EnergyGate が塞いでいる。speech 害ゼロ。

両 engine 共通: **speech への false-drop は -45dBFS で 0件**(EnergyGate 単体では speech を一切落とさない。最も静かな speech でも -39dB で -45 に ~6dB マージン)。

## 5. 結論

**Q1「機能しているか?」→ YES。** EnergyGate は無音 non_speech 101件(うち 83件はデジタル無音)を正しく落とし、speech は1件も落とさない。

**Q2「必要か?」→ signal family 依存。** 単純な「冗長だから不要」ではない:

| engine signal | EnergyGate marginal | 判定 |
|---|---|---|
| **avg_logprob**(reazonspeech / qwen3asr / voxtral) | unique=0(完全冗長) | 品質面は不要だが **無害 + ASR 計算節約**(無音で engine を回さない pre-ASR gate) |
| **no_speech_prob**(whispers2t) | **unique=76(+11.2pt)** | **相補的で必要**。Whisper の無音幻聴を塞ぐ唯一の guard |
| token_confidence_mean(parakeet/canary) | 未測定(要 follow-up) | avg_logprob 系と同様に engine 側の silence 挙動に依存 |

**推奨: EnergyGate は既定 ON を維持。** 理由: (1) speech 害ゼロ(-45dBFS は安全)、(2) WhisperS2T では品質必須、(3) 全 engine で無音時 ASR を短絡する安価な計算節約 gate。user の「問題(冗長)」仮説は avg_logprob engine に限って部分的に真だが、**削除すると WhisperS2T で無音幻聴が 76/676 (11%) 増える**ため削除は不可。

**注意点(過小評価バイアス)**: 本 corpus の non_speech は ESC-50/MUSAN の大音量環境音に偏り、EnergyGate に不利な下限条件。実 mic の VAD 誤検出はより低音量寄りで、EnergyGate の実効果は本結果より高い可能性が高い(realtime 分布での検証は Layer 5 = 内部 Task #394)。

**follow-up 候補**: (a) token_confidence 系(parakeet/canary)の測定で全 signal family をカバー、(b) `-45dBFS` 閾値の sweep で最適点確認(現状 speech マージン ~6dB は保守的で、より高くしても speech 害ゼロを保ちつつ非音声捕捉を増やせる可能性)。

## 6. 副次発見(別 issue 候補)

- `FileTranscriptionPipeline`(バッチ)はどのフィルターも通さない。CLI file 経路は実 API と乖離(`engine=` を取らない・`transcribe`/`to_srt` 不在)で `TypeError` の疑い。
- `compression_ratio_threshold` は予約 field で `should_reject` から未参照 = 死にシグナル(既存 Finding F5 = 内部 Task #391 と一致)。

## 再現コマンド

```pwsh
uv run python -m benchmarks.confidence_calibration.energygate_ablation \
    --engine reazonspeech --corpus-dir .tmp/calibration_corpus_full \
    --filter-by-language ja --output .tmp/energygate_ablation_reazonspeech.json
uv run python -m benchmarks.confidence_calibration.energygate_ablation \
    --engine whispers2t --corpus-dir .tmp/calibration_corpus_full \
    --filter-by-language ja --output .tmp/energygate_ablation_whispers2t.json
```
