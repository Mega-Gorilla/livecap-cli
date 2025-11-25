# VAD (Voice Activity Detection) 比較分析

> **作成日:** 2025-11-25
> **目的:** livecap-cli の VAD バックエンド選択のための参考資料
> **関連:** Phase 1 実装 (#69)

---

## 1. 概要

VAD（Voice Activity Detection）は音声区間検出を行うコンポーネントで、リアルタイム文字起こしにおいて以下の役割を担う：

- 発話開始/終了の検出
- 無音区間のフィルタリング
- ASR エンジンへの入力セグメント生成

本ドキュメントでは、livecap-cli での採用候補となる VAD を比較分析する。

---

## 2. 候補 VAD 一覧

| VAD | 正式名称 | ライセンス | 実装 | サイズ |
|-----|----------|-----------|------|--------|
| **Silero VAD** | Silero Voice Activity Detector | MIT | PyTorch (ONNX/JIT) | ~2MB |
| **TenVAD** | TEN Voice Activity Detector | 独自（複雑） | C | 300-500KB |
| **JaVAD** | Just Another Voice Activity Detector | MIT | PyTorch | 中程度 |
| **NeMo VAD** | NVIDIA NeMo MarbleNet | Apache 2.0 | PyTorch (GPU推奨) | 重い |
| **WebRTC VAD** | WebRTC Voice Activity Detector | BSD | C | 軽量 |

---

## 3. 精度比較

### 3.1 重要な前提

**VAD の精度評価はデータセットに強く依存する。** 各ベンダーは自社に有利なデータセットでベンチマークを公開する傾向があり、単純な比較は困難。

### 3.2 AVA-Speech ベンチマーク

JaVAD 作者による比較。映画音声（AVA-Speech 18.5時間）での評価。

| モデル | Precision | Recall | **F1** | **AUROC** | CPU時間 |
|--------|-----------|--------|--------|-----------|---------|
| NeMo VAD | 0.7676 | 0.9526 | 0.8502 | 0.9201 | 56.94s |
| Silero VAD | **0.9678** | 0.6503 | 0.9050 | 0.9169 | 695.58s |
| JaVAD tiny | 0.9263 | 0.8846 | 0.8961 | 0.9550 | 476.93s |
| JaVAD balanced | 0.9284 | 0.8938 | 0.9108 | 0.9642 | 220.00s |
| **JaVAD precise** | 0.9359 | 0.8980 | **0.9166** | **0.9696** | 236.61s |

**出典:** [JaVAD GitHub](https://github.com/skrbnv/javad)

**考察:**
- **JaVAD precise** が F1/AUROC 共に最高
- **Silero** は Precision が非常に高いが Recall が低い（発話を取りこぼしやすい）
- **NeMo** は最速だが精度は低め
- **Silero** は最も遅い

### 3.3 Silero マルチドメイン評価

Silero が公開している 17時間のマルチドメインデータセット（AliMeeting, Earnings21, MSDWild, AISHELL-4, VoxConverse, Libriparty 等）での評価。

| モデル | ROC-AUC | Accuracy |
|--------|---------|----------|
| WebRTC | 0.73 | 0.74 |
| TenVAD | 0.93 | 0.87 |
| Silero v5 | 0.96 | 0.91 |
| **Silero v6** | **0.97** | **0.92** |

**出典:** [Silero VAD Wiki - Quality Metrics](https://github.com/snakers4/silero-vad/wiki/Quality-Metrics)

**考察:**
- **Silero v6** が最高スコア
- **TenVAD** も WebRTC より大幅に良いが、Silero v6 には及ばない
- Silero 自身が用意したデータセットなので、Silero に有利なバイアスの可能性あり

### 3.4 TenVAD 自己評価

TenVAD が公開している LibriSpeech + GigaSpeech + DNS Challenge での PR カーブ比較。

**結果:** PR カーブ全域で TenVAD が Silero/WebRTC を上回ると主張

**出典:** [TenVAD GitHub](https://github.com/TEN-framework/ten-vad), [TenVAD HuggingFace](https://huggingface.co/TEN-framework/ten-vad)

**考察:**
- 具体的な F1 数値は非公開
- TenVAD 側のデータセットなので、TenVAD に有利なバイアスの可能性あり
- 発話終了検出の遅延が Silero より小さいと主張

### 3.5 精度まとめ

| データセット条件 | 1位 | 2位 | 3位 |
|-----------------|-----|-----|-----|
| AVA-Speech（映画音声） | **JaVAD** | Silero | NeMo |
| Silero マルチドメイン | **Silero v6** | TenVAD | WebRTC |
| TenVAD 自己評価 | **TenVAD** | Silero | WebRTC |

**結論:** データセットによって評価が逆転するため、**実際の使用環境でのベンチマークが必要**。

---

## 4. 速度・リソース比較

### 4.1 RTF (Real-Time Factor) 比較

TenVAD の HuggingFace ページより。RTF が低いほど高速。

| プラットフォーム | CPU | TenVAD RTF | Silero RTF |
|-----------------|-----|------------|------------|
| Linux | Xeon Gold 6348 | **0.0086** | 0.0127 |
| Linux | Ryzen 9 5900X | 0.0150 | - |
| Windows | i7-10710U | 0.0150 | - |
| macOS | M1 | 0.0160 | - |

**出典:** [TenVAD HuggingFace](https://huggingface.co/TEN-framework/ten-vad)

### 4.2 AVA-Speech での CPU 処理時間

18.5時間の音声を処理するのにかかった時間（Ryzen 9 3900XT）。

| モデル | 処理時間 | 相対速度 |
|--------|----------|----------|
| **NeMo VAD** | 56.94s | 最速 |
| JaVAD balanced | 220.00s | 3.9x |
| JaVAD precise | 236.61s | 4.2x |
| JaVAD tiny | 476.93s | 8.4x |
| **Silero VAD** | 695.58s | 12.2x（最遅） |

**出典:** [JaVAD GitHub](https://github.com/skrbnv/javad)

### 4.3 ライブラリサイズ

| VAD | サイズ |
|-----|--------|
| **TenVAD** | 300-500KB（最小） |
| WebRTC | 軽量 |
| Silero | 2.16MB (JIT) / 2.22MB (ONNX) |
| JaVAD | 中程度 |
| NeMo | 重い（依存含む） |

### 4.4 速度まとめ

| 観点 | 最速 | 中間 | 最遅 |
|------|------|------|------|
| RTF | TenVAD | - | Silero |
| CPU処理時間 | NeMo | JaVAD | Silero |
| ライブラリサイズ | TenVAD | WebRTC | NeMo |

---

## 5. ライセンス比較

| VAD | ライセンス | 商用利用 | 備考 |
|-----|-----------|----------|------|
| **Silero VAD** | MIT | ○ | 制限なし |
| **JaVAD** | MIT | ○ | 制限なし |
| **NeMo VAD** | Apache 2.0 | ○ | 制限なし |
| **WebRTC VAD** | BSD | ○ | 制限なし |
| **TenVAD** | 独自 | △ | 複雑、要確認 |

**TenVAD のライセンス懸念:**
- 独自のライセンス体系
- 商用利用の条件が不明確
- livecap-cli での採用には慎重な検討が必要

---

## 6. 各 VAD の特徴詳細

### 6.1 Silero VAD

**リポジトリ:** https://github.com/snakers4/silero-vad

**長所:**
- 最も広く使われている（コミュニティ・ドキュメント充実）
- MIT ライセンス
- ONNX 対応（livecap-core は既に onnxruntime に依存）
- マルチドメインで安定した精度
- v5/v6 で 3 倍高速化

**短所:**
- AVA-Speech ベンチマークでは JaVAD に精度で劣る
- Recall が低い傾向（発話を取りこぼしやすい）

#### Silero VAD v5/v6 パラメータ詳細

**固定設定（v5 以降）:**

| 設定 | 16kHz | 8kHz |
|------|-------|------|
| チャンクサイズ | 512 samples (32ms) | 256 samples (32ms) |
| サンプリングレート | 16000 Hz | 8000 Hz |

**get_speech_timestamps パラメータ:**

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `threshold` | 0.5 | 音声判定閾値（0.0-1.0） |
| `sampling_rate` | 16000 | サンプリングレート |
| `min_speech_duration_ms` | 250 | 最小音声持続時間 |
| `max_speech_duration_s` | inf | 最大音声持続時間 |
| `min_silence_duration_ms` | 100 | 最小無音区間 |
| `speech_pad_ms` | 30 | 音声前後のパディング |
| `neg_threshold` | threshold - 0.15 | ノイズ閾値 |

**VADIterator パラメータ（ストリーミング用）:**

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `threshold` | 0.5 | 音声判定閾値 |
| `sampling_rate` | 16000 | サンプリングレート |
| `min_silence_duration_ms` | 100 | 最小無音区間 |
| `speech_pad_ms` | 30 | パディング |

**ストリーミング使用例:**

```python
from silero_vad import load_silero_vad, VADIterator

model = load_silero_vad(onnx=True)
vad_iterator = VADIterator(
    model,
    threshold=0.5,
    sampling_rate=16000,
    min_silence_duration_ms=100,
    speech_pad_ms=30,
)

# チャンク処理
for chunk in audio_chunks:
    speech_dict = vad_iterator(chunk, return_seconds=True)
    if speech_dict:
        print(speech_dict)

# 状態リセット（新しい音声ストリーム開始時）
vad_iterator.reset_states()
```

**直接モデル呼び出し:**

```python
# 単一チャンクの確率取得
speech_prob = model(chunk, sampling_rate).item()
```

**参考リンク:**
- [Silero VAD GitHub](https://github.com/snakers4/silero-vad)
- [Quality Metrics](https://github.com/snakers4/silero-vad/wiki/Quality-Metrics)
- [FAQ](https://github.com/snakers4/silero-vad/wiki/FAQ)

### 6.2 TenVAD

**リポジトリ:** https://github.com/TEN-framework/ten-vad

**長所:**
- 最軽量（300-500KB）
- 最高速（RTF 最小）
- 発話終了検出の遅延が小さい
- C 実装で依存が少ない

**短所:**
- ライセンスが複雑（商用利用に懸念）
- Silero マルチドメイン評価では Silero v6 に劣る
- コミュニティが小さい

### 6.3 JaVAD

**リポジトリ:** https://github.com/skrbnv/javad

**名称:** Just Another Voice Activity Detector（日本語特化ではない）

**長所:**
- AVA-Speech で最高精度（F1/AUROC）
- MIT ライセンス
- balanced/precise/tiny の 3 モデルで用途に応じた選択可能
- Silero より高速

**短所:**
- 知名度が低い（Silero と比較して）
- PyTorch 依存
- マルチドメイン評価データなし

### 6.4 NeMo VAD (MarbleNet)

**ドキュメント:** https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speech_classification/models.html

**長所:**
- CPU 処理が最速
- NVIDIA エコシステムとの統合
- Apache 2.0 ライセンス

**短所:**
- 精度が他より低い（AVA-Speech ベンチマーク）
- 依存が重い（NeMo toolkit 全体）
- GPU 前提の設計

### 6.5 WebRTC VAD

**長所:**
- 軽量
- BSD ライセンス
- 実績がある

**短所:**
- 精度が最も低い
- 古い実装

---

## 7. livecap-cli での選択基準

### 7.1 要件

| 要件 | 優先度 | 説明 |
|------|--------|------|
| ライセンス | 必須 | 商用利用可能であること |
| CPU 動作 | 必須 | GPU なしでも動作 |
| 低レイテンシ | 高 | リアルタイム処理に必要 |
| 精度 | 高 | 発話の取りこぼし・誤検出を最小化 |
| 軽量 | 中 | インストールサイズ、メモリ使用量 |
| 保守性 | 中 | コミュニティ、ドキュメントの充実度 |

### 7.2 評価マトリクス

| VAD | ライセンス | CPU動作 | レイテンシ | 精度 | 軽量 | 保守性 | 総合 |
|-----|-----------|---------|-----------|------|------|--------|------|
| Silero | ◎ | ○ | △ | ○ | ○ | ◎ | **B+** |
| TenVAD | △ | ◎ | ◎ | ○ | ◎ | △ | **B** |
| JaVAD | ◎ | ○ | ○ | ◎ | ○ | △ | **B+** |
| NeMo | ◎ | △ | ○ | △ | × | ○ | **C+** |
| WebRTC | ◎ | ◎ | ◎ | × | ◎ | ○ | **C** |

---

## 8. 決定事項

### 8.1 Phase 1 での方針（決定済み）

1. **Silero VAD をデフォルトとして採用**
   - MIT ライセンス（商用利用に問題なし）
   - 広く使われており、コミュニティ・ドキュメントが充実
   - ONNX 対応（livecap-core は既に onnxruntime に依存）

2. **プラグイン可能な設計**
   - `VADBackend` Protocol を定義
   - 将来的に他のバックエンド（JaVAD 等）を追加可能

3. **TenVAD の除外**
   - ライセンス懸念のため `pyproject.toml` から削除
   - デフォルトおよび optional dependency としても採用しない

### 8.2 将来の検討事項（優先度低）

1. **実データでのベンチマーク**
   - livecap-cli の実際のユースケース（VRChat/Discord 音声等）で評価
   - 日本語・英語混在環境での精度比較
   - ベンチマーク結果に基づき、必要に応じて他の VAD を追加または切り替え

2. **追加候補**
   - **JaVAD**: AVA-Speech で最高精度、MIT ライセンス
   - **NeMo VAD**: GPU 環境での高速処理が必要な場合

---

## 9. 参考リンク

- [Silero VAD GitHub](https://github.com/snakers4/silero-vad)
- [Silero VAD Quality Metrics](https://github.com/snakers4/silero-vad/wiki/Quality-Metrics)
- [TenVAD GitHub](https://github.com/TEN-framework/ten-vad)
- [TenVAD HuggingFace](https://huggingface.co/TEN-framework/ten-vad)
- [JaVAD GitHub](https://github.com/skrbnv/javad)
- [NVIDIA NeMo VAD Docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speech_classification/models.html)

---

## 変更履歴

| 日付 | 変更内容 |
|------|----------|
| 2025-11-25 | 初版作成 |
