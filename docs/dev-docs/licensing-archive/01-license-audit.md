# ライセンス監査結果

**作成日**: 2025-10-24
**監査対象**: LiveCap v2.1.0.1の全依存ライブラリ
**結論**: ✅ **デュアルライセンス実現可能（問題なし）**

---

## 📋 目次

1. [監査結果サマリー](#監査結果サマリー)
2. [ライセンス分類別詳細](#ライセンス分類別詳細)
3. [重要コンポーネントの詳細分析](#重要コンポーネントの詳細分析)
4. [リスク評価](#リスク評価)
5. [推奨事項](#推奨事項)

---

## 監査結果サマリー

### ✅ 結論：デュアルライセンスは100%実現可能

すべての依存ライブラリがGPLv3と互換性があり、商用利用も許可されています。

### ライセンス分類

| ライセンス | コンポーネント数 | GPL互換性 | 商用利用 | デュアルライセンス影響 |
|------------|-----------------|-----------|----------|----------------------|
| **MIT** | 12+ | ✅ 互換 | ✅ 許可 | 問題なし |
| **Apache 2.0** | 8+ | ✅ 互換 | ✅ 許可 | 問題なし |
| **BSD-3-Clause** | 7+ | ✅ 互換 | ✅ 許可 | 問題なし |
| **ISC** | 1 | ✅ 互換 | ✅ 許可 | 問題なし |
| **LGPL v3** | 1 (PySide6) | ✅ 互換* | ✅ 許可* | 問題なし* |
| **NVIDIA Open Model** | 1 (Riva) | ✅ 互換 | ✅ 許可 | 輸出規制に注意 |

\* 動的リンクの場合

### 重要な発見

1. **PySide6 (LGPL v3)**
   - ✅ 既に動的リンクで実装済み
   - ✅ PyInstallerビルドでも分離可能
   - ✅ アプリ本体のライセンスに制約なし

2. **NVIDIA Components**
   - ✅ NeMoはApache 2.0（商用OK）
   - ✅ Riva-Translate-4BはNVIDIA Open Model License（商用OK）
   - ⚠️ 輸出規制あり（U.S. Export Control Laws）

3. **その他の依存関係**
   - すべてPermissiveライセンス
   - 商用利用制限なし
   - コピーレフト要件なし（LGPL除く、ただし動的リンク）

---

## ライセンス分類別詳細

### MIT License（最も自由）

**特徴**: 著作権表示のみ必要、他は無制限

| パッケージ | 用途 | 重要度 |
|-----------|------|--------|
| **Whisper S2T** | 音声認識エンジン | 高 |
| **sounddevice** | オーディオ入出力 | 高 |
| **PyYAML** | 設定ファイル | 中 |
| **onnxruntime** | ONNX推論 | 高 |
| **pydub** | オーディオ処理 | 中 |
| **tqdm** | プログレスバー | 低 |
| **omegaconf** | 設定管理 | 中 |
| **hydra-core** | 設定フレームワーク | 中 |
| **moviepy** | 動画処理 | 中 |
| **deep-translator** | 翻訳API | 中 |

**GPLv3との互換性**: ✅ 完全互換
**商用利用**: ✅ 無制限
**帰属表示要件**: ✅ `THIRD_PARTY_LICENSES.md`で対応済み

---

### Apache License 2.0

**特徴**: 特許条項あり、著作権・変更通知必要

| パッケージ | 用途 | 重要度 |
|-----------|------|--------|
| **NVIDIA NeMo** | ASRフレームワーク | 高 |
| **ReazonSpeech** | 日本語ASR | 高 |
| **transformers** | HuggingFace Transformers | 高 |
| **MistralAI Voxtral** | 多言語ASR | 高 |
| **aiohttp** | 非同期HTTP | 中 |
| **ffmpeg-python** | FFmpeg wrapper | 中 |

**GPLv3との互換性**: ✅ 互換（GPLv3がApache 2.0を明示的に許可）
**商用利用**: ✅ 無制限
**特許条項**: ✅ 特許権の自動付与（ユーザーに有利）
**帰属表示要件**: ✅ `THIRD_PARTY_LICENSES.md`で対応済み

---

### BSD-3-Clause License

**特徴**: 非常に自由、著作権表示のみ

| パッケージ | 用途 | 重要度 |
|-----------|------|--------|
| **PyTorch** | 深層学習フレームワーク | 最高 |
| **torchaudio** | PyTorchオーディオ | 高 |
| **torchvision** | PyTorchビジョン | 低 |
| **NumPy** | 数値計算 | 高 |
| **SciPy** | 科学計算 | 高 |
| **soundfile** | 音声ファイルI/O | 高 |
| **websockets** | WebSocketサーバー | 中 |
| **markdown** | Markdown処理 | 低 |

**GPLv3との互換性**: ✅ 完全互換
**商用利用**: ✅ 無制限
**帰属表示要件**: ✅ `THIRD_PARTY_LICENSES.md`で対応済み

---

### ISC License

**特徴**: MITとほぼ同等（より簡潔）

| パッケージ | 用途 | 重要度 |
|-----------|------|--------|
| **librosa** | 音声信号処理 | 高 |

**GPLv3との互換性**: ✅ 完全互換
**商用利用**: ✅ 無制限

---

### LGPL v3（重要）

**特徴**: 動的リンクなら親ソフトは自由なライセンス可

| パッケージ | 用途 | 重要度 | リンク方法 |
|-----------|------|--------|-----------|
| **PySide6** | GUIフレームワーク | 最高 | 動的リンク |

#### PySide6の詳細分析

**現在の実装状態**:
```python
# src/gui_main.py で import
from PySide6.QtWidgets import QApplication
# → 動的リンク（Pythonの標準的なimport）
```

**PyInstallerビルド後**:
```
app.exe (LiveCapバイナリ)
└─ _internal/
   └─ PySide6/
      ├─ QtCore.pyd  ← 分離されたDLL
      ├─ QtGui.pyd
      └─ QtWidgets.pyd
```

**LGPL v3要件への対応**:
- ✅ **動的リンク**: PySide6はDLL/PYDファイルとして分離
- ✅ **ソースコード提供**: `THIRD_PARTY_LICENSES.md`でURL明記
- ✅ **置き換え可能**: ユーザーは`_internal/PySide6/`を置き換え可能
- ✅ **ライセンス通知**: アプリの"About"ダイアログに表示済み

**結論**: ✅ **問題なし** - LiveCap本体は任意のライセンス適用可能

---

### NVIDIA Open Model License

**特徴**: NVIDIAの独自ライセンス、商用利用可だが輸出規制あり

| モデル | 用途 | ライセンス |
|-------|------|-----------|
| **Riva-Translate-4B-Instruct** | 翻訳モデル | NVIDIA Open Model License |

**ライセンス要件**:
- ✅ 商用利用許可
- ✅ 改変・配布許可
- ⚠️ 帰属表示必須: "Licensed by NVIDIA Corporation under the NVIDIA Open Model License"
- ⚠️ 輸出規制: U.S. Export Administration Regulations (EAR) 対象

**対応状況**:
- ✅ 帰属表示: `THIRD_PARTY_LICENSES.md`および`about_dialog.py`で実装済み
- ⚠️ 輸出規制: ユーザー責任（通常の使用では問題なし）

**GPLv3との互換性**: ✅ 問題なし（商用利用可、ソース公開義務なし）

---

## 重要コンポーネントの詳細分析

### PyTorch（BSD-3-Clause）

**重要度**: 最高（プロジェクトの基盤）

```
PyTorch 2.5.1+cu124
├─ BSD-3-Clause License
├─ Copyright (c) Meta Platforms, Inc.
└─ 依存関係も主にBSD/MIT
```

**デュアルライセンスへの影響**: なし
**商用利用**: 完全に自由
**再配布**: ソースコード不要

---

### NVIDIA NeMo（Apache 2.0）

**重要度**: 高（主要ASRエンジン）

```
NeMo Toolkit 1.24.0
├─ Apache License 2.0
├─ Copyright (c) NVIDIA Corporation
└─ Parakeet/Canaryモデルを含む
```

**デュアルライセンスへの影響**: なし
**特許条項**: ✅ NVIDIAから特許ライセンス自動付与
**再配布**: ソースコード不要、著作権表示のみ

---

### PySide6（LGPL v3）

**重要度**: 最高（GUIフレームワーク）

```
PySide6 6.8.0.2
├─ LGPL v3 License
├─ 動的リンク（Python import）
└─ Qt 6.8.0ベース
```

**LGPL遵守状況チェック**:
- ✅ ソースコード提供先を明記
- ✅ LGPLライセンス全文を同梱（`licenses/COPYING.LGPLv3`）
- ✅ 動的リンクの実装
- ✅ ユーザーによる置き換え方法を文書化

**結論**: ✅ 完全準拠 - LiveCap本体のライセンスに制約なし

---

## リスク評価

### 🟢 リスクなし

以下のライセンスは完全に自由で、デュアルライセンスに影響なし：

- MIT License
- Apache 2.0
- BSD-3-Clause
- ISC License

### 🟡 要注意（既に対応済み）

#### PySide6 (LGPL v3)
- **リスク**: 静的リンクするとLiveCapもLGPLになる
- **対策**: ✅ 動的リンクで実装済み
- **ステータス**: 問題なし

#### NVIDIA Riva-Translate-4B
- **リスク**: 輸出規制違反
- **対策**: ユーザー責任として利用規約に明記
- **ステータス**: 通常使用では問題なし

### 🔴 リスク高（該当なし）

GPL v2, AGPL等のコピーレフトライセンスは使用していません。

---

## 推奨事項

### ✅ 実施済み

1. **THIRD_PARTY_LICENSES.md作成**
   - すべての依存ライブラリを列挙
   - 著作権表示とソースURL
   - PySide6の置き換え方法を文書化

2. **Aboutダイアログ**
   - PySide6のLGPL通知
   - 主要ライブラリのリンク
   - ライセンスドキュメントへのリンク

3. **動的リンク実装**
   - PySide6を分離可能な形で実装
   - PyInstallerでも分離構造を維持

### 🔄 OSS化前に追加推奨

1. **README.mdにライセンス表示**
   ```markdown
   ## Licenses

   LiveCap: GPL v3 / Commercial License
   Dependencies: See THIRD_PARTY_LICENSES.md
   ```

2. **CONTRIBUTING.mdでライセンス説明**
   - 貢献者はGPL v3に同意
   - CLAの説明（後述）

3. **COPYRIGHTファイル追加**（オプション）
   ```
   LiveCap Copyright (c) 2025 Pine Lab

   Third-party components:
   - See THIRD_PARTY_LICENSES.md
   ```

---

## ライセンス互換性マトリックス

| 依存ライブラリ | GPLv3互換 | 商用OK | アプリへの制約 |
|---------------|----------|--------|---------------|
| MIT | ✅ | ✅ | なし |
| Apache 2.0 | ✅ | ✅ | なし |
| BSD | ✅ | ✅ | なし |
| ISC | ✅ | ✅ | なし |
| LGPL v3 | ✅* | ✅* | なし* |
| NVIDIA Open Model | ✅ | ✅ | 輸出規制のみ |

\* 動的リンクの場合

---

## 参考資料

### 公式ライセンステキスト
- [GPL v3](https://www.gnu.org/licenses/gpl-3.0.html)
- [LGPL v3](https://www.gnu.org/licenses/lgpl-3.0.html)
- [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- [MIT License](https://opensource.org/licenses/MIT)
- [BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause)

### ライセンス互換性
- [GNU License Compatibility](https://www.gnu.org/licenses/license-list.html)
- [Apache 2.0 and GPLv3 Compatibility](https://www.apache.org/licenses/GPL-compatibility.html)

### LGPL動的リンク
- [LGPL and Dynamic Linking](https://www.gnu.org/licenses/gpl-faq.html#LGPLStaticVsDynamic)
- [Qt LGPL Obligations](https://www.qt.io/licensing/open-source-lgpl-obligations)

---

## 改訂履歴

| 日付 | バージョン | 変更内容 |
|------|-----------|---------|
| 2025-10-24 | 1.0 | 初版作成 |

---

## 次のステップ

1. ✅ この監査結果を確認
2. → [02-dual-license-strategy.md](02-dual-license-strategy.md) でデュアルライセンスの詳細を確認
3. → [03-steam-integration.md](03-steam-integration.md) でSteam対応を確認
