# Issue #221: Qwen3-ASR エンジン統合

> **Status**: 📋 PLANNING
> **作成日**: 2026-02-04
> **関連 Issue**: #221

---

## 1. 概要

Alibaba Cloud Qwen チームが開発した [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) を新しい ASR エンジンとして livecap-cli に統合する。

### 1.1 Qwen3-ASR の特徴

| 特徴 | 詳細 |
|------|------|
| 性能 | Whisper-large-v3 を上回る WER（特に中国語） |
| モデルサイズ | 0.6B / 1.7B の2バリエーション |
| 対応言語 | 30言語 + 22中国語方言（日本語含む） |
| 推論モード | オフライン / ストリーミング |
| タイムスタンプ | ForcedAligner による高精度アライメント |
| ライセンス | Apache 2.0 |

### 1.2 性能ベンチマーク

| データセット | Qwen3-ASR-1.7B | Whisper-large-v3 | 備考 |
|-------------|----------------|------------------|------|
| Librispeech clean (WER) | **1.63%** | 1.51% | 英語 |
| Librispeech other (WER) | **3.38%** | 3.97% | 英語（ノイズあり） |
| AISHELL-2 (WER) | **2.71%** | 5.06% | 中国語 |

---

## 2. 事前調査結果

### 2.1 Windows 環境動作テスト（2026-02-04）

> **Note**: このテストは Python 3.13 で実施されましたが、livecap-cli のサポート範囲は
> `>=3.10,<3.13` です。Python 3.10-3.12 での追加検証が必要です。

| テスト | 結果 |
|--------|------|
| nagisa インストール | ✅ Python 3.13 で動作 |
| qwen-asr インストール | ✅ 全依存関係が解決 |
| モジュールインポート | ✅ 成功 |
| CPU モデルロード | ✅ 5.79秒 |
| 英語音声認識 | ✅ 17.69秒 |
| 中国語音声認識 | ✅ 4.52秒 |

### 2.2 依存関係

#### 必須依存

> **Note**: 以下のバージョンは qwen-asr 0.0.6 時点で観測された現在のピンです。
> 将来のバージョンで変更される可能性があります。

```
qwen-asr
├── transformers==4.57.6
├── accelerate==1.12.0
├── nagisa==0.2.11 (日本語トークナイザー)
├── soynlp==0.0.493 (韓国語NLP)
├── qwen-omni-utils
│   ├── av
│   ├── librosa ⚠️
│   └── pillow
├── librosa ⚠️
├── soundfile
├── gradio (不要)
└── flask (不要)
```

#### バージョン互換性

| パッケージ | qwen-asr | livecap-cli (現在) | 互換性 |
|-----------|----------|-------------------|--------|
| transformers | ==4.57.6 | >=4.57.0 | ✅ |
| librosa | - | - | ⚠️ PyInstaller問題 (#219) |

### 2.3 リスク評価

| リスク | 深刻度 | 対策 |
|--------|--------|------|
| PyInstaller 循環インポート (librosa) | 中 | #219 の対策を適用 |
| 不要な依存関係 (gradio, flask) | 低 | インストール時の警告のみ |
| GPU メモリ消費 | 低 | 0.6B モデルで対応可能 |

---

## 3. 設計

### 3.1 エンジン構成

```
livecap_cli/engines/
├── qwen3asr_engine.py      # 新規作成
├── qwen3asr_utils.py       # 新規作成（オプション）
└── metadata.py             # エンジンメタデータ追加
```

### 3.2 エンジンクラス設計

```python
# qwen3asr_engine.py

from .base_engine import BaseEngine

class Qwen3ASREngine(BaseEngine):
    """Qwen3-ASR 音声認識エンジン"""

    def __init__(
        self,
        device: Optional[str] = None,
        language: Optional[str] = None,  # None = 自動検出
        model_name: str = "Qwen/Qwen3-ASR-0.6B",
        use_forced_aligner: bool = False,
        engine_id: str = "qwen3asr",  # メタデータから渡される識別子
        **kwargs,
    ):
        # engine_id を使用して qwen3asr / qwen3asr_large を区別
        self.engine_name = engine_id
        self.language = language
        self.model_name = model_name
        self.use_forced_aligner = use_forced_aligner
        super().__init__(device, **kwargs)

    # Template Method 実装
    def get_model_metadata(self) -> Dict[str, Any]: ...
    def _check_dependencies(self) -> None: ...
    def _prepare_model_directory(self) -> Path: ...
    def _download_model(self, model_path: Path, progress_callback, model_manager=None) -> None: ...
    def _load_model_from_path(self, model_path: Path) -> Any: ...
    def _configure_model(self) -> None: ...

    # TranscriptionEngine プロトコル実装
    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[str, float]: ...
    def get_engine_name(self) -> str: ...
    def get_required_sample_rate(self) -> int: ...
    def cleanup(self) -> None: ...
```

### 3.3 メタデータ定義

```python
# metadata.py に追加

"qwen3asr": EngineInfo(
    id="qwen3asr",
    display_name="Qwen3-ASR 0.6B",
    description="High-accuracy multilingual ASR supporting 30+ languages",
    supported_languages=[
        "zh", "en", "yue", "ar", "de", "fr", "es", "pt", "id", "it",
        "ko", "ru", "th", "vi", "ja", "tr", "hi", "ms", "nl", "sv",
        "da", "fi", "pl", "cs", "fil", "fa", "el", "hu", "mk", "ro"
    ],
    requires_download=True,
    model_size="1.2GB",
    device_support=["cpu", "cuda"],
    streaming=False,  # 初期実装ではオフラインのみ
    module=".qwen3asr_engine",
    class_name="Qwen3ASREngine",
    default_params={
        "model_name": "Qwen/Qwen3-ASR-0.6B",
        "use_forced_aligner": False,
        "engine_id": "qwen3asr",  # エンジン識別子をクラスに渡す
    }
),

"qwen3asr_large": EngineInfo(
    id="qwen3asr_large",
    display_name="Qwen3-ASR 1.7B",
    description="State-of-the-art multilingual ASR with best accuracy",
    supported_languages=[...],  # 同上
    requires_download=True,
    model_size="3.4GB",
    device_support=["cuda"],  # 1.7B は GPU 推奨
    streaming=False,
    module=".qwen3asr_engine",
    class_name="Qwen3ASREngine",
    default_params={
        "model_name": "Qwen/Qwen3-ASR-1.7B",
        "use_forced_aligner": False,
        "engine_id": "qwen3asr_large",  # エンジン識別子をクラスに渡す
    }
),
```

### 3.4 依存関係管理

```toml
# pyproject.toml に追加

[project.optional-dependencies]
"engines-qwen3asr" = [
    "qwen-asr>=0.0.6",
    "torch",
]
```

---

## 4. 実装フェーズ

### Phase 1: 基本実装 (MVP)

**目標**: 最小限の機能で動作するエンジンを実装

#### 4.1.1 タスク

- [ ] `qwen3asr_engine.py` の作成
  - [ ] `Qwen3ASREngine` クラスの実装
  - [ ] `BaseEngine` の Template Method 実装
  - [ ] `transcribe()` メソッドの実装
- [ ] `metadata.py` にエンジン情報を追加
- [ ] `pyproject.toml` に `engines-qwen3asr` extra を追加
- [ ] 基本的なユニットテストの作成

#### 4.1.2 スコープ

| 含む | 含まない |
|------|---------|
| オフライン推論 | ストリーミング推論 |
| 言語自動検出 | ForcedAligner |
| CPU/GPU サポート | vLLM バックエンド |
| 0.6B モデル | 1.7B モデル（後で追加） |

### Phase 2: 機能拡張

**目標**: 追加機能の実装

#### 4.2.1 タスク

- [ ] 1.7B モデルサポート (`qwen3asr_large`)
- [ ] ForcedAligner 統合（タイムスタンプ出力）
- [ ] PyInstaller 互換性対策
- [ ] 統合テストの作成

### Phase 3: 最適化（将来）

**目標**: パフォーマンスと UX の改善

#### 4.3.1 タスク

- [ ] ストリーミング推論の実装（vLLM バックエンド）
- [ ] モデルキャッシュの最適化
- [ ] 既存エンジンとの性能比較ベンチマーク

---

## 5. テスト計画

### 5.1 ユニットテスト

```python
# tests/core/engines/test_qwen3asr_engine.py

class TestQwen3ASREngine:
    def test_engine_creation(self): ...
    def test_check_dependencies(self): ...
    def test_transcribe_english(self): ...
    def test_transcribe_chinese(self): ...
    def test_transcribe_japanese(self): ...
    def test_language_auto_detect(self): ...
```

### 5.2 統合テスト

```python
# tests/integration/engines/test_qwen3asr_integration.py

class TestQwen3ASRIntegration:
    def test_file_transcription(self): ...
    def test_stream_transcriber_integration(self): ...
```

### 5.3 pytest マーカー

```python
@pytest.mark.engine_smoke  # エンジン基本動作テスト
@pytest.mark.gpu           # GPU 必須テスト用
@pytest.mark.network       # ネットワークアクセス必須（モデルダウンロード）
@pytest.mark.slow          # 長時間テスト（モデルロード、推論）
```

### 5.4 テスト実行例

```bash
# 基本テスト（ネットワーク・GPU不要のテストのみ）
pytest tests/core/engines/test_qwen3asr_engine.py -m "not network and not gpu"

# フルテスト（モデルダウンロード含む）
pytest tests/core/engines/test_qwen3asr_engine.py -m "engine_smoke"

# GPU テスト
pytest tests/core/engines/test_qwen3asr_engine.py -m "gpu"
```

---

## 6. ドキュメント更新

### 6.1 更新対象

- [ ] `README.md` - エンジン一覧に Qwen3-ASR を追加
- [ ] `docs/reference/engines.md` - エンジン詳細ドキュメント
- [ ] `CLAUDE.md` - 開発ガイドの更新（必要に応じて）

### 6.2 CLI ヘルプ

```bash
$ livecap-cli engines
Available engines:
  ...
  qwen3asr        Qwen3-ASR 0.6B - High-accuracy multilingual ASR (30+ languages)
  qwen3asr_large  Qwen3-ASR 1.7B - State-of-the-art multilingual ASR
```

---

## 7. リスクと緩和策

### 7.1 技術的リスク

| リスク | 確率 | 影響 | 緩和策 |
|--------|------|------|--------|
| PyInstaller 循環インポート | 中 | 中 | #219 の対策パターンを適用 |
| transformers バージョン衝突 | 低 | 高 | CI で互換性テスト |
| GPU メモリ不足 | 低 | 中 | 0.6B モデルをデフォルトに |

### 7.2 スケジュールリスク

| リスク | 確率 | 影響 | 緩和策 |
|--------|------|------|--------|
| qwen-asr API 変更 | 低 | 中 | バージョン固定、CI 監視 |
| 依存関係の非互換 | 低 | 高 | 独立した extra で分離 |

---

## 8. 成功基準

### 8.1 MVP (Phase 1)

- [ ] `livecap-cli transcribe -e qwen3asr audio.wav` が動作する
- [ ] 英語・中国語・日本語の音声認識が正常に動作する
- [ ] 言語自動検出が機能する
- [ ] CPU / GPU 両方で動作する

### 8.2 完成 (Phase 2)

- [ ] PyInstaller ビルドで動作する
- [ ] ForcedAligner によるタイムスタンプ出力が動作する
- [ ] 統合テストがすべてパスする

---

## 9. 参考リンク

- [Qwen3-ASR GitHub](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-ASR-0.6B Hugging Face](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Qwen3-ASR-1.7B Hugging Face](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [qwen-asr PyPI](https://pypi.org/project/qwen-asr/)
- [Issue #221](https://github.com/Mega-Gorilla/livecap-cli/issues/221)
