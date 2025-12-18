# Issue #166: detect_device() の責務分離とシグネチャ改善

> **Status**: READY FOR IMPLEMENTATION
> **作成日:** 2025-12-05
> **関連 Issue:** #166
> **親 Issue:** #64 (Epic: livecap-cli リファクタリング)

---

## 1. 目的

`detect_device()` 関数のシグネチャを改善し、責務を明確化する。

**変更内容**: 戻り値を `Tuple[str, str]` から `str` に変更

---

## 2. 背景

### 2.1 現在の実装

```python
# livecap_cli/utils/__init__.py:38-64
def detect_device(requested_device: Optional[str], engine_name: str) -> Tuple[str, str]:
    """Returns: (device, compute_type)"""
    if requested_device not in ("cpu",):
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda", "float16"
        except ImportError:
            pass
    return "cpu", "float32"
```

### 2.2 問題点

| 問題 | 詳細 |
|------|------|
| **関心の分離ができていない** | デバイス検出と量子化タイプ選択が混在 |
| **戻り値の設計が不適切** | `compute_type` は WhisperS2T (CTranslate2) 固有の概念 |
| **全エンジンで2番目の値が未使用** | 調査により確認済み |

### 2.3 現在の使用状況（調査結果）

| エンジン | 使用方法 | 2番目の値 |
|----------|---------|----------|
| WhisperS2T | `device_result[0]` | **無視** (独自の `_resolve_compute_type()`) |
| Parakeet | `self.torch_device, _ = ...` | **破棄** |
| Canary | `self.torch_device, _ = ...` | **破棄** |
| Voxtral | `self.torch_device, self.device_str = ...` | **保存されるが未使用** |
| ReazonSpeech | 使用しない | N/A |

### 2.4 WhisperS2T の compute_type 処理

WhisperS2T は既に `compute_type` をパラメータ経由で取得している：

```python
# whispers2t_engine.py:141-146
def _resolve_compute_type(self, compute_type: str) -> str:
    """compute_typeを解決（autoの場合はデバイスに応じて最適化）"""
    if compute_type != "auto":
        return compute_type  # ユーザー指定を尊重
    # auto: CPU→int8（1.5倍高速）、GPU→float16
    return "int8" if self.device == "cpu" else "float16"
```

**結論**: `detect_device()` から `compute_type` を返す必要はない。

---

## 3. 設計

### 3.1 新しいシグネチャ

```python
def detect_device(requested_device: Optional[str], engine_name: str) -> str:
    """
    デバイスを検出して返す。

    Args:
        requested_device: 要求されたデバイス ("cuda", "cpu", None=auto)
        engine_name: エンジン名（ログ用）

    Returns:
        使用するデバイス ("cuda" または "cpu")
    """
```

### 3.2 実装

```python
def detect_device(requested_device: Optional[str], engine_name: str) -> str:
    """
    デバイスを検出して返す。

    Args:
        requested_device: 要求されたデバイス ("cuda", "cpu", None=auto)
        engine_name: エンジン名（ログ用）

    Returns:
        使用するデバイス ("cuda" または "cpu")
    """
    logger = logging.getLogger(__name__)

    if requested_device == "cpu":
        logger.info("Using CPU for %s (explicitly requested).", engine_name)
        return "cpu"

    try:
        import torch
        version = torch.__version__
        if torch.cuda.is_available():
            logger.info("Using CUDA for %s (PyTorch %s).", engine_name, version)
            return "cuda"

        if "+cpu" in version:
            logger.warning("PyTorch CPU build detected (%s); falling back to CPU for %s.", version, engine_name)
        else:
            logger.warning("CUDA unavailable (PyTorch %s); falling back to CPU for %s.", version, engine_name)
    except ImportError:
        logger.warning("PyTorch not installed; using CPU for %s.", engine_name)

    return "cpu"
```

### 3.3 呼び出し側の変更

#### WhisperS2T

```python
# Before (whispers2t_engine.py:109-112)
device_result = detect_device(device, "WhisperS2T")
self.device = device_result[0] if isinstance(device_result, tuple) else device_result

# After
self.device = detect_device(device, "WhisperS2T")
```

#### Parakeet

```python
# Before (parakeet_engine.py:122)
self.torch_device, _ = detect_device(device, "Parakeet")

# After
self.torch_device = detect_device(device, "Parakeet")
```

#### Canary

```python
# Before (canary_engine.py:93)
self.torch_device, _ = detect_device(device, "Canary")

# After
self.torch_device = detect_device(device, "Canary")
```

#### Voxtral

```python
# Before (voxtral_engine.py:93)
self.torch_device, self.device_str = detect_device(device, "Voxtral")

# After
self.torch_device = detect_device(device, "Voxtral")
# device_str は未使用のため削除
```

---

## 4. 修正対象ファイル

| ファイル | 変更内容 |
|---------|----------|
| `livecap_cli/utils/__init__.py` | `detect_device()` の戻り値を `str` に変更 |
| `livecap_cli/engines/whispers2t_engine.py` | タプル処理を削除、直接代入に変更 |
| `livecap_cli/engines/parakeet_engine.py` | タプルアンパックを削除 |
| `livecap_cli/engines/canary_engine.py` | タプルアンパックを削除 |
| `livecap_cli/engines/voxtral_engine.py` | タプルアンパック削除、未使用変数 `device_str` 削除 |

---

## 5. 実装順序

```
Phase 1: detect_device() の修正
  1. livecap_cli/utils/__init__.py の戻り値を str に変更
  2. 型ヒントを Tuple[str, str] から str に変更
  3. docstring を更新

Phase 2: エンジン呼び出し側の更新
  4. whispers2t_engine.py を更新
  5. parakeet_engine.py を更新
  6. canary_engine.py を更新
  7. voxtral_engine.py を更新（device_str 削除含む）

Phase 3: テスト
  8. テスト実行・確認
```

---

## 6. テスト計画

### 6.1 既存テスト

`detect_device()` の直接テストは存在しない。各エンジンのテストで間接的に検証される。

### 6.2 確認項目

- [ ] `pytest tests/core/engines/test_engine_factory.py` がパス
- [ ] 各エンジンが正しくデバイスを取得できる
- [ ] CPU/CUDA の自動検出が正常に動作

---

## 7. 完了条件

- [ ] `detect_device()` が `str` を返す
- [ ] 型ヒントが `str` に更新されている
- [ ] WhisperS2T がタプル処理なしでデバイスを取得している
- [ ] Parakeet/Canary/Voxtral がタプルアンパックなしでデバイスを取得している
- [ ] Voxtral から未使用の `device_str` 変数が削除されている
- [ ] 全テストがパス

---

## 8. リスク評価

### 8.1 破壊的変更

**影響範囲**: `detect_device()` を直接使用している外部コードがある場合は破壊的変更。

**緩和策**:
- livecap-core 内部でのみ使用されているため、影響は限定的
- `__all__` にエクスポートされているが、主に内部使用

### 8.2 後方互換性

必要であれば、一時的に両方の戻り値形式をサポートする互換レイヤーを追加可能：

```python
# 非推奨（必要な場合のみ）
def detect_device_legacy(...) -> Tuple[str, str]:
    device = detect_device(...)
    compute_type = "float16" if device == "cuda" else "float32"
    return device, compute_type
```

**推奨**: 破壊的変更を許容し、シンプルに実装する。

---

## 9. 関連情報

### 9.1 参考リンク

- Issue #166: https://github.com/Mega-Gorilla/livecap-core/issues/166
- 親 Issue #64: Epic リファクタリング
- [CTranslate2 Quantization](https://opennmt.net/CTranslate2/quantization.html)

### 9.2 調査で判明した事実

1. **全エンジンが `compute_type` を使用していない** - 調査により確認
2. **WhisperS2T は既に正しく実装されている** - `_resolve_compute_type()` でパラメータから解決
3. **Voxtral の `device_str` は完全に未使用** - grep で確認済み
