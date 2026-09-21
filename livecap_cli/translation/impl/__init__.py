"""
翻訳エンジン実装

各翻訳エンジンの実装を格納するサブパッケージ。

- google.py: Google Translate
- opus_mt.py: OPUS-MT via CTranslate2 (``translation-local`` extra)
- riva_instruct.py: Riva-Translate-4B-Instruct (``translation-riva`` extra)

**ここで実装 module を import しない** (Issue #454)。``TranslatorFactory`` は
``TranslatorInfo.module`` を ``importlib`` で遅延 import する設計で、この ``__init__`` が
``opus_mt`` / ``riva_instruct`` を eager import すると、Google しか使わない場合でも
torch / transformers / ctranslate2 (数百 MB、数秒) が読み込まれる。さらに別スレッドで
``scipy.signal.resample_poly`` が走っていると、scipy が ``sys.modules["torch"]`` の
初期化途中の module に当たって ``AttributeError`` で落ちる (livecap-gui の実機ログ)。
実装クラスは ``TranslatorFactory.create_translator(<id>)`` か、各 module を直接 import して使う。
"""

from __future__ import annotations

__all__: list[str] = []
