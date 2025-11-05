"""
NeMo JIT compilation patch for PyInstaller compatibility
"""
import sys
import os

def patch_nemo_jit():
    """PyInstaller環境でNeMoのJITコンパイルを無効化"""
    # PyInstallerで実行されているかチェック
    if not (getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')):
        return
        
    # JITコンパイルを無効化
    os.environ['PYTORCH_JIT'] = '0'
    
    # torch.jit.scriptをパッチして、単に元の関数を返すようにする
    try:
        import torch.jit
        
        # すでにパッチ済みかチェック
        if hasattr(torch.jit.script, '_pyinstaller_patched'):
            return
            
        original_script = torch.jit.script
        
        def patched_script(obj, *args, **kwargs):
            """JITコンパイルをスキップして元のオブジェクトを返す"""
            # 関数の場合はそのまま返す
            if callable(obj):
                return obj
            # それ以外の場合は通常のscriptを試みる
            try:
                return original_script(obj, *args, **kwargs)
            except:
                # エラーが発生した場合は元のオブジェクトを返す
                return obj
        
        # パッチ済みマーカーを設定
        patched_script._pyinstaller_patched = True
        torch.jit.script = patched_script
        print("[INFO] NeMo JIT compilation patched for PyInstaller")
    except ImportError:
        pass

# 自動的にパッチを適用
patch_nemo_jit()