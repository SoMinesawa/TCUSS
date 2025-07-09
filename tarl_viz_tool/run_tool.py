#!/usr/bin/env python3
"""
TARL Clustering Visualization Tool Launcher

このスクリプトはTARLビジュアライゼーションツールを実行するためのランチャーです。
"""

import sys
import os
from pathlib import Path

# TCUSSのルートディレクトリをパスに追加
tcuss_root = Path(__file__).parent.parent
sys.path.insert(0, str(tcuss_root))

# 依存関係の確認
def check_dependencies():
    """依存関係の確認"""
    try:
        import numpy
        import open3d
        import MinkowskiEngine
        import sklearn
        import hdbscan
        import matplotlib
        print("✓ すべての依存関係が確認されました")
        return True
    except ImportError as e:
        print(f"✗ 依存関係のエラー: {e}")
        print("必要なパッケージをインストールしてください:")
        print("pip install numpy open3d scikit-learn hdbscan matplotlib")
        print("pip install MinkowskiEngine")
        return False

if __name__ == "__main__":
    print("TARL Clustering Visualization Tool を起動しています...")
    
    # 依存関係の確認
    if not check_dependencies():
        sys.exit(1)
    
    # メインプログラムの実行
    try:
        from tarl_viz_tool.main import main
        main()
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1) 