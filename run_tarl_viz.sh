#!/bin/bash

# TARL Clustering Visualization Tool
# 実行スクリプト

echo "================================================"
echo "  TARL Clustering Visualization Tool"
echo "  TARLクラスタリング ハイパーパラメータ最適化ツール"
echo "================================================"

# 仮想環境の確認
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "仮想環境: $VIRTUAL_ENV"
else
    echo "警告: 仮想環境が検出されませんでした"
fi

# Pythonの確認
PYTHON_VERSION=$(python --version 2>&1)
echo "Python: $PYTHON_VERSION"

# 実行ディレクトリの確認
if [[ ! -d "tarl_viz_tool" ]]; then
    echo "エラー: tarl_viz_tool ディレクトリが見つかりません"
    echo "TCUSSのルートディレクトリから実行してください"
    exit 1
fi

# デフォルトパラメータの設定
DEFAULT_DATA_PATH="data/dataset/semantickitti/dataset/sequences"
DEFAULT_SEQ="00"

# 引数の処理
DATA_PATH=${1:-$DEFAULT_DATA_PATH}
SEQ=${2:-$DEFAULT_SEQ}

echo "データパス: $DATA_PATH"
echo "シーケンス: $SEQ"

# データパスの確認
if [[ ! -d "$DATA_PATH" ]]; then
    echo "エラー: データパスが見つかりません: $DATA_PATH"
    echo "正しいパスを指定してください"
    exit 1
fi

# シーケンスディレクトリの確認
if [[ ! -d "$DATA_PATH/$SEQ" ]]; then
    echo "エラー: シーケンスディレクトリが見つかりません: $DATA_PATH/$SEQ"
    echo "利用可能なシーケンス:"
    ls -1 "$DATA_PATH" | grep -E '^[0-9]{2}$' | head -10
    exit 1
fi

echo "================================================"
echo "ツールを起動しています..."

# メインプログラムの実行
python -m tarl_viz_tool.main \
    --data_path "$DATA_PATH" \
    --seq "$SEQ" \
    "$@"

echo "ツールが終了しました" 