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

# 引数の処理（位置引数とオプション引数を分離）
DATA_PATH="$DEFAULT_DATA_PATH"
SEQ="$DEFAULT_SEQ"
EXTRA_ARGS=""

# 引数を解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --*)
            # オプション引数（--で始まる）はEXTRA_ARGSに追加
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
        *)
            # 位置引数の処理
            if [[ -z "$DATA_PATH_SET" ]]; then
                DATA_PATH="$1"
                DATA_PATH_SET=true
            elif [[ -z "$SEQ_SET" ]]; then
                SEQ="$1"
                SEQ_SET=true
            else
                # 追加の位置引数もEXTRA_ARGSに追加
                EXTRA_ARGS="$EXTRA_ARGS $1"
            fi
            shift
            ;;
    esac
done

echo "データパス: $DATA_PATH"
echo "シーケンス: $SEQ"
if [[ -n "$EXTRA_ARGS" ]]; then
    echo "追加オプション:$EXTRA_ARGS"
fi

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

# === Conda環境の有効化 ===
echo "tcuss-cuml環境を有効化中..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tcuss-cuml

# === 完全なOpenGL ソフトウェアレンダリング設定 ===
# ライブラリパスの設定
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# 完全にソフトウェアレンダリングを強制
export LIBGL_ALWAYS_SOFTWARE=1
export LIBGL_ALWAYS_INDIRECT=1
export GALLIUM_DRIVER=llvmpipe
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe

# OpenGLバージョンのオーバーライド（vispy対応）
export MESA_GL_VERSION_OVERRIDE=3.3
export MESA_GLSL_VERSION_OVERRIDE=330
export GL_VERSION=3.3

# GLXを強制的に有効化
export LIBGL_ALWAYS_GL=1

# === Qt設定（OpenGLコンテキスト問題解決） ===
export QT_OPENGL=software
export QT_XCB_GL_INTEGRATION=software
export QT_OPENGL_IMPL=software
export QT_X11_NO_MITSHM=1
export QT_GRAPHICSSYSTEM=native

# === vispy固有の設定 ===
export VISPY_GL_LIB="/home/minesawa/anaconda3/envs/tcuss-cuml/x86_64-conda_cos6-linux-gnu/sysroot/usr/lib64/libGL.so.1.2.0"
export VISPY_GL_API=gl+
export VISPY_BACKEND=PyQt5

# === マウスイベントエラー回避設定 ===
export VISPY_INTERACTIVE=False

# === EGLの設定 ===
export EGL_PLATFORM=surfaceless

# DISPLAYの設定
export DISPLAY="${DISPLAY:-:0}"

echo "完全なソフトウェアレンダリング設定を有効にしました"

# メインプログラムの実行
python -m tarl_viz_tool.main \
    --data_path "$DATA_PATH" \
    --seq "$SEQ" \
    --use_vispy_simple \
    $EXTRA_ARGS

echo "ツールが終了しました" 