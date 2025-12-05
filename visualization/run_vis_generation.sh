#!/bin/bash
# TCUSS可視化用データ生成実行スクリプト

# 使用方法の表示
show_usage() {
    echo "使用方法:"
    echo "  $0 [オプション]"
    echo ""
    echo "オプション:"
    echo "  --sequences SEQ1 SEQ2 ...    特定のシーケンスのみ処理（例: --sequences 00 01 02）"
    echo "  --data_path PATH             点群データのパス（デフォルト: data/users/minesawa/semantickitti/growsp）"
    echo "  --original_data_path PATH    オリジナルデータのパス（デフォルト: data/dataset/semantickitti/dataset/sequences）"
    echo "  --patchwork_path PATH        パッチワークデータのパス（デフォルト: data/users/minesawa/semantickitti/patchwork）"
    echo "  --voxel_size SIZE            ボクセルサイズ（デフォルト: 0.15）"
    echo "  --r_crop RADIUS              クロッピング半径（デフォルト: 50.0）"
    echo "  --scan_window SIZE           スキャンウィンドウサイズ（デフォルト: 12）"
    echo "  --debug                      デバッグモード（最初の3ウィンドウのみ処理）"
    echo "  --help                       このヘルプメッセージを表示"
    echo ""
    echo "例:"
    echo "  $0                              # すべてのシーケンスを処理"
    echo "  $0 --sequences 00 01            # シーケンス00と01のみ処理"
    echo "  $0 --sequences 02 --voxel_size 0.2  # シーケンス02をボクセルサイズ0.2で処理"
    echo "  $0 --debug --sequences 00       # シーケンス00の最初の3ウィンドウのみをデバッグモードで処理"
}

# デフォルト値
DATA_PATH="data/users/minesawa/semantickitti/growsp"
ORIGINAL_DATA_PATH="data/dataset/semantickitti/dataset/sequences"
PATCHWORK_PATH="data/users/minesawa/semantickitti/patchwork"
VOXEL_SIZE="0.15"
R_CROP="50.0"
SCAN_WINDOW="12"
SEQUENCES=""
DEBUG_MODE=""

# 引数の解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --sequences)
            shift
            SEQUENCES="--sequences"
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                SEQUENCES="$SEQUENCES $1"
                shift
            done
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --original_data_path)
            ORIGINAL_DATA_PATH="$2"
            shift 2
            ;;
        --patchwork_path)
            PATCHWORK_PATH="$2"
            shift 2
            ;;
        --voxel_size)
            VOXEL_SIZE="$2"
            shift 2
            ;;
        --r_crop)
            R_CROP="$2"
            shift 2
            ;;
        --scan_window)
            SCAN_WINDOW="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE="--debug"
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "未知のオプション: $1"
            show_usage
            exit 1
            ;;
    esac
done

# スクリプトのディレクトリを取得
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Anaconda環境をアクティベート
echo "Anaconda環境 'tcuss-cuml' をアクティベート中..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tcuss-cuml

# 現在のディレクトリをプロジェクトルートに変更
cd "$PROJECT_ROOT"

# パラメータの表示
echo "==================== 可視化データ生成 開始 ===================="
echo "設定パラメータ:"
echo "  データパス: $DATA_PATH"
echo "  オリジナルデータパス: $ORIGINAL_DATA_PATH"
echo "  パッチワークパス: $PATCHWORK_PATH"
echo "  ボクセルサイズ: $VOXEL_SIZE"
echo "  クロッピング半径: $R_CROP"
echo "  スキャンウィンドウサイズ: $SCAN_WINDOW"
if [ -n "$SEQUENCES" ]; then
    echo "  処理対象シーケンス: $SEQUENCES"
else
    echo "  処理対象シーケンス: すべて"
fi
if [ -n "$DEBUG_MODE" ]; then
    echo "  デバッグモード: 有効"
else
    echo "  デバッグモード: 無効"
fi
echo "============================================================"

# Python スクリプトの実行
python visualization/generate_vis_data.py \
    --data_path "$DATA_PATH" \
    --original_data_path "$ORIGINAL_DATA_PATH" \
    --patchwork_path "$PATCHWORK_PATH" \
    --voxel_size "$VOXEL_SIZE" \
    --r_crop "$R_CROP" \
    --scan_window "$SCAN_WINDOW" \
    $SEQUENCES \
    $DEBUG_MODE

# 実行結果の確認
if [ $? -eq 0 ]; then
    echo ""
    echo "==================== 生成完了 ===================="
    echo "可視化用データが以下のディレクトリに保存されました:"
    echo "  data/users/minesawa/semantickitti/vis/sequences/"
    echo ""
    echo "各シーケンスのディレクトリ構造:"
    echo "  sequences/{seq_id}/"
    echo "    ├── velodyne/{frame_id}.ply           # 座標データ"
    echo "    ├── labels/{frame_id}.ply             # ラベルデータ"
    echo "    ├── agg_coordinates/{start}-{end}.ply # 集約座標データ"
    echo "    └── agg_segments/{start}-{end}.ply    # 集約セグメント付きデータ"
    echo "=================================================="
else
    echo ""
    echo "==================== エラー ===================="
    echo "可視化データの生成中にエラーが発生しました。"
    echo "上記のエラーメッセージを確認してください。"
    echo "=============================================="
    exit 1
fi 