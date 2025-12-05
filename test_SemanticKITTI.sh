#!/bin/bash
# TCUSS: テストと提出用スクリプト
# 使用方法: ./test_SemanticKITTI.sh <model_dir> <submission_name>

set -e  # エラーがあった時点で処理を停止

# デフォルト値を設定
DATA_PATH="data/users/minesawa/semantickitti/growsp"
SP_PATH="data/users/minesawa/semantickitti/growsp_sp"
DATASET_PATH="data/dataset/semantickitti/dataset/"
DEBUG=""
USE_BEST=""

# 使い方の表示関数
function show_usage {
    echo "使用方法: $0 <model_dir> <submission_name> [options]"
    echo ""
    echo "必須引数:"
    echo "  <model_dir>       - モデルのチェックポイントを含むディレクトリパス"
    echo "  <submission_name> - 提出用zipファイルの名前 (例: submission.zip)"
    echo ""
    echo "オプション:"
    echo "  --data_path PATH  - 点群データパス (デフォルト: $DATA_PATH)"
    echo "  --sp_path PATH    - 初期スーパーポイントパス (デフォルト: $SP_PATH)"
    echo "  --dataset PATH    - SemanticKITTIデータセットパス (デフォルト: $DATASET_PATH)"
    echo "  --debug           - デバッグモード (最新のチェックポイントのみ評価)"
    echo "  --use_best        - bestモデル（best_model.pth, best_classifier.pth）を使用"
    echo ""
    echo "例:"
    echo "  $0 data/users/minesawa/semantickitti/growsp_model_original growsp.zip --debug"
    echo "  $0 data/users/minesawa/semantickitti/growsp_model_original growsp_best.zip --use_best"
    exit 1
}

# 引数が足りない場合
if [ "$#" -lt 2 ]; then
    show_usage
fi

# 必須引数を取得
MODEL_DIR="$1"
SUBMISSION_NAME="$2"
shift 2

# オプション引数の処理
while [ "$#" -gt 0 ]; do
    case "$1" in
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --sp_path)
            SP_PATH="$2"
            shift 2
            ;;
        --dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        --use_best)
            USE_BEST="--use_best"
            shift
            ;;
        *)
            echo "エラー: 不明なオプション $1"
            show_usage
            ;;
    esac
done

# ディレクトリ確認
if [ ! -d "$MODEL_DIR" ]; then
    echo "エラー: モデルディレクトリ $MODEL_DIR が存在しません"
    exit 1
fi

echo "=========== TCUSS テスト処理開始 ==========="
echo "モデルディレクトリ: $MODEL_DIR"
echo "提出ファイル名: $SUBMISSION_NAME"
echo "データパス: $DATA_PATH"
echo "SPパス: $SP_PATH"
echo "データセットパス: $DATASET_PATH"
if [ -n "$DEBUG" ]; then
    echo "モード: デバッグ (最新のチェックポイントのみ)"
elif [ -n "$USE_BEST" ]; then
    echo "モード: bestモデル使用 (best_model.pth, best_classifier.pth)"
else
    echo "モード: 通常 (全チェックポイント評価)"
fi
echo "==========================================="

# 各コマンドを順番に実行
echo "1. テスト実行: テストデータでの予測実行"
python test_SemanticKITTI.py \
    --data_path "$DATA_PATH" \
    --sp_path "$SP_PATH" \
    --save_path "$MODEL_DIR" \
    $DEBUG \
    $USE_BEST || { echo "テスト実行に失敗しました"; exit 1; }

echo "2. 評価: 公式評価スクリプトで精度計測"
CURRENT_DIR=$(pwd)
cd ~/repos/semantic-kitti-api || { echo "semantic-kitti-apiディレクトリへの移動に失敗しました"; exit 1; }
python evaluate_semantics.py -d "$CURRENT_DIR/$DATASET_PATH" -p "$CURRENT_DIR/$MODEL_DIR/pred_result/" || { echo "評価に失敗しました"; exit 1; }
cd "$CURRENT_DIR" || { echo "元のディレクトリへの移動に失敗しました"; exit 1; }

echo "3. 提出ファイル作成: ZIPアーカイブを作成"
RESULT_DIR="$MODEL_DIR/pred_result"
cd "$RESULT_DIR" || { echo "結果ディレクトリへの移動に失敗しました"; exit 1; }
zip -r -q "$SUBMISSION_NAME" . || { echo "ZIP作成に失敗しました"; exit 1; }

echo "4. 提出ファイル検証: フォーマット検証"
cd "$CURRENT_DIR" || { echo "元のディレクトリへの移動に失敗しました"; exit 1; }
python ~/repos/semantic-kitti-api/validate_submission.py "$RESULT_DIR/$SUBMISSION_NAME" "$CURRENT_DIR/$DATASET_PATH" || { echo "提出ファイル検証に失敗しました"; exit 1; }

# 転送処理（必要に応じて）
if [ -f "$RESULT_DIR/$SUBMISSION_NAME" ]; then
    echo "5. ファイル転送: ローカルPCに転送"
    scp -P 2222 -i ~/.ssh/id_ishikawa "$RESULT_DIR/$SUBMISSION_NAME" somin@localhost:"C:\\Users\\somin\\Downloads" || { echo "ファイル転送に失敗しました"; exit 1; }
    echo "ファイル転送完了: C:\\Users\\somin\\Downloads\\$SUBMISSION_NAME"
else
    echo "エラー: 提出ファイルが作成されませんでした"
    exit 1
fi

echo "=========== 処理完了 ==========="
echo "提出ファイル: $RESULT_DIR/$SUBMISSION_NAME"
echo "サイズ: $(du -h "$RESULT_DIR/$SUBMISSION_NAME" | cut -f1) bytes"