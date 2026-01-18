#!/bin/bash
# TCUSS: テストと提出用スクリプト
# 使用方法: ./test_SemanticKITTI.sh <model_dir> <submission_name>

set -e  # エラーがあった時点で処理を停止

# デフォルト値を設定
DATA_PATH="data/users/minesawa/semantickitti/growsp"
DATASET_PATH="data/dataset/semantickitti/dataset/"
CONFIG_PATH="config/stc.yaml"
DEBUG=""
USE_BEST=""
EPOCH=""
SKIP_INFERENCE=""
SEARCH_SEED=""
TARGET_MIOU="16.5"
TARGET_OACC="45.0"
SEED_START="0"
SEED_MODE="sequential"
MAX_TRIALS="0"
DEVICE_ID="0"

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
    echo "  --dataset PATH    - SemanticKITTIデータセットパス (デフォルト: $DATASET_PATH)"
    echo "  --debug           - デバッグモード (最新のチェックポイントのみ評価)"
    echo "  --use_best        - bestモデル（best_model.pth, best_classifier.pth）を使用"
    echo "  --epoch N         - 指定したepochのみ評価 (model_N_checkpoint.pth, cls_N_checkpoint.pth)"
    echo "  --skip_inference  - 推論（pred_result生成）をスキップして、既存pred_resultを評価/ZIP/検証する"
    echo "  --search_seed     - val(08)で mIoU/oAcc が閾値を超えるまで KMeans seed を探索して固定（そのseedで推論）"
    echo "  --config PATH     - seed探索で使用するTCUSS設定yaml (デフォルト: $CONFIG_PATH)"
    echo "  --target_miou X   - seed探索のmIoU閾値 (デフォルト: $TARGET_MIOU)"
    echo "  --target_oacc X   - seed探索のoAcc閾値 (デフォルト: $TARGET_OACC)"
    echo "  --seed_start N    - seed探索の開始seed (sequential時) / RNG seed (random時) (デフォルト: $SEED_START)"
    echo "  --seed_mode MODE  - seed探索のモード: sequential|random (デフォルト: $SEED_MODE)"
    echo "  --max_trials N    - seed探索の最大試行回数（0で無限） (デフォルト: $MAX_TRIALS)"
    echo "  --device_id N     - seed探索時に使用するGPU ID (デフォルト: $DEVICE_ID)"
    echo ""
    echo "例:"
    echo "  $0 data/users/minesawa/semantickitti/growsp_model_original growsp.zip --debug"
    echo "  $0 data/users/minesawa/semantickitti/growsp_model_original growsp_best.zip --use_best"
    echo "  $0 data/users/minesawa/semantickitti/growsp_model_original growsp_epoch80.zip --epoch 80"
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
        --dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        --config)
            CONFIG_PATH="$2"
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
        --epoch)
            EPOCH="$2"
            shift 2
            ;;
        --skip_inference)
            SKIP_INFERENCE="1"
            shift
            ;;
        --search_seed)
            SEARCH_SEED="1"
            shift
            ;;
        --target_miou)
            TARGET_MIOU="$2"
            shift 2
            ;;
        --target_oacc)
            TARGET_OACC="$2"
            shift 2
            ;;
        --seed_start)
            SEED_START="$2"
            shift 2
            ;;
        --seed_mode)
            SEED_MODE="$2"
            shift 2
            ;;
        --max_trials)
            MAX_TRIALS="$2"
            shift 2
            ;;
        --device_id)
            DEVICE_ID="$2"
            shift 2
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

# オプションの整合性チェック（フォールバックせずエラーで終了）
if [ -n "$DEBUG" ] && [ -n "$USE_BEST" ]; then
    echo "エラー: --debug と --use_best は同時に指定できません"
    exit 1
fi
if [ -n "$EPOCH" ] && [ -n "$DEBUG" ]; then
    echo "エラー: --epoch と --debug は同時に指定できません"
    exit 1
fi
if [ -n "$EPOCH" ] && [ -n "$USE_BEST" ]; then
    echo "エラー: --epoch と --use_best は同時に指定できません"
    exit 1
fi
if [ -n "$EPOCH" ] && ! [[ "$EPOCH" =~ ^[0-9]+$ ]]; then
    echo "エラー: --epoch には0以上の整数を指定してください: $EPOCH"
    exit 1
fi

# seed探索時の整合性チェック（フォールバックせずエラーで終了）
if [ -n "$SEARCH_SEED" ]; then
    if [ -z "$EPOCH" ]; then
        echo "エラー: --search_seed を指定する場合は --epoch を指定してください（評価対象チェックポイントが必要）"
        exit 1
    fi
    if [ -n "$DEBUG" ] || [ -n "$USE_BEST" ]; then
        echo "エラー: --search_seed と --debug/--use_best は同時に指定できません（epoch固定で探索するため）"
        exit 1
    fi
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "エラー: --config が存在しません: $CONFIG_PATH"
        exit 1
    fi
    if [ -z "$TARGET_MIOU" ] || [ -z "$TARGET_OACC" ] || [ -z "$SEED_START" ] || [ -z "$SEED_MODE" ] || [ -z "$MAX_TRIALS" ] || [ -z "$DEVICE_ID" ]; then
        echo "エラー: seed探索オプションが不正です（空の値があります）"
        exit 1
    fi
    if [ "$SEED_MODE" != "sequential" ] && [ "$SEED_MODE" != "random" ]; then
        echo "エラー: --seed_mode は sequential または random を指定してください: $SEED_MODE"
        exit 1
    fi
fi

# --skip_inference の場合、推論系オプションは使われないので明示
if [ -n "$SKIP_INFERENCE" ] && { [ -n "$DEBUG" ] || [ -n "$USE_BEST" ] || [ -n "$EPOCH" ]; }; then
    echo "注意: --skip_inference が指定されているため、--debug/--use_best/--epoch は推論には使用されません"
fi

echo "=========== TCUSS テスト処理開始 ==========="
echo "モデルディレクトリ: $MODEL_DIR"
echo "提出ファイル名: $SUBMISSION_NAME"
echo "データパス: $DATA_PATH"
echo "データセットパス: $DATASET_PATH"
echo "TCUSS config: $CONFIG_PATH"
if [ -n "$EPOCH" ]; then
    echo "モード: epoch指定 (epoch=$EPOCH)"
elif [ -n "$DEBUG" ]; then
    echo "モード: デバッグ (最新のチェックポイントのみ)"
elif [ -n "$USE_BEST" ]; then
    echo "モード: bestモデル使用 (best_model.pth, best_classifier.pth)"
else
    echo "モード: 通常 (全チェックポイント評価)"
fi
if [ -n "$SEARCH_SEED" ]; then
    echo "seed探索: 有効 (target_mIoU=$TARGET_MIOU, target_oAcc=$TARGET_OACC, seed_start=$SEED_START, seed_mode=$SEED_MODE, max_trials=$MAX_TRIALS, device_id=$DEVICE_ID)"
fi
echo "==========================================="

# epoch指定用のオプション配列
EPOCH_OPT=()
if [ -n "$EPOCH" ]; then
    EPOCH_OPT=(--epoch "$EPOCH")
fi

# seed探索で見つかったseedを渡すためのオプション配列
SEED_OPT=()
FOUND_SEED_FILE=""
FOUND_SEED=""
if [ -n "$SEARCH_SEED" ]; then
    echo "0. Seed探索: val(08)で条件達成するKMeans seedを探索"
    FOUND_SEED_FILE="$MODEL_DIR/found_kmeans_seed_epoch${EPOCH}.txt"
    LOG_CSV="$MODEL_DIR/kmeans_seed_search_epoch${EPOCH}.csv"
    python search_kmeans_seed_SemanticKITTI.py \
        --config "$CONFIG_PATH" \
        --save_path "$MODEL_DIR" \
        --epoch "$EPOCH" \
        --target_miou "$TARGET_MIOU" \
        --target_oacc "$TARGET_OACC" \
        --seed_start "$SEED_START" \
        --seed_mode "$SEED_MODE" \
        --max_trials "$MAX_TRIALS" \
        --device "$DEVICE_ID" \
        --out_seed_file "$FOUND_SEED_FILE" \
        --log_csv "$LOG_CSV" || { echo "Seed探索に失敗しました"; exit 1; }

    if [ ! -f "$FOUND_SEED_FILE" ]; then
        echo "エラー: Seed探索結果ファイルが作成されませんでした: $FOUND_SEED_FILE"
        exit 1
    fi
    FOUND_SEED=$(cat "$FOUND_SEED_FILE" | tr -d ' \t\r\n')
    if [ -z "$FOUND_SEED" ]; then
        echo "エラー: Seed探索結果が空です: $FOUND_SEED_FILE"
        exit 1
    fi
    if ! [[ "$FOUND_SEED" =~ ^[0-9]+$ ]]; then
        echo "エラー: Seed探索結果が整数ではありません: $FOUND_SEED"
        exit 1
    fi
    echo "Seed探索完了: kmeans_seed=$FOUND_SEED (saved: $FOUND_SEED_FILE)"
    SEED_OPT=(--seed "$FOUND_SEED")
fi

# 各コマンドを順番に実行
if [ -z "$SKIP_INFERENCE" ]; then
    echo "1. テスト実行: テストデータでの予測実行"
    python test_SemanticKITTI.py \
        --data_path "$DATA_PATH" \
        --save_path "$MODEL_DIR" \
        "${EPOCH_OPT[@]}" \
        "${SEED_OPT[@]}" \
        $DEBUG \
        $USE_BEST || { echo "テスト実行に失敗しました"; exit 1; }
else
    echo "1. テスト実行: --skip_inference のためスキップ（既存pred_resultを使用）"
    if [ ! -d "$MODEL_DIR/pred_result/sequences" ]; then
        echo "エラー: 既存pred_resultが見つかりません: $MODEL_DIR/pred_result/sequences"
        exit 1
    fi
fi

echo "2. 評価: 公式評価スクリプトで精度計測"
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SEMANTIC_KITTI_API_DIR="$SCRIPT_DIR/../semantic-kitti-api"
cd "$SEMANTIC_KITTI_API_DIR" || { echo "semantic-kitti-apiディレクトリへの移動に失敗しました: $SEMANTIC_KITTI_API_DIR"; exit 1; }
python evaluate_semantics.py -d "$CURRENT_DIR/$DATASET_PATH" -p "$CURRENT_DIR/$MODEL_DIR/pred_result/" || { echo "評価に失敗しました"; exit 1; }
cd "$CURRENT_DIR" || { echo "元のディレクトリへの移動に失敗しました"; exit 1; }

echo "3. 提出ファイル作成: ZIPアーカイブを作成"
RESULT_DIR="$MODEL_DIR/pred_result"
cd "$RESULT_DIR" || { echo "結果ディレクトリへの移動に失敗しました"; exit 1; }
zip -r -q "$SUBMISSION_NAME" . || { echo "ZIP作成に失敗しました"; exit 1; }

echo "4. 提出ファイル検証: フォーマット検証"
cd "$CURRENT_DIR" || { echo "元のディレクトリへの移動に失敗しました"; exit 1; }
python "$SEMANTIC_KITTI_API_DIR/validate_submission.py" "$RESULT_DIR/$SUBMISSION_NAME" "$CURRENT_DIR/$DATASET_PATH" || { echo "提出ファイル検証に失敗しました"; exit 1; }

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