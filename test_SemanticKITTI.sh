#!/bin/bash

# Check if required arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <model_dir> <submission_name>"
    echo "Example: $0 /mnt/urashima/users/minesawa/semantickitti/growsp_model_original growsp.zip"
    exit 1
fi

# Assign command line arguments to variables
MODEL_DIR="$1"
SUBMISSION_NAME="$2"
DATASET_PATH="/mnt/urashima/dataset/semantickitti/dataset/"

# Execute commands in sequence
python test_SemanticKITTI.py \
    --data_path /mnt/urashima/users/minesawa/semantickitti/growsp \
    --sp_path /mnt/urashima/users/minesawa/semantickitti/growsp_sp \
    --save_path "$MODEL_DIR" \
    --debug && \
cd ~/repos/semantic-kitti-api && \
python evaluate_semantics.py -d "$DATASET_PATH" -p "$MODEL_DIR/pred_result/" && \
cd "$MODEL_DIR/pred_result" && \
zip -r -q "$SUBMISSION_NAME" . && \
python ~/repos/semantic-kitti-api/validate_submission.py "$MODEL_DIR/pred_result/$SUBMISSION_NAME" "$DATASET_PATH" && \
cd ~/repos/TCUSS && \
scp -P 2222 -i ~/.ssh/id_ishikawa "$MODEL_DIR/pred_result/$SUBMISSION_NAME" somin@localhost:"C:\Users\somin\Downloads"

# Check if the last command was successful
if [ $? -eq 0 ]; then
    echo "Process completed successfully!"
    echo "File transferred to: C:\\Users\\somin\\Downloads\\$SUBMISSION_NAME"
else
    echo "An error occurred during the process."
    exit 1
fi