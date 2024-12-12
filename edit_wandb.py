from wandb import Api
import time
import statistics

def update_run_summary_with_stats(run, key):
    """
    指定されたrunとkeyに基づいて、最近のデータから平均と標準偏差を計算し、summaryに更新します。

    Args:
        run: wandbのRunオブジェクト
        key: 計算する指標のキー
    """
    history = run.scan_history()
    values = [row[key] for row in history if key in row]
    values = [value for value in values if value is not None]
    try:
        recent_values = [values[i] for i in [-1, -11, -21, -31, -41] if i >= -len(values)]
        mean_value = statistics.mean(recent_values)
        std_value = statistics.stdev(recent_values) if len(recent_values) > 1 else 0.0
        run.summary[f'mean_recent_{key}'] = mean_value
        run.summary[f'std_recent_{key}'] = std_value
        run.summary.update()
    except Exception as e:
        print(f"Error processing key {key}: {e}")


api = Api()
# runs = api.runs("TCUSS", filters={"display_name": "debug"})
runs = api.runs("TCUSS", filters={"state": "finished"})

for run in runs:
    # update_run_summary_with_stats(run, "mIoU")
    update_run_summary_with_stats(run, "mAcc")
    update_run_summary_with_stats(run, "oAcc")
    time.sleep(5)
    
    