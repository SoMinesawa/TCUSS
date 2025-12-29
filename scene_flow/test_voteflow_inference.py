"""
VoteFlow 推論の単体テストスクリプト

使い方:
  CUDA_LAUNCH_BLOCKING=1 WANDB_MODE=disabled \
    python scene_flow/test_voteflow_inference.py --config config/default.yaml --index 0

GPUは自動で決定（GPUが複数ならcuda:1以降を優先、なければcuda:0）。
"""

import argparse
import torch

from lib.config import TCUSSConfig
from datasets.SemanticKITTI import KITTIstc
from scene_flow.voteflow_wrapper import VoteFlowWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML設定ファイルパス")
    parser.add_argument("--index", type=int, default=0, help="取得するサンプルインデックス")
    args = parser.parse_args()

    # 設定読み込み
    config = TCUSSConfig.from_yaml(args.config)

    # データセット準備（ランダムサンプリングを初期化）
    dataset = KITTIstc(config)
    dataset._random_select_samples()
    if len(dataset) == 0:
        raise RuntimeError("KITTIstc のサンプルが0件です。data_path/select_numを確認してください。")

    idx = args.index % len(dataset)
    (
        coords_t,
        coords_t1,
        coords_t_original,
        coords_t1_original,
        sp_labels_t,
        sp_labels_t1,
        pose_t,
        pose_t1,
    ) = dataset[idx]

    # VoteFlow推論
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if device_count > 1:
        device = "cuda:1"
    else:
        device = "cuda:0"

    vf = VoteFlowWrapper(
        checkpoint_path=config.stc.scene_flow.checkpoint,
        voxel_size=config.stc.scene_flow.voxel_size,
        point_cloud_range=config.stc.scene_flow.point_cloud_range,
        device=device,
    )

    with torch.no_grad():
        flow, valid_idx = vf.compute_flow(
            coords_t_original.astype("float32"),
            coords_t1_original.astype("float32"),
            pose_t,
            pose_t1,
        )

    print(f"pc0 shape: {coords_t_original.shape}, pc1 shape: {coords_t1_original.shape}")
    print(f"flow shape: {flow.shape}, valid_idx: {len(valid_idx)}")


if __name__ == "__main__":
    main()


