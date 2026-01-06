#!/usr/bin/env python3
"""
STC (Superpoint Time Consistency) 対応可視化スクリプト

連続する2フレーム（t, t+n）で対応するSuperpointを同じ色で可視化する。
VoteFlowの結果を使ってSuperpoint間の対応を計算し、PLYファイルとして保存する。

使用例:
    # t と t+1 の対応を可視化（エゴモーション除去なし）
    python scripts/visualize_stc_correspondence.py \
        --seq 0 --frame 100 \
        --checkpoint data/users/minesawa/semantickitti/onlyGrowSP/model_30_checkpoint.pth \
        --num_sp 50

    # t と t+1 の対応を可視化（エゴモーション除去あり）
    python scripts/visualize_stc_correspondence.py \
        --seq 0 --frame 100 \
        --checkpoint data/users/minesawa/semantickitti/onlyGrowSP/model_30_checkpoint.pth \
        --num_sp 50 --remove_ego_motion

    # t と t+12 の対応を可視化（累積flow使用、エゴモーション除去あり）
    python scripts/visualize_stc_correspondence.py \
        --seq 0 --frame 100 --frame_gap 12 \
        --checkpoint data/users/minesawa/semantickitti/onlyGrowSP/model_30_checkpoint.pth \
        --num_sp 50 --remove_ego_motion
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import h5py
import MinkowskiEngine as ME
from scipy.spatial import cKDTree
import random

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 再現性のためシードを固定
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

from models.fpn import Res16FPN18
from lib.utils import get_kmeans_labels
from lib.helper_ply import write_ply
from scene_flow.correspondence import (
    compute_point_correspondence_filtered,
    compute_superpoint_correspondence_matrix,
    compute_ego_motion_flow
)


def load_model(checkpoint_path: str, device: str = 'cuda:0'):
    """モデルをロード"""
    model = Res16FPN18(
        in_channels=3,  # input_dim
        out_channels=128,  # feats_dim
        conv1_kernel_size=5,
        config=None
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"モデルをロードしました: {checkpoint_path}")
    return model


def extract_features(model, coords: np.ndarray, voxel_size: float = 0.15, device: str = 'cuda:0'):
    """モデルから特徴量を抽出"""
    with torch.no_grad():
        # バッチID（0）を追加
        batch_idx = np.zeros((len(coords), 1), dtype=np.float32)
        coords_with_batch = np.hstack([batch_idx, coords])
        coords_tensor = torch.from_numpy(coords_with_batch).float().to(device)
        
        in_field = ME.TensorField(coords_tensor[:, 1:] * voxel_size, coords_tensor, device=device)
        feats = model(in_field)
        return feats.detach().cpu()


def load_frame_data_raw(h5_path: str, seq: int, frame: int):
    """1フレームのrawデータをロード（voxelize前）"""
    seq_str = str(seq).zfill(2)
    frame_str = str(frame).zfill(6)
    
    h5_file_path = os.path.join(h5_path, f"{seq_str}.h5")
    if not os.path.exists(h5_file_path):
        raise FileNotFoundError(f"H5ファイルが見つかりません: {h5_file_path}")
    
    with h5py.File(h5_file_path, 'r') as h5_file:
        if frame_str not in h5_file:
            raise KeyError(f"フレーム {frame_str} がH5ファイルに存在しません")
        
        frame_data = h5_file[frame_str]
        coords = frame_data['lidar'][:]  # (N, 3)
        pose = frame_data['pose'][:]  # (4, 4)
        ground_mask = frame_data['ground_mask'][:]  # (N,)
        
        if 'flow_est_fixed' in frame_data:
            flow = frame_data['flow_est_fixed'][:]  # (N, 3)
        else:
            flow = np.zeros_like(coords)
    
    return {
        'coords': coords,
        'pose': pose,
        'flow': flow,
        'ground_mask': ground_mask
    }


def load_frame_data(h5_path: str, seq: int, frame: int, sp_path: str, voxel_size: float = 0.15, r_crop: float = 50.0):
    """1フレームのデータをロード"""
    seq_str = str(seq).zfill(2)
    frame_str = str(frame).zfill(6)
    
    h5_file_path = os.path.join(h5_path, f"{seq_str}.h5")
    if not os.path.exists(h5_file_path):
        raise FileNotFoundError(f"H5ファイルが見つかりません: {h5_file_path}")
    
    with h5py.File(h5_file_path, 'r') as h5_file:
        if frame_str not in h5_file:
            raise KeyError(f"フレーム {frame_str} がH5ファイルに存在しません")
        
        frame_data = h5_file[frame_str]
        coords = frame_data['lidar'][:]  # (N, 3)
        pose = frame_data['pose'][:]  # (4, 4)
        ground_mask = frame_data['ground_mask'][:]  # (N,)
        
        if 'flow_est_fixed' in frame_data:
            flow = frame_data['flow_est_fixed'][:]  # (N, 3)
        else:
            flow = np.zeros_like(coords)
    
    # オリジナル座標を保存
    coords_original = coords.copy()
    
    # 中心化
    means = coords.mean(0)
    coords_centered = coords - means
    
    # SPラベルをロード
    sp_file = os.path.join(sp_path, seq_str, f"{frame_str}_superpoint.npy")
    if not os.path.exists(sp_file):
        raise FileNotFoundError(f"SPファイルが見つかりません: {sp_file}")
    sp_labels_full = np.load(sp_file)
    
    # Voxelize
    scale = 1 / voxel_size
    coords_scaled = np.floor(coords_centered * scale)
    result = ME.utils.sparse_quantize(
        np.ascontiguousarray(coords_scaled),
        return_index=True,
        return_inverse=True
    )
    # MinkowskiEngineのバージョンによって戻り値が異なる
    if len(result) == 3:
        coords_vox, unique_map, inverse_map = result
    else:
        coords_vox, _, _, unique_map, inverse_map = result
    
    if isinstance(coords_vox, torch.Tensor):
        coords_vox = coords_vox.numpy()
    coords_vox = coords_vox.astype(np.float32)
    
    # r_cropでクロップ
    mask = np.sqrt(((coords_vox * voxel_size) ** 2).sum(-1)) < r_crop
    coords_vox = coords_vox[mask]
    
    # 各データもマッピング
    sp_labels = sp_labels_full[unique_map][mask]
    ground_mask_vox = ground_mask[unique_map][mask]
    flow_vox = flow[unique_map][mask]
    coords_original_masked = coords_original[unique_map][mask]
    
    return {
        'coords_vox': coords_vox,
        'coords_original': coords_original_masked,
        'sp_labels': sp_labels,
        'pose': pose,
        'flow': flow_vox,
        'ground_mask': ground_mask_vox,
        'unique_map': unique_map,
        'mask': mask
    }


def compute_accumulated_flow(
    h5_path: str, 
    seq: int, 
    start_frame: int, 
    frame_gap: int,
    start_coords: np.ndarray,
    unique_map: np.ndarray,
    mask: np.ndarray,
    remove_ego_motion: bool = True
):
    """
    累積flowを計算する。
    
    t→t+1→t+2→...→t+n のフローを累積してt→t+nの総移動量を計算。
    
    Args:
        h5_path: H5ファイルのパス
        seq: シーケンス番号
        start_frame: 開始フレーム
        frame_gap: フレーム間隔（n）
        start_coords: 開始フレームの座標 (voxelize後, crop後)
        unique_map: voxelize時のユニークマップ
        mask: r_crop後のマスク
        remove_ego_motion: エゴモーションを除去するか
        
    Returns:
        accumulated_flow: 累積flow [N, 3]
        warped_coords: warp後の座標 [N, 3]
    """
    # 初期位置（生座標）
    raw_data = load_frame_data_raw(h5_path, seq, start_frame)
    start_coords = raw_data['coords'][unique_map][mask].copy()  # crop後の座標
    accumulated_flow = np.zeros_like(start_coords)
    
    # 最近傍探索用の座標（常にstep_flowで更新して、元の点群との対応を追跡）
    current_coords_for_nn = start_coords.copy()
    
    ego_str = "without ego" if remove_ego_motion else "with ego"
    print(f"  累積flow計算 ({ego_str}): t={start_frame} -> t+{frame_gap}={start_frame + frame_gap}")
    
    for step in range(frame_gap):
        current_frame = start_frame + step
        next_frame = current_frame + 1
        
        # 現在フレームのrawデータ
        curr_raw = load_frame_data_raw(h5_path, seq, current_frame)
        next_raw = load_frame_data_raw(h5_path, seq, next_frame)
        
        # 現在フレームの全点のflow
        curr_flow = curr_raw['flow']
        curr_pose = curr_raw['pose']
        next_pose = next_raw['pose']
        
        # 現在の位置から最近傍を見つけてflowを取得
        # （累積移動後の点は元の点群と一致しないので、最近傍探索が必要）
        if step == 0:
            # 最初のステップは直接取得可能
            step_flow = curr_flow[unique_map][mask]
            # エゴモーション計算用の座標
            coords_for_ego = start_coords
        else:
            # warpした位置から元の点群への最近傍探索
            tree = cKDTree(curr_raw['coords'])
            distances, indices = tree.query(current_coords_for_nn, k=1)
            step_flow = curr_flow[indices]
            # エゴモーション計算用の座標
            coords_for_ego = current_coords_for_nn
        
        # エゴモーション除去
        if remove_ego_motion:
            ego_flow = compute_ego_motion_flow(coords_for_ego, curr_pose, next_pose)
            object_flow = step_flow - ego_flow
        else:
            object_flow = step_flow
        
        # 累積
        accumulated_flow += object_flow
        # 最近傍探索用の座標は常にstep_flowで更新（元の点群との対応を追跡）
        current_coords_for_nn = current_coords_for_nn + step_flow
        
        if (step + 1) % 5 == 0 or step == frame_gap - 1:
            print(f"    Step {step+1}/{frame_gap}: 累積flow norm平均 = {np.linalg.norm(accumulated_flow, axis=1).mean():.3f}m")
    
    # 最終的なwarp後の座標は、開始座標 + 累積flow
    warped_coords = start_coords + accumulated_flow
    
    return accumulated_flow, warped_coords


def compute_superpoints_kmeans(coords_vox: np.ndarray, feats: torch.Tensor, init_sp: np.ndarray, k: int):
    """初期SPからk-meansでSuperpoint統合"""
    valid_mask = init_sp >= 0
    if not valid_mask.any():
        print("警告: 有効なSPが見つかりません")
        return init_sp
    
    valid_feats = feats[valid_mask]
    valid_sp = init_sp[valid_mask].astype(np.int64)
    
    # 初期SPを連番にリマップ
    unique_sp = np.unique(valid_sp)
    sp_mapping = {old: new for new, old in enumerate(unique_sp)}
    remapped_sp = np.array([sp_mapping[s] for s in valid_sp])
    num_init_sp = len(unique_sp)
    
    if num_init_sp <= k:
        print(f"初期SP数({num_init_sp}) <= k({k})のため、そのまま使用します")
        return init_sp
    
    # 初期SPごとの特徴量を集約
    region_corr = F.one_hot(torch.from_numpy(remapped_sp), num_classes=num_init_sp).float()
    per_region_num = region_corr.sum(0, keepdims=True).t()
    region_feats = F.linear(region_corr.t(), valid_feats.t()) / per_region_num
    region_feats = F.normalize(region_feats, dim=-1)
    
    # k-meansで統合
    sp_idx = get_kmeans_labels(n_clusters=k, pcds=region_feats).long().cpu().numpy()
    
    # 元の点に戻す
    new_sp = np.full_like(init_sp, -1)
    new_sp[valid_mask] = sp_idx[remapped_sp]
    
    return new_sp


def generate_distinct_colors(n: int, seed: int = 42):
    """n個の区別しやすい色を生成"""
    np.random.seed(seed)
    colors = np.zeros((n, 3), dtype=np.uint8)
    
    # HSVでできるだけ離れた色相を使う
    for i in range(n):
        hue = (i * 137.508) % 360  # 黄金角で分散
        sat = 0.7 + np.random.random() * 0.3
        val = 0.7 + np.random.random() * 0.3
        
        # HSV to RGB
        h = hue / 60.0
        c = val * sat
        x = c * (1 - abs(h % 2 - 1))
        m = val - c
        
        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        colors[i] = [int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)]
    
    return colors


def save_ply(coords: np.ndarray, colors: np.ndarray, labels: np.ndarray, output_path: str):
    """PLYファイルを保存（指定フォーマット）"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # ヘッダーを手動で書く
    n_points = len(coords)
    header = f"""ply
format ascii 1.0
element vertex {n_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property int label
end_header
"""
    
    with open(output_path, 'w') as f:
        f.write(header)
        for i in range(n_points):
            f.write(f"{coords[i, 0]:.6f} {coords[i, 1]:.6f} {coords[i, 2]:.6f} ")
            f.write(f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]} ")
            f.write(f"{labels[i]}\n")
    
    print(f"保存完了: {output_path}")


def colorize_superpoints_independent(sp_labels: np.ndarray, num_sp: int, seed: int = 42):
    """
    Superpointラベルに基づいて各点に色を割り当てる（各時刻で独立）
    
    Args:
        sp_labels: 各点のSPラベル [N]
        num_sp: Superpoint数（色パレットのサイズ）
        seed: 乱数シード
        
    Returns:
        colors: 各点の色 [N, 3]
    """
    # SPごとにユニークな色を生成
    sp_colors = generate_distinct_colors(num_sp, seed=seed)
    
    # 点ごとに色を割り当て
    colors = np.zeros((len(sp_labels), 3), dtype=np.uint8)
    
    for i, sp_id in enumerate(sp_labels):
        if sp_id >= 0:
            colors[i] = sp_colors[sp_id % num_sp]
        else:
            colors[i] = [0, 0, 0]  # 無効点は黒
    
    return colors


def main():
    parser = argparse.ArgumentParser(description='STC対応可視化')
    parser.add_argument('--seq', type=int, default=0, help='シーケンス番号')
    parser.add_argument('--frame', type=int, default=100, help='開始フレーム番号')
    parser.add_argument('--frame_gap', type=int, default=1, help='フレーム間隔（n）。t と t+n の対応を可視化')
    parser.add_argument('--checkpoint', type=str, 
                        default='data/users/minesawa/semantickitti/onlyGrowSP/model_30_checkpoint.pth',
                        help='モデルチェックポイントパス')
    parser.add_argument('--num_sp', type=int, default=50, help='Superpoint数（k）')
    parser.add_argument('--h5_path', type=str, 
                        default='data/dataset/semantickitti/voteflow_preprocess_fixed',
                        help='VoteFlow H5ファイルパス')
    parser.add_argument('--sp_path', type=str,
                        default='data/users/minesawa/semantickitti/growsp_sp',
                        help='初期SPパス')
    parser.add_argument('--output_dir', type=str,
                        default='data/users/minesawa/semantickitti/vis_stc_corr',
                        help='出力ディレクトリ')
    parser.add_argument('--voxel_size', type=float, default=0.15)
    parser.add_argument('--r_crop', type=float, default=50.0)
    parser.add_argument('--distance_threshold', type=float, default=0.3, help='対応点距離閾値')
    parser.add_argument('--min_points', type=int, default=5, help='SP対応に必要な最小点数')
    parser.add_argument('--remove_ego_motion', action='store_true', default=False, 
                        help='エゴモーションを除去するか（デフォルト: False）')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    end_frame = args.frame + args.frame_gap
    
    print("="*60)
    print("STC対応可視化スクリプト")
    print("="*60)
    print(f"シーケンス: {args.seq}, フレーム: {args.frame} -> {end_frame} (gap={args.frame_gap})")
    print(f"Superpoint数: {args.num_sp}")
    print()
    
    # 1. モデルロード
    print("[1/6] モデルをロード中...")
    model = load_model(args.checkpoint, args.device)
    
    # 2. フレームデータロード
    print("[2/6] フレームデータをロード中...")
    data_t = load_frame_data(args.h5_path, args.seq, args.frame, args.sp_path, args.voxel_size, args.r_crop)
    data_tn = load_frame_data(args.h5_path, args.seq, end_frame, args.sp_path, args.voxel_size, args.r_crop)
    
    print(f"  時刻t: {len(data_t['coords_vox'])} 点, 初期SP数: {len(np.unique(data_t['sp_labels'][data_t['sp_labels'] >= 0]))}")
    print(f"  時刻t+{args.frame_gap}: {len(data_tn['coords_vox'])} 点, 初期SP数: {len(np.unique(data_tn['sp_labels'][data_tn['sp_labels'] >= 0]))}")
    
    # 3. 特徴量抽出
    print("[3/6] 特徴量を抽出中...")
    feats_t = extract_features(model, data_t['coords_vox'], args.voxel_size, args.device)
    feats_tn = extract_features(model, data_tn['coords_vox'], args.voxel_size, args.device)
    
    # 4. k-meansでSuperpoint統合
    print(f"[4/6] k-means (k={args.num_sp}) でSuperpoint統合中...")
    sp_t = compute_superpoints_kmeans(data_t['coords_vox'], feats_t, data_t['sp_labels'], args.num_sp)
    sp_tn = compute_superpoints_kmeans(data_tn['coords_vox'], feats_tn, data_tn['sp_labels'], args.num_sp)
    
    print(f"  時刻t: 統合後SP数 = {len(np.unique(sp_t[sp_t >= 0]))}")
    print(f"  時刻t+{args.frame_gap}: 統合後SP数 = {len(np.unique(sp_tn[sp_tn >= 0]))}")
    
    # 点群統計情報
    print()
    print("=== 点群統計情報 ===")
    print(f"[時刻t={args.frame}]")
    coords_t = data_t['coords_original']
    print(f"  点数: {len(coords_t)}")
    print(f"  平均座標: x={coords_t[:, 0].mean():.3f}, y={coords_t[:, 1].mean():.3f}, z={coords_t[:, 2].mean():.3f}")
    print(f"  標準偏差: x={coords_t[:, 0].std():.3f}, y={coords_t[:, 1].std():.3f}, z={coords_t[:, 2].std():.3f}")
    print(f"  範囲: x=[{coords_t[:, 0].min():.2f}, {coords_t[:, 0].max():.2f}], "
          f"y=[{coords_t[:, 1].min():.2f}, {coords_t[:, 1].max():.2f}], "
          f"z=[{coords_t[:, 2].min():.2f}, {coords_t[:, 2].max():.2f}]")
    
    print(f"[時刻t+{args.frame_gap}={args.frame + args.frame_gap}]")
    coords_tn = data_tn['coords_original']
    print(f"  点数: {len(coords_tn)}")
    print(f"  平均座標: x={coords_tn[:, 0].mean():.3f}, y={coords_tn[:, 1].mean():.3f}, z={coords_tn[:, 2].mean():.3f}")
    print(f"  標準偏差: x={coords_tn[:, 0].std():.3f}, y={coords_tn[:, 1].std():.3f}, z={coords_tn[:, 2].std():.3f}")
    print(f"  範囲: x=[{coords_tn[:, 0].min():.2f}, {coords_tn[:, 0].max():.2f}], "
          f"y=[{coords_tn[:, 1].min():.2f}, {coords_tn[:, 1].max():.2f}], "
          f"z=[{coords_tn[:, 2].min():.2f}, {coords_tn[:, 2].max():.2f}]")
    print()
    
    # 5. 対応点計算
    print("[5/6] VoteFlowを使ってSuperpoint対応を計算中...")
    
    ego_str = "without ego" if args.remove_ego_motion else "with ego"
    print(f"  エゴモーション除去: {args.remove_ego_motion} ({ego_str})")
    
    # frame_gap > 1 の場合は累積flowを計算
    if args.frame_gap > 1:
        accumulated_flow, warped_coords = compute_accumulated_flow(
            args.h5_path, args.seq, args.frame, args.frame_gap,
            data_t['coords_original'],
            data_t['unique_map'], data_t['mask'],
            remove_ego_motion=args.remove_ego_motion
        )
    else:
        # frame_gap == 1 の場合は通常のflow
        accumulated_flow = data_t['flow']
        warped_coords = data_t['coords_original'] + data_t['flow']
    
    correspondence_t, _ = compute_point_correspondence_filtered(
        data_t['coords_original'],
        data_tn['coords_original'],
        accumulated_flow,
        sp_t,
        sp_tn,
        data_t['pose'],
        data_tn['pose'],
        data_t['ground_mask'],
        data_tn['ground_mask'],
        distance_threshold=args.distance_threshold,
        remove_ego_motion=(args.frame_gap == 1 and args.remove_ego_motion),  # gap>1の場合は既に処理済み
        exclude_ground=True
    )
    
    corr_matrix, unique_sp_t, unique_sp_tn = compute_superpoint_correspondence_matrix(
        correspondence_t, sp_t, sp_tn, min_points=args.min_points
    )
    
    print(f"  対応点数: {np.sum(correspondence_t >= 0)}")
    print(f"  対応行列形状: {corr_matrix.shape}")
    
    # 各SP_tの最大対応先を取得
    sp_t_to_sp_tn = {}
    for i, sp_id_t in enumerate(unique_sp_t):
        if corr_matrix[i].sum() >= args.min_points:
            # 最も対応点が多いSP_tn
            best_idx = np.argmax(corr_matrix[i])
            sp_t_to_sp_tn[sp_id_t] = unique_sp_tn[best_idx]
    
    print(f"  有効な対応ペア数: {len(sp_t_to_sp_tn)}")
    
    # 6. 色付けして保存
    print("[6/6] 色付けしてPLY保存中...")
    
    # 対応があるペアに同じ色を割り当てる
    pair_colors = generate_distinct_colors(len(sp_t_to_sp_tn) + 1)
    unmatched_color = np.array([128, 128, 128], dtype=np.uint8)
    
    # SP_t -> 色のマッピング
    sp_t_color_map = {}
    sp_tn_color_map = {}
    color_idx = 0
    for sp_id_t, sp_id_tn in sp_t_to_sp_tn.items():
        color = pair_colors[color_idx]
        sp_t_color_map[sp_id_t] = color
        sp_tn_color_map[sp_id_tn] = color
        color_idx += 1
    
    # 点ごとに色を割り当て
    colors_t = np.zeros((len(sp_t), 3), dtype=np.uint8)
    colors_tn = np.zeros((len(sp_tn), 3), dtype=np.uint8)
    
    for i, sp_id in enumerate(sp_t):
        if sp_id in sp_t_color_map:
            colors_t[i] = sp_t_color_map[sp_id]
        elif sp_id >= 0:
            colors_t[i] = unmatched_color
        else:
            colors_t[i] = [0, 0, 0]
    
    for i, sp_id in enumerate(sp_tn):
        if sp_id in sp_tn_color_map:
            colors_tn[i] = sp_tn_color_map[sp_id]
        elif sp_id >= 0:
            colors_tn[i] = unmatched_color
        else:
            colors_tn[i] = [0, 0, 0]
    
    # 出力ディレクトリ（frame_gapごとに分ける）
    seq_str = str(args.seq).zfill(2)
    frame_str = str(args.frame).zfill(6)
    frame_str_n = str(end_frame).zfill(6)
    ego_suffix = "_without_ego" if args.remove_ego_motion else "_with_ego"
    
    output_dir_gap = os.path.join(args.output_dir, f"gap{args.frame_gap}")
    os.makedirs(output_dir_gap, exist_ok=True)
    
    output_path_t = os.path.join(output_dir_gap, f"seq{seq_str}_frame{frame_str}_t{ego_suffix}.ply")
    output_path_tn = os.path.join(output_dir_gap, f"seq{seq_str}_frame{frame_str_n}_tn{ego_suffix}.ply")
    
    save_ply(data_t['coords_original'], colors_t, sp_t, output_path_t)
    save_ply(data_tn['coords_original'], colors_tn, sp_tn, output_path_tn)
    
    # === 追加: 元の点群とwarp後の点群を保存（Superpoint無し） ===
    print()
    print("[追加] 元の点群とwarp後の点群を保存中...")
    
    # 元の点群（t時刻）（緑色）
    colors_orig = np.tile([0, 255, 0], (len(data_t['coords_original']), 1)).astype(np.uint8)
    labels_orig = np.zeros(len(data_t['coords_original']), dtype=np.int32)
    output_path_orig = os.path.join(output_dir_gap, f"seq{seq_str}_frame{frame_str}_original.ply")
    save_ply(data_t['coords_original'], colors_orig, labels_orig, output_path_orig)
    
    # warp後の点群（赤色）+ 正解の点群（青色）を1つのファイルに結合
    colors_warped = np.tile([255, 0, 0], (len(warped_coords), 1)).astype(np.uint8)
    labels_warped = np.ones(len(warped_coords), dtype=np.int32)
    
    colors_target = np.tile([0, 0, 255], (len(data_tn['coords_original']), 1)).astype(np.uint8)
    labels_target = np.full(len(data_tn['coords_original']), 2, dtype=np.int32)
    
    # 結合
    combined_coords = np.vstack([warped_coords, data_tn['coords_original']])
    combined_colors = np.vstack([colors_warped, colors_target])
    combined_labels = np.concatenate([labels_warped, labels_target])
    
    output_path_warped = os.path.join(output_dir_gap, f"seq{seq_str}_frame{frame_str}_warped{ego_suffix}.ply")
    save_ply(combined_coords, combined_colors, combined_labels, output_path_warped)
    
    # 統計情報も保存
    stats_path = os.path.join(output_dir_gap, f"seq{seq_str}_frame{frame_str}_stats{ego_suffix}.txt")
    with open(stats_path, 'w') as f:
        f.write(f"STC Correspondence Visualization Stats\n")
        f.write(f"="*50 + "\n")
        f.write(f"Sequence: {args.seq}\n")
        f.write(f"Frame t: {args.frame}\n")
        f.write(f"Frame t+n: {end_frame} (gap={args.frame_gap})\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Remove ego motion: {args.remove_ego_motion}\n")
        f.write(f"\n")
        
        f.write(f"=== Point Cloud Statistics ===\n")
        f.write(f"[Time t={args.frame}]\n")
        f.write(f"  Points: {len(coords_t)}\n")
        f.write(f"  Mean: x={coords_t[:, 0].mean():.3f}, y={coords_t[:, 1].mean():.3f}, z={coords_t[:, 2].mean():.3f}\n")
        f.write(f"  Std: x={coords_t[:, 0].std():.3f}, y={coords_t[:, 1].std():.3f}, z={coords_t[:, 2].std():.3f}\n")
        f.write(f"[Time t+{args.frame_gap}={args.frame + args.frame_gap}]\n")
        f.write(f"  Points: {len(coords_tn)}\n")
        f.write(f"  Mean: x={coords_tn[:, 0].mean():.3f}, y={coords_tn[:, 1].mean():.3f}, z={coords_tn[:, 2].mean():.3f}\n")
        f.write(f"  Std: x={coords_tn[:, 0].std():.3f}, y={coords_tn[:, 1].std():.3f}, z={coords_tn[:, 2].std():.3f}\n")
        f.write(f"\n")
        
        f.write(f"Points (voxelized, t): {len(data_t['coords_vox'])}\n")
        f.write(f"Points (voxelized, t+n): {len(data_tn['coords_vox'])}\n")
        f.write(f"Superpoints (k): {args.num_sp}\n")
        f.write(f"Matched pairs: {len(sp_t_to_sp_tn)}\n")
        f.write(f"Correspondence count: {np.sum(correspondence_t >= 0)}\n")
        f.write(f"\n")
        
        ego_label = "without ego motion" if args.remove_ego_motion else "with ego motion"
        f.write(f"Accumulated Flow Stats ({ego_label}):\n")
        f.write(f"  Mean flow magnitude: {np.linalg.norm(accumulated_flow, axis=1).mean():.3f} m\n")
        f.write(f"  Max flow magnitude: {np.linalg.norm(accumulated_flow, axis=1).max():.3f} m\n")
        f.write(f"  Min flow magnitude: {np.linalg.norm(accumulated_flow, axis=1).min():.3f} m\n")
        f.write(f"\n")
        
        f.write(f"=== Warped PLY File Contents ===\n")
        f.write(f"  Red points (label=1): Warped prediction ({len(warped_coords)} points)\n")
        f.write(f"  Blue points (label=2): Ground truth at t+{args.frame_gap} ({len(data_tn['coords_original'])} points)\n")
        f.write(f"  Total points in warped PLY: {len(combined_coords)}\n")
        f.write(f"\n")
        
        f.write(f"Matched SP pairs:\n")
        for sp_id_t, sp_id_tn in sp_t_to_sp_tn.items():
            corr_count = corr_matrix[np.where(unique_sp_t == sp_id_t)[0][0], np.where(unique_sp_tn == sp_id_tn)[0][0]]
            f.write(f"  SP_t={sp_id_t} -> SP_t+n={sp_id_tn} (対応点数: {int(corr_count)})\n")
    
    print(f"統計情報を保存: {stats_path}")
    
    # === 追加: 各時刻で独立にSuperpointを色付けしたPLYも保存 ===
    print()
    print("[追加] 各時刻の独立したSuperpoint色付けPLYを保存中...")
    
    # 時刻t - 独立した色付け
    colors_t_indep = colorize_superpoints_independent(sp_t, args.num_sp, seed=42)
    output_path_t_indep = os.path.join(output_dir_gap, f"seq{seq_str}_frame{frame_str}_t_sp_only{ego_suffix}.ply")
    save_ply(data_t['coords_original'], colors_t_indep, sp_t, output_path_t_indep)
    
    # 時刻t+n - 独立した色付け（同じseedで色が対応するように）
    colors_tn_indep = colorize_superpoints_independent(sp_tn, args.num_sp, seed=42)
    output_path_tn_indep = os.path.join(output_dir_gap, f"seq{seq_str}_frame{frame_str_n}_tn_sp_only{ego_suffix}.ply")
    save_ply(data_tn['coords_original'], colors_tn_indep, sp_tn, output_path_tn_indep)
    
    print()
    print("完了しました！")
    print(f"出力ディレクトリ: {output_dir_gap}")
    print(f"エゴモーション除去: {args.remove_ego_motion}")
    print(f"")
    print(f"出力ファイル:")
    print(f"  [対応ベース色付け（Superpoint対応可視化）]")
    print(f"  - {output_path_t}")
    print(f"  - {output_path_tn}")
    print(f"  [独立SP色付け（クラスタリング確認用）]")
    print(f"  - {output_path_t_indep}")
    print(f"  - {output_path_tn_indep}")
    print(f"  [元の点群とwarp後の点群（Flow確認用）]")
    print(f"  - {output_path_orig} (緑: t時刻の元の点群)")
    ego_description = "物体の動きのみ" if args.remove_ego_motion else "VoteFlowの生の推定結果"
    print(f"  - {output_path_warped}")
    print(f"      赤: warp後の点群（予測: {ego_description}）")
    print(f"      青: t+{args.frame_gap}時刻の実際の点群（正解）")
    print(f"  [統計情報]")
    print(f"  - {stats_path}")


if __name__ == "__main__":
    main()
