#!/usr/bin/env python3
"""
STC (Superpoint Time Consistency) 対応可視化スクリプト

連続する2フレーム（t, t+n）で対応するSuperpointを同じ色で可視化する。
VoteFlowの結果を使ってSuperpoint間の対応を計算し、PLYファイルとして保存する。

使用例:
    # t と t+1 の対応を可視化
    python scripts/visualize_stc_correspondence.py \
        --seq 0 --frame 100 \
        --checkpoint data/users/minesawa/semantickitti/onlyGrowSP/model_30_checkpoint.pth \
        --num_sp 50

    # t と t+12 の対応を可視化（累積flow使用）
    python scripts/visualize_stc_correspondence.py \
        --seq 0 --frame 100 --frame_gap 12 \
        --checkpoint data/users/minesawa/semantickitti/onlyGrowSP/model_30_checkpoint.pth \
        --num_sp 50
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

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

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
        start_coords: 開始フレームの座標 (voxelize後、crop後)
        unique_map: voxelize時のユニークマップ
        mask: r_crop後のマスク
        remove_ego_motion: エゴモーションを除去するか
        
    Returns:
        accumulated_flow: 累積flow [N, 3]
    """
    # 初期位置（生座標）
    raw_data = load_frame_data_raw(h5_path, seq, start_frame)
    current_coords = raw_data['coords'][unique_map][mask].copy()  # crop後の座標
    accumulated_flow = np.zeros_like(current_coords)
    
    print(f"  累積flow計算: t={start_frame} -> t+{frame_gap}={start_frame + frame_gap}")
    
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
        else:
            # warpした位置から元の点群への最近傍探索
            tree = cKDTree(curr_raw['coords'])
            distances, indices = tree.query(current_coords, k=1)
            step_flow = curr_flow[indices]
        
        # エゴモーション除去
        if remove_ego_motion:
            ego_flow = compute_ego_motion_flow(current_coords, curr_pose, next_pose)
            object_flow = step_flow - ego_flow
        else:
            object_flow = step_flow
        
        # 累積
        accumulated_flow += object_flow
        current_coords = current_coords + step_flow  # 次のステップ用に位置を更新
        
        if (step + 1) % 5 == 0 or step == frame_gap - 1:
            print(f"    Step {step+1}/{frame_gap}: 累積flow norm平均 = {np.linalg.norm(accumulated_flow, axis=1).mean():.3f}m")
    
    return accumulated_flow


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
    
    # 5. 対応点計算
    print("[5/6] VoteFlowを使ってSuperpoint対応を計算中...")
    
    # frame_gap > 1 の場合は累積flowを計算
    if args.frame_gap > 1:
        accumulated_flow = compute_accumulated_flow(
            args.h5_path, args.seq, args.frame, args.frame_gap,
            data_t['coords_original'],
            data_t['unique_map'], data_t['mask'],
            remove_ego_motion=True
        )
    else:
        # frame_gap == 1 の場合は通常のflow
        accumulated_flow = data_t['flow']
        # エゴモーション除去は compute_point_correspondence_filtered 内で行われる
    
    correspondence_t, _ = compute_point_correspondence_filtered(
        data_t['coords_original'],
        data_tn['coords_original'],
        accumulated_flow if args.frame_gap > 1 else data_t['flow'],
        sp_t,
        sp_tn,
        data_t['pose'],
        data_tn['pose'],
        data_t['ground_mask'],
        data_tn['ground_mask'],
        distance_threshold=args.distance_threshold,
        remove_ego_motion=(args.frame_gap == 1),  # gap>1の場合は既に除去済み
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
    
    # 出力ディレクトリ
    seq_str = str(args.seq).zfill(2)
    frame_str = str(args.frame).zfill(6)
    frame_str_n = str(end_frame).zfill(6)
    gap_suffix = f"_gap{args.frame_gap}" if args.frame_gap > 1 else ""
    
    output_path_t = os.path.join(args.output_dir, f"seq{seq_str}_frame{frame_str}{gap_suffix}_t.ply")
    output_path_tn = os.path.join(args.output_dir, f"seq{seq_str}_frame{frame_str_n}{gap_suffix}_tn.ply")
    
    save_ply(data_t['coords_original'], colors_t, sp_t, output_path_t)
    save_ply(data_tn['coords_original'], colors_tn, sp_tn, output_path_tn)
    
    # 統計情報も保存
    stats_path = os.path.join(args.output_dir, f"seq{seq_str}_frame{frame_str}{gap_suffix}_stats.txt")
    with open(stats_path, 'w') as f:
        f.write(f"STC Correspondence Visualization Stats\n")
        f.write(f"="*50 + "\n")
        f.write(f"Sequence: {args.seq}\n")
        f.write(f"Frame t: {args.frame}\n")
        f.write(f"Frame t+n: {end_frame} (gap={args.frame_gap})\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"\n")
        f.write(f"Points (t): {len(data_t['coords_vox'])}\n")
        f.write(f"Points (t+n): {len(data_tn['coords_vox'])}\n")
        f.write(f"Superpoints (k): {args.num_sp}\n")
        f.write(f"Matched pairs: {len(sp_t_to_sp_tn)}\n")
        f.write(f"Correspondence count: {np.sum(correspondence_t >= 0)}\n")
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
    output_path_t_indep = os.path.join(args.output_dir, f"seq{seq_str}_frame{frame_str}{gap_suffix}_t_sp_only.ply")
    save_ply(data_t['coords_original'], colors_t_indep, sp_t, output_path_t_indep)
    
    # 時刻t+n - 独立した色付け（同じseedで色が対応するように）
    colors_tn_indep = colorize_superpoints_independent(sp_tn, args.num_sp, seed=42)
    output_path_tn_indep = os.path.join(args.output_dir, f"seq{seq_str}_frame{frame_str_n}{gap_suffix}_tn_sp_only.ply")
    save_ply(data_tn['coords_original'], colors_tn_indep, sp_tn, output_path_tn_indep)
    
    print()
    print("完了しました！")
    print(f"出力ファイル:")
    print(f"  [対応ベース色付け]")
    print(f"  - {output_path_t}")
    print(f"  - {output_path_tn}")
    print(f"  [独立SP色付け（クラスタリング確認用）]")
    print(f"  - {output_path_t_indep}")
    print(f"  - {output_path_tn_indep}")
    print(f"  [統計情報]")
    print(f"  - {stats_path}")


if __name__ == "__main__":
    main()
