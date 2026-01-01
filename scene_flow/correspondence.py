"""
対応点計算モジュール

Scene flowを使って点の対応を計算するためのユーティリティ関数。
VoteFlowに依存しない純粋なNumPy実装。

主要な機能:
- エゴモーション除去（object_flow計算）
- SPに属する点のみでのマッチング
- 地面点の除外（オプション）
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Optional


def compute_ego_motion_flow(
    points: np.ndarray,
    pose_t: np.ndarray,
    pose_t1: np.ndarray
) -> np.ndarray:
    """
    エゴモーションによるflowを計算
    
    Args:
        points: 時刻tの点群 [N, 3]
        pose_t: 時刻tのワールド姿勢行列 [4, 4]
        pose_t1: 時刻t+1のワールド姿勢行列 [4, 4]
        
    Returns:
        ego_flow: エゴモーションによる移動ベクトル [N, 3]
    """
    # t+1座標系からt座標系への変換
    ego_pose = np.linalg.inv(pose_t1) @ pose_t
    
    # 各点にエゴモーション変換を適用
    # ego_flow = points @ R.T + t - points
    ego_flow = points @ ego_pose[:3, :3].T + ego_pose[:3, 3] - points
    
    return ego_flow


def compute_object_flow(
    flow: np.ndarray,
    points: np.ndarray,
    pose_t: np.ndarray,
    pose_t1: np.ndarray
) -> np.ndarray:
    """
    エゴモーションを除去した物体固有のflowを計算
    
    Args:
        flow: 全体のflow（エゴモーション込み）[N, 3]
        points: 時刻tの点群 [N, 3]
        pose_t: 時刻tのワールド姿勢行列 [4, 4]
        pose_t1: 時刻t+1のワールド姿勢行列 [4, 4]
        
    Returns:
        object_flow: 物体固有の移動ベクトル [N, 3]
    """
    ego_flow = compute_ego_motion_flow(points, pose_t, pose_t1)
    object_flow = flow - ego_flow
    return object_flow


def compute_point_correspondence(
    points_t: np.ndarray,
    points_t1: np.ndarray,
    flow: np.ndarray,
    valid_indices_t: np.ndarray,
    distance_threshold: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scene flowを使って点の対応を計算（KD-Treeによる高速実装）
    
    ※後方互換性のために残している基本版。新しいコードでは
    compute_point_correspondence_filtered を使用することを推奨。
    
    Args:
        points_t: 時刻tの点群 [N, 3]
        points_t1: 時刻t+1の点群 [M, 3]
        flow: 時刻tの有効点に対するフロー [N_valid, 3]
        valid_indices_t: 時刻tの有効点インデックス [N_valid]
        distance_threshold: 対応点とみなす距離の閾値 (meters)
        
    Returns:
        correspondence_t: 時刻tの各点に対応するt+1の点インデックス [N]、対応なしは-1
        correspondence_distances: 対応点までの距離 [N]、対応なしはinf
    """
    N = len(points_t)
    correspondence_t = -np.ones(N, dtype=np.int64)
    correspondence_distances = np.full(N, np.inf, dtype=np.float32)
    
    if len(flow) == 0 or len(points_t1) == 0 or len(valid_indices_t) == 0:
        return correspondence_t, correspondence_distances
    
    # 有効な点について、フロー適用後の位置を計算
    valid_points_t = points_t[valid_indices_t]  # [N_valid, 3]
    warped_points = valid_points_t + flow  # [N_valid, 3]
    
    # KD-Treeを構築して高速な最近傍探索を実行
    tree = cKDTree(points_t1)
    
    # 全有効点に対して一括で最近傍探索（distance_upper_boundで閾値以上は無視）
    distances, indices = tree.query(
        warped_points, 
        k=1, 
        distance_upper_bound=distance_threshold
    )
    
    # 有効な対応のみを設定（閾値内の点のみ）
    valid_mask = distances <= distance_threshold
    valid_orig_indices = valid_indices_t[valid_mask]
    
    correspondence_t[valid_orig_indices] = indices[valid_mask]
    correspondence_distances[valid_orig_indices] = distances[valid_mask].astype(np.float32)
    
    return correspondence_t, correspondence_distances


def compute_point_correspondence_filtered(
    points_t: np.ndarray,
    points_t1: np.ndarray,
    flow: np.ndarray,
    sp_labels_t: np.ndarray,
    sp_labels_t1: np.ndarray,
    pose_t: Optional[np.ndarray] = None,
    pose_t1: Optional[np.ndarray] = None,
    ground_mask_t: Optional[np.ndarray] = None,
    ground_mask_t1: Optional[np.ndarray] = None,
    distance_threshold: float = 0.3,
    remove_ego_motion: bool = True,
    exclude_ground: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scene flowを使って点の対応を計算（改良版）
    
    以下の改良を含む：
    1. エゴモーション除去（オプション）
    2. SPに属する点のみでKD-Tree構築
    3. 地面点の除外（オプション）
    
    Args:
        points_t: 時刻tの点群 [N, 3]
        points_t1: 時刻t+1の点群 [M, 3]
        flow: 時刻tの各点に対するフロー [N, 3]（全点分）
        sp_labels_t: 時刻tの各点のSuperpointラベル [N]
        sp_labels_t1: 時刻t+1の各点のSuperpointラベル [M]
        pose_t: 時刻tのワールド姿勢行列 [4, 4]（エゴモーション除去用）
        pose_t1: 時刻t+1のワールド姿勢行列 [4, 4]（エゴモーション除去用）
        ground_mask_t: 時刻tの地面マスク [N]（Trueが地面）
        ground_mask_t1: 時刻t+1の地面マスク [M]（Trueが地面）
        distance_threshold: 対応点とみなす距離の閾値 (meters)
        remove_ego_motion: エゴモーションを除去するかどうか
        exclude_ground: 地面点を除外するかどうか
        
    Returns:
        correspondence_t: 時刻tの各点に対応するt+1の点インデックス [N]、対応なしは-1
        correspondence_distances: 対応点までの距離 [N]、対応なしはinf
    """
    N = len(points_t)
    M = len(points_t1)
    correspondence_t = -np.ones(N, dtype=np.int64)
    correspondence_distances = np.full(N, np.inf, dtype=np.float32)
    
    if N == 0 or M == 0:
        return correspondence_t, correspondence_distances
    
    # === 1. エゴモーション除去 ===
    if remove_ego_motion and pose_t is not None and pose_t1 is not None:
        object_flow = compute_object_flow(flow, points_t, pose_t, pose_t1)
    else:
        object_flow = flow
    
    # === 2. 時刻tの有効点を決定（SPに属する & 地面でない） ===
    valid_mask_t = sp_labels_t >= 0
    if exclude_ground and ground_mask_t is not None:
        valid_mask_t = valid_mask_t & ~ground_mask_t
    
    # === 3. 時刻t+1の有効点を決定（SPに属する & 地面でない） ===
    valid_mask_t1 = sp_labels_t1 >= 0
    if exclude_ground and ground_mask_t1 is not None:
        valid_mask_t1 = valid_mask_t1 & ~ground_mask_t1
    
    valid_indices_t = np.where(valid_mask_t)[0]
    valid_indices_t1 = np.where(valid_mask_t1)[0]
    
    if len(valid_indices_t) == 0 or len(valid_indices_t1) == 0:
        return correspondence_t, correspondence_distances
    
    # === 4. 有効点のみでKD-Tree構築 ===
    valid_points_t1 = points_t1[valid_indices_t1]  # [K, 3]
    tree = cKDTree(valid_points_t1)
    
    # === 5. 有効な時刻t点をflow移動して最近傍探索 ===
    valid_points_t = points_t[valid_indices_t]  # [L, 3]
    valid_flow = object_flow[valid_indices_t]  # [L, 3]
    warped_points = valid_points_t + valid_flow  # [L, 3]
    
    distances, local_indices = tree.query(
        warped_points,
        k=1,
        distance_upper_bound=distance_threshold
    )
    
    # === 6. 結果をマッピング ===
    # local_indices はvalid_points_t1内でのインデックス
    # 元のpoints_t1でのインデックスに変換
    match_mask = distances <= distance_threshold
    matched_t_indices = valid_indices_t[match_mask]
    matched_local_t1_indices = local_indices[match_mask]
    
    # valid_indices_t1を使って元のインデックスに変換
    matched_t1_indices = valid_indices_t1[matched_local_t1_indices]
    
    correspondence_t[matched_t_indices] = matched_t1_indices
    correspondence_distances[matched_t_indices] = distances[match_mask].astype(np.float32)
    
    return correspondence_t, correspondence_distances


def compute_superpoint_correspondence_matrix(
    correspondence_t: np.ndarray,
    sp_labels_t: np.ndarray,
    sp_labels_t1: np.ndarray,
    min_points: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    点の対応からSuperpoint間の対応行列を計算
    
    Args:
        correspondence_t: 時刻tの各点に対応するt+1の点インデックス [N]
        sp_labels_t: 時刻tの各点のSuperpointラベル [N]
        sp_labels_t1: 時刻t+1の各点のSuperpointラベル [M]
        min_points: SPペアを有効とする最小対応点数
        
    Returns:
        corr_matrix: 対応行列 [num_sp_t, num_sp_t1]
        unique_sp_t: 有効なSPインデックス（時刻t）
        unique_sp_t1: 有効なSPインデックス（時刻t+1）
    """
    # SPの数を取得（-1を除く）
    unique_sp_t = np.unique(sp_labels_t[sp_labels_t >= 0])
    unique_sp_t1 = np.unique(sp_labels_t1[sp_labels_t1 >= 0])
    
    num_sp_t = len(unique_sp_t)
    num_sp_t1 = len(unique_sp_t1)
    
    if num_sp_t == 0 or num_sp_t1 == 0:
        return np.zeros((0, 0), dtype=np.float32), np.array([]), np.array([])
    
    # SP IDをインデックスにマッピング
    sp_t_to_idx = {sp: i for i, sp in enumerate(unique_sp_t)}
    sp_t1_to_idx = {sp: i for i, sp in enumerate(unique_sp_t1)}
    
    # 対応行列を初期化
    corr_matrix = np.zeros((num_sp_t, num_sp_t1), dtype=np.float32)
    
    # 対応があるかを集計
    for pt_idx, (sp_t, corr_idx) in enumerate(zip(sp_labels_t, correspondence_t)):
        if sp_t < 0 or corr_idx < 0:
            continue
        
        sp_t1 = sp_labels_t1[corr_idx]
        if sp_t1 < 0:
            continue
        
        i = sp_t_to_idx[sp_t]
        j = sp_t1_to_idx[sp_t1]
        corr_matrix[i, j] += 1
    
    return corr_matrix, unique_sp_t, unique_sp_t1


if __name__ == "__main__":
    # テスト
    import numpy as np
    
    print("=== 基本テスト ===")
    # テストデータ
    points_t = np.random.randn(100, 3).astype(np.float32)
    points_t1 = points_t + np.random.randn(100, 3).astype(np.float32) * 0.1
    
    # ダミーフロー
    flow = points_t1[:50] - points_t[:50]
    valid_indices = np.arange(50)
    
    # 対応点計算（基本版）
    correspondence, distances = compute_point_correspondence(points_t, points_t1, flow, valid_indices, 0.5)
    print(f'Correspondence shape: {correspondence.shape}')
    print(f'Valid correspondences: {np.sum(correspondence >= 0)}')
    
    print("\n=== 改良版テスト ===")
    # 改良版テスト
    sp_labels_t = np.repeat(np.arange(10), 10)
    sp_labels_t1 = np.repeat(np.arange(10), 10)
    ground_mask_t = np.zeros(100, dtype=bool)
    ground_mask_t[:10] = True  # 最初の10点を地面とする
    ground_mask_t1 = np.zeros(100, dtype=bool)
    ground_mask_t1[:10] = True
    
    # ダミーpose
    pose_t = np.eye(4, dtype=np.float32)
    pose_t1 = np.eye(4, dtype=np.float32)
    pose_t1[:3, 3] = [0.1, 0.0, 0.0]  # 少し移動
    
    # 全点分のflow
    full_flow = points_t1 - points_t
    
    correspondence_filtered, distances_filtered = compute_point_correspondence_filtered(
        points_t, points_t1, full_flow,
        sp_labels_t, sp_labels_t1,
        pose_t, pose_t1,
        ground_mask_t, ground_mask_t1,
        distance_threshold=0.5,
        remove_ego_motion=True,
        exclude_ground=True
    )
    print(f'Filtered correspondence shape: {correspondence_filtered.shape}')
    print(f'Filtered valid correspondences: {np.sum(correspondence_filtered >= 0)}')
    
    # 地面点（最初の10点）の対応がないことを確認
    ground_correspondences = correspondence_filtered[:10]
    print(f'Ground point correspondences (should be all -1): {ground_correspondences}')
    
    print("\n=== SP対応行列テスト ===")
    corr_matrix, unique_sp_t, unique_sp_t1 = compute_superpoint_correspondence_matrix(
        correspondence_filtered, sp_labels_t, sp_labels_t1, min_points=1
    )
    print(f'Correspondence matrix shape: {corr_matrix.shape}')
    print(f'Unique SPs t: {len(unique_sp_t)}, t1: {len(unique_sp_t1)}')
    
    print('\nAll tests passed!')
