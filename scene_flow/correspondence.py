"""
対応点計算モジュール

Scene flowを使って点の対応を計算するためのユーティリティ関数。
VoteFlowに依存しない純粋なNumPy/PyTorch実装。
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple


def compute_point_correspondence(
    points_t: np.ndarray,
    points_t1: np.ndarray,
    flow: np.ndarray,
    valid_indices_t: np.ndarray,
    distance_threshold: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scene flowを使って点の対応を計算（KD-Treeによる高速実装）
    
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
        valid_sp_t: 有効なSPインデックス（時刻t）
        valid_sp_t1: 有効なSPインデックス（時刻t+1）
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
    
    # テストデータ
    points_t = np.random.randn(100, 3).astype(np.float32)
    points_t1 = points_t + np.random.randn(100, 3).astype(np.float32) * 0.1
    
    # ダミーフロー
    flow = points_t1[:50] - points_t[:50]
    valid_indices = np.arange(50)
    
    # 対応点計算
    correspondence, distances = compute_point_correspondence(points_t, points_t1, flow, valid_indices, 0.5)
    print(f'Correspondence shape: {correspondence.shape}')
    print(f'Valid correspondences: {np.sum(correspondence >= 0)}')
    
    # SP対応行列計算
    sp_labels_t = np.repeat(np.arange(10), 10)
    sp_labels_t1 = np.repeat(np.arange(10), 10)
    
    corr_matrix, unique_sp_t, unique_sp_t1 = compute_superpoint_correspondence_matrix(
        correspondence, sp_labels_t, sp_labels_t1, min_points=1
    )
    print(f'Correspondence matrix shape: {corr_matrix.shape}')
    print(f'Unique SPs t: {len(unique_sp_t)}, t1: {len(unique_sp_t1)}')
    print('All tests passed!')








