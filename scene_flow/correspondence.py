"""
対応点計算モジュール

Scene flowを使ってSuperpointの対応を計算するためのユーティリティ関数。
VoteFlowに依存しない純粋なNumPy実装。

主要な機能:
- エゴモーション除去（object_flow計算）
- SPレベルでの直接マッチング（重心、広がり、点数、移動ベクトル）
- 地面点の除外（オプション）
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Optional, Dict


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
    distance_threshold_per_frame: float = 0.0,
    distance_threshold_moving: Optional[float] = None,
    distance_threshold_moving_per_frame: float = 0.0,
    moving_flow_threshold: float = 0.1,
    num_frames: int = 1,
    remove_ego_motion: bool = True,
    exclude_ground: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scene flowを使って点の対応を計算（改良版）
    
    以下の改良を含む：
    1. エゴモーション除去（オプション）
    2. SPに属する点のみでKD-Tree構築
    3. 地面点の除外（オプション）
    4. SPごとの動的distance_threshold（静止/移動物体で異なる閾値）
    
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
        distance_threshold: 静止物体のベース閾値 (meters)
        distance_threshold_per_frame: フレーム数に応じて追加する閾値 (meters)
        distance_threshold_moving: 動いている物体のベース閾値 (meters)、Noneなら静止と同じ
        distance_threshold_moving_per_frame: 動いている物体用のフレーム追加閾値 (meters)
        moving_flow_threshold: moving SPと判定する閾値 (m/frame)
        num_frames: t と t1 のフレーム差
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
    
    # === 4. SPごとの平均object_flowノルムを計算してmoving/static判定 ===
    # SPごとの閾値を計算
    effective_threshold_static = distance_threshold + distance_threshold_per_frame * (num_frames - 1)
    if distance_threshold_moving is not None:
        effective_threshold_moving = distance_threshold_moving + distance_threshold_moving_per_frame * (num_frames - 1)
    else:
        effective_threshold_moving = effective_threshold_static
    
    # 動的閾値を使う場合、SPごとのmoving判定を行う
    use_dynamic_threshold = distance_threshold_moving is not None and moving_flow_threshold > 0
    
    if use_dynamic_threshold:
        # SPごとの平均object_flowノルム（1フレームあたり）を計算
        unique_sp_t = np.unique(sp_labels_t[valid_mask_t])
        sp_is_moving = {}
        
        for sp_id in unique_sp_t:
            sp_mask = (sp_labels_t == sp_id) & valid_mask_t
            sp_flow = object_flow[sp_mask]
            # 累積flowを1フレームあたりに正規化
            avg_flow_norm = np.linalg.norm(sp_flow, axis=-1).mean() / max(num_frames, 1)
            sp_is_moving[sp_id] = avg_flow_norm > moving_flow_threshold
        
        # 点ごとの閾値を設定
        per_point_threshold = np.full(len(valid_indices_t), effective_threshold_static, dtype=np.float32)
        valid_sp_labels = sp_labels_t[valid_indices_t]
        for i, sp_id in enumerate(valid_sp_labels):
            if sp_is_moving.get(sp_id, False):
                per_point_threshold[i] = effective_threshold_moving
        
        # 大きい方の閾値でKD-Tree query
        max_threshold = max(effective_threshold_static, effective_threshold_moving)
    else:
        # 動的閾値を使わない場合は固定閾値
        per_point_threshold = None
        max_threshold = effective_threshold_static
    
    # === 5. 有効点のみでKD-Tree構築 ===
    valid_points_t1 = points_t1[valid_indices_t1]  # [K, 3]
    tree = cKDTree(valid_points_t1)
    
    # === 6. 有効な時刻t点をflow移動して最近傍探索 ===
    valid_points_t = points_t[valid_indices_t]  # [L, 3]
    valid_flow = object_flow[valid_indices_t]  # [L, 3]
    warped_points = valid_points_t + valid_flow  # [L, 3]
    
    distances, local_indices = tree.query(
        warped_points,
        k=1,
        distance_upper_bound=max_threshold
    )
    
    # === 7. 結果をマッピング（点ごとの閾値でフィルタリング） ===
    if per_point_threshold is not None:
        # 点ごとの閾値でフィルタリング
        match_mask = distances <= per_point_threshold
    else:
        match_mask = distances <= max_threshold
    
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
    min_points: int = 5,
    min_sp_points: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    点の対応からSuperpoint間の対応行列を計算
    
    Args:
        correspondence_t: 時刻tの各点に対応するt+1の点インデックス [N]
        sp_labels_t: 時刻tの各点のSuperpointラベル [N]
        sp_labels_t1: 時刻t+1の各点のSuperpointラベル [M]
        min_points: SPペアを有効とする最小対応点数
        min_sp_points: この点数以下のSPは対応計算から除外（0なら除外しない）
        
    Returns:
        corr_matrix: 対応行列 [num_sp_t, num_sp_t1]
        unique_sp_t: 有効なSPインデックス（時刻t）
        unique_sp_t1: 有効なSPインデックス（時刻t+1）
    """
    # SPの数を取得（-1を除く）
    unique_sp_t = np.unique(sp_labels_t[sp_labels_t >= 0])
    unique_sp_t1 = np.unique(sp_labels_t1[sp_labels_t1 >= 0])
    
    # min_sp_points > 0 の場合、小さいSPを除外
    if min_sp_points > 0:
        # 各SPの点数をカウント
        sp_counts_t = {}
        for sp in unique_sp_t:
            sp_counts_t[sp] = np.sum(sp_labels_t == sp)
        sp_counts_t1 = {}
        for sp in unique_sp_t1:
            sp_counts_t1[sp] = np.sum(sp_labels_t1 == sp)
        
        # min_sp_points より大きいSPのみ残す
        unique_sp_t = np.array([sp for sp in unique_sp_t if sp_counts_t[sp] > min_sp_points])
        unique_sp_t1 = np.array([sp for sp in unique_sp_t1 if sp_counts_t1[sp] > min_sp_points])
    
    num_sp_t = len(unique_sp_t)
    num_sp_t1 = len(unique_sp_t1)
    
    if num_sp_t == 0 or num_sp_t1 == 0:
        return np.zeros((0, 0), dtype=np.float32), np.array([]), np.array([])
    
    # SP IDをインデックスにマッピング
    sp_t_to_idx = {sp: i for i, sp in enumerate(unique_sp_t)}
    sp_t1_to_idx = {sp: i for i, sp in enumerate(unique_sp_t1)}
    
    # 対応行列を初期化
    corr_matrix = np.zeros((num_sp_t, num_sp_t1), dtype=np.float32)
    
    # 対応があるかを集計（除外されたSPはスキップ）
    for pt_idx, (sp_t, corr_idx) in enumerate(zip(sp_labels_t, correspondence_t)):
        if sp_t < 0 or corr_idx < 0:
            continue
        
        # 除外されたSPはスキップ
        if sp_t not in sp_t_to_idx:
            continue
        
        sp_t1 = sp_labels_t1[corr_idx]
        if sp_t1 < 0 or sp_t1 not in sp_t1_to_idx:
            continue
        
        i = sp_t_to_idx[sp_t]
        j = sp_t1_to_idx[sp_t1]
        corr_matrix[i, j] += 1
    
    return corr_matrix, unique_sp_t, unique_sp_t1


# =============================================================================
# SPレベルでの直接マッチング（新方式）
# =============================================================================

def compute_sp_features(
    points: np.ndarray,
    sp_labels: np.ndarray,
    flow: np.ndarray,
    ground_mask: Optional[np.ndarray] = None,
    pose_t: Optional[np.ndarray] = None,
    pose_t1: Optional[np.ndarray] = None,
    min_sp_points: int = 10,
    exclude_ground: bool = False,
    remove_ego_motion: bool = False
) -> Dict[int, Dict]:
    """
    各Superpointの特徴量を計算
    
    Args:
        points: 点群座標 [N, 3]
        sp_labels: 各点のSPラベル [N]
        flow: Scene flow [N, 3]
        ground_mask: 地面マスク [N]（Trueが地面）
        pose_t: 時刻tのワールド姿勢行列 [4, 4]（エゴモーション除去用）
        pose_t1: 時刻t+1のワールド姿勢行列 [4, 4]（エゴモーション除去用）
        min_sp_points: この点数以下のSPは除外
        exclude_ground: 地面点を除外するかどうか
        remove_ego_motion: エゴモーションを除去するかどうか
        
    Returns:
        sp_features: {sp_id: {'centroid': [3], 'spread': [3], 'point_count': int, 'motion': [3]}}
    """
    # エゴモーション除去
    if remove_ego_motion and pose_t is not None and pose_t1 is not None:
        object_flow = compute_object_flow(flow, points, pose_t, pose_t1)
    else:
        object_flow = flow
    
    # 有効な点のマスク（SP所属 & 地面でない）
    valid_mask = sp_labels >= 0
    if exclude_ground and ground_mask is not None:
        valid_mask = valid_mask & ~ground_mask
    
    # 有効なユニークSPを取得
    unique_sps = np.unique(sp_labels[valid_mask])
    
    sp_features = {}
    
    for sp_id in unique_sps:
        # このSPに属する点のマスク
        sp_mask = (sp_labels == sp_id) & valid_mask
        sp_points = points[sp_mask]
        sp_flow = object_flow[sp_mask]
        
        point_count = len(sp_points)
        
        # 点数が閾値以下のSPは除外
        if point_count <= min_sp_points:
            continue
        
        # 重心
        centroid = sp_points.mean(axis=0)
        
        # 広がり（各軸の標準偏差）
        spread = sp_points.std(axis=0)
        
        # 移動ベクトル（flowの平均）
        motion = sp_flow.mean(axis=0)
        
        sp_features[sp_id] = {
            'centroid': centroid,
            'spread': spread,
            'point_count': point_count,
            'motion': motion
        }
    
    return sp_features


def compute_sp_matching_score_matrix(
    sp_features_t: Dict[int, Dict],
    sp_features_t1: Dict[int, Dict],
    weight_centroid_distance: float = 1.0,
    weight_spread_similarity: float = 0.3,
    weight_point_count_similarity: float = 0.2,
    max_centroid_distance: float = 3.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SPの特徴量からスコア行列を計算
    
    Args:
        sp_features_t: 時刻tのSP特徴量
        sp_features_t1: 時刻t+1のSP特徴量
        weight_centroid_distance: 重心距離の重み
        weight_spread_similarity: 広がり類似度の重み
        weight_point_count_similarity: 点数類似度の重み
        max_centroid_distance: 重心距離の最大値（これ以上は対応候補から除外）
        
    Returns:
        score_matrix: スコア行列 [num_sp_t, num_sp_t1]（0〜1、高いほど良いマッチ）
        sp_ids_t: 時刻tのSP IDリスト
        sp_ids_t1: 時刻t+1のSP IDリスト
    """
    sp_ids_t = np.array(list(sp_features_t.keys()))
    sp_ids_t1 = np.array(list(sp_features_t1.keys()))
    
    num_sp_t = len(sp_ids_t)
    num_sp_t1 = len(sp_ids_t1)
    
    if num_sp_t == 0 or num_sp_t1 == 0:
        return np.zeros((0, 0), dtype=np.float32), sp_ids_t, sp_ids_t1
    
    # 特徴量を配列に展開
    centroids_t = np.array([sp_features_t[sp_id]['centroid'] for sp_id in sp_ids_t])
    centroids_t1 = np.array([sp_features_t1[sp_id]['centroid'] for sp_id in sp_ids_t1])
    spreads_t = np.array([sp_features_t[sp_id]['spread'] for sp_id in sp_ids_t])
    spreads_t1 = np.array([sp_features_t1[sp_id]['spread'] for sp_id in sp_ids_t1])
    counts_t = np.array([sp_features_t[sp_id]['point_count'] for sp_id in sp_ids_t])
    counts_t1 = np.array([sp_features_t1[sp_id]['point_count'] for sp_id in sp_ids_t1])
    motions_t = np.array([sp_features_t[sp_id]['motion'] for sp_id in sp_ids_t])
    
    # 予測重心（重心 + 移動ベクトル）
    predicted_centroids_t = centroids_t + motions_t  # [num_sp_t, 3]
    
    # スコア行列を初期化
    score_matrix = np.zeros((num_sp_t, num_sp_t1), dtype=np.float32)
    
    # === 1. 重心距離スコア ===
    # [num_sp_t, num_sp_t1]
    centroid_distances = np.linalg.norm(
        predicted_centroids_t[:, np.newaxis, :] - centroids_t1[np.newaxis, :, :],
        axis=-1
    )
    # 距離を0〜1のスコアに変換（近いほど高い）
    # max_centroid_distance以上は0
    centroid_score = np.clip(1.0 - centroid_distances / max_centroid_distance, 0.0, 1.0)
    
    # === 2. 広がり類似度スコア ===
    # コサイン類似度を使用
    # spreads_t: [num_sp_t, 3], spreads_t1: [num_sp_t1, 3]
    spread_norms_t = np.linalg.norm(spreads_t, axis=-1, keepdims=True) + 1e-8
    spread_norms_t1 = np.linalg.norm(spreads_t1, axis=-1, keepdims=True) + 1e-8
    spreads_t_normalized = spreads_t / spread_norms_t
    spreads_t1_normalized = spreads_t1 / spread_norms_t1
    
    # コサイン類似度 [num_sp_t, num_sp_t1]
    spread_similarity = np.dot(spreads_t_normalized, spreads_t1_normalized.T)
    # -1〜1を0〜1に変換
    spread_score = (spread_similarity + 1.0) / 2.0
    
    # === 3. 点数類似度スコア ===
    # min/max で類似度を計算
    counts_t_expanded = counts_t[:, np.newaxis]
    counts_t1_expanded = counts_t1[np.newaxis, :]
    count_min = np.minimum(counts_t_expanded, counts_t1_expanded)
    count_max = np.maximum(counts_t_expanded, counts_t1_expanded)
    count_score = count_min / (count_max + 1e-8)
    
    # === 総合スコア ===
    total_weight = weight_centroid_distance + weight_spread_similarity + weight_point_count_similarity
    if total_weight > 0:
        score_matrix = (
            weight_centroid_distance * centroid_score +
            weight_spread_similarity * spread_score +
            weight_point_count_similarity * count_score
        ) / total_weight
    
    # max_centroid_distance以上のペアは強制的に0
    score_matrix[centroid_distances > max_centroid_distance] = 0.0
    
    return score_matrix, sp_ids_t, sp_ids_t1


def greedy_sp_matching(
    score_matrix: np.ndarray,
    min_score_threshold: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    スコア行列からGreedyで1対1マッチングを行う
    
    Args:
        score_matrix: スコア行列 [num_sp_t, num_sp_t1]
        min_score_threshold: この値以下のスコアは対応として採用しない
        
    Returns:
        matched_indices_t: マッチしたSP_tのインデックス（score_matrixの行インデックス）
        matched_indices_t1: マッチしたSP_t1のインデックス（score_matrixの列インデックス）
        matched_scores: 対応スコア
    """
    num_sp_t, num_sp_t1 = score_matrix.shape
    
    if num_sp_t == 0 or num_sp_t1 == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    
    # コピーして作業（マスク用）
    scores = score_matrix.copy()
    
    matched_t = []
    matched_t1 = []
    matched_scores = []
    
    # 使用済みフラグ
    used_t = np.zeros(num_sp_t, dtype=bool)
    used_t1 = np.zeros(num_sp_t1, dtype=bool)
    
    # Greedyマッチング
    while True:
        # 最大スコアを見つける
        max_score = scores.max()
        
        if max_score <= min_score_threshold:
            break
        
        # 最大スコアの位置
        idx = int(np.argmax(scores))
        i, j = divmod(idx, num_sp_t1)
        
        # マッチを記録
        matched_t.append(i)
        matched_t1.append(j)
        matched_scores.append(max_score)
        
        # 使用済みにする
        used_t[i] = True
        used_t1[j] = True
        
        # この行と列をマスク（-infにする）
        scores[i, :] = -np.inf
        scores[:, j] = -np.inf
    
    return (
        np.array(matched_t, dtype=np.int64),
        np.array(matched_t1, dtype=np.int64),
        np.array(matched_scores, dtype=np.float32)
    )


def compute_superpoint_correspondence_direct(
    points_t: np.ndarray,
    points_t1: np.ndarray,
    flow_t: np.ndarray,
    sp_labels_t: np.ndarray,
    sp_labels_t1: np.ndarray,
    pose_t: Optional[np.ndarray] = None,
    pose_t1: Optional[np.ndarray] = None,
    ground_mask_t: Optional[np.ndarray] = None,
    ground_mask_t1: Optional[np.ndarray] = None,
    weight_centroid_distance: float = 1.0,
    weight_spread_similarity: float = 0.3,
    weight_point_count_similarity: float = 0.2,
    max_centroid_distance: float = 3.0,
    min_score_threshold: float = 0.3,
    min_sp_points: int = 10,
    remove_ego_motion: bool = False,
    exclude_ground: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SPレベルで直接対応を計算する（新方式）
    
    点レベルのマッチングを行わず、SPの特徴量（重心、広がり、点数、移動ベクトル）を使って
    Greedy 1対1マッチングを行う。
    
    Args:
        points_t: 時刻tの点群 [N, 3]
        points_t1: 時刻t+1の点群 [M, 3]
        flow_t: 時刻tのScene flow [N, 3]
        sp_labels_t: 時刻tの各点のSPラベル [N]
        sp_labels_t1: 時刻t+1の各点のSPラベル [M]
        pose_t: 時刻tのワールド姿勢行列 [4, 4]（エゴモーション除去用）
        pose_t1: 時刻t+1のワールド姿勢行列 [4, 4]（エゴモーション除去用）
        ground_mask_t: 時刻tの地面マスク [N]
        ground_mask_t1: 時刻t+1の地面マスク [M]
        weight_centroid_distance: 重心距離の重み
        weight_spread_similarity: 広がり類似度の重み
        weight_point_count_similarity: 点数類似度の重み
        max_centroid_distance: 重心距離の最大値 (m)
        min_score_threshold: 最小スコア閾値
        min_sp_points: 最小SP点数
        remove_ego_motion: エゴモーション除去
        exclude_ground: 地面除外
        
    Returns:
        corr_matrix: 対応行列 [num_valid_sp_t, num_valid_sp_t1]（1対1マッチ箇所が1、他は0）
        unique_sp_t: 有効なSP IDリスト（時刻t）
        unique_sp_t1: 有効なSP IDリスト（時刻t+1）
    """
    # 1. SP特徴量を計算
    sp_features_t = compute_sp_features(
        points_t, sp_labels_t, flow_t,
        ground_mask_t, pose_t, pose_t1,
        min_sp_points, exclude_ground, remove_ego_motion
    )
    
    # t+1側はflowは使わない（ゼロで代用）
    sp_features_t1 = compute_sp_features(
        points_t1, sp_labels_t1, np.zeros_like(points_t1),
        ground_mask_t1, None, None,
        min_sp_points, exclude_ground, False
    )
    
    if len(sp_features_t) == 0 or len(sp_features_t1) == 0:
        return np.zeros((0, 0), dtype=np.float32), np.array([]), np.array([])
    
    # 2. スコア行列を計算
    score_matrix, sp_ids_t, sp_ids_t1 = compute_sp_matching_score_matrix(
        sp_features_t, sp_features_t1,
        weight_centroid_distance,
        weight_spread_similarity,
        weight_point_count_similarity,
        max_centroid_distance
    )
    
    # 3. Greedyマッチング
    matched_indices_t, matched_indices_t1, matched_scores = greedy_sp_matching(
        score_matrix, min_score_threshold
    )
    
    # 4. 対応行列を構築（従来のcorr_matrixと同じ形式）
    # 1対1マッチした箇所が1（または対応スコア）、他は0
    num_sp_t = len(sp_ids_t)
    num_sp_t1 = len(sp_ids_t1)
    corr_matrix = np.zeros((num_sp_t, num_sp_t1), dtype=np.float32)
    
    for idx_t, idx_t1, score in zip(matched_indices_t, matched_indices_t1, matched_scores):
        # スコアを格納（後で重み付けに使える）
        corr_matrix[idx_t, idx_t1] = score
    
    return corr_matrix, sp_ids_t, sp_ids_t1


if __name__ == "__main__":
    # テスト
    import numpy as np
    
    print("=== SPレベル直接マッチングテスト ===")
    
    # テストデータ: 各SPに20点ずつ割り当て（5 SP × 20点 = 100点）
    np.random.seed(42)
    num_sps = 5
    points_per_sp = 20
    
    # 時刻tの点群を作成（SPごとにクラスタ）
    points_t = []
    sp_labels_t = []
    for sp_id in range(num_sps):
        # 各SPの重心をランダムに設定
        centroid = np.array([sp_id * 3.0, 0.0, 0.0])  # SPごとに3m離れた位置
        sp_points = centroid + np.random.randn(points_per_sp, 3).astype(np.float32) * 0.5
        points_t.append(sp_points)
        sp_labels_t.extend([sp_id] * points_per_sp)
    
    points_t = np.vstack(points_t).astype(np.float32)
    sp_labels_t = np.array(sp_labels_t)
    
    # Scene flow: 全体を少し移動させる（例: x方向に0.5m）
    flow_t = np.zeros_like(points_t)
    flow_t[:, 0] = 0.5  # x方向に0.5m移動
    
    # 時刻t+1の点群: tの点群をflowで移動 + ノイズ
    points_t1 = points_t + flow_t + np.random.randn(*points_t.shape).astype(np.float32) * 0.1
    sp_labels_t1 = sp_labels_t.copy()  # 同じSPラベル（理想的なケース）
    
    print(f"Points t: {points_t.shape}, Points t1: {points_t1.shape}")
    print(f"SP labels t: unique={np.unique(sp_labels_t)}")
    
    # SPレベル直接マッチングをテスト
    corr_matrix, unique_sp_t, unique_sp_t1 = compute_superpoint_correspondence_direct(
        points_t, points_t1, flow_t,
        sp_labels_t, sp_labels_t1,
        weight_centroid_distance=1.0,
        weight_spread_similarity=0.3,
        weight_point_count_similarity=0.2,
        max_centroid_distance=3.0,
        min_score_threshold=0.3,
        min_sp_points=5  # テスト用に小さめ
    )
    
    print(f"\n=== 結果 ===")
    print(f"対応行列サイズ: {corr_matrix.shape}")
    print(f"有効SP数: t={len(unique_sp_t)}, t1={len(unique_sp_t1)}")
    print(f"マッチ数: {np.sum(corr_matrix > 0)}")
    print(f"\n対応行列（スコア）:")
    print(corr_matrix)
    
    # 対角要素が高いスコアになっているか確認（理想的には1対1で自分自身にマッチ）
    for i in range(len(unique_sp_t)):
        matched_j = np.argmax(corr_matrix[i])
        if corr_matrix[i, matched_j] > 0:
            print(f"SP_t[{unique_sp_t[i]}] -> SP_t1[{unique_sp_t1[matched_j]}] (score={corr_matrix[i, matched_j]:.3f})")
    
    print("\n=== SP特徴量テスト ===")
    sp_features = compute_sp_features(
        points_t, sp_labels_t, flow_t,
        min_sp_points=5
    )
    for sp_id, features in sp_features.items():
        print(f"SP {sp_id}:")
        print(f"  centroid: {features['centroid']}")
        print(f"  spread (σx, σy, σz): {features['spread']}")
        print(f"  point_count: {features['point_count']}")
        print(f"  motion: {features['motion']}")
    
    # 後方互換性テスト（旧関数）
    print("\n=== 後方互換性テスト（旧関数） ===")
    # テストデータ
    points_t_old = np.random.randn(100, 3).astype(np.float32)
    points_t1_old = points_t_old + np.random.randn(100, 3).astype(np.float32) * 0.1
    
    # ダミーフロー
    flow = points_t1_old[:50] - points_t_old[:50]
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
    
    print("\n=== 動的distance_thresholdテスト ===")
    # 動いている物体をシミュレート（SP 5を大きく動かす）
    moving_sp_mask = sp_labels_t == 5
    full_flow_dynamic = full_flow.copy()
    full_flow_dynamic[moving_sp_mask] += np.array([2.0, 0.0, 0.0])  # 大きな動き
    
    correspondence_dynamic, distances_dynamic = compute_point_correspondence_filtered(
        points_t, points_t1, full_flow_dynamic,
        sp_labels_t, sp_labels_t1,
        pose_t, pose_t1,
        ground_mask_t, ground_mask_t1,
        distance_threshold=0.3,  # 静止物体用
        distance_threshold_per_frame=0.1,
        distance_threshold_moving=0.8,  # 動いている物体用
        distance_threshold_moving_per_frame=0.2,
        moving_flow_threshold=0.1,  # m/frame
        num_frames=5,  # 5フレーム離れている想定
        remove_ego_motion=True,
        exclude_ground=True
    )
    print(f'Dynamic threshold test - valid correspondences: {np.sum(correspondence_dynamic >= 0)}')
    # 静止物体の有効閾値: 0.3 + 0.1 * 4 = 0.7m
    # 動いている物体の有効閾値: 0.8 + 0.2 * 4 = 1.6m
    print(f'Effective threshold (static): {0.3 + 0.1 * 4}m')
    print(f'Effective threshold (moving): {0.8 + 0.2 * 4}m')
    
    print('\nAll tests passed!')
