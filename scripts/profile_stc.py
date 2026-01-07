#!/usr/bin/env python3
"""STC処理のボトルネック特定用プロファイリングスクリプト"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from lib.config import TCUSSConfig
from datasets.SemanticKITTI import KITTIstc, KITTItcuss_stc, _global_to_seq_local

def profile_getitem(config, num_samples=5):
    """__getitem__の各ステップの時間を計測"""
    
    print("=" * 60)
    print("STC DataLoader __getitem__ プロファイリング")
    print("=" * 60)
    
    # KITTIstcを初期化
    dataset = KITTIstc(config)
    
    # テスト用にシーンペアを設定（連続フレーム）
    test_indices_t = list(range(100, 100 + num_samples))
    test_indices_t2 = list(range(101, 101 + num_samples))
    dataset.set_scene_pairs(test_indices_t, test_indices_t2)
    
    print(f"\nサンプル数: {num_samples}")
    
    # 時間計測用
    times = {
        'get_item_one_scene_t': [],
        'get_item_one_scene_t2': [],
        'compute_sp_correspondence': [],
        'data_augmentation': [],
        'total': []
    }
    
    for i in range(num_samples):
        seq_t, idx_t = dataset.scene_locates[i]
        seq_t2, idx_t2 = dataset.scene_diff_locates[i]
        
        print(f"\n--- Sample {i}: seq={seq_t}, idx_t={idx_t}, idx_t2={idx_t2} ---")
        
        # 全体の時間
        t_start = time.time()
        
        # 1. _get_item_one_scene (t)
        t0 = time.time()
        coords_t, coords_t_original, sp_labels_t, pose_t, flow_t, ground_mask_t = dataset._get_item_one_scene(seq_t, idx_t)
        t1 = time.time()
        times['get_item_one_scene_t'].append(t1 - t0)
        print(f"  _get_item_one_scene (t): {t1-t0:.3f}s, points={len(coords_t)}, unique_sps={len(np.unique(sp_labels_t[sp_labels_t >= 0]))}")
        
        # 2. _get_item_one_scene (t2)
        t0 = time.time()
        coords_t2, coords_t2_original, sp_labels_t2, pose_t2, _, ground_mask_t2 = dataset._get_item_one_scene(seq_t2, idx_t2)
        t1 = time.time()
        times['get_item_one_scene_t2'].append(t1 - t0)
        print(f"  _get_item_one_scene (t2): {t1-t0:.3f}s, points={len(coords_t2)}, unique_sps={len(np.unique(sp_labels_t2[sp_labels_t2 >= 0]))}")
        
        # 3. SP対応計算
        t0 = time.time()
        corr_matrix, unique_sp_t, unique_sp_t2 = dataset._compute_sp_correspondence_direct(
            coords_t_original,
            coords_t2_original,
            flow_t,
            sp_labels_t,
            sp_labels_t2,
            pose_t,
            pose_t2,
            ground_mask_t,
            ground_mask_t2,
            weight_centroid_distance=dataset.weight_centroid_distance,
            weight_spread_similarity=dataset.weight_spread_similarity,
            weight_point_count_similarity=dataset.weight_point_count_similarity,
            max_centroid_distance=dataset.max_centroid_distance,
            min_score_threshold=dataset.min_score_threshold,
            min_sp_points=dataset.min_sp_points,
            remove_ego_motion=dataset.remove_ego_motion,
            exclude_ground=dataset.exclude_ground
        )
        t1 = time.time()
        times['compute_sp_correspondence'].append(t1 - t0)
        print(f"  compute_sp_correspondence: {t1-t0:.3f}s, corr_matrix={corr_matrix.shape}, matches={np.sum(corr_matrix > 0)}")
        
        # 4. データ拡張
        t0 = time.time()
        coords_t_aug = dataset.augs(coords_t.copy())
        coords_t2_aug = dataset.augs(coords_t2.copy())
        t1 = time.time()
        times['data_augmentation'].append(t1 - t0)
        print(f"  data_augmentation: {t1-t0:.3f}s")
        
        t_end = time.time()
        times['total'].append(t_end - t_start)
        print(f"  TOTAL: {t_end - t_start:.3f}s")
    
    # サマリー
    print("\n" + "=" * 60)
    print("サマリー (平均時間)")
    print("=" * 60)
    for key, values in times.items():
        avg = np.mean(values)
        std = np.std(values)
        print(f"  {key}: {avg:.3f}s ± {std:.3f}s")
    
    total_avg = np.mean(times['total'])
    print(f"\n1サンプルあたり平均: {total_avg:.3f}s")
    print(f"バッチサイズ8の場合の推定時間: {total_avg * 8:.3f}s")
    
    return times


def profile_sp_matching_detail(config, num_samples=3):
    """SP対応計算の内部の時間を詳細に計測"""
    
    print("\n" + "=" * 60)
    print("SP対応計算 内部詳細プロファイリング")
    print("=" * 60)
    
    from scene_flow.correspondence import (
        compute_sp_features,
        compute_sp_matching_score_matrix,
        greedy_sp_matching,
        compute_object_flow
    )
    
    dataset = KITTIstc(config)
    test_indices_t = list(range(100, 100 + num_samples))
    test_indices_t2 = list(range(101, 101 + num_samples))
    dataset.set_scene_pairs(test_indices_t, test_indices_t2)
    
    times = {
        'compute_sp_features_t': [],
        'compute_sp_features_t1': [],
        'compute_score_matrix': [],
        'greedy_matching': [],
    }
    
    for i in range(num_samples):
        seq_t, idx_t = dataset.scene_locates[i]
        seq_t2, idx_t2 = dataset.scene_diff_locates[i]
        
        coords_t, coords_t_original, sp_labels_t, pose_t, flow_t, ground_mask_t = dataset._get_item_one_scene(seq_t, idx_t)
        coords_t2, coords_t2_original, sp_labels_t2, pose_t2, _, ground_mask_t2 = dataset._get_item_one_scene(seq_t2, idx_t2)
        
        print(f"\n--- Sample {i}: SPs_t={len(np.unique(sp_labels_t[sp_labels_t >= 0]))}, SPs_t2={len(np.unique(sp_labels_t2[sp_labels_t2 >= 0]))} ---")
        
        # 1. compute_sp_features (t)
        t0 = time.time()
        sp_features_t = compute_sp_features(
            coords_t_original, sp_labels_t, flow_t,
            ground_mask_t, pose_t, pose_t2,
            dataset.min_sp_points, dataset.exclude_ground, dataset.remove_ego_motion
        )
        t1 = time.time()
        times['compute_sp_features_t'].append(t1 - t0)
        print(f"  compute_sp_features (t): {t1-t0:.3f}s, valid_sps={len(sp_features_t)}")
        
        # 2. compute_sp_features (t1)
        t0 = time.time()
        sp_features_t1 = compute_sp_features(
            coords_t2_original, sp_labels_t2, np.zeros_like(coords_t2_original),
            ground_mask_t2, None, None,
            dataset.min_sp_points, dataset.exclude_ground, False
        )
        t1 = time.time()
        times['compute_sp_features_t1'].append(t1 - t0)
        print(f"  compute_sp_features (t1): {t1-t0:.3f}s, valid_sps={len(sp_features_t1)}")
        
        # 3. compute_score_matrix
        t0 = time.time()
        score_matrix, sp_ids_t, sp_ids_t1 = compute_sp_matching_score_matrix(
            sp_features_t, sp_features_t1,
            dataset.weight_centroid_distance,
            dataset.weight_spread_similarity,
            dataset.weight_point_count_similarity,
            dataset.max_centroid_distance
        )
        t1 = time.time()
        times['compute_score_matrix'].append(t1 - t0)
        print(f"  compute_score_matrix: {t1-t0:.3f}s, matrix_size={score_matrix.shape}")
        
        # 4. greedy_matching
        t0 = time.time()
        matched_t, matched_t1, matched_scores = greedy_sp_matching(
            score_matrix, dataset.min_score_threshold
        )
        t1 = time.time()
        times['greedy_matching'].append(t1 - t0)
        print(f"  greedy_matching: {t1-t0:.3f}s, matches={len(matched_t)}")
    
    # サマリー
    print("\n" + "=" * 60)
    print("SP対応計算 内部サマリー (平均時間)")
    print("=" * 60)
    for key, values in times.items():
        avg = np.mean(values)
        print(f"  {key}: {avg:.3f}s")


def profile_with_real_dataset(config, num_samples=10):
    """実際のKITTItcuss_stcデータセットでプロファイリング"""
    
    print("=" * 60)
    print("実際のKITTItcuss_stcでプロファイリング")
    print("=" * 60)
    
    # データセット作成
    dataset = KITTItcuss_stc(config)
    
    # プロファイリング有効化
    dataset.kittistc._profile_enabled = True
    
    # シーンペア設定（クラスタリング後に呼ばれるはず）
    # 手動で設定
    from datasets.SemanticKITTI import get_stc_scene_pairs
    scene_idx_t1, scene_idx_t2 = get_stc_scene_pairs(
        dataset.kittistc.file,
        config.select_num,
        config.scan_window
    )
    dataset.kittistc.set_scene_pairs(scene_idx_t1, scene_idx_t2)
    
    print(f"シーンペア数: {len(dataset.kittistc)}")
    
    times = []
    for i in range(min(num_samples, len(dataset.kittistc))):
        print(f"\n--- Sample {i} ---")
        t0 = time.time()
        try:
            data = dataset.kittistc[i]
            t1 = time.time()
            times.append(t1 - t0)
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    if times:
        print(f"\n=== サマリー ===")
        print(f"平均時間: {np.mean(times):.3f}s")
        print(f"バッチサイズ8の推定: {np.mean(times) * 8:.3f}s")


if __name__ == "__main__":
    config = TCUSSConfig.from_yaml('config/stc.yaml')
    
    # 実際のデータセットでプロファイリング
    profile_with_real_dataset(config, num_samples=10)

