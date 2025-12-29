#!/usr/bin/env python
"""
SemanticKITTI init SP可視化ツール

SemanticKITTIの点群データからinit SP（初期スーパーポイント）を計算し、
各SPを色分けしたPLYファイルとして出力します。

使用方法:
    python visualize_initSP_SemanticKITTI.py --data_path /path/to/sequences --output_path ./output
    python visualize_initSP_SemanticKITTI.py --seq_id 00 --frame_id 000000  # 特定フレームのみ
    python visualize_initSP_SemanticKITTI.py --seq_id 00  # 特定シーケンスの全フレーム
"""

import open3d as o3d
import numpy as np
from pathlib import Path
import argparse
import os
import sys
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import colorsys

# プロジェクトルートをパスに追加
BASE_DIR = Path(__file__).parent.absolute()
ROOT_DIR = BASE_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(BASE_DIR))

from lib.helper_ply import write_ply


def generate_distinct_colors(n_colors: int, seed: int = 42) -> np.ndarray:
    """
    視覚的に区別しやすい色を生成
    
    Args:
        n_colors: 必要な色の数
        seed: 乱数シード
    
    Returns:
        (n_colors, 3) のRGB配列 (0-255)
    """
    np.random.seed(seed)
    colors = []
    
    # HSVカラースペースで均等に分布した色を生成
    for i in range(n_colors):
        hue = (i * 0.618033988749895) % 1.0  # 黄金角を使用して均等に分布
        saturation = 0.7 + np.random.random() * 0.3  # 0.7-1.0
        value = 0.7 + np.random.random() * 0.3  # 0.7-1.0
        
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append([int(c * 255) for c in rgb])
    
    return np.array(colors, dtype=np.uint8)


def load_velodyne_bin(bin_path: str) -> np.ndarray:
    """
    SemanticKITTI velodyne binファイルを読み込む
    
    Returns:
        (N, 4) の配列 [x, y, z, remission]
    """
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points


def compute_init_sp(
    coords: np.ndarray,
    distance_threshold: float = 0.1,
    dbscan_eps: float = 0.2,
    dbscan_min_points: int = 1
) -> np.ndarray:
    """
    init SP（初期スーパーポイント）を計算
    
    処理内容:
    1. RANSACで地面平面を検出
    2. 非地面点をDBSCANでクラスタリング
    3. 地面点は1つのSPとして扱う
    
    Args:
        coords: (N, 3) 点群座標
        distance_threshold: RANSACの距離閾値
        dbscan_eps: DBSCANのeps
        dbscan_min_points: DBSCANの最小点数
    
    Returns:
        (N,) SPラベル配列
    """
    # Open3D点群を作成
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    
    # RANSAC で地面平面を検出
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=1000
    )
    road_index = np.array(inliers)
    
    # 非地面点のインデックス
    all_indices = np.arange(coords.shape[0])
    other_index = np.setdiff1d(all_indices, road_index)
    
    # 非地面点をDBSCANでクラスタリング
    if len(other_index) > 0:
        other_coords = coords[other_index]
        pcd_other = o3d.geometry.PointCloud()
        pcd_other.points = o3d.utility.Vector3dVector(other_coords)
        other_region_idx = np.array(pcd_other.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_points))
    else:
        other_region_idx = np.array([], dtype=np.int64)
    
    # SPラベルを構築
    sp_labels = -np.ones(coords.shape[0], dtype=np.int64)
    if len(other_index) > 0:
        sp_labels[other_index] = other_region_idx
    
    # 地面点は最後のSPラベルとして追加
    if len(road_index) > 0:
        max_label = other_region_idx.max() if len(other_region_idx) > 0 else -1
        sp_labels[road_index] = max_label + 1
    
    return sp_labels


def process_single_frame(
    bin_path: str,
    output_path: str,
    distance_threshold: float = 0.1,
    dbscan_eps: float = 0.2,
    dbscan_min_points: int = 1,
    max_colors: int = 50000
) -> dict:
    """
    単一フレームを処理してPLYファイルを出力
    
    Returns:
        処理結果の統計情報
    """
    bin_path = Path(bin_path)
    output_path = Path(output_path)
    
    # 点群を読み込み
    points = load_velodyne_bin(str(bin_path))
    coords = points[:, :3].astype(np.float32)
    
    # 座標を中心化（オプション）
    coords_centered = coords - coords.mean(axis=0)
    
    # init SPを計算
    sp_labels = compute_init_sp(
        coords_centered,
        distance_threshold=distance_threshold,
        dbscan_eps=dbscan_eps,
        dbscan_min_points=dbscan_min_points
    )
    
    # 統計情報
    unique_labels = np.unique(sp_labels)
    num_sp = len(unique_labels[unique_labels >= 0])
    
    # 色を生成
    colors = generate_distinct_colors(max(num_sp, max_colors))
    
    # 各点に色を割り当て
    point_colors = np.zeros((coords.shape[0], 3), dtype=np.uint8)
    for i, label in enumerate(sp_labels):
        if label >= 0:
            point_colors[i] = colors[label % len(colors)]
        else:
            point_colors[i] = [128, 128, 128]  # 未割当点はグレー
    
    # 出力ディレクトリを作成
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # PLYファイルを出力（元の座標を使用）
    write_ply(
        str(output_path),
        [coords, point_colors],
        ['x', 'y', 'z', 'red', 'green', 'blue']
    )
    
    return {
        'input': str(bin_path),
        'output': str(output_path),
        'num_points': coords.shape[0],
        'num_sp': num_sp,
    }


def process_sequence(
    data_path: str,
    output_path: str,
    seq_id: str,
    frame_id: str = None,
    distance_threshold: float = 0.1,
    dbscan_eps: float = 0.2,
    dbscan_min_points: int = 1,
    max_workers: int = 8
) -> list:
    """
    シーケンス（または特定フレーム）を処理
    
    Args:
        data_path: SemanticKITTIのsequencesディレクトリ
        output_path: 出力ディレクトリ
        seq_id: シーケンスID (e.g., "00")
        frame_id: フレームID (e.g., "000000")、Noneなら全フレーム
        distance_threshold: RANSACの距離閾値
        dbscan_eps: DBSCANのeps
        dbscan_min_points: DBSCANの最小点数
        max_workers: 並列処理のワーカー数
    
    Returns:
        処理結果のリスト
    """
    data_path = Path(data_path)
    output_path = Path(output_path)
    
    velodyne_dir = data_path / seq_id / 'velodyne'
    
    if not velodyne_dir.exists():
        raise ValueError(f"Velodyne directory not found: {velodyne_dir}")
    
    # 処理対象のフレームを取得
    if frame_id is not None:
        bin_files = [velodyne_dir / f"{frame_id}.bin"]
        if not bin_files[0].exists():
            raise ValueError(f"Frame not found: {bin_files[0]}")
    else:
        bin_files = sorted(velodyne_dir.glob("*.bin"))
    
    print(f"Processing sequence {seq_id}: {len(bin_files)} frames")
    
    results = []
    
    if max_workers <= 1 or len(bin_files) == 1:
        # シングルスレッド処理
        for bin_file in tqdm(bin_files, desc=f"Seq {seq_id}"):
            output_file = output_path / seq_id / f"{bin_file.stem}_initsp.ply"
            result = process_single_frame(
                str(bin_file),
                str(output_file),
                distance_threshold=distance_threshold,
                dbscan_eps=dbscan_eps,
                dbscan_min_points=dbscan_min_points
            )
            results.append(result)
    else:
        # 並列処理
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for bin_file in bin_files:
                output_file = output_path / seq_id / f"{bin_file.stem}_initsp.ply"
                future = executor.submit(
                    process_single_frame,
                    str(bin_file),
                    str(output_file),
                    distance_threshold,
                    dbscan_eps,
                    dbscan_min_points
                )
                futures[future] = bin_file
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Seq {seq_id}"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    bin_file = futures[future]
                    print(f"Error processing {bin_file}: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='SemanticKITTI init SPを可視化（PLYファイル出力）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
    # 全シーケンス処理
    python visualize_initSP_SemanticKITTI.py --data_path /path/to/sequences --output_path ./output

    # 特定シーケンスのみ
    python visualize_initSP_SemanticKITTI.py --data_path /path/to/sequences --seq_id 00

    # 特定フレームのみ
    python visualize_initSP_SemanticKITTI.py --data_path /path/to/sequences --seq_id 00 --frame_id 000000
        """
    )
    parser.add_argument('--data_path', type=str, 
                        default='data/dataset/semantickitti/dataset/sequences',
                        help='SemanticKITTI sequences directory')
    parser.add_argument('--output_path', type=str,
                        default='data/initSP_visualization',
                        help='Output directory for PLY files')
    parser.add_argument('--seq_id', type=str, default=None,
                        help='Specific sequence ID to process (e.g., "00")')
    parser.add_argument('--frame_id', type=str, default=None,
                        help='Specific frame ID to process (e.g., "000000")')
    parser.add_argument('--distance_threshold', type=float, default=0.1,
                        help='RANSAC distance threshold')
    parser.add_argument('--dbscan_eps', type=float, default=0.2,
                        help='DBSCAN eps parameter')
    parser.add_argument('--dbscan_min_points', type=int, default=1,
                        help='DBSCAN min_points parameter')
    parser.add_argument('--max_workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--train_only', action='store_true',
                        help='Process only training sequences (00-10)')
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = ROOT_DIR / data_path
    
    output_path = Path(args.output_path)
    if not output_path.is_absolute():
        output_path = ROOT_DIR / output_path
    
    print(f"Data path: {data_path}")
    print(f"Output path: {output_path}")
    
    if not data_path.exists():
        print(f"Error: Data path not found: {data_path}")
        sys.exit(1)
    
    # 処理対象シーケンスを決定
    if args.seq_id is not None:
        seq_list = [args.seq_id]
    else:
        seq_list = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
        if args.train_only:
            seq_list = [s for s in seq_list if int(s) <= 10]
    
    print(f"Sequences to process: {seq_list}")
    
    # 各シーケンスを処理
    all_results = []
    for seq_id in seq_list:
        velodyne_dir = data_path / seq_id / 'velodyne'
        if not velodyne_dir.exists():
            print(f"Skipping sequence {seq_id}: velodyne directory not found")
            continue
        
        try:
            results = process_sequence(
                str(data_path),
                str(output_path),
                seq_id,
                frame_id=args.frame_id,
                distance_threshold=args.distance_threshold,
                dbscan_eps=args.dbscan_eps,
                dbscan_min_points=args.dbscan_min_points,
                max_workers=args.max_workers
            )
            all_results.extend(results)
        except Exception as e:
            print(f"Error processing sequence {seq_id}: {e}")
            continue
    
    # 統計情報を出力
    if all_results:
        total_points = sum(r['num_points'] for r in all_results)
        total_sp = sum(r['num_sp'] for r in all_results)
        avg_sp = total_sp / len(all_results)
        
        print(f"\n=== Summary ===")
        print(f"Processed frames: {len(all_results)}")
        print(f"Total points: {total_points:,}")
        print(f"Total superpoints: {total_sp:,}")
        print(f"Average superpoints per frame: {avg_sp:.1f}")
        print(f"Output directory: {output_path}")


if __name__ == '__main__':
    main()




