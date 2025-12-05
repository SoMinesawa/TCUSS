#!/usr/bin/env python3
"""
TCUSS可視化用データ生成スクリプト

KITTItemporalクラスと同様の処理を行い、連続する12フレームを処理して
自動生成されたセグメンテーションラベルを可視化用に保存する。
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import argparse
import numpy as np
import torch
import hdbscan
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional, Union

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.helper_ply import read_ply, write_ply
from lib.config import TCUSSConfig
import MinkowskiEngine as ME


class VisualizationDataGenerator:
    """可視化用データ生成クラス"""
    
    def __init__(self, config: TCUSSConfig):
        self.config = config
        self.seq_to_scan_num = {0: 4541, 1: 1101, 2: 4661, 3: 801, 4: 271, 5: 2761, 6: 1101, 7: 1101, 9: 1591, 10: 1201}
        self.debug_mode = False  # デバッグモード
        
        # 出力ディレクトリの設定
        self.output_dir = "data/users/minesawa/semantickitti/vis"
        
        # ファイルリストの構築
        self.file_list = []
        seq_list = np.sort(os.listdir(self.config.data_path))
        for seq_id in seq_list:
            seq_path = os.path.join(self.config.data_path, seq_id)
            if seq_id in ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']:
                for f in np.sort(os.listdir(seq_path)):
                    self.file_list.append(os.path.join(seq_path, f))
    
    def generate_all_sequences(self):
        """すべてのシーケンスの可視化データを生成"""
        for seq in self.seq_to_scan_num.keys():
            print(f"Processing sequence {seq:02d}...")
            self._process_sequence(seq)
    
    def _process_sequence(self, seq: int):
        """指定されたシーケンスを12フレームずつ処理"""
        scan_num = self.seq_to_scan_num[seq]
        scan_window = self.config.scan_window
        
        # frame 0~11, frame 12~23, frame 24~35... のように処理
        window_count = 0
        for start_idx in tqdm(range(0, scan_num, scan_window), desc=f"Seq {seq:02d}"):
            end_idx = min(start_idx + scan_window, scan_num)
            if end_idx - start_idx < scan_window:
                # 最後のウィンドウが12フレームに満たない場合はスキップ
                continue
            
            # デバッグモードでは最初の3ウィンドウのみ処理
            if self.debug_mode and window_count >= 3:
                print(f"デバッグモード: 3ウィンドウ処理完了、シーケンス {seq:02d} をスキップします")
                break
            
            # 12フレームの処理
            scene_idx_in_window = [(seq, start_idx + i) for i in range(scan_window)]
            self._process_window(scene_idx_in_window)
            
            window_count += 1
    
    def _process_window(self, scene_idx_in_window: List[Tuple[int, int]]):
        """12フレームのウィンドウを処理してセグメンテーション結果を保存"""
        # 各フレームのデータをロード
        coords_list, labels_list, unique_map_list, mask_list = [], [], [], []
        
        for seq, idx in scene_idx_in_window:
            coords, labels, unique_map, mask = self._load_and_preprocess_frame(seq, idx)
            coords_list.append(coords)
            labels_list.append(labels)
            unique_map_list.append(unique_map)
            mask_list.append(mask)
        
        # 点群を集約
        agg_coords, agg_ground_labels, elements_nums = self._aggregate_pcds(
            scene_idx_in_window, coords_list, unique_map_list, mask_list, labels_list
        )
        
        # クラスタリング実行
        agg_segs = self._clusterize_pcds(agg_coords, agg_ground_labels)
        
        # 集約データの保存
        seq = scene_idx_in_window[0][0]  # 全フレームが同じシーケンス
        window_start_idx = scene_idx_in_window[0][1]  # ウィンドウの開始インデックス
        self._save_aggregated_data(seq, window_start_idx, agg_coords, agg_segs)
        
        # 各フレームのセグメントを抽出して保存
        start_idx = 0
        for i, (elements_num, (seq, idx)) in enumerate(zip(elements_nums, scene_idx_in_window)):
            frame_segs = agg_segs[start_idx:start_idx + elements_num]
            self._save_frame_data(seq, idx, coords_list[i], frame_segs)
            start_idx += elements_num
    
    def _load_and_preprocess_frame(self, seq: int, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """1フレームのデータをロードして前処理を行う"""
        # PLYファイル読み込み
        ply_path = self._tuple_to_path((seq, idx))
        data = read_ply(ply_path)
        coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
        feats = np.array(data['remission'])[:, np.newaxis]
        labels = np.array(data['class'])
        
        # 座標の正規化
        coords_original = coords.copy()
        means = coords.mean(0)
        coords -= means
        
        # Voxelization
        coords, feats, labels, unique_map, inverse_map = self._voxelize(coords, feats, labels)
        coords = coords.astype(np.float32)
        
        # Cropping
        mask = np.sqrt(((coords * self.config.voxel_size) ** 2).sum(-1)) < self.config.r_crop
        coords, feats, labels = coords[mask], feats[mask], labels[mask]
        
        # 元の座標系に戻す
        coords = coords_original[unique_map][mask]
        
        return coords, labels, unique_map, mask
    
    def _voxelize(self, coords, feats, labels) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Voxelization処理"""
        scale = 1 / self.config.voxel_size
        coords = np.floor(coords * scale)
        coords, feats, labels, unique_map, inverse_map = ME.utils.sparse_quantize(
            np.ascontiguousarray(coords), feats, labels=labels, ignore_label=-1, return_index=True, return_inverse=True
        )
        return coords, feats, labels, unique_map, inverse_map
    
    def _aggregate_pcds(self, scene_idx_in_window: List[Tuple[int, int]], 
                       coords_list: List[np.ndarray], unique_map_list: List[np.ndarray], 
                       mask_list: List[np.ndarray], labels_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """複数フレームの点群を集約"""
        poses = self._load_poses(scene_idx_in_window[0][0])
        points_set = np.empty((0, 3))
        ground_label = np.empty((0, 1))
        element_nums = []
        
        for (seq, idx), coords, unique_map, mask, labels in zip(scene_idx_in_window, coords_list, unique_map_list, mask_list, labels_list):
            pose = poses[idx]
            coords = self._apply_transform(coords, pose)
            
            # Ground label (Patchwork)
            g_set = np.fromfile(self._tuple_to_patchwork_path((seq, idx)), dtype=np.uint32)
            g_set = g_set[unique_map]
            g_set = g_set[mask]
            g_set = g_set.reshape((-1))[:, np.newaxis]
            
            points_set = np.vstack((points_set, coords))
            ground_label = np.vstack((ground_label, g_set))
            element_nums.append(coords.shape[0])
        
        # 最後のフレームの座標系に変換
        last_pose = poses[scene_idx_in_window[-1][1]]
        points_set = self._undo_transform(points_set, last_pose)
        
        return points_set, ground_label, element_nums
    
    def _clusterize_pcds(self, agg_coords: np.ndarray, agg_ground_labels: np.ndarray) -> np.ndarray:
        """HDBSCANを用いてクラスタリング"""
        # 地面マスクの作成
        mask_ground = agg_ground_labels == 1
        mask_ground = mask_ground.flatten()
        non_ground_coords = agg_coords[~mask_ground]
        
        if len(non_ground_coords) > 0:
            # HDBSCANクラスタリング
            clusterer = hdbscan.HDBSCAN(
                algorithm='best',
                metric='euclidean',
                min_cluster_size=100,
                min_samples=20,
                alpha=1.0,
                cluster_selection_epsilon=0.8, # こいつ重要かも 0.6だとちょっと分けすぎで、0.8だと訳なさすぎる
                cluster_selection_method='eom',
                leaf_size=100,
                approx_min_span_tree=True,
                gen_min_span_tree=True
            )
            # default
            # clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1., approx_min_span_tree=True,　
            #     gen_min_span_tree=True, leaf_size=100,
            #     metric='euclidean', min_cluster_size=20, min_samples=None
            # )
            labels = clusterer.fit_predict(non_ground_coords)
            
            # デバッグ情報の出力
            if hasattr(self, 'debug_mode') and self.debug_mode:
                noise_ratio = (labels == -1).mean() if len(labels) > 0 else 0
                unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
                n_clusters = len(unique_labels) if len(unique_labels) > 0 else 0
                max_cluster_size = counts.max() if len(counts) > 0 else 0
                min_cluster_size = counts.min() if len(counts) > 0 else 0
                
                print(f"  HDBSCAN結果: クラスタ数={n_clusters}, ノイズ率={noise_ratio:.1%}, "
                      f"最大クラスタ={max_cluster_size}点, 最小クラスタ={min_cluster_size}点")
            
            # ノイズポイント（-1ラベル）を適切に処理
            if (labels == -1).any():
                max_valid_label = labels[labels >= 0].max() if (labels >= 0).any() else -1
                labels[labels == -1] = max_valid_label + 1
            
            dynamic_ground_label = labels.max() + 1 if len(labels) > 0 else 0
        else:
            labels = np.array([])
            dynamic_ground_label = 0
        
        # セグメンテーション配列の作成
        agg_segs = np.zeros_like(agg_ground_labels).flatten()
        agg_segs[~mask_ground] = labels
        agg_segs[mask_ground] = dynamic_ground_label
        
        # デバッグ情報の出力（最終結果）
        if hasattr(self, 'debug_mode') and self.debug_mode:
            unique_final_labels, final_counts = np.unique(agg_segs, return_counts=True)
            ground_points = final_counts[unique_final_labels == dynamic_ground_label][0] if dynamic_ground_label in unique_final_labels else 0
            print(f"  最終結果: 総セグメント数={len(unique_final_labels)}, 地面点数={ground_points}, 総点数={len(agg_segs)}")
        
        return agg_segs.astype(np.int64)
    
    def _save_frame_data(self, seq: int, idx: int, coords: np.ndarray, segments: np.ndarray):
        """フレームのデータを指定された形式で保存"""
        # 出力ディレクトリの作成
        seq_dir = os.path.join(self.output_dir, "sequences", f"{seq:02d}")
        velodyne_dir = os.path.join(seq_dir, "velodyne")
        labels_dir = os.path.join(seq_dir, "labels")
        
        os.makedirs(velodyne_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # 座標データの保存（velodyneディレクトリ）
        coords_path = os.path.join(velodyne_dir, f"{idx:06d}.ply")
        write_ply(coords_path, [coords], ['x', 'y', 'z'])
        
        # ラベルデータの保存（labelsディレクトリ）
        labels_path = os.path.join(labels_dir, f"{idx:06d}.ply")
        # ラベル情報のみを保存
        write_ply(labels_path, [segments.reshape(-1, 1)], ['label'])
    
    def _save_aggregated_data(self, seq: int, window_start_idx: int, agg_coords: np.ndarray, agg_segs: np.ndarray):
        """集約されたデータを保存"""
        # 出力ディレクトリの作成
        seq_dir = os.path.join(self.output_dir, "sequences", f"{seq:02d}")
        agg_coords_dir = os.path.join(seq_dir, "agg_coordinates")
        agg_segs_dir = os.path.join(seq_dir, "agg_segments")
        
        os.makedirs(agg_coords_dir, exist_ok=True)
        os.makedirs(agg_segs_dir, exist_ok=True)
        
        # 集約座標データの保存
        agg_coords_path = os.path.join(agg_coords_dir, f"{window_start_idx:06d}-{window_start_idx+11:06d}.ply")
        write_ply(agg_coords_path, [agg_coords], ['x', 'y', 'z'])
        
        # 集約セグメンテーションデータの保存
        agg_segs_path = os.path.join(agg_segs_dir, f"{window_start_idx:06d}-{window_start_idx+11:06d}.ply")
        write_ply(agg_segs_path, [agg_segs.reshape(-1, 1).astype(np.int32)], ['segment'])
    
    def _load_poses(self, seq: int) -> np.ndarray:
        """ポーズファイルをロード"""
        calib_fname = self._seq_to_calib_path(seq)
        poses_fname = self._seq_to_poses_path(seq)
        calibration = self._parse_calibration(calib_fname)
        poses_file = open(poses_fname)

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        poses = []
        for line in poses_file:
            values = [float(v) for v in line.strip().split()]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses
    
    def _apply_transform(self, points: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """ポーズ変換を適用"""
        hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
        return np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)[:, :3]

    def _undo_transform(self, points: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """ポーズ変換を元に戻す"""
        hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
        return np.sum(np.expand_dims(hpoints, 2) * np.linalg.inv(pose).T, axis=1)[:, :3]
    
    def _parse_calibration(self, filename: str) -> Dict:
        """キャリブレーションファイルを解析"""
        calib = {}
        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            calib[key] = pose
        calib_file.close()
        return calib
    
    # パス生成関数
    def _tuple_to_path(self, tup: Tuple[int, int]) -> str:
        return os.path.join(self.config.data_path, str(tup[0]).zfill(2), str(tup[1]).zfill(6) + '.ply')

    def _tuple_to_patchwork_path(self, tup: Tuple[int, int]) -> str:
        return os.path.join(self.config.patchwork_path, str(tup[0]).zfill(2), str(tup[1]).zfill(6) + '.label')
    
    def _seq_to_calib_path(self, seq: int) -> str:
        return os.path.join(self.config.original_data_path, str(seq).zfill(2), 'calib.txt')
    
    def _seq_to_poses_path(self, seq: int) -> str:
        return os.path.join(self.config.original_data_path, str(seq).zfill(2), 'poses.txt')


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='TCUSS可視化用データ生成')
    
    # 基本的な設定のみを引数として受け取る
    parser.add_argument('--data_path', type=str, default='data/users/minesawa/semantickitti/growsp',
                        help='点群データパス')
    parser.add_argument('--original_data_path', type=str, default='data/dataset/semantickitti/dataset/sequences',
                        help='オリジナルデータパス')
    parser.add_argument('--patchwork_path', type=str, default='data/users/minesawa/semantickitti/patchwork',
                        help='パッチワークデータパス')
    parser.add_argument('--voxel_size', type=float, default=0.15, help='ボクセルサイズ')
    parser.add_argument('--r_crop', type=float, default=50.0, help='クロッピング半径')
    parser.add_argument('--scan_window', type=int, default=12, help='スキャンウィンドウサイズ')
    parser.add_argument('--sequences', type=str, nargs='+', default=None,
                        help='処理するシーケンス番号（例: --sequences 00 01 02）')
    parser.add_argument('--debug', action='store_true', help='デバッグモード（最初の3ウィンドウのみ処理）')
    
    args = parser.parse_args()
    
    # 設定オブジェクトの作成（最小限の設定のみ）
    config = TCUSSConfig(
        name="vis_data_gen",
        data_path=args.data_path,
        original_data_path=args.original_data_path,
        patchwork_path=args.patchwork_path,
        voxel_size=args.voxel_size,
        r_crop=args.r_crop,
        scan_window=args.scan_window
    )
    
    # データ生成器の初期化
    generator = VisualizationDataGenerator(config)
    
    # デバッグモードの設定
    if args.debug:
        generator.debug_mode = True
        print("デバッグモード: 最初の3ウィンドウのみ処理します")
    
    if args.sequences:
        # 指定されたシーケンスのみを処理
        for seq_str in args.sequences:
            seq = int(seq_str)
            if seq in generator.seq_to_scan_num:
                print(f"Processing sequence {seq:02d}...")
                generator._process_sequence(seq)
            else:
                print(f"Warning: Sequence {seq:02d} not found in available sequences")
    else:
        # すべてのシーケンスを処理
        generator.generate_all_sequences()
    
    print("可視化用データの生成が完了しました！")


if __name__ == '__main__':
    main() 