"""
VoteFlow Wrapper

VoteFlowモデルをTCUSSパイプラインで使用するためのラッパークラス。
推論専用で、学習済みチェックポイントからscene flowを計算します。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from omegaconf import DictConfig
import sys
import os

# VoteFlowのパスを追加
VOTEFLOW_PATH = os.path.join(os.path.dirname(__file__), 'VoteFlow')
sys.path.insert(0, VOTEFLOW_PATH)

from src.trainer import ModelWrapper

# 対応点計算関数をcorrespondence.pyから再エクスポート
from scene_flow.correspondence import (
    compute_point_correspondence,
    compute_superpoint_correspondence_matrix
)


class VoteFlowWrapper:
    """
    VoteFlowモデルのラッパークラス
    
    TCUSSのデータ形式からVoteFlowの入力形式に変換し、
    scene flowを計算して点の対応を取得します。
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        voxel_size: List[float] = [0.2, 0.2, 6],
        point_cloud_range: List[float] = [-51.2, -51.2, -3, 51.2, 51.2, 3],
        device: str = "cuda:0"
    ):
        """
        Args:
            checkpoint_path: VoteFlowのチェックポイントパス
            voxel_size: VoteFlowのボクセルサイズ
            point_cloud_range: 点群の範囲
            device: 使用するデバイス
        """
        self.device = device
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        
        # チェックポイントからモデルをロード
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """VoteFlowモデルをロード"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"VoteFlow checkpoint not found: {checkpoint_path}")
        
        # VoteFlowディレクトリに一時的に移動してHydraのパス解決を正しく行う
        original_cwd = os.getcwd()
        try:
            os.chdir(VOTEFLOW_PATH)
            
            # チェックポイントの絶対パス
            abs_checkpoint_path = os.path.join(original_cwd, checkpoint_path) if not os.path.isabs(checkpoint_path) else checkpoint_path
            
            # チェックポイントからハイパーパラメータを取得
            checkpoint = torch.load(abs_checkpoint_path, map_location=self.device)
            
            # 設定を作成
            cfg = DictConfig(checkpoint["hyper_parameters"]["cfg"])
            
            # 古いクラス名を新しいクラス名に修正
            if cfg.model.target._target_ == 'src.models.SFVoxelModel':
                cfg.model.target._target_ = 'src.models.VoteFlow'
            
            # モデルをロード
            model = ModelWrapper.load_from_checkpoint(
                abs_checkpoint_path,
                cfg=cfg,
                eval=True,
                map_location=self.device
            )
            model = model.to(self.device)
            
            return model
        finally:
            os.chdir(original_cwd)
    
    @torch.no_grad()
    def compute_flow(
        self,
        points_t: np.ndarray,
        points_t1: np.ndarray,
        pose_t: Optional[np.ndarray] = None,
        pose_t1: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        2フレーム間のscene flowを計算
        
        Args:
            points_t: 時刻tの点群 [N, 3]
            points_t1: 時刻t+1の点群 [M, 3]
            pose_t: 時刻tのポーズ（4x4行列）、Noneの場合は単位行列
            pose_t1: 時刻t+1のポーズ（4x4行列）、Noneの場合は単位行列
            
        Returns:
            flow: 時刻tの各点に対するフロー [N_valid, 3]
            valid_indices: 有効な点のインデックス [N_valid]
        """
        device = torch.device(self.device)
        prev_device = torch.cuda.current_device() if device.type == "cuda" and torch.cuda.is_available() else None
        if device.type == "cuda":
            torch.cuda.set_device(device)
        try:
            # ポーズがない場合は単位行列
            if pose_t is None:
                pose_t = np.eye(4, dtype=np.float32)
            if pose_t1 is None:
                pose_t1 = np.eye(4, dtype=np.float32)
            
            # 点群を範囲内にフィルタリング（念のため連続メモリ化）
            points_t_filtered, mask_t = self._filter_points(np.ascontiguousarray(points_t))
            points_t1_filtered, mask_t1 = self._filter_points(np.ascontiguousarray(points_t1))
            
            # フィルタ後のインデックスを保持（元のインデックス復元用）
            idx_t_filtered = np.where(mask_t)[0]
            
            if len(points_t_filtered) == 0 or len(points_t1_filtered) == 0:
                return np.zeros((0, 3), dtype=np.float32), np.array([], dtype=np.int64)
            
            # バッチ作成
            batch = self._create_batch(points_t_filtered, points_t1_filtered, pose_t, pose_t1)
            
            # 推論
            res_dict = self.model.model(batch)
            
            # フローを取得
            flow = res_dict['flow'][0].cpu().numpy()  # [N_valid, 3]
            valid_indices = res_dict['pc0_valid_point_idxes'][0].cpu().numpy()
            
            # 元のインデックスに変換（サブサンプル対応）
            original_valid_indices = idx_t_filtered[valid_indices]
            
            return flow, original_valid_indices
        finally:
            if prev_device is not None:
                torch.cuda.set_device(prev_device)
    
    def _filter_points(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """点群を範囲内にフィルタリング"""
        x_min, y_min, z_min = self.point_cloud_range[:3]
        x_max, y_max, z_max = self.point_cloud_range[3:]
        
        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )
        
        return points[mask], mask
    
    def _create_batch(
        self,
        points_t: np.ndarray,
        points_t1: np.ndarray,
        pose_t: np.ndarray,
        pose_t1: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """VoteFlow用のバッチを作成（単一サンプル）"""
        batch = {
            'pc0': torch.from_numpy(points_t).float().unsqueeze(0).to(self.device),
            'pc1': torch.from_numpy(points_t1).float().unsqueeze(0).to(self.device),
            'pose0': torch.from_numpy(pose_t).float().unsqueeze(0).to(self.device),
            'pose1': torch.from_numpy(pose_t1).float().unsqueeze(0).to(self.device),
        }
        return batch
    
    @torch.no_grad()
    def compute_flow_batch(
        self,
        points_t_list: List[np.ndarray],
        points_t1_list: List[np.ndarray],
        pose_t_list: Optional[List[np.ndarray]] = None,
        pose_t1_list: Optional[List[np.ndarray]] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        複数サンプルのscene flowを一括計算（バッチ処理、NaNパディング使用）
        
        VoteFlowの学習時と同様に、異なるサイズの点群をNaNでパディングして
        バッチ処理を行う。
        
        Args:
            points_t_list: 時刻tの点群リスト [N_i, 3] × batch_size
            points_t1_list: 時刻t+1の点群リスト [M_i, 3] × batch_size
            pose_t_list: 時刻tのポーズリスト（4x4行列）、Noneの場合は単位行列
            pose_t1_list: 時刻t+1のポーズリスト（4x4行列）、Noneの場合は単位行列
            
        Returns:
            results: [(flow, valid_indices), ...] のリスト
        """
        batch_size = len(points_t_list)
        if batch_size == 0:
            return []
        
        device = torch.device(self.device)
        prev_device = torch.cuda.current_device() if device.type == "cuda" and torch.cuda.is_available() else None
        if device.type == "cuda":
            torch.cuda.set_device(device)
        
        try:
            # デフォルトポーズ
            if pose_t_list is None:
                pose_t_list = [np.eye(4, dtype=np.float32) for _ in range(batch_size)]
            if pose_t1_list is None:
                pose_t1_list = [np.eye(4, dtype=np.float32) for _ in range(batch_size)]
            
            # 各サンプルをフィルタリング
            pc0_tensors = []
            pc1_tensors = []
            pose0_list = []
            pose1_list = []
            idx_t_filtered_list = []  # 元インデックス復元用
            valid_sample_indices = []  # 有効なサンプルのインデックス
            
            for i in range(batch_size):
                points_t_filtered, mask_t = self._filter_points(np.ascontiguousarray(points_t_list[i]))
                points_t1_filtered, mask_t1 = self._filter_points(np.ascontiguousarray(points_t1_list[i]))
                
                if len(points_t_filtered) == 0 or len(points_t1_filtered) == 0:
                    # 空のサンプルはスキップ
                    idx_t_filtered_list.append(None)
                    continue
                
                idx_t_filtered_list.append(np.where(mask_t)[0])
                valid_sample_indices.append(i)
                
                pc0_tensors.append(torch.from_numpy(points_t_filtered).float())
                pc1_tensors.append(torch.from_numpy(points_t1_filtered).float())
                pose0_list.append(torch.from_numpy(pose_t_list[i]).float().to(self.device))
                pose1_list.append(torch.from_numpy(pose_t1_list[i]).float().to(self.device))
            
            # 結果を格納するリスト
            results = [None] * batch_size
            
            if len(pc0_tensors) == 0:
                # 全サンプルが空の場合
                return [(np.zeros((0, 3), dtype=np.float32), np.array([], dtype=np.int64)) for _ in range(batch_size)]
            
            # NaNでパディングして同じサイズに揃える（VoteFlowの学習時と同様）
            pc0_padded = torch.nn.utils.rnn.pad_sequence(
                pc0_tensors, batch_first=True, padding_value=float('nan')
            ).to(self.device)
            pc1_padded = torch.nn.utils.rnn.pad_sequence(
                pc1_tensors, batch_first=True, padding_value=float('nan')
            ).to(self.device)
            
            # バッチを作成（パディング済みテンソル形式）
            batch = {
                'pc0': pc0_padded,   # [B, max_N, 3]
                'pc1': pc1_padded,   # [B, max_M, 3]
                'pose0': pose0_list, # リスト形式
                'pose1': pose1_list, # リスト形式
            }
            
            # 推論（一括処理）
            res_dict = self.model.model(batch)
            
            # 結果をアンパック
            for batch_idx, orig_idx in enumerate(valid_sample_indices):
                flow = res_dict['flow'][batch_idx].cpu().numpy()
                valid_indices = res_dict['pc0_valid_point_idxes'][batch_idx].cpu().numpy()
                
                # 元のインデックスに変換
                idx_t_filtered = idx_t_filtered_list[orig_idx]
                if idx_t_filtered is not None and len(valid_indices) > 0:
                    original_valid_indices = idx_t_filtered[valid_indices]
                else:
                    original_valid_indices = np.array([], dtype=np.int64)
                
                results[orig_idx] = (flow, original_valid_indices)
            
            # 空サンプルの結果を埋める
            for i in range(batch_size):
                if results[i] is None:
                    results[i] = (np.zeros((0, 3), dtype=np.float32), np.array([], dtype=np.int64))
            
            return results
            
        finally:
            if prev_device is not None:
                torch.cuda.set_device(prev_device)


if __name__ == "__main__":
    # テスト
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, 
                        default='scene_flow/VoteFlow/checkpoints/voteflow_best_m8n128_ori.ckpt')
    args = parser.parse_args()
    
    # モデルのロードテスト
    wrapper = VoteFlowWrapper(args.checkpoint)
    print("VoteFlow model loaded successfully!")
    
    # ダミーデータでテスト
    points_t = np.random.randn(1000, 3).astype(np.float32) * 10
    points_t1 = np.random.randn(1000, 3).astype(np.float32) * 10
    
    flow, valid_indices = wrapper.compute_flow(points_t, points_t1)
    print(f"Flow shape: {flow.shape}, Valid indices: {len(valid_indices)}")

