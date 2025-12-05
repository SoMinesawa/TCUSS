import torch
import numpy as np
import MinkowskiEngine as ME
import os
from torch.utils.data import Dataset
import sys
sys.path.append('..')

from lib.helper_ply import read_ply
from vis_sp_config import VisualizationConfig


class VisualizationDataset(Dataset):
    """Superpoint可視化用のデータセット"""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.file = []
        self.name = []
        
        # データファイルの収集
        for seq_id in self.config.sequences:
            seq_path = os.path.join(self.config.data_path, seq_id)
            if os.path.exists(seq_path):
                for f in sorted(os.listdir(seq_path)):
                    if f.endswith('.ply'):
                        self.file.append(os.path.join(seq_path, f))
                        self.name.append(f'/{seq_id}/{f[:-4]}')  # シーケンスIDとファイル名（拡張子なし）を含める
        
        # max_scenesによる制限
        if self.config.max_scenes is not None and len(self.file) > self.config.max_scenes:
            self.file = self.file[:self.config.max_scenes]
            self.name = self.name[:self.config.max_scenes]
            
        print(f"データセットに {len(self.file)} のシーンが含まれています")
    
    def __len__(self):
        return len(self.file)
    
    def voxelize(self, coords, feats, labels):
        """ボクセル化"""
        scale = 1 / self.config.voxel_size
        coords = np.floor(coords * scale)
        coords, feats, labels, unique_map, inverse_map = ME.utils.sparse_quantize(
            np.ascontiguousarray(coords), feats, labels=labels, 
            ignore_label=-1, return_index=True, return_inverse=True
        )
        return coords, feats, labels, unique_map, inverse_map
    
    def augment_coords_to_feats(self, coords, feats, labels=None):
        """座標を特徴量として使用するための変換"""
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0  # Z軸の中心は0にしない
        norm_coords = (coords - coords_center)
        return norm_coords, feats, labels
    
    def __getitem__(self, index):
        file_path = self.file[index]
        scene_name = self.name[index]
        
        # PLYファイルの読み込み
        data = read_ply(file_path)
        coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
        feats = np.array(data['remission'])[:, np.newaxis]
        labels = np.array(data['class'])
        
        # 座標の中心化
        coords_original = coords.copy()
        coords -= coords.mean(0)
        
        # ボクセル化
        coords, feats, labels, unique_map, inverse_map = self.voxelize(coords, feats, labels)
        coords = coords.astype(np.float32)
        
        # クロッピング
        mask = np.sqrt(((coords * self.config.voxel_size) ** 2).sum(-1)) < self.config.r_crop
        coords, feats, labels = coords[mask], feats[mask], labels[mask]
        
        # 初期スーパーポイントのロード
        region_file = os.path.join(self.config.sp_path, scene_name.strip('/') + '_superpoint.npy')
        if not os.path.exists(region_file):
            print(f"警告: スーパーポイントファイルが見つかりません: {region_file}")
            region = np.arange(len(coords))  # フォールバック: 各点を個別のスーパーポイントとする
        else:
            region = np.load(region_file)
            region = region[unique_map]
            region = region[mask]
        
        # 座標の特徴量変換
        coords, feats, labels = self.augment_coords_to_feats(coords, feats, labels)
        
        # 元の座標（変換前）も保存
        original_coords = coords_original[unique_map][mask]
        
        return {
            'coords': coords,
            'feats': feats,
            'labels': labels,
            'region': region,
            'original_coords': original_coords,
            'scene_name': scene_name,
            'inverse_map': inverse_map
        } 