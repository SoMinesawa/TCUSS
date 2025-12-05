#!/usr/bin/env python3
"""
Superpointの可視化ツール
TCUSSプロジェクトの学習済みモデルを使用してSuperpointを生成し、色付けして保存する
"""

import torch
import numpy as np
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os

# プロジェクトルートを追加
sys.path.append('..')

from vis_sp_config import VisualizationConfig
from vis_sp_dataset import VisualizationDataset
from vis_sp_utils import (
    load_model, extract_features, compute_superpoints, 
    colorize_point_cloud, save_colored_pointcloud, save_superpoint_labels
)


class SuperpointVisualizer:
    """Superpointの可視化を行うクラス"""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"使用デバイス: {self.device}")
        
        # モデルのロード
        self.model, self.classifier = load_model(config, self.device)
        
        # データセットの作成
        self.dataset = VisualizationDataset(config)
        
        # データローダーの作成（バッチサイズ1で処理）
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)
    
    def process_single_scene(self, data_batch):
        """単一シーンの処理"""
        # バッチから単一データを取得
        coords = data_batch['coords'][0].numpy()  # バッチ次元を除去
        feats = data_batch['feats'][0].numpy()
        labels = data_batch['labels'][0].numpy()
        region = data_batch['region'][0].numpy()
        original_coords = data_batch['original_coords'][0].numpy()
        scene_name = data_batch['scene_name'][0]
        
        print(f"処理中: {scene_name}")
        print(f"点数: {len(coords)}, 初期スーパーポイント数: {len(np.unique(region[region != -1]))}")
        
        # バッチ次元を追加して特徴量抽出用の座標を準備
        batch_coords = np.hstack([np.zeros((len(coords), 1)), coords])
        batch_coords_tensor = torch.from_numpy(batch_coords).float().to(self.device)
        
        # 特徴量抽出
        with torch.no_grad():
            in_field = ME.TensorField(
                batch_coords_tensor[:, 1:] * self.config.voxel_size, 
                batch_coords_tensor, 
                device=self.device
            )
            extracted_feats = self.model(in_field).detach().cpu()
        
        # スーパーポイントの計算
        superpoint_labels = compute_superpoints(
            coords, extracted_feats, region, self.config.current_growsp
        )
        
        unique_sp_labels = np.unique(superpoint_labels[superpoint_labels != -1])
        print(f"統合後スーパーポイント数: {len(unique_sp_labels)}")
        
        # 色付け
        point_colors = colorize_point_cloud(original_coords, superpoint_labels)
        
        # 保存
        saved_files = []
        
        # 色付き点群の保存
        output_file = save_colored_pointcloud(
            original_coords, point_colors, 
            self.config.output_path, scene_name, 
            self.config.file_extension
        )
        saved_files.append(output_file)
        
        # スーパーポイントラベルの保存
        label_file = save_superpoint_labels(
            superpoint_labels, self.config.output_path, 
            scene_name, 'npy'
        )
        saved_files.append(label_file)
        
        return saved_files
    
    def visualize_all(self):
        """全シーンの可視化処理"""
        print(f"可視化を開始します。対象シーン数: {len(self.dataset)}")
        
        total_saved_files = []
        failed_scenes = []
        
        for i, data_batch in enumerate(tqdm(self.dataloader, desc="処理中")):
            try:
                saved_files = self.process_single_scene(data_batch)
                total_saved_files.extend(saved_files)
                
                if self.config.debug and i >= 2:  # デバッグモードでは3シーンのみ処理
                    break
                    
            except Exception as e:
                scene_name = data_batch['scene_name'][0] if 'scene_name' in data_batch else f"scene_{i}"
                print(f"エラー - {scene_name}: {str(e)}")
                failed_scenes.append(scene_name)
                continue
        
        # 結果の表示
        print(f"\n可視化処理完了!")
        print(f"成功: {len(total_saved_files)//2} シーン")
        print(f"失敗: {len(failed_scenes)} シーン")
        
        if failed_scenes:
            print(f"失敗したシーン: {failed_scenes}")
        
        print(f"保存先: {self.config.output_path}")
        
        return total_saved_files, failed_scenes


def main():
    """メイン関数"""
    # 設定の読み込み
    config = VisualizationConfig.from_parse_args()
    
    if config.debug:
        print("デバッグモードが有効です")
        config.max_scenes = 3
    
    print(f"設定:")
    print(f"  データパス: {config.data_path}")
    print(f"  モデルパス: {config.model_path}")
    print(f"  分類器パス: {config.classifier_path}")
    print(f"  出力パス: {config.output_path}")
    print(f"  対象シーケンス: {config.sequences}")
    print(f"  現在のスーパーポイント数: {config.current_growsp}")
    print(f"  最大処理シーン数: {config.max_scenes}")
    
    # 出力ディレクトリの作成
    os.makedirs(config.output_path, exist_ok=True)
    
    # 可視化処理の実行
    visualizer = SuperpointVisualizer(config)
    saved_files, failed_scenes = visualizer.visualize_all()
    
    print(f"\n処理完了。保存されたファイル数: {len(saved_files)}")


if __name__ == '__main__':
    main() 