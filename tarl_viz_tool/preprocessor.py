"""
Data Preprocessor

TCUSSのKITTItemporalクラスと同じ前処理パイプラインを実装
- ボクセル化
- 球状クロッピング
- 座標正規化
- 時系列データ集約
"""

import numpy as np
import MinkowskiEngine as ME
from typing import Dict, List, Any, Optional
import logging
import time


class DataPreprocessor:
    """点群データの前処理クラス（TCUSSのKITTItemporalクラス準拠）"""
    
    def __init__(self, voxel_size: float = 0.15, r_crop: float = 50.0, 
                 scan_window: int = 12, logger: logging.Logger = None):
        """
        Args:
            voxel_size: ボクセルサイズ（TCUSSデフォルト: 0.15）
            r_crop: クロッピング半径（TCUSSデフォルト: 50.0）
            scan_window: 時系列ウィンドウサイズ（TCUSSデフォルト: 12）
            logger: ロガー
        """
        self.voxel_size = voxel_size
        self.r_crop = r_crop
        self.scan_window = scan_window
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info(f"前処理器を初期化: voxel_size={voxel_size}, r_crop={r_crop}")
    
    def process_frame(self, frame_data: Dict[str, Any], aggregate_temporal: bool = True) -> Dict[str, Any]:
        """単一フレームの前処理
        
        Args:
            frame_data: データローダーから得られるフレームデータ
            aggregate_temporal: 時系列集約を行うかどうか
            
        Returns:
            前処理済みデータ
        """
        start_time = time.time()
        
        try:
            coords = frame_data['coords'].copy()
            feats = frame_data['feats'].copy()
            labels = frame_data['labels'].copy()
            ground_labels = frame_data['ground_labels'].copy()
            
            # デバッグ情報
            self.logger.debug(f"データ形状: coords={coords.shape}, feats={feats.shape}, labels={labels.shape}")
            
            # 元の座標を保存（時系列集約用）
            coords_original = coords.copy()
            
            # 1. 中心化
            coords_mean = coords.mean(0)
            coords -= coords_mean
            
            # 2. ボクセル化
            coords_vox, feats_vox, labels_vox, unique_map, inverse_map = self._voxelize(
                coords, feats, labels
            )
            
            # 3. 球状クロッピング
            mask_crop = np.sqrt(((coords_vox * self.voxel_size) ** 2).sum(-1)) < self.r_crop
            coords_cropped = coords_vox[mask_crop]
            feats_cropped = feats_vox[mask_crop]
            labels_cropped = labels_vox[mask_crop]
            
            # 地面ラベルも同様に処理
            if len(ground_labels) == len(frame_data['coords']):
                # unique_mapとcrop_maskを適用
                ground_labels_processed = ground_labels[unique_map][mask_crop]
            else:
                # サイズが合わない場合は0で埋める
                ground_labels_processed = np.zeros(len(coords_cropped), dtype=np.uint32)
            
            # 4. 座標正規化 (_augment_coords_to_feats準拠)
            coords_center = coords_cropped.mean(0, keepdims=True)
            # Z軸中心を0に設定（3次元データの場合のみ）
            if coords_center.shape[1] >= 3:
                coords_center[0, 2] = 0  # Z軸は0に固定
            coords_normalized = coords_cropped - coords_center
            
            # 前処理済みデータの準備
            processed_data = {
                'coords': coords_normalized.astype(np.float32),
                'coords_original': coords_original[unique_map][mask_crop],
                'feats': feats_cropped.astype(np.float32),
                'labels': labels_cropped.astype(np.int32),
                'ground_labels': ground_labels_processed,
                'unique_map': unique_map,
                'inverse_map': inverse_map,
                'crop_mask': mask_crop,
                'coords_mean': coords_mean,
                'coords_center': coords_center,
                'frame_info': {
                    'seq': frame_data['seq'],
                    'frame': frame_data['frame'],
                    'original_points': len(frame_data['coords']),
                    'after_voxel': len(coords_vox),
                    'after_crop': len(coords_cropped)
                }
            }
            
            processing_time = time.time() - start_time
            self.logger.debug(f"前処理完了: {processing_time:.3f}秒, "
                            f"{len(frame_data['coords'])}→{len(coords_cropped)}点")
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"前処理中にエラー: {e}")
            raise
    
    def process_temporal_sequence(self, data_loader, seq: str, center_frame: int) -> Dict[str, Any]:
        """時系列シーケンスの前処理と集約（TCUSSのKITTItemporalクラス準拠）
        
        Args:
            data_loader: データローダー
            seq: シーケンス番号
            center_frame: 中心フレーム番号
            
        Returns:
            集約済み点群データ
        """
        try:
            # ウィンドウ内のフレーム範囲を計算
            half_window = self.scan_window // 2
            start_frame = max(0, center_frame - half_window)
            end_frame = center_frame + half_window
            
            # 各フレームのデータを読み込み・前処理
            frame_coords = []
            frame_labels = []
            frame_ground_labels = []
            frame_unique_maps = []
            frame_masks = []
            frame_poses = []
            
            for frame_idx in range(start_frame, end_frame + 1):
                if frame_idx > data_loader.get_max_frame(seq):
                    break
                    
                frame_data = data_loader.load_frame(seq, frame_idx)
                if frame_data is None:
                    continue
                
                # 前処理（時系列集約なし）
                processed = self.process_frame(frame_data, aggregate_temporal=False)
                
                frame_coords.append(processed['coords_original'])
                frame_labels.append(processed['labels'])
                frame_ground_labels.append(processed['ground_labels'])
                frame_unique_maps.append(processed['unique_map'])
                frame_masks.append(processed['crop_mask'])
                frame_poses.append(frame_data['pose'])
            
            if not frame_coords:
                raise ValueError("有効なフレームデータがありません")
            
            # 時系列集約の実行
            aggregated_data = self._aggregate_temporal_data(
                frame_coords=frame_coords,
                frame_labels=frame_labels,
                frame_ground_labels=frame_ground_labels,
                frame_unique_maps=frame_unique_maps,
                frame_masks=frame_masks,
                frame_poses=frame_poses,
                center_frame_idx=center_frame - start_frame
            )
            
            return aggregated_data
            
        except Exception as e:
            self.logger.error(f"時系列前処理エラー: {e}")
            raise
    
    def _voxelize(self, coords: np.ndarray, feats: np.ndarray, labels: np.ndarray):
        """ボクセル化（MinkowskiEngine使用、TCUSSのKITTItemporalクラス準拠）"""
        scale = 1.0 / self.voxel_size
        coords_quantized = np.floor(coords * scale)
        
        coords_vox, feats_vox, labels_vox, unique_map, inverse_map = ME.utils.sparse_quantize(
            np.ascontiguousarray(coords_quantized), 
            feats, 
            labels=labels, 
            ignore_label=-1, 
            return_index=True, 
            return_inverse=True
        )
        
        return coords_vox, feats_vox, labels_vox, unique_map, inverse_map
    
    def _aggregate_temporal_data(self, frame_coords: List[np.ndarray], 
                                frame_labels: List[np.ndarray],
                                frame_ground_labels: List[np.ndarray],
                                frame_unique_maps: List[np.ndarray],
                                frame_masks: List[np.ndarray],
                                frame_poses: List[np.ndarray],
                                center_frame_idx: int) -> Dict[str, Any]:
        """時系列データの集約（TCUSSの_aggretate_pcds準拠）"""
        
        if not frame_poses or len(frame_poses) <= center_frame_idx:
            raise ValueError("ポーズ情報が不足しています")
        
        # 集約用の配列を初期化
        points_set = np.empty((0, 3))
        ground_label_set = np.empty((0, 1))
        label_set = np.empty((0, 1))
        element_nums = []
        
        # 各フレームを中心フレームの座標系に変換して集約
        center_pose = frame_poses[center_frame_idx]
        
        for i, (coords, labels, ground_labels, unique_map, mask, pose) in enumerate(
            zip(frame_coords, frame_labels, frame_ground_labels, 
                frame_unique_maps, frame_masks, frame_poses)
        ):
            try:
                # ポーズ変換を適用
                transformed_coords = self._apply_transform(coords, pose)
                
                # 地面ラベルの処理
                if len(ground_labels) == len(coords):
                    g_set = ground_labels.reshape(-1, 1)
                else:
                    g_set = np.zeros((len(coords), 1), dtype=np.uint32)
                
                # データを集約
                points_set = np.vstack((points_set, transformed_coords))
                ground_label_set = np.vstack((ground_label_set, g_set))
                label_set = np.vstack((label_set, labels.reshape(-1, 1)))
                element_nums.append(len(coords))
                
            except Exception as e:
                self.logger.warning(f"フレーム{i}の変換でエラー: {e}")
                continue
        
        # 最終的に中心フレームの座標系に統一
        final_coords = self._undo_transform(points_set, center_pose)
        
        return {
            'coords': final_coords.astype(np.float32),
            'ground_labels': ground_label_set.flatten().astype(np.uint32),
            'labels': label_set.flatten().astype(np.int32),
            'element_nums': element_nums,
            'total_points': len(final_coords)
        }
    
    def _apply_transform(self, points: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """ポーズ変換の適用（TCUSSのKITTItemporalクラス準拠）"""
        if points.shape[1] != 3:
            raise ValueError(f"点群は3次元である必要があります: {points.shape}")
        
        # 同次座標に変換
        hpoints = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))
        
        # 変換を適用
        transformed = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)
        
        return transformed[:, :3]
    
    def _undo_transform(self, points: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """ポーズ変換の逆変換（TCUSSのKITTItemporalクラス準拠）"""
        if points.shape[1] != 3:
            raise ValueError(f"点群は3次元である必要があります: {points.shape}")
        
        # 同次座標に変換
        hpoints = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))
        
        # 逆変換を適用
        inv_pose = np.linalg.inv(pose)
        transformed = np.sum(np.expand_dims(hpoints, 2) * inv_pose.T, axis=1)
        
        return transformed[:, :3]
    
    def update_parameters(self, **kwargs):
        """パラメータの動的更新"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                old_value = getattr(self, key)
                setattr(self, key, value)
                self.logger.info(f"パラメータ更新: {key}: {old_value} → {value}")
            else:
                self.logger.warning(f"不明なパラメータ: {key}")
    
    def get_preprocessing_stats(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """前処理統計の取得"""
        stats = {
            'original_points': processed_data['frame_info']['original_points'],
            'after_voxel': processed_data['frame_info']['after_voxel'],
            'after_crop': processed_data['frame_info']['after_crop'],
            'reduction_ratio': processed_data['frame_info']['after_crop'] / processed_data['frame_info']['original_points'],
            'voxel_size': self.voxel_size,
            'crop_radius': self.r_crop
        }
        return stats 