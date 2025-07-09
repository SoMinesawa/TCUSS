"""
SemanticKITTI Data Loader

SemanticKITTIデータセットからの点群データ、ラベル、ポーズ情報の読み込みを行う
生のBINファイルを直接読み込み、PLYファイルへの変換は不要
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import logging
import yaml

# TCUSSの既存ライブラリを使用
from lib.helper_ply import read_ply


class SemanticKITTILoader:
    """SemanticKITTIデータセットのローダークラス"""
    
    def __init__(self, data_path: str, original_data_path: str = None, 
                 patchwork_path: str = None, logger: logging.Logger = None, 
                 use_ply: bool = True):
        """
        Args:
            data_path: データファイルのベースパス（PLYまたはBIN）
            original_data_path: ポーズ情報用のパス
            patchwork_path: 地面ラベル用のパス
            logger: ロガー
            use_ply: PLYファイルを使用するか（False: BINファイルを使用）
        """
        self.data_path = Path(data_path)
        self.original_data_path = Path(original_data_path) if original_data_path else self.data_path
        self.patchwork_path = Path(patchwork_path) if patchwork_path else None
        self.logger = logger or logging.getLogger(__name__)
        self.use_ply = use_ply
        
        # SemanticKITTIラベルマッピング（learning_map）
        self.learning_map = self._load_learning_map()
        
        # SemanticKITTIクラス情報
        self.label_to_names = {
            0: 'unlabeled',
            1: 'car',
            2: 'bicycle', 
            3: 'motorcycle',
            4: 'truck',
            5: 'other-vehicle',
            6: 'person',
            7: 'bicyclist',
            8: 'motorcyclist',
            9: 'road',
            10: 'parking',
            11: 'sidewalk',
            12: 'other-ground',
            13: 'building',
            14: 'fence',
            15: 'vegetation',
            16: 'trunk',
            17: 'terrain',
            18: 'pole',
            19: 'traffic-sign'
        }
        
        # クラス色（SemanticKITTI標準色をRGBで定義）
        self.class_colors = {
            0: [0, 0, 0],           # unlabeled - black
            1: [100, 150, 245],     # car - blue
            2: [100, 230, 245],     # bicycle - cyan
            3: [30, 60, 150],       # motorcycle - dark blue
            4: [80, 30, 180],       # truck - purple
            5: [0, 0, 255],         # other-vehicle - red
            6: [255, 30, 30],       # person - red
            7: [255, 40, 200],      # bicyclist - magenta
            8: [150, 30, 90],       # motorcyclist - dark magenta
            9: [255, 0, 255],       # road - magenta
            10: [255, 150, 255],    # parking - light magenta
            11: [75, 0, 75],        # sidewalk - dark magenta
            12: [175, 0, 75],       # other-ground - dark pink
            13: [255, 200, 0],      # building - yellow
            14: [255, 120, 50],     # fence - orange
            15: [0, 175, 0],        # vegetation - green
            16: [135, 60, 0],       # trunk - brown
            17: [150, 240, 80],     # terrain - light green
            18: [255, 240, 150],    # pole - light yellow
            19: [255, 0, 0]         # traffic-sign - red
        }
        
        # キャッシュ
        self._pose_cache = {}
        self._frame_cache = {}
        
        self.logger.info(f"SemanticKITTIローダーを初期化: {self.data_path}, use_ply={self.use_ply}")
    
    def _load_learning_map(self) -> Dict[int, int]:
        """SemanticKITTIのラベルマッピングを読み込み"""
        # デフォルトのラベルマッピング（data_prepare/semantic-kitti.yamlから）
        default_map = {
            0: 0,     # unlabeled
            1: 0,     # outlier -> unlabeled
            10: 1,    # car
            11: 2,    # bicycle
            13: 5,    # bus -> other-vehicle
            15: 3,    # motorcycle
            16: 5,    # on-rails -> other-vehicle
            18: 4,    # truck
            20: 5,    # other-vehicle
            30: 6,    # person
            31: 7,    # bicyclist
            32: 8,    # motorcyclist
            40: 9,    # road
            44: 10,   # parking
            48: 11,   # sidewalk
            49: 12,   # other-ground
            50: 13,   # building
            51: 14,   # fence
            52: 0,    # other-structure -> unlabeled
            60: 9,    # lane-marking -> road
            70: 15,   # vegetation
            71: 16,   # trunk
            72: 17,   # terrain
            80: 18,   # pole
            81: 19,   # traffic-sign
            99: 0,    # other-object -> unlabeled
            252: 1,   # moving-car -> car
            253: 7,   # moving-bicyclist -> bicyclist
            254: 6,   # moving-person -> person
            255: 8,   # moving-motorcyclist -> motorcyclist
            256: 5,   # moving-on-rails -> other-vehicle
            257: 5,   # moving-bus -> other-vehicle
            258: 4,   # moving-truck -> truck
            259: 5,   # moving-other-vehicle -> other-vehicle
        }
        
        return default_map
    
    def get_available_sequences(self) -> list:
        """利用可能なシーケンス一覧を取得"""
        sequences = []
        if self.data_path.exists():
            for seq_dir in sorted(self.data_path.iterdir()):
                if seq_dir.is_dir() and seq_dir.name.isdigit():
                    sequences.append(seq_dir.name)
        return sequences
    
    def get_max_frame(self, seq: str) -> int:
        """指定シーケンスの最大フレーム番号を取得"""
        if self.use_ply:
            seq_path = self.data_path / seq
            if not seq_path.exists():
                return 0
            
            ply_files = list(seq_path.glob("*.ply"))
            if not ply_files:
                return 0
            
            max_frame = max([int(f.stem) for f in ply_files])
            return max_frame
        else:
            # BINファイル用
            seq_path = self.data_path / seq / "velodyne"
            if not seq_path.exists():
                return 0
            
            bin_files = list(seq_path.glob("*.bin"))
            if not bin_files:
                return 0
            
            max_frame = max([int(f.stem) for f in bin_files])
            return max_frame
    
    def frame_exists(self, seq: str, frame: int) -> bool:
        """指定フレームが存在するかチェック"""
        if self.use_ply:
            ply_path = self.data_path / seq / f"{frame:06d}.ply"
            return ply_path.exists()
        else:
            bin_path = self.data_path / seq / "velodyne" / f"{frame:06d}.bin"
            return bin_path.exists()
    
    def load_frame(self, seq: str, frame: int) -> Optional[Dict[str, Any]]:
        """指定フレームのデータを読み込み
        
        Returns:
            Dict containing:
                - coords: 点群座標 (N, 3)
                - feats: 特徴量 (N, 1) - remission
                - labels: ラベル (N,) - class
                - ground_labels: 地面ラベル (N,) - 0 or 1
                - pose: ポーズ行列 (4, 4)
        """
        cache_key = f"{seq}_{frame:06d}"
        if cache_key in self._frame_cache:
            return self._frame_cache[cache_key]
        
        try:
            if self.use_ply:
                # PLYファイルの読み込み
                ply_path = self.data_path / seq / f"{frame:06d}.ply"
                if not ply_path.exists():
                    self.logger.error(f"PLYファイルが見つかりません: {ply_path}")
                    return None
                
                data = read_ply(ply_path)
                
                # 座標とfeatures
                coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
                feats = np.array(data['remission'], dtype=np.float32)[:, np.newaxis]
                labels = np.array(data['class'], dtype=np.int32)
                
            else:
                # BINファイルの読み込み
                bin_path = self.data_path / seq / "velodyne" / f"{frame:06d}.bin"
                if not bin_path.exists():
                    self.logger.error(f"BINファイルが見つかりません: {bin_path}")
                    return None
                
                # 生のbinファイルを読み込み
                scan = np.fromfile(bin_path, dtype=np.float32)
                scan = scan.reshape((-1, 4))
                
                coords = scan[:, 0:3].astype(np.float32)
                feats = scan[:, 3:4].astype(np.float32)  # remission
                
                # ラベルファイルの読み込み
                labels = self._load_labels(seq, frame)
                if labels is None:
                    # ラベルが無い場合は全て0（unlabeled）
                    labels = np.zeros(len(coords), dtype=np.int32)
            
            # ポーズ情報の読み込み
            pose = self.load_pose(seq, frame)
            
            # 地面ラベルの読み込み
            ground_labels = self.load_ground_labels(seq, frame)
            if ground_labels is None:
                # 地面ラベルがない場合は全て0（非地面）として扱う
                ground_labels = np.zeros(len(coords), dtype=np.uint32)
            
            frame_data = {
                'coords': coords,
                'feats': feats, 
                'labels': labels,
                'ground_labels': ground_labels,
                'pose': pose,
                'seq': seq,
                'frame': frame
            }
            
            # キャッシュに保存（メモリ使用量を制限）
            if len(self._frame_cache) < 10:  # 最大10フレームをキャッシュ
                self._frame_cache[cache_key] = frame_data
            
            self.logger.debug(f"フレーム{seq}/{frame:06d}を読み込み: {len(coords)}点")
            return frame_data
            
        except Exception as e:
            self.logger.error(f"フレーム{seq}/{frame:06d}の読み込みでエラー: {e}")
            return None
    
    def load_pose(self, seq: str, frame: int) -> Optional[np.ndarray]:
        """ポーズ情報を読み込み"""
        try:
            if seq in self._pose_cache:
                poses = self._pose_cache[seq]
            else:
                poses = self._load_poses(seq)
                self._pose_cache[seq] = poses
            
            if poses is not None and frame < len(poses):
                return poses[frame]
            else:
                # ポーズ情報がない場合は単位行列を返す
                return np.eye(4, dtype=np.float32)
                
        except Exception as e:
            self.logger.warning(f"ポーズ読み込みエラー {seq}/{frame}: {e}")
            return np.eye(4, dtype=np.float32)
    
    def _load_poses(self, seq: str) -> Optional[np.ndarray]:
        """シーケンス全体のポーズ情報を読み込み（TCUSSのKITTItemporalクラス準拠）"""
        try:
            calib_path = self.original_data_path / seq / "calib.txt"
            poses_path = self.original_data_path / seq / "poses.txt"
            
            if not calib_path.exists() or not poses_path.exists():
                self.logger.warning(f"ポーズファイルが見つかりません: {seq}")
                return None
            
            # キャリブレーション情報の読み込み
            calibration = self._parse_calibration(calib_path)
            
            # ポーズファイルの読み込み
            with open(poses_path, 'r') as poses_file:
                poses = []
                Tr = calibration.get("Tr", np.eye(4))
                Tr_inv = np.linalg.inv(Tr)
                
                for line in poses_file:
                    values = [float(v) for v in line.strip().split()]
                    
                    pose = np.zeros((4, 4))
                    pose[0, 0:4] = values[0:4]
                    pose[1, 0:4] = values[4:8]
                    pose[2, 0:4] = values[8:12]
                    pose[3, 3] = 1.0
                    
                    # Tr変換を適用
                    final_pose = np.matmul(Tr_inv, np.matmul(pose, Tr))
                    poses.append(final_pose)
                
                return np.array(poses)
                
        except Exception as e:
            self.logger.warning(f"ポーズ読み込みエラー {seq}: {e}")
            return None
    
    def _parse_calibration(self, calib_path: Path) -> Dict[str, np.ndarray]:
        """キャリブレーションファイルの解析（TCUSSのKITTItemporalクラス準拠）"""
        calib = {}
        
        try:
            with open(calib_path, 'r') as calib_file:
                for line in calib_file:
                    if ':' not in line:
                        continue
                        
                    key, content = line.strip().split(":", 1)
                    values = [float(v) for v in content.strip().split()]
                    
                    if len(values) >= 12:
                        pose = np.zeros((4, 4))
                        pose[0, 0:4] = values[0:4]
                        pose[1, 0:4] = values[4:8]
                        pose[2, 0:4] = values[8:12]
                        pose[3, 3] = 1.0
                        calib[key] = pose
                        
        except Exception as e:
            self.logger.warning(f"キャリブレーション解析エラー: {e}")
        
        return calib
    
    def _load_labels(self, seq: str, frame: int) -> Optional[np.ndarray]:
        """SemanticKITTIラベルファイルを読み込み（BINファイル用）"""
        try:
            label_path = self.data_path / seq / "labels" / f"{frame:06d}.label"
            if not label_path.exists():
                self.logger.debug(f"ラベルファイルが見つかりません: {label_path}")
                return None
            
            # ラベルファイルの読み込み
            labels = np.fromfile(label_path, dtype=np.uint32)
            
            # 下位16bitがセマンティクスラベル
            sem_labels = labels & 0xFFFF
            
            # learning_mapを適用
            max_key = max(self.learning_map.keys()) if self.learning_map else 0
            remap_lut = np.zeros((max_key + 100), dtype=np.int32)
            for k, v in self.learning_map.items():
                remap_lut[k] = v
            
            # セマンティクスラベルをマッピング
            mapped_labels = remap_lut[sem_labels]
            
            return mapped_labels.astype(np.int32)
            
        except Exception as e:
            self.logger.warning(f"ラベル読み込みエラー {seq}/{frame}: {e}")
            return None
    
    def load_ground_labels(self, seq: str, frame: int) -> Optional[np.ndarray]:
        """地面ラベルを読み込み（patchwork）"""
        if self.patchwork_path is None:
            return None
        
        try:
            label_path = self.patchwork_path / seq / f"{frame:06d}.label"
            if label_path.exists():
                ground_labels = np.fromfile(label_path, dtype=np.uint32)
                return ground_labels
            else:
                self.logger.debug(f"地面ラベルファイルが見つかりません: {label_path}")
                return None
                
        except Exception as e:
            self.logger.warning(f"地面ラベル読み込みエラー {seq}/{frame}: {e}")
            return None
    
    def get_class_color(self, class_id: int) -> Tuple[float, float, float]:
        """クラスIDに対応する色を取得（0-1の範囲）"""
        if class_id in self.class_colors:
            color = self.class_colors[class_id]
            return (color[0]/255.0, color[1]/255.0, color[2]/255.0)
        else:
            # 未知のクラスは白色
            return (1.0, 1.0, 1.0)
    
    def get_class_name(self, class_id: int) -> str:
        """クラスIDに対応するクラス名を取得"""
        return self.label_to_names.get(class_id, f"unknown_{class_id}")
    
    def clear_cache(self):
        """キャッシュをクリア"""
        self._frame_cache.clear()
        self._pose_cache.clear()
        self.logger.debug("キャッシュをクリアしました") 