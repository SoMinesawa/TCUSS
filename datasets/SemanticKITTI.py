import torch
import numpy as np
from lib.helper_ply import read_ply, write_ply
from torch.utils.data import Dataset
import MinkowskiEngine as ME
import random
import os
import sys
import open3d as o3d
import h5py
from lib.aug_tools import rota_coords, scale_coords, trans_coords
from lib.utils import get_kmeans_labels
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional, Union, Any, Set
# import cuml  # GPU版（コメントアウト）

# 移動物体のraw labelリスト（SemanticKITTI定義）
# 252: moving-car, 253: moving-bicyclist, 254: moving-person, 255: moving-motorcyclist,
# 256: moving-on-rails, 257: moving-bus, 258: moving-truck, 259: moving-other-vehicle
MOVING_RAW_LABELS: Set[int] = {252, 253, 254, 255, 256, 257, 258, 259}

# SemanticKITTIの各シーケンスのスキャン数
SEQ_TO_SCAN_NUM: Dict[int, int] = {
    0: 4541, 1: 1101, 2: 4661, 3: 801, 4: 271, 
    5: 2761, 6: 1101, 7: 1101, 9: 1591, 10: 1201
}
TOTAL_TRAIN_SCANS = sum(SEQ_TO_SCAN_NUM.values())  # 19130


def generate_scene_pairs(
    select_num: int, 
    scan_window: int, 
    seed: int
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """シーンペア（t1, t2）を生成する
    
    全GPUで同じ結果を得るため、固定シードを使用する。
    t1をランダムに選択し、t2をt1からscan_window以内でランダムに選択する。
    
    cluster_loaderのシーン数がおおよそselect_numになるように、
    ペア数は select_num // 2 とする。
    （t1とt2がほぼ異なるので、ユニークシーン数 ≈ select_num）
    
    Args:
        select_num: cluster_loaderに使用するおおよそのシーン数
        scan_window: t1からt2を選ぶ際の最大フレーム差
        seed: 乱数シード（全GPUで同じ値を使用すること）
    
    Returns:
        scene_pairs: [(t1_global_idx, t2_global_idx), ...] のリスト（長さ = select_num // 2）
        scene_idx_t1: t1のグローバルインデックスリスト
        scene_idx_t2: t2のグローバルインデックスリスト
    """
    rng = np.random.RandomState(seed)
    
    # ペア数 = select_num // 2 （cluster_loaderのシーン数がselect_num程度になるように）
    num_pairs = select_num // 2
    
    # t1をランダムに選択
    t1_global_indices = rng.choice(TOTAL_TRAIN_SCANS, num_pairs, replace=False).tolist()
    
    scene_pairs = []
    scene_idx_t1 = []
    scene_idx_t2 = []
    
    for t1_global in t1_global_indices:
        # グローバルインデックスからseq, local_idxに変換
        seq, local_idx = _global_to_seq_local(t1_global)
        seq_scan_num = SEQ_TO_SCAN_NUM[seq]
        
        # t2の候補範囲を計算（同じシーケンス内でscan_window以内）
        t2_min = max(0, local_idx - scan_window)
        t2_max = min(seq_scan_num - 1, local_idx + scan_window)
        
        # t1自身を除いた候補からランダム選択
        t2_candidates = [i for i in range(t2_min, t2_max + 1) if i != local_idx]
        if len(t2_candidates) == 0:
            # 候補がない場合（非常にまれ）はt1と同じにする
            t2_local = local_idx
        else:
            t2_local = rng.choice(t2_candidates)
        
        # ローカルインデックスからグローバルインデックスに変換
        t2_global = _seq_local_to_global(seq, t2_local)
        
        scene_pairs.append((t1_global, t2_global))
        scene_idx_t1.append(t1_global)
        scene_idx_t2.append(t2_global)
    
    return scene_pairs, scene_idx_t1, scene_idx_t2


def _global_to_seq_local(global_idx: int) -> Tuple[int, int]:
    """グローバルインデックスからシーケンス番号とローカルインデックスに変換"""
    cumsum = 0
    for seq, scan_num in SEQ_TO_SCAN_NUM.items():
        if global_idx < cumsum + scan_num:
            return seq, global_idx - cumsum
        cumsum += scan_num
    raise ValueError(f"Invalid global index: {global_idx}")


def _seq_local_to_global(seq: int, local_idx: int) -> int:
    """シーケンス番号とローカルインデックスからグローバルインデックスに変換"""
    cumsum = 0
    for s, scan_num in SEQ_TO_SCAN_NUM.items():
        if s == seq:
            return cumsum + local_idx
        cumsum += scan_num
    raise ValueError(f"Invalid sequence: {seq}")


def get_unique_scene_indices(scene_idx_t1: List[int], scene_idx_t2: List[int]) -> List[int]:
    """t1とt2の重複を除いたユニークなシーンインデックスを取得"""
    return list(set(scene_idx_t1 + scene_idx_t2))


class cfl_collate_fn:
    """データセットの出力を適切なフォーマットに変換するコレート関数
    
    Note: KITTItrainではMixupでcoordsだけが連結されるため、
    coordsとfeats/labels/normals/regionのサイズが異なる（feats側は「元シーン側のみ」）。
    一方で、backbone入力用に「coords側（Mixup後）と同じ点数のremission」を
    remission_allとして別途返す。
    
    indsはモデル出力（coordsサイズ）から original 部分を取り出すためのインデックスなので、
    オフセットはcoordsの点数で計算する必要がある。
    
    Returns:
        coords_batch, feats_batch, normal_batch, labels_batch, inverse_batch, 
        pseudo_batch, inds_batch, region_batch, index, feats_sizes, unique_vals_list
        
        feats_sizes: 各シーンのfeatsサイズのリスト（バッチ分割用）
        unique_vals_list: 各シーンのunique_vals（連番ID→元init SP IDマッピング）のリスト
    """

    def __call__(self, list_data):
        coords, feats, normals, labels, inverse_map, pseudo, inds, region, index, unique_vals, remission_all = list(zip(*list_data))
        coords_batch, feats_batch, normal_batch, labels_batch, inverse_batch, pseudo_batch, inds_batch = [], [], [], [], [], [], []
        region_batch = []
        remission_all_batch = []
        feats_sizes = []  # 各シーンのfeatsサイズを記録
        unique_vals_list = []  # 各シーンのunique_vals（連番ID→元init SP IDマッピング）
        accm_coords = 0   # inds用オフセット（coordsのサイズで計算）
        for batch_id, _ in enumerate(coords):
            num_coords = coords[batch_id].shape[0]
            num_feats = feats[batch_id].shape[0]
            feats_sizes.append(num_feats)
            unique_vals_list.append(unique_vals[batch_id])
            coords_batch.append(torch.cat((torch.ones(num_coords, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            normal_batch.append(torch.from_numpy(normals[batch_id]))
            labels_batch.append(torch.from_numpy(labels[batch_id]).int())
            inverse_batch.append(torch.from_numpy(inverse_map[batch_id]))
            pseudo_batch.append(torch.from_numpy(pseudo[batch_id]))
            # indsのオフセットはcoordsの点数で計算（モデル出力がcoordsサイズに対応するため）
            inds_batch.append(torch.from_numpy(inds[batch_id] + accm_coords).int())
            region_batch.append(torch.from_numpy(region[batch_id])[:,None])
            remission_all_batch.append(torch.from_numpy(remission_all[batch_id]))
            accm_coords += num_coords

        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()
        feats_batch = torch.cat(feats_batch, 0).float()
        normal_batch = torch.cat(normal_batch, 0).float()
        labels_batch = torch.cat(labels_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()
        pseudo_batch = torch.cat(pseudo_batch, -1)
        inds_batch = torch.cat(inds_batch, 0)
        region_batch = torch.cat(region_batch, 0)
        remission_all_batch = torch.cat(remission_all_batch, 0).float()

        return (
            coords_batch,
            feats_batch,
            normal_batch,
            labels_batch,
            inverse_batch,
            pseudo_batch,
            inds_batch,
            region_batch,
            index,
            feats_sizes,
            unique_vals_list,
            remission_all_batch,
        )


class KITTItrain(Dataset):
    """学習用のKITTIデータセット"""
    
    def __init__(self, args, scene_idx, split='train'):
        self.args = args
        self.label_to_names = {0: 'unlabeled',
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
                               19: 'traffic-sign'}
        self.mode = 'train'
        self.split = split
        self.val_split = '08'
        self.file = []

        seq_list = np.sort(os.listdir(self.args.data_path))
        for seq_id in seq_list:
            seq_path = os.path.join(self.args.data_path, seq_id)
            if self.split == 'train':
                if seq_id in ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']:
                    for f in np.sort(os.listdir(seq_path)):
                        self.file.append(os.path.join(seq_path, f))

            elif self == 'val':
                if seq_id == '08':
                    for f in np.sort(os.listdir(seq_path)):
                        self.file.append(os.path.join(seq_path, f))
                    scene_idx = range(len(self.file))

        '''初期の拡張処理設定'''
        self.trans_coords = trans_coords(shift_ratio=50)  # 50%
        self.rota_coords = rota_coords(rotation_bound = ((-np.pi/32, np.pi/32), (-np.pi/32, np.pi/32), (-np.pi, np.pi)))
        self.scale_coords = scale_coords(scale_bound=(0.9, 1.1))

        self.random_select_sample(scene_idx)
    
    def __len__(self):
        return len(self.file_selected) 

    def random_select_sample(self, scene_idx):
        self.name = []
        self.file_selected = []
        for i in scene_idx:
            self.file_selected.append(self.file[i])
            self.name.append(self.file[i][0:-4].replace(self.args.data_path, ''))


    def augs(self, coords):
        coords = self.rota_coords(coords)
        coords = self.trans_coords(coords)
        coords = self.scale_coords(coords)
        return coords


    def augment_coords_to_feats(self, coords, feats, labels=None):
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = (coords - coords_center)
        return norm_coords, feats, labels

    def voxelize(self, coords, feats, labels):
        scale = 1 / self.args.voxel_size
        coords = np.floor(coords * scale)
        coords, feats, labels, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), feats, labels=labels, ignore_label=-1, return_index=True, return_inverse=True)
        # return coords.numpy(), feats, labels, unique_map, inverse_map.numpy()
        return coords, feats, labels, unique_map, inverse_map


    def __getitem__(self, index):
        file = self.file_selected[index]
        data = read_ply(file)
        coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T # (123008, 3)
        feats = np.array(data['remission'])[:, np.newaxis] # (123008, 1)
        labels = np.array(data['class']) #(123008,)
        coords = coords.astype(np.float32)
        coords -= coords.mean(0)

        coords, feats, labels, unique_map, inverse_map = self.voxelize(coords, feats, labels) # (123008, x) -> (41342, x)
        coords = coords.astype(np.float32)

        mask = np.sqrt(((coords*self.args.voxel_size)**2).sum(-1))< self.args.r_crop
        coords, feats, labels = coords[mask], feats[mask], labels[mask] # (41342, x) -> (39521, x)

        region_file = self.args.sp_path + '/' +self.name[index] + '_superpoint.npy'
        region = np.load(region_file) # (123008,)
        region = region[unique_map] # (41342,)
        region = region[mask] # (39521,)

        coords = self.augs(coords)

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(coords)
        # o3d.io.write_point_cloud('mixup_before.pcd', pcd)

        # Mixup: model入力のcoords側だけを連結する設計（lossは元シーン側のみ）。
        # ただし backbone 入力で remission を使う場合、coordsと同じ点数のremissionが必要なので、
        # remission_all（coordsと同じ点数）を別途作って返す。
        inds = np.arange(coords.shape[0])
        remission_all = feats  # 元シーン側（coordsと同じ点数）

        '''Start Mixup(if you want to use Mixup, comment out the following)'''
        mix = random.randint(0, len(self.name)-1)

        data_mix = read_ply(self.file_selected[mix])
        coords_mix = np.array([data_mix['x'], data_mix['y'], data_mix['z']], dtype=np.float32).T
        feats_mix = np.array(data_mix['remission'])[:, np.newaxis]
        labels_mix = np.array(data_mix['class'])
        feats_mix = feats_mix.astype(np.float32)
        coords_mix = coords_mix.astype(np.float32)
        coords_mix -= coords_mix.mean(0)

        coords_mix, feats_mix, _, unique_map_mix, _ = self.voxelize(coords_mix, feats_mix, labels_mix)
        coords_mix = coords_mix.astype(np.float32)

        mask_mix = np.sqrt(((coords_mix * self.args.voxel_size) ** 2).sum(-1)) < self.args.r_crop
        coords_mix, feats_mix = coords_mix[mask_mix], feats_mix[mask_mix]
        #
        coords_mix = self.augs(coords_mix)
        coords = np.concatenate((coords, coords_mix), axis=0)
        remission_all = np.concatenate((remission_all, feats_mix), axis=0)
        ''' End Mixup'''
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(coords)
        # o3d.io.write_point_cloud('mixup_after.pcd', pcd)

        coords, feats, labels = self.augment_coords_to_feats(coords, feats, labels) # (67911, 3), (39521, 1), (39521,) -> (67911, 3), (39521, 1), (39521,)
        labels -= 1
        labels[labels == self.args.ignore_label - 1] = self.args.ignore_label

        '''mode must be cluster or train'''
        if self.mode == 'cluster':
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coords[inds])# mixupしなくなったからindsいらない
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
            normals = np.array(pcd.normals)

            region[labels==-1] = -1

            for q in np.unique(region):
                mask = q == region
                if mask.sum() < self.args.drop_threshold and q != -1:
                    region[mask] = -1

            # regionを連番化（GrowSP学習のscatter_操作用）
            # unique_vals: 連番ID → 元のinit SP IDのマッピング
            # STC用にget_kittisp_featureでsp_mappingを計算・保存する際に使用
            valid_region = region[region != -1]
            unique_vals = np.unique(valid_region)
            unique_vals.sort()
            valid_region = np.searchsorted(unique_vals, valid_region)
            region[region != -1] = valid_region

            # np.long は非推奨なので int64 で置換
            pseudo = -np.ones_like(labels).astype(np.int64)

        else:
            normals = np.zeros_like(coords)
            scene_name = self.name[index]
            file_path = os.path.join(self.args.pseudo_label_path, scene_name.lstrip("/") + '.npy')            
            pseudo = np.array(np.load(file_path), dtype=np.int64)
            
            # trainモードでは連番化しないので、unique_valsは空
            unique_vals = np.array([], dtype=np.int64)
            
            # サイズ不一致チェック
            expected_size = len(inds)
            if len(pseudo) != expected_size:
                # デバッグ: 保存時に記録されたサイズを確認
                size_file = file_path.replace('.npy', '_size.txt')
                recorded_size = None
                if os.path.exists(size_file):
                    with open(size_file, 'r') as f:
                        recorded_size = int(f.read().strip())
                
                # デバッグ: 同じファイルを再読み込みして処理し、サイズを確認
                debug_file = self.file_selected[index]
                debug_data = read_ply(debug_file)
                debug_coords = np.array([debug_data['x'], debug_data['y'], debug_data['z']], dtype=np.float32).T
                debug_feats = np.array(debug_data['remission'])[:, np.newaxis]
                debug_labels = np.array(debug_data['class'])
                debug_coords = debug_coords.astype(np.float32)
                debug_coords -= debug_coords.mean(0)
                debug_coords, debug_feats, debug_labels, _, _ = self.voxelize(debug_coords, debug_feats, debug_labels)
                debug_coords = debug_coords.astype(np.float32)
                debug_mask = np.sqrt(((debug_coords * self.args.voxel_size) ** 2).sum(-1)) < self.args.r_crop
                debug_size = debug_mask.sum()
                
                raise ValueError(
                    f"pseudo ファイルのサイズが一致しません: "
                    f"scene={scene_name}, pseudo={len(pseudo)}, expected={expected_size}, "
                    f"labels={len(labels)}, recorded_size={recorded_size}, "
                    f"debug_reprocess_size={debug_size}, ply_file={debug_file}"
                )

        return coords, feats, normals, labels, inverse_map, pseudo, inds, region, index, unique_vals, remission_all


class KITTIval(Dataset):
    def __init__(self, args, split='val'):
        self.args = args
        self.label_to_names = {0: 'unlabeled',
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
                               19: 'traffic-sign'}
        self.name = []
        self.mode = 'val'
        self.split = split
        self.val_split = '08'
        self.file = []

        seq_list = np.sort(os.listdir(self.args.data_path))
        for seq_id in seq_list:
            seq_path = os.path.join(self.args.data_path, seq_id)
            if seq_id in ['08']:
                for f in np.sort(os.listdir(seq_path)):
                    self.file.append(os.path.join(seq_path, f))
                    self.name.append(os.path.join(seq_path, f)[0:-4].replace(self.args.data_path, ''))
        if args.eval_select_num < len(self.file):
            random_indices = random.sample(range(len(self.file)), args.eval_select_num)
            self.file = [self.file[i] for i in random_indices]
            self.name = [self.name[i] for i in random_indices]
    
    def _load_raw_labels(self, seq_id: str, frame_id: str) -> np.ndarray:
        """元の.labelファイルからraw labelsを読み込む
        
        Args:
            seq_id: シーケンスID（例: '08'）
            frame_id: フレームID（例: '000000'）
        
        Returns:
            raw_labels: 元のセマンティックラベル（252-259は移動物体）
        """
        label_path = os.path.join(
            self.args.original_data_path, 
            seq_id, 
            'labels', 
            f'{frame_id}.label'
        )
        if not os.path.exists(label_path):
            raise FileNotFoundError(f'Raw label file not found: {label_path}')
        
        label = np.fromfile(label_path, dtype=np.uint32)
        sem_label = label & 0xFFFF  # semantic label in lower half
        return sem_label.astype(np.int32)

    def augment_coords_to_feats(self, coords, feats, labels=None):
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = (coords - coords_center)
        return norm_coords, feats, labels

    def voxelize(self, coords, feats, labels):
        scale = 1 / self.args.voxel_size
        coords = np.floor(coords * scale)
        coords, feats, labels, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), feats, labels=labels, ignore_label=-1, return_index=True, return_inverse=True)
        # return coords.numpy(), feats, labels, unique_map, inverse_map.numpy()
        return coords, feats, labels, unique_map, inverse_map


    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        file = self.file[index]
        data = read_ply(file)
        coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
        feats = np.array(data['remission'])[:, np.newaxis]
        labels = np.array(data['class'])
        
        # 元座標を保持（距離計算用）
        original_coords = coords.copy()
        
        coords = coords.astype(np.float32)
        coords -= coords.mean(0)

        coords, feats, _, unique_map, inverse_map = self.voxelize(coords, feats, labels)
        coords = coords.astype(np.float32)

        region_file = self.args.sp_path + '/' +self.name[index] + '_superpoint.npy'
        region = np.load(region_file)
        region = region[unique_map]

        coords, feats, labels = self.augment_coords_to_feats(coords, feats, labels)
        
        # raw labelsを読み込む（移動物体判定用）
        # name[index]は '/08/000000' のような形式
        name_parts = self.name[index].strip('/').split('/')
        seq_id = name_parts[0]  # '08'
        frame_id = name_parts[1]  # '000000'
        raw_labels = self._load_raw_labels(seq_id, frame_id)
        
        # NOTE:
        # growspのPLY内ラベルは 0..19（semantic-kitti.yaml の learning_map と同じ）。
        # 学習/テスト推論は labels -= 1 により unlabeled(0) を ignore(-1) にし、
        # 1..19 を 0..18 に詰めて semantic_class=19 と整合させている。
        # valだけ前処理が違うと、unlabeledが評価に入って traffic-sign(19) が落ちるため、
        # train/test と同じ前処理に揃える。
        labels -= 1
        labels[labels == self.args.ignore_label - 1] = self.args.ignore_label
        return coords, feats, np.ascontiguousarray(labels), inverse_map, region, index, original_coords, raw_labels


class KITTItest(Dataset):
    def __init__(self, args, split='test'):
        self.args = args
        self.label_to_names = {0: 'unlabeled',
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
                               19: 'traffic-sign'}
        self.name = []
        # self.mode = 'test'
        self.split = split
        self.file = []

        seq_list = np.sort(os.listdir(self.args.data_path))
        # valid_seqs = ['08'] if self.args.debug else [str(i).zfill(2) for i in range(22)]
        valid_seqs = ['08'] + [str(i).zfill(2) for i in range(11, 22)]
        # valid_seqs = ['08']
        for seq_id in seq_list:
            seq_path = os.path.join(self.args.data_path, seq_id)
            if seq_id in valid_seqs:
                for f in np.sort(os.listdir(seq_path)):
                    self.file.append(os.path.join(seq_path, f))
                    self.name.append(os.path.join(seq_path, f)[0:-4].replace(self.args.data_path, ''))
        # random_indices = random.sample(range(len(self.file)), 10)
        # self.file = [self.file[i] for i in random_indices]
        # self.name = [self.name[i] for i in random_indices]


    def augment_coords_to_feats(self, coords, feats, labels=None):
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = (coords - coords_center)
        return norm_coords, feats, labels

    def voxelize(self, coords, feats, labels):
        scale = 1 / self.args.voxel_size
        coords = np.floor(coords * scale)
        coords, feats, labels, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), feats, labels=labels, ignore_label=-1, return_index=True, return_inverse=True)
        # return coords.numpy(), feats, labels, unique_map, inverse_map.numpy()
        return coords, feats, labels, unique_map, inverse_map


    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        file = self.file[index]
        data = read_ply(file)
        coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
        feats = np.array(data['remission'])[:, np.newaxis]
        labels = np.array(data['class'])
        coords = coords.astype(np.float32)
        coords -= coords.mean(0)

        coords, feats, _, _, inverse_map = self.voxelize(coords, feats, labels)
        coords = coords.astype(np.float32)

        coords, feats, labels = self.augment_coords_to_feats(coords, feats, labels)
        labels -= 1
        labels[labels == self.args.ignore_label - 1] = self.args.ignore_label
        return coords, feats, np.ascontiguousarray(labels), inverse_map, index, file


class cfl_collate_fn_val:

    def __call__(self, list_data):
        coords, feats, labels, inverse_map, region, index, original_coords, raw_labels = list(zip(*list_data))
        coords_batch, feats_batch, inverse_batch, labels_batch = [], [], [], []
        region_batch = []
        original_coords_batch = []
        raw_labels_batch = []
        
        # inverse_mapのオフセット計算用（batch_size>1対応）
        accm_voxel_num = 0
        
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(
                torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            # inverse_mapにオフセットを追加（batch_size>1対応）
            inverse_batch.append(torch.from_numpy(inverse_map[batch_id]) + accm_voxel_num)
            accm_voxel_num += num_points  # 次のバッチ用にオフセットを更新
            labels_batch.append(torch.from_numpy(labels[batch_id]).int())
            region_batch.append(torch.from_numpy(region[batch_id])[:, None])
            # 追加: 元座標とraw labels
            original_coords_batch.append(torch.from_numpy(original_coords[batch_id]).float())
            raw_labels_batch.append(torch.from_numpy(raw_labels[batch_id]).int())
        #
        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()
        feats_batch = torch.cat(feats_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()
        labels_batch = torch.cat(labels_batch, 0).int()
        region_batch = torch.cat(region_batch, 0)
        original_coords_batch = torch.cat(original_coords_batch, 0)
        raw_labels_batch = torch.cat(raw_labels_batch, 0)

        return coords_batch, feats_batch, inverse_batch, labels_batch, index, region_batch, original_coords_batch, raw_labels_batch


class cfl_collate_fn_test:

    def __call__(self, list_data):
        coords, feats, labels, inverse_map, index, file_path = list(zip(*list_data))
        coords_batch, feats_batch, inverse_batch, labels_batch = [], [], [], []
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(
                torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            inverse_batch.append(torch.from_numpy(inverse_map[batch_id]))
            labels_batch.append(torch.from_numpy(labels[batch_id]).int())
        #
        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()
        feats_batch = torch.cat(feats_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()
        labels_batch = torch.cat(labels_batch, 0).int()

        return coords_batch, feats_batch, inverse_batch, labels_batch, index, file_path


class KITTIstc(Dataset):
    """STC (Superpoint Time Consistency) 学習用データセット
    
    連続する2フレーム（時刻tとt2）のデータを返す。
    scan_windowに対応し、t2はt1からscan_window以内のフレーム。
    VoteFlowで事前計算されたScene Flowデータ（H5ファイル）を使用する。
    
    データセット側でSP対応計算まで行い、train_stcには対応行列を渡す。
    統合SPラベルはsp_id_pathから読み込む（get_kittisp_featureで保存されたもの）。
    
    シーン選択は外部から set_scene_pairs() で設定する。
    """
    
    def __init__(self, args):
        self.args = args
        
        # scan_window = 1 の強制チェック
        if args.scan_window != 1:
            raise ValueError(
                f"STC（SPレベル直接マッチング）ではscan_window=1のみサポートしています。"
                f"現在の設定: scan_window={args.scan_window}"
            )
        
        self.file = []
        seq_list = np.sort(os.listdir(self.args.data_path))
        for seq_id in seq_list:
            seq_path = os.path.join(self.args.data_path, seq_id)
            if seq_id in ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']:
                for f in np.sort(os.listdir(seq_path)):
                    self.file.append(os.path.join(seq_path, f))
        
        self.scene_locates = []  # [(seq, local_idx), ...] for t1
        self.scene_diff_locates = []  # [(seq, local_idx), ...] for t2
        self.trans_coords = trans_coords(shift_ratio=50)
        self.rota_coords = rota_coords(rotation_bound=((-np.pi/32, np.pi/32), (-np.pi/32, np.pi/32), (-np.pi, np.pi)))
        self.scale_coords = scale_coords(scale_bound=(0.9, 1.1))
        
        # VoteFlow前処理済みH5ファイルのパス
        self.voteflow_h5_path = args.stc.voteflow_preprocess_path
        # H5ファイルハンドルのキャッシュ（シーケンス番号 -> ファイルハンドル）
        self._h5_handles: Dict[int, h5py.File] = {}
        
        # SPマッチング設定（新方式）
        sp_matching = args.stc.sp_matching
        self.weight_centroid_distance = sp_matching.weight_centroid_distance
        self.weight_spread_similarity = sp_matching.weight_spread_similarity
        self.weight_point_count_similarity = sp_matching.weight_point_count_similarity
        self.weight_remission_similarity = getattr(sp_matching, "weight_remission_similarity", 0.0)
        self.max_centroid_distance = sp_matching.max_centroid_distance
        self.min_score_threshold = sp_matching.min_score_threshold
        self.min_sp_points = sp_matching.min_sp_points
        self.remove_ego_motion = sp_matching.remove_ego_motion
        self.exclude_ground = sp_matching.exclude_ground
        
        # 対応計算関数をインポート（SPレベル直接マッチング）
        from scene_flow.correspondence import compute_superpoint_correspondence_direct
        self._compute_sp_correspondence_direct = compute_superpoint_correspondence_direct
    
    def augs(self, coords):
        """データ拡張（KITTItrainと同様）
        
        voxel座標に対して回転、平行移動、スケーリングを適用する。
        t1とt2で別々に呼び出すことで、異なるランダム拡張が適用される。
        """
        coords = self.rota_coords(coords)
        coords = self.trans_coords(coords)
        coords = self.scale_coords(coords)
        return coords
    
    def set_scene_pairs(self, scene_idx_t1: List[int], scene_idx_t2: List[int]):
        """シーンペアを設定する
        
        Args:
            scene_idx_t1: t1のグローバルインデックスリスト
            scene_idx_t2: t2のグローバルインデックスリスト
        """
        self.scene_locates = [_global_to_seq_local(idx) for idx in scene_idx_t1]
        self.scene_diff_locates = [_global_to_seq_local(idx) for idx in scene_idx_t2]
    
    def _get_h5_handle(self, seq: int) -> h5py.File:
        """H5ファイルハンドルを取得（キャッシュ機構付き）"""
        if seq not in self._h5_handles:
            h5_path = os.path.join(self.voteflow_h5_path, f'{str(seq).zfill(2)}.h5')
            if not os.path.exists(h5_path):
                raise FileNotFoundError(f'VoteFlow H5ファイルが見つかりません: {h5_path}')
            self._h5_handles[seq] = h5py.File(h5_path, 'r')
        return self._h5_handles[seq]
    
    def _close_h5_handles(self):
        """H5ファイルハンドルを閉じる"""
        for handle in self._h5_handles.values():
            handle.close()
        self._h5_handles.clear()
    
    def __del__(self):
        """デストラクタでH5ファイルを閉じる"""
        self._close_h5_handles()
    
    def __len__(self):
        return len(self.scene_locates)
    
    def __getitem__(self, index):
        seq_t, idx_t = self.scene_locates[index]
        seq_t2, idx_t2 = self.scene_diff_locates[index]
        
        # scene_name（特徴量取得用）
        scene_name_t = self._tuple_to_scene_name((seq_t, idx_t))
        scene_name_t2 = self._tuple_to_scene_name((seq_t2, idx_t2))
        
        # 時刻tのデータ（座標、統合SPラベル、pose、Scene Flow）
        coords_t, coords_t_original, sp_labels_t, pose_t, flow_t, ground_mask_t, remission_t = self._get_item_one_scene(seq_t, idx_t)
        
        # 時刻t2のデータ
        coords_t2, coords_t2_original, sp_labels_t2, pose_t2, _, ground_mask_t2, remission_t2 = self._get_item_one_scene(seq_t2, idx_t2)
        
        # SP対応行列を計算（SPレベル直接マッチング）
        # scan_window=1なので、flow_tをそのまま使用
        corr_matrix, unique_sp_t, unique_sp_t2 = self._compute_sp_correspondence_direct(
            coords_t_original,
            coords_t2_original,
            flow_t,
            sp_labels_t,
            sp_labels_t2,
            pose_t,
            pose_t2,
            ground_mask_t,
            ground_mask_t2,
            remission_t=remission_t,
            remission_t1=remission_t2,
            weight_centroid_distance=self.weight_centroid_distance,
            weight_spread_similarity=self.weight_spread_similarity,
            weight_point_count_similarity=self.weight_point_count_similarity,
            weight_remission_similarity=self.weight_remission_similarity,
            max_centroid_distance=self.max_centroid_distance,
            min_score_threshold=self.min_score_threshold,
            min_sp_points=self.min_sp_points,
            remove_ego_motion=self.remove_ego_motion,
            exclude_ground=self.exclude_ground
        )
        
        # データ拡張を適用（t1とt2で別々のランダム拡張）
        # 対応計算はオリジナル座標で完了済み、特徴抽出用にデータ拡張を適用
        # voxel座標に対して適用（KITTItrainと同様）
        coords_t_aug = self.augs(coords_t.copy())
        coords_t2_aug = self.augs(coords_t2.copy())

        out = {
            'coords_t': coords_t_aug,  # データ拡張後（特徴抽出用）
            'coords_t2': coords_t2_aug,  # データ拡張後（特徴抽出用）
            'sp_labels_t': sp_labels_t,
            'sp_labels_t2': sp_labels_t2,
            'corr_matrix': corr_matrix,
            'unique_sp_t': unique_sp_t,
            'unique_sp_t2': unique_sp_t2,
            'scene_name_t': scene_name_t,
            'scene_name_t2': scene_name_t2
        }
        # backbone入力にremissionを使う場合は必須
        if int(getattr(self.args, "input_dim", 0)) == 4:
            if remission_t is None or remission_t2 is None:
                raise ValueError(
                    "input_dim=4 ですが remission_t/remission_t2 が取得できませんでした。"
                    "PLYにremissionが存在するか、data_pathが正しいか確認してください。"
                )
            out["remission_t"] = remission_t.reshape(-1, 1).astype(np.float32, copy=False)
            out["remission_t2"] = remission_t2.reshape(-1, 1).astype(np.float32, copy=False)
        return out
    
    def _tuple_to_scene_name(self, tup: Tuple[int, int]) -> str:
        """シーン名を生成（KITTItrainと同じフォーマット）"""
        return f'/{str(tup[0]).zfill(2)}/{str(tup[1]).zfill(6)}'
    
    def _get_item_one_scene(self, seq: int, idx: int):
        """1シーンのデータをH5ファイルから取得
        
        統合SPラベルは、元のinit SPファイルにsp_mappingを適用して作成する。
        
        Args:
            seq: シーケンス番号
            idx: フレームインデックス
            
        Returns:
            coords_vox: voxel化された座標 [N, 3]
            coords_original_masked: オリジナル座標（mask後）[N, 3]
            sp_labels: SPラベル [N]
            pose: ワールド姿勢行列 [4, 4]
            flow_vox: Scene flow [N, 3]
            ground_mask_vox: 地面マスク [N]
        """
        h5_file = self._get_h5_handle(seq)
        timestamp = str(idx).zfill(6)
        
        if timestamp not in h5_file:
            raise KeyError(f'タイムスタンプ {timestamp} がH5ファイル（seq={seq}）に存在しません')
        
        frame_data = h5_file[timestamp]
        
        # H5ファイルからデータを読み込み
        coords = frame_data['lidar'][:]  # (N, 3) Velodyne座標系
        pose = frame_data['pose'][:]  # (4, 4)
        ground_mask = frame_data['ground_mask'][:]  # (N,) bool
        
        # Scene Flowデータを取得（存在する場合）
        if 'flow_est_fixed' in frame_data:
            flow = frame_data['flow_est_fixed'][:]  # (N, 3)
        else:
            flow = np.zeros_like(coords)  # フローがない場合はゼロ
        
        coords_original = coords.copy()
        means = coords.mean(0)
        coords = coords - means
        
        # voxelize
        coords_vox, _, _, unique_map, _ = self._voxelize(coords)
        coords_vox = coords_vox.astype(np.float32)
        
        # r_cropでマスク
        mask = np.sqrt(((coords_vox * self.args.voxel_size) ** 2).sum(-1)) < self.args.r_crop
        coords_vox = coords_vox[mask]
        
        # 統合SPラベルを作成: 元のinit SPにsp_mappingを適用
        # 1. 元のinit SPファイルを読み込む（元の点群サイズ）
        seq_str = str(seq).zfill(2)
        idx_str = str(idx).zfill(6)
        init_sp_file = os.path.join(self.args.sp_path, seq_str, f'{idx_str}_superpoint.npy')
        init_sp_labels = np.load(init_sp_file)  # (N,) 元の点群サイズ
        
        # 2. sp_mappingを読み込む（init SP → 統合SPのマッピング）
        sp_mapping_file = os.path.join(self.args.sp_id_path, seq_str, f'{idx_str}_sp_mapping.npy')
        if os.path.exists(sp_mapping_file):
            sp_mapping = np.load(sp_mapping_file)  # (max_init_sp_id+1,)
            
            # 3. init SPラベルに対してマッピングを適用
            # init_sp_labels == -1 の点（ノイズ）は -1 のまま
            # init_sp_labels >= len(sp_mapping) の場合は -1 にする
            # sp_mapping[init_sp_id] == -1 の場合も -1 にする（サンプリングされなかったSP）
            sp_labels_full = np.full_like(init_sp_labels, -1, dtype=np.int32)
            valid_init_mask = (init_sp_labels >= 0) & (init_sp_labels < len(sp_mapping))
            mapped_values = sp_mapping[init_sp_labels[valid_init_mask]]
            # sp_mapping内の-1は未定義なので、それ以外のみ適用
            sp_labels_full[valid_init_mask] = np.where(mapped_values >= 0, mapped_values, -1)
        else:
            # sp_mappingが存在しない場合（最初のクラスタリング前）はエラー
            raise FileNotFoundError(
                f"sp_mappingファイルが見つかりません: {sp_mapping_file}\n"
                f"STCを使用するには、まずクラスタリングを実行してsp_mappingを生成してください。"
            )
        
        # SPラベル、ground_mask、flowもマッピング
        sp_labels = sp_labels_full[unique_map][mask]
        ground_mask_vox = ground_mask[unique_map][mask]
        flow_vox = flow[unique_map][mask]

        # remission（強度）は必要な場合のみPLYから取得（H5には入っていない想定）
        remission_vox = None
        need_remission = (float(getattr(self, "weight_remission_similarity", 0.0)) != 0.0) or (int(getattr(self.args, "input_dim", 0)) == 4)
        if need_remission:
            ply_path = os.path.join(self.args.data_path, seq_str, f"{idx_str}.ply")
            if not os.path.exists(ply_path):
                raise FileNotFoundError(f"PLYが見つかりません（remission取得用）: {ply_path}")
            ply = read_ply(ply_path)
            if "remission" not in ply.dtype.names:
                raise KeyError(f"PLYに'remission'が存在しません: {ply_path} (fields={ply.dtype.names})")
            remission_full = np.asarray(ply["remission"], dtype=np.float32).reshape(-1)
            if remission_full.shape[0] != coords_original.shape[0]:
                raise ValueError(
                    "PLY remission と H5 lidar の点数が一致しません: "
                    f"ply={remission_full.shape[0]}, h5={coords_original.shape[0]}, ply_path={ply_path}"
                )
            remission_vox = remission_full[unique_map][mask]
        
        # オリジナル座標（mask後）
        coords_original_masked = coords_original[unique_map][mask]
        
        return coords_vox, coords_original_masked, sp_labels, pose, flow_vox, ground_mask_vox, remission_vox
    
    def _voxelize(self, coords):
        """点群をvoxel化"""
        coords = np.ascontiguousarray(coords)

        res = ME.utils.sparse_quantize(
            coords,
            return_index=True,
            return_inverse=True,
            quantization_size=self.args.voxel_size,
            return_maps_only=True
        )

        if len(res) == 2:
            unique_map, inverse_map = res
        else:
            quantized_coords, voxel_idx, inverse_map, voxel_counts = res
            original_idx = np.arange(len(coords))
            unique_map = original_idx[voxel_idx]

        quantized_coords = coords[unique_map]
        quantized_coords = (quantized_coords / self.args.voxel_size)
        
        return quantized_coords, None, None, unique_map, inverse_map
    
    def _tuple_to_sp_path(self, tup):
        """init SPラベルファイルのパス"""
        return os.path.join(self.args.sp_path, str(tup[0]).zfill(2), str(tup[1]).zfill(6) + '_superpoint.npy')
    
    def _tuple_to_sp_id_path(self, tup):
        """統合SPラベルファイルのパス（get_kittisp_featureで保存されたもの）"""
        return os.path.join(self.args.sp_id_path, str(tup[0]).zfill(2), str(tup[1]).zfill(6) + '_sp_id.npy')


class cfl_collate_fn_stc:
    """STC用collate関数
    
    KITTIstcが返すdictをバッチ化する。
    coords_t/coords_t2はMinkowskiEngine用にバッチIDを付与。
    SP対応行列などはリストのまま返す。
    """
    
    def __call__(self, list_data):
        # list_dataは[{...}, {...}, ...]の形式
        coords_t_batch = []
        coords_t2_batch = []
        remission_t_batch = []
        remission_t2_batch = []
        sp_labels_t_list = []
        sp_labels_t2_list = []
        corr_matrix_list = []
        unique_sp_t_list = []
        unique_sp_t2_list = []
        scene_name_t_list = []
        scene_name_t2_list = []
        
        for batch_id, data in enumerate(list_data):
            coords_t = data['coords_t']
            coords_t2 = data['coords_t2']
            remission_t = data.get('remission_t', None)
            remission_t2 = data.get('remission_t2', None)
            
            num_points_t = coords_t.shape[0]
            num_points_t2 = coords_t2.shape[0]
            
            coords_t_batch.append(
                torch.cat((torch.ones(num_points_t, 1).int() * batch_id, 
                          torch.from_numpy(coords_t).int()), 1)
            )
            coords_t2_batch.append(
                torch.cat((torch.ones(num_points_t2, 1).int() * batch_id, 
                          torch.from_numpy(coords_t2).int()), 1)
            )
            if remission_t is not None:
                remission_t_batch.append(torch.from_numpy(remission_t))
            if remission_t2 is not None:
                remission_t2_batch.append(torch.from_numpy(remission_t2))
            
            sp_labels_t_list.append(torch.from_numpy(data['sp_labels_t']).long())
            sp_labels_t2_list.append(torch.from_numpy(data['sp_labels_t2']).long())
            corr_matrix_list.append(torch.from_numpy(data['corr_matrix']).float())
            unique_sp_t_list.append(data['unique_sp_t'])
            unique_sp_t2_list.append(data['unique_sp_t2'])
            scene_name_t_list.append(data['scene_name_t'])
            scene_name_t2_list.append(data['scene_name_t2'])
        
        coords_t_batch = torch.cat(coords_t_batch, 0).float()
        coords_t2_batch = torch.cat(coords_t2_batch, 0).float()
        remission_t_out = torch.cat(remission_t_batch, 0).float() if len(remission_t_batch) > 0 else None
        remission_t2_out = torch.cat(remission_t2_batch, 0).float() if len(remission_t2_batch) > 0 else None
        
        out = {
            'coords_t': coords_t_batch,
            'coords_t2': coords_t2_batch,
            'sp_labels_t': sp_labels_t_list,
            'sp_labels_t2': sp_labels_t2_list,
            'corr_matrix': corr_matrix_list,
            'unique_sp_t': unique_sp_t_list,
            'unique_sp_t2': unique_sp_t2_list,
            'scene_name_t': scene_name_t_list,
            'scene_name_t2': scene_name_t2_list
        }
        if remission_t_out is not None and remission_t2_out is not None:
            out['remission_t'] = remission_t_out
            out['remission_t2'] = remission_t2_out
        return out


class KITTItcuss_stc(Dataset):
    """TCUSS学習用の複合データセット
    
    stc.enabled=True: GrowSP + STC
    stc.enabled=False: GrowSPのみ（STCもTARLも不要）
    
    シーン選択は外部から set_scene_pairs() で設定する。
    これにより、trainsetとclustersetで同じシーンを使用でき、
    マルチGPUでも全プロセスで同じシーンを使用できる。
    """
    
    def __init__(self, args):
        self.args = args
        self.phase = 0
        self.stc_enabled = args.stc.enabled
        
        # GrowSP用データセット（初期状態では空のscene_idx）
        self.kittitrain_t1 = KITTItrain(args, scene_idx=[], split='train')
        self.kittitrain_t2 = KITTItrain(args, scene_idx=[], split='train')
        
        # STC用データセット（STC有効時のみ）
        if self.stc_enabled:
            self.kittistc = KITTIstc(args)
        else:
            self.kittistc = None
        
        self.scene_idx_t1 = []
        self.scene_idx_t2 = []
        self.scene_idx_all = []
    
    def set_scene_pairs(self, scene_idx_t1: List[int], scene_idx_t2: List[int]):
        """シーンペアを設定する
        
        Args:
            scene_idx_t1: t1のグローバルインデックスリスト
            scene_idx_t2: t2のグローバルインデックスリスト
        """
        self.scene_idx_t1 = scene_idx_t1
        self.scene_idx_t2 = scene_idx_t2
        self.scene_idx_all = get_unique_scene_indices(scene_idx_t1, scene_idx_t2)
        
        # GrowSP用データセットにシーンを設定
        self.kittitrain_t1.random_select_sample(scene_idx_t1)
        self.kittitrain_t2.random_select_sample(scene_idx_t2)
        
        # STC用データセットにシーンを設定（STC有効時のみ）
        if self.stc_enabled and self.kittistc is not None:
            self.kittistc.set_scene_pairs(scene_idx_t1, scene_idx_t2)
    
    def __len__(self):
        return len(self.scene_idx_t1)
    
    def __getitem__(self, index):
        growsp_t1 = self.kittitrain_t1.__getitem__(index)
        growsp_t2 = self.kittitrain_t2.__getitem__(index)
        
        if self.stc_enabled:
            stc_data = self.kittistc.__getitem__(index)
        else:
            stc_data = None
        
        return growsp_t1, growsp_t2, stc_data


class cfl_collate_fn_tcuss_stc:
    """TCUSS用collate関数
    
    stc_dataがNoneの場合（STC無効時）はstc=Noneを返す
    """
    
    def __init__(self):
        self.growsp_collate = cfl_collate_fn()
        self.stc_collate = cfl_collate_fn_stc()
    
    def __call__(self, list_data):
        growsp_t1_data, growsp_t2_data, stc_data = list(zip(*list_data))
        
        growsp_t1 = self.growsp_collate(growsp_t1_data)
        growsp_t2 = self.growsp_collate(growsp_t2_data)
        
        # STC無効時はstc_dataがNoneのタプル
        if stc_data[0] is None:
            stc = None
        else:
            stc = self.stc_collate(stc_data)
        
        return growsp_t1, growsp_t2, stc