import torch
import numpy as np
from lib.helper_ply import read_ply, write_ply
from torch.utils.data import Dataset
import MinkowskiEngine as ME
import random
import os
import open3d as o3d
from lib.aug_tools import rota_coords, scale_coords, trans_coords
from lib.utils import get_kmeans_labels

from typing import List, Tuple
class cfl_collate_fn:

    def __call__(self, list_data):
        coords, feats, normals, labels, inverse_map, pseudo, inds, region, index = list(zip(*list_data))
        coords_batch, feats_batch, normal_batch, labels_batch, inverse_batch, pseudo_batch, inds_batch = [], [], [], [], [], [], []
        region_batch = []
        accm_num = 0
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            normal_batch.append(torch.from_numpy(normals[batch_id]))
            labels_batch.append(torch.from_numpy(labels[batch_id]).int())
            inverse_batch.append(torch.from_numpy(inverse_map[batch_id]))
            pseudo_batch.append(torch.from_numpy(pseudo[batch_id]))
            inds_batch.append(torch.from_numpy(inds[batch_id] + accm_num).int())
            region_batch.append(torch.from_numpy(region[batch_id])[:,None])
            accm_num += coords[batch_id].shape[0]

        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()#.int()
        feats_batch = torch.cat(feats_batch, 0).float()
        normal_batch = torch.cat(normal_batch, 0).float()
        labels_batch = torch.cat(labels_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()
        pseudo_batch = torch.cat(pseudo_batch, -1)
        inds_batch = torch.cat(inds_batch, 0)
        region_batch = torch.cat(region_batch, 0)

        return coords_batch, feats_batch, normal_batch, labels_batch, inverse_batch, pseudo_batch, inds_batch, region_batch, index


class cfl_collate_fn_temporal:

    def __call__(self, list_data):
        coords_q, coords_k, segs_q, segs_k = list(zip(*list_data))
        coords_q_batch, coords_k_batch = [], []
        for batch_id, _ in enumerate(coords_q):
            num_points_q = coords_q[batch_id].shape[0]
            coords_q_batch.append(torch.cat((torch.ones(num_points_q, 1).int() * batch_id, torch.from_numpy(coords_q[batch_id]).int()), 1))

        for batch_id, _ in enumerate(coords_k):
            num_points_k = coords_k[batch_id].shape[0]
            coords_k_batch.append(torch.cat((torch.ones(num_points_k, 1).int() * batch_id, torch.from_numpy(coords_k[batch_id]).int()), 1))

        # Concatenate all lists
        coords_q_batch = torch.cat(coords_q_batch, 0).float()
        coords_k_batch = torch.cat(coords_k_batch, 0).float()
        
        segs_q = [torch.from_numpy(seg_q) for seg_q in segs_q]
        segs_k = [torch.from_numpy(seg_k) for seg_k in segs_k]
        return coords_q_batch, coords_k_batch, segs_q, segs_k

class KITTItrain(Dataset):
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

        '''Initial Augmentations'''
        self.trans_coords = trans_coords(shift_ratio=50)  ### 50%
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


    def __len__(self):
        return len(self.file_selected)

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

        # ''' Take Mixup as an Augmentation'''
        inds = np.arange(coords.shape[0])
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

            # region = np.array([3, -1, 2, 3, 5, -1])が
            valid_region = region[region != -1]
            unique_vals = np.unique(valid_region)
            unique_vals.sort()
            valid_region = np.searchsorted(unique_vals, valid_region)

            region[region != -1] = valid_region
            # region = np.array([1, -1, 0, 1, 2, -1])のようになる

            pseudo = -np.ones_like(labels).astype(np.long)

        else:
            normals = np.zeros_like(coords)
            scene_name = self.name[index]
            file_path = self.args.pseudo_label_path + '/' + scene_name + '.npy'
            pseudo = np.array(np.load(file_path), dtype=np.long)


        return coords, feats, normals, labels, inverse_map, pseudo, inds, region, index



class KITTItemporal(Dataset):
    def __init__(self, args):
        self.args = args
        self.n_clusters = 0
        self.seq_to_scan_num = {0: 4541, 1: 1101, 2: 4661, 3: 801, 4: 271, 5: 2761, 6: 1101, 7: 1101, 9: 1591, 10: 1201}
        scan_range = list(range(-1*self.args.scan_window+1, self.args.scan_window))
        scan_range.remove(0)
        
        self.file = []
        seq_list = np.sort(os.listdir(self.args.data_path))
        for seq_id in seq_list:
            seq_path = os.path.join(self.args.data_path, seq_id)
            if seq_id in ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']:
                for f in np.sort(os.listdir(seq_path)):
                    self.file.append(os.path.join(seq_path, f))
        
        self.scene_locates, self.scene_diff_locates, self.window_start_locates = self._random_select_samples(scan_range) # [(seq, idx), ...]
        
        self.trans_coords = trans_coords(shift_ratio=50)  ### 50%
        self.rota_coords = rota_coords(rotation_bound = ((-np.pi/32, np.pi/32), (-np.pi/32, np.pi/32), (-np.pi, np.pi)))
        self.scale_coords = scale_coords(scale_bound=(0.9, 1.1))
        
        
    def _random_select_samples(self, scan_range:List[int]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
        scene_idx = np.random.choice(19130, self.args.select_num, replace=False).tolist()
        scene_locates = []
        for idx in scene_idx:
            scan_num = 0
            for seq, seq_scan_num in self.seq_to_scan_num.items():
                if idx < scan_num + seq_scan_num:
                    scene_locates.append((seq, idx - scan_num))
                    break
                scan_num += seq_scan_num
        
        scene_diff = np.random.choice(scan_range, self.args.select_num, replace=True).tolist()
        window_pattern = [random.randint(0, self.args.scan_window-abs(diff)-1) for diff in scene_diff]
        scene_diff_locates = []
        window_start_locates = []
        for (seq, idx), diff, pat in zip(scene_locates, scene_diff, window_pattern):
            t2_idx = idx + diff
            # この処理だと稀にt1とt2が重なることがあるけどそれもよき
            if t2_idx >= self.seq_to_scan_num[seq]:
                t2_idx = self.seq_to_scan_num[seq] - 1
            elif t2_idx < 0:
                t2_idx = 0
            scene_diff_locates.append((seq, t2_idx))
            # self.window_start_locate.append((seq, x))をしたい。ただし、idxとt2_idxを含むウィンドウの候補の複数考えられる場合があるので、候補からランダムにウィンドウを決定したときの最初のインデックスをxとする。
            window_idx = min(idx, t2_idx) - pat
            if window_idx < 0:
                window_idx = 0
            elif window_idx > self.seq_to_scan_num[seq] - self.args.scan_window:
                window_idx = self.seq_to_scan_num[seq] - self.args.scan_window
            window_start_locates.append((seq, window_idx))
    
        return scene_locates, scene_diff_locates, window_start_locates


    def __getitem__(self, index):
        seq_t1, idx_t1 = self.scene_locates[index]
        seq_t2, idx_t2 = self.scene_diff_locates[index]
        coords_t1, *_ = self._get_item_one_scene(seq_t1, idx_t1, aug=True)
        coords_t2, *_ = self._get_item_one_scene(seq_t2, idx_t2, aug=True)
        
        scene_idx_in_window = [(self.window_start_locates[index][0], self.window_start_locates[index][1]+i) for i in range(self.args.scan_window)]
        coords_tn, unique_map_tn, inverse_map_tn, mask_tn = map(list, zip(*[self._get_item_one_scene(seq, idx, False) for seq, idx in scene_idx_in_window]))
        agg_coords, agg_ground_labels, elements_nums = self._aggretate_pcds(scene_idx_in_window, coords_tn, unique_map_tn, mask_tn)
        agg_segs = self._clusterize_pcds(agg_coords, agg_ground_labels)
        segs_tn = []
        for elements_num in elements_nums:
            segs_tn.append(agg_segs[:elements_num])
            agg_segs = agg_segs[elements_num:]
        idx_t1_in_window = scene_idx_in_window.index((seq_t1, idx_t1))
        idx_t2_in_window = scene_idx_in_window.index((seq_t2, idx_t2))
        segs_t1 = segs_tn[idx_t1_in_window]
        segs_t2 = segs_tn[idx_t2_in_window]
        return coords_t1, coords_t2, segs_t1, segs_t2

    
    def _aggretate_pcds(self, scene_idx_in_window, coords_tn, unique_map_tn, mask_tn):
        poses = self._load_poses(scene_idx_in_window[0][0])
        points_set = np.empty((0, 3))
        ground_label = np.empty((0, 1))
        element_nums = []
        for (seq, idx), coords, unique_map, mask in zip(scene_idx_in_window, coords_tn, unique_map_tn, mask_tn):
            pose = poses[idx]
            coords = self._apply_transform(coords, pose)
            g_set = np.fromfile(self._tuple_to_patchwork_path((seq, idx)), dtype=np.uint32)
            g_set = g_set[unique_map]
            g_set = g_set[mask]
            g_set = g_set.reshape((-1))[:, np.newaxis]
            points_set = np.vstack((points_set, coords))
            ground_label = np.vstack((ground_label, g_set))
            element_nums.append(coords.shape[0])
        last_pose = poses[scene_idx_in_window[-1][1]]
        points_set = self._undo_transform(points_set, last_pose)
        return points_set, ground_label, element_nums
        
    
    def _clusterize_pcds(self, agg_coords, agg_ground_labels):
        mask_ground = agg_ground_labels == 1
        mask_ground = mask_ground.flatten()
        non_ground_coords = agg_coords[~mask_ground]
        labels = get_kmeans_labels(self.n_clusters-1, non_ground_coords).detach().cpu().numpy()
        agg_segs = np.zeros_like(agg_ground_labels).flatten()
        agg_segs[~mask_ground] = labels
        agg_segs[mask_ground] = self.n_clusters-1
        return agg_segs
        
        
    def _get_item_one_scene(self, seq:int, idx:int, aug:bool=True):
        coords, feats, labels = self._load_ply(seq, idx)
        coords_original = coords.copy()
        means = coords.mean(0)
        coords -= means
        
        coords, feats, labels, unique_map, inverse_map = self._voxelize(coords, feats, labels) # (123008, x) -> (41342, x)
        coords = coords.astype(np.float32)
        
        mask = np.sqrt(((coords*self.args.voxel_size)**2).sum(-1))< self.args.r_crop
        coords, feats, labels = coords[mask], feats[mask], labels[mask] # (41342, x) -> (39521, x)
        
        if aug:
            coords = self._augs(coords)
            coords, feats, labels = self._augment_coords_to_feats(coords, feats, labels)
        else:
            coords = coords_original[unique_map][mask]
        
        return coords, unique_map, inverse_map, mask # オリジナルのtrainでは、coords, pseudo_labels, indsしかつかってない。
    
    
    def _load_ply(self, seq:int, idx:int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ply_path = self._tuple_to_path((seq, idx))
        data = read_ply(ply_path)
        coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T # (123008, 3)
        feats = np.array(data['remission'])[:, np.newaxis] # (123008, 1)
        labels = np.array(data['class']) # (123008,)
        coords = coords.astype(np.float32)
        return coords, feats, labels
    
    
    def _augs(self, coords):
        coords = self.rota_coords(coords)
        coords = self.trans_coords(coords)
        coords = self.scale_coords(coords)
        return coords


    def _augment_coords_to_feats(self, coords, feats, labels=None):
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = (coords - coords_center)
        return norm_coords, feats, labels


    def _voxelize(self, coords, feats, labels) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        scale = 1 / self.args.voxel_size
        coords = np.floor(coords * scale)
        coords, feats, labels, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), feats, labels=labels, ignore_label=-1, return_index=True, return_inverse=True)
        return coords, feats, labels, unique_map, inverse_map
    
    
    def _load_poses(self, seq:int) -> np.ndarray:
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
    
    
    def _apply_transform(self, points, pose):
        hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
        return np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)[:,:3]


    def _undo_transform(self, points, pose):
        hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
        return np.sum(np.expand_dims(hpoints, 2) * np.linalg.inv(pose).T, axis=1)[:,:3]
    
    
    def _parse_calibration(self, filename):
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
    
    
    def _tuple_to_path(self, tup:Tuple[int, int]) -> str:
        return os.path.join(self.args.data_path, str(tup[0]).zfill(2), str(tup[1]).zfill(6) + '.ply')

    def _tuple_to_sp_path(self, tup:Tuple[int, int]) -> str:
        return os.path.join(self.args.sp_path, str(tup[0]).zfill(2), str(tup[1]).zfill(6) + '_superpoint.npy')

    def _tuple_to_psuedo_path(self, tup:Tuple[int, int]) -> str:
        return os.path.join(self.args.pseudo_label_path, str(tup[0]).zfill(2), str(tup[1]).zfill(6) + '.npy')
    
    def _seq_to_calib_path(self, seq:int) -> str:
        return os.path.join(self.args.original_data_path, str(seq).zfill(2), 'calib.txt')
    
    def _seq_to_poses_path(self, seq:int) -> str:
        return os.path.join(self.args.original_data_path, str(seq).zfill(2), 'poses.txt')
    
    def _tuple_to_patchwork_path(self, tup:Tuple[int, int]) -> str:
        return os.path.join(self.args.patchwork_path, str(tup[0]).zfill(2), str(tup[1]).zfill(6) + '.label')

    def __len__(self):
        return len(self.scene_locates)
    


    

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
            if self.split == 'val':
                if seq_id == '08':
                    for f in np.sort(os.listdir(seq_path)):
                        self.file.append(os.path.join(seq_path, f))
                        self.name.append(os.path.join(seq_path, f)[0:-4].replace(self.args.data_path, ''))


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

        coords, feats, _, unique_map, inverse_map = self.voxelize(coords, feats, labels)
        coords = coords.astype(np.float32)

        region_file = self.args.sp_path + '/' +self.name[index] + '_superpoint.npy'
        region = np.load(region_file)
        region = region[unique_map]

        coords, feats, labels = self.augment_coords_to_feats(coords, feats, labels)
        labels = labels -1

        return coords, feats, np.ascontiguousarray(labels), inverse_map, region, index


class cfl_collate_fn_val:

    def __call__(self, list_data):
        coords, feats, labels, inverse_map, region, index = list(zip(*list_data))
        coords_batch, feats_batch, inverse_batch, labels_batch = [], [], [], []
        region_batch = []
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(
                torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            inverse_batch.append(torch.from_numpy(inverse_map[batch_id]))
            labels_batch.append(torch.from_numpy(labels[batch_id]).int())
            region_batch.append(torch.from_numpy(region[batch_id])[:, None])
        #
        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()
        feats_batch = torch.cat(feats_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()
        labels_batch = torch.cat(labels_batch, 0).int()
        region_batch = torch.cat(region_batch, 0)

        return coords_batch, feats_batch, inverse_batch, labels_batch, index, region_batch
