import torch
import numpy as np
from lib.helper_ply import read_ply
from torch.utils.data import Dataset
import MinkowskiEngine as ME
import os
import open3d as o3d
from lib.aug_tools import rota_coords, scale_coords, trans_coords
import array
from typing import Optional


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




class KITTItrain(Dataset):
    def __init__(self, args, scene_idx, mode, scene_idx_diff=None, window_pattern=None, split='train'):
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
        self.mode = mode
        self.split = split
        self.val_split = '08'
        self.seq_to_scan_num = {0: 4541, 1: 1101, 2: 4661, 3: 801, 4: 271, 5: 2761, 6: 1101, 7: 1101, 9: 1591, 10: 1201}

        # seq_list = np.sort(os.listdir(self.args.data_path))
        # for seq_id in seq_list:
        #     seq_path = os.path.join(self.args.data_path, seq_id)
        #     if self.split == 'train':
        #         if seq_id in self.train_seq_ids:
        #             for f in np.sort(os.listdir(seq_path)):
        #                 self.file.append(os.path.join(seq_path, f))
        #                 self.file_nums[seq_id] += 1

        #     elif self == 'val':
        #         if seq_id == '08':
        #             for f in np.sort(os.listdir(seq_path)):
        #                 self.file.append(os.path.join(seq_path, f))
        #             scene_idx = range(len(self.file))

        '''Initial Augmentations'''
        self.trans_coords = trans_coords(shift_ratio=50)  ### 50%
        self.rota_coords = rota_coords(rotation_bound = ((-np.pi/32, np.pi/32), (-np.pi/32, np.pi/32), (-np.pi, np.pi)))
        self.scale_coords = scale_coords(scale_bound=(0.9, 1.1))
        
        self.random_select_sample(scene_idx, scene_idx_diff, window_pattern)
    
    

    def random_select_sample(self, scene_idx:list[int], scene_idx_diff:Optional[list[int]]=None, window_pattern:Optional[list[int]]=None):
        self.scene_locate = []
        for idx in scene_idx:
            scan_num = 0
            for seq, seq_scan_num in self.seq_to_scan_num.items():
                if idx < scan_num + seq_scan_num:
                    self.scene_locate.append((seq, idx - scan_num))
                    break
                scan_num += seq_scan_num
        
        if self.mode == 'train':
            self.scene_diff_locate = []
            self.window_start_locate = []
            for (seq, idx), diff, pat in zip(self.scene_locate, scene_idx_diff, window_pattern):
                t2_idx = idx + diff
                # この処理だと稀にt1とt2が重なることがあるけどそれもよき
                if t2_idx >= self.seq_to_scan_num[seq]:
                    t2_idx = self.seq_to_scan_num[seq]
                elif t2_idx < 0:
                    t2_idx = 0
                self.scene_diff_locate.append((seq, t2_idx))
                # self.window_start_locate.append((seq, x))をしたい。ただし、idxとt2_idxを含むウィンドウの候補の複数考えられる場合があるので、候補からランダムにウィンドウを決定したときの最初のインデックスをxとする。
                window_idx = min(idx, t2_idx) - pat
                if window_idx < 0:
                    window_idx = 0
                elif window_idx > self.seq_to_scan_num[seq] - self.args.scan_window + 1:
                    window_idx = self.seq_to_scan_num[seq] - self.args.scan_window + 1
                self.window_start_locate.append((seq, window_idx))
    
    
    def tuple_to_path(self, tup:tuple[int, int]) -> str:
        return os.path.join(self.args.data_path, str(tup[0]).zfill(2), str(tup[1]).zfill(6) + '.ply')

    def tuple_to_sp_path(self, tup:tuple[int, int]) -> str:
        return os.path.join(self.args.sp_path, str(tup[0]).zfill(2), str(tup[1]).zfill(6) + '_superpoint.npy')

    def tuple_to_psuedo_path(self, tup:tuple[int, int]) -> str:
        return os.path.join(self.args.pseudo_label_path, str(tup[0]).zfill(2), str(tup[1]).zfill(6) + '.npy')
    
    def tuple_to_calib_path(self, tup:tuple[int, int]) -> str:
        return os.path.join("~/dataset/semantickitti/dataset/sequences/", str(tup[0]).zfill(2), 'calib.txt')
    
    def tuple_to_poses_path(self, tup:tuple[int, int]) -> str:
        return os.path.join("~/dataset/semantickitti/dataset/sequences/", str(tup[0]).zfill(2), 'poses.txt')
    
    # def tuple_to_patchwork_path(self, tup:tuple[int, int]) -> str:
    
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
        return coords, feats, labels, unique_map, inverse_map


    def __len__(self):
        return len(self.scene_locate)
    

    def __getitem__(self, index): # 近い時間のを2つ選ぼうよ
        file = self.tuple_to_path(self.scene_locate[index])
        data = read_ply(file)
        coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
        feats = np.array(data['remission'])[:, np.newaxis]
        labels = np.array(data['class'])
        coords = coords.astype(np.float32)
        coords -= coords.mean(0)

        coords, feats, labels, unique_map, inverse_map = self.voxelize(coords, feats, labels)
        coords = coords.astype(np.float32)

        mask = np.sqrt(((coords*self.args.voxel_size)**2).sum(-1))< self.args.r_crop
        coords, feats, labels = coords[mask], feats[mask], labels[mask]

        region_file = self.args.sp_path + '/' +self.name[index] + '_superpoint.npy'
        region = np.load(region_file)
        region = region[unique_map]
        region = region[mask]

        coords = self.augs(coords)

        ''' Take Mixup as an Augmentation'''
        inds = np.arange(coords.shape[0])
        # mix = random.randint(0, len(self.name)-1)

        # data_mix = read_ply(self.file_selected[mix])
        # coords_mix = np.array([data_mix['x'], data_mix['y'], data_mix['z']], dtype=np.float32).T
        # feats_mix = np.array(data_mix['remission'])[:, np.newaxis]
        # labels_mix = np.array(data_mix['class'])
        # feats_mix = feats_mix.astype(np.float32)
        # coords_mix = coords_mix.astype(np.float32)
        # coords_mix -= coords_mix.mean(0)

        # coords_mix, feats_mix, _, unique_map_mix, _ = self.voxelize(coords_mix, feats_mix, labels_mix)
        # coords_mix = coords_mix.astype(np.float32)

        # mask_mix = np.sqrt(((coords_mix * self.args.voxel_size) ** 2).sum(-1)) < self.args.r_crop
        # coords_mix, feats_mix = coords_mix[mask_mix], feats_mix[mask_mix]
        # #
        # coords_mix = self.augs(coords_mix)
        # coords = np.concatenate((coords, coords_mix), axis=0)
        ''' End Mixup'''

        coords, feats, labels = self.augment_coords_to_feats(coords, feats, labels)
        labels -= 1
        labels[labels == self.args.ignore_label - 1] = self.args.ignore_label

        '''mode must be cluster or train'''
        if self.mode == 'cluster':
            pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(coords[inds])
            pcd.points = o3d.utility.Vector3dVector(coords)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
            normals = np.array(pcd.normals)

            region[labels==-1] = -1

            for q in np.unique(region):
                mask = q == region
                if mask.sum() < self.args.drop_threshold and q != -1:
                    region[mask] = -1

            valid_region = region[region != -1]
            unique_vals = np.unique(valid_region)
            unique_vals.sort()
            valid_region = np.searchsorted(unique_vals, valid_region)

            region[region != -1] = valid_region

            pseudo = -np.ones_like(labels).astype(np.long)
            
            # t2
            file_k = self.tuple_to_path(self.scene_diff_locate[index])
            data_k = read_ply(file_k)
            coords_k = np.array([data_k['x'], data_k['y'], data_k['z']], dtype=np.float32).T
            feats_k = np.array(data_k['remission'])[:, np.newaxis]
            labels_k = np.array(data_k['class'])
            coords_k = coords_k.astype(np.float32)
            coords_k -= coords_k.mean(0)

            coords_k, feats_k, labels_k, unique_map_k, inverse_map_k = self.voxelize(coords_k, feats_k, labels_k)
            coords_k = coords_k.astype(np.float32)
            
            mask_k = np.sqrt(((coords_k*self.args.voxel_size)**2).sum(-1))< self.args.r_crop
            coords_k, feats_k, labels_k = coords_k[mask_k], feats_k[mask_k], labels_k[mask_k]
            
            region_file_k = self.tuple_to_sp_path(self.scene_diff_locate[index])
            region_k = np.load(region_file_k)
            region_k = region_k[unique_map_k]
            region_k = region_k[mask_k]
            
            coords_k = self.augs(coords_k)
            
            inds_k = np.arange(coords_k.shape[0])
            
            coords_k, feats_k, labels_k = self.augment_coords_to_feats(coords_k, feats_k, labels_k)
            labels_k -= 1
            labels_k[labels_k == self.args.ignore_label - 1] = self.args.ignore_label
            
            pcd_k = o3d.geometry.PointCloud()
            pcd_k.points = o3d.utility.Vector3dVector(coords_k)
            pcd_k.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
            normals_k = np.array(pcd_k.normals)
            
            region_k[labels_k==-1] = -1
            
            for q in np.unique(region_k):
                mask = q == region_k
                if mask.sum() < self.args.drop_threshold and q != -1:
                    region_k[mask] = -1
            
            valid_region_k = region_k[region_k != -1]
            unique_vals_k = np.unique(valid_region_k)
            unique_vals_k.sort()
            valid_region_k = np.searchsorted(unique_vals_k, valid_region_k)
            
            region_k[region_k != -1] = valid_region_k
            
            pseudo_k = -np.ones_like(labels_k).astype(np.long)
            
            coord_list, feat_list, normal_list, inverse_map_list, region_list = self._agg_and_seg(index)
            
            return coords, feats, normals, labels, inverse_map, pseudo, inds, region, index, \
                   coords_k, feats_k, normals_k, labels_k, inverse_map_k, pseudo_k, inds_k, region_k,\
                   coord_list, feat_list, normal_list, inverse_map_list, region_list

        elif self.mode == 'train':
            normals = np.zeros_like(coords)
            file_path = self.tuple_to_psuedo_path(self.scene_locate[index])
            pseudo = np.array(np.load(file_path), dtype=np.long)
            return coords, feats, normals, labels, inverse_map, pseudo, inds, region, index

    # SPの作成にはmodel_kが必要なので、ここでは複数ファイルのデータのまとめを(1)get_itemとほぼ同じ処理をしたものと(2)各pcdをvoxelize→poseをapply_transformしたものを返す
    # データ拡張どうするか
    def _agg_and_seg(self, index:int):
        ids_in_window = [self.window_start_locate[index][1]+i for i in range(self.args.scan_window)]
        files = [self.tuple_to_path((self.window_start_locate[index][0], self.window_start_locate[index]+i)) for i in range(self.args.scan_window)]
        datas = [read_ply(file) for file in files]
        coords = [np.array([data['x'], data['y'], data['z']], dtype=np.float32).T for data in datas]
        feats = [np.array(data['remission'])[:, np.newaxis] for data in datas]
        labels = [np.array(data['class']) for data in datas]
        coords = [coord.astype(np.float32) for coord in coords]
        
        '''begin prepare world coords'''
        world_coords, *_ = [self.voxelize(coord, feat, label) for coord, feat, label in zip(coords, feats, labels)]
        poses = self._load_poses(index)
        world_coords = [self._apply_transform(coord, poses[id]) for coord, id in zip(world_coords, ids_in_window)]
        '''end prepare world coords'''
        
        coords = [coord - coord.mean(0) for coord in coords]
        coords, feats, labels, unique_maps, inverse_maps = [self.voxelize(coord, feat, label) for coord, feat, label in zip(coords, feats, labels)]
        coords = [coord.astype(np.float32) for coord in coords]
        
        masks = [np.sqrt(((coord*self.args.voxel_size)**2).sum(-1)) < self.args.r_crop for coord in coords]
        coords = [coord[mask] for coord, mask in zip(coords, masks)]
        feats = [feat[mask] for feat, mask in zip(feats, masks)]
        labels = [label[mask] for label, mask in zip(labels, masks)]
        
        regions = [np.load(self.tuple_to_sp_path((self.window_start_locate[index][0], self.window_start_locate[index]+i))) for i in range(self.args.scan_window)]
        regions = [region[unique_map] for region, unique_map in zip(regions, unique_maps)]
        regions = [region[mask] for region, mask in zip(regions, masks)]
        
        coords = [self.augs(coord) for coord in coords] # これはしない
        coords, feats, labels = [self.augment_coords_to_feats(coord, feat, label) for coord, feat, label in zip(coords, feats, labels)]
        labels = [label - 1 for label in labels]
        labels = [label if label != self.args.ignore_label - 1 else self.args.ignore_label for label in labels]

        pcds = [o3d.geometry.PointCloud() for _ in range(self.args.scan_window)]
        normals = []
        for i, (pcd, coord) in enumerate(zip(pcds, coords)):
            pcd.points = o3d.utility.Vector3dVector(coord)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
            normals[i] = np.array(pcd.normals)
            
        for region in regions:
            region[labels==-1] = -1
            
        for region in regions:
            for q in np.unique(region):
                mask = q == region
                if mask.sum() < self.args.drop_threshold and q != -1:
                    region[mask] = -1
        
        valid_regions = [region[region != -1] for region in regions]
        unique_vals = [np.unique(valid_region) for valid_region in valid_regions]
        # unique_vals.sort() sortはしてくれてるはず
        valid_regions = [np.searchsorted(unique_val, valid_region) for unique_val, valid_region in zip(unique_vals, valid_regions)]
        
        for region, valid_region in zip(regions, valid_regions):
            region[region != -1] = valid_region
        
        return coords, feats, normals, inverse_maps, regions, world_coords
        
    def _load_poses(self, index:int):
        calib_fname = self.tuple_to_calib_path((self.window_start_locate[index]))
        poses_fname = self.tuple_to_poses_path((self.window_start_locate[index]))
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
    
    
    def _apply_transform(points, pose):
        hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
        return np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)[:,:3]


    def _undo_transform(points, pose):
        hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
        return np.sum(np.expand_dims(hpoints, 2) * np.linalg.inv(pose).T, axis=1)[:,:3]
    
    
    def _parse_calibration(filename):
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


'''
# current_growspがNoneの場合には、qだけを計算すれば良くて、kやlistは計算する必要がない！？
# 出力に追加で欲しいもの：projection_headにいれるためのSPのラベル
def get_kittisp_feature(args, loader:DataLoader, model_q:Res16FPNBase, model_k:Res16FPNBase, current_growsp:Optional[int]):
    print('computing point feats ....')
    point_feats_list_q = []
    point_feats_list_k = []
    point_labels_list_q = []
    point_labels_list_k = []
    all_sp_index_q = []
    all_sp_index_k = []
    model_q.eval()
    model_k.eval()
    context_q = []
    context_k = []
    
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(loader)):
            coords_q, features_q, normals_q, labels_q, inverse_map_q, pseudo_labels_q, inds_q, region_q, index, coords_k, feats_k, normals_k, labels_k, inverse_map_k, pseudo_k, inds_k, region_k, coord_list, feat_list, normal_list, inverse_map_list, region_list = data
            
            region_q = region_q.squeeze()
            region_k = region_k.squeeze()
            coord_list = [coord.squeeze() for coord in coord_list]
            scene_locate_q = loader.scene_locate[index]
            scene_locate_k = loader.scene_diff_locate[index]
            scene_locate_list = [(loader.window_start_locate[index][0], loader.window_start_locate[index][1]+i) for i in range(args.scan_window)]
            gt_k = labels_k.clone()
            gt_q = labels_q.clone()
            raw_region_q = region_q.clone()
            raw_region_k = region_k.clone()
            raw_region_list = [region.clone() for region in region_list]

            in_field_q = ME.TensorField(coords_q[:, 1:]*args.voxel_size, coords_q, device='cpu')
            in_field_k = ME.TensorField(coords_k[:, 1:]*args.voxel_size, coords_k, device='cpu')
            in_field_list = [ME.TensorField(coord[:, 1:]*args.voxel_size, coord, device='cpu') for coord in coord_list]

            feats_q = model_q(in_field_q).cuda()
            feats_k = model_k(in_field_k).cuda()
            feat_list_q = [model_q(in_field).cuda() for in_field in in_field_list]
            feat_list_k = [model_k(in_field).cuda() for in_field in in_field_list]
            
            # feats = feats[inds.long()]

            valid_mask_q = region_q!=-1
            valid_mask_k = region_k!=-1
            valid_mask_list = [region!=-1 for region in region_list]
            # features = features[inds.long()].cuda()
            features_q = features_q[valid_mask_q].cuda()
            features_k = feats_k[valid_mask_k].cuda()
            features_list = [feat[valid_mask].cuda() for feat, valid_mask in zip(feat_list, valid_mask_list)]
            # normals = normals[inds.long()].cuda()
            normals_q = normals_q[valid_mask_q].cuda()
            normals_k = normals_k[valid_mask_k].cuda()
            normals_list = [normal[valid_mask].cuda() for normal, valid_mask in zip(normal_list, valid_mask_list)]
            feats_q = feats_q[valid_mask_q]
            feats_k = feats_k[valid_mask_k]
            feats_q_list = [feat[valid_mask] for feat, valid_mask in zip(feat_list_q, valid_mask_list)]
            feats_k_list = [feat[valid_mask] for feat, valid_mask in zip(feat_list_k, valid_mask_list)]
            labels_q = labels_q[valid_mask_q]
            labels_k = labels_k[valid_mask_k]
            region_q = region_q[valid_mask_q].long()
            region_k = region_k[valid_mask_k].long()
            region_list = [region[valid_mask].long() for region, valid_mask in zip(region_list, valid_mask_list)]
            ##
            pc_remission_q = features_q
            pc_remission_k = features_k
            pc_remission_list = [feat for feat in features_list]
            ##
            region_num_q = len(torch.unique(region_q))
            region_num_k = len(torch.unique(region_k))
            region_num_list = [len(torch.unique(region)) for region in region_list]
            region_corr_q = torch.zeros(region_q.size(0), region_num_q)
            region_corr_k = torch.zeros(region_k.size(0), region_num_k)
            region_corr_list = [torch.zeros(region.size(0), region_num) for region, region_num in zip(region_list, region_num_list)]
            region_corr_q.scatter_(1, region_q.view(-1, 1), 1)
            region_corr_k.scatter_(1, region_k.view(-1, 1), 1)
            for region_coor, region_list in zip(region_corr_list, region_list):
                region_coor.scatter_(1, region_list.view(-1, 1), 1)
            region_corr_q = region_corr_q.cuda()##[N, M]
            region_corr_k = region_corr_k.cuda()
            region_corr_list = [region_corr.cuda() for region_corr in region_corr_list]
            per_region_num_q = region_corr_q.sum(0, keepdims=True).t()
            per_region_num_k = region_corr_k.sum(0, keepdims=True).t()
            per_region_num_list = [region_corr.sum(0, keepdims=True).t() for region_corr in region_corr_list]
            ###
            region_feats_q = F.linear(region_corr_q.t(), feats_q.t())/per_region_num_q
            region_feats_k = F.linear(region_corr_k.t(), feats_k.t())/per_region_num_k
            region_feats_q_list = [F.linear(region_corr.t(), feat.t())/per_region_num for region_corr, feat, per_region_num in zip(region_corr_list, feats_q_list, per_region_num_list)]
            region_feats_k_list = [F.linear(region_corr.t(), feat.t())/per_region_num for region_corr, feat, per_region_num in zip(region_corr_list, feats_k_list, per_region_num_list)]
            if current_growsp is not None:
                region_feats_q = F.normalize(region_feats_q, dim=-1)
                region_feats_k = F.normalize(region_feats_k, dim=-1)
                region_feats_q_list = [F.normalize(region_feat, dim=-1) for region_feat in region_feats_q_list]
                region_feats_k_list = [F.normalize(region_feat, dim=-1) for region_feat in region_feats_k_list]
                #
                if region_feats_q.size(0) < current_growsp:
                    n_segments_q = region_feats_q.size(0)
                else:
                    n_segments_q = current_growsp
                sp_idx_q = torch.from_numpy(KMeans(n_clusters=n_segments_q, n_init=5, random_state=0, n_jobs=5).fit_predict(region_feats_q.cpu().numpy())).long()
                if region_feats_k.size(0) < current_growsp:
                    n_segments_k = region_feats_k.size(0)
                else:
                    n_segments_k = current_growsp
                sp_idx_k = torch.from_numpy(KMeans(n_clusters=n_segments_k, n_init=5, random_state=0, n_jobs=5).fit_predict(region_feats_k.cpu().numpy())).long()
                # region_feats_listの要素region_featsについて、region_feats.size(0) < current_growspの場合には、エラーを出力
                assert all([region_feat.size(0) >= current_growsp for region_feat in region_feats_q_list]), "region_feat.size(0) < current_growsp"
                assert all([region_feat.size(0) >= current_growsp for region_feat in region_feats_k_list]), "region_feat.size(0) < current_growsp"
                n_segments = current_growsp
                sp_idx_q_list = [torch.from_numpy(KMeans(n_clusters=n_segments, n_init=5, random_state=0, n_jobs=8).fit_predict(region_feat.cpu().numpy())).long() for region_feat in region_feats_q_list]
                sp_idx_k_list = [torch.from_numpy(KMeans(n_clusters=n_segments, n_init=5, random_state=0, n_jobs=8).fit_predict(region_feat.cpu().numpy())).long() for region_feat in region_feats_k_list]
            else:
                feats_q = region_feats_q
                feat_k = region_feats_k
                sp_idx_q = torch.tensor(range(region_feats_q.size(0)))
                sp_idx_k = torch.tensor(range(region_feats_k.size(0)))

            neural_region_q = sp_idx_q[region_q]
            neural_region_k = sp_idx_k[region_k]
            neural_region_q_list = [sp_idx[region] for sp_idx, region in zip(sp_idx_q_list, region_list)]
            neural_region_k_list = [sp_idx[region] for sp_idx, region in zip(sp_idx_k_list, region_list)]
            pfh_q = []
            pfh_k = []
            pfh_list = []

            neural_region_num_q = len(torch.unique(neural_region_q))
            neural_region_num_k = len(torch.unique(neural_region_k))
            neural_region_num_q_list = [len(torch.unique(neural_region)) for neural_region in neural_region_q_list]
            neural_region_num_k_list = [len(torch.unique(neural_region)) for neural_region in neural_region_k_list]
            neural_region_corr_q = torch.zeros(neural_region_q.size(0), neural_region_num_q)
            neural_region_corr_k = torch.zeros(neural_region_k.size(0), neural_region_num_k)
            neural_region_corr_list = [torch.zeros(neural_region.size(0), neural_region_num) for neural_region, neural_region_num in zip(neural_region_list, neural_region_num_list)]
            neural_region_corr_q.scatter_(1, neural_region_q.view(-1, 1), 1)
            neural_region_corr_k.scatter_(1, neural_region_k.view(-1, 1), 1)
            for neural_region_coor, neural_region_list in zip(neural_region_corr_list, neural_region_list):
                neural_region_coor.scatter_(1, neural_region_list.view(-1, 1), 1)
            neural_region_corr_q = neural_region_corr_q.cuda()
            neural_region_corr_k = neural_region_corr_k.cuda()
            per_neural_region_num_q = neural_region_corr_q.sum(0, keepdims=True).t()
            per_neural_region_num_k = neural_region_corr_k.sum(0, keepdims=True).t()
            #
            final_remission_q = F.linear(neural_region_corr_q.t(), pc_remission_q.t())/per_neural_region_num_q
            final_remission_k = F.linear(neural_region_corr_k.t(), pc_remission_k.t())/per_neural_region_num_k
            #
            if current_growsp is not None:
                feats_q = F.linear(neural_region_corr_q.t(), feats_q.t()) / per_neural_region_num_q
                feats_k = F.linear(neural_region_corr_k.t(), feats_k.t()) / per_neural_region_num_k
                feats_q = F.normalize(feats_q, dim=-1)
                feats_k = F.normalize(feats_k, dim=-1)
            #
            for p in torch.unique(neural_region_q):
                if p!=-1:
                    mask = p==neural_region_q
                    pfh_q.append(compute_hist(normals_q[mask]).unsqueeze(0))

            for p in torch.unique(neural_region_k):
                if p!=-1:
                    mask = p==neural_region_k
                    pfh_k.append(compute_hist(normals_k[mask]).unsqueeze(0))
                    
            pfh_q = torch.cat(pfh_q, dim=0)
            pfh_k = torch.cat(pfh_k, dim=0)
            feats_q = F.normalize(feats_q, dim=-1)
            feats_k = F.normalize(feats_k, dim=-1)
            # #
            feats_q = torch.cat((feats_q, args.c_rgb*final_remission_q, args.c_shape*pfh_q), dim=-1)
            feats_k = torch.cat((feats_k, args.c_rgb*final_remission_k, args.c_shape*pfh_k), dim=-1)
            feats_q = F.normalize(feats_q, dim=-1)
            feats_k = F.normalize(feats_k, dim=-1)

            point_feats_list_q.append(feats_q.cpu())
            point_feats_list_k.append(feats_k.cpu())
            point_labels_list_q.append(labels_q.cpu())
            point_labels_list_k.append(labels_k.cpu())

            all_sp_index_q.append(neural_region_q)
            all_sp_index_k.append(neural_region_k)
            context_q.append((scene_locate_q, gt_q, raw_region_q))
            context_k.append((scene_locate_k, gt_k, raw_region_k))
            

            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))

    return point_feats_list_q, point_labels_list_q, all_sp_index_q, context_q, point_feats_list_k, point_labels_list_k, all_sp_index_k, context_k
'''