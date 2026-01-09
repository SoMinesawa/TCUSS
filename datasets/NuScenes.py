import os
import random
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import torch
from torch.utils.data import Dataset
import MinkowskiEngine as ME
import open3d as o3d

from lib.helper_ply import read_ply
from lib.aug_tools import rota_coords, scale_coords, trans_coords


def _require_nuscenes_devkit():
    try:
        from nuscenes.utils import splits as _splits  # noqa: F401
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "nuScenes用には nuscenes-devkit が必要です。"
            "tcuss_vf 環境で `pip install nuscenes-devkit==1.1.11` を実行してください。"
        ) from e


def get_unique_scene_indices(scene_idx_t1: List[int], scene_idx_t2: List[int]) -> List[int]:
    """t1とt2の重複を除いたユニークなシーンインデックスを取得"""
    return list(set(scene_idx_t1 + scene_idx_t2))


@lru_cache(maxsize=8)
def _build_file_index(data_path: str, split: str, frame_stride: int) -> Tuple[List[str], List[str], List[int], List[int]]:
    """data_path配下のPLYを走査し、グローバルインデックス用のファイルリストとscene境界情報を作る。

    Returns:
        files: グローバル順のPLYファイルパス一覧（len = TOTAL）
        scenes: scene名一覧（sorted）
        scene_counts: scenes[i] のフレーム数
        scene_offsets: scenes[i] の開始グローバルindex（prefix sum）
    """
    _require_nuscenes_devkit()
    from nuscenes.utils import splits as nuscenes_splits

    if split not in ("train", "val"):
        raise ValueError(f"Invalid split: {split} (expected 'train' or 'val')")

    if frame_stride <= 0:
        raise ValueError(f"frame_stride must be > 0: {frame_stride}")

    allowed: Set[str]
    if split == "train":
        allowed = set(nuscenes_splits.train)
    else:
        allowed = set(nuscenes_splits.val)

    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"data_path が存在しません: {data_path}")

    # scene directories: e.g., scene-0001
    scenes = sorted(
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d)) and d in allowed
    )
    if len(scenes) == 0:
        raise FileNotFoundError(
            f"data_path 配下に split='{split}' のsceneが見つかりません: data_path={data_path}"
        )

    files: List[str] = []
    scene_counts: List[int] = []
    scene_offsets: List[int] = []

    offset = 0
    for scene in scenes:
        scene_dir = os.path.join(data_path, scene)
        scene_files = sorted(
            os.path.join(scene_dir, f)
            for f in os.listdir(scene_dir)
            if f.endswith(".ply")
        )
        if frame_stride != 1:
            filtered: List[str] = []
            for p in scene_files:
                base = os.path.basename(p)
                # expected: "000123_k.ply" or "000123_s.ply"
                idx_str = base.split("_", 1)[0]
                if len(idx_str) != 6 or not idx_str.isdigit():
                    raise ValueError(f"Unexpected nuScenes frame filename: {base} (path={p})")
                frame_idx = int(idx_str)
                if frame_idx % frame_stride == 0:
                    filtered.append(p)
            scene_files = filtered
        if len(scene_files) == 0:
            continue
        scene_offsets.append(offset)
        scene_counts.append(len(scene_files))
        files.extend(scene_files)
        offset += len(scene_files)

    if len(files) == 0:
        raise FileNotFoundError(f"PLYが見つかりません: data_path={data_path}, split={split}")

    # scenes/offsets/counts length should match, but scenes may include empty dirs which are skipped.
    # Build scenes_filtered aligned to offsets.
    scenes_filtered: List[str] = []
    for scene in scenes:
        scene_dir = os.path.join(data_path, scene)
        if any(f.endswith(".ply") for f in os.listdir(scene_dir)):
            scenes_filtered.append(scene)

    if len(scenes_filtered) != len(scene_counts):
        raise ValueError(
            f"Internal index mismatch: scenes_filtered={len(scenes_filtered)} != scene_counts={len(scene_counts)}"
        )

    return files, scenes_filtered, scene_counts, scene_offsets


def _global_to_scene_local(global_idx: int, scene_offsets: List[int], scene_counts: List[int]) -> Tuple[int, int]:
    """グローバルindex -> (scene_idx, local_idx)"""
    if global_idx < 0:
        raise ValueError(f"Invalid global index: {global_idx}")
    # scene_offsets is increasing, length = num_scenes
    # find rightmost offset <= global_idx
    lo, hi = 0, len(scene_offsets) - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if scene_offsets[mid] <= global_idx:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        raise ValueError(f"Invalid global index: {global_idx}")
    local = global_idx - scene_offsets[best]
    if local < 0 or local >= scene_counts[best]:
        raise ValueError(f"Invalid local index computed: global={global_idx}, scene_idx={best}, local={local}")
    return best, local


def _scene_local_to_global(scene_idx: int, local_idx: int, scene_offsets: List[int], scene_counts: List[int]) -> int:
    if scene_idx < 0 or scene_idx >= len(scene_offsets):
        raise ValueError(f"Invalid scene_idx: {scene_idx}")
    if local_idx < 0 or local_idx >= scene_counts[scene_idx]:
        raise ValueError(f"Invalid local_idx: {local_idx} for scene_idx={scene_idx}")
    return scene_offsets[scene_idx] + local_idx


def generate_scene_pairs(
    config,
    select_num: int,
    scan_window: int,
    seed: int,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """nuScenes用: シーンペア（t1, t2）を生成する（10Hz想定）。

    Args:
        config: TCUSSConfig（data_path参照に必要）
        select_num: cluster_loaderに使用するおおよそのシーン数
        scan_window: t1からt2を選ぶ際の最大フレーム差（10Hzステップ）
        seed: 乱数シード
    """
    files, scenes, scene_counts, scene_offsets = _build_file_index(
        config.data_path, split="train", frame_stride=int(config.frame_stride)
    )
    total_scans = len(files)
    if total_scans <= 0:
        raise ValueError(f"Unexpected total scans: {total_scans}")

    if select_num <= 0:
        raise ValueError(f"select_num must be > 0: {select_num}")
    if scan_window < 0:
        raise ValueError(f"scan_window must be >= 0: {scan_window}")

    rng = np.random.RandomState(seed)
    num_pairs = select_num // 2
    if num_pairs <= 0:
        raise ValueError(f"select_num is too small: {select_num}")

    if num_pairs > total_scans:
        raise ValueError(f"select_num={select_num} is too large for total_scans={total_scans}")

    t1_global_indices = rng.choice(total_scans, num_pairs, replace=False).tolist()

    scene_pairs: List[Tuple[int, int]] = []
    scene_idx_t1: List[int] = []
    scene_idx_t2: List[int] = []

    for t1_global in t1_global_indices:
        scene_idx, local_idx = _global_to_scene_local(t1_global, scene_offsets, scene_counts)
        scene_len = scene_counts[scene_idx]

        t2_min = max(0, local_idx - scan_window)
        t2_max = min(scene_len - 1, local_idx + scan_window)
        t2_candidates = [i for i in range(t2_min, t2_max + 1) if i != local_idx]
        if len(t2_candidates) == 0:
            t2_local = local_idx
        else:
            t2_local = int(rng.choice(t2_candidates))
        t2_global = _scene_local_to_global(scene_idx, t2_local, scene_offsets, scene_counts)

        scene_pairs.append((t1_global, t2_global))
        scene_idx_t1.append(t1_global)
        scene_idx_t2.append(t2_global)

    return scene_pairs, scene_idx_t1, scene_idx_t2


class cfl_collate_fn:
    """KITTI版と同等のcollate（coordsとfeatsのサイズ差、indsオフセットに対応）"""

    def __call__(self, list_data):
        coords, feats, normals, labels, inverse_map, pseudo, inds, region, index, unique_vals = list(zip(*list_data))
        coords_batch, feats_batch, normal_batch, labels_batch, inverse_batch, pseudo_batch, inds_batch = [], [], [], [], [], [], []
        region_batch = []
        feats_sizes = []
        unique_vals_list = []
        accm_coords = 0
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
            inds_batch.append(torch.from_numpy(inds[batch_id] + accm_coords).int())
            region_batch.append(torch.from_numpy(region[batch_id])[:, None])
            accm_coords += num_coords

        coords_batch = torch.cat(coords_batch, 0).float()
        feats_batch = torch.cat(feats_batch, 0).float()
        normal_batch = torch.cat(normal_batch, 0).float()
        labels_batch = torch.cat(labels_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()
        pseudo_batch = torch.cat(pseudo_batch, -1)
        inds_batch = torch.cat(inds_batch, 0)
        region_batch = torch.cat(region_batch, 0)

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
        )


class NuScenesTrain(Dataset):
    """nuScenes GrowSP用データセット（train/cluster共用）

    前処理済みPLY（data_prepare_nuScenes.py）と initSP（*_superpoint.npy）と疑似ラベルを使用する。
    """

    def __init__(self, args, scene_idx: List[int], split: str = "train"):
        _require_nuscenes_devkit()
        self.args = args
        self.split = split
        self.mode = "train"

        files, _, _, _ = _build_file_index(self.args.data_path, split="train", frame_stride=int(self.args.frame_stride))
        self.file: List[str] = files

        # data augmentation
        self.trans_coords = trans_coords(shift_ratio=50)
        self.rota_coords = rota_coords(rotation_bound=((-np.pi / 32, np.pi / 32), (-np.pi / 32, np.pi / 32), (-np.pi, np.pi)))
        self.scale_coords = scale_coords(scale_bound=(0.9, 1.1))

        self.random_select_sample(scene_idx)

    def __len__(self):
        return len(self.file_selected)

    def random_select_sample(self, scene_idx: List[int]):
        self.name: List[str] = []
        self.file_selected: List[str] = []
        for i in scene_idx:
            self.file_selected.append(self.file[i])
            self.name.append(self.file[i][0:-4].replace(self.args.data_path, ""))

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
        coords, feats, labels, unique_map, inverse_map = ME.utils.sparse_quantize(
            np.ascontiguousarray(coords),
            feats,
            labels=labels,
            ignore_label=-1,
            return_index=True,
            return_inverse=True,
        )
        return coords, feats, labels, unique_map, inverse_map

    def __getitem__(self, index):
        file = self.file_selected[index]
        data = read_ply(file)
        coords = np.array([data["x"], data["y"], data["z"]], dtype=np.float32).T
        feats = np.array(data["remission"], dtype=np.float32)[:, np.newaxis]
        labels = np.array(data["class"], dtype=np.int32)

        coords = coords.astype(np.float32)
        coords -= coords.mean(0)

        coords, feats, labels, unique_map, inverse_map = self.voxelize(coords, feats, labels)
        coords = coords.astype(np.float32)

        mask = np.sqrt(((coords * self.args.voxel_size) ** 2).sum(-1)) < self.args.r_crop
        coords, feats, labels = coords[mask], feats[mask], labels[mask]

        region_file = self.args.sp_path + "/" + self.name[index] + "_superpoint.npy"
        region = np.load(region_file)
        region = region[unique_map]
        region = region[mask]

        coords = self.augs(coords)

        inds = np.arange(coords.shape[0])

        # Mixup (same as KITTItrain)
        mix = random.randint(0, len(self.name) - 1)
        data_mix = read_ply(self.file_selected[mix])
        coords_mix = np.array([data_mix["x"], data_mix["y"], data_mix["z"]], dtype=np.float32).T
        feats_mix = np.array(data_mix["remission"], dtype=np.float32)[:, np.newaxis]
        labels_mix = np.array(data_mix["class"], dtype=np.int32)
        coords_mix = coords_mix.astype(np.float32)
        coords_mix -= coords_mix.mean(0)

        coords_mix, feats_mix, _, unique_map_mix, _ = self.voxelize(coords_mix, feats_mix, labels_mix)
        coords_mix = coords_mix.astype(np.float32)
        mask_mix = np.sqrt(((coords_mix * self.args.voxel_size) ** 2).sum(-1)) < self.args.r_crop
        coords_mix, _ = coords_mix[mask_mix], feats_mix[mask_mix]
        coords_mix = self.augs(coords_mix)
        coords = np.concatenate((coords, coords_mix), axis=0)

        coords, feats, labels = self.augment_coords_to_feats(coords, feats, labels)

        # label: 0..16 -> -1..15
        labels = labels.astype(np.int32) - 1

        if self.mode == "cluster":
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coords[inds])
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
            normals = np.array(pcd.normals)

            # nuScenesでは sweep のGTが存在しないため、labels==-1 で region を落とさない
            for q in np.unique(region):
                mask_q = q == region
                if mask_q.sum() < self.args.drop_threshold and q != -1:
                    region[mask_q] = -1

            valid_region = region[region != -1]
            unique_vals = np.unique(valid_region)
            unique_vals.sort()
            valid_region = np.searchsorted(unique_vals, valid_region)
            region[region != -1] = valid_region

            pseudo = -np.ones_like(labels).astype(np.int64)
        else:
            normals = np.zeros_like(coords)
            scene_name = self.name[index]
            file_path = os.path.join(self.args.pseudo_label_path, scene_name.lstrip("/") + ".npy")
            pseudo = np.array(np.load(file_path), dtype=np.int64)

            unique_vals = np.array([], dtype=np.int64)

            expected_size = len(inds)
            if len(pseudo) != expected_size:
                raise ValueError(
                    f"pseudo ファイルのサイズが一致しません: scene={scene_name}, pseudo={len(pseudo)}, expected={expected_size}, ply_file={file}"
                )

        return coords, feats, normals, labels, inverse_map, pseudo, inds, region, index, unique_vals


class NuScenesVal(Dataset):
    """nuScenes評価用データセット（val split + keyframeのみ）"""

    def __init__(self, args):
        _require_nuscenes_devkit()
        self.args = args
        self.mode = "val"

        # valは全keyframeを評価対象にするため、frame_strideは常に1
        files, _, _, _ = _build_file_index(self.args.data_path, split="val", frame_stride=1)
        # keyframeのみ（ファイル名が *_k.ply のもの）
        self.file: List[str] = [p for p in files if p.endswith("_k.ply")]
        if len(self.file) == 0:
            raise FileNotFoundError(f"val split の keyframe PLY が見つかりません: data_path={self.args.data_path}")

        # name is relative path without extension
        self.name: List[str] = [p[0:-4].replace(self.args.data_path, "") for p in self.file]

        # 評価数制限
        if args.eval_select_num < len(self.file):
            random_indices = random.sample(range(len(self.file)), args.eval_select_num)
            self.file = [self.file[i] for i in random_indices]
            self.name = [self.name[i] for i in random_indices]

    def voxelize(self, coords, feats, labels):
        scale = 1 / self.args.voxel_size
        coords = np.floor(coords * scale)
        coords, feats, labels, unique_map, inverse_map = ME.utils.sparse_quantize(
            np.ascontiguousarray(coords),
            feats,
            labels=labels,
            ignore_label=-1,
            return_index=True,
            return_inverse=True,
        )
        return coords, feats, labels, unique_map, inverse_map

    def augment_coords_to_feats(self, coords, feats, labels=None):
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = (coords - coords_center)
        return norm_coords, feats, labels

    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        file = self.file[index]
        data = read_ply(file)
        coords = np.array([data["x"], data["y"], data["z"]], dtype=np.float32).T
        feats = np.array(data["remission"], dtype=np.float32)[:, np.newaxis]
        labels = np.array(data["class"], dtype=np.int32)  # (N,) original points

        original_coords = coords.copy()
        original_labels = labels.copy()
        coords = coords.astype(np.float32)
        coords -= coords.mean(0)

        # voxelize: we keep ORIGINAL labels for evaluation (same design as KITTIval)
        coords, feats, _labels_vox, unique_map, inverse_map = self.voxelize(coords, feats, labels)
        coords = coords.astype(np.float32)

        region_file = self.args.sp_path + "/" + self.name[index] + "_superpoint.npy"
        region = np.load(region_file)
        region = region[unique_map]

        coords, feats, _ = self.augment_coords_to_feats(coords, feats, labels=None)

        # label: 0..16 -> -1..15
        labels_eval = original_labels.astype(np.int32) - 1
        labels_eval[labels_eval < 0] = self.args.ignore_label

        return coords, feats, np.ascontiguousarray(labels_eval), inverse_map, region, index, original_coords


class cfl_collate_fn_val:
    def __call__(self, list_data):
        coords, feats, labels, inverse_map, region, index, original_coords = list(zip(*list_data))
        coords_batch, feats_batch, inverse_batch, labels_batch = [], [], [], []
        region_batch = []
        original_coords_batch = []
        accm_voxel_num = 0

        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            inverse_batch.append(torch.from_numpy(inverse_map[batch_id]) + accm_voxel_num)
            accm_voxel_num += num_points
            labels_batch.append(torch.from_numpy(labels[batch_id]).int())
            region_batch.append(torch.from_numpy(region[batch_id])[:, None])
            original_coords_batch.append(torch.from_numpy(original_coords[batch_id]).float())

        coords_batch = torch.cat(coords_batch, 0).float()
        feats_batch = torch.cat(feats_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()
        labels_batch = torch.cat(labels_batch, 0).int()
        region_batch = torch.cat(region_batch, 0)
        original_coords_batch = torch.cat(original_coords_batch, 0)

        return coords_batch, feats_batch, inverse_batch, labels_batch, index, region_batch, original_coords_batch


class NuScenestcuss(Dataset):
    """GrowSP用: t1とt2を返す複合データセット（STCなし）"""

    def __init__(self, args):
        self.args = args
        self.phase = 0
        self.train_t1 = NuScenesTrain(args, scene_idx=[], split="train")
        self.train_t2 = NuScenesTrain(args, scene_idx=[], split="train")
        self.scene_idx_t1: List[int] = []
        self.scene_idx_t2: List[int] = []

    def set_scene_pairs(self, scene_idx_t1: List[int], scene_idx_t2: List[int]):
        self.scene_idx_t1 = scene_idx_t1
        self.scene_idx_t2 = scene_idx_t2
        self.train_t1.random_select_sample(scene_idx_t1)
        self.train_t2.random_select_sample(scene_idx_t2)

    def __len__(self):
        return len(self.scene_idx_t1)

    def __getitem__(self, index):
        growsp_t1 = self.train_t1.__getitem__(index)
        growsp_t2 = self.train_t2.__getitem__(index)
        return growsp_t1, growsp_t2, None


class cfl_collate_fn_tcuss:
    """TCUSS用collate（GrowSPのみ：stc=None）"""

    def __init__(self):
        self.growsp_collate = cfl_collate_fn()

    def __call__(self, list_data):
        growsp_t1_data, growsp_t2_data, _ = list(zip(*list_data))
        growsp_t1 = self.growsp_collate(growsp_t1_data)
        growsp_t2 = self.growsp_collate(growsp_t2_data)
        return growsp_t1, growsp_t2, None


