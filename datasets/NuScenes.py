import os
import random
from collections import OrderedDict
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Optional, Set, cast

import numpy as np
import torch
from torch.utils.data import Dataset
import MinkowskiEngine as ME
import open3d as o3d
import h5py

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
        data = cast(Any, read_ply(file))
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
        data_mix = cast(Any, read_ply(self.file_selected[mix]))
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
        if labels is None:
            raise RuntimeError("NuScenesTrain.augment_coords_to_feats returned labels=None (unexpected)")

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
        data = cast(Any, read_ply(file))
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


class NuScenesstc(Dataset):
    """STC (Superpoint Time Consistency) 学習用データセット（nuScenes版）

    - VoteFlow preprocess の H5（`stc.voteflow_preprocess_path/scene-xxxx.h5`）から
      `lidar/pose/ground_mask/flow_est` を読み込む
    - GrowSP の init superpoint（`sp_path`）と、クラスタリング時に生成された
      `sp_id_path/*_sp_mapping.npy` を用いて「統合SPラベル」を構築する
    - `scene_flow.correspondence.compute_superpoint_correspondence_direct` により
      SP対応行列（スコア行列）を計算して返す

    重要:
    - 現状のnuScenes preprocessの `flow_est` は「10Hz相当（1scan飛ばし）」として生成されている前提。
      そのため stc では `scan_window=1`（隣接ペア）を前提とし、それ以外はエラーとする。
    """

    def __init__(self, args):
        self.args = args

        # STCは隣接ペアのみ（flow_estが10Hz相当のため）
        if int(args.scan_window) != 1:
            raise ValueError(
                "nuScenesのSTCでは scan_window=1 のみサポートしています。"
                f"現在の設定: scan_window={args.scan_window}"
            )

        # nuScenesのflow_estは「1scan飛ばし(=2フレーム)」で作られている前提なので frame_stride=2 を要求
        if int(args.frame_stride) != 2:
            raise ValueError(
                "nuScenesのSTCでは frame_stride=2（20Hz->10Hz）を前提に `flow_est` を使用します。"
                f"現在の設定: frame_stride={args.frame_stride}"
            )

        # 10Hz（frame_stride後）のファイルインデックスを構築（global index -> file path）
        files, scenes, scene_counts, scene_offsets = _build_file_index(
            self.args.data_path, split="train", frame_stride=int(self.args.frame_stride)
        )
        self._files = files
        self._scenes = scenes
        self._scene_counts = scene_counts
        self._scene_offsets = scene_offsets

        self.scene_idx_t1: List[int] = []
        self.scene_idx_t2: List[int] = []

        # augmentation（GrowSP trainと同様）
        self.trans_coords = trans_coords(shift_ratio=50)
        self.rota_coords = rota_coords(rotation_bound=((-np.pi / 32, np.pi / 32), (-np.pi / 32, np.pi / 32), (-np.pi, np.pi)))
        self.scale_coords = scale_coords(scale_bound=(0.9, 1.1))

        # H5ファイルハンドルのキャッシュ（scene_name -> handle）
        # nuScenesはscene数が多いので、開きっぱなしにするとFD枯渇し得る。LRUで上限を設ける。
        self._h5_handles: "OrderedDict[str, h5py.File]" = OrderedDict()
        self._max_open_h5 = 16

        # tokens.txt のキャッシュ（scene_name -> tokens list）
        self._tokens_cache: Dict[str, List[str]] = {}

        # SPマッチング設定
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

        from scene_flow.correspondence import compute_superpoint_correspondence_direct
        self._compute_sp_correspondence_direct = compute_superpoint_correspondence_direct

    def augs(self, coords: np.ndarray) -> np.ndarray:
        coords = self.rota_coords(coords)
        coords = self.trans_coords(coords)
        coords = self.scale_coords(coords)
        return coords

    def set_scene_pairs(self, scene_idx_t1: List[int], scene_idx_t2: List[int]):
        self.scene_idx_t1 = scene_idx_t1
        self.scene_idx_t2 = scene_idx_t2

    def __len__(self):
        return len(self.scene_idx_t1)

    def _get_h5_handle(self, scene_name: str) -> h5py.File:
        if scene_name in self._h5_handles:
            # LRU更新
            self._h5_handles.move_to_end(scene_name)
            return self._h5_handles[scene_name]

        else:
            h5_path = os.path.join(self.args.stc.voteflow_preprocess_path, f"{scene_name}.h5")
            if not os.path.exists(h5_path):
                raise FileNotFoundError(f"VoteFlow preprocess H5 が見つかりません: {h5_path}")
            self._h5_handles[scene_name] = h5py.File(h5_path, "r")
            self._h5_handles.move_to_end(scene_name)

            # LRU eviction
            while len(self._h5_handles) > int(self._max_open_h5):
                old_scene, old_handle = self._h5_handles.popitem(last=False)
                try:
                    old_handle.close()
                except Exception:
                    # 終了処理中などでh5py内部が破棄されている場合があるため、ここでは握りつぶす
                    pass

            return self._h5_handles[scene_name]

    def _close_h5_handles(self):
        for h in list(self._h5_handles.values()):
            try:
                h.close()
            except Exception:
                # 終了処理中などでh5py内部が破棄されている場合があるため、ここでは握りつぶす
                pass
        self._h5_handles.clear()

    def __del__(self):
        try:
            self._close_h5_handles()
        except Exception:
            # __del__ では例外を出すと警告になるため無視
            pass

    def _load_tokens(self, scene_name: str) -> List[str]:
        if scene_name in self._tokens_cache:
            return self._tokens_cache[scene_name]
        token_txt = os.path.join(self.args.data_path, scene_name, "tokens.txt")
        if not os.path.exists(token_txt):
            raise FileNotFoundError(
                f"tokens.txt が見つかりません: {token_txt}\n"
                "data_prepare/data_prepare_nuScenes.py を実行して tokens.txt を生成してください。"
            )
        with open(token_txt, "r") as f:
            tokens = [line.strip() for line in f if line.strip()]
        if len(tokens) == 0:
            raise ValueError(f"tokens.txt が空です: {token_txt}")
        self._tokens_cache[scene_name] = tokens
        return tokens

    @staticmethod
    def _parse_scene_and_frame_from_ply(ply_path: str) -> Tuple[str, int]:
        """PLYパスから (scene_name, frame_idx_original) を抽出する。

        例: .../growsp/scene-0001/000002_s.ply -> ('scene-0001', 2)
        """
        scene_name = os.path.basename(os.path.dirname(ply_path))
        base = os.path.basename(ply_path)
        idx_str = base.split("_", 1)[0]
        if len(idx_str) != 6 or not idx_str.isdigit():
            raise ValueError(f"Unexpected nuScenes frame filename: {base} (path={ply_path})")
        return scene_name, int(idx_str)

    def _voxelize(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """点群をvoxel化（KITTIstc互換の戻り値処理）"""
        coords = np.ascontiguousarray(coords)

        res = ME.utils.sparse_quantize(
            coords,
            return_index=True,
            return_inverse=True,
            quantization_size=float(self.args.voxel_size),
            return_maps_only=True,
        )

        # MinkowskiEngineのバージョン差吸収（戻り値が2 or 4）
        if len(res) == 2:
            unique_map, inverse_map = res
        else:
            # (quantized_coords, voxel_idx, inverse_map, voxel_counts)
            _, voxel_idx, inverse_map, _ = res
            original_idx = np.arange(len(coords))
            unique_map = original_idx[voxel_idx]

        quantized_coords = coords[unique_map]
        quantized_coords = (quantized_coords / float(self.args.voxel_size))
        return quantized_coords, unique_map, inverse_map

    def _get_item_one_frame(self, ply_path: str, require_flow: bool = True):
        """1フレーム分のSTC用データをロードしてvoxelize/cropし、統合SPラベルを作る。

        Args:
            ply_path: GrowSP PLYのパス
            require_flow: True の場合、H5グループに `flow_est` が必須（STCのsource側）。
                          False の場合、`flow_est` が無くても許容（target側。flowは使用しない）。
        """
        scene_name, frame_idx_original = self._parse_scene_and_frame_from_ply(ply_path)
        tokens = self._load_tokens(scene_name)
        if frame_idx_original < 0 or frame_idx_original >= len(tokens):
            raise ValueError(
                f"frame_idx out of range for tokens.txt: scene={scene_name}, idx={frame_idx_original}, tokens={len(tokens)}"
            )
        token = tokens[frame_idx_original]

        h5_file = self._get_h5_handle(scene_name)
        if token not in h5_file:
            raise KeyError(f"token がH5に存在しません: scene={scene_name}, token={token}, h5={h5_file.filename}")
        g = cast(h5py.Group, h5_file[token])

        # raw data (order must match GrowSP PLY and initSP)
        coords = np.asarray(cast(h5py.Dataset, g["lidar"])[:], dtype=np.float32)  # (N, 3)
        pose = np.asarray(cast(h5py.Dataset, g["pose"])[:], dtype=np.float32)  # (4, 4)
        ground_mask = np.asarray(cast(h5py.Dataset, g["ground_mask"])[:], dtype=bool)  # (N,)
        remission_full = None
        remission_vox = None
        # remission（強度）は必要な場合のみPLYから取得（GrowSP PLYの順序に揃える）
        if float(getattr(self, "weight_remission_similarity", 0.0)) != 0.0:
            ply = read_ply(ply_path)
            if "remission" not in ply.dtype.names:
                raise KeyError(f"PLYに'remission'が存在しません: {ply_path} (fields={ply.dtype.names})")
            remission_full = np.asarray(ply["remission"], dtype=np.float32).reshape(-1)
            if remission_full.shape[0] != coords.shape[0]:
                raise ValueError(
                    "PLY remission と H5 lidar の点数が一致しません: "
                    f"ply={remission_full.shape[0]}, h5={coords.shape[0]}, ply_path={ply_path}, h5={h5_file.filename}"
                )

        flow = None
        if require_flow:
            if "flow_est" not in g:
                hint = ""
                # flow_est が (t -> t+frame_stride) で作られている場合、末尾フレームでは存在しないことがある
                try:
                    gap = int(self.args.frame_stride)
                except Exception:
                    gap = 0
                if gap > 0 and frame_idx_original >= (len(tokens) - gap):
                    hint = (
                        "\n補足: frame_idx がシーン末尾に近いため、flow_est が無いのは仕様の可能性があります。"
                        f" frame_idx={frame_idx_original}, tokens={len(tokens)}, frame_stride={gap}"
                    )
                raise KeyError(
                    f"flow_est がH5に存在しません: scene={scene_name}, token={token}, h5={h5_file.filename}{hint}"
                )
            flow = np.asarray(cast(h5py.Dataset, g["flow_est"])[:], dtype=np.float32)  # (N, 3)
            if flow.shape != coords.shape:
                raise ValueError(
                    "flow_est の形状が lidar と一致しません: "
                    f"scene={scene_name}, token={token}, flow={flow.shape}, lidar={coords.shape}, h5={h5_file.filename}"
                )

        coords_original = coords.copy()
        means = coords.mean(0)
        coords_centered = coords - means

        # voxelize
        coords_vox, unique_map, inverse_map = self._voxelize(coords_centered)
        coords_vox = coords_vox.astype(np.float32)

        # r_crop
        mask = np.sqrt(((coords_vox * self.args.voxel_size) ** 2).sum(-1)) < float(self.args.r_crop)
        coords_vox = coords_vox[mask]

        # init SP
        rel_name = ply_path[0:-4].replace(self.args.data_path, "")  # '/scene-0001/000002_s'
        init_sp_file = os.path.join(self.args.sp_path, rel_name.lstrip("/") + "_superpoint.npy")
        if not os.path.exists(init_sp_file):
            raise FileNotFoundError(f"init superpoint が見つかりません: {init_sp_file}")
        init_sp_labels = np.load(init_sp_file)  # (N,)
        if init_sp_labels.shape[0] != coords_original.shape[0]:
            raise ValueError(
                "init superpoint の点数がH5(lidar)と一致しません: "
                f"scene={scene_name}, token={token}, init_sp={init_sp_labels.shape[0]}, lidar={coords_original.shape[0]}, file={init_sp_file}"
            )

        # sp_mapping (init SP -> merged SP)
        sp_mapping_file = os.path.join(self.args.sp_id_path, rel_name.lstrip("/") + "_sp_mapping.npy")
        if not os.path.exists(sp_mapping_file):
            raise FileNotFoundError(
                f"sp_mappingファイルが見つかりません: {sp_mapping_file}\n"
                "STCを使用するには、まずクラスタリングを実行してsp_mappingを生成してください。"
            )
        sp_mapping = np.load(sp_mapping_file)  # (max_init_sp_id+1,)

        sp_labels_full = np.full_like(init_sp_labels, -1, dtype=np.int32)
        valid_init_mask = (init_sp_labels >= 0) & (init_sp_labels < len(sp_mapping))
        mapped = sp_mapping[init_sp_labels[valid_init_mask]]
        sp_labels_full[valid_init_mask] = np.where(mapped >= 0, mapped, -1)

        sp_labels = sp_labels_full[unique_map][mask]
        ground_mask_vox = ground_mask[unique_map][mask]
        flow_vox = None
        if require_flow:
            # require_flow=True の場合のみ使用される
            assert flow is not None
            flow_vox = flow[unique_map][mask]
        coords_original_masked = coords_original[unique_map][mask]
        if remission_full is not None:
            remission_vox = remission_full[unique_map][mask]

        return {
            "scene_name": rel_name,  # '/scene-0001/000002_s'
            "frame_idx_original": frame_idx_original,
            "token": token,
            "coords_vox": coords_vox,
            "coords_original": coords_original_masked,
            "sp_labels": sp_labels,
            "pose": pose,
            "flow_vox": flow_vox,
            "ground_mask_vox": ground_mask_vox,
            "remission_vox": remission_vox,
            "inverse_map": inverse_map,
            "unique_map": unique_map,
            "mask": mask,
        }

    def __getitem__(self, index):
        if len(self.scene_idx_t1) == 0 or len(self.scene_idx_t2) == 0:
            raise RuntimeError("NuScenesstc.set_scene_pairs() が呼ばれていません")
        g1 = int(self.scene_idx_t1[index])
        g2 = int(self.scene_idx_t2[index])
        if g1 < 0 or g1 >= len(self._files) or g2 < 0 or g2 >= len(self._files):
            raise ValueError(f"global index out of range: g1={g1}, g2={g2}, total={len(self._files)}")

        ply1 = self._files[g1]
        ply2 = self._files[g2]

        scene1, idx1 = self._parse_scene_and_frame_from_ply(ply1)
        scene2, idx2 = self._parse_scene_and_frame_from_ply(ply2)
        if scene1 != scene2:
            raise ValueError(f"STC pair must be in the same scene: {scene1} vs {scene2} (g1={g1}, g2={g2})")

        # STCは時間順に揃える（flowは早い時刻のものを使う）
        if idx1 <= idx2:
            src_ply, tgt_ply = ply1, ply2
            src_idx, tgt_idx = idx1, idx2
        else:
            src_ply, tgt_ply = ply2, ply1
            src_idx, tgt_idx = idx2, idx1

        expected_gap = int(self.args.frame_stride)
        actual_gap = int(tgt_idx - src_idx)
        if actual_gap != expected_gap:
            raise ValueError(
                "nuScenes STCでは隣接(10Hz)ペアのみサポートしています。"
                f"expected_gap(frame_stride)={expected_gap}, actual_gap={actual_gap}. "
                f"src={os.path.basename(src_ply)}, tgt={os.path.basename(tgt_ply)}"
            )

        # source側のみ flow_est が必須。target側は flow を使用しないため欠損を許容する。
        src = self._get_item_one_frame(src_ply, require_flow=True)
        tgt = self._get_item_one_frame(tgt_ply, require_flow=False)

        # SP対応行列を計算（SPレベル直接マッチング）
        corr_matrix, unique_sp_t, unique_sp_t2 = self._compute_sp_correspondence_direct(
            src["coords_original"],
            tgt["coords_original"],
            src["flow_vox"],
            src["sp_labels"],
            tgt["sp_labels"],
            src["pose"],
            tgt["pose"],
            src["ground_mask_vox"],
            tgt["ground_mask_vox"],
            remission_t=src["remission_vox"],
            remission_t1=tgt["remission_vox"],
            weight_centroid_distance=self.weight_centroid_distance,
            weight_spread_similarity=self.weight_spread_similarity,
            weight_point_count_similarity=self.weight_point_count_similarity,
            weight_remission_similarity=self.weight_remission_similarity,
            max_centroid_distance=self.max_centroid_distance,
            min_score_threshold=self.min_score_threshold,
            min_sp_points=self.min_sp_points,
            remove_ego_motion=self.remove_ego_motion,
            exclude_ground=self.exclude_ground,
        )

        # augmentation（tとt2で別々に適用）
        coords_t_aug = self.augs(src["coords_vox"].copy())
        coords_t2_aug = self.augs(tgt["coords_vox"].copy())

        return {
            "coords_t": coords_t_aug,
            "coords_t2": coords_t2_aug,
            "sp_labels_t": src["sp_labels"],
            "sp_labels_t2": tgt["sp_labels"],
            "corr_matrix": corr_matrix,
            "unique_sp_t": unique_sp_t,
            "unique_sp_t2": unique_sp_t2,
            "scene_name_t": src["scene_name"],
            "scene_name_t2": tgt["scene_name"],
        }


class cfl_collate_fn_stc:
    """STC用collate関数（nuScenes版）

    NuScenesstc が返すdictをバッチ化する。
    coords_t/coords_t2はMinkowskiEngine用にバッチIDを付与。
    SP対応行列などはリストのまま返す。
    """

    def __call__(self, list_data):
        coords_t_batch = []
        coords_t2_batch = []
        sp_labels_t_list = []
        sp_labels_t2_list = []
        corr_matrix_list = []
        unique_sp_t_list = []
        unique_sp_t2_list = []
        scene_name_t_list = []
        scene_name_t2_list = []

        for batch_id, data in enumerate(list_data):
            coords_t = data["coords_t"]
            coords_t2 = data["coords_t2"]
            num_points_t = coords_t.shape[0]
            num_points_t2 = coords_t2.shape[0]

            coords_t_batch.append(
                torch.cat(
                    (torch.ones(num_points_t, 1).int() * batch_id, torch.from_numpy(coords_t).int()),
                    1,
                )
            )
            coords_t2_batch.append(
                torch.cat(
                    (torch.ones(num_points_t2, 1).int() * batch_id, torch.from_numpy(coords_t2).int()),
                    1,
                )
            )

            sp_labels_t_list.append(torch.from_numpy(data["sp_labels_t"]).long())
            sp_labels_t2_list.append(torch.from_numpy(data["sp_labels_t2"]).long())
            corr_matrix_list.append(torch.from_numpy(data["corr_matrix"]).float())
            unique_sp_t_list.append(data["unique_sp_t"])
            unique_sp_t2_list.append(data["unique_sp_t2"])
            scene_name_t_list.append(data["scene_name_t"])
            scene_name_t2_list.append(data["scene_name_t2"])

        coords_t_batch = torch.cat(coords_t_batch, 0).float()
        coords_t2_batch = torch.cat(coords_t2_batch, 0).float()

        return {
            "coords_t": coords_t_batch,
            "coords_t2": coords_t2_batch,
            "sp_labels_t": sp_labels_t_list,
            "sp_labels_t2": sp_labels_t2_list,
            "corr_matrix": corr_matrix_list,
            "unique_sp_t": unique_sp_t_list,
            "unique_sp_t2": unique_sp_t2_list,
            "scene_name_t": scene_name_t_list,
            "scene_name_t2": scene_name_t2_list,
        }


class NuScenestcuss(Dataset):
    """TCUSS用の複合データセット（nuScenes）

    stc.enabled=True: GrowSP + STC
    stc.enabled=False: GrowSPのみ
    """

    def __init__(self, args):
        self.args = args
        self.phase = 0
        self.stc_enabled = bool(args.stc.enabled)
        self.train_t1 = NuScenesTrain(args, scene_idx=[], split="train")
        self.train_t2 = NuScenesTrain(args, scene_idx=[], split="train")
        self.nuscstc = NuScenesstc(args) if self.stc_enabled else None
        self.scene_idx_t1: List[int] = []
        self.scene_idx_t2: List[int] = []

    def set_scene_pairs(self, scene_idx_t1: List[int], scene_idx_t2: List[int]):
        self.scene_idx_t1 = scene_idx_t1
        self.scene_idx_t2 = scene_idx_t2
        self.train_t1.random_select_sample(scene_idx_t1)
        self.train_t2.random_select_sample(scene_idx_t2)
        if self.stc_enabled and self.nuscstc is not None:
            self.nuscstc.set_scene_pairs(scene_idx_t1, scene_idx_t2)

    def __len__(self):
        return len(self.scene_idx_t1)

    def __getitem__(self, index):
        growsp_t1 = self.train_t1.__getitem__(index)
        growsp_t2 = self.train_t2.__getitem__(index)
        if self.stc_enabled and self.nuscstc is not None:
            stc_data = self.nuscstc.__getitem__(index)
        else:
            stc_data = None
        return growsp_t1, growsp_t2, stc_data


class cfl_collate_fn_tcuss:
    """TCUSS用collate（nuScenes）

    stc_dataがNoneの場合（STC無効時）はstc=Noneを返す
    """

    def __init__(self):
        self.growsp_collate = cfl_collate_fn()
        self.stc_collate = cfl_collate_fn_stc()

    def __call__(self, list_data):
        growsp_t1_data, growsp_t2_data, stc_data = list(zip(*list_data))
        growsp_t1 = self.growsp_collate(growsp_t1_data)
        growsp_t2 = self.growsp_collate(growsp_t2_data)
        if stc_data[0] is None:
            stc = None
        else:
            stc = self.stc_collate(stc_data)
        return growsp_t1, growsp_t2, stc


