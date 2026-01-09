import argparse
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

try:
    from sklearn.utils.linear_assignment_ import linear_assignment  # type: ignore

    def assignment_function(cost_matrix):
        return linear_assignment(cost_matrix)

except ModuleNotFoundError:
    from scipy.optimize import linear_sum_assignment

    def assignment_function(cost_matrix):
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        return np.stack([row_idx, col_idx], axis=1)


from datasets.NuScenes import NuScenesVal, cfl_collate_fn_val


def eval_once(
    args: Union[argparse.Namespace, Any],
    model: torch.nn.Module,
    val_loader: DataLoader,
    classifier: torch.nn.Module,
    epoch: int,
    device: int = 0,
    is_main_process: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """1回の評価（nuScenes: keyframeのみ）"""
    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_distances: List[torch.Tensor] = []

    iterator = val_loader
    if is_main_process:
        iterator = tqdm(val_loader, desc=f"Eval Epoch: {epoch}")

    for data in iterator:
        with torch.no_grad():
            coords, features, inverse_map, labels, index, region, original_coords = data

            in_field = ME.TensorField(coords[:, 1:] * args.voxel_size, coords, device=device)
            feats = model(in_field)
            feats = F.normalize(feats, dim=1)

            scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
            preds = torch.argmax(scores, dim=1).cpu()
            preds = preds[inverse_map.long()]

            # distance in xy-plane (same as KITTI eval style)
            distances = torch.sqrt(original_coords[:, 0] ** 2 + original_coords[:, 1] ** 2)

            valid_mask = labels != args.ignore_label
            preds = preds[valid_mask]
            labels = labels[valid_mask]
            distances = distances[valid_mask]

            all_preds.append(preds)
            all_labels.append(labels)
            all_distances.append(distances)

    return all_preds, all_labels, all_distances


def compute_metrics_from_histogram(histogram: np.ndarray, sem_num: int, matching: np.ndarray) -> Tuple[float, float, float]:
    total = histogram.sum()
    if total == 0:
        return 0.0, 0.0, 0.0

    hist_new = np.zeros((sem_num, sem_num))
    for idx in range(sem_num):
        hist_new[:, idx] = histogram[:, matching[idx, 1]]

    o_Acc = histogram[matching[:, 0], matching[:, 1]].sum() / total * 100.0

    class_totals = histogram.sum(1)
    class_totals[class_totals == 0] = 1
    m_Acc = np.mean(histogram[matching[:, 0], matching[:, 1]] / class_totals) * 100.0

    tp = np.diag(hist_new)
    fp = np.sum(hist_new, 0) - tp
    fn = np.sum(hist_new, 1) - tp
    IoUs = (100 * tp) / (tp + fp + fn + 1e-8)
    m_IoU = float(np.nanmean(IoUs))
    return float(o_Acc), float(m_Acc), float(m_IoU)


def compute_distance_histograms(
    preds: np.ndarray,
    labels: np.ndarray,
    distances: np.ndarray,
    distance_bins: List[int],
    sem_num: int,
) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    valid_mask = (labels >= 0) & (labels < sem_num)

    prev_dist = 0
    for dist in distance_bins:
        mask = valid_mask & (distances >= prev_dist) & (distances < dist)
        key = f"{prev_dist}-{dist}"
        if mask.sum() > 0:
            histogram = np.bincount(
                sem_num * labels[mask] + preds[mask],
                minlength=sem_num**2,
            ).reshape(sem_num, sem_num)
        else:
            histogram = np.zeros((sem_num, sem_num), dtype=np.int64)
        result[key] = histogram
        prev_dist = dist

    mask = valid_mask & (distances >= distance_bins[-1])
    key = f"{distance_bins[-1]}+"
    if mask.sum() > 0:
        histogram = np.bincount(
            sem_num * labels[mask] + preds[mask],
            minlength=sem_num**2,
        ).reshape(sem_num, sem_num)
    else:
        histogram = np.zeros((sem_num, sem_num), dtype=np.int64)
    result[key] = histogram
    return result


def compute_distance_metrics_from_histograms(
    distance_histograms: Dict[str, np.ndarray],
    sem_num: int,
    matching: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key, hist in distance_histograms.items():
        count = int(hist.sum())
        if count > 0:
            oAcc, mAcc, mIoU = compute_metrics_from_histogram(hist, sem_num, matching)
        else:
            oAcc, mAcc, mIoU = 0.0, 0.0, 0.0
        out[key] = {"oAcc": oAcc, "mAcc": mAcc, "mIoU": mIoU, "count": count}
    return out


def eval_ddp(
    epoch: int,
    args: Union[argparse.Namespace, Any],
    model: torch.nn.Module,
    classifier: torch.nn.Module,
    local_rank: int = 0,
    world_size: int = 1,
    is_main_process: bool = True,
) -> Tuple[float, float, float, str, Dict[str, float], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """DDP対応の評価（nuScenes）"""
    use_ddp = world_size > 1
    device = f"cuda:{local_rank}"

    model.eval()
    classifier.eval()

    # フォールバック禁止（YAMLに必須）
    eval_batch_size = args.eval_batch_size
    persistent_workers = args.persistent_workers
    prefetch_factor = args.prefetch_factor
    num_workers = args.eval_workers

    val_dataset = NuScenesVal(args)
    val_sampler = None
    if use_ddp:
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)

    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        collate_fn=cfl_collate_fn_val(),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        sampler=val_sampler,
        shuffle=False,
    )

    preds, labels, distances = eval_once(
        args, model, val_loader, classifier, epoch, device=local_rank, is_main_process=is_main_process
    )

    if len(preds) > 0:
        local_preds = torch.cat(preds)
        local_labels = torch.cat(labels)
        local_distances = torch.cat(distances)
    else:
        local_preds = torch.tensor([], dtype=torch.long)
        local_labels = torch.tensor([], dtype=torch.long)
        local_distances = torch.tensor([], dtype=torch.float)

    sem_num = int(args.semantic_class)
    local_preds_np = local_preds.numpy()
    local_labels_np = local_labels.numpy()
    local_distances_np = local_distances.numpy()

    mask = (local_labels_np >= 0) & (local_labels_np < sem_num)
    if mask.sum() > 0:
        local_histogram = np.bincount(
            sem_num * local_labels_np[mask] + local_preds_np[mask],
            minlength=sem_num**2,
        ).reshape(sem_num, sem_num)
    else:
        local_histogram = np.zeros((sem_num, sem_num), dtype=np.int64)

    local_distance_histograms = None
    if hasattr(args, "evaluation") and args.evaluation.distance_evaluation:
        local_distance_histograms = compute_distance_histograms(
            local_preds_np, local_labels_np, local_distances_np, args.evaluation.distance_bins, sem_num
        )

    if use_ddp:
        histogram_tensor = torch.from_numpy(local_histogram).to(device)
        dist.all_reduce(histogram_tensor, op=dist.ReduceOp.SUM)
        histogram = histogram_tensor.cpu().numpy()

        if local_distance_histograms is not None:
            distance_histograms: Optional[Dict[str, np.ndarray]] = {}
            for key, local_hist in local_distance_histograms.items():
                hist_tensor = torch.from_numpy(local_hist.astype(np.int64)).to(device)
                dist.all_reduce(hist_tensor, op=dist.ReduceOp.SUM)
                distance_histograms[key] = hist_tensor.cpu().numpy()
        else:
            distance_histograms = None
    else:
        histogram = local_histogram
        distance_histograms = local_distance_histograms

    matching = assignment_function(histogram.max() - histogram)
    o_Acc, m_Acc, m_IoU = compute_metrics_from_histogram(histogram, sem_num, matching)

    # IoU string / dict (simple)
    hist_new = np.zeros((sem_num, sem_num))
    for idx in range(sem_num):
        hist_new[:, idx] = histogram[:, matching[idx, 1]]
    tp = np.diag(hist_new)
    fp = np.sum(hist_new, 0) - tp
    fn = np.sum(hist_new, 1) - tp
    IoUs = (100 * tp) / (tp + fp + fn + 1e-8)

    s = "| mIoU {:5.2f} | ".format(m_IoU)
    IoU_list: List[float] = []
    for IoU in IoUs:
        s += "{:5.2f} ".format(IoU)
        IoU_list.append(float(IoU))
    IoU_dict = {f"IoU_{i:02d}": v for i, v in enumerate(IoU_list)}

    distance_metrics: Dict[str, Dict[str, float]] = {}
    if distance_histograms is not None:
        distance_metrics = compute_distance_metrics_from_histograms(distance_histograms, sem_num, matching)

    moving_static_metrics: Dict[str, Dict[str, float]] = {}  # nuScenesでは未使用
    return o_Acc, m_Acc, m_IoU, s, IoU_dict, distance_metrics, moving_static_metrics


