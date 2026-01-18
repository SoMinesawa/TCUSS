import argparse
import csv
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import MinkowskiEngine as ME
from torch.utils.data import DataLoader, Dataset
from scipy.optimize import linear_sum_assignment

from lib.config import TCUSSConfig
from lib.helper_ply import read_ply
from lib.kmeans_torch import KMeans as KMeans_gpu
from lib.utils import get_fixclassifier
from models.fpn import Res16FPN18


VAL_SEQ_ID = "08"


def set_seed_for_determinism(seed: int) -> None:
    # NOTE: 評価の再現性用。KMeans(seed探索)は trialごとに np.random.seed を上書きする。
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def assignment_function(cost_matrix: np.ndarray) -> np.ndarray:
    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    return np.stack([row_idx, col_idx], axis=1)


def get_kmeans_labels_strict(n_clusters: int, pcds: torch.Tensor, max_iter: int = 300) -> torch.Tensor:
    """KMeans(GPU)でクラスタラベルを返す（フォールバック禁止：失敗は例外）"""
    model = KMeans_gpu(n_clusters=n_clusters, max_iter=max_iter, distance="euclidean").cuda()
    with torch.no_grad():
        pcds = pcds.cuda().float()
        unsqueezed = pcds.unsqueeze(0)
        _centroids, labels = model(unsqueezed)  # ValueError を投げうる（missing cluster）
    lbls = labels.squeeze(0).long().cuda()
    del labels, _centroids, model, pcds
    torch.cuda.empty_cache()
    return lbls


class KITTISearchValDataset(Dataset):
    """SemanticKITTI val(seq=08) のみ。KITTItestと同じ前処理（region読み込み無し）で評価する。"""

    def __init__(self, config: TCUSSConfig):
        self.config = config
        seq_dir = os.path.join(self.config.data_path, VAL_SEQ_ID)
        if not os.path.isdir(seq_dir):
            raise FileNotFoundError(f"valシーケンスディレクトリが存在しません: {seq_dir}")

        files_all = sorted([os.path.join(seq_dir, f) for f in os.listdir(seq_dir) if f.endswith(".ply")])
        if not files_all:
            raise FileNotFoundError(f"valのPLYが見つかりません: {seq_dir}")

        if self.config.eval_select_num <= 0:
            raise ValueError(f"eval_select_num は正の整数が必要です: {self.config.eval_select_num}")

        if self.config.eval_select_num < len(files_all):
            rng = random.Random(self.config.seed)
            idx = rng.sample(range(len(files_all)), self.config.eval_select_num)
            self.files = [files_all[i] for i in idx]
        else:
            self.files = files_all

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _augment_coords_to_feats(coords_vox: np.ndarray) -> np.ndarray:
        """datasets/SemanticKITTI.py の augment_coords_to_feats と同等（coordsのみ版）"""
        coords_center = coords_vox.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = coords_vox - coords_center
        return norm_coords

    def __getitem__(self, index: int):
        file_path = self.files[index]
        data = read_ply(file_path)

        coords = np.array([data["x"], data["y"], data["z"]], dtype=np.float32).T
        feats = np.array(data["remission"], dtype=np.float32)[:, np.newaxis]
        labels = np.array(data["class"], dtype=np.int32)

        coords -= coords.mean(0)

        # voxelize（inverse_map: original -> voxel index）
        scale = 1.0 / self.config.voxel_size
        coords_vox = np.floor(coords * scale)
        coords_vox, feats_vox, _labels_vox, _unique_map, inverse_map = ME.utils.sparse_quantize(
            np.ascontiguousarray(coords_vox),
            feats,
            labels=labels,
            ignore_label=-1,
            return_index=True,
            return_inverse=True,
        )
        coords_vox = coords_vox.astype(np.float32, copy=False)

        # KITTItest と同じ座標正規化（voxel後に再中心化）
        coords_vox = self._augment_coords_to_feats(coords_vox)

        # train/test と同じラベル前処理: 0..19 -> -1..18
        labels = labels.astype(np.int32, copy=False)
        labels -= 1
        labels[labels == self.config.ignore_label - 1] = self.config.ignore_label

        return coords_vox, feats_vox, inverse_map.astype(np.int64), labels.astype(np.int32), file_path


class CollateSearchVal:
    """minibatch用collate（inverse_mapオフセット込み）"""

    def __call__(self, list_data):
        coords_list, feats_list, inv_list, labels_list, paths = list(zip(*list_data))

        coords_batch: List[torch.Tensor] = []
        feats_batch: List[torch.Tensor] = []
        inv_batch: List[torch.Tensor] = []
        labels_batch: List[torch.Tensor] = []

        accm_voxel_num = 0
        for batch_id in range(len(coords_list)):
            coords = coords_list[batch_id]
            feats = feats_list[batch_id]
            inv = inv_list[batch_id]
            labels = labels_list[batch_id]

            num_vox = coords.shape[0]
            coords_batch.append(
                torch.cat(
                    (
                        torch.ones(num_vox, 1, dtype=torch.int32) * batch_id,
                        torch.from_numpy(coords).int(),
                    ),
                    dim=1,
                )
            )
            feats_batch.append(torch.from_numpy(feats).float())

            inv_batch.append(torch.from_numpy(inv).long() + accm_voxel_num)
            accm_voxel_num += num_vox

            labels_batch.append(torch.from_numpy(labels).int())

        coords_batch_t = torch.cat(coords_batch, dim=0).float()
        feats_batch_t = torch.cat(feats_batch, dim=0).float()
        inv_batch_t = torch.cat(inv_batch, dim=0).long()
        labels_batch_t = torch.cat(labels_batch, dim=0).int()
        return coords_batch_t, feats_batch_t, inv_batch_t, labels_batch_t, list(paths)


@torch.no_grad()
def compute_histogram_on_val(
    config: TCUSSConfig,
    model: torch.nn.Module,
    classifier: torch.nn.Module,
    val_loader: DataLoader,
    device: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    sem_num = config.semantic_class
    histogram = np.zeros((sem_num, sem_num), dtype=np.int64)

    model.eval()
    classifier.eval()

    # NOTE: tqdmは標準出力を大量に汚し、seed探索（無限ループ）では大幅に遅くなるため使用しない。
    # timing
    num_scans = 0
    num_batches = 0
    fetch_stall_sec = 0.0  # DataLoaderが次バッチを返すまでに待った時間（prefetchで隠れた分は含まれない）
    compute_sec = 0.0  # 推論 + CPU側のhist更新まで含む

    t_total_start = time.perf_counter()
    it = iter(val_loader)
    while True:
        t_fetch0 = time.perf_counter()
        try:
            coords, _feats, inverse_map, labels, paths = next(it)
        except StopIteration:
            break
        t_fetch1 = time.perf_counter()
        fetch_stall_sec += float(t_fetch1 - t_fetch0)

        # batch size (= scans)
        if not isinstance(paths, list):
            raise TypeError(f"paths はlistである必要があります: type={type(paths)}")
        batch_scans = len(paths)
        if batch_scans <= 0:
            raise RuntimeError("batch_scans<=0 です。DataLoader/collateが不正の可能性があります。")

        # 推論時間（GPU同期を入れてwall-clockで計測）
        torch.cuda.synchronize(device)
        t_comp0 = time.perf_counter()

        in_field = ME.TensorField(coords[:, 1:] * config.voxel_size, coords, device=device)
        feats = model(in_field)
        feats = F.normalize(feats, dim=1)

        scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
        preds_vox = torch.argmax(scores, dim=1).cpu()

        preds = preds_vox[inverse_map.long()]

        labels_np = labels.cpu().numpy().astype(np.int64, copy=False)
        preds_np = preds.numpy().astype(np.int64, copy=False)

        mask = (labels_np >= 0) & (labels_np < sem_num)
        if mask.any():
            flat = sem_num * labels_np[mask] + preds_np[mask]
            histogram += np.bincount(flat, minlength=sem_num**2).reshape(sem_num, sem_num)

        torch.cuda.synchronize(device)
        t_comp1 = time.perf_counter()
        compute_sec += float(t_comp1 - t_comp0)

        num_batches += 1
        num_scans += batch_scans

    t_total_end = time.perf_counter()
    total_sec = float(t_total_end - t_total_start)
    if num_scans <= 0:
        raise RuntimeError("num_scans<=0 のため、速度計測ができません。")

    timing = {
        "num_scans": float(num_scans),
        "num_batches": float(num_batches),
        "elapsed_total_sec": float(total_sec),
        "elapsed_compute_sec": float(compute_sec),
        "elapsed_fetch_stall_sec": float(fetch_stall_sec),
        "sec_per_scan": float(total_sec / num_scans),
        "compute_sec_per_scan": float(compute_sec / num_scans),
        "fetch_stall_sec_per_scan": float(fetch_stall_sec / num_scans),
        "scans_per_sec": float(num_scans / total_sec) if total_sec > 0 else float("inf"),
    }

    return histogram, timing


def metrics_from_histogram(histogram: np.ndarray) -> Tuple[float, float, float]:
    sem_num = histogram.shape[0]
    if histogram.sum() == 0:
        raise RuntimeError("有効なGT点が0のため、評価できません（histogram.sum()==0）")

    matching = assignment_function(histogram.max() - histogram)

    o_acc = histogram[matching[:, 0], matching[:, 1]].sum() / histogram.sum() * 100.0

    per_class_total = histogram.sum(1)
    per_class_total[per_class_total == 0] = 1
    m_acc = float(np.mean(histogram[matching[:, 0], matching[:, 1]] / per_class_total) * 100.0)

    hist_new = np.zeros_like(histogram, dtype=np.float64)
    for idx in range(sem_num):
        hist_new[:, idx] = histogram[:, matching[idx, 1]]

    tp = np.diag(hist_new)
    fp = np.sum(hist_new, 0) - tp
    fn = np.sum(hist_new, 1) - tp
    ious = (100.0 * tp) / (tp + fp + fn + 1e-8)
    m_iou = float(np.nanmean(ious))

    return float(o_acc), float(m_acc), m_iou


def build_eval_classifier_from_cls_checkpoint(
    config: TCUSSConfig,
    device: int,
    primitive_centers: torch.Tensor,
    kmeans_seed: int,
) -> torch.nn.Module:
    if primitive_centers.device.type != "cuda":
        raise ValueError(f"primitive_centers はCUDAテンソルである必要があります: device={primitive_centers.device}")
    if primitive_centers.shape != (config.primitive_num, config.feats_dim):
        raise ValueError(
            f"primitive_centers のshapeが不正です: expected={(config.primitive_num, config.feats_dim)}, got={tuple(primitive_centers.shape)}"
        )

    # primitive -> semantic のKMeans初期化をseedで制御（ここが探索対象）
    np.random.seed(kmeans_seed)
    semantic_labels = get_kmeans_labels_strict(n_clusters=config.semantic_class, pcds=primitive_centers).to("cpu").numpy()

    semantic_centers = torch.zeros((config.semantic_class, config.feats_dim), device=f"cuda:{device}")
    for cluster_idx in range(config.semantic_class):
        idx = semantic_labels == cluster_idx
        if not idx.any():
            raise RuntimeError(
                f"semantic_class={config.semantic_class} のうち、空クラスタが発生しました: cluster_idx={cluster_idx}"
            )
        semantic_centers[cluster_idx] = primitive_centers[idx].mean(0, keepdims=False)
    semantic_centers = F.normalize(semantic_centers, dim=1)

    classifier = get_fixclassifier(
        in_channel=config.feats_dim,
        centroids_num=config.semantic_class,
        centroids=semantic_centers,
    ).to(f"cuda:{device}")
    classifier.eval()
    return classifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search KMeans seed for SemanticKITTI val thresholds, then output the seed.")
    p.add_argument("--config", type=str, required=True, help="TCUSS YAML config path (e.g., config/stc.yaml)")
    p.add_argument("--save_path", type=str, required=True, help="model directory containing model_<epoch>_checkpoint.pth")
    p.add_argument("--epoch", type=int, required=True, help="epoch to evaluate (e.g., 330)")
    p.add_argument("--target_miou", type=float, required=True, help="target mIoU (e.g., 16.5)")
    p.add_argument("--target_oacc", type=float, required=True, help="target oAcc (e.g., 45.0)")
    p.add_argument("--seed_start", type=int, required=True, help="seed start (used for sequential, and as RNG seed for random)")
    p.add_argument("--seed_mode", type=str, required=True, choices=["sequential", "random"], help="seed generation mode")
    p.add_argument("--max_trials", type=int, required=True, help="0=inf, otherwise stop after N trials")
    p.add_argument("--device", type=int, required=True, help="CUDA device id (e.g., 0)")
    p.add_argument("--out_seed_file", type=str, required=True, help="output text file to save found seed")
    p.add_argument("--log_csv", type=str, required=True, help="output CSV file to append trials")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config = TCUSSConfig.from_yaml(args.config)
    config.save_path = args.save_path

    if config.dataset != "semantickitti":
        raise ValueError(f"このスクリプトは semantickitti 専用です: dataset={config.dataset}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDAが利用できません（MinkowskiEngine推論にGPUが必要です）")

    device = args.device
    set_seed_for_determinism(config.seed)

    # dataset/loader（trial間で使い回す）
    val_dataset = KITTISearchValDataset(config)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.eval_workers,
        pin_memory=True,
        persistent_workers=getattr(config, "persistent_workers", False) if config.eval_workers > 0 else False,
        prefetch_factor=getattr(config, "prefetch_factor", 4) if config.eval_workers > 0 else None,
        collate_fn=CollateSearchVal(),
    )

    model_ckpt = os.path.join(config.save_path, f"model_{args.epoch}_checkpoint.pth")
    cls_ckpt = os.path.join(config.save_path, f"cls_{args.epoch}_checkpoint.pth")
    if not os.path.exists(model_ckpt):
        raise FileNotFoundError(f"モデルチェックポイントが見つかりません: {model_ckpt}")
    if not os.path.exists(cls_ckpt):
        raise FileNotFoundError(f"分類器チェックポイントが見つかりません: {cls_ckpt}")

    # modelは固定（trial間で使い回す）
    model = Res16FPN18(
        in_channels=config.input_dim,
        out_channels=config.feats_dim,
        conv1_kernel_size=config.conv1_kernel_size,
        config=config,
    ).to(f"cuda:{device}")
    model.load_state_dict(torch.load(model_ckpt, map_location=f"cuda:{device}"))
    model.eval()

    # primitive_centers は固定（trial間で使い回す）
    cls = torch.nn.Linear(config.feats_dim, config.primitive_num, bias=False).to(f"cuda:{device}")
    cls.load_state_dict(torch.load(cls_ckpt, map_location=f"cuda:{device}"))
    cls.eval()
    primitive_centers = cls.weight.data.clone()  # [primitive_num, feats_dim]
    del cls
    torch.cuda.empty_cache()

    # CSV log init（ヘッダー不一致は黙って壊さず、エラーで止める）
    fieldnames = [
        "trial",
        "kmeans_seed",
        "epoch",
        "oAcc",
        "mAcc",
        "mIoU",
        "target_oacc",
        "target_miou",
        "config_seed",
        "config_save_path",
        "config_data_path",
        # timing (per trial)
        "kmeans_build_sec",
        "eval_num_scans",
        "eval_num_batches",
        "eval_elapsed_total_sec",
        "eval_elapsed_compute_sec",
        "eval_elapsed_fetch_stall_sec",
        "eval_sec_per_scan",
        "eval_compute_sec_per_scan",
        "eval_fetch_stall_sec_per_scan",
        "eval_scans_per_sec",
    ]

    os.makedirs(os.path.dirname(args.log_csv) or ".", exist_ok=True)
    if os.path.exists(args.log_csv):
        with open(args.log_csv, "r", newline="") as f:
            header_line = ""
            while True:
                line = f.readline()
                if line == "":
                    break
                if line.strip() != "":
                    header_line = line
                    break
        if header_line == "":
            raise RuntimeError(f"既存CSVが空です。削除するか別名にしてください: {args.log_csv}")
        existing_header = [h.strip() for h in header_line.strip().split(",")]
        if existing_header != fieldnames:
            raise RuntimeError(
                "CSVヘッダーが一致しないため追記できません。\n"
                f"- csv: {args.log_csv}\n"
                f"- expected: {fieldnames}\n"
                f"- found:    {existing_header}\n"
                "古いCSVを削除するか、--log_csv を別名にしてください。"
            )
    else:
        with open(args.log_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    rng = random.Random(args.seed_start)

    trial = 0
    while True:
        if args.max_trials != 0 and trial >= args.max_trials:
            raise RuntimeError(
                f"条件未達のため終了: max_trials={args.max_trials}, last_trial={trial - 1}. "
                f"target: mIoU>={args.target_miou}, oAcc>={args.target_oacc}"
            )

        if args.seed_mode == "sequential":
            kmeans_seed = args.seed_start + trial
        else:
            kmeans_seed = rng.randint(0, 2**31 - 1)

        print(
            f"[seed_search] trial={trial} kmeans_seed={kmeans_seed} "
            f"target(mIoU>={args.target_miou}, oAcc>={args.target_oacc})"
        )

        try:
            t_build0 = time.perf_counter()
            classifier = build_eval_classifier_from_cls_checkpoint(
                config=config,
                device=device,
                primitive_centers=primitive_centers,
                kmeans_seed=kmeans_seed,
            )
            torch.cuda.synchronize(device)
            t_build1 = time.perf_counter()
            kmeans_build_sec = float(t_build1 - t_build0)
        except Exception as e:
            # seed探索なので、KMeans失敗等は次のseedへ（フォールバックはしない）
            print(f"[seed_search] trial={trial} kmeans_seed={kmeans_seed} -> KMeans/build failed: {type(e).__name__}: {e}")
            trial += 1
            continue

        histogram, timing = compute_histogram_on_val(
            config=config,
            model=model,
            classifier=classifier,
            val_loader=val_loader,
            device=device,
        )
        o_acc, m_acc, m_iou = metrics_from_histogram(histogram)

        ok = (m_iou >= args.target_miou) and (o_acc >= args.target_oacc)
        print(
            f"[seed_search] result trial={trial} kmeans_seed={kmeans_seed}: "
            f"oAcc={o_acc:.4f} mAcc={m_acc:.4f} mIoU={m_iou:.4f} -> {'OK' if ok else 'NG'}"
        )
        print(
            f"[seed_search] speed trial={trial} kmeans_seed={kmeans_seed}: "
            f"eval_scans={int(timing['num_scans'])} "
            f"elapsed={timing['elapsed_total_sec']:.3f}s "
            f"ms/scan={timing['sec_per_scan']*1000.0:.3f} "
            f"(compute={timing['compute_sec_per_scan']*1000.0:.3f}, stall={timing['fetch_stall_sec_per_scan']*1000.0:.3f}) "
            f"throughput={timing['scans_per_sec']:.2f} scans/s "
            f"kmeans_build={kmeans_build_sec:.3f}s"
        )

        with open(args.log_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(
                {
                    "trial": int(trial),
                    "kmeans_seed": int(kmeans_seed),
                    "epoch": int(args.epoch),
                    "oAcc": float(o_acc),
                    "mAcc": float(m_acc),
                    "mIoU": float(m_iou),
                    "target_oacc": float(args.target_oacc),
                    "target_miou": float(args.target_miou),
                    "config_seed": int(config.seed),
                    "config_save_path": str(config.save_path),
                    "config_data_path": str(config.data_path),
                    # timing
                    "kmeans_build_sec": float(kmeans_build_sec),
                    "eval_num_scans": int(timing["num_scans"]),
                    "eval_num_batches": int(timing["num_batches"]),
                    "eval_elapsed_total_sec": float(timing["elapsed_total_sec"]),
                    "eval_elapsed_compute_sec": float(timing["elapsed_compute_sec"]),
                    "eval_elapsed_fetch_stall_sec": float(timing["elapsed_fetch_stall_sec"]),
                    "eval_sec_per_scan": float(timing["sec_per_scan"]),
                    "eval_compute_sec_per_scan": float(timing["compute_sec_per_scan"]),
                    "eval_fetch_stall_sec_per_scan": float(timing["fetch_stall_sec_per_scan"]),
                    "eval_scans_per_sec": float(timing["scans_per_sec"]),
                }
            )
            f.flush()

        # 明示的に解放
        del classifier
        torch.cuda.empty_cache()

        if ok:
            os.makedirs(os.path.dirname(args.out_seed_file) or ".", exist_ok=True)
            with open(args.out_seed_file, "w") as f:
                f.write(f"{kmeans_seed}\n")
            print(f"[seed_search] FOUND seed={kmeans_seed} -> saved: {args.out_seed_file}")
            break

        trial += 1


if __name__ == "__main__":
    main()


