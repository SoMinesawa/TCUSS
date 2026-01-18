import os
import re
import csv
import random
from dataclasses import asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import MinkowskiEngine as ME
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from lib.config import TCUSSConfig
from lib.helper_ply import read_ply
from lib.utils import get_fixclassifier, get_kmeans_labels
from models.fpn import Res16FPN18


# scikit-learnバージョンに応じてlinear_assignmentを選択（フォールバックしない: import失敗は例外で停止）
try:
    from sklearn.utils.linear_assignment_ import linear_assignment  # type: ignore

    def assignment_function(cost_matrix: np.ndarray) -> np.ndarray:
        # sklearn版は直接 (N, 2) で返る
        return linear_assignment(cost_matrix)

except ModuleNotFoundError:
    from scipy.optimize import linear_sum_assignment

    def assignment_function(cost_matrix: np.ndarray) -> np.ndarray:
        # scipy版は (row_idx, col_idx) のタプル
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        # 同じ形状 (N, 2) に変形して返す
        return np.stack([row_idx, col_idx], axis=1)


VAL_SEQ_ID = "08"
CLASS_NAMES_19 = [
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]


def set_seed(seed: int) -> None:
    """乱数シードを設定（評価の再現性用）"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def _extract_epoch_from_filename(filename: str) -> Optional[int]:
    m = re.search(r"model_(\d+)_checkpoint\.pth$", filename)
    if m is None:
        return None
    return int(m.group(1))


def list_available_epochs(save_path: str) -> List[int]:
    if not os.path.isdir(save_path):
        raise FileNotFoundError(f"save_path が存在しません: {save_path}")

    files = os.listdir(save_path)
    epochs: List[int] = []
    for f in files:
        e = _extract_epoch_from_filename(f)
        if e is None:
            continue
        model_ckpt = os.path.join(save_path, f"model_{e}_checkpoint.pth")
        cls_ckpt = os.path.join(save_path, f"cls_{e}_checkpoint.pth")
        if not os.path.exists(model_ckpt):
            raise FileNotFoundError(f"モデルチェックポイントが見つかりません: {model_ckpt}")
        if not os.path.exists(cls_ckpt):
            raise FileNotFoundError(f"分類器チェックポイントが見つかりません: {cls_ckpt}")
        epochs.append(e)

    epochs = sorted(set(epochs))
    if not epochs:
        raise FileNotFoundError(
            f"model_<epoch>_checkpoint.pth が見つかりません: save_path={save_path}"
        )
    return epochs


class KITTISweepValDataset(Dataset):
    """val(seq=08) 用: 余計なI/O（SP/RAW label）を避け、評価に必要な最小情報だけ返す。"""

    def __init__(self, config: TCUSSConfig):
        self.config = config

        seq_dir = os.path.join(self.config.data_path, VAL_SEQ_ID)
        if not os.path.isdir(seq_dir):
            raise FileNotFoundError(f"valシーケンスディレクトリが存在しません: {seq_dir}")

        files_all = sorted(
            [os.path.join(seq_dir, f) for f in os.listdir(seq_dir) if f.endswith(".ply")]
        )
        if not files_all:
            raise FileNotFoundError(f"valのPLYが見つかりません: {seq_dir}")

        # eval_select_num は「評価に使用するフレーム数」。小さい場合は固定seedでサンプリング。
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
        coords_vox, feats_vox, _, _, inverse_map = ME.utils.sparse_quantize(
            np.ascontiguousarray(coords_vox),
            feats_vox := feats,  # featsは現状使っていないが、sparse_quantizeのAPIに合わせる
            labels=labels,
            ignore_label=-1,
            return_index=True,
            return_inverse=True,
        )

        # train/testと同じラベル前処理: 0..19 -> -1..18
        labels = labels.astype(np.int32, copy=False)
        labels -= 1
        labels[labels == self.config.ignore_label - 1] = self.config.ignore_label

        return coords_vox, feats_vox, inverse_map.astype(np.int64), labels.astype(np.int32), file_path


class CollateSweepVal:
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
) -> np.ndarray:
    sem_num = config.semantic_class
    histogram = np.zeros((sem_num, sem_num), dtype=np.int64)

    model.eval()
    classifier.eval()

    for coords, _feats, inverse_map, labels, _paths in tqdm(val_loader, desc="val inference", leave=False):
        # MinkowskiEngine TensorField（特徴はxyz、入力次元=3）
        in_field = ME.TensorField(coords[:, 1:] * config.voxel_size, coords, device=device)
        feats = model(in_field)
        feats = F.normalize(feats, dim=1)

        scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
        preds_vox = torch.argmax(scores, dim=1).cpu()

        # voxel -> original points
        preds = preds_vox[inverse_map.long()]

        labels_np = labels.cpu().numpy().astype(np.int64, copy=False)
        preds_np = preds.numpy().astype(np.int64, copy=False)

        mask = (labels_np >= 0) & (labels_np < sem_num)
        if mask.any():
            flat = sem_num * labels_np[mask] + preds_np[mask]
            histogram += np.bincount(flat, minlength=sem_num ** 2).reshape(sem_num, sem_num)

    return histogram


def metrics_from_histogram(histogram: np.ndarray) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """Hungarian後の oAcc, mAcc, mIoU と、matching, IoUs(%) を返す"""
    sem_num = histogram.shape[0]
    if histogram.sum() == 0:
        raise RuntimeError("有効なGT点が0のため、評価できません（histogram.sum()==0）")

    matching = assignment_function(histogram.max() - histogram)

    o_acc = histogram[matching[:, 0], matching[:, 1]].sum() / histogram.sum() * 100.0

    # classごとの正解率（GT側で正規化）
    per_class_total = histogram.sum(1)
    per_class_total[per_class_total == 0] = 1
    m_acc = np.mean(histogram[matching[:, 0], matching[:, 1]] / per_class_total) * 100.0

    # matchingを適用してpred列を並べ替え
    hist_new = np.zeros_like(histogram, dtype=np.float64)
    for idx in range(sem_num):
        hist_new[:, idx] = histogram[:, matching[idx, 1]]

    tp = np.diag(hist_new)
    fp = np.sum(hist_new, 0) - tp
    fn = np.sum(hist_new, 1) - tp
    ious = (100.0 * tp) / (tp + fp + fn + 1e-8)
    m_iou = float(np.nanmean(ious))

    return float(o_acc), float(m_acc), m_iou, matching, ious.astype(np.float64)


def build_fixed_classifier_from_checkpoint(
    config: TCUSSConfig,
    cls_ckpt_path: str,
    device: int,
) -> torch.nn.Module:
    """cls_{epoch}_checkpoint.pth から primitive を semantic_class にマージして固定classifierを作る"""
    if not os.path.exists(cls_ckpt_path):
        raise FileNotFoundError(f"分類器チェックポイントが見つかりません: {cls_ckpt_path}")

    cls = torch.nn.Linear(config.feats_dim, config.primitive_num, bias=False).to(f"cuda:{device}")
    cls.load_state_dict(torch.load(cls_ckpt_path, map_location=f"cuda:{device}"))
    cls.eval()

    primitive_centers = cls.weight.data  # [primitive_num, feats_dim]
    cluster_pred = (
        get_kmeans_labels(n_clusters=config.semantic_class, pcds=primitive_centers)
        .to("cpu")
        .detach()
        .numpy()
        .astype(np.int64, copy=False)
    )

    centroids = torch.zeros((config.semantic_class, config.feats_dim), device=f"cuda:{device}")
    for cluster_idx in range(config.semantic_class):
        idx = cluster_pred == cluster_idx
        if not idx.any():
            raise RuntimeError(
                f"semantic_class={config.semantic_class} のうち、空クラスタが発生しました: cluster_idx={cluster_idx}"
            )
        cluster_avg = primitive_centers[idx].mean(0, keepdims=True)
        centroids[cluster_idx] = cluster_avg

    centroids = F.normalize(centroids, dim=1)
    classifier = get_fixclassifier(
        in_channel=config.feats_dim,
        centroids_num=config.semantic_class,
        centroids=centroids,
    ).to(f"cuda:{device}")
    classifier.eval()
    return classifier


def main() -> None:
    config = TCUSSConfig.from_parse_args()
    if config.dataset != "semantickitti":
        raise ValueError(f"このスクリプトは semantickitti 専用です: dataset={config.dataset}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDAが利用できません（MinkowskiEngine推論にGPUが必要です）")

    device = 0
    set_seed(config.seed)

    epochs = list_available_epochs(config.save_path)
    print(f"Found {len(epochs)} epochs under save_path: {config.save_path}")
    print(f"Val sequence: {VAL_SEQ_ID}, eval_select_num={config.eval_select_num}, eval_batch_size={config.eval_batch_size}")

    # val loader（毎epochで使い回す）
    val_dataset = KITTISweepValDataset(config)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.eval_workers,
        pin_memory=True,
        persistent_workers=getattr(config, "persistent_workers", False) if config.eval_workers > 0 else False,
        prefetch_factor=getattr(config, "prefetch_factor", 4) if config.eval_workers > 0 else None,
        collate_fn=CollateSweepVal(),
    )

    # モデルは1回だけ作って重みだけ差し替える（初期化コスト削減）
    model = Res16FPN18(
        in_channels=config.input_dim,
        out_channels=config.feats_dim,
        conv1_kernel_size=config.conv1_kernel_size,
        config=config,
    ).to(f"cuda:{device}")
    model.eval()

    out_csv = os.path.join(config.save_path, "val_epoch_sweep_1.csv")
    out_best = os.path.join(config.save_path, "best_epoch_val_1.txt")

    results: List[Dict[str, float]] = []
    best_epoch = None
    best_miou = -1.0

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "oAcc",
                "mAcc",
                "mIoU",
            ]
            + [f"IoU_{n}" for n in CLASS_NAMES_19],
        )
        writer.writeheader()

        for epoch in tqdm(epochs, desc="epoch sweep"):
            model_ckpt = os.path.join(config.save_path, f"model_{epoch}_checkpoint.pth")
            cls_ckpt = os.path.join(config.save_path, f"cls_{epoch}_checkpoint.pth")
            if not os.path.exists(model_ckpt):
                raise FileNotFoundError(f"モデルチェックポイントが見つかりません: {model_ckpt}")
            if not os.path.exists(cls_ckpt):
                raise FileNotFoundError(f"分類器チェックポイントが見つかりません: {cls_ckpt}")

            # load model weights
            model.load_state_dict(torch.load(model_ckpt, map_location=f"cuda:{device}"))
            model.eval()

            # build classifier from cls checkpoint
            classifier = build_fixed_classifier_from_checkpoint(config, cls_ckpt, device=device)

            # streaming histogram
            histogram = compute_histogram_on_val(
                config=config,
                model=model,
                classifier=classifier,
                val_loader=val_loader,
                device=device,
            )
            o_acc, m_acc, m_iou, _matching, ious = metrics_from_histogram(histogram)

            row: Dict[str, float] = {
                "epoch": float(epoch),
                "oAcc": o_acc,
                "mAcc": m_acc,
                "mIoU": m_iou,
            }
            for name, iou in zip(CLASS_NAMES_19, ious.tolist()):
                row[f"IoU_{name}"] = float(iou)
            writer.writerow(row)
            f.flush()

            results.append(row)
            if m_iou > best_miou:
                best_miou = m_iou
                best_epoch = epoch

            # 明示的に解放（epoch sweep時のVRAM断片化回避）
            del classifier
            torch.cuda.empty_cache()

    if best_epoch is None:
        raise RuntimeError("best_epoch が決定できませんでした（結果が空）")

    with open(out_best, "w") as f:
        f.write(f"{best_epoch}\n")

    print("========== val epoch sweep done ==========")
    print(f"CSV: {out_csv}")
    print(f"Best epoch (by mIoU): {best_epoch} (mIoU={best_miou:.4f})")
    print(f"Best epoch saved to: {out_best}")


if __name__ == "__main__":
    main()



