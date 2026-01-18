import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Set, TYPE_CHECKING, cast

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import MinkowskiEngine as ME
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "MinkowskiEngine が見つかりません。環境を確認してください。"
    ) from e

from lib.config import TCUSSConfig
from lib.utils import get_fixclassifier, get_kmeans_labels
from lib.helper_ply import read_ply
from models.fpn import Res16FPN18

if TYPE_CHECKING:
    from nuscenes.nuscenes import NuScenes  # pragma: no cover


def _require(d: Dict[str, Any], keys: List[str], ctx: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"{ctx} に必須キーが不足しています: {missing}")


def _require_bool(d: Dict[str, Any], key: str, ctx: str) -> bool:
    if key not in d:
        raise ValueError(f"{ctx} に必須キー '{key}' がありません")
    v = d[key]
    if not isinstance(v, bool):
        raise TypeError(f"{ctx}.{key} は bool である必要があります: type={type(v)} value={v}")
    return v


def _require_str(d: Dict[str, Any], key: str, ctx: str) -> str:
    if key not in d:
        raise ValueError(f"{ctx} に必須キー '{key}' がありません")
    v = d[key]
    if not isinstance(v, str) or len(v) == 0:
        raise TypeError(f"{ctx}.{key} は空でない str である必要があります: type={type(v)} value={v}")
    return v


def _require_int(d: Dict[str, Any], key: str, ctx: str) -> int:
    if key not in d:
        raise ValueError(f"{ctx} に必須キー '{key}' がありません")
    v = d[key]
    if not isinstance(v, int):
        raise TypeError(f"{ctx}.{key} は int である必要があります: type={type(v)} value={v}")
    return v


@dataclass(frozen=True)
class SubmitConfig:
    train_config: str
    model_dir: str
    checkpoint_type: str  # "best" | "epoch"
    checkpoint_epoch: Optional[int]
    results_dir: str
    zip_path: str
    dataroot: str
    version_val: str
    version_test: str
    lidar_sensor: str
    meta: Dict[str, bool]


def load_submit_config(yaml_path: str) -> SubmitConfig:
    import yaml

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"config が見つかりません: {yaml_path}")

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise TypeError(f"config YAML は dict である必要があります: type={type(cfg)} path={yaml_path}")

    train_config = _require_str(cfg, "train_config", "config")
    model_dir = _require_str(cfg, "model_dir", "config")
    results_dir = _require_str(cfg, "results_dir", "config")
    zip_path = _require_str(cfg, "zip_path", "config")

    _require(cfg, ["checkpoint"], "config")
    ckpt = cfg["checkpoint"]
    if not isinstance(ckpt, dict):
        raise TypeError(f"config.checkpoint は dict である必要があります: type={type(ckpt)}")

    ckpt_type = _require_str(ckpt, "type", "config.checkpoint")
    if ckpt_type not in ("best", "epoch"):
        raise ValueError(f"config.checkpoint.type が不正です: {ckpt_type} (expected 'best' or 'epoch')")
    ckpt_epoch: Optional[int] = None
    if ckpt_type == "epoch":
        ckpt_epoch = _require_int(ckpt, "epoch", "config.checkpoint")
        if ckpt_epoch < 0:
            raise ValueError(f"config.checkpoint.epoch は0以上である必要があります: {ckpt_epoch}")
    else:
        # best のときは epoch を指定しない（指定されていたら明示的にエラー）
        if "epoch" in ckpt:
            raise ValueError("config.checkpoint.type='best' のとき config.checkpoint.epoch は指定しないでください")

    _require(cfg, ["nuscenes"], "config")
    ns = cfg["nuscenes"]
    if not isinstance(ns, dict):
        raise TypeError(f"config.nuscenes は dict である必要があります: type={type(ns)}")
    dataroot = _require_str(ns, "dataroot", "config.nuscenes")
    version_val = _require_str(ns, "version_val", "config.nuscenes")
    version_test = _require_str(ns, "version_test", "config.nuscenes")
    lidar_sensor = _require_str(ns, "lidar_sensor", "config.nuscenes")

    _require(cfg, ["meta"], "config")
    meta = cfg["meta"]
    if not isinstance(meta, dict):
        raise TypeError(f"config.meta は dict である必要があります: type={type(meta)}")
    meta_out = {
        "use_camera": _require_bool(meta, "use_camera", "config.meta"),
        "use_lidar": _require_bool(meta, "use_lidar", "config.meta"),
        "use_radar": _require_bool(meta, "use_radar", "config.meta"),
        "use_map": _require_bool(meta, "use_map", "config.meta"),
        "use_external": _require_bool(meta, "use_external", "config.meta"),
    }

    return SubmitConfig(
        train_config=train_config,
        model_dir=model_dir,
        checkpoint_type=ckpt_type,
        checkpoint_epoch=ckpt_epoch,
        results_dir=results_dir,
        zip_path=zip_path,
        dataroot=dataroot,
        version_val=version_val,
        version_test=version_test,
        lidar_sensor=lidar_sensor,
        meta=meta_out,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # NOTE: pyright の型スタブ差分により torch.backends が未定義扱いになることがある。
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    torch.backends.cudnn.enabled = False  # type: ignore[attr-defined]


def read_nuscenes_lidar_bin(bin_path: str) -> Tuple[np.ndarray, int]:
    """nuScenes LIDAR_TOP .pcd.bin を読み込む（点の順序は維持する）.

    Returns:
        xyz: (N, 3) float32
        n_points: N
    """
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"lidar bin が見つかりません: {bin_path}")
    arr = np.fromfile(bin_path, dtype=np.float32)
    if arr.size % 5 != 0:
        raise ValueError(
            "pcd.bin の要素数が5で割り切れません（想定: x,y,z,intensity,ring）: "
            f"size={arr.size}, path={bin_path}"
        )
    pts = arr.reshape(-1, 5)
    xyz = pts[:, 0:3].astype(np.float32, copy=False)
    return xyz, int(xyz.shape[0])


def _build_semantic_classifier(train_cfg: TCUSSConfig, cls_weight: torch.Tensor) -> torch.nn.Module:
    """primitive classifier の重心(primitive_num)を k=semantic_class でkmeansし、semantic_class分類器を返す。"""
    primitive_centers = cls_weight.data  # (primitive_num, feats_dim)
    k = int(train_cfg.semantic_class)
    if primitive_centers.shape[0] != int(train_cfg.primitive_num):
        raise ValueError(
            "cls.weight の形状が config.primitive_num と一致しません: "
            f"cls={tuple(primitive_centers.shape)}, primitive_num={train_cfg.primitive_num}"
        )
    if primitive_centers.shape[1] != int(train_cfg.feats_dim):
        raise ValueError(
            "cls.weight の形状が config.feats_dim と一致しません: "
            f"cls={tuple(primitive_centers.shape)}, feats_dim={train_cfg.feats_dim}"
        )
    if k <= 0:
        raise ValueError(f"semantic_class は正である必要があります: {k}")

    # kmeans (torch) -> (primitive_num,) in [0..k-1]
    semantic_labels = get_kmeans_labels(k, primitive_centers).to("cpu").detach().numpy()
    if semantic_labels.shape[0] != primitive_centers.shape[0]:
        raise ValueError(
            "kmeansラベル数がprimitive_numと一致しません: "
            f"labels={semantic_labels.shape}, primitive_num={primitive_centers.shape[0]}"
        )

    centroids = torch.zeros((k, int(train_cfg.feats_dim)), device=primitive_centers.device, dtype=primitive_centers.dtype)
    for cls_idx in range(k):
        mask = semantic_labels == cls_idx
        if mask.sum() <= 0:
            raise ValueError(f"kmeansに空クラスタが発生しました: cls_idx={cls_idx} k={k}")
        centroids[cls_idx] = primitive_centers[mask].mean(0)

    centroids = F.normalize(centroids, dim=1)
    classifier = get_fixclassifier(
        in_channel=int(train_cfg.feats_dim),
        centroids_num=k,
        centroids=centroids,
    ).to(primitive_centers.device)
    classifier.eval()
    return classifier


def _infer_points(
    model: torch.nn.Module,
    classifier: torch.nn.Module,
    coords: torch.Tensor,
    inverse_map: torch.Tensor,
    voxel_size: float,
    device_id: int,
) -> torch.Tensor:
    """voxel化済み coords/inverse_map から、元点群順序の予測 (N,) を返す（0..K-1）。"""
    # NOTE: 既存実装（trainer/eval）と同様に、coordsはCPUのままTensorFieldを作る。
    # device_id は local_rank 相当（int）を渡す。
    in_field = ME.TensorField(coords[:, 1:] * float(voxel_size), coords, device=int(device_id))
    feats = cast(torch.Tensor, model(in_field))
    feats = F.normalize(feats, dim=1)

    weight = cast(torch.Tensor, getattr(classifier, "weight"))
    scores = F.linear(F.normalize(feats), F.normalize(weight))
    preds_vox = torch.argmax(scores, dim=1).cpu()
    preds_points = preds_vox[inverse_map.long()]  # inverse_map はCPU想定
    return preds_points


def _iter_val_keyframe_ply(data_path: str, split: str) -> List[str]:
    """data_path配下から nuScenes split(train/val/test) の keyframe PLY一覧を得る。"""
    try:
        from nuscenes.utils import splits as nuscenes_splits
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "nuScenes用には nuscenes-devkit が必要です。"
            "tcuss_vf 環境で `pip install nuscenes-devkit==1.1.11` を実行してください。"
        ) from e

    if split not in ("train", "val", "test"):
        raise ValueError(f"Invalid split: {split} (expected 'train'|'val'|'test')")

    allowed: Set[str]
    if split == "train":
        allowed = set(nuscenes_splits.train)
    elif split == "val":
        allowed = set(nuscenes_splits.val)
    else:
        allowed = set(nuscenes_splits.test)

    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"data_path が存在しません: {data_path}")

    scenes = sorted(d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and d in allowed)
    if len(scenes) == 0:
        raise FileNotFoundError(f"data_path 配下に split='{split}' のsceneが見つかりません: data_path={data_path}")

    files: List[str] = []
    for scene in scenes:
        scene_dir = os.path.join(data_path, scene)
        scene_files = sorted(
            os.path.join(scene_dir, f)
            for f in os.listdir(scene_dir)
            if f.endswith("_k.ply")
        )
        files.extend(scene_files)

    if len(files) == 0:
        raise FileNotFoundError(f"keyframe PLY が見つかりません: data_path={data_path}, split={split}")
    return files


def _voxelize_centered(coords_xyz: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """中心化->voxelize し、(coords_vox, inverse_map) を返す。

    coords_vox: (M, 3) float/int (MEに渡す前にint化する)
    inverse_map: (N,) int (original point -> voxel index)
    """
    if coords_xyz.ndim != 2 or coords_xyz.shape[1] != 3:
        raise ValueError(f"coords_xyz shape must be (N,3): {coords_xyz.shape}")

    coords = coords_xyz.astype(np.float32, copy=False)
    coords = coords - coords.mean(0, keepdims=True)

    scale = 1.0 / float(voxel_size)
    coords_scaled = np.floor(coords * scale)

    # ME.utils.sparse_quantize の戻り値はバージョン差があり得るため、既存コードと同じ呼び方に寄せる
    feats_dummy = np.zeros((coords_scaled.shape[0], 1), dtype=np.float32)
    labels_dummy = np.zeros((coords_scaled.shape[0],), dtype=np.int32)
    res = ME.utils.sparse_quantize(
        np.ascontiguousarray(coords_scaled),
        feats_dummy,
        labels=labels_dummy,
        ignore_label=-1,
        return_index=True,
        return_inverse=True,
    )
    if len(res) != 5:
        raise ValueError(f"Unexpected sparse_quantize return length: {len(res)}")
    coords_vox, _feats_vox, _labels_vox, _unique_map, inverse_map = res
    return coords_vox, inverse_map


def compute_val_matching(
    train_cfg: TCUSSConfig,
    model: torch.nn.Module,
    classifier: torch.nn.Module,
    dataroot: str,
    version_val: str,
    device_id: int,
) -> np.ndarray:
    """val split (keyframe) で Hungarian matching を計算し、pred_id -> gt_id の写像を返す。

    Returns:
        pred_to_gt: (K,) int64, where K = semantic_class, values in [0..K-1]
    """
    # val PLYは train_cfg.data_path にある前提（data_prepare_nuScenes.py の出力）
    val_files = _iter_val_keyframe_ply(train_cfg.data_path, split="val")

    # NOTE: version_val / dataroot は validate用にも使うのでここでも存在確認だけする
    if not os.path.isdir(dataroot):
        raise FileNotFoundError(f"nuScenes dataroot が存在しません: {dataroot}")
    if not os.path.isdir(os.path.join(dataroot, version_val)):
        raise FileNotFoundError(f"nuScenes version dir が存在しません: {os.path.join(dataroot, version_val)}")

    sem_num = int(train_cfg.semantic_class)
    histogram = np.zeros((sem_num, sem_num), dtype=np.int64)

    # DataLoaderは使わず逐次（I/O依存が強く、batch化するとメモリが不安定になりやすい）
    iterator = tqdm(val_files, desc="Compute val matching (keyframes)")
    for ply_path in iterator:
        data = cast(Dict[str, Any], read_ply(ply_path))
        coords_xyz = np.array([data["x"], data["y"], data["z"]], dtype=np.float32).T  # (N,3)
        labels_raw = np.array(data["class"], dtype=np.int32)  # (N,) in 0..16 (0 is void)

        coords_vox, inverse_map = _voxelize_centered(coords_xyz, float(train_cfg.voxel_size))

        # labels: 0..16 -> -1..15
        labels_eval = labels_raw.astype(np.int32) - 1
        labels_eval[labels_eval < 0] = int(train_cfg.ignore_label)

        coords_t = torch.cat(
            [
                torch.zeros((coords_vox.shape[0], 1), dtype=torch.int32),
                torch.from_numpy(coords_vox).int(),
            ],
            dim=1,
        ).float()
        inverse_t = torch.from_numpy(inverse_map).long()

        preds_points = _infer_points(
            model=model,
            classifier=classifier,
            coords=coords_t,
            inverse_map=inverse_t,
            voxel_size=float(train_cfg.voxel_size),
            device_id=device_id,
        ).cpu().numpy()

        labels_np = labels_eval
        valid = (labels_np >= 0) & (labels_np < sem_num)
        if valid.any():
            hist = np.bincount(
                sem_num * labels_np[valid].astype(np.int64) + preds_points[valid].astype(np.int64),
                minlength=sem_num**2,
            ).reshape(sem_num, sem_num)
            histogram += hist

    if histogram.sum() == 0:
        raise ValueError(
            "val split で有効なGTが空です（histogram.sum()==0）。"
            "data_prepare_nuScenes.py の trainval 出力（keyframeのclass）が正しく入っているか確認してください。"
        )

    # Hungarian matching: rows=GT, cols=pred
    try:
        from scipy.optimize import linear_sum_assignment
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("scipy が見つかりません。environment.yml を確認してください。") from e

    row_idx, col_idx = linear_sum_assignment(histogram.max() - histogram)
    m = np.stack([row_idx, col_idx], axis=1).astype(np.int64)
    m = m[np.argsort(m[:, 0])]

    pred_to_gt = np.full((sem_num,), -1, dtype=np.int64)
    for gt_id, pred_id in m:
        pred_to_gt[pred_id] = gt_id

    if (pred_to_gt < 0).any():
        raise ValueError(f"pred_to_gt に未割当が存在します: {pred_to_gt}")

    return pred_to_gt


def collect_test_keyframe_tokens(
    dataroot: str,
    version_test: str,
    lidar_sensor: str,
) -> Tuple['NuScenes', List[str], Dict[str, str]]:
    """test split の keyframe sample_data_token 一覧と、token->pcd.binパスを返す。"""
    try:
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils import splits as nuscenes_splits
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "nuScenes用には nuscenes-devkit が必要です。"
            "tcuss_vf 環境で `pip install nuscenes-devkit==1.1.11` を実行してください。"
        ) from e

    if not os.path.isdir(dataroot):
        raise FileNotFoundError(f"nuScenes dataroot が存在しません: {dataroot}")
    version_dir = os.path.join(dataroot, version_test)
    if not os.path.isdir(version_dir):
        raise FileNotFoundError(f"nuScenes version dir が存在しません: {version_dir}")

    nusc = NuScenes(version=version_test, dataroot=dataroot, verbose=True)

    scene_names = list(nuscenes_splits.test)
    if len(scene_names) == 0:
        raise ValueError("nuscenes.utils.splits.test が空です。nuscenes-devkit のsplit定義を確認してください。")

    tokens: List[str] = []
    token_to_bin: Dict[str, str] = {}

    for scene_name in tqdm(scene_names, desc="Collect test keyframe tokens"):
        scene_tokens = nusc.field2token("scene", "name", scene_name)
        if not isinstance(scene_tokens, list) or len(scene_tokens) != 1:
            raise ValueError(f"scene name -> token が一意に解決できません: scene_name={scene_name}, tokens={scene_tokens}")
        scene = nusc.get("scene", scene_tokens[0])
        sample_token = scene["first_sample_token"]
        if not isinstance(sample_token, str) or len(sample_token) == 0:
            raise ValueError(f"scene.first_sample_token が不正です: scene={scene_name}, token={sample_token}")

        while sample_token:
            sample = nusc.get("sample", sample_token)
            if "data" not in sample or lidar_sensor not in sample["data"]:
                raise KeyError(f"sample.data に {lidar_sensor} がありません: sample_token={sample_token}")
            sd_token = sample["data"][lidar_sensor]

            sd = nusc.get("sample_data", sd_token)
            if "filename" not in sd:
                raise KeyError(f"sample_data.filename がありません: token={sd_token}")
            if "is_key_frame" not in sd:
                raise KeyError(f"sample_data.is_key_frame がありません: token={sd_token}")
            if not bool(sd["is_key_frame"]):
                raise ValueError(
                    "test提出はkeyframe前提です。sampleのLIDAR_TOPがkeyframeではありません: "
                    f"scene={scene_name}, sample_token={sample_token}, sample_data_token={sd_token}"
                )

            bin_path = os.path.join(dataroot, sd["filename"])
            if not os.path.exists(bin_path):
                raise FileNotFoundError(f"lidar bin が存在しません: {bin_path} (token={sd_token})")

            tokens.append(sd_token)
            token_to_bin[sd_token] = bin_path

            sample_token = sample.get("next", "")

    if len(tokens) == 0:
        raise ValueError("test keyframe token が0件です。dataroot/version/splits を確認してください。")

    # 重複チェック
    if len(tokens) != len(set(tokens)):
        raise ValueError("test keyframe token に重複があります（想定外）。")

    return nusc, tokens, token_to_bin


def validate_submission_with_devkit(nusc: 'NuScenes', results_dir: str, eval_set: str, verbose: bool) -> None:
    """nuscenes-devkit の validate_submission を直接呼ぶ（公式チェックと同一）。"""
    try:
        from nuscenes.eval.lidarseg.validate_submission import validate_submission as _validate_submission
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "nuscenes-devkit の validate_submission が import できません。"
            "tcuss_vf 環境で `pip install nuscenes-devkit==1.1.11` を確認してください。"
        ) from e

    # NOTE: devkit実装は assert で検証する。Python -O で実行すると assert が無効になるので注意。
    _validate_submission(
        nusc=nusc,
        results_folder=os.path.abspath(results_dir),
        eval_set=str(eval_set),
        verbose=bool(verbose),
    )


def make_zip(results_dir: str, zip_path: str) -> None:
    import zipfile

    if os.path.exists(zip_path):
        raise FileExistsError(f"zip_path が既に存在します: {zip_path}")

    base_dir = os.path.abspath(results_dir)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"results_dir が存在しません: {base_dir}")

    zip_abs = os.path.abspath(zip_path)
    # zipがresults_dir配下だとzip自身を取り込んで壊れるので禁止
    if os.path.commonpath([base_dir, zip_abs]) == base_dir:
        raise ValueError(
            "zip_path は results_dir の外に置いてください（zip自身を取り込む事故防止）: "
            f"results_dir={base_dir}, zip_path={zip_abs}"
        )
    zip_parent = os.path.dirname(zip_abs)
    if zip_parent and not os.path.isdir(zip_parent):
        os.makedirs(zip_parent, exist_ok=True)

    # zipの直下に test/ と lidarseg/ が来るよう、results_dir配下の相対パスで格納
    with zipfile.ZipFile(zip_abs, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(base_dir):
            for fn in files:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, base_dir)
                zf.write(full, rel)


def main() -> None:
    parser = argparse.ArgumentParser(description="TCUSS - nuScenes LiDARSeg test submission generator")
    parser.add_argument("--config", type=str, required=True, help="YAML config path (see config/nuscenes_lidarseg_test.yaml)")
    args = parser.parse_args()

    submit_cfg = load_submit_config(args.config)
    train_cfg = TCUSSConfig.from_yaml(submit_cfg.train_config)
    if train_cfg.dataset != "nuscenes":
        raise ValueError(f"train_config.dataset must be 'nuscenes': {train_cfg.dataset}")
    if int(train_cfg.semantic_class) != 16:
        raise ValueError(
            "nuScenes LiDARSeg 提出は semantic_class=16 前提です。"
            f"現在の設定: semantic_class={train_cfg.semantic_class}"
        )
    if submit_cfg.lidar_sensor != "LIDAR_TOP":
        raise ValueError(f"lidar_sensor は 'LIDAR_TOP' のみサポートします: {submit_cfg.lidar_sensor}")

    # 重要: fallback禁止。必要パスは全て存在必須。
    for p in [submit_cfg.model_dir, train_cfg.data_path, submit_cfg.dataroot]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"必要パスが存在しません: {p}")

    set_seed(int(train_cfg.seed))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_id = 0
    if device.type != "cuda":
        raise RuntimeError("CUDAが利用できません。フォールバックせず終了します。")

    # checkpoints
    if submit_cfg.checkpoint_type == "best":
        model_path = os.path.join(submit_cfg.model_dir, "best_model.pth")
        cls_path = os.path.join(submit_cfg.model_dir, "best_classifier.pth")
    else:
        assert submit_cfg.checkpoint_epoch is not None
        model_path = os.path.join(submit_cfg.model_dir, f"model_{submit_cfg.checkpoint_epoch}_checkpoint.pth")
        cls_path = os.path.join(submit_cfg.model_dir, f"cls_{submit_cfg.checkpoint_epoch}_checkpoint.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model checkpoint が見つかりません: {model_path}")
    if not os.path.exists(cls_path):
        raise FileNotFoundError(f"classifier checkpoint が見つかりません: {cls_path}")

    # output dirs
    results_dir = os.path.abspath(submit_cfg.results_dir)
    if os.path.exists(results_dir):
        # 既存ディレクトリは事故りやすいのでエラー（手動で消す）
        if os.path.isdir(results_dir) and len(os.listdir(results_dir)) == 0:
            pass
        else:
            raise FileExistsError(f"results_dir が既に存在します（空にしてから再実行してください）: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "test"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "lidarseg", "test"), exist_ok=True)

    # submission.json
    submission_json = os.path.join(results_dir, "test", "submission.json")
    with open(submission_json, "w") as f:
        json.dump({"meta": submit_cfg.meta}, f, indent=2, sort_keys=True)

    # load model
    model = Res16FPN18(
        in_channels=int(train_cfg.input_dim),
        out_channels=int(train_cfg.feats_dim),
        conv1_kernel_size=int(train_cfg.conv1_kernel_size),
        config=train_cfg,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    cls = torch.nn.Linear(int(train_cfg.feats_dim), int(train_cfg.primitive_num), bias=False).to(device)
    cls.load_state_dict(torch.load(cls_path, map_location=device))
    cls.eval()

    semantic_classifier = _build_semantic_classifier(train_cfg, cls.weight)

    # val matching (pred_id -> gt_id, both in 0..K-1)
    pred_to_gt = compute_val_matching(
        train_cfg=train_cfg,
        model=model,
        classifier=semantic_classifier,
        dataroot=os.path.abspath(submit_cfg.dataroot),
        version_val=submit_cfg.version_val,
        device_id=device_id,
    )

    # save matching for debug (submissionフォルダには入れない)
    pred_to_gt_path = os.path.abspath(submit_cfg.zip_path) + ".pred_to_gt.npy"
    if os.path.exists(pred_to_gt_path):
        raise FileExistsError(f"pred_to_gt の保存先が既に存在します: {pred_to_gt_path}")
    np.save(pred_to_gt_path, pred_to_gt)

    # test tokens
    nusc_test, tokens, token_to_bin = collect_test_keyframe_tokens(
        dataroot=os.path.abspath(submit_cfg.dataroot),
        version_test=submit_cfg.version_test,
        lidar_sensor=submit_cfg.lidar_sensor,
    )

    sem_num = int(train_cfg.semantic_class)
    out_pred_dir = os.path.join(results_dir, "lidarseg", "test")

    # inference on test (batched forward is難しいので逐次で堅牢に)
    for token in tqdm(tokens, desc="Infer test (keyframes)"):
        bin_path = token_to_bin[token]
        xyz, n_points = read_nuscenes_lidar_bin(bin_path)
        coords_vox, inverse_map = _voxelize_centered(xyz, float(train_cfg.voxel_size))

        coords_t = torch.cat(
            [
                torch.zeros((coords_vox.shape[0], 1), dtype=torch.int32),
                torch.from_numpy(coords_vox).int(),
            ],
            dim=1,
        ).float()
        inverse_t = torch.from_numpy(inverse_map).long()

        preds_points = _infer_points(
            model=model,
            classifier=semantic_classifier,
            coords=coords_t,
            inverse_map=inverse_t,
            voxel_size=float(train_cfg.voxel_size),
            device_id=device_id,
        ).cpu().numpy()

        if preds_points.shape[0] != n_points:
            raise ValueError(
                "予測点数が入力点数と一致しません: "
                f"token={token}, preds={preds_points.shape[0]}, n_points={n_points}"
            )

        # unsupervised pred(0..K-1) -> GT id(0..K-1) -> submission label(1..K)
        mapped = pred_to_gt[preds_points.astype(np.int64)]
        submit_label = (mapped + 1).astype(np.uint8)
        if submit_label.min() < 1 or submit_label.max() > sem_num:
            raise ValueError(
                "提出ラベル範囲が不正です（1..K）: "
                f"token={token}, min={int(submit_label.min())}, max={int(submit_label.max())}, K={sem_num}"
            )

        out_path = os.path.join(out_pred_dir, f"{token}_lidarseg.bin")
        submit_label.tofile(out_path)

    # validate (official devkit)
    validate_submission_with_devkit(
        nusc=nusc_test,
        results_dir=results_dir,
        eval_set="test",
        verbose=True,
    )

    # zip
    make_zip(results_dir=results_dir, zip_path=submit_cfg.zip_path)

    print("=== nuScenes LiDARSeg submission ready ===")
    print(f"results_dir: {results_dir}")
    print(f"zip_path: {os.path.abspath(submit_cfg.zip_path)}")


if __name__ == "__main__":
    main()


