import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Dict, Iterator, List, Tuple, Optional

import numpy as np
import open3d as o3d
import yaml
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
sys.path.append(str(ROOT_DIR))

from lib.helper_ply import read_ply, write_ply


def load_config(config_path: str) -> dict:
    """設定ファイルを読み込み、必須キーの存在を確認する（フォールバック禁止）。"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    required_keys = [
        "input_path",
        "sp_path",
        "target_scenes",
        "ground_detection_method",
        "clustering_method",
        "vis",
        "vis_gt",
        "max_workers",
        "chunksize",
        "semantic_class",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"設定ファイルに必須キー '{key}' がありません: {config_path}")

    # bool options validation（文字列 "false" などを許さない）
    for k in ["vis", "vis_gt"]:
        if not isinstance(config[k], bool):
            raise ValueError(f"{k} は bool である必要があります: value={config[k]} type={type(config[k])}")

    gdm = config["ground_detection_method"]
    if gdm == "ransac":
        if "ransac" not in config:
            raise ValueError("ground_detection_method='ransac' のため 'ransac' セクションが必要です")
        for k in ["distance_threshold", "ransac_n", "num_iterations"]:
            if k not in config["ransac"]:
                raise ValueError(f"ransac に必須キー '{k}' がありません")
    elif gdm == "patchwork++":
        # patchwork++: ground labels are precomputed per raw scan
        if "patchwork" not in config:
            raise ValueError("ground_detection_method='patchwork++' のため 'patchwork' セクションが必要です")
        pw = config["patchwork"]
        for k in ["path", "ground_label"]:
            if k not in pw:
                raise ValueError(f"patchwork に必須キー '{k}' がありません")
        if not os.path.isdir(pw["path"]):
            raise FileNotFoundError(f"patchwork.path が存在しません: {pw['path']}")
        gl = int(pw["ground_label"])
        if gl < 0:
            raise ValueError(f"patchwork.ground_label は0以上である必要があります: {gl}")

        # nuScenes meta required to map token -> filename
        if "nuscenes" not in config:
            raise ValueError("ground_detection_method='patchwork++' のため 'nuscenes' セクションが必要です")
        ns = config["nuscenes"]
        for k in ["dataroot", "version"]:
            if k not in ns:
                raise ValueError(f"nuscenes に必須キー '{k}' がありません")
        if not os.path.isdir(ns["dataroot"]):
            raise FileNotFoundError(f"nuscenes.dataroot が存在しません: {ns['dataroot']}")
        version_dir = os.path.join(ns["dataroot"], ns["version"])
        if not os.path.isdir(version_dir):
            raise FileNotFoundError(f"nuscenes version dir が存在しません: {version_dir}")
    else:
        raise ValueError(
            f"無効な ground_detection_method: {gdm}（'ransac' or 'patchwork++'）"
        )

    cm = config["clustering_method"]
    if cm == "dbscan":
        if "dbscan" not in config:
            raise ValueError("clustering_method='dbscan' のため 'dbscan' セクションが必要です")
        for k in ["eps", "min_points"]:
            if k not in config["dbscan"]:
                raise ValueError(f"dbscan に必須キー '{k}' がありません")
    elif cm == "hdbscan":
        if "hdbscan" not in config:
            raise ValueError("clustering_method='hdbscan' のため 'hdbscan' セクションが必要です")
        for k in ["min_cluster_size", "min_samples", "metric", "cluster_selection_method"]:
            if k not in config["hdbscan"]:
                raise ValueError(f"hdbscan に必須キー '{k}' がありません")
    else:
        raise ValueError(f"無効な clustering_method: {cm}（'dbscan' or 'hdbscan'）")

    # target_scenes validation
    ts = config["target_scenes"]
    if isinstance(ts, str):
        if ts not in ("all", "first"):
            raise ValueError(
                "target_scenes が文字列の場合は 'all' または 'first' を指定してください: "
                f"target_scenes={ts}"
            )
    elif isinstance(ts, list):
        if len(ts) == 0:
            raise ValueError("target_scenes がリストの場合、空リストは無効です（少なくとも1つ指定してください）")
        for s in ts:
            if not isinstance(s, str):
                raise ValueError(f"target_scenes リスト内は文字列のみ許可: got={type(s)}")
    else:
        raise ValueError(
            "target_scenes は 'all'/'first' 文字列、または scene名のリストで指定してください: "
            f"type={type(ts)}"
        )

    return config


def ransac_ground_detection(coords: np.ndarray, config: dict) -> np.ndarray:
    """RANSACによる地面検出（インデックス配列を返す）"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)

    rc = config["ransac"]
    _, inliers = pcd.segment_plane(
        distance_threshold=float(rc["distance_threshold"]),
        ransac_n=int(rc["ransac_n"]),
        num_iterations=int(rc["num_iterations"]),
    )
    return np.array(inliers, dtype=np.int64)


def patchwork_ground_detection(
    label_path: str,
    num_points: int,
    ground_label: int,
) -> np.ndarray:
    """Patchwork++の地面ラベルファイルから地面点インデックスを取得する。

    nuScenes用のpatchwork++出力は 0/1 の二値ラベルを uint32 で保存している想定。
    """
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Patchwork++ label file not found: {label_path}")

    size = os.path.getsize(label_path)
    if size % 4 != 0:
        raise ValueError(f"Patchwork++ label file size is not multiple of 4 (uint32): size={size}, path={label_path}")

    labels = np.fromfile(label_path, dtype=np.uint32)
    if labels.shape[0] != num_points:
        raise ValueError(
            f"Patchwork++ ラベルの点数({labels.shape[0]})と入力点群の点数({num_points})が一致しません: {label_path}"
        )

    ground_indices = np.where(labels.astype(np.int64) == int(ground_label))[0]
    return ground_indices.astype(np.int64, copy=False)


def cluster_non_ground(coords: np.ndarray, config: dict) -> np.ndarray:
    """非地面点をクラスタリングし、クラスタID（-1含む）を返す"""
    cm = config["clustering_method"]
    if coords.shape[0] == 0:
        return np.array([], dtype=np.int64)

    if cm == "dbscan":
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        dc = config["dbscan"]
        labels = np.array(
            pcd.cluster_dbscan(
                eps=float(dc["eps"]),
                min_points=int(dc["min_points"]),
            ),
            dtype=np.int64,
        )
        return labels

    # hdbscan
    import hdbscan  # environment.yml で導入済み

    hc = config["hdbscan"]
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(hc["min_cluster_size"]),
        min_samples=int(hc["min_samples"]),
        metric=hc["metric"],
        cluster_selection_method=hc["cluster_selection_method"],
    )
    labels = clusterer.fit_predict(coords).astype(np.int64)
    return labels


def construct_superpoints(
    ply_path: str,
    config: dict,
    patchwork_label_path: Optional[str] = None,
):
    """単一のPLYから init superpoint を構築して *_superpoint.npy を保存する"""
    f = Path(ply_path)
    # read_ply は structured ndarray を返すが、型スタブが無いため Any 扱いにする
    data: Any = read_ply(str(f))
    coords = np.vstack((data["x"], data["y"], data["z"])).T.astype(np.float32, copy=False)

    # center for stable plane detection
    coords_centered = coords - coords.mean(0, keepdims=True)

    # scene_id / frame_name
    scene_id = f.parent.name  # e.g. scene-0001
    frame_name = f.name  # e.g. 000123.ply
    rel_name = os.path.join(scene_id, frame_name)  # for output relative path

    # ground detection
    gdm = config["ground_detection_method"]
    if gdm == "ransac":
        ground_index = ransac_ground_detection(coords_centered, config)
    else:
        if patchwork_label_path is None:
            raise ValueError(
                "patchwork++ が指定されていますが patchwork_label_path が渡されていません。"
                f"ply_path={ply_path}"
            )
        ground_index = patchwork_ground_detection(
            patchwork_label_path,
            num_points=coords.shape[0],
            ground_label=int(config["patchwork"]["ground_label"]),
        )
    ground_set = set(ground_index.tolist())
    other_index = np.array([i for i in range(coords.shape[0]) if i not in ground_set], dtype=np.int64)

    # clustering
    if other_index.shape[0] > 0:
        other_coords = coords_centered[other_index]
        other_region_idx = cluster_non_ground(other_coords, config)
    else:
        other_region_idx = np.array([], dtype=np.int64)

    # assign labels: ground is one cluster
    sp_labels = -np.ones((coords.shape[0],), dtype=np.int64)
    if other_index.shape[0] > 0:
        sp_labels[other_index] = other_region_idx
        ground_sp_label = int(other_region_idx.max()) + 1 if other_region_idx.size > 0 else 0
    else:
        ground_sp_label = 0
    sp_labels[ground_index] = ground_sp_label

    # save
    out_dir = os.path.join(config["sp_path"], scene_id)
    os.makedirs(out_dir, exist_ok=True)
    out_npy = os.path.join(config["sp_path"], rel_name[:-4] + "_superpoint.npy")
    np.save(out_npy, sp_labels.astype(np.int64))

    # vis
    if bool(config["vis"]):
        vis_dir = os.path.join(config["sp_path"], "vis", scene_id)
        os.makedirs(vis_dir, exist_ok=True)
        # simple colors per sp id
        uniq = np.unique(sp_labels[sp_labels >= 0])
        color_map: Dict[int, np.ndarray] = {}
        for i, sp in enumerate(uniq):
            color = np.array([(37 * i) % 255, (17 * i) % 255, (97 * i) % 255], dtype=np.uint8)
            color_map[int(sp)] = color
        colors = np.zeros((coords.shape[0], 3), dtype=np.uint8)
        for i in range(coords.shape[0]):
            sp = int(sp_labels[i])
            colors[i] = color_map.get(sp, np.array([0, 0, 0], dtype=np.uint8))
        out_vis = os.path.join(vis_dir, frame_name)
        write_ply(out_vis, [coords, colors], ["x", "y", "z", "red", "green", "blue"])

    # vis_gt (semantic ground-truth; keyframes only)
    if bool(config["vis_gt"]):
        base = f.name
        if not base.endswith(".ply"):
            raise ValueError(f"Unexpected PLY filename (expected *.ply): {base} (path={ply_path})")
        stem = base[:-4]
        parts = stem.split("_")
        if len(parts) < 2:
            raise ValueError(
                "vis_gt=true の場合、ファイル名から keyframe/sweep を判定するため "
                "nuScenes PLY名は '<6digits>_..._<k|s>.ply' を想定します: "
                f"got={base} (path={ply_path})"
            )
        idx_str = parts[0]
        if len(idx_str) != 6 or not idx_str.isdigit():
            raise ValueError(
                "vis_gt=true の場合、先頭に6桁のlocal indexが必要です（data_prepare_nuScenes.py の出力形式）: "
                f"got={base} (path={ply_path})"
            )
        kf_flag = parts[-1]
        if kf_flag == "s":
            # sweep は lidarseg GT が無いので保存しない
            return out_npy
        if kf_flag != "k":
            raise ValueError(
                "vis_gt=true の場合、ファイル名末尾の種別は 'k' or 's' を想定します: "
                f"got={base} (path={ply_path})"
            )

        required_fields = ["class", "red", "green", "blue"]
        names = getattr(data.dtype, "names", None)
        if names is None:
            raise ValueError(
                "vis_gt=true には入力PLYが structured PLY である必要があります（'class'/'red'/'green'/'blue' を含む）: "
                f"{ply_path}"
            )
        for rf in required_fields:
            if rf not in names:
                raise ValueError(
                    f"vis_gt=true には入力PLYに '{rf}' が必要です（data_prepare_nuScenes.py の出力を使用してください）: {ply_path}"
                )

        gt_labels = data["class"].astype(np.int32, copy=False)
        gt_colors = np.vstack((data["red"], data["green"], data["blue"])).T.astype(np.uint8, copy=False)

        vis_dir = os.path.join(config["sp_path"], "vis", scene_id)
        os.makedirs(vis_dir, exist_ok=True)
        out_gt = os.path.join(vis_dir, frame_name[:-4] + "_gt.ply")
        write_ply(
            out_gt,
            [coords, gt_colors, gt_labels.reshape(-1, 1)],
            ["x", "y", "z", "red", "green", "blue", "label"],
        )

    return out_npy


def construct_superpoints_patchwork_task(task: Tuple[str, str], config: dict) -> str:
    """(ply_path, patchwork_label_path) を受け取り、init SP を保存する（ProcessPool用）"""
    ply_path, label_path = task
    return construct_superpoints(ply_path, config, patchwork_label_path=label_path)


def main():
    config_path = os.path.join(BASE_DIR, "config_initialSP_nuscenes.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
    config = load_config(config_path)

    input_path = config["input_path"]
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"input_path が存在しません: {input_path}")

    input_root = Path(input_path)

    # 対象シーン決定
    scene_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()])
    if len(scene_dirs) == 0:
        raise FileNotFoundError(f"input_path 配下にシーンディレクトリが見つかりません: {input_path}")

    target_scenes = config["target_scenes"]
    if isinstance(target_scenes, str):
        if target_scenes == "all":
            selected_scene_dirs = scene_dirs
        else:  # "first"
            selected_scene_dirs = [scene_dirs[0]]
    else:
        # list[str]
        selected_scene_dirs = []
        for s in target_scenes:
            p = input_root / s
            if not p.exists():
                raise FileNotFoundError(f"target_scenes に指定されたシーンが見つかりません: {p}")
            if not p.is_dir():
                raise ValueError(f"target_scenes に指定されたパスがディレクトリではありません: {p}")
            selected_scene_dirs.append(p)

    # collect ply files only under selected scenes (recursive)
    ply_paths = []
    for scene_dir in selected_scene_dirs:
        ply_paths.extend(str(p) for p in scene_dir.rglob("*.ply"))
    ply_paths = sorted(ply_paths)
    if len(ply_paths) == 0:
        raise FileNotFoundError(
            f"PLYが見つかりません: input_path={input_path}, selected_scenes={[p.name for p in selected_scene_dirs]}"
        )

    print(f"Selected scenes: {[p.name for p in selected_scene_dirs]}")
    print(f"Found {len(ply_paths)} ply files under {input_path}")
    print(f"Output sp_path: {config['sp_path']}")
    print(f"ground: {config['ground_detection_method']} / clustering: {config['clustering_method']}")

    # patchwork++ 用に (ply_path -> label_path) を解決（メインプロセスで実行して、ワーカーはI/Oのみ）
    gdm = config["ground_detection_method"]
    if gdm == "patchwork++":
        try:
            from nuscenes.nuscenes import NuScenes
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "patchwork++ を使うには nuscenes-devkit が必要です。"
                "tcuss_vf 環境で `pip install nuscenes-devkit==1.1.11` を実行してください。"
            ) from e

        ns = config["nuscenes"]
        nusc = NuScenes(version=ns["version"], dataroot=ns["dataroot"], verbose=False)
        patchwork_root = config["patchwork"]["path"]

        # cache tokens per scene
        tokens_cache: Dict[str, List[str]] = {}

        def iter_tasks() -> Iterator[Tuple[str, str]]:
            for ply_path in ply_paths:
                pf = Path(ply_path)
                scene_id = pf.parent.name
                base = pf.name
                idx_str = base.split("_", 1)[0]
                if len(idx_str) != 6 or not idx_str.isdigit():
                    raise ValueError(f"Unexpected nuScenes PLY filename: {base} (path={ply_path})")
                local_idx = int(idx_str)

                if scene_id not in tokens_cache:
                    token_file = input_root / scene_id / "tokens.txt"
                    if not token_file.exists():
                        raise FileNotFoundError(
                            f"tokens.txt が見つかりません（data_prepare_nuScenes.py の出力に含まれます）: {token_file}"
                        )
                    with open(token_file, "r") as f:
                        tokens_cache[scene_id] = [line.strip() for line in f if line.strip()]

                tokens = tokens_cache[scene_id]
                if local_idx < 0 or local_idx >= len(tokens):
                    raise ValueError(
                        f"local_idx が tokens.txt の範囲外です: scene={scene_id}, local_idx={local_idx}, tokens={len(tokens)}, ply={ply_path}"
                    )
                token = tokens[local_idx]
                sd = nusc.get("sample_data", token)
                if "filename" not in sd:
                    raise ValueError(f"sample_data に filename がありません: token={token}")
                rel = sd["filename"]  # e.g. samples/LIDAR_TOP/xxx.pcd.bin
                if not rel.endswith(".pcd.bin"):
                    raise ValueError(f"Unexpected lidar filename (expected .pcd.bin): {rel} (token={token})")
                rel_label = rel[: -len(".pcd.bin")] + ".label"
                label_path = os.path.join(patchwork_root, rel_label)
                if not os.path.exists(label_path):
                    raise FileNotFoundError(f"patchwork++ label が見つかりません: {label_path} (token={token})")
                yield (ply_path, label_path)

        tasks = iter_tasks()
        process_func = partial(construct_superpoints_patchwork_task, config=config)
    else:
        tasks = ply_paths
        process_func = partial(construct_superpoints, config=config)
    max_workers = int(config["max_workers"])
    chunksize = int(config["chunksize"])
    if chunksize <= 0:
        raise ValueError(f"chunksize は正の値である必要があります: {chunksize}")

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for _ in tqdm(
            pool.map(process_func, tasks, chunksize=chunksize),
            total=len(ply_paths),
            desc="initialSP",
        ):
            pass

    print("Done.")


if __name__ == "__main__":
    main()


