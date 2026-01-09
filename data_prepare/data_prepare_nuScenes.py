import os
import sys
import json
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple, Optional

import numpy as np
import yaml
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
sys.path.append(str(ROOT_DIR))

from lib.helper_ply import write_ply


_G_REMAP_LUT: Optional[np.ndarray] = None
_G_COLOR_LUT: Optional[np.ndarray] = None
_G_INTENSITY_DIVISOR: Optional[float] = None


def _init_worker(remap_lut: np.ndarray, color_lut: np.ndarray, intensity_divisor: float) -> None:
    global _G_REMAP_LUT, _G_COLOR_LUT, _G_INTENSITY_DIVISOR
    _G_REMAP_LUT = remap_lut
    _G_COLOR_LUT = color_lut
    _G_INTENSITY_DIVISOR = float(intensity_divisor)


def _convert_one(task: Tuple[str, str, str, str]) -> str:
    """1フレームをPLYへ変換（ProcessPool用ワーカー）.

    Args:
        task: (lidar_bin_path, lidarseg_bin_path_or_empty, out_ply_path, token)
    Returns:
        out_ply_path
    """
    lidar_bin, lidarseg_bin, out_ply, token = task

    if _G_REMAP_LUT is None or _G_COLOR_LUT is None or _G_INTENSITY_DIVISOR is None:
        raise RuntimeError("Worker globals are not initialized. initializer was not called.")

    xyz, intensity = read_nuscenes_lidar_bin(lidar_bin)

    if lidarseg_bin:
        if not os.path.exists(lidarseg_bin):
            raise FileNotFoundError(f"lidarseg bin が見つかりません: {lidarseg_bin} (token={token})")
        raw = np.fromfile(lidarseg_bin, dtype=np.uint8)
        if raw.shape[0] != xyz.shape[0]:
            raise ValueError(
                f"lidarseg の点数が一致しません: token={token}, "
                f"lidar_points={xyz.shape[0]}, lidarseg={raw.shape[0]}, label_file={lidarseg_bin}"
            )
        labels = _G_REMAP_LUT[raw.astype(np.int32)]
    else:
        labels = np.zeros((xyz.shape[0],), dtype=np.int32)

    colors = _G_COLOR_LUT[labels.clip(0, 16)]
    remission = (intensity / _G_INTENSITY_DIVISOR).astype(np.float32, copy=False)

    out_dir = os.path.dirname(out_ply)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ok = write_ply(
        out_ply,
        [xyz, colors, labels.reshape(-1, 1).astype(np.int32), remission.astype(np.float32)],
        ["x", "y", "z", "red", "green", "blue", "class", "remission"],
    )
    if not ok:
        raise RuntimeError(f"write_ply failed: {out_ply}")
    return out_ply


def load_config(config_path: str) -> dict:
    """設定ファイルを読み込み、必須キーの存在を確認する（フォールバック禁止）。"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    required_keys = [
        "dataroot",
        "version",
        "preprocess_path",
        "output_path",
        "max_workers",
        "chunksize",
        "use_preprocess_index",
        "lidar_sensor",
        "intensity_divisor",
        "label_remap",
    ]
    for k in required_keys:
        if k not in config:
            raise ValueError(f"設定ファイルに必須キー '{k}' がありません: {config_path}")

    # label_remap のチェック（0..31が全て存在）
    label_remap = config["label_remap"]
    if not isinstance(label_remap, dict):
        raise ValueError(f"label_remap は dict である必要があります: type={type(label_remap)}")
    missing = [i for i in range(32) if i not in label_remap]
    if missing:
        raise ValueError(f"label_remap に不足しているキーがあります: {missing}")

    # intensity_divisor
    intensity_div = float(config["intensity_divisor"])
    if intensity_div <= 0:
        raise ValueError(f"intensity_divisor は正の値である必要があります: {intensity_div}")

    # parallelism
    max_workers = int(config["max_workers"])
    chunksize = int(config["chunksize"])
    if max_workers <= 0:
        raise ValueError(f"max_workers は正の値である必要があります: {max_workers}")
    if chunksize <= 0:
        raise ValueError(f"chunksize は正の値である必要があります: {chunksize}")

    return config


def build_lidarseg_map(dataroot: str, version: str) -> Dict[str, str]:
    """sample_data_token -> lidarseg label file path"""
    lidarseg_json = os.path.join(dataroot, version, "lidarseg.json")
    if not os.path.exists(lidarseg_json):
        raise FileNotFoundError(f"lidarseg.json が見つかりません: {lidarseg_json}")

    with open(lidarseg_json, "r") as f:
        records = json.load(f)

    token_to_file: Dict[str, str] = {}
    for r in records:
        token = r.get("sample_data_token")
        filename = r.get("filename")
        if token is None or filename is None:
            raise ValueError(f"lidarseg.json のレコード形式が不正です: {r}")
        token_to_file[token] = os.path.join(dataroot, filename)

    return token_to_file


def load_preprocess_index(preprocess_path: str) -> Dict[str, List[str]]:
    """VoteFlow preprocess の index_total.pkl を読み込み、scene -> token list を構築（順序保持）。"""
    index_path = os.path.join(preprocess_path, "index_total.pkl")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"index_total.pkl が見つかりません: {index_path}")

    with open(index_path, "rb") as f:
        index_total = pickle.load(f)

    if not isinstance(index_total, list) or len(index_total) == 0:
        raise ValueError(f"index_total.pkl の形式が不正です: type={type(index_total)}, len={len(index_total) if hasattr(index_total,'__len__') else 'N/A'}")

    scene_to_tokens: Dict[str, List[str]] = {}
    for row in index_total:
        if not (isinstance(row, list) or isinstance(row, tuple)) or len(row) != 2:
            raise ValueError(f"index_total の行形式が不正です: {row}")
        scene_name, token = row
        if not isinstance(scene_name, str) or not isinstance(token, str):
            raise ValueError(f"index_total の型が不正です: scene={type(scene_name)}, token={type(token)}")
        scene_to_tokens.setdefault(scene_name, []).append(token)

    return scene_to_tokens


def make_color_map(num_classes: int = 17) -> np.ndarray:
    """0..num_classes-1 のラベルに対する簡易カラーマップ（uint8）"""
    # 0は黒（ignore）
    colors = np.zeros((num_classes, 3), dtype=np.uint8)
    base = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [255, 127, 0],
        [127, 255, 0],
        [0, 255, 127],
        [0, 127, 255],
        [127, 0, 255],
        [255, 0, 127],
        [200, 200, 200],
        [100, 100, 100],
        [180, 120, 60],
        [60, 120, 180],
    ], dtype=np.uint8)
    # 1..16に割当
    for i in range(1, min(num_classes, 17)):
        colors[i] = base[(i - 1) % len(base)]
    return colors


def read_nuscenes_lidar_bin(bin_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """nuScenes LIDAR_TOP .pcd.bin を読み込む。

    Returns:
        xyz: (N, 3) float32
        intensity: (N, 1) float32 (raw, not normalized)
    """
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"lidar bin が見つかりません: {bin_path}")
    arr = np.fromfile(bin_path, dtype=np.float32)
    if arr.size % 5 != 0:
        raise ValueError(f"pcd.bin の要素数が5で割り切れません（想定: x,y,z,intensity,ring）: size={arr.size}, path={bin_path}")
    pts = arr.reshape(-1, 5)
    xyz = pts[:, 0:3].astype(np.float32, copy=False)
    intensity = pts[:, 3:4].astype(np.float32, copy=False)
    return xyz, intensity


def main():
    # 設定ファイルの読み込み
    config_path = os.path.join(BASE_DIR, "config_nuscenes_prepare.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
    config = load_config(config_path)

    dataroot = config["dataroot"]
    version = config["version"]
    preprocess_path = config["preprocess_path"]
    output_path = config["output_path"]
    use_preprocess_index = bool(config["use_preprocess_index"])
    intensity_divisor = float(config["intensity_divisor"])
    max_workers = int(config["max_workers"])
    chunksize = int(config["chunksize"])

    # label remap LUT（0..31 -> 0..16）
    remap_dict = {int(k): int(v) for k, v in config["label_remap"].items()}
    remap_lut = np.zeros((32,), dtype=np.int32)
    for i in range(32):
        remap_lut[i] = remap_dict[i]

    os.makedirs(output_path, exist_ok=True)

    # lidarseg token -> label file
    lidarseg_map = build_lidarseg_map(dataroot, version)

    # 20Hz token index（VoteFlow preprocess; keyframe + sweep）
    if not use_preprocess_index:
        raise ValueError("use_preprocess_index=false は未サポートです（フォールバック禁止）。")
    scene_to_tokens = load_preprocess_index(preprocess_path)

    # nuScenes-devkit で token -> filename を解決
    try:
        from nuscenes.nuscenes import NuScenes
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "nuscenes-devkit が見つかりません。tcuss_vf 環境で `pip install nuscenes-devkit==1.1.11` を実行してください。"
        ) from e

    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

    color_lut = make_color_map(num_classes=17)  # 0..16

    scenes = sorted(scene_to_tokens.keys())
    print(f"Found scenes in preprocess index: {len(scenes)}")
    print(f"Output: {output_path}")

    total_frames = sum(len(scene_to_tokens[s]) for s in scenes)
    print(f"Total frames (20Hz, key+sweep): {total_frames}")
    print(f"ProcessPool: max_workers={max_workers}, chunksize={chunksize}")

    def iter_tasks():
        for scene_name in scenes:
            tokens = scene_to_tokens[scene_name]
            if len(tokens) == 0:
                continue

            scene_out_dir = os.path.join(output_path, scene_name)
            os.makedirs(scene_out_dir, exist_ok=True)

            # token list を保存（後でSTC等で使える）
            token_txt = os.path.join(scene_out_dir, "tokens.txt")
            with open(token_txt, "w") as f:
                for t in tokens:
                    f.write(t + "\n")

            for local_idx, token in enumerate(tokens):
                sd = nusc.get("sample_data", token)
                if "filename" not in sd:
                    raise ValueError(f"sample_data に filename がありません: token={token}")
                if "is_key_frame" not in sd:
                    raise ValueError(f"sample_data に is_key_frame がありません: token={token}")

                lidar_bin = os.path.join(dataroot, sd["filename"])
                label_file = lidarseg_map.get(token, "")

                suffix = "k" if bool(sd["is_key_frame"]) else "s"
                out_ply = os.path.join(scene_out_dir, f"{local_idx:06d}_{suffix}.ply")
                yield (lidar_bin, label_file, out_ply, token)

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(remap_lut, color_lut, intensity_divisor),
    ) as pool:
        for _ in tqdm(
            pool.map(_convert_one, iter_tasks(), chunksize=chunksize),
            total=total_frames,
            desc="Convert PLY",
        ):
            pass

    print("Done.")


if __name__ == "__main__":
    main()


