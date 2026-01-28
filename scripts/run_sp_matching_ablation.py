#!/usr/bin/env python3
"""
STC: Superpoint matching のアブレーション実行スクリプト

目的:
  - base YAML（例: config/stc.yaml）を読み込み
  - 以下4つの重みを {0, 0.5, 1.0} で総当たり（3^4=81）
      - weight_centroid_distance
      - weight_spread_similarity
      - weight_point_count_similarity
      - weight_remission_similarity
  - 各実験ごとに YAML を生成し、順番に学習を実行する
  - 各実験は total_epochs（デフォルト200）で終了させる（max_epoch=[phase0, phase1] の合計を total_epochs に合わせる）

例:
  # YAML生成だけ（実行しない）
  python scripts/run_sp_matching_ablation.py --dry-run

  # 8GPUで順次実行（torchrun）
  python scripts/run_sp_matching_ablation.py --nproc-per-node 8
"""

from __future__ import annotations

import argparse
import copy
import itertools
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import yaml


def _fmt_weight(w: float) -> str:
    """ファイル名/実験名用に重みを安全な文字列へ変換"""
    s = f"{float(w):g}"  # 0 / 0.5 / 1
    s = s.replace("-", "m").replace(".", "p")
    return s


def _compute_max_epoch_2phase(base_max_epoch: Any, total_epochs: int) -> List[int]:
    """2phase想定のmax_epochを total_epochs に合わせて作る（両phase>=1を保証）。"""
    if total_epochs < 2:
        raise ValueError(f"total_epochs must be >= 2: {total_epochs}")

    if not isinstance(base_max_epoch, list) or len(base_max_epoch) != 2:
        raise ValueError(f"base max_epoch must be a list of length 2, got: {base_max_epoch}")

    base0 = int(base_max_epoch[0])
    if base0 < 1:
        base0 = 1

    phase0 = min(base0, total_epochs - 1)  # phase1を最低1残す
    phase1 = total_epochs - phase0
    if phase1 < 1:
        raise ValueError(f"Invalid phase split: phase0={phase0}, phase1={phase1}, total={total_epochs}")

    return [int(phase0), int(phase1)]


def _make_cmd(
    launcher: str,
    nproc_per_node: int,
    train_script: str,
    config_path: str,
) -> List[str]:
    if launcher == "torchrun":
        return [
            "torchrun",
            f"--nproc_per_node={int(nproc_per_node)}",
            train_script,
            "--config",
            config_path,
        ]
    if launcher == "python":
        return [sys.executable, train_script, "--config", config_path]
    raise ValueError(f"Unknown launcher: {launcher}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run STC sp_matching ablation (grid search)")
    parser.add_argument("--base-config", type=str, default="config/stc.yaml", help="ベース設定YAML")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="config/ablations/stc_sp_matching",
        help="生成したYAMLの出力ディレクトリ",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0],
        help="各重みで試す値（例: 0 0.5 1.0）",
    )
    parser.add_argument("--total-epochs", type=int, default=200, help="各実験の総エポック数（max_epoch合計）")
    parser.add_argument("--train-script", type=str, default="train_SemanticKITTI.py", help="学習スクリプト")
    parser.add_argument("--launcher", choices=["torchrun", "python"], default="torchrun", help="起動方法")
    parser.add_argument("--nproc-per-node", type=int, default=1, help="torchrun時のGPUプロセス数")
    parser.add_argument(
        "--start-idx",
        type=int,
        default=1,
        help="実行するgridの開始番号（1始まり, inclusive）",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="実行するgridの終了番号（1始まり, inclusive）。未指定なら最後まで。",
    )
    parser.add_argument("--dry-run", action="store_true", help="YAML生成とコマンド表示のみ（実行しない）")
    parser.add_argument(
        "--if-exists",
        choices=["skip", "error", "overwrite"],
        default="skip",
        help="save_pathが既に存在する場合の挙動",
    )
    args = parser.parse_args()

    base_config_path = Path(args.base_config)
    if not base_config_path.exists():
        raise FileNotFoundError(f"base config not found: {base_config_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with base_config_path.open("r") as f:
        base_cfg: Dict[str, Any] = yaml.safe_load(f)

    base_name = str(base_cfg.get("name", base_config_path.stem))
    base_save_path = str(base_cfg.get("save_path", "runs"))
    base_max_epoch = base_cfg.get("max_epoch", [100, 100])
    max_epoch = _compute_max_epoch_2phase(base_max_epoch, int(args.total_epochs))

    grid: List[Tuple[float, float, float, float]] = list(itertools.product(args.weights, repeat=4))
    total = len(grid)

    start_idx = int(args.start_idx)
    end_idx = total if args.end_idx is None else int(args.end_idx)
    if start_idx < 1 or start_idx > total:
        raise ValueError(f"--start-idx must be in [1, {total}], got: {start_idx}")
    if end_idx < start_idx or end_idx > total:
        raise ValueError(f"--end-idx must be in [{start_idx}, {total}], got: {end_idx}")

    for idx, (wcd, ws, wpc, wr) in enumerate(grid, start=1):
        if idx < start_idx or idx > end_idx:
            continue
        exp_id = (
            f"wcd{_fmt_weight(wcd)}_"
            f"ws{_fmt_weight(ws)}_"
            f"wpc{_fmt_weight(wpc)}_"
            f"wr{_fmt_weight(wr)}_"
            f"e{int(args.total_epochs)}"
        )
        name = f"{base_name}_ablate_{exp_id}"
        save_path = os.path.join(base_save_path, "ablation", name)

        if Path(save_path).exists():
            if args.if_exists == "skip":
                print(f"[{idx:03d}/{total:03d}] SKIP (save_path exists): {name} -> {save_path}")
                continue
            if args.if_exists == "error":
                raise FileExistsError(f"save_path already exists: {save_path} (name={name})")
            # overwrite: 何もしない（学習側で上書き/追記される可能性がある点に注意）

        cfg = copy.deepcopy(base_cfg)
        cfg["name"] = name
        cfg["save_path"] = save_path
        cfg["max_epoch"] = max_epoch
        cfg["resume"] = False
        cfg["wandb_run_id"] = None

        stc = cfg.setdefault("stc", {})
        spm = stc.setdefault("sp_matching", {})
        spm["weight_centroid_distance"] = float(wcd)
        spm["weight_spread_similarity"] = float(ws)
        spm["weight_point_count_similarity"] = float(wpc)
        spm["weight_remission_similarity"] = float(wr)

        cfg_path = out_dir / f"{name}.yaml"
        with cfg_path.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        cmd = _make_cmd(
            launcher=str(args.launcher),
            nproc_per_node=int(args.nproc_per_node),
            train_script=str(args.train_script),
            config_path=str(cfg_path),
        )

        print(f"[{idx:03d}/{total:03d}] RUN: {name}")
        print("  CMD:", " ".join(cmd))

        if args.dry_run:
            continue

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()



