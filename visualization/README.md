# TCUSS 可視化データ生成ツール

このツールは、TCUSSプロジェクトにおける自動生成ラベルの精度を視覚化するために、ラベル付き点群をPLYファイル形式で保存するツールです。

## 概要

- KITTItemporalクラスと同様の前処理を実行
- 連続する12フレームをaggregate & segmentationして自動ラベルを生成
- frame 0~11, frame12-23, frame24-35...の順で処理
- 各フレームの座標データとセグメント情報をPLY形式で保存
- vispy+GPU可視化に対応

## 機能

### データ処理フロー

1. **前処理**: KITTItemporalと同じ前処理（voxelization, cropping等）
2. **集約**: 連続12フレームの点群を同一座標系に変換・集約
3. **セグメンテーション**: HDBSCANクラスタリングによる自動ラベル生成
4. **分割**: 各フレームに対応する点とラベルを抽出
5. **保存**: PLY形式で座標とセグメント情報を保存

### 出力形式

```
data/users/minesawa/semantickitti/vis/sequences/
├── 00/
│   ├── velodyne/
│   │   ├── 000000.ply      # 座標データのみ (x, y, z)
│   │   ├── 000001.ply
│   │   └── ...
│   ├── labels/
│   │   ├── 000000.ply      # ラベルデータのみ (label)
│   │   ├── 000001.ply
│   │   └── ...
│   ├── agg_coordinates/
│   │   ├── 000000-000011.ply  # 集約座標データ (x, y, z)
│   │   ├── 000012-000023.ply
│   │   └── ...
│   └── agg_segments/
│       ├── 000000-000011.ply  # 集約セグメント付きデータ (x, y, z, segment)
│       ├── 000012-000023.ply
│       └── ...
├── 01/
└── ...
```

## 使用方法

### 1. 基本的な実行方法

```bash
# すべてのシーケンスを処理
./visualization/run_vis_generation.sh

# 特定のシーケンスのみ処理
./visualization/run_vis_generation.sh --sequences 00 01 02
```

### 2. Pythonスクリプトを直接実行

```bash
# conda環境をアクティベート
conda activate tcuss

# すべてのシーケンスを処理
python visualization/generate_vis_data.py

# 特定のシーケンスのみ処理
python visualization/generate_vis_data.py --sequences 00 01

# カスタム設定で実行
python visualization/generate_vis_data.py \
    --data_path "data/users/minesawa/semantickitti/growsp" \
    --voxel_size 0.2 \
    --r_crop 60.0 \
    --sequences 02
```

### 3. パラメータ説明

| パラメータ | デフォルト値 | 説明 |
|-----------|------------|------|
| `--data_path` | `data/users/minesawa/semantickitti/growsp` | 点群データのパス |
| `--original_data_path` | `data/dataset/semantickitti/dataset/sequences` | オリジナルSemanticKITTIデータのパス |
| `--patchwork_path` | `data/users/minesawa/semantickitti/patchwork` | パッチワーク地面検出結果のパス |
| `--voxel_size` | `0.15` | ボクセルサイズ |
| `--r_crop` | `50.0` | クロッピング半径 |
| `--scan_window` | `12` | スキャンウィンドウサイズ |
| `--sequences` | `None` | 処理対象シーケンス（指定しない場合は全て） |

## 前提条件

### 必要なファイル構造

```
data/
├── users/minesawa/semantickitti/
│   ├── growsp/           # PLY形式の点群データ
│   │   ├── 00/
│   │   ├── 01/
│   │   └── ...
│   └── patchwork/        # 地面検出結果
│       ├── 00/
│       ├── 01/
│       └── ...
└── dataset/semantickitti/dataset/sequences/  # オリジナルSemanticKITTI
    ├── 00/
    │   ├── calib.txt
    │   ├── poses.txt
    │   └── ...
    ├── 01/
    └── ...
```

### 必要なライブラリ

- numpy
- torch
- MinkowskiEngine
- hdbscan
- tqdm
- lib.helper_ply (プロジェクト内)
- lib.config (プロジェクト内)

## 処理の詳細

### 1. データの前処理

KITTItemporalクラスと同じ前処理を適用：

- PLYファイルの読み込み
- 座標の正規化（重心を原点に）
- Voxelization（指定されたボクセルサイズで）
- Spherical cropping（指定された半径で）

### 2. 集約処理

- 連続12フレームの点群を同一座標系に変換
- ポーズファイル（poses.txt）を使用した座標変換
- Patchworkによる地面ラベルの統合

### 3. セグメンテーション

HDBSCANアルゴリズムを使用：
- 地面点は別途処理
- 非地面点に対してクラスタリング実行
- ノイズポイントは適切に処理

### 4. 保存処理

各フレームについて：
- `velodyne/{frame_id}.ply`: 座標データのみ (x, y, z)
- `labels/{frame_id}.ply`: ラベルデータのみ (label)

集約データについて（12フレーム単位）：
- `agg_coordinates/{start_idx}-{end_idx}.ply`: 集約された座標データ (x, y, z)
- `agg_segments/{start_idx}-{end_idx}.ply`: 集約されたセグメント付きデータ (x, y, z, segment)

## トラブルシューティング

### よくあるエラー

1. **ファイルが見つからない**
   ```
   FileNotFoundError: [Errno 2] No such file or directory
   ```
   → データパスの設定を確認してください

2. **メモリ不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   → 処理するシーケンスを制限するか、バッチサイズを調整してください

3. **権限エラー**
   ```
   PermissionError: [Errno 13] Permission denied
   ```
   → 出力ディレクトリの書き込み権限を確認してください

### デバッグ方法

1. 小さなシーケンスで試す：
   ```bash
   ./visualization/run_vis_generation.sh --sequences 04  # seq 04は271フレームで小さい
   ```

2. ログの確認：
   - 処理の進行状況はtqdmプログレスバーで表示
   - エラーメッセージは標準エラー出力に出力

## 可視化方法

生成されたPLYファイルは以下のツールで可視化できます：

- Open3D
- MeshLab
- CloudCompare
- 自作のvispy+GPU可視化ツール

### 簡単な可視化例（Open3D）

```python
import open3d as o3d
import numpy as np

# セグメント付きデータの読み込み
pcd = o3d.io.read_point_cloud("data/users/minesawa/semantickitti/vis/sequences/00/segments/000000.ply")
points = np.asarray(pcd.points)

# セグメント情報は別途読み込みが必要（PLYファイルの仕様による）
# または自作の読み込み関数を使用
```

## ライセンス

このツールはTCUSSプロジェクトの一部です。 