# SemanticKITTI init SP 可視化ツール

## 概要

このツールは、SemanticKITTIの点群データからinit SP（初期スーパーポイント）を計算し、各SPを色分けしたPLYファイルとして出力します。

### init SPとは

init SPは、TCUSS（およびGrowSP）で使用される初期スーパーポイントです。以下の処理で生成されます：

1. **RANSAC**: 地面平面を検出し、地面点を1つのSPとしてグループ化
2. **DBSCAN**: 非地面点をクラスタリングし、各クラスタを個別のSPとして扱う

これにより、物体ごとに近似的なセグメンテーションが得られます。

## 使用方法

### 環境設定

```bash
cd /home/minesawa/repos/TCUSS
source tcuss_310/bin/activate  # または適切な仮想環境
```

### 基本的な使い方

```bash
# 特定のフレームを処理
python data_prepare/visualize_initSP_SemanticKITTI.py \
    --seq_id 00 \
    --frame_id 000000

# 特定のシーケンス全体を処理
python data_prepare/visualize_initSP_SemanticKITTI.py \
    --seq_id 00

# 学習用シーケンス（00-10）を全て処理
python data_prepare/visualize_initSP_SemanticKITTI.py \
    --train_only

# 全シーケンスを処理
python data_prepare/visualize_initSP_SemanticKITTI.py
```

### オプション

| オプション | デフォルト | 説明 |
|------------|-----------|------|
| `--data_path` | `data/dataset/semantickitti/dataset/sequences` | SemanticKITTIのsequencesディレクトリ |
| `--output_path` | `data/initSP_visualization` | PLYファイルの出力先 |
| `--seq_id` | None | 処理するシーケンスID（例: "00"） |
| `--frame_id` | None | 処理するフレームID（例: "000000"） |
| `--distance_threshold` | 0.1 | RANSACの距離閾値（m） |
| `--dbscan_eps` | 0.2 | DBSCANのeps（m） |
| `--dbscan_min_points` | 1 | DBSCANの最小点数 |
| `--max_workers` | 8 | 並列処理のワーカー数 |
| `--train_only` | False | 学習用シーケンス（00-10）のみ処理 |

### 出力

出力されるPLYファイルは以下の形式です：
- ファイル名: `{frame_id}_initsp.ply`
- 属性: x, y, z, red, green, blue
- 各init SPは異なる色で表示されます

## 可視化

出力されたPLYファイルは以下のソフトウェアで表示できます：

- **CloudCompare** (推奨): https://www.cloudcompare.org/
- **MeshLab**: https://www.meshlab.net/
- **Open3D**: Pythonコードで表示

### Open3Dで表示する例

```python
import open3d as o3d

# PLYファイルを読み込み
pcd = o3d.io.read_point_cloud("data/initSP_visualization/00/000000_initsp.ply")

# 表示
o3d.visualization.draw_geometries([pcd])
```

## 大規模処理

全シーケンスを処理する場合は、時間がかかるためバックグラウンドで実行することをお勧めします：

```bash
nohup python data_prepare/visualize_initSP_SemanticKITTI.py --train_only > initsp_vis.log 2>&1 &
```

## 関連ファイル

- `initialSP_prepare_SemanticKITTI.py`: init SPの.npyファイルを生成（学習用）
- `data_prepare_SemanticKITTI.py`: 生データをPLY形式に変換




