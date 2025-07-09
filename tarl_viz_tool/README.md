# TARL Clustering Visualization Tool

TARLクラスタリングのハイパーパラメータ最適化のためのビジュアライゼーションツールです。

## 概要

このツールは、SemanticKITTIデータセットを使用してTARLクラスタリングの性能を視覚的に評価し、最適なハイパーパラメータを決定するためのツールです。

### 主な機能

1. **3つの表示モード**
   - Ground-truth：SemanticKITTI標準のセマンティックラベル表示
   - K-means：K-meansクラスタリング結果の表示
   - HDBSCAN：HDBSCANクラスタリング結果の表示

2. **インタラクティブな操作**
   - 矢印キーによるフレーム間移動
   - 1-3キーによる表示モード切り替え
   - マウスによる視点操作

3. **実行時間の計測**
   - 各クラスタリング手法の実行時間を計測・表示

4. **TCUSSとの整合性**
   - TCUSSのKITTItemporalクラスと同じ前処理パイプラインを使用
   - 実際のモデル学習時と同じ条件でクラスタリングを実行

## インストール

### 依存関係

```bash
pip install numpy open3d scikit-learn hdbscan matplotlib
```

### MinkowskiEngineのインストール

```bash
# CUDAが利用可能な場合
pip install MinkowskiEngine

# CPUのみの場合
pip install MinkowskiEngine --no-deps
```

## 使用方法

### 基本的な使用方法

```bash
# TCUSSディレクトリから実行
python -m tarl_viz_tool.main --data_path data/dataset/semantickitti/dataset/sequences --seq 00
```

### コマンドライン引数

#### データパス設定
- `--data_path`：PLYファイルのパス（デフォルト: `data/dataset/semantickitti/dataset/sequences`）
- `--original_data_path`：ポーズ情報用のパス（デフォルト: `data/dataset/semantickitti/dataset/sequences`）
- `--patchwork_path`：地面ラベル用のパス（デフォルト: `data/users/minesawa/semantickitti/patchwork`）

#### 表示設定
- `--seq`：シーケンス番号（デフォルト: `00`）
- `--start_frame`：開始フレーム（デフォルト: `0`）
- `--end_frame`：終了フレーム（デフォルト: 自動検出）

#### 前処理パラメータ
- `--voxel_size`：ボクセルサイズ（デフォルト: `0.15`）
- `--r_crop`：クロッピング半径（デフォルト: `50.0`）
- `--scan_window`：時系列ウィンドウサイズ（デフォルト: `12`）

#### クラスタリングパラメータ
**HDBSCAN**
- `--min_cluster_size`：最小クラスターサイズ（デフォルト: `20`）
- `--min_samples`：最小サンプル数（デフォルト: `50`）
- `--cluster_selection_epsilon`：クラスター選択エプシロン（デフォルト: `0.0`）

**K-means**
- `--n_clusters`：クラスター数（デフォルト: `50`）
- `--max_iter`：最大イテレーション数（デフォルト: `300`）

#### 表示オプション
- `--window_width`：ウィンドウ幅（デフォルト: `1200`）
- `--window_height`：ウィンドウ高さ（デフォルト: `800`）
- `--point_size`：点のサイズ（デフォルト: `2.0`）

### 使用例

```bash
# シーケンス08で大きなクラスターを使用
python -m tarl_viz_tool.main --seq 08 --min_cluster_size 100 --n_clusters 30

# 小さなボクセルサイズで詳細な分析
python -m tarl_viz_tool.main --voxel_size 0.1 --r_crop 30.0

# デバッグモード
python -m tarl_viz_tool.main --debug --verbose
```

## 操作方法

### キーボード操作

| キー | 機能 |
|------|------|
| ← / → | 前/次のフレーム |
| 1 | Ground-truth表示 |
| 2 | K-means結果表示 |
| 3 | HDBSCAN結果表示 |
| R | 視点リセット |
| S | スクリーンショット保存 |
| H | ヘルプ表示 |
| Q / ESC | 終了 |

### マウス操作

| 操作 | 機能 |
|------|------|
| 左クリック + ドラッグ | 視点回転 |
| 右クリック + ドラッグ | 視点移動 |
| スクロール | ズーム |

## 出力結果

### スクリーンショット
- `S`キーで現在の表示をPNG形式で保存
- 保存先：`tarl_viz_results/tarl_viz_YYYYMMDD_HHMMSS.png`

### ログ情報
- クラスタリング実行時間
- 各フレームの統計情報
- 前処理の詳細

## トラブルシューティング

### よくある問題

1. **PLYファイルが見つからない**
   ```
   エラー: PLYファイルが見つかりません
   ```
   → データパスとシーケンス番号を確認してください

2. **MinkowskiEngineエラー**
   ```
   ImportError: MinkowskiEngine
   ```
   → MinkowskiEngineのインストールを確認してください

3. **Open3Dの表示エラー**
   ```
   Open3D visualization error
   ```
   → ディスプレイ設定とOpenGLドライバーを確認してください

### パフォーマンスの最適化

1. **メモリ使用量の削減**
   - `r_crop`値を小さくする
   - `voxel_size`を大きくする

2. **実行速度の向上**
   - `n_clusters`を調整する
   - `min_cluster_size`を大きくする

## 開発者向け情報

### ファイル構造

```
tarl_viz_tool/
├── __init__.py          # パッケージ初期化
├── main.py              # メインエントリーポイント
├── data_loader.py       # SemanticKITTIデータローダー
├── preprocessor.py      # データ前処理
├── clustering.py        # クラスタリング管理
├── visualizer.py        # Open3Dビジュアライザー
├── utils.py            # ユーティリティ関数
└── README.md           # このファイル
```

### 拡張方法

1. **新しいクラスタリング手法の追加**
   - `clustering.py`に新しいメソッドを追加
   - `visualizer.py`に表示モードを追加

2. **新しいデータ形式の対応**
   - `data_loader.py`を拡張

3. **カスタムビジュアライゼーション**
   - `visualizer.py`を拡張

## ライセンス

このツールはTCUSSプロジェクトの一部として開発されました。

## 更新履歴

- **v1.0.0** (2024-01-XX)
  - 初期リリース
  - Ground-truth、K-means、HDBSCAN表示機能
  - インタラクティブな操作機能
  - 実行時間計測機能