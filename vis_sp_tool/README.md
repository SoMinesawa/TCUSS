# Superpoint 可視化ツール

TCUSSプロジェクトで学習されたモデルを使用して、Superpointを可視化するためのツールです。

## 概要

このツールは以下の機能を提供します：

- 学習済みモデルを用いた特徴量抽出
- 初期スーパーポイントのK-meansクラスタリングによる統合
- Superpointごとに色付けされた点群の生成
- PLYファイルまたはその他の形式での保存
- SemanticKITTIデータセットに対応

## ファイル構成

- `visualize_superpoints.py`: メインの実行スクリプト
- `vis_sp_config.py`: 設定クラス
- `vis_sp_dataset.py`: データセットクラス
- `vis_sp_utils.py`: ユーティリティ関数
- `README.md`: このファイル

## 必要な依存関係

- PyTorch
- MinkowskiEngine
- NumPy
- tqdm
- TCUSSプロジェクトのライブラリ（lib/, models/）

## 使用方法

### 基本的な使用方法

```bash
cd vis_sp_tool
python visualize_superpoints.py \
    --model_path /path/to/trained_model.pth \
    --classifier_path /path/to/trained_classifier.pth \
    --current_growsp 50
```

### オプション引数

#### 必須引数
- `--model_path`: 学習済みモデルファイルのパス
- `--classifier_path`: 学習済み分類器ファイルのパス

#### 任意引数
- `--data_path`: 点群データパス（デフォルト: `data/users/minesawa/semantickitti/growsp`）
- `--sp_path`: 初期スーパーポイントパス（デフォルト: `data/users/minesawa/semantickitti/growsp_sp`）
- `--output_path`: 出力パス（デフォルト: `data/users/minesawa/semantickitti/vis_sp`）
- `--current_growsp`: 統合後のスーパーポイント数（デフォルト: None = 初期SPをそのまま使用）
- `--sequences`: 対象シーケンス（デフォルト: ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']）
- `--max_scenes`: 処理する最大シーン数（デフォルト: None = 全て）
- `--file_extension`: 出力ファイル拡張子（デフォルト: ply）
- `--debug`: デバッグモード（3シーンのみ処理）

### 使用例

#### 80個のスーパーポイントで可視化（学習開始時の設定）
```bash
python visualize_superpoints.py \
    --model_path ../data/users/minesawa/semantickitti/growsp_model/model_100_checkpoint.pth \
    --classifier_path ../data/users/minesawa/semantickitti/growsp_model/cls_100_checkpoint.pth \
    --current_growsp 80 \
    --sequences 00 01 02
```

#### 30個のスーパーポイントで可視化（学習終了時の設定）
```bash
python visualize_superpoints.py \
    --model_path ../data/users/minesawa/semantickitti/growsp_model/model_450_checkpoint.pth \
    --classifier_path ../data/users/minesawa/semantickitti/growsp_model/cls_450_checkpoint.pth \
    --current_growsp 30 \
    --sequences 00 01 02
```

#### 初期スーパーポイントをそのまま可視化
```bash
python visualize_superpoints.py \
    --model_path ../data/users/minesawa/semantickitti/growsp_model/model_100_checkpoint.pth \
    --classifier_path ../data/users/minesawa/semantickitti/growsp_model/cls_100_checkpoint.pth \
    --sequences 00
```

#### デバッグモードで実行
```bash
python visualize_superpoints.py \
    --model_path ../data/users/minesawa/semantickitti/growsp_model/model_100_checkpoint.pth \
    --classifier_path ../data/users/minesawa/semantickitti/growsp_model/cls_100_checkpoint.pth \
    --current_growsp 50 \
    --debug
```

## 出力形式

出力ファイルは以下の構造で保存されます：

```
data/users/minesawa/semantickitti/vis_sp/
└── sequences/
    ├── 00/
    │   ├── velodyne/
    │   │   ├── 000000.ply  # 色付き点群
    │   │   ├── 000001.ply
    │   │   └── ...
    │   └── labels/
    │       ├── 000000.npy  # スーパーポイントラベル
    │       ├── 000001.npy
    │       └── ...
    ├── 01/
    │   └── ...
    └── ...
```

### 出力ファイルの内容

#### velodyne/ディレクトリ（色付き点群）
- PLYファイル形式
- 各点にRGB色情報が付与
- 同じSuperpointに属する点は同じ色で表示

#### labels/ディレクトリ（スーパーポイントラベル）
- NumPy配列形式（.npy）
- 各点に対応するスーパーポイントのラベル番号
- -1は無効な点を示す

## 注意事項

1. TCUSSプロジェクトのルートディレクトリから実行することを前提としています
2. GPUが利用可能な環境での実行を推奨します
3. 大量のデータを処理する場合はディスク容量にご注意ください
4. デバッグモードは開発・テスト用です

## トラブルシューティング

### よくあるエラー

1. **モデルファイルが見つからない**
   - `--model_path`と`--classifier_path`が正しく指定されているか確認してください

2. **メモリ不足エラー**
   - `--max_scenes`で処理するシーン数を制限してください

3. **CUDA関連エラー**
   - GPUドライバーとCUDAのバージョンを確認してください

4. **スーパーポイントファイルが見つからない**
   - `--sp_path`が正しく設定されているか確認してください

### デバッグ方法

1. `--debug`オプションを使用して少数のシーンで動作確認
2. `--max_scenes 1`で単一シーンのみ処理してテスト

## 参考情報

このツールはTCUSSプロジェクトの学習プロセスと同じデータ前処理を使用します：

- ボクセルサイズ: 0.15m
- クロッピング半径: 50m
- データ拡張: 座標正規化のみ（回転・平行移動なし） 