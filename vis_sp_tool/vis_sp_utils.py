import torch
import torch.nn.functional as F
import numpy as np
import MinkowskiEngine as ME
import os
import sys
sys.path.append('..')

from lib.helper_ply import write_ply
from lib.utils import get_kmeans_labels
from models.fpn import Res16FPN18
from vis_sp_config import VisualizationConfig


def load_model(config: VisualizationConfig, device='cuda:0'):
    """学習済みモデルと分類器をロード"""
    # モデルの初期化
    model = Res16FPN18(
        in_channels=config.input_dim, 
        out_channels=config.feats_dim, 
        conv1_kernel_size=5,  # デフォルト値
        config=None  # とりあえずNone
    ).to(device)
    
    # 学習済み重みのロード
    if os.path.exists(config.model_path):
        checkpoint = torch.load(config.model_path, map_location=device)
        
        # チェックポイント形式の場合
        if 'model_q_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_q_state_dict'])
            print(f"チェックポイントからモデルをロードしました: {config.model_path}")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print(f"state_dictからモデルをロードしました: {config.model_path}")
        else:
            # 直接state_dictの場合
            model.load_state_dict(checkpoint)
            print(f"モデルをロードしました: {config.model_path}")
    else:
        raise FileNotFoundError(f"モデルファイルが見つかりません: {config.model_path}")
    
    # 分類器のロード（必要に応じて）
    classifier = None
    if config.classifier_path and os.path.exists(config.classifier_path):
        try:
            classifier = torch.load(config.classifier_path, map_location=device)
            print(f"分類器をロードしました: {config.classifier_path}")
        except Exception as e:
            print(f"分類器のロードに失敗しました: {e}")
    
    model.eval()
    return model, classifier


def extract_features(model, coords, device='cuda:0'):
    """モデルから特徴量を抽出"""
    with torch.no_grad():
        coords_tensor = torch.from_numpy(coords).float().to(device)
        in_field = ME.TensorField(coords_tensor[:, 1:] * 0.15, coords_tensor, device=device)
        feats = model(in_field)
        return feats.detach().cpu()


def compute_superpoints(coords, feats, region, current_growsp=None):
    """初期スーパーポイントから統合スーパーポイントを計算"""
    if current_growsp is None:
        # current_growspがNoneの場合は初期スーパーポイントを使用
        return region
    
    # 有効な領域のマスク
    valid_mask = region != -1
    if not valid_mask.any():
        print("警告: 有効なスーパーポイントが見つかりません")
        return region
    
    # 有効な領域のみを処理
    valid_coords = coords[valid_mask]
    valid_feats = feats[valid_mask]
    valid_region = region[valid_mask].astype(np.int64)
    
    # 初期スーパーポイントのラベルを連続する整数に再マッピング
    unique_regions = np.unique(valid_region)
    region_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_regions)}
    remapped_region = np.array([region_mapping[r] for r in valid_region])
    region_num = len(unique_regions)
    
    if region_num <= current_growsp:
        print(f"初期スーパーポイント数({region_num})が目標数({current_growsp})以下のため、そのまま使用します")
        return region
    
    # 初期スーパーポイントの特徴量を集約
    region_corr = F.one_hot(torch.from_numpy(remapped_region), num_classes=region_num).float()
    per_region_num = region_corr.sum(0, keepdims=True).t()
    region_feats = F.linear(region_corr.t(), valid_feats.t()) / per_region_num
    region_feats = F.normalize(region_feats, dim=-1)
    
    # K-meansクラスタリングで統合
    sp_idx = get_kmeans_labels(n_clusters=current_growsp, pcds=region_feats).long().cpu().numpy()
    
    # 統合結果を元の領域にマッピング
    neural_region = np.full_like(region, -1)
    neural_region[valid_mask] = sp_idx[remapped_region]
    
    return neural_region


def generate_colors(num_classes):
    """クラス数に応じた色を生成"""
    np.random.seed(42)  # 再現可能な色生成のため
    colors = np.random.randint(0, 256, size=(num_classes, 3), dtype=np.uint8)
    
    # 背景色（無効ラベル用）を黒に設定
    colors = np.vstack([[0, 0, 0], colors])
    
    return colors


def colorize_point_cloud(coords, superpoint_labels, colors=None):
    """スーパーポイントラベルに基づいて点群を色付け"""
    unique_labels = np.unique(superpoint_labels)
    num_classes = len(unique_labels[unique_labels != -1])  # -1（無効ラベル）を除く
    
    if colors is None:
        colors = generate_colors(num_classes)
    
    # ラベルを0から始まるインデックスにマッピング
    point_colors = np.zeros((len(coords), 3), dtype=np.uint8)
    
    for i, label in enumerate(unique_labels):
        if label == -1:
            # 無効ラベルは黒
            point_colors[superpoint_labels == label] = [0, 0, 0]
        else:
            # 有効ラベルは生成された色
            color_idx = label % len(colors)
            point_colors[superpoint_labels == label] = colors[color_idx]
    
    return point_colors


def save_colored_pointcloud(coords, colors, output_path, scene_name, file_extension='ply'):
    """色付きの点群をファイルに保存"""
    # 出力ディレクトリの作成
    seq_id = scene_name.split('/')[1]
    file_name = scene_name.split('/')[-1]
    
    # 指定された形式に従ってパスを作成
    # data/user/minesawa/semantickitti/vis_sp/sequences/{seq}/{velodyne}/{name}.{拡張子}
    output_dir = os.path.join(output_path, 'sequences', seq_id, 'velodyne')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f'{file_name}.{file_extension}')
    
    if file_extension == 'ply':
        # PLYファイルとして保存
        field_list = [coords.astype(np.float32), colors.astype(np.uint8)]
        field_names = ['x', 'y', 'z', 'red', 'green', 'blue']
        write_ply(output_file, field_list, field_names)
    else:
        # その他の形式の場合（numpy形式など）
        data = np.hstack([coords, colors])
        np.save(output_file, data)
    
    print(f"保存完了: {output_file}")
    return output_file


def save_superpoint_labels(superpoint_labels, output_path, scene_name, file_extension='npy'):
    """スーパーポイントラベルを保存"""
    seq_id = scene_name.split('/')[1]
    file_name = scene_name.split('/')[-1]
    
    # data/user/minesawa/semantickitti/vis_sp/sequences/{seq}/{labels}/{name}.{拡張子}
    output_dir = os.path.join(output_path, 'sequences', seq_id, 'labels')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f'{file_name}.{file_extension}')
    
    if file_extension == 'npy':
        np.save(output_file, superpoint_labels)
    else:
        # テキスト形式で保存
        np.savetxt(output_file, superpoint_labels, fmt='%d')
    
    print(f"スーパーポイントラベル保存完了: {output_file}")
    return output_file 