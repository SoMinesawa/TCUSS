import open3d as o3d
import numpy as np
from scipy import stats
from os.path import join, exists, dirname, abspath
import sys

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from lib.helper_ply import read_ply, write_ply
import time
import os
import yaml
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from tqdm import tqdm
import hdbscan

# カラーマップ初期化
colormap = []
for _ in range(10000):
    for k in range(12):
        colormap.append(plt.cm.Set3(k))
    for k in range(9):
        colormap.append(plt.cm.Set1(k))
    for k in range(8):
        colormap.append(plt.cm.Set2(k))
colormap.append((0, 0, 0, 0))
colormap = np.array(colormap)


def load_config(config_path: str) -> dict:
    """設定ファイルを読み込み、必須キーの存在を確認する"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 必須キーのチェック
    required_keys = [
        'input_path',
        'sp_path',
        'ground_detection_method',
        'clustering_method',
        'vis',
        'max_workers',
        'semantic_class',
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"設定ファイルに必須キー '{key}' がありません: {config_path}")
    
    # clustering_method に応じたチェック
    if config['clustering_method'] == 'dbscan':
        if 'dbscan' not in config:
            raise ValueError(f"clustering_method='dbscan' の場合、'dbscan' 設定が必要です")
        dbscan_required = ['eps', 'min_points']
        for key in dbscan_required:
            if key not in config['dbscan']:
                raise ValueError(f"設定ファイルの 'dbscan' に必須キー '{key}' がありません: {config_path}")
    elif config['clustering_method'] == 'hdbscan':
        if 'hdbscan' not in config:
            raise ValueError(f"clustering_method='hdbscan' の場合、'hdbscan' 設定が必要です")
        hdbscan_required = ['min_cluster_size', 'min_samples']
        for key in hdbscan_required:
            if key not in config['hdbscan']:
                raise ValueError(f"設定ファイルの 'hdbscan' に必須キー '{key}' がありません: {config_path}")
    else:
        raise ValueError(f"無効な clustering_method: {config['clustering_method']}。'dbscan' または 'hdbscan' を指定してください")
    
    # ground_detection_method に応じた追加チェック
    if config['ground_detection_method'] == 'ransac':
        if 'ransac' not in config:
            raise ValueError(f"ground_detection_method='ransac' の場合、'ransac' 設定が必要です")
        ransac_required = ['distance_threshold', 'ransac_n', 'num_iterations']
        for key in ransac_required:
            if key not in config['ransac']:
                raise ValueError(f"設定ファイルの 'ransac' に必須キー '{key}' がありません")
    elif config['ground_detection_method'] == 'patchwork++':
        if 'patchwork' not in config:
            raise ValueError(f"ground_detection_method='patchwork++' の場合、'patchwork' 設定が必要です")
        patchwork_required = ['path', 'ground_label']
        for key in patchwork_required:
            if key not in config['patchwork']:
                raise ValueError(f"設定ファイルの 'patchwork' に必須キー '{key}' がありません")
    else:
        raise ValueError(f"無効な ground_detection_method: {config['ground_detection_method']}。'ransac' または 'patchwork++' を指定してください")
    
    return config


def ransac_ground_detection(coords: np.ndarray, config: dict) -> np.ndarray:
    """RANSACによる地面検出"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    
    ransac_config = config['ransac']
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=ransac_config['distance_threshold'],
        ransac_n=ransac_config['ransac_n'],
        num_iterations=ransac_config['num_iterations']
    )
    return np.array(inliers)


def patchwork_ground_detection(seq_id: str, frame_name: str, num_points: int, config: dict) -> np.ndarray:
    """Patchwork++による地面検出"""
    patchwork_config = config['patchwork']
    
    # ラベルファイルパスを構築
    # frame_name は "000000.ply" のような形式
    frame_id = Path(frame_name).stem  # "000000"
    label_path = join(patchwork_config['path'], seq_id, 'predictions', f'{frame_id}.label')
    
    if not exists(label_path):
        raise FileNotFoundError(f"Patchwork++ ラベルファイルが見つかりません: {label_path}")
    
    # SemanticKITTI形式のラベルを読み込む（uint32、下位16ビットがセマンティックラベル）
    labels = np.fromfile(label_path, dtype=np.uint32)
    semantic_labels = labels & 0xFFFF  # 下位16ビットを取得
    
    if len(semantic_labels) != num_points:
        raise ValueError(f"Patchwork++ ラベルの点数({len(semantic_labels)})と入力点群の点数({num_points})が一致しません: {label_path}")
    
    # 地面ラベルに該当するインデックスを返す
    ground_indices = np.where(semantic_labels == patchwork_config['ground_label'])[0]
    return ground_indices


def construct_superpoints(path: str, config: dict):
    """
    単一のplyファイルに対してsuperpointを構築する
    
    Args:
        path: plyファイルへのパス
        config: 設定辞書
    """
    f = Path(path)
    data = read_ply(f)
    coords = np.vstack((data['x'], data['y'], data['z'])).T.copy()
    labels = data['class'].copy()
    labels -= 1
    coords_float = coords.astype(np.float32)
    coords_centered = coords_float - coords_float.mean(0)

    # シーケンスIDとフレーム名を取得
    seq_id = f.parts[-2]  # "00", "01", etc.
    frame_name = f.name  # "000000.ply"
    name = join(seq_id, frame_name)

    # 地面検出
    if config['ground_detection_method'] == 'ransac':
        ground_index = ransac_ground_detection(coords_centered, config)
    else:  # patchwork++
        ground_index = patchwork_ground_detection(seq_id, frame_name, coords.shape[0], config)
    
    # 地面以外の点を取得
    ground_set = set(ground_index)
    other_index = np.array([i for i in range(coords.shape[0]) if i not in ground_set])
    
    # 地面以外の点をクラスタリング
    if len(other_index) > 0:
        other_coords = coords_centered[other_index]
        
        if config['clustering_method'] == 'dbscan':
            # DBSCAN (Open3D)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(other_coords)
            dbscan_config = config['dbscan']
            other_region_idx = np.array(pcd.cluster_dbscan(
                eps=dbscan_config['eps'],
                min_points=dbscan_config['min_points']
            ))
        else:
            # HDBSCAN
            hdbscan_config = config['hdbscan']
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=hdbscan_config['min_cluster_size'],
                min_samples=hdbscan_config['min_samples'],
                metric=hdbscan_config.get('metric', 'euclidean'),
                cluster_selection_method=hdbscan_config.get('cluster_selection_method', 'eom')
            )
            other_region_idx = clusterer.fit_predict(other_coords)
    else:
        other_region_idx = np.array([])

    # superpointラベルの割り当て
    sp_labels = -np.ones_like(labels)
    if len(other_index) > 0:
        sp_labels[other_index] = other_region_idx
        ground_sp_label = other_region_idx.max() + 1 if len(other_region_idx) > 0 else 0
    else:
        ground_sp_label = 0
    sp_labels[ground_index] = ground_sp_label

    # 結果の保存
    sp_output_dir = join(config['sp_path'], seq_id)
    if not os.path.exists(sp_output_dir):
        os.makedirs(sp_output_dir, exist_ok=True)
    np.save(join(config['sp_path'], name[:-4] + '_superpoint.npy'), sp_labels)

    # 可視化
    if config['vis']:
        vis_path = join(config['sp_path'], 'vis', seq_id)
        if not os.path.exists(vis_path):
            os.makedirs(vis_path, exist_ok=True)
        
        colors = np.zeros_like(coords)
        for p in range(colors.shape[0]):
            colors[p] = 255 * (colormap[sp_labels[p].astype(np.int32)])[:3]
        colors = colors.astype(np.uint8)

        out_coords = np.vstack((data['x'], data['y'], data['z'])).T
        write_ply(vis_path + '/' + f.name, [out_coords, colors], ['x', 'y', 'z', 'red', 'green', 'blue'])

    # SPからGTへのマッピング
    sp2gt = -np.ones_like(labels)
    for sp in np.unique(sp_labels):
        if sp != -1:
            sp_mask = sp == sp_labels
            sp2gt[sp_mask] = stats.mode(labels[sp_mask])[0][0]

    return (labels, sp2gt)


def main():
    # 設定ファイルの読み込み
    config_path = join(BASE_DIR, 'config_initialSP_1.yaml')
    if not exists(config_path):
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
    
    config = load_config(config_path)
    
    # 処理対象シーケンスの取得
    target_sequences = config.get('sequences', None)
    if target_sequences is not None and len(target_sequences) > 0:
        # 指定されたシーケンスをゼロ埋め2桁文字列のセットに変換
        target_seq_set = set(f'{seq:02d}' for seq in target_sequences)
    else:
        target_seq_set = None  # 全シーケンス処理
    
    print(f'設定ファイル: {config_path}')
    print(f'地面検出方法: {config["ground_detection_method"]}')
    print(f'クラスタリング方法: {config["clustering_method"]}')
    if config['clustering_method'] == 'dbscan':
        print(f'  DBSCAN設定: eps={config["dbscan"]["eps"]}, min_points={config["dbscan"]["min_points"]}')
    else:
        hc = config['hdbscan']
        print(f'  HDBSCAN設定: min_cluster_size={hc["min_cluster_size"]}, min_samples={hc["min_samples"]}, '
              f'metric={hc.get("metric", "euclidean")}, cluster_selection_method={hc.get("cluster_selection_method", "eom")}')
    if target_seq_set:
        print(f'処理対象シーケンス: {sorted(target_seq_set)}')
    else:
        print('処理対象シーケンス: 全シーケンス')
    print('start constructing initial superpoints')
    
    trainval_path_list, test_path_list = [], []
    
    seq_list = np.sort(os.listdir(config['input_path']))
    
    for seq_id in tqdm(seq_list, desc='Scanning sequences'):
        # 特定のシーケンスのみを処理する場合はフィルタリング
        if target_seq_set is not None and seq_id not in target_seq_set:
            continue
        
        seq_path = join(config['input_path'], seq_id)
        if int(seq_id) < 11:
            for f in np.sort(os.listdir(seq_path)):
                if f.endswith('.ply'):
                    trainval_path_list.append(os.path.join(seq_path, f))
        else:
            for f in np.sort(os.listdir(seq_path)):
                if f.endswith('.ply'):
                    test_path_list.append(os.path.join(seq_path, f))
    
    print(f'TrainVal: {len(trainval_path_list)} files, Test: {len(test_path_list)} files')
    
    # functools.partialでconfigを固定した関数を作成（pickle可能）
    process_func = partial(construct_superpoints, config=config)
    
    pool = ProcessPoolExecutor(max_workers=config['max_workers'])
    
    # TrainValセットの処理
    if len(trainval_path_list) > 0:
        futures = [pool.submit(process_func, path) for path in trainval_path_list]
        
        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing trainval'):
            results.append(future.result())
        result = results
        
        print('end constructing initial superpoints (trainval)')
        
        # 精度評価 (trainval)
        all_labels, all_sp2gt = [], []
        for (labels, sp2gt) in result:
            mask = (sp2gt != -1)
            labels, sp2gt = labels[mask].astype(np.int32), sp2gt[mask].astype(np.int32)
            all_labels.append(labels)
            all_sp2gt.append(sp2gt)
        
        all_labels, all_sp2gt = np.concatenate(all_labels), np.concatenate(all_sp2gt)
        sem_num = config['semantic_class']
        mask = (all_labels >= 0) & (all_labels < sem_num)
        histogram = np.bincount(sem_num * all_labels[mask] + all_sp2gt[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)
        o_Acc = histogram[range(sem_num), range(sem_num)].sum() / histogram.sum()
        tp = np.diag(histogram)
        fp = np.sum(histogram, 0) - tp
        fn = np.sum(histogram, 1) - tp
        IoUs = tp / (tp + fp + fn + 1e-8)
        m_IoU = np.nanmean(IoUs)
        s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
        for IoU in IoUs:
            s += '{:5.2f} '.format(100 * IoU)
        print(' Acc: {:.5f}  TrainVal IoU'.format(o_Acc), s)
    else:
        print('TrainValセット: 処理対象なし')
    
    # テストセットの処理
    if len(test_path_list) > 0:
        futures = [pool.submit(process_func, path) for path in test_path_list]
        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing test'):
            results.append(future.result())
        print('end constructing initial superpoints (test)')
    else:
        print('テストセット: 処理対象なし')


if __name__ == '__main__':
    main()
