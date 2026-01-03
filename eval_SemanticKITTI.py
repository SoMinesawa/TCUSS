import torch
import torch.nn.functional as F
import torch.distributed as dist
from datasets.SemanticKITTI import KITTIval, cfl_collate_fn_val
import numpy as np
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
try:
    # 古い scikit-learn がある場合（バージョン0.22.2等）
    from sklearn.utils.linear_assignment_ import linear_assignment
    def assignment_function(cost_matrix):
        # sklearn版は直接 (N, 2) で返る
        return linear_assignment(cost_matrix)
except ModuleNotFoundError:
    # scikit-learn版が無い or 新しい scikit-learn（linear_assignment_ が削除済み）の場合
    from scipy.optimize import linear_sum_assignment
    def assignment_function(cost_matrix):
        # scipy版は (row_idx, col_idx) のタプル
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        # 同じ形状 (N, 2) に変形して返す
        return np.stack([row_idx, col_idx], axis=1)
from models.fpn import Res16FPN18
from lib.utils import get_fixclassifier, get_kmeans_labels
from lib.helper_ply import read_ply, write_ply
import warnings
import argparse
import random
import os
import re
from tqdm import tqdm
from typing import Tuple, Dict, List, Optional, Union, Any, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from lib.config import TCUSSConfig

# 移動物体のraw labelリスト（SemanticKITTI定義）
MOVING_RAW_LABELS: Set[int] = {252, 253, 254, 255, 256, 257, 258, 259}

###
def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する関数"""
    parser = argparse.ArgumentParser(description='TCUSS - SemanticKITTI Evaluation')
    parser.add_argument('--data_path', type=str, default='data/users/minesawa/semantickitti/growsp',
                        help='点群データパス')
    parser.add_argument('--sp_path', type=str, default='data/users/minesawa/semantickitti/growsp_sp',
                        help='初期スーパーポイントパス')
    parser.add_argument('--save_path', type=str, default='data/users/minesawa/semantickitti/growsp_model',
                        help='モデル保存パス')
    ###
    parser.add_argument('--bn_momentum', type=float, default=0.02, help='バッチ正規化のパラメータ')
    parser.add_argument('--conv1_kernel_size', type=int, default=5, help='第1畳み込み層のカーネルサイズ')
    ####
    parser.add_argument('--workers', type=int, default=24, help='データローディング用ワーカー数')
    parser.add_argument('--cluster_workers', type=int, default=24, help='クラスタリング用ワーカー数')
    parser.add_argument('--seed', type=int, default=2022, help='乱数シード')
    parser.add_argument('--voxel_size', type=float, default=0.15, help='SparseConvのボクセルサイズ')
    parser.add_argument('--input_dim', type=int, default=3, help='ネットワーク入力次元')
    parser.add_argument('--primitive_num', type=int, default=500, help='学習に使用するプリミティブ数')
    parser.add_argument('--semantic_class', type=int, default=19, help='意味クラス数')
    parser.add_argument('--feats_dim', type=int, default=128, help='出力特徴次元')
    parser.add_argument('--ignore_label', type=int, default=-1, help='無効ラベル')
    parser.add_argument('--eval_epoch', type=int, help='評価する特定のエポック（指定しない場合は全エポック）')
    parser.add_argument('--eval_select_num', type=int, default=4071, help='評価に使用するデータ数')
    parser.add_argument('--eval_backward_num', type=int, default=1, help='eval_epoch指定時に逆順で評価するエポック数')
    return parser.parse_args()

def set_seed(seed: int) -> None:
    """乱数シードを設定する関数
    
    Args:
        seed: 乱数シード
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def eval_once(
    args: argparse.Namespace, 
    model: torch.nn.Module, 
    test_loader: DataLoader, 
    classifier: torch.nn.Module, 
    epoch: int,
    device: int = 0,
    is_main_process: bool = True
) -> Tuple[List, List, List, List]:
    """一回の評価を実行する関数（DDP対応版）
    
    Args:
        args: コマンドライン引数
        model: 評価するモデル
        test_loader: テストデータローダー
        classifier: 分類器
        epoch: 評価するエポック
        device: GPUデバイスID
        is_main_process: メインプロセスかどうか（tqdm表示用）
    
    Returns:
        all_preds: すべての予測結果
        all_label: すべてのラベル
        all_distances: すべての距離（原点からの2D距離）
        all_is_moving: すべての移動物体フラグ
    """
    all_preds, all_label, all_distances, all_is_moving = [], [], [], []
    
    # メインプロセスのみtqdmで表示
    iterator = test_loader
    if is_main_process:
        iterator = tqdm(test_loader, desc=f'Eval Epoch: {epoch}')
    
    for data in iterator:
        with torch.no_grad():
            coords, features, inverse_map, labels, index, region, original_coords, raw_labels = data

            # TensorFieldを作成（DDPデバイスを指定）
            in_field = ME.TensorField(coords[:, 1:] * args.voxel_size, coords, device=device)
            feats = model(in_field)
            feats = F.normalize(feats, dim=1)

            # スコア計算と予測
            scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
            preds = torch.argmax(scores, dim=1).cpu()

            preds = preds[inverse_map.long()]
            
            # 距離を計算（原点からの2D距離）
            distances = torch.sqrt(original_coords[:, 0]**2 + original_coords[:, 1]**2)
            
            # 移動物体フラグを計算
            is_moving = torch.tensor([int(r.item()) in MOVING_RAW_LABELS for r in raw_labels], dtype=torch.bool)
            
            # ignore_labelを除外
            valid_mask = labels != args.ignore_label
            preds = preds[valid_mask]
            labels = labels[valid_mask]
            distances = distances[valid_mask]
            is_moving = is_moving[valid_mask]
            
            all_preds.append(preds)
            all_label.append(labels)
            all_distances.append(distances)
            all_is_moving.append(is_moving)

    return all_preds, all_label, all_distances, all_is_moving


def compute_metrics_for_subset(
    preds: np.ndarray, 
    labels: np.ndarray, 
    sem_num: int,
    matching: np.ndarray
) -> Tuple[float, float, float]:
    """サブセットに対してメトリクスを計算する
    
    Args:
        preds: 予測ラベル
        labels: 正解ラベル
        sem_num: セマンティッククラス数
        matching: ハンガリアンマッチング結果（全体で計算済み）
    
    Returns:
        o_Acc, m_Acc, m_IoU
    """
    if len(preds) == 0 or len(labels) == 0:
        return 0.0, 0.0, 0.0
    
    mask = (labels >= 0) & (labels < sem_num)
    if not mask.any():
        return 0.0, 0.0, 0.0
    
    histogram = np.bincount(
        sem_num * labels[mask] + preds[mask], 
        minlength=sem_num ** 2
    ).reshape(sem_num, sem_num)
    
    return compute_metrics_from_histogram(histogram, sem_num, matching)


def compute_metrics_from_histogram(
    histogram: np.ndarray,
    sem_num: int,
    matching: np.ndarray
) -> Tuple[float, float, float]:
    """混同行列からメトリクスを計算する
    
    Args:
        histogram: 混同行列 (sem_num, sem_num)
        sem_num: セマンティッククラス数
        matching: ハンガリアンマッチング結果
    
    Returns:
        o_Acc, m_Acc, m_IoU
    """
    total = histogram.sum()
    if total == 0:
        return 0.0, 0.0, 0.0
    
    # 既存のマッチングを使用してhistogramを再配置
    hist_new = np.zeros((sem_num, sem_num))
    for idx in range(sem_num):
        hist_new[:, idx] = histogram[:, matching[idx, 1]]
    
    o_Acc = histogram[matching[:, 0], matching[:, 1]].sum() / total * 100.0
    
    # クラスごとの正解数
    class_totals = histogram.sum(1)
    class_totals[class_totals == 0] = 1  # 0除算防止
    m_Acc = np.mean(histogram[matching[:, 0], matching[:, 1]] / class_totals) * 100.0
    
    # IoU計算
    tp = np.diag(hist_new)
    fp = np.sum(hist_new, 0) - tp
    fn = np.sum(hist_new, 1) - tp
    IoUs = (100 * tp) / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    
    return o_Acc, m_Acc, m_IoU


def compute_distance_histograms(
    preds: np.ndarray,
    labels: np.ndarray,
    distances: np.ndarray,
    distance_bins: List[int],
    sem_num: int
) -> Dict[str, np.ndarray]:
    """距離帯ごとの混同行列を計算する（histogram形式）
    
    Args:
        preds: 予測ラベル
        labels: 正解ラベル
        distances: 距離
        distance_bins: 距離区切り
        sem_num: セマンティッククラス数
    
    Returns:
        距離帯ごとの混同行列辞書
    """
    result = {}
    valid_mask = (labels >= 0) & (labels < sem_num)
    
    prev_dist = 0
    for dist in distance_bins:
        mask = valid_mask & (distances >= prev_dist) & (distances < dist)
        key = f'{prev_dist}-{dist}'
        
        if mask.sum() > 0:
            histogram = np.bincount(
                sem_num * labels[mask] + preds[mask],
                minlength=sem_num ** 2
            ).reshape(sem_num, sem_num)
        else:
            histogram = np.zeros((sem_num, sem_num), dtype=np.int64)
        
        result[key] = histogram
        prev_dist = dist
    
    # 最後の距離帯以降
    mask = valid_mask & (distances >= distance_bins[-1])
    key = f'{distance_bins[-1]}+'
    
    if mask.sum() > 0:
        histogram = np.bincount(
            sem_num * labels[mask] + preds[mask],
            minlength=sem_num ** 2
        ).reshape(sem_num, sem_num)
    else:
        histogram = np.zeros((sem_num, sem_num), dtype=np.int64)
    
    result[key] = histogram
    
    return result


def compute_moving_static_histograms(
    preds: np.ndarray,
    labels: np.ndarray,
    is_moving: np.ndarray,
    sem_num: int
) -> Dict[str, np.ndarray]:
    """移動/静止別の混同行列を計算する（histogram形式）
    
    Args:
        preds: 予測ラベル
        labels: 正解ラベル
        is_moving: 移動物体フラグ
        sem_num: セマンティッククラス数
    
    Returns:
        移動/静止別の混同行列辞書
    """
    result = {}
    valid_mask = (labels >= 0) & (labels < sem_num)
    
    # 移動物体
    mask = valid_mask & is_moving
    if mask.sum() > 0:
        histogram = np.bincount(
            sem_num * labels[mask] + preds[mask],
            minlength=sem_num ** 2
        ).reshape(sem_num, sem_num)
    else:
        histogram = np.zeros((sem_num, sem_num), dtype=np.int64)
    result['moving'] = histogram
    
    # 静止物体
    mask = valid_mask & ~is_moving
    if mask.sum() > 0:
        histogram = np.bincount(
            sem_num * labels[mask] + preds[mask],
            minlength=sem_num ** 2
        ).reshape(sem_num, sem_num)
    else:
        histogram = np.zeros((sem_num, sem_num), dtype=np.int64)
    result['static'] = histogram
    
    return result


def compute_distance_metrics_from_histograms(
    distance_histograms: Dict[str, np.ndarray],
    sem_num: int,
    matching: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """距離帯ごとの混同行列からメトリクスを計算する
    
    Args:
        distance_histograms: 距離帯ごとの混同行列辞書
        sem_num: セマンティッククラス数
        matching: ハンガリアンマッチング結果
    
    Returns:
        距離帯ごとのメトリクス辞書
    """
    result = {}
    
    for key, histogram in distance_histograms.items():
        count = int(histogram.sum())
        if count > 0:
            o_Acc, m_Acc, m_IoU = compute_metrics_from_histogram(histogram, sem_num, matching)
        else:
            o_Acc, m_Acc, m_IoU = 0.0, 0.0, 0.0
        
        result[key] = {
            'oAcc': o_Acc,
            'mAcc': m_Acc,
            'mIoU': m_IoU,
            'count': count
        }
    
    return result


def compute_moving_static_metrics_from_histograms(
    moving_static_histograms: Dict[str, np.ndarray],
    sem_num: int,
    matching: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """移動/静止別の混同行列からメトリクスを計算する
    
    Args:
        moving_static_histograms: 移動/静止別の混同行列辞書
        sem_num: セマンティッククラス数
        matching: ハンガリアンマッチング結果
    
    Returns:
        移動/静止ごとのメトリクス辞書
    """
    result = {}
    
    for key, histogram in moving_static_histograms.items():
        count = int(histogram.sum())
        if count > 0:
            o_Acc, m_Acc, m_IoU = compute_metrics_from_histogram(histogram, sem_num, matching)
        else:
            o_Acc, m_Acc, m_IoU = 0.0, 0.0, 0.0
        
        result[key] = {
            'oAcc': o_Acc,
            'mAcc': m_Acc,
            'mIoU': m_IoU,
            'count': count
        }
    
    return result


def eval_ddp(
    epoch: int, 
    args: Union[argparse.Namespace, 'TCUSSConfig'],
    model: torch.nn.Module,
    classifier: torch.nn.Module,
    local_rank: int = 0,
    world_size: int = 1,
    is_main_process: bool = True
) -> Tuple[float, float, float, str, Dict[str, float], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """モデルを評価する関数（DDP対応版）
    
    Args:
        epoch: 評価するエポック
        args: コマンドライン引数またはTCUSSConfig
        model: 評価するモデル（DDPラップされている可能性あり）
        classifier: 分類器
        local_rank: このプロセスのローカルランク
        world_size: 総プロセス数
        is_main_process: メインプロセスかどうか
    
    Returns:
        o_Acc: 全体の精度
        m_Acc: 平均精度
        m_IoU: 平均IoU
        s: IoU情報の文字列
        IoU_dict: クラスごとのIoU辞書
        distance_metrics: 距離別メトリクス
        moving_static_metrics: 移動/静止別メトリクス
    """
    use_ddp = world_size > 1
    device = f"cuda:{local_rank}"
    
    # モデルを評価モードに
    model.eval()
    classifier.eval()
    
    # 評価用バッチサイズとワーカー数を取得
    eval_batch_size = getattr(args, 'eval_batch_size', 32)
    persistent_workers = getattr(args, 'persistent_workers', True)
    prefetch_factor = getattr(args, 'prefetch_factor', 4)
    num_workers = getattr(args, 'eval_workers', args.cluster_workers)
    
    # 検証データセットを作成
    val_dataset = KITTIval(args)
    
    # DDP時はDistributedSamplerを使用
    val_sampler = None
    if use_ddp:
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=eval_batch_size, 
        collate_fn=cfl_collate_fn_val(), 
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        sampler=val_sampler,
        shuffle=False
    )

    # 評価を実行（各GPUで一部のデータを処理）
    preds, labels, distances, is_moving = eval_once(
        args, model, val_loader, classifier, epoch, 
        device=local_rank, is_main_process=is_main_process
    )
    
    # 結果を連結
    if len(preds) > 0:
        local_preds = torch.cat(preds)
        local_labels = torch.cat(labels)
        local_distances = torch.cat(distances)
        local_is_moving = torch.cat(is_moving)
    else:
        local_preds = torch.tensor([], dtype=torch.long)
        local_labels = torch.tensor([], dtype=torch.long)
        local_distances = torch.tensor([], dtype=torch.float)
        local_is_moving = torch.tensor([], dtype=torch.bool)
    
    sem_num = args.semantic_class
    
    # numpy配列に変換
    local_preds_np = local_preds.numpy()
    local_labels_np = local_labels.numpy()
    local_distances_np = local_distances.numpy()
    local_is_moving_np = local_is_moving.numpy()
    
    # ローカルhistogramを計算（グローバルメトリクス用）
    mask = (local_labels_np >= 0) & (local_labels_np < sem_num)
    if mask.sum() > 0:
        local_histogram = np.bincount(
            sem_num * local_labels_np[mask] + local_preds_np[mask], 
            minlength=sem_num ** 2
        ).reshape(sem_num, sem_num)
    else:
        local_histogram = np.zeros((sem_num, sem_num), dtype=np.int64)
    
    # 距離別・移動/静止別のローカルhistogramを計算
    local_distance_histograms = None
    local_moving_static_histograms = None
    
    if hasattr(args, 'evaluation') and args.evaluation.distance_evaluation:
        local_distance_histograms = compute_distance_histograms(
            local_preds_np, local_labels_np, local_distances_np,
            args.evaluation.distance_bins, sem_num
        )
    
    if hasattr(args, 'evaluation') and args.evaluation.moving_static_evaluation:
        local_moving_static_histograms = compute_moving_static_histograms(
            local_preds_np, local_labels_np, local_is_moving_np, sem_num
        )
    
    # DDP時はhistogramを集約（全点データではなくhistogramのみ）
    if use_ddp:
        # グローバルhistogramを集約
        histogram_tensor = torch.from_numpy(local_histogram).to(device)
        dist.all_reduce(histogram_tensor, op=dist.ReduceOp.SUM)
        histogram = histogram_tensor.cpu().numpy()
        
        # 距離別histogramを集約
        if local_distance_histograms is not None:
            distance_histograms = {}
            for key, local_hist in local_distance_histograms.items():
                hist_tensor = torch.from_numpy(local_hist.astype(np.int64)).to(device)
                dist.all_reduce(hist_tensor, op=dist.ReduceOp.SUM)
                distance_histograms[key] = hist_tensor.cpu().numpy()
        else:
            distance_histograms = None
        
        # 移動/静止別histogramを集約
        if local_moving_static_histograms is not None:
            moving_static_histograms = {}
            for key, local_hist in local_moving_static_histograms.items():
                hist_tensor = torch.from_numpy(local_hist.astype(np.int64)).to(device)
                dist.all_reduce(hist_tensor, op=dist.ReduceOp.SUM)
                moving_static_histograms[key] = hist_tensor.cpu().numpy()
        else:
            moving_static_histograms = None
    else:
        histogram = local_histogram
        distance_histograms = local_distance_histograms
        moving_static_histograms = local_moving_static_histograms
    
    # ハンガリアンマッチング（集約後のグローバルhistogramで決定）
    matching = assignment_function(histogram.max() - histogram)
    o_Acc = histogram[matching[:, 0], matching[:, 1]].sum() / histogram.sum() * 100.
    m_Acc = np.mean(histogram[matching[:, 0], matching[:, 1]] / histogram.sum(1)) * 100
    hist_new = np.zeros((sem_num, sem_num))
    for idx in range(sem_num):
        hist_new[:, idx] = histogram[:, matching[idx, 1]]

    # 最終評価指標を計算
    tp = np.diag(hist_new)
    fp = np.sum(hist_new, 0) - tp
    fn = np.sum(hist_new, 1) - tp
    IoUs = (100 * tp) / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    IoU_list = []
    class_name_IoU_list = ["IoU_unlabeled", "IoU_car", "IoU_bicycle", "IoU_motorcycle", "IoU_truck", "IoU_other-vehicle", "IoU_person", "IoU_bicyclist", "IoU_motorcyclist", "IoU_road", "IoU_parking", "IoU_sidewalk", "IoU_other-ground", "IoU_building", "IoU_fence", "IoU_vegetation", "IoU_trunck", "IoU_terrian", "IoU_pole", "IoU_traffic-sign"]
    s = '| mIoU {:5.2f} | '.format(m_IoU)
    for IoU in IoUs:
        s += '{:5.2f} '.format(IoU)
        IoU_list.append(IoU)
    IoU_dict = dict(zip(class_name_IoU_list, IoU_list))
    
    # 距離別メトリクス計算（集約後のhistogramとマッチングを使用）
    distance_metrics = {}
    if distance_histograms is not None:
        distance_metrics = compute_distance_metrics_from_histograms(
            distance_histograms, sem_num, matching
        )
    
    # 移動/静止別メトリクス計算（集約後のhistogramとマッチングを使用）
    moving_static_metrics = {}
    if moving_static_histograms is not None:
        moving_static_metrics = compute_moving_static_metrics_from_histograms(
            moving_static_histograms, sem_num, matching
        )
    
    return o_Acc, m_Acc, m_IoU, s, IoU_dict, distance_metrics, moving_static_metrics


def eval(epoch: int, args: Union[argparse.Namespace, 'TCUSSConfig']) -> Tuple[float, float, float, str, Dict[str, float], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """モデルを評価する関数（スタンドアロン実行用）
    
    Args:
        epoch: 評価するエポック
        args: コマンドライン引数またはTCUSSConfig
    
    Returns:
        o_Acc: 全体の精度
        m_Acc: 平均精度
        m_IoU: 平均IoU
        s: IoU情報の文字列
        IoU_dict: クラスごとのIoU辞書
        distance_metrics: 距離別メトリクス（distance_evaluation有効時のみ、無効時は空辞書）
        moving_static_metrics: 移動/静止別メトリクス（moving_static_evaluation有効時のみ、無効時は空辞書）
    """
    # モデルを読み込み
    model = Res16FPN18(in_channels=args.input_dim, out_channels=args.feats_dim, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
    model.load_state_dict(torch.load(os.path.join(args.save_path, f'model_{epoch}_checkpoint.pth')))
    model.eval()

    # 分類器を読み込み
    cls = torch.nn.Linear(args.feats_dim, args.primitive_num, bias=False).cuda()
    cls.load_state_dict(torch.load(os.path.join(args.save_path, f'cls_{epoch}_checkpoint.pth')))
    cls.eval()

    # プリミティブ中心をクラスタリング
    primitive_centers = cls.weight.data  # [500, 128]
    print('Merging Primitives')
    cluster_pred = get_kmeans_labels(n_clusters=args.semantic_class, pcds=primitive_centers).to('cpu').detach().numpy()
    
    # クラス中心を計算
    centroids = torch.zeros((args.semantic_class, args.feats_dim))
    for cluster_idx in range(args.semantic_class):
        indices = cluster_pred == cluster_idx
        cluster_avg = primitive_centers[indices].mean(0, keepdims=True)
        centroids[cluster_idx] = cluster_avg
    
    centroids = F.normalize(centroids, dim=1)
    classifier = get_fixclassifier(in_channel=args.feats_dim, centroids_num=args.semantic_class, centroids=centroids).cuda()
    classifier.eval()

    # 評価用バッチサイズとワーカー数を取得（grad不要なので大きくできる）
    eval_batch_size = getattr(args, 'eval_batch_size', 32)
    persistent_workers = getattr(args, 'persistent_workers', True)
    prefetch_factor = getattr(args, 'prefetch_factor', 4)
    num_workers = getattr(args, 'eval_workers', args.cluster_workers)
    
    # 検証データセットを作成
    val_dataset = KITTIval(args)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=eval_batch_size, 
        collate_fn=cfl_collate_fn_val(), 
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )

    # 評価を実行
    preds, labels, distances, is_moving = eval_once(args, model, val_loader, classifier, epoch)
    
    # 結果を連結
    all_preds = torch.cat(preds).numpy()
    all_labels = torch.cat(labels).numpy()
    all_distances = torch.cat(distances).numpy()
    all_is_moving = torch.cat(is_moving).numpy()

    # 教師なし評価：予測をGTにマッチング
    # classifierは19クラスなので、予測も0-18の範囲
    sem_num = args.semantic_class
    mask = (all_labels >= 0) & (all_labels < sem_num)
    histogram = np.bincount(sem_num * all_labels[mask] + all_preds[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)
    
    # ハンガリアンマッチング
    matching = assignment_function(histogram.max() - histogram)
    o_Acc = histogram[matching[:, 0], matching[:, 1]].sum() / histogram.sum() * 100.
    m_Acc = np.mean(histogram[matching[:, 0], matching[:, 1]] / histogram.sum(1)) * 100
    hist_new = np.zeros((sem_num, sem_num))
    for idx in range(sem_num):
        hist_new[:, idx] = histogram[:, matching[idx, 1]]

    # 最終評価指標を計算
    tp = np.diag(hist_new)
    fp = np.sum(hist_new, 0) - tp
    fn = np.sum(hist_new, 1) - tp
    IoUs = (100 * tp) / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    IoU_list = []
    class_name_IoU_list = ["IoU_unlabeled", "IoU_car", "IoU_bicycle", "IoU_motorcycle", "IoU_truck", "IoU_other-vehicle", "IoU_person", "IoU_bicyclist", "IoU_motorcyclist", "IoU_road", "IoU_parking", "IoU_sidewalk", "IoU_other-ground", "IoU_building", "IoU_fence", "IoU_vegetation", "IoU_trunck", "IoU_terrian", "IoU_pole", "IoU_traffic-sign"]
    s = '| mIoU {:5.2f} | '.format(m_IoU)
    for IoU in IoUs:
        s += '{:5.2f} '.format(IoU)
        IoU_list.append(IoU)
    IoU_dict = dict(zip(class_name_IoU_list, IoU_list))
    
    # 距離別メトリクス計算（histogram形式で計算してメトリクスを算出）
    distance_metrics = {}
    if hasattr(args, 'evaluation') and args.evaluation.distance_evaluation:
        distance_histograms = compute_distance_histograms(
            all_preds, all_labels, all_distances,
            args.evaluation.distance_bins, sem_num
        )
        distance_metrics = compute_distance_metrics_from_histograms(
            distance_histograms, sem_num, matching
        )
    
    # 移動/静止別メトリクス計算（histogram形式で計算してメトリクスを算出）
    moving_static_metrics = {}
    if hasattr(args, 'evaluation') and args.evaluation.moving_static_evaluation:
        moving_static_histograms = compute_moving_static_histograms(
            all_preds, all_labels, all_is_moving, sem_num
        )
        moving_static_metrics = compute_moving_static_metrics_from_histograms(
            moving_static_histograms, sem_num, matching
        )
    
    return o_Acc, m_Acc, m_IoU, s, IoU_dict, distance_metrics, moving_static_metrics


def extract_epoch_from_filename(filename: str) -> Optional[int]:
    """ファイル名からエポック番号を抽出する関数
    
    Args:
        filename: チェックポイントファイル名
    
    Returns:
        抽出されたエポック番号、または見つからない場合はNone
    """
    match = re.search(r'model_(\d+)_checkpoint\.pth', filename)
    if match:
        return int(match.group(1))
    return None


def find_epochs_backward(start_epoch: int, eval_backward_num: int, available_epochs: List[int]) -> List[int]:
    """指定されたエポックから逆順に存在するエポックを指定数見つける関数
    
    Args:
        start_epoch: 開始エポック
        eval_backward_num: 見つけるエポック数
        available_epochs: 利用可能なエポックのリスト
    
    Returns:
        見つかったエポックのリスト
    """
    found_epochs = []
    current_epoch = start_epoch
    
    while len(found_epochs) < eval_backward_num and current_epoch >= 0:
        if current_epoch in available_epochs:
            found_epochs.append(current_epoch)
        current_epoch -= 1
    
    return found_epochs


def calculate_statistics(results: List[Tuple[int, float, float, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """評価結果の統計を計算する関数
    
    Args:
        results: (epoch, oAcc, mAcc, mIoU) のリスト
    
    Returns:
        means: 平均値の辞書
        stds: 標準偏差の辞書
    """
    if not results:
        return {}, {}
    
    epochs = [r[0] for r in results]
    o_accs = [r[1] for r in results]
    m_accs = [r[2] for r in results]
    m_ious = [r[3] for r in results]
    
    means = {
        'oAcc': float(np.mean(o_accs)),
        'mAcc': float(np.mean(m_accs)),
        'mIoU': float(np.mean(m_ious))
    }
    
    stds = {
        'oAcc': float(np.std(o_accs, ddof=1)) if len(o_accs) > 1 else 0.0,
        'mAcc': float(np.std(m_accs, ddof=1)) if len(m_accs) > 1 else 0.0,
        'mIoU': float(np.std(m_ious, ddof=1)) if len(m_ious) > 1 else 0.0
    }
    
    return means, stds


if __name__ == '__main__':
    # 引数を解析
    args = parse_args()
    
    # シードを設定
    set_seed(args.seed)
    
    # 検証用データパスを取得
    val_paths = []
    val_datas = []
    seq_list = np.sort(os.listdir(args.data_path))
    for seq_id in seq_list:
        seq_path = os.path.join(args.data_path, seq_id)
        if seq_id in ['08']:
            for f in np.sort(os.listdir(seq_path)):
                val_path = os.path.join(seq_path, f)
                val_paths.append(val_path)
                val_datas.append(read_ply(val_path))

    # チェックポイントファイルのリストを取得
    checkpoint_files = [f for f in os.listdir(args.save_path) if f.endswith('_checkpoint.pth') and f.startswith('model_')]
    
    # ファイル名からエポック番号を抽出
    epoch_numbers = [extract_epoch_from_filename(f) for f in checkpoint_files]
    
    # Noneを除去して、エポック番号でソート
    epoch_numbers = sorted([e for e in epoch_numbers if e is not None])
    
    # 特定のエポックが指定されている場合
    if args.eval_epoch is not None:
        if args.eval_backward_num == 1:
            # 単一エポック評価（従来の動作）
            if args.eval_epoch in epoch_numbers:
                print(f"指定されたエポック {args.eval_epoch} を評価します")
                o_Acc, m_Acc, m_IoU, s, IoU_dict, dist_metrics, ms_metrics = eval(args.eval_epoch, args)
                print('Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} mIoU {:.2f}'.format(args.eval_epoch, o_Acc, m_Acc, m_IoU))
                print(s)
            else:
                print(f"エラー: エポック {args.eval_epoch} のチェックポイントが見つかりません")
        else:
            # 複数エポック逆順評価
            target_epochs = find_epochs_backward(args.eval_epoch, args.eval_backward_num, epoch_numbers)
            
            if not target_epochs:
                print(f"エラー: エポック {args.eval_epoch} から逆順に評価可能なチェックポイントが見つかりません")
            elif len(target_epochs) < args.eval_backward_num:
                print(f"警告: 要求された {args.eval_backward_num} エポック中、{len(target_epochs)} エポックのみ見つかりました")
                print(f"評価対象エポック: {target_epochs}")
            else:
                print(f"エポック {args.eval_epoch} から逆順に {args.eval_backward_num} エポックを評価します")
                print(f"評価対象エポック: {target_epochs}")
            
            # 複数エポックの評価を実行
            multi_results = []
            for epoch in target_epochs:
                print(f"\n--- エポック {epoch} の評価開始 ---")
                o_Acc, m_Acc, m_IoU, s, IoU_dict, dist_metrics, ms_metrics = eval(epoch, args)
                multi_results.append((epoch, o_Acc, m_Acc, m_IoU))
                print('Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} mIoU {:.2f}'.format(epoch, o_Acc, m_Acc, m_IoU))
                print(s)
            
            # 統計計算と表示
            if len(multi_results) > 0:
                means, stds = calculate_statistics(multi_results)
                
                print(f"\n========== {len(multi_results)} エポック評価統計 ==========")
                print(f"評価したエポック: {[r[0] for r in multi_results]}")
                print(f"oAcc - 平均: {means['oAcc']:.2f}, 標準偏差: {stds['oAcc']:.2f}")
                print(f"mAcc - 平均: {means['mAcc']:.2f}, 標準偏差: {stds['mAcc']:.2f}")
                print(f"mIoU - 平均: {means['mIoU']:.2f}, 標準偏差: {stds['mIoU']:.2f}")
                
                # 個別結果の表示
                print("\n個別結果:")
                for epoch, o_Acc, m_Acc, m_IoU in multi_results:
                    print(f"Epoch {epoch}: oAcc {o_Acc:.2f}, mAcc {m_Acc:.2f}, mIoU {m_IoU:.2f}")
    else:
        # 通常モードでは全てのチェックポイントを評価
        print(f"合計 {len(epoch_numbers)} エポックのチェックポイントを評価します")
        
        best_miou = 0
        best_epoch = 0
        results = []
        
        for epoch in epoch_numbers:
            o_Acc, m_Acc, m_IoU, s, IoU_dict, dist_metrics, ms_metrics = eval(epoch, args)
            results.append((epoch, o_Acc, m_Acc, m_IoU))
            print('Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} mIoU {:.2f}'.format(epoch, o_Acc, m_Acc, m_IoU))
            print(s)
            
            if m_IoU > best_miou:
                best_miou = m_IoU
                best_epoch = epoch
        
        # 結果の概要を表示
        print("\n========== 評価結果サマリー ==========")
        print(f"評価したエポック数: {len(epoch_numbers)}")
        print(f"最高 mIoU: {best_miou:.2f} (エポック {best_epoch})")
        
        # 上位5つの結果を表示
        if len(results) > 1:
            sorted_results = sorted(results, key=lambda x: x[3], reverse=True)[:5]
            print("\n上位5つの結果:")
            for epoch, o_Acc, m_Acc, m_IoU in sorted_results:
                print(f"Epoch {epoch}: oAcc {o_Acc:.2f}, mAcc {m_Acc:.2f}, mIoU {m_IoU:.2f}")
