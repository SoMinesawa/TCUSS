import torch
import torch.nn.functional as F
from datasets.SemanticKITTI import KITTItest, cfl_collate_fn_test
import numpy as np
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
from models.fpn import Res16FPN18
from lib.utils import get_fixclassifier, get_kmeans_labels
import argparse
import random
import os
import yaml
from tqdm import tqdm
from typing import Tuple, Dict, List, Optional, Union, Any

# scikit-learnバージョンに応じてlinear_assignmentを選択
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

# SemanticKITTIの設定ファイルを読み込み
data_config = os.path.join('data_prepare', 'semantic-kitti.yaml')
DATA = yaml.safe_load(open(data_config, 'r'))
learning_map_inv = DATA["learning_map_inv"]


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する関数"""
    parser = argparse.ArgumentParser(description='TCUSS - SemanticKITTI Testing')
    parser.add_argument('--data_path', type=str, default='data/users/minesawa/semantickitti/growsp',
                        help='点群データパス')
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
    parser.add_argument('--primitive_num', type=int, default=128, help='学習に使用するプリミティブ数')
    parser.add_argument('--semantic_class', type=int, default=19, help='意味クラス数')
    parser.add_argument('--feats_dim', type=int, default=128, help='出力特徴次元')
    parser.add_argument('--ignore_label', type=int, default=-1, help='無効ラベル')
    parser.add_argument('--debug', action='store_true', help='デバッグモード')
    parser.add_argument('--use_best', action='store_true', help='bestモデル（best_model.pth, best_classifier.pth）を使用')
    parser.add_argument('--epoch', type=int, help='指定したepochのみ評価 (model_<epoch>_checkpoint.pth, cls_<epoch>_checkpoint.pth)')
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
    classifier: torch.nn.Module
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """一回の推論を実行し、(1) Hungarian用の混同行列、(2) raw予測の保存先と最終保存先の対応を返す。
    
    Args:
        args: コマンドライン引数
        model: 評価するモデル
        test_loader: テストデータローダー
        classifier: 分類器
    
    Returns:
        histogram: (semantic_class, semantic_class) の混同行列（GT=行, pred=列）
        raw_and_save_paths: [(raw_pred_path, final_label_path), ...]
    """
    # 結果保存ディレクトリ（最終提出物）
    save_dir = os.path.join(args.save_path, 'pred_result', 'sequences')
    os.makedirs(save_dir, exist_ok=True)

    # raw予測保存ディレクトリ（提出物に含めない）
    raw_dir = os.path.join(args.save_path, 'pred_result_raw', 'sequences')
    os.makedirs(raw_dir, exist_ok=True)

    sem_num = args.semantic_class
    histogram = np.zeros((sem_num, sem_num), dtype=np.int64)
    raw_and_save_paths: List[Tuple[str, str]] = []
    
    for data in tqdm(test_loader):
        with torch.no_grad():
            coords, features, inverse_map, labels, index, file_paths = data

            # TensorFieldを作成
            in_field = ME.TensorField(coords[:, 1:] * args.voxel_size, coords, device=0)
            feats = model(in_field)
            feats = F.normalize(feats, dim=1)

            # スコア計算と予測
            scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
            preds = torch.argmax(scores, dim=1).cpu()
            preds = preds[inverse_map.long()]

            # Hungarian用の混同行列を逐次更新（全点を溜めない）
            # 既存実装: mask = (all_no_ignore_labels>=0)&(all_no_ignore_labels<sem_num) と同等
            preds_np = preds.numpy().astype(np.int64, copy=False)
            labels_np = labels.numpy().astype(np.int64, copy=False)
            valid_mask = (labels_np >= 0) & (labels_np < sem_num)
            if valid_mask.any():
                flat = sem_num * labels_np[valid_mask] + preds_np[valid_mask]
                local_hist = np.bincount(flat, minlength=sem_num ** 2).reshape(sem_num, sem_num)
                histogram += local_hist

            # 相対パスを取得
            relative_path = os.path.relpath(file_paths[0], args.data_path)
            sequence, filename = os.path.split(relative_path)
            base = os.path.splitext(filename)[0]
            filename_label = base + '.label'
            filename_raw = base + '.bin'
            
            # 保存先のディレクトリを作成
            sequence_dir = os.path.join(save_dir, sequence, 'predictions')
            os.makedirs(sequence_dir, exist_ok=True)

            raw_sequence_dir = os.path.join(raw_dir, sequence, 'predictions')
            os.makedirs(raw_sequence_dir, exist_ok=True)
            
            # 保存パスを設定
            save_path = os.path.join(sequence_dir, filename_label)
            raw_path = os.path.join(raw_sequence_dir, filename_raw)

            # raw予測（cluster id: 0..sem_num-1）をディスクへ退避（uint8で十分）
            preds_np.astype(np.uint8, copy=False).tofile(raw_path)
            raw_and_save_paths.append((raw_path, save_path))

            # メモリ解放
            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))
    
    return histogram, raw_and_save_paths


def eval(epoch: int, args: argparse.Namespace) -> Tuple[float, float, float, str, Dict[str, float]]:
    """モデルを評価する関数
    
    Args:
        epoch: 評価するエポック
        args: コマンドライン引数
    
    Returns:
        o_Acc: 全体の精度
        m_Acc: 平均精度
        m_IoU: 平均IoU
        s: IoU情報の文字列
        IoU_dict: クラスごとのIoU辞書
    """
    # モデル/分類器のチェックポイント存在確認（フォールバックせずエラーで終了）
    model_ckpt_path = os.path.join(args.save_path, f'model_{epoch}_checkpoint.pth')
    cls_ckpt_path = os.path.join(args.save_path, f'cls_{epoch}_checkpoint.pth')
    if not os.path.exists(model_ckpt_path):
        raise FileNotFoundError(f"モデルチェックポイントが見つかりません: {model_ckpt_path}")
    if not os.path.exists(cls_ckpt_path):
        raise FileNotFoundError(f"分類器チェックポイントが見つかりません: {cls_ckpt_path}")

    # モデルを読み込み
    model = Res16FPN18(in_channels=args.input_dim, out_channels=args.feats_dim, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
    model.load_state_dict(torch.load(model_ckpt_path))
    model.eval()

    # 分類器を読み込み
    cls = torch.nn.Linear(args.feats_dim, args.primitive_num, bias=False).cuda()
    cls.load_state_dict(torch.load(cls_ckpt_path))
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

    # テストデータセットを作成
    test_dataset = KITTItest(args)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=cfl_collate_fn_test(), num_workers=args.cluster_workers, pin_memory=True)

    # 推論を実行（全点を溜めずに混同行列を作る + raw予測を退避）
    histogram, raw_and_save_paths = eval_once(args, model, test_loader, classifier)

    sem_num = args.semantic_class
    if histogram.sum() == 0:
        raise RuntimeError(
            "Hungarianマッチング用の有効GT点が0です。"
            "（labelsが全てignore/範囲外の可能性）"
        )
    
    # ハンガリアンマッチング
    m = assignment_function(histogram.max() - histogram)
    o_Acc = histogram[m[:, 0], m[:, 1]].sum() / histogram.sum() * 100.
    m_Acc = np.mean(histogram[m[:, 0], m[:, 1]] / histogram.sum(1)) * 100
    hist_new = np.zeros((sem_num, sem_num))
    for idx in range(sem_num):
        hist_new[:, idx] = histogram[:, m[idx, 1]]

    # 予測結果を保存
    # cluster id -> semantic class への写像（Hungarian結果は置換）
    cluster_to_class = np.empty((sem_num,), dtype=np.int64)
    for row in m:
        cluster_to_class[int(row[1])] = int(row[0])

    for raw_path, save_path in raw_and_save_paths:
        raw_pred = np.fromfile(raw_path, dtype=np.uint8).astype(np.int64, copy=False)
        mapped = cluster_to_class[raw_pred]  # 0..sem_num-1
        mapped = mapped + 1
        mapped_inv = np.vectorize(learning_map_inv.get)(mapped).astype(np.uint32)
        mapped_inv.tofile(save_path)
        os.remove(raw_path)
    
    # 最終評価指標を計算
    tp = np.diag(hist_new)
    fp = np.sum(hist_new, 0) - tp
    fn = np.sum(hist_new, 1) - tp
    IoUs = (100 * tp) / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    IoU_list = []
    class_name_IoU_list = ["IoU_unlabeled", "IoU_car", "IoU_bicycle", "IoU_motorcycle", "IoU_truck", "IoU_other-vehicle", "IoU_person", "IoU_bicyclist", "IoU_motorcyclist", "IoU_road", "IoU_parking", "IoU_sidewalk", "IoU_other-ground", "IoU_building", "IoU_fence", "IoU_vegetation", "IoU_trunck", "IoU_terrian", "IoU_pole", "IoU_traffic-sign"]
    s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
    for IoU in IoUs:
        s += '{:5.2f} '.format(IoU)
        IoU_list.append(IoU)
    IoU_dict = dict(zip(class_name_IoU_list, IoU_list))
    return o_Acc, m_Acc, m_IoU, s, IoU_dict


def eval_best(args: argparse.Namespace) -> Tuple[float, float, float, str, Dict[str, float]]:
    """bestモデルを評価する関数
    
    Args:
        args: コマンドライン引数
    
    Returns:
        o_Acc: 全体の精度
        m_Acc: 平均精度
        m_IoU: 平均IoU
        s: IoU情報の文字列
        IoU_dict: クラスごとのIoU辞書
    """
    # bestモデルのパスを確認
    best_model_path = os.path.join(args.save_path, 'best_model.pth')
    best_classifier_path = os.path.join(args.save_path, 'best_classifier.pth')
    
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"bestモデルファイルが見つかりません: {best_model_path}")
    if not os.path.exists(best_classifier_path):
        raise FileNotFoundError(f"best分類器ファイルが見つかりません: {best_classifier_path}")
    
    print(f"Loading best model from: {best_model_path}")
    print(f"Loading best classifier from: {best_classifier_path}")
    
    # モデルを読み込み
    model = Res16FPN18(in_channels=args.input_dim, out_channels=args.feats_dim, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # 分類器を読み込み
    cls = torch.nn.Linear(args.feats_dim, args.primitive_num, bias=False).cuda()
    cls.load_state_dict(torch.load(best_classifier_path))
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

    # テストデータセットを作成
    test_dataset = KITTItest(args)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=cfl_collate_fn_test(), num_workers=args.cluster_workers, pin_memory=True)

    # 推論を実行（全点を溜めずに混同行列を作る + raw予測を退避）
    histogram, raw_and_save_paths = eval_once(args, model, test_loader, classifier)

    sem_num = args.semantic_class
    if histogram.sum() == 0:
        raise RuntimeError(
            "Hungarianマッチング用の有効GT点が0です。"
            "（labelsが全てignore/範囲外の可能性）"
        )
    
    # ハンガリアンマッチング
    m = assignment_function(histogram.max() - histogram)
    o_Acc = histogram[m[:, 0], m[:, 1]].sum() / histogram.sum() * 100.
    m_Acc = np.mean(histogram[m[:, 0], m[:, 1]] / histogram.sum(1)) * 100
    hist_new = np.zeros((sem_num, sem_num))
    for idx in range(sem_num):
        hist_new[:, idx] = histogram[:, m[idx, 1]]

    # 予測結果を保存
    cluster_to_class = np.empty((sem_num,), dtype=np.int64)
    for row in m:
        cluster_to_class[int(row[1])] = int(row[0])

    for raw_path, save_path in raw_and_save_paths:
        raw_pred = np.fromfile(raw_path, dtype=np.uint8).astype(np.int64, copy=False)
        mapped = cluster_to_class[raw_pred]  # 0..sem_num-1
        mapped = mapped + 1
        mapped_inv = np.vectorize(learning_map_inv.get)(mapped).astype(np.uint32)
        mapped_inv.tofile(save_path)
        os.remove(raw_path)
    
    # 最終評価指標を計算
    tp = np.diag(hist_new)
    fp = np.sum(hist_new, 0) - tp
    fn = np.sum(hist_new, 1) - tp
    IoUs = (100 * tp) / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    IoU_list = []
    class_name_IoU_list = ["IoU_unlabeled", "IoU_car", "IoU_bicycle", "IoU_motorcycle", "IoU_truck", "IoU_other-vehicle", "IoU_person", "IoU_bicyclist", "IoU_motorcyclist", "IoU_road", "IoU_parking", "IoU_sidewalk", "IoU_other-ground", "IoU_building", "IoU_fence", "IoU_vegetation", "IoU_trunck", "IoU_terrian", "IoU_pole", "IoU_traffic-sign"]
    s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
    for IoU in IoUs:
        s += '{:5.2f} '.format(IoU)
        IoU_list.append(IoU)
    IoU_dict = dict(zip(class_name_IoU_list, IoU_list))
    return o_Acc, m_Acc, m_IoU, s, IoU_dict


def extract_epoch_from_filename(filename: str) -> Optional[int]:
    """ファイル名からエポック番号を抽出する関数
    
    Args:
        filename: チェックポイントファイル名
    
    Returns:
        抽出されたエポック番号、または見つからない場合はNone
    """
    import re
    match = re.search(r'model_(\d+)_checkpoint\.pth', filename)
    if match:
        return int(match.group(1))
    return None


if __name__ == '__main__':
    # 引数を解析
    args = parse_args()
    
    # シードを設定
    set_seed(args.seed)

    # オプションの整合性チェック（フォールバックせずエラーで終了）
    if args.epoch is not None and args.epoch < 0:
        raise ValueError(f"--epoch には0以上の整数を指定してください: {args.epoch}")
    if args.epoch is not None and (args.use_best or args.debug):
        raise ValueError("--epoch と --use_best/--debug は同時に指定できません")

    # epoch指定がある場合はそのepochのみ評価
    if args.epoch is not None:
        o_Acc, m_Acc, m_IoU, s, IoU_dict = eval(args.epoch, args)
        print('Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} mIoU {:.2f}'.format(args.epoch, o_Acc, m_Acc, m_IoU))
        print(s)
    # bestモデルを使用する場合
    elif args.use_best:
        o_Acc, m_Acc, m_IoU, s, IoU_dict = eval_best(args)
        print('Best Model: oAcc {:.2f}  mAcc {:.2f} mIoU {:.2f}'.format(o_Acc, m_Acc, m_IoU))
        print(s)
    # デバッグモードの場合は最新のチェックポイントのみ評価
    elif args.debug:
        checkpoint_files = [f for f in os.listdir(args.save_path) if f.endswith('_checkpoint.pth') and f.startswith('model_')]
        epoch_numbers = [extract_epoch_from_filename(f) for f in checkpoint_files]
        epoch_numbers = sorted([e for e in epoch_numbers if e is not None], reverse=True)
        
        if epoch_numbers:
            latest_epoch = epoch_numbers[0]
            o_Acc, m_Acc, m_IoU, s, IoU_dict = eval(latest_epoch, args)
            print('Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} mIoU {:.2f}'.format(latest_epoch, o_Acc, m_Acc, m_IoU))
            print(s)
        else:
            print("No checkpoint files found.")
    else:
        # 通常モードでは最新のチェックポイントを評価
        checkpoint_files = [f for f in os.listdir(args.save_path) if f.endswith('_checkpoint.pth') and f.startswith('model_')]
        epoch_numbers = [extract_epoch_from_filename(f) for f in checkpoint_files]
        epoch_numbers = sorted([e for e in epoch_numbers if e is not None])
        
        # 最新のnum_latest_epochs個のepochのみを評価対象とする
        num_latest_epochs = 5  # 評価する最新epoch数
        epoch_numbers = epoch_numbers[-num_latest_epochs:] if len(epoch_numbers) > num_latest_epochs else epoch_numbers
        
        print(f"Evaluating latest {len(epoch_numbers)} epochs: {epoch_numbers}")
        
        best_miou = 0
        best_epoch = 0
        
        for epoch in epoch_numbers:
            o_Acc, m_Acc, m_IoU, s, IoU_dict = eval(epoch, args)
            print('Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} mIoU {:.2f}'.format(epoch, o_Acc, m_Acc, m_IoU))
            print(s)
            
            if m_IoU > best_miou:
                best_miou = m_IoU
                best_epoch = epoch
        
        print(f"Best mIoU: {best_miou:.2f} at epoch {best_epoch}")
