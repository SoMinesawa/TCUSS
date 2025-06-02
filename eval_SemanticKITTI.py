import torch
import torch.nn.functional as F
from datasets.SemanticKITTI import KITTIval, cfl_collate_fn_val
import numpy as np
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
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
from typing import Tuple, Dict, List, Optional, Union, Any

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
    parser.add_argument('--workers', type=int, default=10, help='データローディング用ワーカー数')
    parser.add_argument('--cluster_workers', type=int, default=10, help='クラスタリング用ワーカー数')
    parser.add_argument('--seed', type=int, default=2022, help='乱数シード')
    parser.add_argument('--voxel_size', type=float, default=0.15, help='SparseConvのボクセルサイズ')
    parser.add_argument('--input_dim', type=int, default=3, help='ネットワーク入力次元')
    parser.add_argument('--primitive_num', type=int, default=500, help='学習に使用するプリミティブ数')
    parser.add_argument('--semantic_class', type=int, default=19, help='意味クラス数')
    parser.add_argument('--feats_dim', type=int, default=128, help='出力特徴次元')
    parser.add_argument('--ignore_label', type=int, default=-1, help='無効ラベル')
    parser.add_argument('--eval_epoch', type=int, help='評価する特定のエポック（指定しない場合は全エポック）')
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

def eval_once(args: argparse.Namespace, model: torch.nn.Module, test_loader: DataLoader, classifier: torch.nn.Module, epoch: int) -> Tuple[List, List]:
    """一回の評価を実行する関数
    
    Args:
        args: コマンドライン引数
        model: 評価するモデル
        test_loader: テストデータローダー
        classifier: 分類器
        epoch: 評価するエポック
    
    Returns:
        all_preds: すべての予測結果
        all_label: すべてのラベル
    """
    all_preds, all_label = [], []
    
    for data in tqdm(test_loader, desc=f'Eval Epoch: {epoch}'):
        with torch.no_grad():
            coords, features, inverse_map, labels, index, region = data

            # TensorFieldを作成
            in_field = ME.TensorField(coords[:, 1:] * args.voxel_size, coords, device=0)
            feats = model(in_field)
            feats = F.normalize(feats, dim=1)

            # スコア計算と予測
            scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
            preds = torch.argmax(scores, dim=1).cpu()

            preds = preds[inverse_map.long()]
            preds = preds[labels!=args.ignore_label]
            labels = labels[labels!=args.ignore_label]
            all_preds.append(preds)
            all_label.append(labels)

            # メモリ解放
            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))

    return all_preds, all_label


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

    # 検証データセットを作成
    val_dataset = KITTIval(args)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=cfl_collate_fn_val(), num_workers=args.cluster_workers, pin_memory=True)

    # 評価を実行
    preds, labels = eval_once(args, model, val_loader, classifier, epoch)
    
    # 結果を連結
    all_preds = torch.cat(preds).numpy()
    all_labels = torch.cat(labels).numpy()

    # 教師なし評価：予測をGTにマッチング
    sem_num = args.semantic_class
    mask = (all_labels >= 0) & (all_labels < sem_num)
    histogram = np.bincount(sem_num * all_labels[mask] + all_preds[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)
    
    # ハンガリアンマッチング
    m = assignment_function(histogram.max() - histogram)
    o_Acc = histogram[m[:, 0], m[:, 1]].sum() / histogram.sum() * 100.
    m_Acc = np.mean(histogram[m[:, 0], m[:, 1]] / histogram.sum(1)) * 100
    hist_new = np.zeros((sem_num, sem_num))
    for idx in range(sem_num):
        hist_new[:, idx] = histogram[:, m[idx, 1]]

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
    match = re.search(r'model_(\d+)_checkpoint\.pth', filename)
    if match:
        return int(match.group(1))
    return None


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
        if args.eval_epoch in epoch_numbers:
            print(f"指定されたエポック {args.eval_epoch} を評価します")
            o_Acc, m_Acc, m_IoU, s, IoU_dict = eval(args.eval_epoch, args)
            print('Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} mIoU {:.2f}'.format(args.eval_epoch, o_Acc, m_Acc, m_IoU))
            print(s)
        else:
            print(f"エラー: エポック {args.eval_epoch} のチェックポイントが見つかりません")
    else:
        # 通常モードでは全てのチェックポイントを評価
        print(f"合計 {len(epoch_numbers)} エポックのチェックポイントを評価します")
        
        best_miou = 0
        best_epoch = 0
        results = []
        
        for epoch in epoch_numbers:
            o_Acc, m_Acc, m_IoU, s, IoU_dict = eval(epoch, args)
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
