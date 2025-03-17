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
###

data_config = os.path.join('data_prepare', 'semantic-kitti.yaml')
DATA = yaml.safe_load(open(data_config, 'r'))
learning_map_inv = DATA["learning_map_inv"]

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Unsuper_3D_Seg')
    parser.add_argument('--data_path', type=str, default='data/users/minesawa/semantickitti/growsp',
                        help='pont cloud data path')
    parser.add_argument('--sp_path', type=str, default='data/users/minesawa/semantickitti/growsp_sp',
                        help='initial sp path')
    parser.add_argument('--save_path', type=str, default='data/users/minesawa/semantickitti/growsp_model',
                        help='model savepath')
    ###
    parser.add_argument('--bn_momentum', type=float, default=0.02, help='batchnorm parameters')
    parser.add_argument('--conv1_kernel_size', type=int, default=5, help='kernel size of 1st conv layers')
    ####
    parser.add_argument('--workers', type=int, default=10, help='how many workers for loading data')
    parser.add_argument('--cluster_workers', type=int, default=10, help='how many workers for loading data in clustering')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--voxel_size', type=float, default=0.15, help='voxel size in SparseConv')
    parser.add_argument('--input_dim', type=int, default=3, help='network input dimension')### 6 for XYZGB
    parser.add_argument('--primitive_num', type=int, default=500, help='how many primitives used in training')
    parser.add_argument('--semantic_class', type=int, default=19, help='ground truth semantic class')
    parser.add_argument('--feats_dim', type=int, default=128, help='output feature dimension')
    parser.add_argument('--ignore_label', type=int, default=-1, help='invalid label')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    return parser.parse_args()


def set_seed(seed):
    """
    Unfortunately, backward() of [interpolate] functional seems to be never deterministic.

    Below are related threads:
    https://github.com/pytorch/pytorch/issues/7068
    https://discuss.pytorch.org/t/non-deterministic-behavior-of-pytorch-upsample-interpolate/42842?u=sbelharbi
    """
    # Use random seed.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def eval_once(args, model, test_loader, classifier):
    
    save_dir = os.path.join(args.save_path, 'pred_result', 'sequences')
    os.makedirs(save_dir, exist_ok=True)
    all_preds, all_preds_no_ignore, all_label, all_label_no_ignore, all_save_path = [], [], [], [], []
    for data in tqdm(test_loader):
        with torch.no_grad():
            coords, features, inverse_map, labels, index, region, file_paths = data

            in_field = ME.TensorField(coords[:, 1:] * args.voxel_size, coords, device=0)
            feats = model(in_field)
            feats = F.normalize(feats, dim=1)

            scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
            preds = torch.argmax(scores, dim=1).cpu()
            preds = preds[inverse_map.long()]
            preds_no_ignore = preds.clone()
            labels_no_ignore = labels.clone()
            if int(os.path.split(os.path.dirname(file_paths[0]))[1]) < 11:
                # preds_no_ignore[labels==args.ignore_label] = args.ignore_label 
                preds = preds_no_ignore[labels!=args.ignore_label]
            labels = labels[labels!=args.ignore_label]
            relative_path = os.path.relpath(file_paths[0], args.data_path)
            sequence, filename = os.path.split(relative_path)
            filename = os.path.splitext(filename)[0] + '.label'
            
            # 保存先のディレクトリを作成
            sequence_dir = os.path.join(save_dir, sequence, 'predictions')
            os.makedirs(sequence_dir, exist_ok=True)
            
            # ラベルを保存
            save_path = os.path.join(sequence_dir, filename)
            all_preds.append(preds), all_preds_no_ignore.append(preds_no_ignore), all_label.append(labels), all_label_no_ignore.append(labels_no_ignore), all_save_path.append(save_path)

            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))
    
    return all_preds, all_preds_no_ignore, all_label, all_label_no_ignore, all_save_path


def eval(epoch, args):

    model = Res16FPN18(in_channels=args.input_dim, out_channels=args.feats_dim, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
    model.load_state_dict(torch.load(os.path.join(args.save_path, 'model_' + str(epoch) + '_checkpoint.pth')))
    model.eval()

    cls = torch.nn.Linear(args.feats_dim, args.primitive_num, bias=False).cuda()
    cls.load_state_dict(torch.load(os.path.join(args.save_path, 'cls_' + str(epoch) + '_checkpoint.pth')))
    cls.eval()

    primitive_centers = cls.weight.data###[500, 128]
    print('Merging Primitives')
    cluster_pred = get_kmeans_labels(n_clusters=args.semantic_class, pcds=primitive_centers).to('cpu').detach().numpy()
    
    '''Compute Class Centers'''
    centroids = torch.zeros((args.semantic_class, args.feats_dim))
    for cluster_idx in range(args.semantic_class):
        indices = cluster_pred ==cluster_idx
        cluster_avg = primitive_centers[indices].mean(0, keepdims=True)
        centroids[cluster_idx] = cluster_avg
    # #
    centroids = F.normalize(centroids, dim=1)
    classifier = get_fixclassifier(in_channel=args.feats_dim, centroids_num=args.semantic_class, centroids=centroids).cuda()
    classifier.eval()

    test_dataset = KITTItest(args)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=cfl_collate_fn_test(), num_workers=args.cluster_workers, pin_memory=True)

    preds, no_ignore_preds, labels, no_ignore_labels, save_paths = eval_once(args, model, test_loader, classifier)
    
    all_preds = torch.cat(preds).numpy()
    all_no_ignore_preds = torch.cat(no_ignore_preds).numpy()
    all_labels = torch.cat(labels).numpy()
    all_no_ignore_labels = torch.cat(no_ignore_labels).numpy()

    '''Unsupervised, Match pred to gt'''
    sem_num = args.semantic_class
    mask = (all_no_ignore_labels >= 0) & (all_no_ignore_labels < sem_num)
    histogram = np.bincount(sem_num * all_no_ignore_labels[mask] + all_no_ignore_preds[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)
    # np.save('histogram.npy', histogram)
    # '''Hungarian Matching'''
    m = assignment_function(histogram.max() - histogram)
    o_Acc = histogram[m[:, 0], m[:, 1]].sum() / histogram.sum()*100.
    m_Acc = np.mean(histogram[m[:, 0], m[:, 1]] / histogram.sum(1))*100
    hist_new = np.zeros((sem_num, sem_num))
    for idx in range(sem_num):
        hist_new[:, idx] = histogram[:, m[idx, 1]]

        
    for no_ignore_pred, save_path in zip(no_ignore_preds, save_paths):
        no_ignore_pred = no_ignore_pred.numpy().astype(np.int64)
        no_ignore_pred_copy = no_ignore_pred.copy()
        for row in m:
            no_ignore_pred_copy[no_ignore_pred == row[1]] = row[0]
        no_ignore_pred_copy = no_ignore_pred_copy + 1
        no_ignore_pred_inv = np.vectorize(learning_map_inv.get)(no_ignore_pred_copy).astype(np.uint32)
        no_ignore_pred_inv.tofile(save_path)
        
    # acc 100%
    # for no_ignore_label, save_path in zip(no_ignore_labels, save_paths):
    #     no_ignore_label = no_ignore_label.numpy().astype(np.uint32)
    #     # label = label + 1
    #     no_ignore_label = np.vectorize(learning_map_inv.get)(no_ignore_label).astype(np.uint32)
        # no_ignore_label.tofile(save_path)

    # '''Final Metrics'''
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


import re

def extract_epoch_from_filename(filename):
    """ファイル名からepoch番号を抽出"""
    match = re.search(r'model_(\d+)_checkpoint\.pth', filename)
    if match:
        return int(match.group(1))
    return None


if __name__ == '__main__':

    args = parse_args()
    seed = args.seed
    set_seed(seed)

    # チェックポイントファイルのリストを取得
    checkpoint_files = [f for f in os.listdir(args.save_path) if f.endswith('.pth')]
    
    # ファイル名からepoch番号を抽出
    epoch_numbers = [extract_epoch_from_filename(f) for f in checkpoint_files]
    
    # Noneを除去して、epoch番号でソート
    epoch_numbers = sorted([e for e in epoch_numbers if e is not None], reverse=True)
    
    # 最も大きい5つのepoch番号を選択
    top_1_epochs = epoch_numbers[:1]

    eval_results = []
    
    # 各epochでevalを実行して結果を保存
    for epoch in top_1_epochs:
        o_Acc, m_Acc, m_IoU, _, _ = eval(epoch, args)
        eval_results.append([o_Acc, m_Acc, m_IoU])
        print(f'Epoch: {epoch}, oAcc: {o_Acc:.2f}, mAcc: {m_Acc:.2f}, mIoU: {m_IoU:.2f}')
    
    # # 結果の平均と標準偏差を計算
    # eval_results = np.array(eval_results)
    # mean_results = np.mean(eval_results, axis=0)
    # std_results = np.std(eval_results, axis=0)

    # print('Mean oAcc: {:.2f}, mAcc: {:.2f}, mIoU: {:.2f}'.format(mean_results[0], mean_results[1], mean_results[2]))
    # print('Std oAcc: {:.2f}, mAcc: {:.2f}, mIoU: {:.2f}'.format(std_results[0], std_results[1], std_results[2]))
