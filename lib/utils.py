import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
from typing import Optional, Tuple
from tqdm import tqdm
import wandb
import time

from lib.kmeans_torch import KMeans as KMeans_gpu
from sklearn.cluster import KMeans as KMeans_sklearn


def get_kittisp_feature(args, loader, model, current_growsp, epoch):
    """点群特徴を計算する関数（バッチ化版）
    
    モデル推論をバッチで実行し、後処理はシーンごとに実行。
    
    Note: KITTItrainではMixupでcoordsだけが連結されるため、
    coordsとfeats/labels/normals/regionのサイズが異なる。
    indsはfeatsの範囲でインデックスされる。
    """
    print('computing point feats ....')
    point_feats_list = []
    point_labels_list = []
    all_sp_index = []
    context = []
    model.eval()
    
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(loader), total=len(loader), desc="get_kittisp_feature"):
            # collate_fnから10要素を受け取る（feats_sizesが追加された）
            coords, features, normals, labels, inverse_map, pseudo_labels, inds, region, index, feats_sizes = data
            
            # バッチサイズを取得（coordsの最初の列がbatch_id）
            batch_ids = coords[:, 0].int()
            actual_batch_size = len(feats_sizes)
            
            # === バッチでモデル推論 ===
            in_field = ME.TensorField(coords[:, 1:] * args.voxel_size, coords, device=0)
            all_feats = model(in_field)
            
            # featsの累積サイズを計算（collate_fnから受け取ったfeats_sizesを使用）
            feats_cumsum = [0] + list(np.cumsum(feats_sizes))
            
            # === シーンごとに後処理 ===
            for local_batch_id in range(actual_batch_size):
                # このシーンのfeats範囲
                feats_start = feats_cumsum[local_batch_id]
                feats_end = feats_cumsum[local_batch_id + 1]
                
                # このシーンのindsを特定
                inds_mask = (inds >= feats_start) & (inds < feats_end)
                scene_inds = (inds[inds_mask] - feats_start).long()
                
                # このシーンのデータを取得（feats/labels/normals/regionは同じサイズ）
                scene_all_features = features[feats_start:feats_end]
                scene_all_normals = normals[feats_start:feats_end]
                scene_all_labels = labels[feats_start:feats_end]
                scene_region = region[feats_start:feats_end].squeeze()
                
                # このシーンのcoordsとモデル出力
                scene_coords_mask = (batch_ids == local_batch_id)
                scene_all_feats = all_feats[scene_coords_mask]
                
                # indsでサンプリングされた点のデータ
                # scene_indsはfeats内の相対インデックス
                scene_feats = scene_all_feats[scene_inds].to("cuda:0")
                scene_features = scene_all_features[scene_inds].to("cuda:0")
                scene_normals = scene_all_normals[scene_inds].to("cuda:0")
                scene_labels = scene_all_labels[scene_inds]
                scene_region_sampled = scene_region[scene_inds]
                
                scene_name = loader.dataset.name[index[local_batch_id]]
                gt = scene_labels.clone()
                raw_region = scene_region_sampled.clone()
                
                valid_mask = scene_region_sampled != -1
                scene_features = scene_features[valid_mask]
                scene_normals = scene_normals[valid_mask]
                scene_feats = scene_feats[valid_mask]
                scene_labels = scene_labels[valid_mask]
                scene_region_valid = scene_region_sampled[valid_mask].long().to("cuda:0")
                
                pc_remission = scene_features
                region_num = len(torch.unique(scene_region_valid))
                region_corr = torch.zeros(scene_region_valid.size(0), region_num).to("cuda:0")
                region_corr.scatter_(1, scene_region_valid.view(-1, 1), 1)
                per_region_num = region_corr.sum(0, keepdims=True).t()

                region_feats = F.linear(region_corr.t(), scene_feats.t()) / per_region_num
                if current_growsp is not None:
                    region_feats = F.normalize(region_feats, dim=-1)
                    if region_feats.size(0) < current_growsp:
                        n_segments = region_feats.size(0)
                    else:
                        n_segments = current_growsp
                    sp_idx = get_kmeans_labels(n_clusters=n_segments, pcds=region_feats).long()
                else:
                    feats = region_feats
                    sp_idx = torch.arange(region_feats.size(0)).to("cuda:0")

                neural_region = sp_idx[scene_region_valid]

                neural_region_num = len(torch.unique(neural_region))
                neural_region_corr = F.one_hot(neural_region, num_classes=neural_region_num).float().to("cuda:0")
                per_neural_region_num = neural_region_corr.sum(0, keepdims=True).t()
                final_remission = F.linear(neural_region_corr.t(), pc_remission.t()) / per_neural_region_num

                if current_growsp is not None:
                    feats = F.linear(neural_region_corr.t(), scene_feats.t()) / per_neural_region_num
                    feats = F.normalize(feats, dim=-1)

                # pfh計算（ベクトル化版で高速化）
                pfh = compute_hist_vectorized(scene_normals, neural_region, neural_region_num)

                feats = F.normalize(feats, dim=-1)
                feats = torch.cat((feats, args.c_rgb * final_remission, args.c_shape * pfh), dim=-1)
                feats = F.normalize(feats, dim=-1)

                # データをCPUに移動
                feats_cpu = feats.cpu()
                labels_cpu = scene_labels.cpu()
                neural_region_cpu = neural_region.cpu().detach().numpy().copy()
                sp_idx_cpu = sp_idx.cpu().detach().numpy().copy()

                # リストにデータを追加
                point_feats_list.append(feats_cpu)
                point_labels_list.append(labels_cpu)
                all_sp_index.append(neural_region_cpu)
                context.append((scene_name, gt, raw_region, sp_idx_cpu))

    return point_feats_list, point_labels_list, all_sp_index, context


def compute_hist_vectorized(normals: torch.Tensor, neural_region: torch.Tensor, neural_region_num: int, bins: int = 10) -> torch.Tensor:
    """法線ヒストグラムをベクトル化して計算（高速化版）
    
    従来のPythonループをGPUで効率的に計算。
    
    Args:
        normals: 法線ベクトル [N, 3]
        neural_region: 各点のSPラベル [N]
        neural_region_num: SP数
        bins: ヒストグラムのビン数
    
    Returns:
        pfh: SP毎の法線ヒストグラム [neural_region_num, bins]
    """
    device = normals.device
    normals = F.normalize(normals, dim=-1)
    
    pfh = torch.zeros(neural_region_num, bins, device=device)
    
    # 各SPのポイント数をカウント
    counts = torch.bincount(neural_region, minlength=neural_region_num)
    
    for p in range(neural_region_num):
        if counts[p] > 1:  # 2点以上必要
            mask = neural_region == p
            sp_normals = normals[mask]
            # コサイン類似度行列の計算
            relation = torch.mm(sp_normals, sp_normals.t())
            # 上三角部分のみ取得（対角含む）
            relation = torch.triu(relation, diagonal=0)
            # ヒストグラム計算
            hist = torch.histc(relation, bins, -1, 1)
            hist_sum = hist.sum()
            if hist_sum > 0:
                hist = hist / hist_sum
            pfh[p] = hist
        elif counts[p] == 1:
            # 1点のみの場合
            pfh[p, bins-1] = 1.0  # 自己相関は1
    
    return pfh



def get_pseudo_kitti(args, context, cluster_pred, all_sub_cluster=None):
    print('computing pseduo labels...')
    all_gt = []
    all_pseudo = []
    all_pseudo_gt = []
    pc_no = 0
    region_num = 0

    for i in range(len(context)):
        # context[i] = (scene_name, labels, region) or (scene_name, labels, region, sp_idx)
        ctx = context[i]
        scene_name, labels, region = ctx[0], ctx[1], ctx[2]

        sub_cluster_pred = all_sub_cluster[pc_no]+ region_num
        valid_mask = region != -1

        labels_tmp = labels[valid_mask]
        pseudo_gt = -torch.ones_like(labels)
        pseudo_gt_tmp = pseudo_gt[valid_mask]

        pseudo = -np.ones_like(labels.numpy()).astype(np.int32)
        pseudo[valid_mask] = cluster_pred[sub_cluster_pred]

        for p in np.unique(sub_cluster_pred):
            if p != -1:
                mask = p == sub_cluster_pred
                sub_cluster_gt = torch.mode(labels_tmp[mask]).values
                pseudo_gt_tmp[mask] = sub_cluster_gt
        pseudo_gt[valid_mask] = pseudo_gt_tmp
        #
        pc_no += 1
        new_region = np.unique(sub_cluster_pred)
        region_num += len(new_region[new_region != -1])

        pseudo_label_folder = args.pseudo_label_path + '/' + scene_name[0:3]
        if not os.path.exists(pseudo_label_folder):
            os.makedirs(pseudo_label_folder)

        pseudo_label_file = args.pseudo_label_path + '/' + scene_name + '.npy'
        np.save(pseudo_label_file, pseudo)

        all_gt.append(labels)
        all_pseudo.append(pseudo)
        all_pseudo_gt.append(pseudo_gt)

    all_gt = np.concatenate(all_gt)
    all_pseudo = np.concatenate(all_pseudo)
    all_pseudo_gt = np.concatenate(all_pseudo_gt)

    return all_pseudo, all_gt, all_pseudo_gt


def get_fixclassifier(in_channel, centroids_num, centroids):
    classifier = nn.Linear(in_features=in_channel, out_features=centroids_num, bias=False)
    centroids = F.normalize(centroids, dim=1)
    classifier.weight.data = centroids
    for para in classifier.parameters():
        para.requires_grad = False
    return classifier


def compute_hist(normal, bins=10, min=-1, max=1):
    ## normal : [N, 3]
    normal = F.normalize(normal)
    relation = torch.mm(normal, normal.t())
    relation = torch.triu(relation, diagonal=0) # top-half matrix
    hist = torch.histc(relation, bins, min, max)
    # hist = torch.histogram(relation, bins, range=(-1, 1))
    hist /= hist.sum()

    return hist
    

def get_kmeans_labels(n_clusters, pcds, max_iter=300):
    """
    KMeansを用いてクラスタリングを行い、各点に対するクラスタラベルを返す。
    エラー
    Args:
        n_clusters (int): クラスタ数
        pcds (np.ndarray | torch.tensor): 点群データ (N, D)?
        max_iter (int): KMeansの最大イテレーション数
    Returns:
        labels (torch.tensor): 各点に対するクラスタラベル (N,)
    """
    model = KMeans_gpu(n_clusters=n_clusters, max_iter=max_iter, distance='euclidean').cuda()
    with torch.no_grad():
        if isinstance(pcds, np.ndarray):
            pcds = torch.from_numpy(pcds)
        pcds = pcds.cuda().float()
        unsqueezed = pcds.unsqueeze(0)
        try:
            centroids, labels = model(unsqueezed)
        # centroids = centroids.squeeze(0)
        # distances = torch.cdist(pcds, centroids)
        # labels = torch.argmin(distances, dim=1)
        except ValueError:
            # print("kmeans-gpu ValueError so use sklearn")
            pcds = pcds.cpu().numpy()
            # np.save("kmeans_error.npy", pcds)
            try:
                labels = torch.from_numpy(KMeans_sklearn(n_clusters=n_clusters, n_init=5, random_state=0).fit_predict(pcds))
            except:
                labels = torch.from_numpy(KMeans_sklearn(n_clusters=n_clusters, n_init=5, random_state=0, n_jobs=5).fit_predict(pcds))
    lbls = labels.float().squeeze(0).cuda()
    del labels, model, pcds
    torch.cuda.empty_cache()
    return lbls


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '1':
        # テスト用のデータを生成して動作確認を行う
        import matplotlib.pyplot as plt
        from sklearn.datasets import make_blobs
        n_samples = 500
        n_features = 2
        n_clusters = 3

        # サンプルデータを生成
        X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)
        print(X.shape)

        # KMeansラベル取得
        labels = get_kmeans_labels(n_clusters, X).cpu().numpy()

        # 結果を散布図で確認
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        plt.title("KMeans Clustering Result")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.colorbar(label="Cluster Label")
        plt.savefig("a.png")
