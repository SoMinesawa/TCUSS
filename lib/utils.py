import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm
import wandb
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from torchclustermetrics import silhouette
import time
import random

# from models.fpn import Res16FPNBase
from kmeans_gpu import KMeans

def get_sp_feature(args, loader, model, current_growsp):
    print('computing point feats ....')
    point_feats_list = []
    point_labels_list = []
    all_sp_index = []
    model.eval()
    context = []
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            coords, features, normals, labels, inverse_map, pseudo_labels, inds, region, index = data

            region = region.squeeze()
            scene_name = loader.dataset.name[index[0]]
            gt = labels.clone()
            raw_region = region.clone()

            in_field = ME.TensorField(features, coords, device=0)

            feats = model(in_field)
            # feats = F.normalize(feats, dim=-1)
            feats = feats[inds.long()]

            valid_mask = region!=-1
            '''Compute avg rgb/xyz/norm for each Superpoints to help merging superpoints'''
            features = features[inds.long()].cuda()
            features = features[valid_mask]
            normals = normals[inds.long()].cuda()
            normals = normals[valid_mask]
            feats = feats[valid_mask]
            labels = labels[valid_mask]
            region = region[valid_mask].long()
            ##
            pc_rgb = features[:, 0:3]
            pc_xyz = features[:, 3:] * args.voxel_size
            ##
            region_num = len(torch.unique(region))
            region_corr = torch.zeros(region.size(0), region_num)#?
            region_corr.scatter_(1, region.view(-1, 1), 1)
            region_corr = region_corr.cuda()##[N, M]
            per_region_num = region_corr.sum(0, keepdims=True).t()
            ###
            region_feats = F.linear(region_corr.t(), feats.t())/per_region_num
            if current_growsp is not None:
                region_rgb = F.linear(region_corr.t(), pc_rgb.t())/per_region_num
                region_xyz = F.linear(region_corr.t(), pc_xyz.t())/per_region_num
                region_norm = F.linear(region_corr.t(), normals.t())/per_region_num

                rgb_w, xyz_w, norm_w = args.w_rgb, args.w_xyz, args.w_norm
                region_feats = F.normalize(region_feats, dim=-1)
                region_feats = torch.cat((region_feats, rgb_w*region_rgb, xyz_w*region_xyz, norm_w*region_norm), dim=-1)
                #
                if region_feats.size(0)<current_growsp:
                    n_segments = region_feats.size(0)
                else:
                    n_segments = current_growsp
                sp_idx = get_kmeans_labels(n_clusters=n_segments, pcds=region_feats).long()
            else:
                feats = region_feats
                sp_idx = torch.tensor(range(region_feats.size(0)))

            neural_region = sp_idx[region]
            pfh = []

            neural_region_num = len(torch.unique(neural_region))
            neural_region_corr = torch.zeros(neural_region.size(0), neural_region_num)
            neural_region_corr.scatter_(1, neural_region.view(-1, 1), 1)
            neural_region_corr = neural_region_corr.cuda()
            per_neural_region_num = neural_region_corr.sum(0, keepdims=True).t()
            #
            '''Compute avg rgb/pfh for each Superpoints to help Primitives Learning'''
            final_rgb = F.linear(neural_region_corr.t(), pc_rgb.t())/per_neural_region_num
            #
            if current_growsp is not None:
                feats = F.linear(neural_region_corr.t(), feats.t()) / per_neural_region_num
                feats = F.normalize(feats, dim=-1)

            for p in torch.unique(neural_region):
                if p!=-1:
                    mask = p==neural_region
                    pfh.append(compute_hist(normals[mask].cpu()).unsqueeze(0).cuda())

            pfh = torch.cat(pfh, dim=0)
            feats = F.normalize(feats, dim=-1)
            # #
            feats = torch.cat((feats, args.c_rgb*final_rgb, args.c_shape*pfh), dim=-1)
            feats = F.normalize(feats, dim=-1)

            point_feats_list.append(feats.cpu())
            point_labels_list.append(labels.cpu())

            all_sp_index.append(neural_region)
            context.append((scene_name, gt, raw_region))

            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))
    return point_feats_list, point_labels_list, all_sp_index, context


def get_kittisp_feature(args, loader, model, current_growsp, epoch):
    print('computing point feats ....')
    point_feats_list = []
    point_labels_list = []
    all_sp_index = []
    model.eval()
    context = []
    sl_scores, db_scores, ch_scores, ts = [], [], [], []
    random_indices = sorted(random.sample(range(len(loader)), 10))
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(loader), total=len(loader)):
            coords, features, normals, labels, inverse_map, pseudo_labels, inds, region, index = data

            region = region.squeeze() #[39521, 1] -> [39521]
            scene_name = loader.dataset.name[index[0]]
            gt = labels.clone()
            raw_region = region.clone()

            '''現状の特徴量を計算'''
            in_field = ME.TensorField(coords[:, 1:]*args.voxel_size, coords, device=0)

            feats = model(in_field) #[67911, 4+3] -> [67911, 4]
            feats = feats[inds.long()].cuda() #[67911, 4] -> [39521, 4]

            valid_mask = region!=-1
            features = features[inds.long()].cuda()
            features = features[valid_mask] # [39521, 1] -> [32629, 1]
            normals = normals[inds.long()].cuda()
            normals = normals[valid_mask] # [39521, 3] -> [32629, 3]
            feats = feats[valid_mask]# [39521, 4] -> [32629, 4]
            labels = labels[valid_mask]# [39521, 1] -> [32629, 1]
            region = region[valid_mask].long() # [39521] -> [32629]
            ##
            pc_remission = features # [32629, 1]
            ## region = [0, 1, 2, 1, 0]
            region_num = len(torch.unique(region))
            region_corr = torch.zeros(region.size(0), region_num)#?
            region_corr.scatter_(1, region.view(-1, 1), 1)
            region_corr = region_corr.cuda()##[N, M]
            per_region_num = region_corr.sum(0, keepdims=True).t()
            if torch.any(per_region_num == 0):
                raise ValueError("per_region_num contains zero, which is not allowed.")
            ### per_region_num = [[1], [2], [2]] (266,1)

            region_feats = F.linear(region_corr.t(), feats.t())/per_region_num # [M, 4] これはM個のinit_SPの特徴量
            if current_growsp is not None:
                region_feats = F.normalize(region_feats, dim=-1)
                #
                if region_feats.size(0) < current_growsp:
                    n_segments = region_feats.size(0)
                else:
                    n_segments = current_growsp
                # print(region_feats.size(0), current_growsp, n_segments)
                sp_idx = get_kmeans_labels(n_clusters=n_segments, pcds=region_feats).long()
                # print(sp_idx)
            else:
                feats = region_feats
                sp_idx = torch.tensor(range(region_feats.size(0))).cuda()

            # kmeansの評価
            if batch_idx in random_indices:
                sl_score, db_score, ch_score, t = calc_cluster_metrics(region_feats, sp_idx.cuda())
                sl_scores.append(sl_score)
                db_scores.append(db_score)
                ch_scores.append(ch_score)
                ts.append(t)
            
            neural_region = sp_idx[region]
            
            pfh = []

            '''kmeansしたあとのSPの特徴量を計算'''
            neural_region_num = len(torch.unique(neural_region))
            # print(neural_region.max(), neural_region.min(), neural_region_num)
            neural_region_corr = F.one_hot(neural_region, num_classes=neural_region_num).float().cuda()
            per_neural_region_num = neural_region_corr.sum(0, keepdims=True).t()
            #
            final_remission = F.linear(neural_region_corr.t(), pc_remission.t())/per_neural_region_num
            #
            if current_growsp is not None:
                feats = F.linear(neural_region_corr.t(), feats.t()) / per_neural_region_num
                feats = F.normalize(feats, dim=-1)
            #
            for p in torch.unique(neural_region):
                if p!=-1:
                    mask = p==neural_region
                    pfh.append(compute_hist(normals[mask]).unsqueeze(0))

            pfh = torch.cat(pfh, dim=0) # [266, 10]
            feats = F.normalize(feats, dim=-1)
            # #
            feats = torch.cat((feats, args.c_rgb*final_remission, args.c_shape*pfh), dim=-1)
            feats = F.normalize(feats, dim=-1) # [266, 4+1+10]

            point_feats_list.append(feats.cpu()) # [266, 4+1+10]
            point_labels_list.append(labels.cpu()) # [32629]

            all_sp_index.append(neural_region.to('cpu').detach().numpy().copy()) # [32629]
            context.append((scene_name, gt, raw_region)) # [39521]

            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))
    
    filtered_db_scores = [x for x in db_scores if x != -1]
    filtered_ch_scores = [x for x in ch_scores if x != -1]
    wandb.log({"epoch": epoch, "SC/Silhouette Score": np.mean(sl_scores), "SC/Davies-Bouldin Score": np.mean(filtered_db_scores), "SC/Calinski-Harabasz Score": np.mean(filtered_ch_scores), "SC/Time": np.mean(ts)})

    return point_feats_list, point_labels_list, all_sp_index, context


def get_pseudo(args, context, cluster_pred, all_sp_index=None):
    print('computing pseduo labels...')
    pseudo_label_folder = args.pseudo_label_path + '/'
    if not os.path.exists(pseudo_label_folder):
        os.makedirs(pseudo_label_folder)
    all_gt = []
    all_pseudo = []
    all_pseudo_gt = []
    pc_no = 0
    region_num = 0

    for i in range(len(context)):
        scene_name, labels, region = context[i]

        sub_cluster_pred = all_sp_index[pc_no]+ region_num
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

        pseudo_label_file = pseudo_label_folder + '/' + scene_name + '.npy'
        np.save(pseudo_label_file, pseudo)

        all_gt.append(labels)
        all_pseudo.append(pseudo)
        all_pseudo_gt.append(pseudo_gt)

    all_gt = np.concatenate(all_gt)
    all_pseudo = np.concatenate(all_pseudo)
    all_pseudo_gt = np.concatenate(all_pseudo_gt)

    return all_pseudo, all_gt, all_pseudo_gt


def get_pseudo_kitti(args, context, cluster_pred, all_sub_cluster=None):
    print('computing pseduo labels...')
    all_gt = []
    all_pseudo = []
    all_pseudo_gt = []
    pc_no = 0
    region_num = 0

    for i in range(len(context)):
        scene_name, labels, region = context[i]

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
    

def compute_segment_feats(feats, segs):
    # segsのユニークな要素とそのインデックスを取得
    unique_segs, inverse_indices = torch.unique(segs, return_inverse=True)
    
    # segsのユニークな数（k）
    seg_num = unique_segs.size(0)
    
    # 各セグメントに対応するfeatsの総和を保持するテンソルを初期化 (k, D)
    seg_feats_sum = torch.zeros(seg_num, feats.size(1), device=feats.device)
    
    # 各セグメントに対応する要素数をカウントするためのテンソルを初期化 (k,)
    seg_counts = torch.zeros(seg_num, device=feats.device)
    
    # seg_feats_sumにfeatsを各セグメントごとに足し合わせる
    seg_feats_sum = seg_feats_sum.index_add(0, inverse_indices.squeeze(), feats)
    
    # 各セグメントごとの要素数をカウント
    seg_counts = seg_counts.index_add(0, inverse_indices.squeeze(), torch.ones_like(segs, dtype=torch.float))
    
    # seg_feats_sumをセグメントごとの要素数で割って平均を計算
    seg_feats = seg_feats_sum / seg_counts[:, None]
    
    return seg_feats


def calc_info_nce(seg_feats_q, seg_feats_k, temperature=0.07):
    sims = F.cosine_similarity(seg_feats_q.unsqueeze(1), seg_feats_k.unsqueeze(0), dim=2) / temperature
    sims = sims.cuda()
    m, n = sims.size()
    num_pos = min(m, n)
    # 行方向の損失を計算
    labels_row = torch.arange(num_pos).to(sims.device)
    loss_row = F.cross_entropy(sims[:num_pos, :], labels_row, ignore_index=-1)
    
    # 列方向の損失を計算
    labels_col = torch.arange(num_pos).to(sims.device)
    loss_col = F.cross_entropy(sims[:, :num_pos].T, labels_col, ignore_index=-1)
    
    # 総損失を計算
    loss = (loss_row + loss_col) / 2
    return loss    


@torch.no_grad()
def copy_minkowski_network_params(source_model, target_model):
    # モデルが同じ構造を持っていることを確認
    assert type(source_model) == type(target_model), "モデルの型が一致しません"

    # ソースモデルの状態辞書を取得
    source_state_dict = source_model.state_dict()

    # ターゲットモデルの状態辞書を取得
    target_state_dict = target_model.state_dict()

    # パラメータをコピー
    for name, param in source_state_dict.items():
        if name in target_state_dict:
            if isinstance(param, ME.SparseTensor):
                # SparseTensorの場合、特別な処理が必要
                target_state_dict[name] = ME.SparseTensor(
                    features=param.F.clone().detach(),
                    coordinates=param.C.clone().detach(),
                    tensor_stride=param.tensor_stride
                )
            else:
                # 通常のテンソルの場合、単純にコピー
                target_state_dict[name].copy_(param.detach())

    # 更新された状態辞書をターゲットモデルにロード
    target_model.load_state_dict(target_state_dict)
    return source_model, target_model


@torch.no_grad()
def momentum_update_key_encoder(model_q, model_k, proj_head_q, proj_head_k, momentum:int=0.999):
    """
    Momentum update of the key encoder
    """
    momentum_update_model(model_q, model_k, momentum)

    for param_q, param_k in zip(proj_head_q.parameters(), proj_head_k.parameters()):
        param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
        
        
@torch.no_grad()
def momentum_update_model(model_q, model_k, momentum:int=0.999):
    """
    Momentum update of the key encoder
    """
    model_q_dict = model_q.state_dict()
    model_k_dict = model_k.state_dict()

    for name, param_q in model_q_dict.items():
        if name in model_k_dict:
            if isinstance(param_q, ME.SparseTensor):
                # SparseTensorの場合、特別な処理が必要
                param_k = model_k_dict[name]
                new_F = param_k.F * momentum + param_q.F.detach() * (1 - momentum)
                model_k_dict[name] = ME.SparseTensor(
                    features=new_F,
                    coordinates=param_k.C, # coordinatesは共通でいいのか？
                    tensor_stride=param_k.tensor_stride
                )
            else:
                # 通常のテンソルの場合、単純に更新
                model_k_dict[name] = model_k_dict[name] * momentum + param_q * (1 - momentum)

    model_k.load_state_dict(model_k_dict)
    

def get_kmeans_labels(n_clusters, pcds, max_iter=300):
    """
    KMeansを用いてクラスタリングを行い、各点に対するクラスタラベルを返す
    Args:
        n_clusters (int): クラスタ数
        pcds (np.ndarray | torch.tensor): 点群データ (N, D)?
        max_iter (int): KMeansの最大イテレーション数
    Returns:
        labels (torch.tensor): 各点に対するクラスタラベル (N,)
    """
    model = KMeans(n_clusters=n_clusters, max_iter=max_iter, distance='euclidean')
    with torch.no_grad():
        if isinstance(pcds, np.ndarray):
            pcds = torch.from_numpy(pcds)
        pcds = pcds.cuda().float()
        unsqueezed = pcds.unsqueeze(0)
        centroids, labels = model(unsqueezed)
        # centroids = centroids.squeeze(0)
        # distances = torch.cdist(pcds, centroids)
        # labels = torch.argmin(distances, dim=1)
    return labels.squeeze(0)


def calc_cluster_metrics(X, labels):
    """
    クラスタリングの評価指標を計算する
    Args:
        X (torch.tensor): データ行列 (N, D)
        labels (np.ndarray): クラスタラベル (N,)
        epoch (int): 現在のエポック数
        method (str): クラスタリングの手法名
    Returns:
        sl_score (float): シルエットスコア
        db_score (float): Davies-Bouldinスコア
        ch_score (float): Calinski-Harabaszスコア
        time (float): 計算時間

    """
    start = time.time()
    with torch.no_grad():
        sl_score = silhouette.score(X, labels)
        X = X.cpu().numpy()
        if type(labels) != np.ndarray:
            labels = labels.cpu().numpy()
        try:
            db_score = davies_bouldin_score(X, labels)
            ch_score = calinski_harabasz_score(X, labels)
        except ValueError:
            db_score = -1
            ch_score = -1
        
    t = time.time() - start
    
    return sl_score, db_score, ch_score, t
    


if __name__ == '__main__':
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