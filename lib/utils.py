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
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
try:
    from torchclustermetrics import silhouette
except :
    pass
import time
import random

# from models.fpn import Res16FPNBase
from lib.kmeans_torch import KMeans as KMeans_gpu
from sklearn.cluster import KMeans as KMeans_sklearn

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
            features = features[inds.long()].to("cuda:0")
            features = features[valid_mask]
            normals = normals[inds.long()].to("cuda:0")
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
            region_corr = region_corr.to("cuda:0")##[N, M]
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
            neural_region_corr = neural_region_corr.to("cuda:0")
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
                    pfh.append(compute_hist(normals[mask].cpu()).unsqueeze(0).to("cuda:0"))

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
    context = []
    model.eval()
    sl_scores, db_scores, ch_scores, ts = [], [], [], []
    random_indices = sorted(random.sample(range(len(loader)), 10)) if args.silhouette else []
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(loader), total=len(loader), desc="get_kittisp_feature in utils.py"):
            coords, features, normals, labels, inverse_map, pseudo_labels, inds, region, index = data

            region = region.squeeze() #[39521, 1] -> [39521]
            scene_name = loader.dataset.name[index[0]]
            gt = labels.clone()
            raw_region = region.clone()

            # 現状の特徴量を計算
            in_field = ME.TensorField(coords[:, 1:] * args.voxel_size, coords, device=0)
            feats = model(in_field) #[67911, 4+3] -> [67911, 4]
            feats = feats[inds.long()].to("cuda:0") #[67911, 4] -> [39521, 4]

            valid_mask = region != -1
            features = features[inds.long()].to("cuda:0")
            features = features[valid_mask] # [39521, 1] -> [32629, 1]
            normals = normals[inds.long()].to("cuda:0")
            normals = normals[valid_mask] # [39521, 3] -> [32629, 3]
            feats = feats[valid_mask]# [39521, 4] -> [32629, 4]
            labels = labels[valid_mask]# [39521, 1] -> [32629, 1]
            region = region[valid_mask].long().to("cuda:0") # [39521] -> [32629]
            ##
            pc_remission = features # [32629, 1]
            ## region = [0, 1, 2, 1, 0]
            region_num = len(torch.unique(region))
            region_corr = torch.zeros(region.size(0), region_num).to("cuda:0")
            region_corr.scatter_(1, region.view(-1, 1), 1)##[N, M] M=266 は M_0、つまり、init superpointの数
            per_region_num = region_corr.sum(0, keepdims=True).t()
            ### per_region_num = [[1], [2], [2]] (266,1) 266個のSuperpointについて、それぞれのに含まれている点の数

            region_feats = F.linear(region_corr.t(), feats.t()) / per_region_num # [M, 4] これはM個のinit_SPの特徴量
            if current_growsp is not None:
                region_feats = F.normalize(region_feats, dim=-1)
                if region_feats.size(0) < current_growsp: # 基本ない？M=266が80よりも小さくなるような場合
                    n_segments = region_feats.size(0)
                else:
                    n_segments = current_growsp
                sp_idx = get_kmeans_labels(n_clusters=n_segments, pcds=region_feats).long()
            else:
                feats = region_feats
                sp_idx = torch.arange(region_feats.size(0)).to("cuda:0")

            # kmeansの評価
            if batch_idx in random_indices:
                sl_score, db_score, ch_score, t = calc_cluster_metrics(region_feats, sp_idx.to("cuda:0"))
                sl_scores.append(sl_score)
                db_scores.append(db_score)
                ch_scores.append(ch_score)
                ts.append(t)

            neural_region = sp_idx[region]

            pfh = []
            '''kmeansしたあとのSPの特徴量を計算'''
            neural_region_num = len(torch.unique(neural_region))
            neural_region_corr = F.one_hot(neural_region, num_classes=neural_region_num).float().to("cuda:0")
            per_neural_region_num = neural_region_corr.sum(0, keepdims=True).t()
            final_remission = F.linear(neural_region_corr.t(), pc_remission.t()) / per_neural_region_num

            if current_growsp is not None:
                feats = F.linear(neural_region_corr.t(), feats.t()) / per_neural_region_num
                feats = F.normalize(feats, dim=-1)

            for p in torch.unique(neural_region):
                if p != -1:
                    mask = p == neural_region
                    pfh.append(compute_hist(normals[mask]).unsqueeze(0))

            pfh = torch.cat(pfh, dim=0) # [266, 10]
            feats = F.normalize(feats, dim=-1)
            feats = torch.cat((feats, args.c_rgb * final_remission, args.c_shape * pfh), dim=-1)
            feats = F.normalize(feats, dim=-1) # [266, 4+1+10]

            # データをCPUに移動
            feats_cpu = feats.cpu()
            labels_cpu = labels.cpu()
            neural_region_cpu = neural_region.cpu().detach().numpy().copy()

            # リストにデータを追加
            point_feats_list.append(feats_cpu) # [266, 4+1+10]
            point_labels_list.append(labels_cpu) # [32629]
            all_sp_index.append(neural_region_cpu) # [32629]
            context.append((scene_name, gt, raw_region)) # [39521]

            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))

    # kmeansの評価結果をログに記録
    filtered_db_scores = [x for x in db_scores if x != -1]
    filtered_ch_scores = [x for x in ch_scores if x != -1]
    if not((len(sl_scores)==0) and (len(filtered_ch_scores)==0) and (len(filtered_db_scores)==0)):
        wandb.log({
            "epoch": epoch,
            "SC/Silhouette Score": np.mean(sl_scores),
            "SC/Davies-Bouldin Score": np.mean(filtered_db_scores),
            "SC/Calinski-Harabasz Score": np.mean(filtered_ch_scores),
            "SC/Time": np.mean(ts)
        })

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
    

def compute_segment_feats(feats: torch.Tensor,
                          segs: torch.Tensor,
                          max_seg_num: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    各セグメントごとの特徴量を平均し、固定サイズのテンソルを返す。
    パディング部分のマスクも同時に返す。

    Args:
        feats: 形状 [N, D] の特徴量テンソル (N: ポイント数, D: 特徴次元)
        segs: 形状 [N] のセグメントIDテンソル (-1 は無効)
        max_seg_num: 出力するセグメント数の最大値 (例: current_growsp)

    Returns:
        seg_feats: 形状 [max_seg_num, D] のテンソル
            存在するセグメントIDの位置に平均特徴量が入り、
            存在しないセグメントIDの位置は 0 で埋められる。
        mask: 形状 [max_seg_num] の boolean テンソル
            セグメントが存在する位置は True、存在しない位置は False。
    """
    # 1. 無効データ (-1) を除外
    valid_mask = segs != -1
    feats = feats[valid_mask]
    segs = segs[valid_mask]

    # 2. 初期化
    seg_feats_sum = torch.zeros(max_seg_num, feats.shape[1], device=feats.device)
    seg_counts = torch.zeros(max_seg_num, device=feats.device)

    # 3. 特徴量の合計とセグメントごとのカウント
    seg_feats_sum.index_add_(0, segs, feats)
    seg_counts.index_add_(0, segs, torch.ones_like(segs, dtype=torch.float, device=feats.device))

    # 4. 平均計算 (ゼロ除算を防ぐ)
    seg_feats = torch.where(
        seg_counts.unsqueeze(-1) > 0,
        seg_feats_sum / seg_counts.unsqueeze(-1),
        torch.zeros_like(seg_feats_sum)
    )

    # 5. 存在しないセグメントのマスク
    mask = seg_counts <= 0

    return seg_feats, mask


def calc_info_nce(seg_feats_q, seg_feats_k, mask_q, mask_k, temperature=0.07):
    assert seg_feats_q.size() == seg_feats_k.size()
    batch_size, seg_num, dim = seg_feats_q.size()
    
    # マスクを展開してブール型に変換（パディング部分がFalse、データ部分がTrue）
    mask_q = ~mask_q  # パディング部分がFalse、データ部分がTrue
    mask_k = ~mask_k

    losses = []
    for i in range(batch_size):
        # 有効なシーケンス長を取得 ok
        valid_feats_q = seg_feats_q[i][mask_q[i]]
        valid_feats_k = seg_feats_k[i][mask_k[i]]
        
        # コサイン類似度の計算 ok
        sims = F.cosine_similarity(valid_feats_q.unsqueeze(1), valid_feats_k.unsqueeze(0), dim=2) / temperature
        
        labels_row = create_label(mask_q[i], mask_k[i], device=sims.device)
        labels_col = create_label(mask_k[i], mask_q[i], device=sims.device)
        
        # 損失の計算
        loss_row = F.cross_entropy(sims, labels_row, ignore_index=-100)
        loss_col = F.cross_entropy(sims.t(), labels_col, ignore_index=-100)
        loss = (loss_row + loss_col) / 2
        losses.append(loss)
    
    # バッチ全体の損失を平均
    total_loss = torch.stack(losses).mean()
    return total_loss


def create_label(mask_row, mask_column, device):
    label_row = torch.nonzero(mask_row, as_tuple=False).view(-1)
    label_column = torch.nonzero(mask_column, as_tuple=False).view(-1)
    result = torch.tensor([x if x in label_column else -100 for x in label_row]).to(device)
    # 検索用のマスクを生成
    index_map = torch.full_like(result, -1).to(device)
    for idx, value in enumerate(label_column):
        index_map[result == value] = idx

    # 結果を保持するテンソルを生成
    result = torch.where(index_map >= 0, index_map, result).to(device)
    return result


@torch.no_grad()
def copy_minkowski_network_params(source_model: nn.Module, target_model: nn.Module):
    """
    source_model のパラメータを target_model に丸ごとコピーする.
    MinkowskiEngine であっても, 学習パラメータ (weight/bias) は普通の torch.Tensor なので、
    SparseTensor を無視して, 単純にパラメータをコピーすればOK.
    """
    for p_src, p_tgt in zip(source_model.parameters(), target_model.parameters()):
        p_tgt.data.copy_(p_src.data)

@torch.no_grad()
def momentum_update_key_encoder(model_q, model_k, proj_head_q, proj_head_k, momentum=0.999):
    """
    MoCoスタイルのMomentum Update:
      - model_qの学習パラメータを参照し,
      - model_kのパラメータを momentum * k + (1 - momentum) * q で更新する
      - proj_head_q, proj_head_k についても同様
    """
    # model本体 (model_q, model_k) のパラメータ更新
    momentum_update_model(model_q, model_k, momentum)
    # proj_head (proj_head_q, proj_head_k) のパラメータ更新
    momentum_update_model(proj_head_q, proj_head_k, momentum)


@torch.no_grad()
def momentum_update_model(model_q: nn.Module, model_k: nn.Module, momentum: float = 0.999):
    """
    model_q の学習パラメータを使って model_k をMomentum Updateする
    ただし, SparseTensor など学習パラメータでないものは更新しない.
    """
    # model_q.parameters() と model_k.parameters() を順番にたどる
    for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
        # param_q, param_k はともに nn.Parameter (torch.Tensor)
        # MinkowskiEngineのMinkowskiConvolutionなどでも weight/bias はTorch Tensor
        # SparseTensorはそもそも model_q.parameters() には含まれないはず
        param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
    

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
    elif len(sys.argv) > 1 and sys.argv[1] == '2':
        # compute_segment_feats関数のテスト
        import torch

        # ダミーデータの生成
        feats = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        segs = torch.tensor([0, 2, 0, -1])
        max_seg_num = 3

        # compute_segment_feats関数の呼び出し
        seg_feats, mask = compute_segment_feats(feats, segs, max_seg_num)

        # 結果の表示
        print("Segment Features:\n", seg_feats)
        print("Mask:\n", mask)
    elif len(sys.argv) > 1 and sys.argv[1] == '3':
        # create_label関数のテスト
        import torch
        label1 = torch.tensor([True, False, True, False, True])
        label2 = torch.tensor([True, True, False, False, True, True])
        print(create_label(label1, label2))
        print(create_label(label2, label1))