import warnings
warnings.filterwarnings('ignore')
import argparse
import time
import os
import numpy as np
import random
from datasets.SemanticKITTI import KITTItrain, cfl_collate_fn, KITTItemporal, KITTItcuss, cfl_collate_fn_tcuss
import torch
import MinkowskiEngine as ME
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.fpn import Res16FPN18, Res16FPNBase
from models.transformer_projector import TransformerProjector
from eval_SemanticKITTI import eval
from lib.utils import get_pseudo_kitti, get_kittisp_feature, get_fixclassifier, copy_minkowski_network_params, compute_segment_feats, momentum_update_key_encoder, calc_info_nce, momentum_update_model, get_kmeans_labels, calc_cluster_metrics
import logging
from os.path import join
import wandb
import torch.multiprocessing as multiprocessing
from lib.helper_ply import read_ply
from tqdm import tqdm
from math import ceil


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(description='PyTorch Unsuper_3D_Seg')
    parser.add_argument('--name', type=str, required=True, help='name of the experiment')
    parser.add_argument('--data_path', type=str, default='data/users/minesawa/semantickitti/growsp',
                        help='point cloud data path')
    parser.add_argument('--sp_path', type=str, default='data/users/minesawa/semantickitti/growsp_sp',
                        help='initial sp path')
    parser.add_argument('--original_data_path', type=str, default='data/dataset/semantickitti/dataset/sequences')
    parser.add_argument('--patchwork_path', type=str, default='data/users/minesawa/semantickitti/patchwork')
    parser.add_argument('--save_path', type=str, default='data/users/minesawa/semantickitti/growsp_model', help='model savepath')
    parser.add_argument('--pseudo_label_path', default='pseudo_label_kitti/', type=str, help='pseudo label save path') # 同時に複数実行する場合のみ、被らないように変更する必要がある
    ##
    parser.add_argument('--max_epoch', type=int, nargs='+', default=[100, 350], help='max epoch for non-growing and growing stage')
    parser.add_argument('--max_iter', type=list, default=[10000, 30000], help='max iter for non-growing and growing stage')
    ###
    parser.add_argument('--bn_momentum', type=float, default=0.02, help='batchnorm parameters')
    parser.add_argument('--conv1_kernel_size', type=int, default=5, help='kernel size of 1st conv layers')
    ####
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate for backbone network')
    parser.add_argument('--tarl_lr', type=float, default=0.0002, help='learning rate for transformer projector and predictor')
    parser.add_argument('--workers', type=int, default=16, help='how many workers for loading data')
    parser.add_argument('--cluster_workers', type=int, default=16, help='how many workers for loading data in clustering')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--log-interval', type=int, default=1000000, help='log interval')
    parser.add_argument('--batch_size', type=int, nargs='+', default=[16, 16], help='batchsize in training[GrowSP, TARL]')
    parser.add_argument('--voxel_size', type=float, default=0.15, help='voxel size in SparseConv')
    parser.add_argument('--input_dim', type=int, default=3, help='network input dimension')### 6 for XYZGB
    parser.add_argument('--primitive_num', type=int, default=500, help='how many primitives used in training')
    parser.add_argument('--semantic_class', type=int, default=19, help='ground truth semantic class')
    parser.add_argument('--feats_dim', type=int, default=128, help='output feature dimension')
    parser.add_argument('--ignore_label', type=int, default=-1, help='invalid label')
    parser.add_argument('--growsp_start', type=int, default=80, help='the start number of growing superpoint')
    parser.add_argument('--growsp_end', type=int, default=30, help='the end number of grwoing superpoint')
    parser.add_argument('--drop_threshold', type=int, default=10, help='ignore superpoints with few points')
    parser.add_argument('--w_rgb', type=float, default=5/5, help='weight for RGB in merging superpoint')
    parser.add_argument('--c_rgb', type=float, default=5, help='weight for RGB in clustering primitives')
    parser.add_argument('--c_shape', type=float, default=5, help='weight for PFH in clustering primitives')
    parser.add_argument('--select_num', type=int, default=1500, help='scene number selected in each round')
    parser.add_argument('--eval_select_num', type=int, default=4071, help='scene number selected in evaluation')
    parser.add_argument('--r_crop', type=float, default=50, help='cropping radius in training')
    parser.add_argument('--cluster_interval', type=int, default=10, help='cluster interval')
    parser.add_argument('--eval_interval', type=int, default=10, help='eval interval')
    parser.add_argument('--silhouette', action='store_true', help='more eval metrics for kmeans')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--scan_window', type=int, default=12, help='scan window size')
    # parser.add_argument('--lmb', type=float, default=0.5, help='lambda for contrastive learning')
    parser.add_argument('--vis', action='store_true', help='visualize')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--wandb_run_id', type=str, help='wandb run id')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='weight decay')
    parser.add_argument('--accum_step', type=int, default=1, help='gradient accumulation step')
    return parser.parse_args()


def main(args, logger):
    run = wandb.init(
        project="TCUSS",
        config=vars(args),
        name = args.name if args.name else None,
        resume= 'must' if args.resume else 'never',
        id = args.wandb_run_id if args.resume else None,
        settings=wandb.Settings(code_dir=".")
    )
    
    model_q = Res16FPN18(in_channels=args.input_dim, out_channels=args.feats_dim, conv1_kernel_size=args.conv1_kernel_size, config=args)
    model_k = Res16FPN18(in_channels=args.input_dim, out_channels=args.feats_dim, conv1_kernel_size=args.conv1_kernel_size, config=args)
    proj_head_q = TransformerProjector(d_model=args.feats_dim, num_layer=1)
    proj_head_k = TransformerProjector(d_model=args.feats_dim, num_layer=1)
    predictor = TransformerProjector(d_model=args.feats_dim, num_layer=1)
    model_q = model_q.to("cuda:0")
    model_k = model_k.to("cuda:0")
    proj_head_q = proj_head_q.to("cuda:0")
    proj_head_k = proj_head_k.to("cuda:0")
    predictor = predictor.to("cuda:0")
    copy_minkowski_network_params(model_q, model_k)
    for param_q, param_k in zip(proj_head_q.parameters(), proj_head_k.parameters()):
        param_k.data.copy_(param_q.data)
        param_k.requires_grad = False
        
    resume_epoch = None
    if args.resume:
        last_epoch = wandb.run.summary.get("epoch", 0)
        resume_epoch = ((last_epoch-1) // args.cluster_interval) * args.cluster_interval + 1
        print(f'Resume from epoch {resume_epoch}')
    
    
    '''Random select 1500 scans to train, will redo in each round'''
    scene_idx = np.random.choice(19130, args.select_num, replace=False).tolist()## SemanticKITTI totally has 19130 training samples
    trainset = KITTItcuss(args)
    train_loader = DataLoader(trainset, batch_size=args.batch_size[0], shuffle=True, collate_fn=cfl_collate_fn_tcuss(), num_workers=args.workers, pin_memory=True, worker_init_fn=worker_init_fn)
    clusterset = KITTItrain(args, scene_idx, 'train')
    cluster_loader = DataLoader(clusterset, batch_size=1, collate_fn=cfl_collate_fn(), num_workers=args.cluster_workers, pin_memory=True)
    
    # パラメータグループを分けて異なる学習率を設定
    backbone_params = list(model_q.parameters())
    transformer_params = list(proj_head_q.parameters()) + list(predictor.parameters())
    
    param_groups = [
        {'params': backbone_params, 'lr': args.lr},
        {'params': transformer_params, 'lr': args.tarl_lr}
    ]
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    
    schedulers = [torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[args.lr, args.tarl_lr], epochs=epoch, steps_per_epoch=ceil(len(train_loader) / args.accum_step)) for epoch in args.max_epoch]
    
    momentum_update_key_encoder(model_q, model_k, proj_head_q, proj_head_k)
    
    is_Growing = False
    for i, (epoch, scheduler) in enumerate(zip(args.max_epoch, schedulers)):
        train_loader.dataset.phase = i
        main_half(args, i, train_loader, cluster_loader, is_Growing, resume_epoch, model_q, model_k, proj_head_q, proj_head_k, predictor, optimizer, scheduler, logger)
        is_Growing = True


def main_half(args, phase, train_loader, cluster_loader, is_Growing, resume_epoch, model_q, model_k, proj_head_q, proj_head_k, predictor, optimizer, scheduler, logger):
    start_epoch = 0 if phase==0 else args.max_epoch[0]
    end_epoch = args.max_epoch[0] if phase==0 else sum(args.max_epoch)
    for epoch in range(start_epoch+1, end_epoch+1):
        if args.resume:
            if epoch < resume_epoch:
                continue    
            elif epoch == resume_epoch:
                checkpoint = torch.load(join(args.save_path, f'checkpoint_epoch_{resume_epoch-1}.pth'))
                model_q.load_state_dict(checkpoint['model_q_state_dict'])
                model_k.load_state_dict(checkpoint['model_k_state_dict'])
                proj_head_q.load_state_dict(checkpoint['proj_head_q_state_dict'])
                proj_head_k.load_state_dict(checkpoint['proj_head_k_state_dict'])
                predictor.load_state_dict(checkpoint['predictor_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if f'scheduler_{phase}_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint[f'scheduler_{phase}_state_dict'])
                if 'np_random_state' in checkpoint:
                    np.random.set_state(checkpoint['np_random_state'])
                if 'torch_random_state' in checkpoint:
                    torch.set_rng_state(checkpoint['torch_random_state'])
                if torch.cuda.is_available() and 'torch_cuda_random_state' in checkpoint and checkpoint['torch_cuda_random_state'] is not None:
                    torch.cuda.set_rng_state(checkpoint['torch_cuda_random_state'])
        if (epoch-1) % args.cluster_interval==0:
            train_loader.dataset.random_select_sample()
            scene_idx = train_loader.dataset.scene_idx_all
            cluster_loader.dataset.random_select_sample(scene_idx)
            classifier, current_growsp = cluster(args, logger, cluster_loader, model_q, epoch, args.max_epoch[0], is_Growing)
        
        train_loader.dataset.kittitemporal.n_clusters = current_growsp
        train(args, train_loader, model_q, model_k, proj_head_q, proj_head_k, predictor, classifier, optimizer, epoch, scheduler, current_growsp)
        
        # eval
        if epoch % args.eval_interval==0:
            torch.save(model_q.state_dict(), join(args.save_path,  'model_' + str(epoch) + '_checkpoint.pth'))
            torch.save(classifier.state_dict(), join(args.save_path, 'cls_' + str(epoch) + '_checkpoint.pth'))
            with torch.no_grad():
                o_Acc, m_Acc, m_IoU, s, IoU_dict = eval(epoch, args)
                logger.info('Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} IoUs'.format(epoch, o_Acc, m_Acc) + s)
                d = {'epoch': epoch, 'oAcc': o_Acc, 'mAcc': m_Acc, 'mIoU': m_IoU}
                d.update(IoU_dict)
                wandb.log(d)
                if args.silhouette:
                    # SPCの評価
                    feats, *_ = get_kittisp_feature(args, cluster_loader, model_q, current_growsp, epoch)
                    sp_feats = torch.cat(feats, dim=0)
                    primitive_labels = get_kmeans_labels(args.primitive_num, sp_feats).to('cpu').detach().numpy() # Semantic Primitive Clustering (SPC)
                    sl_score, db_score, ch_score, t = calc_cluster_metrics(sp_feats, primitive_labels)
                    wandb.log({'epoch': epoch, 'SPC/Silhouette': sl_score, 'SPC/Davies-Bouldin': db_score, 'SPC/Calinski-Harabasz': ch_score, 'SPC/time': t})
        if epoch % args.cluster_interval==0:
            torch.save({
                'epoch': epoch,
                'model_q_state_dict': model_q.state_dict(),
                'model_k_state_dict': model_k.state_dict(),
                'proj_head_q_state_dict': proj_head_q.state_dict(),
                'proj_head_k_state_dict': proj_head_k.state_dict(),
                'predictor_state_dict': predictor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                f'scheduler_{phase}_state_dict': scheduler.state_dict(),
                'np_random_state': np.random.get_state(),
                'torch_random_state': torch.get_rng_state(),
                'torch_cuda_random_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            }, join(args.save_path, f'checkpoint_epoch_{epoch}.pth'))


def cluster(args, logger, cluster_loader:DataLoader, model_q:Res16FPNBase, epoch:int, start_grow_epoch:int=None, is_Growing:bool=False):
    time_start = time.time()
    cluster_loader.dataset.mode = 'cluster'

    current_growsp = None
    if is_Growing:
        current_growsp = int(args.growsp_start - ((epoch - start_grow_epoch)/args.max_epoch[1])*(args.growsp_start - args.growsp_end))
        if current_growsp < args.growsp_end:
            current_growsp = args.growsp_end
        logger.info('Epoch: {}, Superpoints Grow to {}'.format(epoch, current_growsp))

    '''Extract Superpoints Feature'''
    feats, labels, sp_index, context = get_kittisp_feature(args, cluster_loader, model_q, current_growsp, epoch) #Superpoint Constructor
    sp_feats = torch.cat(feats, dim=0)### will do Kmeans with geometric distance
    primitive_labels = get_kmeans_labels(args.primitive_num, sp_feats).to('cpu').detach().numpy() # Semantic Primitive Clustering (SPC)

    sp_feats = sp_feats[:,0:args.feats_dim]### drop geometric feature

    '''Compute Primitive Centers'''
    primitive_centers = torch.zeros((args.primitive_num, args.feats_dim))
    for cluster_idx in range(args.primitive_num):
        indices = primitive_labels == cluster_idx
        cluster_avg = sp_feats[indices].mean(0, keepdims=True)
        primitive_centers[cluster_idx] = cluster_avg
    primitive_centers = F.normalize(primitive_centers, dim=1)
    classifier = get_fixclassifier(in_channel=args.feats_dim, centroids_num=args.primitive_num, centroids=primitive_centers)

    '''Compute and Save Pseudo Labels'''
    all_pseudo, all_gt, all_pseudo_gt = get_pseudo_kitti(args, context, primitive_labels, sp_index)
    logger.info('labelled points ratio %.2f clustering time: %.2fs', (all_pseudo!=-1).sum()/all_pseudo.shape[0], time.time() - time_start)

    '''Check Superpoint/Primitive Acc in Training'''
    sem_num = args.semantic_class
    mask = (all_pseudo_gt!=-1)
    histogram = np.bincount(sem_num* all_gt.astype(np.int32)[mask] + all_pseudo_gt.astype(np.int32)[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)    # hungarian matching
    o_Acc = histogram[range(sem_num), range(sem_num)].sum()/histogram.sum()*100
    tp = np.diag(histogram)
    fp = np.sum(histogram, 0) - tp
    fn = np.sum(histogram, 1) - tp
    IoUs = tp / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
    for IoU in IoUs:
        s += '{:5.2f} '.format(100 * IoU)
    logger.info('Superpoints oAcc {:.2f} IoUs'.format(o_Acc) + s)

    pseudo_class2gt = -np.ones_like(all_gt)
    for i in range(args.primitive_num):
        mask = all_pseudo==i
        pseudo_class2gt[mask] = torch.mode(torch.from_numpy(all_gt[mask])).values
    mask = (pseudo_class2gt!=-1)&(all_gt!=-1)
    histogram = np.bincount(sem_num* all_gt.astype(np.int32)[mask] + pseudo_class2gt.astype(np.int32)[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)    # hungarian matching
    o_Acc = histogram[range(sem_num), range(sem_num)].sum()/histogram.sum()*100
    tp = np.diag(histogram)
    fp = np.sum(histogram, 0) - tp
    fn = np.sum(histogram, 1) - tp
    IoUs = tp / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
    for IoU in IoUs:
        s += '{:5.2f} '.format(100 * IoU)
    logger.info('Primitives oAcc {:.2f} IoUs'.format(o_Acc) + s)
    return classifier.to("cuda:0"), current_growsp


def train(args, train_loader, model_q, model_k, proj_head_q, proj_head_k, predictor, classifier, optimizer, epoch, scheduler, current_growsp):
    model_q.train()
    model_k.train()
    proj_head_q.train()
    proj_head_k.train()
    predictor.train()
    optimizer.zero_grad()
    loss_growsp_display = 0.0
    loss_tarl_display = 0.0
    for i, data in tqdm(enumerate(train_loader), desc='Train Epoch: {}'.format(epoch), total=len(train_loader)):
        growsp_t1_data, growsp_t2_data, tarl_data = data
        # growsp用の関数作るか、dataとmodel_qだけ入力して、lossを計算する
        growsp_t1_loss = train_growsp(args, growsp_t1_data, model_q, classifier)
        growsp_t2_loss = train_growsp(args, growsp_t2_data, model_q, classifier)
        if tarl_data is not None:
            tarl_loss = train_tarl(args, tarl_data, model_q, model_k, proj_head_q, proj_head_k, predictor, current_growsp) / args.accum_step
        else:
            tarl_loss = torch.tensor(0.0, device="cuda")
        growsp_loss = (growsp_t1_loss + growsp_t2_loss) / args.accum_step
        loss = growsp_loss + tarl_loss
        loss_growsp_display += growsp_loss.item()
        loss_tarl_display += tarl_loss.item()
        loss.backward()
        if ((i+1) % args.accum_step == 0) or (i == len(train_loader)-1):
            optimizer.step()
            wandb.log({
                'epoch': epoch, 
                'backbone_lr': optimizer.param_groups[0]['lr'], 
                'transformer_lr': optimizer.param_groups[1]['lr']
            })
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            momentum_update_key_encoder(model_q, model_k, proj_head_q, proj_head_k)
            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))
    wandb.log({'epoch': epoch, 'loss_growsp': loss_growsp_display, 'loss_tarl': loss_tarl_display})


def train_tarl(args, tarl_data, model_q, model_k, proj_head_q, proj_head_k, predictor, current_growsp):
    coords_q, coords_k, segs_q, segs_k = tarl_data
    loss = train_contrast_half(coords_q, coords_k, segs_q, segs_k, model_q, model_k, proj_head_q, proj_head_k, predictor, current_growsp)
    loss += train_contrast_half(coords_k, coords_q, segs_k, segs_q, model_q, model_k, proj_head_q, proj_head_k, predictor, current_growsp)
    return loss
    

def train_growsp(args, growsp_data, model_q, classifier):
    loss = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label).to("cuda:0")
    coords, pseudo_labels, inds = growsp_data
    in_field = ME.TensorField(coords[:, 1:]*args.voxel_size, coords, device=0)
    feats = model_q(in_field)
    feats = feats[inds.long()]
    feats = F.normalize(feats, dim=-1)
    pseudo_labels_comp = pseudo_labels.long().to("cuda:0")
    logits = F.linear(F.normalize(feats), F.normalize(classifier.weight))
    loss_sem = loss(logits * 5, pseudo_labels_comp).mean()
    return loss_sem


def train_contrast_half(coords_q, coords_k, segs_q, segs_k, model_q, model_k, proj_head_q, proj_head_k, predictor, current_growsp):
    in_field_q = ME.TensorField(coords_q[:, 1:] * args.voxel_size, coords_q, device=0)
    # checkpoint = torch.load(join(f'/mnt/urashima/users/minesawa/semantickitti/seg_feat/checkpoint_epoch_220.pth'))
    # checkpoint = torch.load(join(f'/mnt/urashima/users/minesawa/semantickitti/growsp_model/model_350_checkpoint.pth'))
    # model_q.load_state_dict(checkpoint['model_q_state_dict'])
    # model_q.load_state_dict(checkpoint)
    feats_q = model_q(in_field_q)
    batch_ids = torch.unique(coords_q[:, 0])
    seg_feats_q_list, mask_q_list = [], []
    for batch_id in batch_ids:
        mask = coords_q[:, 0] == batch_id
        scene_feats_q = feats_q[mask]
        scene_segs_q = segs_q[int(batch_id)].to("cuda:0")
        scene_seg_feats_q, mask_q = compute_segment_feats(scene_feats_q, scene_segs_q, max_seg_num=current_growsp)
        seg_feats_q_list.append(scene_seg_feats_q)
        mask_q_list.append(mask_q)
        # point_cloud_data = torch.cat((coords_q[mask][:, 1:], segs_q[int(batch_id)].unsqueeze(1)), dim=1)
        # np.savetxt("coords_q.csv", point_cloud_data.cpu().detach().numpy(), delimiter=",")

    padded_seg_feats_q = torch.stack(seg_feats_q_list, dim=0)
    batch_mask_q = torch.stack(mask_q_list, dim=0)
    
    # プロジェクションヘッドに入力
    proj_feats_q = proj_head_q(padded_seg_feats_q, enc_mask=batch_mask_q)

    # プレディクターに入力
    pred_feats_q = predictor(proj_feats_q, enc_mask=batch_mask_q)
    pred_feats_q = F.normalize(pred_feats_q, dim=-1)

    with torch.no_grad():
        in_field_k = ME.TensorField(coords_k[:, 1:] * args.voxel_size, coords_k, device=0)
        feats_k = model_k(in_field_k)
        batch_ids = torch.unique(coords_k[:, 0])
        seg_feats_k_list , mask_k_list = [], []
        for batch_id in batch_ids:
            mask = coords_k[:, 0] == batch_id
            scene_feats_k = feats_k[mask]
            scene_segs_k = segs_k[int(batch_id)].to("cuda:0")
            scene_seg_feats_k, mask_k = compute_segment_feats(scene_feats_k, scene_segs_k, max_seg_num=current_growsp)
            seg_feats_k_list.append(scene_seg_feats_k)
            mask_k_list.append(mask_k)
            # point_cloud_data = torch.cat((coords_k[mask][:, 1:], segs_k[int(batch_id)].unsqueeze(1)), dim=1)
            # np.savetxt("coords_k.csv", point_cloud_data.cpu().detach().numpy(), delimiter=",")

        padded_seg_feats_k = torch.stack(seg_feats_k_list, dim=0)
        batch_mask_k = torch.stack(mask_k_list, dim=0)

        # プロジェクションヘッドに入力
        proj_feats_k = proj_head_k(padded_seg_feats_k, enc_mask=batch_mask_k)
        proj_feats_k = F.normalize(proj_feats_k, dim=-1)

    # ロスを計算
    loss = calc_info_nce(pred_feats_q, proj_feats_k, batch_mask_q, batch_mask_k)
    # torch.cuda.empty_cache()
    return loss
    

from torch.optim.lr_scheduler import LambdaLR

class LambdaStepLR(LambdaLR):
  def __init__(self, optimizer, lr_lambda, last_step=-1):
    super(LambdaStepLR, self).__init__(optimizer, lr_lambda, last_step)

  @property
  def last_step(self):
    """Use last_epoch for the step counter"""
    return self.last_epoch

  @last_step.setter
  def last_step(self, v):
    self.last_epoch = v

class PolyLR(LambdaStepLR):
  """DeepLab learning rate policy"""
  def __init__(self, optimizer, max_iter=30000, power=0.9, last_step=-1):
    super(PolyLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1))**power, last_step)

def worker_init_fn(worker_id):
    if torch.cuda.device_count()-1 == 0:
        gpu_id = 0
    else:
        gpu_id = ( worker_id % (torch.cuda.device_count()-1)) + 1  # GPUをラウンドロビンで選択(GPU0はmodel用)
    torch.cuda.set_device(gpu_id)
    # WorkerごとにユニークなシードをNumPyに設定
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)

    return logger

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


if __name__ == '__main__':
    args = parse_args()
    if multiprocessing.get_start_method() == 'fork':
        multiprocessing.set_start_method('spawn', force=True)
    '''Setup logger'''
    if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
    logger = set_logger(os.path.join(args.save_path, 'train.log'))

    '''Random Seed'''
    seed = args.seed
    set_seed(seed)
    main(args, logger)
