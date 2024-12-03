import warnings
warnings.filterwarnings('ignore')
import argparse
import time
import os
import numpy as np
import random
from datasets.SemanticKITTI import KITTItrain, cfl_collate_fn, KITTItemporal, cfl_collate_fn_temporal
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


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(description='PyTorch Unsuper_3D_Seg')
    parser.add_argument('--data_path', type=str, default='/mnt/data/users/minesawa/semantickitti/growsp',
                        help='point cloud data path')
    parser.add_argument('--sp_path', type=str, default='/mnt/data/users/minesawa/semantickitti/growsp_sp',
                        help='initial sp path')
    parser.add_argument('--original_data_path', type=str, default='/mnt/data/dataset/semantickitti/dataset/sequences')
    parser.add_argument('--patchwork_path', type=str, default='/mnt/data/users/minesawa/semantickitti/patchwork')
    parser.add_argument('--save_path', type=str, default='/mnt/data/users/minesawa/semantickitti/growsp_model', help='model savepath')
    parser.add_argument('--load_path', default='/mnt/data/users/minesawa/semantickitti/growsp_model', type=str, help='model load path')
    parser.add_argument('--pseudo_label_path', default='pseudo_label_kitti/', type=str, help='pseudo label save path') # 同時に複数実行する場合のみ、被らないように変更する必要がある
    ##
    parser.add_argument('--max_epoch', type=int, nargs='+', default=[100, 350], help='max epoch for non-growing and growing stage')
    parser.add_argument('--max_iter', type=list, default=[10000, 30000], help='max iter for non-growing and growing stage')
    ###
    parser.add_argument('--bn_momentum', type=float, default=0.02, help='batchnorm parameters')
    parser.add_argument('--conv1_kernel_size', type=int, default=5, help='kernel size of 1st conv layers')
    ####
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--workers', type=int, default=16, help='how many workers for loading data')
    parser.add_argument('--temporal_workers', type=int, default=16, help='how many workers for loading data')
    parser.add_argument('--cluster_workers', type=int, default=16, help='how many workers for loading data in clustering')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--log-interval', type=int, default=80, help='log interval')
    parser.add_argument('--batch_size', type=int, nargs='+', default=[16, 8], help='batchsize in training[GrowSP, TARL]')
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
    parser.add_argument('--r_crop', type=float, default=50, help='cropping radius in training')
    parser.add_argument('--cluster_interval', type=int, default=10, help='cluster interval')
    parser.add_argument('--eval_interval', type=int, default=1, help='eval interval')
    parser.add_argument('--silhouette', action='store_true', help='more eval metrics for kmeans')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--scan_window', type=int, default=12, help='scan window size')
    parser.add_argument('--contrast_select_num', type=int, default=1500, help='contrastive select number')
    parser.add_argument('--contrast_lr', type=float, default=1e-3, help='contrastive learning rate')  # TARLでは0.0002
    parser.add_argument('--run_stage', type=int, default=0, help='Stage to train  0 or 1 or 2 (0=all)')
    parser.add_argument('--lmb', type=float, default=0.5, help='lambda for contrastive learning')
    parser.add_argument('--vis', action='store_true', help='visualize')
    return parser.parse_args()


def main(args, logger):
    
    run = wandb.init(
        project="TCUSS",
        config={
            "contrast_select_num": args.contrast_select_num,
            "contrast_lr": args.contrast_lr,
            "scan_window": args.scan_window,
            "select_num": args.select_num,
            "max_epoch_1": args.max_epoch[0],
            "max_epoch_2": args.max_epoch[1],
            "cluster_interval": args.cluster_interval,
            "batch_size_growsp": args.batch_size[0],
            "batch_size_tarl": args.batch_size[1],
            "lr": args.lr,
            "workers": args.workers,
            "temporal_workers": args.temporal_workers,
            "cluster_workers": args.cluster_workers,
            "run_stage": args.run_stage,
        },
        name = 'GrowSP-more-metric-poseidon',
    )
    
    '''Random select 1500 scans to train, will redo in each round'''
    scene_idx = np.random.choice(19130, args.select_num, replace=False).tolist()## SemanticKITTI totally has 19130 training samples
    
    trainset = KITTItrain(args, scene_idx, 'train')
    train_loader = DataLoader(trainset, batch_size=args.batch_size[0], shuffle=True, collate_fn=cfl_collate_fn(), num_workers=args.workers, pin_memory=True, worker_init_fn=worker_init_fn)

    clusterset = KITTItrain(args, scene_idx, 'train')
    cluster_loader = DataLoader(clusterset, batch_size=1, collate_fn=cfl_collate_fn(), num_workers=args.cluster_workers, pin_memory=True)

    if args.run_stage == 0 or args.run_stage == 1:
        '''Prepare Model/Optimizer'''
        model_q = Res16FPN18(in_channels=args.input_dim, out_channels=args.feats_dim, conv1_kernel_size=args.conv1_kernel_size, config=args)
        model_q = model_q.cuda()
        optimizer = torch.optim.AdamW(model_q.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.max_iter[0])
        loss = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
        start_grow_epoch = 0

        '''Train and Cluster'''
        '''Superpoints will not Grow in 1st Stage'''
        is_Growing = False
        for epoch in range(1, args.max_epoch[0]+1):

            '''Take 10 epochs as a round'''
            if (epoch-1) % args.cluster_interval==0:
                scene_idx = np.random.choice(19130, args.select_num, replace=False)
                train_loader.dataset.random_select_sample(scene_idx)
                cluster_loader.dataset.random_select_sample(scene_idx)

                classifier, current_growsp = cluster(args, logger, cluster_loader, model_q, epoch, start_grow_epoch, is_Growing)
            train(train_loader, logger, model_q, optimizer, loss, epoch, scheduler, classifier)
            torch.save(model_q.state_dict(), join(args.save_path,  'model_' + str(epoch) + '_checkpoint.pth'))
            torch.save(classifier.state_dict(), join(args.save_path, 'cls_' + str(epoch) + '_checkpoint.pth'))
            
            if epoch % args.eval_interval==0:
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

            # iterations = (epoch+10) * len(train_loader)
            # if iterations > args.max_iter[0]:
            #     # start_grow_epoch = epoch
            #     break
        
        torch.save({
            'start_grow_epoch': start_grow_epoch,
            'model_q_state_dict': model_q.state_dict(),
            'epoch': epoch,
            }, join(args.save_path, 'model_checkpoint_stage1.pth'))

    if args.run_stage == 0 or args.run_stage == 2:
        '''Superpoints will grow in 2nd Stage'''
        logger.info('#################################')
        logger.info('### Superpoints Begin Growing ###')
        logger.info('#################################')
        model_q = Res16FPN18(in_channels=args.input_dim, out_channels=args.feats_dim, conv1_kernel_size=args.conv1_kernel_size, config=args)
        if args.run_stage == 2:
            checkpoint = torch.load(join(args.load_path, 'model_checkpoint_stage1.pth'))
            start_grow_epoch = args.max_epoch[0]
        else:
            checkpoint = torch.load(join(args.save_path, 'model_checkpoint_stage1.pth'))
            start_grow_epoch = checkpoint['epoch']
        model_q.load_state_dict(checkpoint['model_q_state_dict'])
        model_q = model_q.cuda()
        is_Growing = True
        current_growsp = args.growsp_start
        optimizer = torch.optim.AdamW(model_q.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.max_epoch[1], steps_per_epoch=len(train_loader))
        loss = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
        
        # for contrastive learning
        proj_head_q = TransformerProjector(d_model=args.feats_dim, num_layer=1)
        model_k = Res16FPN18(in_channels=args.input_dim, out_channels=args.feats_dim, conv1_kernel_size=args.conv1_kernel_size, config=args)
        proj_head_k = TransformerProjector(d_model=args.feats_dim, num_layer=1)

        copy_minkowski_network_params(model_q, model_k)

        for param_q, param_k in zip(proj_head_q.parameters(), proj_head_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        predictor = TransformerProjector(d_model=args.feats_dim, num_layer=1)
        proj_head_q = proj_head_q.cuda()
        model_k = model_k.cuda()
        proj_head_k = proj_head_k.cuda()
        predictor = predictor.cuda()
        optimizer_contrast = torch.optim.AdamW(list(model_q.parameters())+list(proj_head_q.parameters())+list(predictor.parameters()), lr=args.contrast_lr)
        
        temporalset = KITTItemporal(args)
        temporal_loader = DataLoader(temporalset, batch_size=args.batch_size[1], shuffle=True, collate_fn=cfl_collate_fn_temporal(), num_workers=args.temporal_workers, pin_memory=True, worker_init_fn=worker_init_fn)
        scheduler_contrast = torch.optim.lr_scheduler.OneCycleLR(optimizer_contrast, max_lr=args.lr, epochs=args.max_epoch[1], steps_per_epoch=len(temporal_loader))

        for epoch in range(1, args.max_epoch[1]+1):
            epoch += start_grow_epoch

            '''Take 10 epochs as a round'''
            if (epoch-1) % args.cluster_interval==0:
                scene_idx = np.random.choice(19130, args.select_num, replace=False)
                train_loader.dataset.random_select_sample(scene_idx)
                cluster_loader.dataset.random_select_sample(scene_idx)

                classifier, current_growsp = cluster(args, logger, cluster_loader, model_q, epoch, start_grow_epoch, is_Growing)
                wandb.log({'epoch': epoch, 'current_growsp': current_growsp})
            # train_contrast(args, logger, temporal_loader, model_q, model_k, proj_head_q, proj_head_k, predictor, optimizer_contrast, epoch, scheduler_contrast, current_growsp)
            train(train_loader, logger, model_q, optimizer, loss, epoch, scheduler, classifier)
            momentum_update_model(model_q, model_k)
            torch.save(model_q.state_dict(), join(args.save_path,  'model_' + str(epoch) + '_checkpoint.pth'))
            torch.save(classifier.state_dict(), join(args.save_path, 'cls_' + str(epoch) + '_checkpoint.pth'))

            if epoch% args.eval_interval==0:
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
    return classifier.cuda(), current_growsp


def train_contrast(args, logger, temporal_loader, model_q, model_k, proj_head_q, proj_head_k, predictor, optimizer, epoch, scheduler, current_growsp):
    temporal_loader.dataset.n_clusters = current_growsp
    model_q.train()
    model_k.train()
    proj_head_q.train()
    proj_head_k.train()
    predictor.train()
    loss_display = 0.0
    time_curr = time.time()
    for batch_idx, data in tqdm(enumerate(temporal_loader), desc='Contrast Epoch: {}'.format(epoch)):

        coords_q, coords_k, segs_q, segs_k = data

        loss = train_contrast_half(coords_q, coords_k, segs_q, segs_k, model_q, model_k, proj_head_q, proj_head_k, predictor)
        loss += train_contrast_half(coords_k, coords_q, segs_k, segs_q, model_q, model_k, proj_head_q, proj_head_k, predictor)
        
        optimizer.zero_grad()
        loss_display += loss.item()
        loss = loss * args.lmb
        loss.backward()
        optimizer.step()
        scheduler.step()
        momentum_update_key_encoder(model_q, model_k, proj_head_q, proj_head_k)

        torch.cuda.empty_cache()
        torch.cuda.synchronize(torch.device("cuda"))
    wandb.log({'epoch': epoch, 'loss_contrast': loss_display, 'loss_contrast x lmb': loss_display*args.lmb})


def train_contrast_half(coords_q, coords_k, segs_q, segs_k, model_q, model_k, proj_head_q, proj_head_k, predictor):
    in_field_q = ME.TensorField(coords_q[:, 1:]*args.voxel_size, coords_q, device=0)
    feats_q = model_q(in_field_q)
    batch_ids = torch.unique(coords_q[:, 0])
    seg_feats_q = torch.empty((0, args.feats_dim), device=0)
    for batch_id in batch_ids:
        mask = coords_q[:, 0] == batch_id
        scene_feats_q = feats_q[mask]
        scene_seg_feats_q = compute_segment_feats(scene_feats_q, segs_q[int(batch_id)].cuda())
        seg_feats_q = torch.cat((seg_feats_q, scene_seg_feats_q), dim=0)
        
    proj_feats_q = proj_head_q(seg_feats_q.unsqueeze(1))
    pred_feats_q = predictor(proj_feats_q).squeeze(1)
        
    with torch.no_grad():
        in_field_k = ME.TensorField(coords_k[:, 1:]*args.voxel_size, coords_k, device=0)
        feats_k = model_k(in_field_k)
        batch_ids = torch.unique(coords_k[:, 0])
        seg_feats_k = torch.empty((0, args.feats_dim), device=0)
        for batch_id in batch_ids:
            mask = coords_k[:, 0] == batch_id
            scene_feats_k = feats_k[mask]
            scene_seg_feats_k = compute_segment_feats(scene_feats_k, segs_k[int(batch_id)].cuda())
            seg_feats_k = torch.cat((seg_feats_k, scene_seg_feats_k), dim=0)
            
        proj_feats_k = proj_head_k(seg_feats_k.unsqueeze(1)).squeeze(1)
        
    loss = calc_info_nce(pred_feats_q, proj_feats_k)
    torch.cuda.empty_cache()
    return loss


def train(train_loader, logger, model_q, optimizer=None, loss=None, epoch=None, scheduler=None, classifier=None):
    # train_loader.dataset.mode = 'train'
    model_q.train()
    loss_display = 0.0
    time_curr = time.time()
    for batch_idx, data in tqdm(enumerate(train_loader), desc='Train Epoch: {}'.format(epoch)):
        iteration = (epoch - 1) * len(train_loader) + batch_idx+1#从1开始

        coords, features, normals, labels, inverse_map, pseudo_labels, inds, region, index = data
        if args.vis:
            print('coords', coords)
        in_field = ME.TensorField(coords[:, 1:]*args.voxel_size, coords, device=0)
        feats = model_q(in_field)

        feats = feats[inds.long()]
        feats = F.normalize(feats, dim=-1)
        #
        pseudo_labels_comp = pseudo_labels.long().cuda()
        logits = F.linear(F.normalize(feats), F.normalize(classifier.weight))
        loss_sem = loss(logits * 5, pseudo_labels_comp).mean()

        loss_display += loss_sem.item()
        optimizer.zero_grad()
        loss_sem.backward()
        optimizer.step()
        scheduler.step()

        torch.cuda.empty_cache()
        torch.cuda.synchronize(torch.device("cuda"))

        if (batch_idx+1) % args.log_interval == 0:
            time_used = time.time() - time_curr
            # loss_display /= args.log_interval # TODO: ここで割るのは合っているか？
            logger.info(
                'Train Epoch: {} [{}/{} ({:.0f}%)]{}, Loss: {:.10f}, lr: {:.3e}, Elapsed time: {:.4f}s({} iters)'.format(
                    epoch, (batch_idx+1), len(train_loader), 100. * (batch_idx+1) / len(train_loader),
                    iteration, loss_display, scheduler.get_lr()[0], time_used, args.log_interval))
            time_curr = time.time()
            # loss_display = 0
    wandb.log({'epoch': epoch, 'train_loss': loss_display})
        

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
