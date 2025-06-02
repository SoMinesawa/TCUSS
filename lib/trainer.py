import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import MinkowskiEngine as ME
import numpy as np
import time
import os
import wandb
import logging
from tqdm import tqdm
from math import ceil
from typing import Dict, List, Tuple, Optional, Union, Any
from torch.utils.data import DataLoader
from os.path import join

from models.fpn import Res16FPN18, Res16FPNBase
from models.transformer_projector import TransformerProjector
from lib.config import TCUSSConfig
from lib.utils import (
    get_pseudo_kitti, get_kittisp_feature, get_fixclassifier, copy_minkowski_network_params,
    compute_segment_feats, momentum_update_key_encoder, calc_info_nce, get_kmeans_labels,
    calc_cluster_metrics
)
from eval_SemanticKITTI import eval


class TCUSSTrainer:
    """TCUSSのトレーニングプロセスを管理するクラス"""
    
    def __init__(self, config: TCUSSConfig, logger: logging.Logger):
        """
        トレーナーの初期化
        
        Args:
            config: トレーニング設定
            logger: ロガー
        """
        self.config = config
        self.logger = logger
        
        # モデルの初期化
        self.model_q = Res16FPN18(
            in_channels=config.input_dim, 
            out_channels=config.feats_dim, 
            conv1_kernel_size=config.conv1_kernel_size, 
            config=config
        ).to("cuda:0")
        
        self.model_k = Res16FPN18(
            in_channels=config.input_dim, 
            out_channels=config.feats_dim, 
            conv1_kernel_size=config.conv1_kernel_size, 
            config=config
        ).to("cuda:0")
        
        self.proj_head_q = TransformerProjector(d_model=config.feats_dim, num_layer=1).to("cuda:0")
        self.proj_head_k = TransformerProjector(d_model=config.feats_dim, num_layer=1).to("cuda:0")
        self.predictor = TransformerProjector(d_model=config.feats_dim, num_layer=1).to("cuda:0")
        
        # モデルパラメータの初期化
        copy_minkowski_network_params(self.model_q, self.model_k)
        for param_q, param_k in zip(self.proj_head_q.parameters(), self.proj_head_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # オプティマイザーとスケジューラーは後で初期化する
        self.optimizer = None
        self.schedulers = None
        self.classifier = None
        self.current_growsp = None
        self.resume_epoch = None
        
    def setup_optimizer(self):
        """オプティマイザの設定"""
        # パラメータグループを分けて異なる学習率を設定
        backbone_params = list(self.model_q.parameters())
        transformer_params = list(self.proj_head_q.parameters()) + list(self.predictor.parameters())
        
        param_groups = [
            {'params': backbone_params, 'lr': self.config.lr},
            {'params': transformer_params, 'lr': self.config.tarl_lr}
        ]
        
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=self.config.weight_decay)
    
    def setup_schedulers(self, train_loader_length: int):
        """スケジューラの設定"""
        steps_per_epoch = ceil(train_loader_length / self.config.accum_step)
        self.schedulers = [
            torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, 
                max_lr=[self.config.lr, self.config.tarl_lr], 
                epochs=epoch, 
                steps_per_epoch=steps_per_epoch
            ) for epoch in self.config.max_epoch
        ]
    
    def init_wandb(self):
        """Weights & Biasesの初期化"""
        run = wandb.init(
            project="TCUSS",
            config=vars(self.config),
            name=self.config.name if self.config.name else None,
            resume='must' if self.config.resume else 'never',
            id=self.config.wandb_run_id if self.config.resume else None,
            settings=wandb.Settings(code_dir=".")
        )
        return run
    
    def resume_from_checkpoint(self, phase: int):
        """チェックポイントから再開"""
        if not self.config.resume:
            return None
            
        last_epoch = wandb.run.summary.get("epoch", 0)
        self.resume_epoch = ((last_epoch-1) // self.config.cluster_interval) * self.config.cluster_interval + 1
        self.logger.info(f'エポック {self.resume_epoch} から再開します')
        
        # 指定されたエポックのチェックポイントを読み込む
        checkpoint_path = join(self.config.save_path, f'checkpoint_epoch_{self.resume_epoch-1}.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model_q.load_state_dict(checkpoint['model_q_state_dict'])
            self.model_k.load_state_dict(checkpoint['model_k_state_dict'])
            self.proj_head_q.load_state_dict(checkpoint['proj_head_q_state_dict'])
            self.proj_head_k.load_state_dict(checkpoint['proj_head_k_state_dict'])
            self.predictor.load_state_dict(checkpoint['predictor_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 現在のフェーズのスケジューラー状態を読み込む
            if f'scheduler_{phase}_state_dict' in checkpoint:
                self.schedulers[phase].load_state_dict(checkpoint[f'scheduler_{phase}_state_dict'])
            
            # 乱数状態を復元
            if 'np_random_state' in checkpoint:
                np.random.set_state(checkpoint['np_random_state'])
            if 'torch_random_state' in checkpoint:
                torch.set_rng_state(checkpoint['torch_random_state'])
            if torch.cuda.is_available() and 'torch_cuda_random_state' in checkpoint and checkpoint['torch_cuda_random_state'] is not None:
                torch.cuda.set_rng_state(checkpoint['torch_cuda_random_state'])
            
            return self.resume_epoch
        else:
            self.logger.warning(f'チェックポイントファイル {checkpoint_path} が見つかりません')
            return None
    
    def save_checkpoint(self, epoch: int, phase: int):
        """チェックポイントの保存"""
        checkpoint = {
            'epoch': epoch,
            'model_q_state_dict': self.model_q.state_dict(),
            'model_k_state_dict': self.model_k.state_dict(),
            'proj_head_q_state_dict': self.proj_head_q.state_dict(),
            'proj_head_k_state_dict': self.proj_head_k.state_dict(),
            'predictor_state_dict': self.predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            f'scheduler_{phase}_state_dict': self.schedulers[phase].state_dict(),
            'np_random_state': np.random.get_state(),
            'torch_random_state': torch.get_rng_state(),
            'torch_cuda_random_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        }
        
        torch.save(checkpoint, join(self.config.save_path, f'checkpoint_epoch_{epoch}.pth'))
    
    def train(self, train_loader: DataLoader, cluster_loader: DataLoader):
        """モデルのトレーニングメイン関数"""
        # Weights & Biasesの初期化
        run = self.init_wandb()
        
        # オプティマイザとスケジューラの設定
        self.setup_optimizer()
        self.setup_schedulers(len(train_loader))
        
        # モデルの更新
        momentum_update_key_encoder(self.model_q, self.model_k, self.proj_head_q, self.proj_head_k)
        
        # トレーニング開始
        is_growing = False
        for i, (epoch, scheduler) in enumerate(zip(self.config.max_epoch, self.schedulers)):
            train_loader.dataset.phase = i
            self.train_phase(i, train_loader, cluster_loader, is_growing)
            is_growing = True
    
    def train_phase(self, phase: int, train_loader: DataLoader, cluster_loader: DataLoader, is_growing: bool):
        """各フェーズのトレーニング"""
        # 開始・終了エポックの設定
        start_epoch = 0 if phase == 0 else self.config.max_epoch[0]
        end_epoch = self.config.max_epoch[0] if phase == 0 else sum(self.config.max_epoch)
        
        # チェックポイントからの再開
        resume_epoch = self.resume_from_checkpoint(phase) if self.config.resume else None
        
        for epoch in range(start_epoch + 1, end_epoch + 1):
            # 再開する場合、指定したエポックまでスキップ
            if resume_epoch and epoch < resume_epoch:
                continue
            
            # クラスタリングの実行
            if (epoch - 1) % self.config.cluster_interval == 0:
                train_loader.dataset.random_select_sample()
                scene_idx = train_loader.dataset.scene_idx_all
                cluster_loader.dataset.random_select_sample(scene_idx)
                self.classifier, self.current_growsp = self.cluster(cluster_loader, epoch, self.config.max_epoch[0], is_growing)
            
            # データセットにクラスタ数を設定
            train_loader.dataset.kittitemporal.n_clusters = self.current_growsp
            
            # 1エポックのトレーニング
            self.train_epoch(train_loader, epoch, phase)
            
            # 評価
            if epoch % self.config.eval_interval == 0:
                self.evaluate(epoch)
            
            # チェックポイントの保存
            if epoch % self.config.cluster_interval == 0:
                self.save_checkpoint(epoch, phase)
    
    def train_epoch(self, train_loader: DataLoader, epoch: int, phase: int):
        """1エポックのトレーニング"""
        # モデルをトレーニングモードに設定
        self.model_q.train()
        self.model_k.train()
        self.proj_head_q.train()
        self.proj_head_k.train()
        self.predictor.train()
        
        # オプティマイザのリセット
        self.optimizer.zero_grad()
        
        # 損失の表示用変数
        loss_growsp_display = 0.0
        loss_tarl_display = 0.0
        
        # バッチごとのトレーニング
        for i, data in tqdm(enumerate(train_loader), desc=f'トレーニングエポック: {epoch}', total=len(train_loader)):
            growsp_t1_data, growsp_t2_data, tarl_data = data
            
            # GrowSP損失の計算
            growsp_t1_loss = self.train_growsp(growsp_t1_data)
            growsp_t2_loss = self.train_growsp(growsp_t2_data)
            
            # TARL損失の計算
            if tarl_data is not None:
                tarl_loss = self.train_tarl(tarl_data) / self.config.accum_step
            else:
                tarl_loss = torch.tensor(0.0, device="cuda")
            
            # 合計損失の計算
            growsp_loss = (growsp_t1_loss + growsp_t2_loss) / self.config.accum_step
            loss = growsp_loss + tarl_loss
            
            # 損失の表示用に加算
            loss_growsp_display += growsp_loss.item()
            loss_tarl_display += tarl_loss.item()
            
            # バックプロパゲーション
            loss.backward()
            
            # 勾配の蓄積ステップに達したか、最後のバッチの場合
            if ((i+1) % self.config.accum_step == 0) or (i == len(train_loader)-1):
                # パラメータの更新
                self.optimizer.step()
                
                # 学習率のログ記録
                wandb.log({
                    'epoch': epoch, 
                    'backbone_lr': self.optimizer.param_groups[0]['lr'], 
                    'transformer_lr': self.optimizer.param_groups[1]['lr']
                })
                
                # スケジューラの更新
                if self.schedulers[phase] is not None:
                    self.schedulers[phase].step()
                
                # オプティマイザのリセット
                self.optimizer.zero_grad()
                
                # モメンタムモデルの更新
                momentum_update_key_encoder(self.model_q, self.model_k, self.proj_head_q, self.proj_head_k)
                
                # メモリの解放
                torch.cuda.empty_cache()
                torch.cuda.synchronize(torch.device("cuda"))
        
        # エポック全体の損失をログに記録
        wandb.log({'epoch': epoch, 'loss_growsp': loss_growsp_display, 'loss_tarl': loss_tarl_display})
    
    def train_growsp(self, growsp_data):
        """GrowSPのトレーニング"""
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.config.ignore_label).to("cuda:0")
        coords, pseudo_labels, inds = growsp_data
        
        # 入力フィールドの作成
        in_field = ME.TensorField(coords[:, 1:] * self.config.voxel_size, coords, device=0)
        
        # 特徴抽出
        feats = self.model_q(in_field)
        feats = feats[inds.long()]
        feats = F.normalize(feats, dim=-1)
        
        # 擬似ラベルの準備
        pseudo_labels_comp = pseudo_labels.long().to("cuda:0")
        
        # ロジットの計算
        logits = F.linear(F.normalize(feats), F.normalize(self.classifier.weight))
        
        # 損失計算
        loss_sem = loss_fn(logits * 5, pseudo_labels_comp).mean()
        return loss_sem
    
    def train_tarl(self, tarl_data):
        """TARLのトレーニング"""
        coords_q, coords_k, segs_q, segs_k = tarl_data
        
        # 順方向と逆方向の損失を計算して合計
        loss = self.train_contrast_half(coords_q, coords_k, segs_q, segs_k)
        loss += self.train_contrast_half(coords_k, coords_q, segs_k, segs_q)
        
        return loss
    
    def train_contrast_half(self, coords_q, coords_k, segs_q, segs_k):
        """コントラスト学習の半分（一方向）"""
        # クエリモデルでの特徴抽出
        in_field_q = ME.TensorField(coords_q[:, 1:] * self.config.voxel_size, coords_q, device=0)
        feats_q = self.model_q(in_field_q)
        
        # バッチIDの取得
        batch_ids = torch.unique(coords_q[:, 0])
        seg_feats_q_list, mask_q_list = [], []
        
        # バッチごとの処理
        for batch_id in batch_ids:
            mask = coords_q[:, 0] == batch_id
            scene_feats_q = feats_q[mask]
            scene_segs_q = segs_q[int(batch_id)].to("cuda:0")
            scene_seg_feats_q, mask_q = compute_segment_feats(scene_feats_q, scene_segs_q, max_seg_num=self.current_growsp)
            seg_feats_q_list.append(scene_seg_feats_q)
            mask_q_list.append(mask_q)
        
        # 特徴とマスクのスタック
        padded_seg_feats_q = torch.stack(seg_feats_q_list, dim=0)
        batch_mask_q = torch.stack(mask_q_list, dim=0)
        
        # プロジェクションヘッドに入力
        proj_feats_q = self.proj_head_q(padded_seg_feats_q, enc_mask=batch_mask_q)
        
        # プレディクターに入力
        pred_feats_q = self.predictor(proj_feats_q, enc_mask=batch_mask_q)
        pred_feats_q = F.normalize(pred_feats_q, dim=-1)
        
        # キーモデルでの特徴抽出（勾配計算なし）
        with torch.no_grad():
            in_field_k = ME.TensorField(coords_k[:, 1:] * self.config.voxel_size, coords_k, device=0)
            feats_k = self.model_k(in_field_k)
            
            # バッチIDの取得
            batch_ids = torch.unique(coords_k[:, 0])
            seg_feats_k_list, mask_k_list = [], []
            
            # バッチごとの処理
            for batch_id in batch_ids:
                mask = coords_k[:, 0] == batch_id
                scene_feats_k = feats_k[mask]
                scene_segs_k = segs_k[int(batch_id)].to("cuda:0")
                scene_seg_feats_k, mask_k = compute_segment_feats(
                    scene_feats_k, scene_segs_k, max_seg_num=self.current_growsp
                )
                seg_feats_k_list.append(scene_seg_feats_k)
                mask_k_list.append(mask_k)
            
            # 特徴とマスクのスタック
            padded_seg_feats_k = torch.stack(seg_feats_k_list, dim=0)
            batch_mask_k = torch.stack(mask_k_list, dim=0)
            
            # プロジェクションヘッドに入力
            proj_feats_k = self.proj_head_k(padded_seg_feats_k, enc_mask=batch_mask_k)
            proj_feats_k = F.normalize(proj_feats_k, dim=-1)
        
        # InfoNCE損失の計算
        loss = calc_info_nce(pred_feats_q, proj_feats_k, batch_mask_q, batch_mask_k)
        return loss
    
    def cluster(self, cluster_loader: DataLoader, epoch: int, start_grow_epoch: Optional[int] = None, is_growing: bool = False):
        """クラスタリングの実行"""
        time_start = time.time()
        cluster_loader.dataset.mode = 'cluster'
        
        # GrowSPのクラスタ数計算
        current_growsp = None
        if is_growing:
            current_growsp = int(self.config.growsp_start - ((epoch - start_grow_epoch)/self.config.max_epoch[1])*(self.config.growsp_start - self.config.growsp_end))
            if current_growsp < self.config.growsp_end:
                current_growsp = self.config.growsp_end
            self.logger.info(f'エポック: {epoch}, スーパーポイントが {current_growsp} に成長')
        
        # スーパーポイント特徴の抽出
        feats, labels, sp_index, context = get_kittisp_feature(
            self.config, cluster_loader, self.model_q, current_growsp, epoch
        )
        sp_feats = torch.cat(feats, dim=0)
        
        # セマンティックプリミティブクラスタリング (SPC)
        primitive_labels = get_kmeans_labels(self.config.primitive_num, sp_feats).to('cpu').detach().numpy()
        
        # 幾何学的特徴を削除
        sp_feats = sp_feats[:, :self.config.feats_dim]
        
        # プリミティブセンターの計算
        primitive_centers = torch.zeros((self.config.primitive_num, self.config.feats_dim))
        for cluster_idx in range(self.config.primitive_num):
            indices = primitive_labels == cluster_idx
            if indices.sum() > 0:  # クラスタにサンプルがある場合のみ
                cluster_avg = sp_feats[indices].mean(0, keepdims=True)
                primitive_centers[cluster_idx] = cluster_avg
        primitive_centers = F.normalize(primitive_centers, dim=1)
        
        # 分類器の作成
        classifier = get_fixclassifier(
            in_channel=self.config.feats_dim, 
            centroids_num=self.config.primitive_num, 
            centroids=primitive_centers
        )
        
        # 疑似ラベルの計算と保存
        all_pseudo, all_gt, all_pseudo_gt = get_pseudo_kitti(
            self.config, context, primitive_labels, sp_index
        )
        self.logger.info(
            'ラベル付けされたポイントの割合 %.2f クラスタリング時間: %.2fs', 
            (all_pseudo != -1).sum() / all_pseudo.shape[0], 
            time.time() - time_start
        )
        
        # トレーニング中のスーパーポイント/プリミティブ精度のチェック
        sem_num = self.config.semantic_class
        mask = (all_pseudo_gt != -1)
        histogram = np.bincount(
            sem_num * all_gt.astype(np.int32)[mask] + all_pseudo_gt.astype(np.int32)[mask], 
            minlength=sem_num ** 2
        ).reshape(sem_num, sem_num)
        
        # 全体精度の計算
        o_Acc = histogram[range(sem_num), range(sem_num)].sum() / histogram.sum() * 100
        
        # IoUの計算
        tp = np.diag(histogram)
        fp = np.sum(histogram, 0) - tp
        fn = np.sum(histogram, 1) - tp
        IoUs = tp / (tp + fp + fn + 1e-8)
        m_IoU = np.nanmean(IoUs)
        
        # IoUの文字列表現
        s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
        for IoU in IoUs:
            s += '{:5.2f} '.format(100 * IoU)
        
        # 結果のログ記録
        self.logger.info('クラスタリング結果 - エポック: {:02d}, オール精度 {:.2f} mIoU {:.2f}'.format(epoch, o_Acc, 100 * m_IoU))
        wandb.log({
            'epoch': epoch, 
            'cluster/oAcc': o_Acc, 
            'cluster/mIoU': m_IoU
        })
        
        return classifier.to("cuda:0"), current_growsp
    
    def evaluate(self, epoch: int):
        """モデルの評価"""
        # モデルの保存
        torch.save(
            self.model_q.state_dict(), 
            join(self.config.save_path, f'model_{epoch}_checkpoint.pth')
        )
        torch.save(
            self.classifier.state_dict(), 
            join(self.config.save_path, f'cls_{epoch}_checkpoint.pth')
        )
        
        # 評価の実行
        with torch.no_grad():
            o_Acc, m_Acc, m_IoU, s, IoU_dict = eval(epoch, self.config)
            self.logger.info('エポック: {:02d}, oAcc {:.2f}  mAcc {:.2f} IoUs'.format(epoch, o_Acc, m_Acc) + s)
            
            # 結果のログ記録
            d = {'epoch': epoch, 'oAcc': o_Acc, 'mAcc': m_Acc, 'mIoU': m_IoU}
            d.update(IoU_dict)
            wandb.log(d)
            
            # シルエットスコアの計算（オプション）
            if self.config.silhouette:
                # SPCの評価
                feats, *_ = get_kittisp_feature(
                    self.config, self.cluster_loader, self.model_q, self.current_growsp, epoch
                )
                sp_feats = torch.cat(feats, dim=0)
                primitive_labels = get_kmeans_labels(
                    self.config.primitive_num, sp_feats
                ).to('cpu').detach().numpy()
                
                # クラスタリングメトリクスの計算
                sl_score, db_score, ch_score, t = calc_cluster_metrics(sp_feats, primitive_labels)
                
                # メトリクスのログ記録
                wandb.log({
                    'epoch': epoch, 
                    'SPC/Silhouette': sl_score, 
                    'SPC/Davies-Bouldin': db_score, 
                    'SPC/Calinski-Harabasz': ch_score, 
                    'SPC/time': t
                }) 