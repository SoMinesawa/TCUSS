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
from lib.stc_loss import compute_sp_features, loss_stc_similarity
from lib.utils import get_kmeans_labels
from lib.scene_flow_prefetcher import SceneFlowPrefetcher
from eval_SemanticKITTI import eval
import sys


def save_vis_sp_ply(
    points: np.ndarray,
    sp_labels: np.ndarray,
    sp_colors: Dict[int, Tuple[int, int, int]],
    filepath: str,
    default_color: Tuple[int, int, int] = (128, 128, 128)
):
    """
    Superpointで色付けした点群をASCII PLYで保存
    
    Args:
        points: 点群座標 [N, 3]
        sp_labels: 各点のSuperpointラベル [N]
        sp_colors: SPラベル -> (R, G, B)のマッピング
        filepath: 保存先パス
        default_color: 対応がないSPの色（デフォルト: グレー）
    """
    N = len(points)
    
    # 各点に色を割り当て
    colors = np.zeros((N, 3), dtype=np.uint8)
    labels = sp_labels.copy()
    
    for i in range(N):
        sp = sp_labels[i]
        if sp in sp_colors:
            colors[i] = sp_colors[sp]
        else:
            colors[i] = default_color
    
    # ASCII PLYとして保存
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property int label\n")
        f.write("end_header\n")
        
        for i in range(N):
            f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                   f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]} {labels[i]}\n")


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
        # VoteFlowラウンドロビン用カウンタ
        self.vf_rr_idx = 0
        
        # モデルパラメータの初期化
        copy_minkowski_network_params(self.model_q, self.model_k)
        for param_q, param_k in zip(self.proj_head_q.parameters(), self.proj_head_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # デバイス確認ログ
        self.logger.info(f'model_q device: {next(self.model_q.parameters()).device}')
        
        # VoteFlowの初期化（STC用、推論のみ）: GPU0以外を優先して複数デバイスにラウンドロビン
        self.voteflow = None  # 後方互換用
        self.voteflow_wrappers = []
        if config.stc.enabled:
            self.logger.info(f'VoteFlowを初期化中: {config.stc.scene_flow.checkpoint}')
            try:
                from scene_flow.voteflow_wrapper import VoteFlowWrapper
                device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
                devices = []
                if device_count > 1:
                    devices = [f'cuda:{i}' for i in range(device_count) if i != 0]
                if not devices:
                    devices = ['cuda:0']
                self.logger.info(f'VoteFlow使用デバイス: {devices}')
                for dev in devices:
                    wrapper = VoteFlowWrapper(
                        checkpoint_path=config.stc.scene_flow.checkpoint,
                        voxel_size=config.stc.scene_flow.voxel_size,
                        point_cloud_range=config.stc.scene_flow.point_cloud_range,
                        device=dev
                    )
                    self.voteflow_wrappers.append(wrapper)
                if self.voteflow_wrappers:
                    self.voteflow = self.voteflow_wrappers[0]
                    self.logger.info(f'VoteFlow初期化完了: {len(self.voteflow_wrappers)} デバイス（並列計算対応）')
                else:
                    raise RuntimeError('VoteFlowWrapperが初期化できませんでした')
            except Exception as e:
                self.logger.error(f'VoteFlow初期化失敗: {e}')
                self.logger.warning('STCを無効化して続行します')
                config.stc.enabled = False
        
        # Scene Flowプリフェッチャーの初期化（STC用、先行計算）
        # 各GPUは1バッチの全サンプルを担当、プリフェッチバッチ数 = GPU数 × 2
        self.scene_flow_prefetcher = None
        if config.stc.enabled and len(self.voteflow_wrappers) > 0:
            n_voteflow_gpus = len(self.voteflow_wrappers)
            # デフォルト: GPU数 × 2（設定があればそれを使用）
            prefetch_batches = getattr(config.stc, 'prefetch_batches', None)
            if prefetch_batches is None:
                prefetch_batches = n_voteflow_gpus * 2
            self.scene_flow_prefetcher = SceneFlowPrefetcher(
                voteflow_wrappers=self.voteflow_wrappers,
                prefetch_batches=prefetch_batches,
                logger=self.logger
            )
            self.logger.info(
                f'SceneFlowPrefetcher初期化完了: '
                f'{n_voteflow_gpus} GPUs × 2 = {prefetch_batches} バッチをプリフェッチ'
            )
        
        # オプティマイザーとスケジューラーは後で初期化する
        self.optimizer = None
        self.schedulers = None
        self.classifier = None
        self.current_growsp = None
        self.resume_epoch = None
        
        # STC用: 統合後SPラベルのキャッシュ（scene_name -> sp_labels）
        self.sp_index_cache: Dict[str, np.ndarray] = {}
        
        # 早期停止関連の変数
        self.best_metric_score = float('-inf')  # best_val
        self.patience_counter = 0  # wait
        self.early_stopped = False
        self.best_epoch = 0
        self.loss_history = []  # train_lossの履歴
    
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
            
            # early stopping状態の復元
            if 'best_metric_score' in checkpoint:
                self.best_metric_score = checkpoint['best_metric_score']
                self.patience_counter = checkpoint['patience_counter']
                self.early_stopped = checkpoint['early_stopped']
                self.best_epoch = checkpoint['best_epoch']
                self.loss_history = checkpoint['loss_history']
                self.logger.info(f'early stopping状態を復元: best_score={self.best_metric_score:.4f}, '
                                f'patience={self.patience_counter}, best_epoch={self.best_epoch}')
            
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
            'torch_cuda_random_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            # early stopping状態の保存
            'best_metric_score': self.best_metric_score,
            'patience_counter': self.patience_counter,
            'early_stopped': self.early_stopped,
            'best_epoch': self.best_epoch,
            'loss_history': self.loss_history
        }
        
        torch.save(checkpoint, join(self.config.save_path, f'checkpoint_epoch_{epoch}.pth'))
    
    def train(self, train_loader: DataLoader, cluster_loader: DataLoader):
        """モデルのトレーニングメイン関数"""
        # vis時はwandbをdisableに設定
        if self.config.vis:
            os.environ["WANDB_MODE"] = "disabled"

        # Weights & Biasesの初期化
        _ = self.init_wandb()
        
        # オプティマイザとスケジューラの設定
        self.setup_optimizer()
        self.setup_schedulers(len(train_loader))
        
        # モデルの更新
        momentum_update_key_encoder(self.model_q, self.model_k, self.proj_head_q, self.proj_head_k)
        
        # トレーニング開始
        is_growing = False
        for i, (epoch, scheduler) in enumerate(zip(self.config.max_epoch, self.schedulers)):
            train_loader.dataset.phase = i # iではなくフェーズを0に固定すれば、それだけでGrowSPのみのトレーニングになる（今のコードでは違うけど）
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
            
            # データセットにクラスタ数を設定（STC/TARLでデータセットが異なるため属性チェック）
            if hasattr(train_loader.dataset, 'kittitemporal'):
                train_loader.dataset.kittitemporal.n_clusters = self.current_growsp
            if hasattr(train_loader.dataset, 'kittistc'):
                # STCデータセット側も必要なら持つようにする（存在しない場合はスキップ）
                if hasattr(train_loader.dataset.kittistc, 'n_clusters'):
                    train_loader.dataset.kittistc.n_clusters = self.current_growsp
            
            # 1エポックのトレーニング
            self.train_epoch(train_loader, epoch, phase)
            
            # 評価
            if epoch % self.config.eval_interval == 0:
                self.evaluate(epoch)
                
            # 早期停止チェック
            if self.early_stopped:
                self.logger.info(f'早期停止によりトレーニングを終了します (エポック {epoch})')
                break
            
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
        loss_temporal_display = 0.0
        
        # タイミング計測用（最初の10バッチのみ詳細計測）
        timing_stats = {
            'dataloader': [], 'growsp_t1': [], 'growsp_t2': [], 
            'stc_total': [], 'backward': [], 'optimizer': [], 'cuda_sync': []
        }
        
        # プリフェッチャーを使用するかどうか
        # Phase 0（Phase 1）ではSTC lossを計算しないのでプリフェッチャー（VoteFlow計算）も不要
        use_prefetcher = (
            self.config.stc.enabled and 
            self.scene_flow_prefetcher is not None and
            phase != 0  # Phase 1ではVoteFlow計算を省略
        )
        
        if use_prefetcher:
            # プリフェッチャーを開始（DataLoaderを渡す）
            self.scene_flow_prefetcher.start(train_loader, len(train_loader))
            self.logger.info(f'SceneFlowPrefetcher開始: バッチ先読み中...')
        else:
            # 従来通りDataLoaderイテレータを使用
            dataloader_iter = iter(train_loader)
        
        t_dataloader_start = time.perf_counter()
        
        for i in tqdm(range(len(train_loader)), desc=f'トレーニングエポック: {epoch}'):
            # === DataLoader/Prefetch時間計測 ===
            if use_prefetcher:
                # プリフェッチャーから事前計算済みのデータを取得
                result = self.scene_flow_prefetcher.get_batch(i, timeout=120.0)
                if result is None:
                    self.logger.warning(f'Prefetch failed for batch {i}, skipping')
                    continue
                growsp_t1_data, growsp_t2_data, tarl_data, prefetched_flow_results = result
            else:
                # 従来通りDataLoaderから取得
                data = next(dataloader_iter)
                growsp_t1_data, growsp_t2_data, tarl_data = data
                prefetched_flow_results = None
            
            torch.cuda.synchronize()  # GPU処理完了を待つ
            t_dataloader_end = time.perf_counter()
            
            # === GrowSP t1 時間計測 ===
            t_growsp_t1_start = time.perf_counter()
            growsp_t1_loss = self.train_growsp(growsp_t1_data)
            torch.cuda.synchronize()
            t_growsp_t1_end = time.perf_counter()
            
            # === GrowSP t2 時間計測 ===
            t_growsp_t2_start = time.perf_counter()
            growsp_t2_loss = self.train_growsp(growsp_t2_data)
            torch.cuda.synchronize()
            t_growsp_t2_end = time.perf_counter()
            
            # === STC/TARL 時間計測 ===
            t_stc_start = time.perf_counter()
            if self.config.stc.enabled:
                # STCモード
                # Phase 0（Phase 1）ではSTC lossを計算しない（VoteFlowやマッチングも省略）
                if phase == 0:
                    temporal_loss = torch.tensor(0.0, device="cuda")
                elif tarl_data is not None:
                    temporal_loss = self.train_stc(
                        tarl_data, 
                        phase=phase,
                        current_growsp=self.current_growsp,
                        profile=(i < 10),
                        prefetched_flow_results=prefetched_flow_results
                    ) / self.config.accum_step
                else:
                    temporal_loss = torch.tensor(0.0, device="cuda")
                temporal_weight = self.config.stc.weight
            else:
                # TARLモード
                if tarl_data is not None:
                    temporal_loss = self.train_tarl(tarl_data) / self.config.accum_step
                else:
                    temporal_loss = torch.tensor(0.0, device="cuda")
                temporal_weight = self.config.lmb
            torch.cuda.synchronize()
            t_stc_end = time.perf_counter()
            
            # 合計損失の計算
            growsp_loss = (growsp_t1_loss + growsp_t2_loss) / self.config.accum_step
            loss = growsp_loss + temporal_weight * temporal_loss
            
            # 損失の表示用に加算
            loss_growsp_display += growsp_loss.item()
            loss_temporal_display += temporal_loss.item() if isinstance(temporal_loss, torch.Tensor) else temporal_loss
            
            # === Backward 時間計測 ===
            t_backward_start = time.perf_counter()
            if loss.grad_fn is not None:
                loss.backward()
            torch.cuda.synchronize()
            t_backward_end = time.perf_counter()
            
            # 勾配の蓄積ステップに達したか、最後のバッチの場合
            if ((i+1) % self.config.accum_step == 0) or (i == len(train_loader)-1):
                # === Optimizer 時間計測 ===
                t_optimizer_start = time.perf_counter()
                self.optimizer.step()
                torch.cuda.synchronize()
                t_optimizer_end = time.perf_counter()
                
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
                
                # === CUDA同期・メモリ解放 時間計測 ===
                t_sync_start = time.perf_counter()
                torch.cuda.empty_cache()
                torch.cuda.synchronize(torch.device("cuda"))
                t_sync_end = time.perf_counter()
            else:
                t_optimizer_end = t_backward_end
                t_sync_end = t_backward_end
            
            # タイミング統計を記録（最初の10バッチ）
            if i < 10:
                timing_stats['dataloader'].append(t_dataloader_end - t_dataloader_start)
                timing_stats['growsp_t1'].append(t_growsp_t1_end - t_growsp_t1_start)
                timing_stats['growsp_t2'].append(t_growsp_t2_end - t_growsp_t2_start)
                timing_stats['stc_total'].append(t_stc_end - t_stc_start)
                timing_stats['backward'].append(t_backward_end - t_backward_start)
                timing_stats['optimizer'].append(t_optimizer_end - t_backward_end)
                timing_stats['cuda_sync'].append(t_sync_end - t_optimizer_end)
                
                total_time = t_sync_end - t_dataloader_start
                self.logger.info(
                    f'[PROFILE batch {i}] total={total_time:.3f}s | '
                    f'dataloader={t_dataloader_end - t_dataloader_start:.3f}s | '
                    f'growsp_t1={t_growsp_t1_end - t_growsp_t1_start:.3f}s | '
                    f'growsp_t2={t_growsp_t2_end - t_growsp_t2_start:.3f}s | '
                    f'stc={t_stc_end - t_stc_start:.3f}s | '
                    f'backward={t_backward_end - t_backward_start:.3f}s | '
                    f'optimizer={t_optimizer_end - t_backward_end:.3f}s | '
                    f'cuda_sync={t_sync_end - t_optimizer_end:.3f}s'
                )
            
            # 次のイテレーション用にDataLoader開始時間を更新
            t_dataloader_start = time.perf_counter()
        
        # プリフェッチャーを停止
        if use_prefetcher:
            prefetch_stats = self.scene_flow_prefetcher.get_stats()
            self.scene_flow_prefetcher.stop()
            self.logger.info(
                f'[PREFETCH STATS] prefetched={prefetch_stats["prefetched"]}, '
                f'cache_hits={prefetch_stats["cache_hits"]}, '
                f'wait_time={prefetch_stats["wait_time"]:.2f}s, '
                f'compute_time={prefetch_stats["compute_time"]:.2f}s'
            )
        
        # タイミング統計のサマリーを出力
        if timing_stats['dataloader']:
            self.logger.info('='*80)
            self.logger.info('[PROFILE SUMMARY] 最初の10バッチの平均時間:')
            for key, values in timing_stats.items():
                if values:
                    avg = sum(values) / len(values)
                    self.logger.info(f'  {key}: {avg:.3f}s ({avg/sum(sum(v) for v in timing_stats.values() if v)*100:.1f}%)')
            self.logger.info('='*80)
        
        # エポック全体の損失をログに記録
        temporal_weight = self.config.stc.weight if self.config.stc.enabled else self.config.lmb
        train_loss = loss_growsp_display + temporal_weight * loss_temporal_display
        self.loss_history.append(train_loss)
        loss_name = 'loss_stc' if self.config.stc.enabled else 'loss_tarl'
        wandb.log({'epoch': epoch, 'loss_growsp': loss_growsp_display, loss_name: loss_temporal_display, 'train_loss': train_loss})
    
    def train_growsp(self, growsp_data):
        """GrowSPのトレーニング
        
        growsp_dataはcollate結果のタプル:
        (coords_batch, feats_batch, normal_batch, labels_batch,
         inverse_batch, pseudo_batch, inds_batch, region_batch, index_batch)
        の9要素が入る想定。ここではcoords, pseudo, indsのみ使用する。
        """
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.config.ignore_label).to("cuda:0")
        coords = growsp_data[0]
        pseudo_labels = growsp_data[5]
        inds = growsp_data[6]
        
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
        
        # デバッグモードでTARLデータを保存
        if self.config.debug:
            print('TARLデータを保存します')
            debug_dir = os.path.join(self.config.save_path, 'debug_tarl')
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            
            # ファイル名の生成
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            # バッチIDを取得
            batch_ids = torch.unique(coords_q[:, 0])
            if len(batch_ids) > 0:
                # 最初のバッチを選択
                batch_id = batch_ids[0].item()
                
                # クエリデータの抽出と保存
                mask_q = coords_q[:, 0] == batch_id
                points_q = coords_q[mask_q, 1:].detach().cpu().numpy() * self.config.voxel_size
                seg_q = segs_q[int(batch_id)].cpu().numpy()
                
                # キーデータの抽出と保存
                mask_k = coords_k[:, 0] == batch_id
                points_k = coords_k[mask_k, 1:].detach().cpu().numpy() * self.config.voxel_size
                seg_k = segs_k[int(batch_id)].cpu().numpy()
                
                # データ保存
                np.save(os.path.join(debug_dir, f'points_q_{timestamp}.npy'), points_q)
                np.save(os.path.join(debug_dir, f'segs_q_{timestamp}.npy'), seg_q)
                np.save(os.path.join(debug_dir, f'points_k_{timestamp}.npy'), points_k)
                np.save(os.path.join(debug_dir, f'segs_k_{timestamp}.npy'), seg_k)
                
                self.logger.info(f'TARLデータを保存しました: {debug_dir}')
            exit()
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
            scene_seg_feats_q, mask_q = compute_segment_feats(
                scene_feats_q, scene_segs_q, max_seg_num=None, 
                ignore_hdbscan_outliers=self.config.ignore_hdbscan_outliers
            )
            seg_feats_q_list.append(scene_seg_feats_q)
            mask_q_list.append(mask_q)
        
        # 特徴とマスクのスタック
        # バッチ内で最大セグメント数を計算
        max_seg_num_in_batch = max(seg_feats.size(0) for seg_feats in seg_feats_q_list)
        
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
                    scene_feats_k, scene_segs_k, max_seg_num=None,
                    ignore_hdbscan_outliers=self.config.ignore_hdbscan_outliers
                )
                seg_feats_k_list.append(scene_seg_feats_k)
                mask_k_list.append(mask_k)
            
            # クエリ側とキー側の最大セグメント数を統一
            max_seg_num_k_in_batch = max(seg_feats.size(0) for seg_feats in seg_feats_k_list)
            final_max_seg_num = max(max_seg_num_in_batch, max_seg_num_k_in_batch)
        
        # クエリ側パディング処理
        padded_seg_feats_q_list = []
        padded_mask_q_list = []
        
        for seg_feats, mask in zip(seg_feats_q_list, mask_q_list):
            current_seg_num = seg_feats.size(0)
            if current_seg_num < final_max_seg_num:
                # 不足分をゼロパディング
                padding_size = final_max_seg_num - current_seg_num
                padded_feats = torch.cat([
                    seg_feats,
                    torch.zeros(padding_size, seg_feats.size(1), device=seg_feats.device)
                ], dim=0)
                padded_mask = torch.cat([
                    mask,
                    torch.ones(padding_size, dtype=torch.bool, device=mask.device)  # パディング部分はTrue（存在しない）
                ], dim=0)
            else:
                padded_feats = seg_feats
                padded_mask = mask
            
            padded_seg_feats_q_list.append(padded_feats)
            padded_mask_q_list.append(padded_mask)
        
        padded_seg_feats_q = torch.stack(padded_seg_feats_q_list, dim=0)
        batch_mask_q = torch.stack(padded_mask_q_list, dim=0)
        
        # プロジェクションヘッドに入力
        proj_feats_q = self.proj_head_q(padded_seg_feats_q, enc_mask=batch_mask_q)
        
        # プレディクターに入力
        pred_feats_q = self.predictor(proj_feats_q, enc_mask=batch_mask_q)
        pred_feats_q = F.normalize(pred_feats_q, dim=-1)
        
        # キー側の特徴抽出は既に上で実行済み
        with torch.no_grad():
            # キー側パディング処理
            padded_seg_feats_k_list = []
            padded_mask_k_list = []
            
            for seg_feats, mask in zip(seg_feats_k_list, mask_k_list):
                current_seg_num = seg_feats.size(0)
                if current_seg_num < final_max_seg_num:
                    # 不足分をゼロパディング
                    padding_size = final_max_seg_num - current_seg_num
                    padded_feats = torch.cat([
                        seg_feats,
                        torch.zeros(padding_size, seg_feats.size(1), device=seg_feats.device)
                    ], dim=0)
                    padded_mask = torch.cat([
                        mask,
                        torch.ones(padding_size, dtype=torch.bool, device=mask.device)  # パディング部分はTrue（存在しない）
                    ], dim=0)
                else:
                    padded_feats = seg_feats
                    padded_mask = mask
                
                padded_seg_feats_k_list.append(padded_feats)
                padded_mask_k_list.append(padded_mask)
            
            padded_seg_feats_k = torch.stack(padded_seg_feats_k_list, dim=0)
            batch_mask_k = torch.stack(padded_mask_k_list, dim=0)
            
            # プロジェクションヘッドに入力
            proj_feats_k = self.proj_head_k(padded_seg_feats_k, enc_mask=batch_mask_k)
            proj_feats_k = F.normalize(proj_feats_k, dim=-1)
        
        # InfoNCE損失の計算
        loss = calc_info_nce(pred_feats_q, proj_feats_k, batch_mask_q, batch_mask_k)
        return loss
    
    def _merge_init_sp_to_growsp(
        self,
        point_feats: torch.Tensor,
        init_sp_labels: torch.Tensor,
        current_growsp: Optional[int]
    ) -> torch.Tensor:
        """
        init SPラベルをSuperpoint Constructor（kmeans）で統合したSPラベルに変換
        
        get_kittisp_featureの処理と同様に、init SPの特徴量を計算してkmeansで統合する。
        
        Args:
            point_feats: 点の特徴量 [N, D]
            init_sp_labels: init SPラベル [N]
            current_growsp: 統合後のSP数
            
        Returns:
            統合後のSPラベル [N]
        """
        device = point_feats.device
        
        # current_growspがNoneの場合はinit SPをそのまま返す
        if current_growsp is None:
            return init_sp_labels
        
        # 有効な点のみを処理（-1以外）
        valid_mask = init_sp_labels >= 0
        if not valid_mask.any():
            return init_sp_labels
        
        valid_feats = point_feats[valid_mask]
        valid_init_labels = init_sp_labels[valid_mask]
        
        # init SPのユニークラベルを取得
        unique_init_sp = torch.unique(valid_init_labels)
        init_sp_num = len(unique_init_sp)
        
        # init SPが統合後の数より少ない場合はそのまま返す
        if init_sp_num <= current_growsp:
            return init_sp_labels
        
        # init SPラベルを連番（0..M-1）にリマップ
        label_to_idx = {label.item(): idx for idx, label in enumerate(unique_init_sp)}
        remapped_labels = torch.tensor(
            [label_to_idx[l.item()] for l in valid_init_labels],
            device=device, dtype=torch.long
        )
        
        # init SPの特徴量を計算（各SPの点特徴量の平均）
        init_sp_corr = F.one_hot(remapped_labels, num_classes=init_sp_num).float()
        per_init_sp_num = init_sp_corr.sum(0, keepdim=True).t()  # [M, 1]
        init_sp_feats = F.linear(init_sp_corr.t(), valid_feats.t()) / per_init_sp_num  # [M, D]
        init_sp_feats = F.normalize(init_sp_feats, dim=-1)
        
        # kmeansでinit SPを統合
        merged_sp_idx = get_kmeans_labels(n_clusters=current_growsp, pcds=init_sp_feats).long()
        
        # 各点のラベルを統合後SPラベルに変換
        merged_labels = torch.full_like(init_sp_labels, -1)
        merged_labels[valid_mask] = merged_sp_idx[remapped_labels]
        
        return merged_labels
    
    def _apply_sp_mapping(
        self,
        init_sp_labels: torch.Tensor,
        sp_idx: np.ndarray
    ) -> torch.Tensor:
        """
        init SPラベルを統合後SPラベルに変換（キャッシュされたマッピングを使用）
        
        Args:
            init_sp_labels: init SPラベル [N]
            sp_idx: init SP ID → 統合SP ID のマッピング [M]
            
        Returns:
            統合後のSPラベル [N]
        """
        device = init_sp_labels.device
        sp_idx_tensor = torch.from_numpy(sp_idx).long().to(device)
        
        # 結果テンソル（-1で初期化）
        merged_labels = torch.full_like(init_sp_labels, -1)
        
        # 有効な点（init SPラベルが-1でなく、マッピングの範囲内）を変換
        valid_mask = (init_sp_labels >= 0) & (init_sp_labels < len(sp_idx_tensor))
        if valid_mask.any():
            merged_labels[valid_mask] = sp_idx_tensor[init_sp_labels[valid_mask]]
        
        return merged_labels
    
    def train_stc(
        self, 
        stc_data: Dict, 
        phase: int = 0,
        current_growsp: Optional[int] = None,
        profile: bool = False,
        prefetched_flow_results: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    ) -> torch.Tensor:
        """
        STC (Superpoint Time Consistency) 損失の計算
        複数GPUでVoteFlowを並列実行して高速化
        
        Phase 0ではSTC lossを計算せず、Phase 1以降でSuperpoint Constructor（kmeans）で
        統合したSP単位で対応をとる。
        
        Args:
            stc_data: STC用データ
            phase: トレーニングフェーズ（0: Phase 1, 1: Phase 2）
            current_growsp: 統合後のSP数（Phase 2で使用）
            profile: 詳細なタイミング計測を行うかどうか
            prefetched_flow_results: 事前計算済みのScene Flow結果（プリフェッチャーから）
        """
        from scene_flow.correspondence import (
            compute_point_correspondence, compute_superpoint_correspondence_matrix
        )
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Phase 0（Phase 1）ではSTC lossを計算しない
        if phase == 0:
            return torch.tensor(0.0, device="cuda:0", requires_grad=False)
        
        coords_t = stc_data['coords_t']
        coords_t1 = stc_data['coords_t1']
        coords_t_original = stc_data['coords_t_original']
        coords_t1_original = stc_data['coords_t1_original']
        sp_labels_t = stc_data['sp_labels_t']  # init SPラベル
        sp_labels_t1 = stc_data['sp_labels_t1']  # init SPラベル
        pose_t = stc_data['pose_t']
        pose_t1 = stc_data['pose_t1']
        scene_name_t = stc_data.get('scene_name_t', None)  # キャッシュ参照用
        scene_name_t1 = stc_data.get('scene_name_t1', None)

        if not self.voteflow_wrappers and prefetched_flow_results is None:
            return torch.tensor(0.0, device="cuda:0")
        
        # プロファイル用タイミング
        if profile:
            t_feat_start = time.perf_counter()
        
        # 特徴抽出
        in_field_t = ME.TensorField(coords_t[:, 1:] * self.config.voxel_size, coords_t, device=0)
        in_field_t1 = ME.TensorField(coords_t1[:, 1:] * self.config.voxel_size, coords_t1, device=0)
        
        feats_t = self.model_q(in_field_t)
        feats_t1 = self.model_q(in_field_t1)
        
        if profile:
            torch.cuda.synchronize()
            t_feat_end = time.perf_counter()
        
        batch_ids = torch.unique(coords_t[:, 0])
        n_samples = len(batch_ids)
        
        # === VoteFlow並列処理用データ収集 ===
        voteflow_tasks = []
        for batch_idx in range(n_samples):
            points_t = coords_t_original[batch_idx]
            points_t1 = coords_t1_original[batch_idx]
            p_t = pose_t[batch_idx]
            p_t1 = pose_t1[batch_idx]
            
            # numpy配列に変換
            if torch.is_tensor(points_t):
                points_t = points_t.numpy()
            if torch.is_tensor(points_t1):
                points_t1 = points_t1.numpy()
            if torch.is_tensor(p_t):
                p_t = p_t.numpy()
            if torch.is_tensor(p_t1):
                p_t1 = p_t1.numpy()
            
            voteflow_tasks.append({
                'batch_idx': batch_idx,
                'points_t': points_t,
                'points_t1': points_t1,
                'pose_t': p_t,
                'pose_t1': p_t1,
            })
        
        # === VoteFlowマルチGPU並列推論（または事前計算結果を使用） ===
        if profile:
            t_vf_start = time.perf_counter()
        
        if prefetched_flow_results is not None:
            # 事前計算済みの結果を使用（プリフェッチャーから）
            flow_results = prefetched_flow_results
            if profile:
                torch.cuda.synchronize()
                t_vf_end = time.perf_counter()
                t_voteflow_total = t_vf_end - t_vf_start
                t_kdtree_total = 0.0
                t_sp_corr_total = 0.0
                t_loss_total = 0.0
                self.logger.info(f'  [STC] Using prefetched flow results (n_samples={n_samples})')
        else:
            # リアルタイムで計算
            n_gpus = len(self.voteflow_wrappers)
            
            if n_gpus == 1:
                # 単一GPUの場合は従来通りバッチ処理
                points_t_list = [task['points_t'] for task in voteflow_tasks]
                points_t1_list = [task['points_t1'] for task in voteflow_tasks]
                pose_t_list = [task['pose_t'] for task in voteflow_tasks]
                pose_t1_list = [task['pose_t1'] for task in voteflow_tasks]
                
                voteflow = self.voteflow_wrappers[0]
                flow_results = voteflow.compute_flow_batch(
                    points_t_list, points_t1_list, pose_t_list, pose_t1_list
                )
            else:
                # 複数GPUの場合はサンプルを分散して並列計算
                # 各GPUに割り当てるサンプルを決定
                gpu_tasks = [[] for _ in range(n_gpus)]
                gpu_task_indices = [[] for _ in range(n_gpus)]  # 元のインデックスを保持
                
                for i, task in enumerate(voteflow_tasks):
                    gpu_id = i % n_gpus
                    gpu_tasks[gpu_id].append(task)
                    gpu_task_indices[gpu_id].append(i)
                
                # 各GPUでの計算をスレッドプールで並列実行
            def compute_on_gpu(gpu_id: int, tasks: list):
                """指定GPUでバッチ計算を実行"""
                if not tasks:
                    return []
                
                wrapper = self.voteflow_wrappers[gpu_id]
                points_t_list = [t['points_t'] for t in tasks]
                points_t1_list = [t['points_t1'] for t in tasks]
                pose_t_list = [t['pose_t'] for t in tasks]
                pose_t1_list = [t['pose_t1'] for t in tasks]
                
                return wrapper.compute_flow_batch(
                    points_t_list, points_t1_list, pose_t_list, pose_t1_list
                )
            
            # スレッドプールで並列実行
            flow_results = [None] * n_samples
            
            with ThreadPoolExecutor(max_workers=n_gpus) as executor:
                futures = {}
                for gpu_id in range(n_gpus):
                    if gpu_tasks[gpu_id]:
                        future = executor.submit(compute_on_gpu, gpu_id, gpu_tasks[gpu_id])
                        futures[future] = gpu_id
                
                for future in as_completed(futures):
                    gpu_id = futures[future]
                    try:
                        results = future.result()
                        # 元のインデックス順に結果を格納
                        for local_idx, result in enumerate(results):
                            original_idx = gpu_task_indices[gpu_id][local_idx]
                            flow_results[original_idx] = result
                    except Exception as e:
                        self.logger.error(f'GPU {gpu_id} VoteFlow error: {e}')
                        # エラーの場合は空の結果を設定
                        for original_idx in gpu_task_indices[gpu_id]:
                            flow_results[original_idx] = (
                                np.zeros((0, 3), dtype=np.float32),
                                np.array([], dtype=np.int64)
                            )
            
            # リアルタイム計算の場合のプロファイル
            if profile:
                torch.cuda.synchronize()
                t_vf_end = time.perf_counter()
                t_voteflow_total = t_vf_end - t_vf_start
                t_kdtree_total = 0.0
                t_sp_corr_total = 0.0
                t_loss_total = 0.0
        
        # === SPラベル統合: キャッシュから取得（kmeansをスキップ） ===
        if profile:
            t_merge_start = time.perf_counter()
        
        # 各サンプルの特徴量とSPラベルを準備
        sample_data_list = []
        merged_labels_results = {}
        use_cache = bool(self.sp_index_cache and scene_name_t and scene_name_t1)
        
        for batch_idx, batch_id in enumerate(batch_ids):
            batch_id_int = int(batch_id.item())
            
            mask_t = coords_t[:, 0] == batch_id_int
            scene_feats_t = feats_t[mask_t]
            
            mask_t1 = coords_t1[:, 0] == batch_id_int
            scene_feats_t1 = feats_t1[mask_t1]
            
            # init SPラベルを取得
            init_sp_labels_t = sp_labels_t[batch_idx].to("cuda:0")
            init_sp_labels_t1 = sp_labels_t1[batch_idx].to("cuda:0")
            
            # キャッシュからinit SP → 統合SPのマッピングを取得
            if use_cache:
                name_t = scene_name_t[batch_idx]
                name_t1 = scene_name_t1[batch_idx]
                
                if name_t in self.sp_index_cache and name_t1 in self.sp_index_cache:
                    # キャッシュヒット: マッピングを使ってinit SPラベルを統合SPラベルに変換
                    sp_idx_t = self.sp_index_cache[name_t]  # init SP ID → 統合SP ID
                    sp_idx_t1 = self.sp_index_cache[name_t1]
                    
                    # init SPラベルをマッピングで変換（-1は維持）
                    scene_sp_labels_t = self._apply_sp_mapping(init_sp_labels_t, sp_idx_t)
                    scene_sp_labels_t1 = self._apply_sp_mapping(init_sp_labels_t1, sp_idx_t1)
                else:
                    # キャッシュミス → フォールバック
                    if profile and batch_idx == 0:
                        self.logger.warning(f'  [STC] Cache miss for {name_t} or {name_t1}, falling back to kmeans')
                    scene_sp_labels_t = self._merge_init_sp_to_growsp(scene_feats_t, init_sp_labels_t, current_growsp)
                    scene_sp_labels_t1 = self._merge_init_sp_to_growsp(scene_feats_t1, init_sp_labels_t1, current_growsp)
            else:
                # キャッシュなし → 従来通りkmeans
                scene_sp_labels_t = self._merge_init_sp_to_growsp(scene_feats_t, init_sp_labels_t, current_growsp)
                scene_sp_labels_t1 = self._merge_init_sp_to_growsp(scene_feats_t1, init_sp_labels_t1, current_growsp)
            
            merged_labels_results[batch_idx] = (scene_sp_labels_t, scene_sp_labels_t1)
            sample_data_list.append({
                'batch_idx': batch_idx,
                'batch_id': batch_id_int,
                'scene_feats_t': scene_feats_t,
                'scene_feats_t1': scene_feats_t1,
            })
        
        if profile:
            torch.cuda.synchronize()
            t_merge_total = time.perf_counter() - t_merge_start
            if use_cache:
                self.logger.info(f'  [STC] Using cached SP labels (n_samples={n_samples})')
        
        # === 各サンプルの対応点計算・損失計算 ===
        total_loss = torch.tensor(0.0, device="cuda:0")
        valid_batch_count = 0
        
        for batch_idx, batch_id in enumerate(batch_ids):
            batch_id = int(batch_id.item())
            
            # 事前計算した特徴量とSPラベルを取得
            sample_data = sample_data_list[batch_idx]
            scene_feats_t = sample_data['scene_feats_t']
            scene_feats_t1 = sample_data['scene_feats_t1']
            scene_sp_labels_t, scene_sp_labels_t1 = merged_labels_results[batch_idx]
            
            # VoteFlow結果を取得
            flow, valid_indices = flow_results[batch_idx]
            
            if len(flow) == 0:
                continue
            
            with torch.no_grad():
                scene_coords_t_orig = voteflow_tasks[batch_idx]['points_t']
                scene_coords_t1_orig = voteflow_tasks[batch_idx]['points_t1']
                
                # KD-Tree対応点計算
                if profile:
                    t_kd_start = time.perf_counter()
                
                correspondence_t, _ = compute_point_correspondence(
                    scene_coords_t_orig,
                    scene_coords_t1_orig,
                    flow,
                    valid_indices,
                    distance_threshold=self.config.stc.correspondence.distance_threshold
                )
                
                if profile:
                    t_kd_end = time.perf_counter()
                    t_kdtree_total += t_kd_end - t_kd_start
                
                # SP対応行列計算
                if profile:
                    t_sp_start = time.perf_counter()
                
                corr_matrix, unique_sp_t, unique_sp_t1 = compute_superpoint_correspondence_matrix(
                    correspondence_t,
                    scene_sp_labels_t.cpu().numpy(),
                    scene_sp_labels_t1.cpu().numpy(),
                    min_points=self.config.stc.correspondence.min_points
                )
                
                if profile:
                    t_sp_end = time.perf_counter()
                    t_sp_corr_total += t_sp_end - t_sp_start
                
                if corr_matrix.size == 0:
                    continue
                
                # === vis_sp: Superpoint対応可視化 ===
                if self.config.vis_sp:
                    self._visualize_sp_correspondence(
                        scene_coords_t_orig, scene_coords_t1_orig,
                        scene_sp_labels_t.cpu().numpy(), scene_sp_labels_t1.cpu().numpy(),
                        corr_matrix, unique_sp_t, unique_sp_t1, batch_idx
                    )
                    # 保存したら終了
                    self.logger.info("vis_sp: Superpointの対応可視化を保存しました。プログラムを終了します。")
                    sys.exit(0)
                
                corr_matrix = torch.from_numpy(corr_matrix).float().to("cuda:0")
            
            # 損失計算
            if profile:
                t_loss_start = time.perf_counter()
            
            sp_feats_t, valid_mask_t = compute_sp_features(
                scene_feats_t, scene_sp_labels_t, 
                num_sp=len(unique_sp_t) if len(unique_sp_t) > 0 else None
            )
            sp_feats_t1, valid_mask_t1 = compute_sp_features(
                scene_feats_t1, scene_sp_labels_t1, 
                num_sp=len(unique_sp_t1) if len(unique_sp_t1) > 0 else None
            )
            
            loss = loss_stc_similarity(
                sp_feats_t,
                sp_feats_t1,
                corr_matrix,
                valid_mask_t,
                valid_mask_t1,
                min_correspondence=self.config.stc.correspondence.min_points
            )
            
            if profile:
                torch.cuda.synchronize()
                t_loss_end = time.perf_counter()
                t_loss_total += t_loss_end - t_loss_start
            
            if loss.grad_fn is not None:
                total_loss = total_loss + loss
                valid_batch_count += 1
        
        if valid_batch_count > 0:
            total_loss = total_loss / valid_batch_count
        
        # プロファイル結果を出力
        if profile:
            n_gpus = len(self.voteflow_wrappers)
            self.logger.info(
                f'  [STC PROFILE] feat_extract={t_feat_end - t_feat_start:.3f}s | '
                f'voteflow={t_voteflow_total:.3f}s ({n_gpus} GPUs, {n_samples} samples) | '
                f'merge_sp={t_merge_total:.3f}s ({t_merge_total/n_samples:.3f}s/sample) | '
                f'kdtree={t_kdtree_total:.3f}s ({t_kdtree_total/n_samples:.3f}s/sample) | '
                f'sp_corr={t_sp_corr_total:.3f}s | '
                f'loss={t_loss_total:.3f}s | '
                f'n_samples={n_samples}'
            )
        
        return total_loss
    
    def _visualize_sp_correspondence(
        self,
        points_t: np.ndarray,
        points_t1: np.ndarray,
        sp_labels_t: np.ndarray,
        sp_labels_t1: np.ndarray,
        corr_matrix: np.ndarray,
        unique_sp_t: np.ndarray,
        unique_sp_t1: np.ndarray,
        batch_idx: int
    ):
        """
        Superpointの対応を可視化してPLYファイルで保存
        
        対応する時刻tと時刻t+nのsuperpointを同じ色で着色し、2つのplyファイルを出力。
        対応がないSPはグレーで表示。
        
        Args:
            points_t: 時刻tの点群 [N, 3]
            points_t1: 時刻t+nの点群 [M, 3]
            sp_labels_t: 時刻tの各点のSPラベル [N]
            sp_labels_t1: 時刻t+nの各点のSPラベル [M]
            corr_matrix: SP対応行列 [num_unique_sp_t, num_unique_sp_t1]
            unique_sp_t: 時刻tのユニークSPラベル
            unique_sp_t1: 時刻t+nのユニークSPラベル
            batch_idx: バッチインデックス
        """
        import random
        
        # 出力ディレクトリ
        output_dir = os.path.join(self.config.save_path, "vis_sp_debug")
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info(f"vis_sp: Superpointの対応可視化を開始 (batch_idx={batch_idx})")
        self.logger.info(f"  時刻t: {len(points_t)} 点, {len(unique_sp_t)} SPs")
        self.logger.info(f"  時刻t+n: {len(points_t1)} 点, {len(unique_sp_t1)} SPs")
        self.logger.info(f"  対応行列サイズ: {corr_matrix.shape}")
        
        # SP IDをインデックスにマッピング
        sp_t_to_idx = {sp: i for i, sp in enumerate(unique_sp_t)}
        sp_t1_to_idx = {sp: i for i, sp in enumerate(unique_sp_t1)}
        idx_to_sp_t = {i: sp for sp, i in sp_t_to_idx.items()}
        idx_to_sp_t1 = {i: sp for sp, i in sp_t1_to_idx.items()}
        
        # 対応するSPペアを特定（各SP_tについて最も対応点数が多いSP_t1を見つける）
        # 双方向で最も対応が強いペアを抽出
        sp_pairs = []  # (sp_t, sp_t1, correspondence_count)
        
        min_correspondence = self.config.stc.correspondence.min_points
        
        for i in range(corr_matrix.shape[0]):
            # このSP_tと最も対応点数が多いSP_t1を見つける
            row = corr_matrix[i]
            if row.max() >= min_correspondence:
                j = row.argmax()
                count = row[j]
                sp_t = idx_to_sp_t[i]
                sp_t1 = idx_to_sp_t1[j]
                sp_pairs.append((sp_t, sp_t1, count))
        
        self.logger.info(f"  有効な対応ペア数: {len(sp_pairs)}")
        
        # 各対応ペアにユニークな色を割り当て
        # HSVを使ってカラフルな色を生成
        def hsv_to_rgb(h, s, v):
            """HSV -> RGB変換 (h: 0-360, s,v: 0-1)"""
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(h/360.0, s, v)
            return (int(r * 255), int(g * 255), int(b * 255))
        
        sp_colors_t = {}   # SP_t -> (R, G, B)
        sp_colors_t1 = {}  # SP_t1 -> (R, G, B)
        
        n_pairs = len(sp_pairs)
        for idx, (sp_t, sp_t1, count) in enumerate(sp_pairs):
            # 色相を等間隔で割り当て
            hue = (idx * 360.0 / max(n_pairs, 1)) % 360
            color = hsv_to_rgb(hue, 0.8, 0.9)
            sp_colors_t[sp_t] = color
            sp_colors_t1[sp_t1] = color
        
        # ログ出力: 対応ペアの詳細（最初の10ペア）
        self.logger.info("  対応ペアの詳細（最初の10ペア）:")
        for sp_t, sp_t1, count in sp_pairs[:10]:
            color = sp_colors_t.get(sp_t, (128, 128, 128))
            self.logger.info(f"    SP_t={sp_t} <-> SP_t1={sp_t1}, count={count}, color={color}")
        
        # PLYファイルとして保存
        filepath_t = os.path.join(output_dir, f"batch{batch_idx}_t.ply")
        filepath_t1 = os.path.join(output_dir, f"batch{batch_idx}_t1.ply")
        
        save_vis_sp_ply(points_t, sp_labels_t, sp_colors_t, filepath_t)
        save_vis_sp_ply(points_t1, sp_labels_t1, sp_colors_t1, filepath_t1)
        
        self.logger.info(f"  保存完了: {filepath_t}")
        self.logger.info(f"  保存完了: {filepath_t1}")
        
        # 統計情報をテキストファイルで保存
        stats_path = os.path.join(output_dir, f"batch{batch_idx}_stats.txt")
        with open(stats_path, 'w') as f:
            f.write(f"=== Superpoint対応可視化統計 (batch_idx={batch_idx}) ===\n\n")
            f.write(f"時刻t: {len(points_t)} 点, {len(unique_sp_t)} SPs\n")
            f.write(f"時刻t+n: {len(points_t1)} 点, {len(unique_sp_t1)} SPs\n")
            f.write(f"対応行列サイズ: {corr_matrix.shape}\n")
            f.write(f"有効な対応ペア数: {len(sp_pairs)}\n")
            f.write(f"min_correspondence設定: {min_correspondence}\n\n")
            f.write("=== 全対応ペア詳細 ===\n")
            for sp_t, sp_t1, count in sorted(sp_pairs, key=lambda x: -x[2]):
                color = sp_colors_t.get(sp_t, (128, 128, 128))
                f.write(f"SP_t={sp_t} <-> SP_t1={sp_t1}, count={int(count)}, color=RGB{color}\n")
        
        self.logger.info(f"  統計情報保存: {stats_path}")
    
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
        
        # STC用: init SP → 統合SP のマッピング（sp_idx）をキャッシュに保存
        # context[i] = (scene_name, gt, raw_region, sp_idx)
        # sp_idx = init SP ID → 統合SP ID のマッピング（点数に依存しない）
        self.sp_index_cache.clear()
        for i, ctx in enumerate(context):
            scene_name = ctx[0]
            sp_idx = ctx[3] if len(ctx) > 3 else None
            if sp_idx is not None:
                self.sp_index_cache[scene_name] = sp_idx
        self.logger.info(f'STC用SPマッピングキャッシュを更新: {len(self.sp_index_cache)} シーン')
        
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
            o_Acc, m_Acc, m_IoU, s, IoU_dict, distance_metrics, moving_static_metrics = eval(epoch, self.config)
            self.logger.info('エポック: {:02d}, oAcc {:.2f}  mAcc {:.2f} IoUs'.format(epoch, o_Acc, m_Acc) + s)
            
            # 結果のログ記録
            d = {'epoch': epoch, 'oAcc': o_Acc, 'mAcc': m_Acc, 'mIoU': m_IoU}
            d.update(IoU_dict)
            wandb.log(d)
            
            # 距離別メトリクスのログ記録
            if distance_metrics:
                self._log_distance_metrics(epoch, distance_metrics)
            
            # 移動/静止別メトリクスのログ記録
            if moving_static_metrics:
                self._log_moving_static_metrics(epoch, moving_static_metrics)
            
            # 早期停止判定
            if self.config.early_stopping and not self.early_stopped:
                self._check_early_stopping(epoch, m_IoU)
            
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
    
    def _log_distance_metrics(self, epoch: int, distance_metrics: dict):
        """距離別メトリクスをwandbにログする
        
        wandb.Tableを使用して、横軸に距離、縦軸に精度指標のグラフを作成可能なデータをログ
        """
        # 距離帯のキーをソート（'0-10', '10-20', ... , '80+'）
        sorted_keys = sorted(distance_metrics.keys(), key=lambda x: int(x.split('-')[0].replace('+', '')))
        
        # 個別の数値をログ（wandb UIでカスタムチャートを作成可能）
        flat_metrics = {'epoch': epoch}
        for key in sorted_keys:
            flat_metrics[f'distance/{key}/oAcc'] = distance_metrics[key]['oAcc']
            flat_metrics[f'distance/{key}/mAcc'] = distance_metrics[key]['mAcc']
            flat_metrics[f'distance/{key}/mIoU'] = distance_metrics[key]['mIoU']
            flat_metrics[f'distance/{key}/count'] = distance_metrics[key]['count']
        wandb.log(flat_metrics)
        
        # wandb.Tableを使用してデータをログ（グラフ作成用）
        # x軸の値（距離帯の上限）を抽出
        table_data = []
        for key in sorted_keys:
            if '+' in key:
                # '80+' のような形式
                distance_upper = int(key.replace('+', '')) + 10
            else:
                # '0-10' のような形式
                distance_upper = int(key.split('-')[1])
            
            table_data.append([
                epoch,
                key,
                distance_upper,
                distance_metrics[key]['oAcc'],
                distance_metrics[key]['mAcc'],
                distance_metrics[key]['mIoU'],
                distance_metrics[key]['count']
            ])
        
        # テーブルをログ
        table = wandb.Table(
            columns=['epoch', 'distance_range', 'distance_upper', 'oAcc', 'mAcc', 'mIoU', 'count'],
            data=table_data
        )
        wandb.log({'distance/metrics_table': table})
    
    def _log_moving_static_metrics(self, epoch: int, moving_static_metrics: dict):
        """移動/静止別メトリクスをwandbにログする"""
        flat_metrics = {'epoch': epoch}
        
        for category in ['moving', 'static']:
            if category in moving_static_metrics:
                flat_metrics[f'{category}/oAcc'] = moving_static_metrics[category]['oAcc']
                flat_metrics[f'{category}/mAcc'] = moving_static_metrics[category]['mAcc']
                flat_metrics[f'{category}/mIoU'] = moving_static_metrics[category]['mIoU']
                flat_metrics[f'{category}/count'] = moving_static_metrics[category]['count']
        
        wandb.log(flat_metrics) 

    def _check_early_stopping(self, epoch: int, val_miou: float):
        """早期停止判定を行う（新仕様）"""
        # lmb依存の収束判定しきい値
        loss_plateau = 0.003 if self.config.lmb <= 1.0 else 0.005 if self.config.lmb <= 2.0 else 0.01
        
        # 改善判定
        if val_miou > self.best_metric_score + self.config.early_stopping_min_delta:
            # 改善
            self.best_metric_score = val_miou
            self.best_epoch = epoch
            self.patience_counter = 0
            
            # ベストモデルの保存
            torch.save(
                self.model_q.state_dict(), 
                join(self.config.save_path, 'best_model.pth')
            )
            torch.save(
                self.classifier.state_dict(), 
                join(self.config.save_path, 'best_classifier.pth')
            )
            
            self.logger.info(f'新しい最良val_mIoU: {val_miou:.4f} (エポック {epoch})')
            wandb.log({
                'epoch': epoch,
                'best_val_miou': self.best_metric_score,
                'best_epoch': self.best_epoch
            })
        else:
            # 停滞
            self.patience_counter += 1
            self.logger.info(f'val_mIoU改善なし。停滞回数: {self.patience_counter}/{self.config.early_stopping_patience}')
            
            if self.patience_counter >= self.config.early_stopping_patience:
                # 収束・過学習判定
                should_stop = False
                stop_reason = ""
                
                # train_loss相対変化率の計算（十分な履歴がある場合）
                if len(self.loss_history) >= self.config.rel_drop_window:
                    recent_loss = self.loss_history[-1]
                    past_loss = self.loss_history[-self.config.rel_drop_window]
                    rel_drop = (past_loss - recent_loss) / past_loss
                    
                    if rel_drop < loss_plateau:
                        # 収束判定
                        should_stop = True
                        stop_reason = f"train_loss収束 (rel_drop={rel_drop:.6f} < {loss_plateau:.6f})"
                
                # 過学習判定
                if val_miou < self.best_metric_score - self.config.overfit_drop:
                    should_stop = True
                    stop_reason = f"過学習検出 (val_mIoU={val_miou:.4f} < {self.best_metric_score:.4f} - {self.config.overfit_drop:.2f})"
                
                if should_stop:
                    # ベストモデルをロードして早期停止
                    self._load_best_model()
                    self.early_stopped = True
                    self.logger.info(f'早期停止: {stop_reason}. 最良エポック{self.best_epoch}のモデルをロード')
                    wandb.log({
                        'epoch': epoch,
                        'early_stopped': True,
                        'early_stop_reason': stop_reason,
                        'early_stop_epoch': epoch,
                        'best_final_epoch': self.best_epoch,
                        'best_final_val_miou': self.best_metric_score
                    })
                else:
                    # 継続 (wait = PATIENCE_VAL - 1)
                    self.patience_counter = self.config.early_stopping_patience - 1
                    self.logger.info("収束・過学習条件未達成。学習継続")
                    
    def _load_best_model(self):
        """ベストモデルをロードする"""
        best_model_path = join(self.config.save_path, 'best_model.pth')
        best_classifier_path = join(self.config.save_path, 'best_classifier.pth')
        
        if os.path.exists(best_model_path) and os.path.exists(best_classifier_path):
            self.model_q.load_state_dict(torch.load(best_model_path))
            self.classifier.load_state_dict(torch.load(best_classifier_path))
            self.logger.info(f'ベストモデル（エポック {self.best_epoch}）をロードしました')
        else:
            self.logger.warning('ベストモデルファイルが見つかりません') 
