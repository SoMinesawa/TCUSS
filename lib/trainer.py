import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import MinkowskiEngine as ME
import numpy as np
import random
import time
import os
import glob
import re
import wandb
import logging
from tqdm import tqdm
from math import ceil
from typing import Dict, List, Tuple, Optional, Union, Any
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from os.path import join

from models.fpn import Res16FPN18, Res16FPNBase
from lib.config import TCUSSConfig
from lib.utils import (
    get_pseudo_kitti, get_kittisp_feature, get_fixclassifier, get_kmeans_labels
)
from datasets.SemanticKITTI import generate_scene_pairs, get_unique_scene_indices
from lib.stc_loss import compute_sp_features, loss_stc_similarity_weighted
from eval_SemanticKITTI import eval, eval_ddp
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
    """TCUSSのトレーニングプロセスを管理するクラス（DDP対応版）"""
    
    def __init__(
        self, 
        config: TCUSSConfig, 
        logger: logging.Logger,
        local_rank: int = 0,
        world_size: int = 1,
        is_main_process: bool = True
    ):
        """
        トレーナーの初期化
        
        Args:
            config: トレーニング設定
            logger: ロガー
            local_rank: このプロセスのローカルランク（GPU ID）
            world_size: 総プロセス数
            is_main_process: メインプロセスかどうか
        """
        self.config = config
        self.logger = logger
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main_process = is_main_process
        self.use_ddp = getattr(config, 'use_ddp', False) and world_size > 1
        
        # デバイス設定
        self.device = f"cuda:{local_rank}"
        
        # モデルの初期化（バックボーンのみ）
        self.model_q = Res16FPN18(
            in_channels=config.input_dim, 
            out_channels=config.feats_dim, 
            conv1_kernel_size=config.conv1_kernel_size, 
            config=config
        ).to(self.device)
        
        # DDPでモデルをラップ
        if self.use_ddp:
            self.model_q = DDP(self.model_q, device_ids=[local_rank], output_device=local_rank)
            if self.is_main_process:
                self.logger.info(f'DDP有効: {world_size}プロセスで分散学習')
        
        # デバイス確認ログ
        model_q_module = self.model_q.module if self.use_ddp else self.model_q
        if self.is_main_process:
            self.logger.info(f'model_q device: {next(model_q_module.parameters()).device}')
        
        # STC設定の確認
        if config.stc.enabled and self.is_main_process:
            self.logger.info(f'STC有効: VoteFlow前処理済みデータを使用 ({config.stc.voteflow_preprocess_path})')
        
        # オプティマイザーとスケジューラーは後で初期化する
        self.optimizer = None
        self.schedulers = None
        self.classifier = None
        self.current_growsp = None
        self.resume_epoch = None
        
        # 早期停止関連の変数
        self.best_metric_score = float('-inf')  # best_val
        self.patience_counter = 0  # wait
        self.early_stopped = False
        self.best_epoch = 0
        self.loss_history = []  # train_lossの履歴
    
    def setup_optimizer(self):
        """オプティマイザの設定"""
        # DDP時は.moduleを通してパラメータにアクセス
        model_q = self.model_q.module if self.use_ddp else self.model_q
        
        # バックボーンのパラメータのみ
        backbone_params = list(model_q.parameters())
        
        self.optimizer = torch.optim.AdamW(
            backbone_params, 
            lr=self.config.lr, 
            weight_decay=self.config.weight_decay
        )
    
    def setup_schedulers(self, train_loader_length: int):
        """スケジューラの設定"""
        steps_per_epoch = ceil(train_loader_length / self.config.accum_step)
        self.schedulers = [
            torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, 
                max_lr=self.config.lr, 
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

    def _find_latest_checkpoint_epoch(self) -> int:
        """save_path 内の最新 checkpoint_epoch_*.pth の epoch を返す（見つからなければ例外）"""
        pattern = join(self.config.save_path, "checkpoint_epoch_*.pth")
        paths = glob.glob(pattern)
        if not paths:
            raise FileNotFoundError(
                f"resume=true ですがチェックポイントが見つかりません: pattern={pattern}. "
                f"save_path={self.config.save_path}"
            )
        epochs: List[int] = []
        for p in paths:
            m = re.search(r"checkpoint_epoch_(\d+)\.pth$", os.path.basename(p))
            if m is None:
                continue
            epochs.append(int(m.group(1)))
        if not epochs:
            raise FileNotFoundError(
                f"checkpoint_epoch_*.pth は存在しますが epoch がパースできません: pattern={pattern}"
            )
        return max(epochs)
    
    def resume_from_checkpoint(self, phase: int):
        """チェックポイントから再開（DDP対応版）
        
        重要:
        - wandbのsummaryは同期遅延/未同期の可能性があるため、save_path 内の最新 checkpoint を正とする
        - DDP時は *全プロセス* で同じチェックポイントをロードし、モデル/optimizer/scheduler状態を揃える
        """
        if not self.config.resume:
            return None

        # まず rank0 が最新チェックポイント epoch を決め、DDP時は全rankへブロードキャスト
        ckpt_epoch_tensor = torch.zeros(1, dtype=torch.int64, device=self.device)
        if self.is_main_process:
            latest_epoch = self._find_latest_checkpoint_epoch()
            ckpt_epoch_tensor[0] = latest_epoch
            self.logger.info(
                f"最新チェックポイント: checkpoint_epoch_{latest_epoch}.pth を使用して再開します"
            )
        if self.use_ddp:
            dist.broadcast(ckpt_epoch_tensor, src=0)
        latest_epoch = int(ckpt_epoch_tensor[0].item())

        checkpoint_path = join(self.config.save_path, f'checkpoint_epoch_{latest_epoch}.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"resume=true ですがチェックポイントが見つかりません: {checkpoint_path}"
            )

        # DDP時も全rankでロードして状態を一致させる（rank0のみロードはNG）
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # DDPラップを解除してロード
        model_q = self.model_q.module if self.use_ddp else self.model_q
        model_q.load_state_dict(checkpoint['model_q_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 現在のフェーズのスケジューラー状態を読み込む（存在しなければエラーにしないが警告）
        sched_key = f'scheduler_{phase}_state_dict'
        if sched_key in checkpoint:
            self.schedulers[phase].load_state_dict(checkpoint[sched_key])
        else:
            if self.is_main_process:
                self.logger.warning(
                    f"チェックポイントに {sched_key} がありません。"
                    "スケジューラは初期状態のまま開始されます（意図した挙動か確認してください）。"
                )

        # 乱数状態は rank0 のみ復元（チェックポイントがrank0で保存されるため）
        if self.is_main_process:
            if 'np_random_state' in checkpoint:
                np.random.set_state(checkpoint['np_random_state'])
            if 'torch_random_state' in checkpoint:
                rng_state = checkpoint['torch_random_state']
                if isinstance(rng_state, torch.Tensor):
                    rng_state = rng_state.cpu().to(torch.uint8)
                torch.set_rng_state(rng_state)
            if torch.cuda.is_available() and 'torch_cuda_random_state' in checkpoint and checkpoint['torch_cuda_random_state'] is not None:
                cuda_rng_state = checkpoint['torch_cuda_random_state']
                if isinstance(cuda_rng_state, torch.Tensor):
                    cuda_rng_state = cuda_rng_state.cpu().to(torch.uint8)
                torch.cuda.set_rng_state(cuda_rng_state, device=self.local_rank)

        # early stopping状態の復元（全rankで同じ値にしておく）
        if 'best_metric_score' in checkpoint:
            self.best_metric_score = checkpoint['best_metric_score']
            self.patience_counter = checkpoint['patience_counter']
            self.early_stopped = checkpoint['early_stopped']
            self.best_epoch = checkpoint['best_epoch']
            self.loss_history = checkpoint['loss_history']
            if self.is_main_process:
                self.logger.info(
                    f'early stopping状態を復元: best_score={self.best_metric_score:.4f}, '
                    f'patience={self.patience_counter}, best_epoch={self.best_epoch}'
                )

        # 再開epochは「最新チェックポイントの次」
        self.resume_epoch = latest_epoch + 1
        if self.is_main_process:
            self.logger.info(f'エポック {self.resume_epoch} から再開します')

        # 念のため同期
        if self.use_ddp:
            dist.barrier()

        return self.resume_epoch
    
    def save_checkpoint(self, epoch: int, phase: int):
        """チェックポイントの保存（メインプロセスのみ）"""
        if not self.is_main_process:
            return
        
        # DDPラップを解除してstate_dictを取得
        model_q_state = self.model_q.module.state_dict() if self.use_ddp else self.model_q.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_q_state_dict': model_q_state,
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
    
    def train(self, train_loader: DataLoader, cluster_loader: DataLoader, train_sampler: Optional[DistributedSampler] = None):
        """モデルのトレーニングメイン関数（DDP対応版）
        
        Args:
            train_loader: トレーニングデータローダー
            cluster_loader: クラスタリングデータローダー
            train_sampler: DDP用のDistributedSampler（DDPなしの場合はNone）
        """
        # vis時はwandbをdisableに設定
        if self.config.vis:
            os.environ["WANDB_MODE"] = "disabled"

        # Weights & Biasesの初期化（メインプロセスのみ）
        if self.is_main_process:
            _ = self.init_wandb()
        else:
            os.environ["WANDB_MODE"] = "disabled"
        
        # オプティマイザとスケジューラの設定
        self.setup_optimizer()
        self.setup_schedulers(len(train_loader))
        
        # train_samplerを保存（train_phaseで使用）
        self.train_sampler = train_sampler
        
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
                # シーンペアを再生成（固定シード + エポック番号で全GPUで同じ結果）
                # これによりtrainsetとclustersetで同じシーンを使用できる
                sync_seed = self.config.seed + epoch
                scene_pairs, scene_idx_t1, scene_idx_t2 = generate_scene_pairs(
                    select_num=self.config.select_num,
                    scan_window=self.config.scan_window,
                    seed=sync_seed
                )
                scene_idx_all = get_unique_scene_indices(scene_idx_t1, scene_idx_t2)
                
                # trainsetとclustersetに同じシーンを設定
                train_loader.dataset.set_scene_pairs(scene_idx_t1, scene_idx_t2)
                cluster_loader.dataset.random_select_sample(scene_idx_all)
                
                # DDP時はバリア同期
                if self.use_ddp:
                    dist.barrier()
                
                self.classifier, self.current_growsp = self.cluster(cluster_loader, epoch, self.config.max_epoch[0], is_growing)
            
            # データセットにクラスタ数を設定（STCでデータセットが異なるため属性チェック）
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
        """1エポックのトレーニング（DDP対応版）"""
        # DDP時は全プロセスでバリア同期してからエポック開始
        if self.use_ddp:
            dist.barrier()
        
        # DDP時はDistributedSamplerにエポックを設定
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        
        # モデルをトレーニングモードに設定
        self.model_q.train()
        
        # オプティマイザのリセット
        self.optimizer.zero_grad()
        
        # 損失の表示用変数
        loss_growsp_display = 0.0
        loss_temporal_display = 0.0
        
        # DataLoaderイテレータを使用
        dataloader_iter = iter(train_loader)
        
        # メインプロセスのみtqdmで進捗表示
        iterator = range(len(train_loader))
        if self.is_main_process:
            iterator = tqdm(iterator, desc=f'トレーニングエポック: {epoch}')
        
        for i in iterator:
            import time as _time
            _t0 = _time.time()
            data = next(dataloader_iter)
            _t1 = _time.time()
            growsp_t1_data, growsp_t2_data, stc_data = data
            
            # 注意: BatchNormのinplace更新問題を回避するため、各損失は個別にbackwardする
            growsp_t1_loss = self.train_growsp(growsp_t1_data) / self.config.accum_step
            if growsp_t1_loss.grad_fn is not None:
                growsp_t1_loss.backward()
            
            growsp_t2_loss = self.train_growsp(growsp_t2_data) / self.config.accum_step
            if growsp_t2_loss.grad_fn is not None:
                growsp_t2_loss.backward()
            _t2 = _time.time()
            
            if self.config.stc.enabled:
                # STCモード（Phase 0からSTC lossを計算）
                if stc_data is not None:
                    temporal_loss = self.train_stc(stc_data) / self.config.accum_step
                    _t3 = _time.time()
                else:
                    temporal_loss = torch.tensor(0.0, device=self.device)
                    _t3 = _t2
                temporal_weight = self.config.stc.weight
                
                # STC損失の個別backward
                if temporal_loss.grad_fn is not None:
                    (temporal_weight * temporal_loss).backward()
            else:
                # GrowSPのみモード（STC無効）
                temporal_loss = torch.tensor(0.0, device=self.device)
                temporal_weight = 0.0
                _t3 = _t2
            
            # 損失の表示用に加算（backwardは既に各損失で個別に実行済み）
            growsp_loss = growsp_t1_loss.item() + growsp_t2_loss.item()
            loss_growsp_display += growsp_loss
            loss_temporal_display += temporal_loss.item() if isinstance(temporal_loss, torch.Tensor) else temporal_loss
            
            # 勾配の蓄積ステップに達したか、最後のバッチの場合
            if ((i+1) % self.config.accum_step == 0) or (i == len(train_loader)-1):
                self.optimizer.step()
                
                # 学習率のログ記録（メインプロセスのみ）
                if self.is_main_process:
                    wandb.log({
                        'epoch': epoch, 
                        'backbone_lr': self.optimizer.param_groups[0]['lr']
                    })
                
                # スケジューラの更新
                if self.schedulers[phase] is not None:
                    self.schedulers[phase].step()
                
                # オプティマイザのリセット
                self.optimizer.zero_grad()
        
        # エポック全体の損失を計算
        temporal_weight = self.config.stc.weight if self.config.stc.enabled else 0.0
        train_loss = loss_growsp_display + temporal_weight * loss_temporal_display
        
        # DDP時はtrain_lossを全プロセスで平均化して同期（early_stoppingの収束判定に使用）
        if self.use_ddp:
            train_loss_tensor = torch.tensor([train_loss], device=self.device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            train_loss = train_loss_tensor.item() / self.world_size
        
        self.loss_history.append(train_loss)
        
        if self.is_main_process:
            loss_name = 'loss_stc' if self.config.stc.enabled else 'loss_temporal'
            num_iters = len(train_loader)
            if num_iters <= 0:
                raise ValueError(f"Unexpected train_loader length: {num_iters}")
            loss_growsp_avg = loss_growsp_display / num_iters
            loss_temporal_avg = loss_temporal_display / num_iters
            train_loss_avg = loss_growsp_avg + temporal_weight * loss_temporal_avg
            wandb.log({
                'epoch': epoch,
                'train/iters': num_iters,
                # 既存のメトリクス（epoch内の合計）
                'loss_growsp': loss_growsp_display,
                loss_name: loss_temporal_display,
                'train_loss': train_loss,
                # 追加メトリクス（epoch内の平均：解釈しやすい）
                'loss_growsp_avg': loss_growsp_avg,
                f'{loss_name}_avg': loss_temporal_avg,
                'train_loss_avg': train_loss_avg,
            })
        
        # vis_spモードでは1 epoch終了後に終了
        if self.config.vis_sp:
            self.logger.info("vis_sp: 1 epoch終了。可視化データは vis_sp_debug/ に保存されました。プログラムを終了します。")
            sys.exit(0)
    
    def train_growsp(self, growsp_data):
        """GrowSPのトレーニング（DDP対応版）
        
        growsp_dataはcollate結果で、11要素タプル:
        (coords, feats, normal, labels, inverse, pseudo, inds, region, index, feats_sizes, unique_vals_list)
        """
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.config.ignore_label).to(self.device)
        
        # データ形式をアンパック（11要素タプルのみ対応）
        if len(growsp_data) != 11:
            raise ValueError(f"Unexpected growsp_data length: {len(growsp_data)}. Expected 11.")
        
        coords = growsp_data[0]
        pseudo_labels = growsp_data[5]
        inds = growsp_data[6]
        feats_sizes = growsp_data[9]
        # unique_vals_list = growsp_data[10]  # STC用（train_growspでは使用しない）
        
        # サイズ不一致チェック
        if len(inds) != len(pseudo_labels):
            raise ValueError(f"inds ({len(inds)}) と pseudo_labels ({len(pseudo_labels)}) のサイズが一致しません")
        
        # 入力フィールドの作成（DDPデバイスを使用）
        in_field = ME.TensorField(coords[:, 1:] * self.config.voxel_size, coords, device=self.local_rank)
        
        # 特徴抽出
        feats = self.model_q(in_field)
        feats = feats[inds.long()]
        feats = F.normalize(feats, dim=-1)
        
        # 擬似ラベルの準備
        pseudo_labels_comp = pseudo_labels.long().to(self.device)
        
        # ロジットの計算
        logits = F.linear(F.normalize(feats), F.normalize(self.classifier.weight))
        
        # 損失計算
        loss_sem = loss_fn(logits * 5, pseudo_labels_comp).mean()
        return loss_sem
    
    def train_stc(self, stc_data: Dict) -> torch.Tensor:
        """
        STC (Superpoint Time Consistency) 損失の計算
        
        対応点計算・SP対応行列計算はデータセット側（KITTIstc）で事前に行われる。
        ここでは特徴抽出とロス計算のみを行う。
        統合SPラベルはsp_id_pathから読み込まれている。
        
        Args:
            stc_data: STC用データ（corr_matrix, sp_labels等を含む）
        """
        import time as _time
        _stc_profile = getattr(self, '_stc_profile_count', 0)
        self._stc_profile_count = _stc_profile + 1
        _do_profile = _stc_profile < 3 and self.is_main_process
        
        if _do_profile: _t0 = _time.time()
        
        coords_t = stc_data['coords_t']
        coords_t2 = stc_data['coords_t2']
        sp_labels_t_list = stc_data['sp_labels_t']  # 統合SPラベル（リスト）
        sp_labels_t2_list = stc_data['sp_labels_t2']  # 統合SPラベル（リスト）
        corr_matrix_list = stc_data['corr_matrix']  # SP対応行列（リスト）
        unique_sp_t_list = stc_data['unique_sp_t']  # ユニークSP（リスト）
        unique_sp_t2_list = stc_data['unique_sp_t2']  # ユニークSP（リスト）
        
        # === 特徴抽出（tとt2を1つのバッチにまとめて1回のforwardで処理） ===
        # これによりBatchNormのinplace更新問題を回避（DDP対応）
        n_t = coords_t.shape[0]
        
        # coords_t2のバッチIDをシフト（forward時のみ、マスク処理では元のIDを使用）
        batch_size = int(coords_t[:, 0].max().item()) + 1
        coords_t2_shifted = coords_t2.clone()
        coords_t2_shifted[:, 0] = coords_t2_shifted[:, 0] + batch_size
        
        # 結合して1回のforward
        coords_combined = torch.cat([coords_t, coords_t2_shifted], dim=0)
        
        in_field_combined = ME.TensorField(
            coords_combined[:, 1:] * self.config.voxel_size, 
            coords_combined, 
            device=self.local_rank
        )
        
        feats_combined = self.model_q(in_field_combined)
        
        # 出力を分割
        feats_t = feats_combined[:n_t]
        feats_t2 = feats_combined[n_t:]
        
        batch_ids = torch.unique(coords_t[:, 0])
        
        # === 各サンプルの損失計算 ===
        total_loss = torch.tensor(0.0, device=self.device)
        valid_batch_count = 0
        
        _loop_times = []
        for batch_idx, batch_id in enumerate(batch_ids):
            if _do_profile: _loop_start = _time.time()
            batch_id_int = int(batch_id.item())
            
            # このバッチの特徴量を取得
            mask_t = coords_t[:, 0] == batch_id_int
            scene_feats_t = feats_t[mask_t]
            
            mask_t2 = coords_t2[:, 0] == batch_id_int
            scene_feats_t2 = feats_t2[mask_t2]
            
            # SPラベルを取得
            scene_sp_labels_t = sp_labels_t_list[batch_idx].to(self.device)
            scene_sp_labels_t2 = sp_labels_t2_list[batch_idx].to(self.device)
            
            # 対応行列を取得
            corr_matrix = corr_matrix_list[batch_idx].to(self.device)
            unique_sp_t = unique_sp_t_list[batch_idx]
            unique_sp_t2 = unique_sp_t2_list[batch_idx]
            
            if corr_matrix.numel() == 0 or len(unique_sp_t) == 0 or len(unique_sp_t2) == 0:
                continue
            
            # === vis_sp: Superpoint対応可視化 ===
            if self.config.vis_sp:
                # coords_originalがないため、voxel座標を使用
                scene_coords_t = (coords_t[mask_t, 1:] * self.config.voxel_size).cpu().numpy()
                scene_coords_t2 = (coords_t2[mask_t2, 1:] * self.config.voxel_size).cpu().numpy()
                # シーン名を取得
                scene_name_t = stc_data['scene_name_t'][batch_idx]
                scene_name_t2 = stc_data['scene_name_t2'][batch_idx]
                self._visualize_sp_correspondence(
                    scene_coords_t, scene_coords_t2,
                    scene_sp_labels_t.cpu().numpy(), scene_sp_labels_t2.cpu().numpy(),
                    corr_matrix.cpu().numpy(), unique_sp_t, unique_sp_t2, batch_idx,
                    scene_name_t, scene_name_t2
                )
                # 1 epoch終了まで可視化を継続（各バッチで保存）
            
            # 損失計算（unique_sp_tはフィルタリング後のSPリストで、corr_matrixの次元と一致）
            sp_feats_t, valid_mask_t, sp_counts_t = compute_sp_features(
                scene_feats_t, scene_sp_labels_t, 
                target_sp_ids=unique_sp_t
            )
            sp_feats_t2, valid_mask_t2, _ = compute_sp_features(
                scene_feats_t2, scene_sp_labels_t2, 
                target_sp_ids=unique_sp_t2
            )
            
            # 新方式: corr_matrixはマッチングスコア（0〜1）を格納（未マッチは0）
            # スコアを重みとして、重み付きコサイン類似度を最大化（=負の値を最小化）
            loss = loss_stc_similarity_weighted(
                sp_feats_t,
                sp_feats_t2,
                corr_matrix,
                valid_mask_t,
                valid_mask_t2
            )
            
            if loss.grad_fn is not None:
                total_loss = total_loss + loss
                valid_batch_count += 1
            
            if _do_profile: _loop_times.append(_time.time() - _loop_start)
        
        if valid_batch_count > 0:
            total_loss = total_loss / valid_batch_count
        
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
        batch_idx: int,
        scene_name_t: str = "",
        scene_name_t2: str = ""
    ):
        """
        Superpointの対応を可視化してPLYファイルで保存
        
        出力ファイル:
        1. batch{idx}_t_independent.ply / batch{idx}_t2_independent.ply
           - 各時刻で独立したSP（対応を考慮せず、各SPに異なる色）
        2. batch{idx}_t_matched.ply / batch{idx}_t2_matched.ply
           - 対応するSPを同じ色で着色（対応がないSPはグレー）
        
        Args:
            points_t: 時刻tの点群 [N, 3]
            points_t1: 時刻t+nの点群 [M, 3]
            sp_labels_t: 時刻tの各点のSPラベル [N]
            sp_labels_t1: 時刻t+nの各点のSPラベル [M]
            corr_matrix: SP対応行列 [num_unique_sp_t, num_unique_sp_t1]
            unique_sp_t: 時刻tのユニークSPラベル
            unique_sp_t1: 時刻t+nのユニークSPラベル
            batch_idx: バッチインデックス
            scene_name_t: 時刻tのシーン名（例: /00/000100）
            scene_name_t2: 時刻t2のシーン名（例: /00/000112）
        """
        import colorsys
        
        def hsv_to_rgb(h, s, v):
            """HSV -> RGB変換 (h: 0-360, s,v: 0-1)"""
            r, g, b = colorsys.hsv_to_rgb(h/360.0, s, v)
            return (int(r * 255), int(g * 255), int(b * 255))
        
        # 出力ディレクトリ
        output_dir = os.path.join(self.config.save_path, "vis_sp_debug")
        os.makedirs(output_dir, exist_ok=True)
        
        # scene_nameからseqとidxを抽出（例: "/00/000100" -> seq="00", idx="000100"）
        def parse_scene_name(scene_name: str):
            parts = scene_name.strip('/').split('/')
            if len(parts) >= 2:
                return parts[0], parts[1]
            return "unknown", "unknown"
        
        seq_t, idx_t = parse_scene_name(scene_name_t)
        seq_t2, idx_t2 = parse_scene_name(scene_name_t2)
        
        # ファイル名のプレフィックス（seq_idx形式）
        prefix_t = f"seq{seq_t}_idx{idx_t}"
        prefix_t2 = f"seq{seq_t2}_idx{idx_t2}"
        
        self.logger.info(f"vis_sp: Superpointの対応可視化を開始 (batch_idx={batch_idx})")
        self.logger.info(f"  時刻t: {scene_name_t} ({len(points_t)} 点, {len(unique_sp_t)} SPs)")
        self.logger.info(f"  時刻t2: {scene_name_t2} ({len(points_t1)} 点, {len(unique_sp_t1)} SPs)")
        self.logger.info(f"  対応行列サイズ: {corr_matrix.shape}")
        
        # ============================================
        # 1. 独立したSPの可視化（対応を考慮しない）
        # ============================================
        # sp_labels内の全SPに色を割り当て（unique_sp_tはフィルタリング後なので使わない）
        all_sp_t = np.unique(sp_labels_t[sp_labels_t >= 0])
        all_sp_t1 = np.unique(sp_labels_t1[sp_labels_t1 >= 0])
        
        # 時刻tの各SPに異なる色を割り当て
        sp_colors_t_independent = {}
        for idx, sp in enumerate(all_sp_t):
            hue = (idx * 360.0 / max(len(all_sp_t), 1)) % 360
            sp_colors_t_independent[sp] = hsv_to_rgb(hue, 0.8, 0.9)
        
        # 時刻t2の各SPに異なる色を割り当て
        sp_colors_t1_independent = {}
        for idx, sp in enumerate(all_sp_t1):
            hue = (idx * 360.0 / max(len(all_sp_t1), 1)) % 360
            sp_colors_t1_independent[sp] = hsv_to_rgb(hue, 0.8, 0.9)
        
        # 独立したSPのPLYファイルを保存
        filepath_t_ind = os.path.join(output_dir, f"{prefix_t}_independent.ply")
        filepath_t1_ind = os.path.join(output_dir, f"{prefix_t2}_independent.ply")
        
        save_vis_sp_ply(points_t, sp_labels_t, sp_colors_t_independent, filepath_t_ind)
        save_vis_sp_ply(points_t1, sp_labels_t1, sp_colors_t1_independent, filepath_t1_ind)
        
        self.logger.info(f"  独立SP保存完了: {filepath_t_ind}")
        self.logger.info(f"  独立SP保存完了: {filepath_t1_ind}")
        
        # ============================================
        # 2. 対応するSPの可視化
        # ============================================
        # SP IDをインデックスにマッピング
        sp_t_to_idx = {sp: i for i, sp in enumerate(unique_sp_t)}
        sp_t1_to_idx = {sp: i for i, sp in enumerate(unique_sp_t1)}
        idx_to_sp_t = {i: sp for sp, i in sp_t_to_idx.items()}
        idx_to_sp_t1 = {i: sp for sp, i in sp_t1_to_idx.items()}
        
        # 対応するSPペアを特定
        # 新方式: corr_matrixはマッチングスコア（0〜1）を格納、非ゼロは有効なマッチ
        sp_pairs = []  # (sp_t, sp_t1, score)
        
        for i in range(corr_matrix.shape[0]):
            row = corr_matrix[i]
            if row.max() > 0:  # スコアが0より大きければ有効なマッチ
                j = row.argmax()
                score = row[j]
                sp_t = idx_to_sp_t[i]
                sp_t1 = idx_to_sp_t1[j]
                sp_pairs.append((sp_t, sp_t1, score))
        
        self.logger.info(f"  有効な対応ペア数: {len(sp_pairs)}")
        
        # 対応付けられた点の割合を計算
        matched_sp_t = set(sp_t for sp_t, _, _ in sp_pairs)
        matched_sp_t1 = set(sp_t1 for _, sp_t1, _ in sp_pairs)
        
        # 各点がマッチしたSPに属するかをカウント
        points_in_matched_sp_t = np.sum(np.isin(sp_labels_t, list(matched_sp_t)))
        points_in_matched_sp_t1 = np.sum(np.isin(sp_labels_t1, list(matched_sp_t1)))
        
        total_points_t = len(points_t)
        total_points_t1 = len(points_t1)
        
        ratio_t = points_in_matched_sp_t / total_points_t * 100 if total_points_t > 0 else 0
        ratio_t1 = points_in_matched_sp_t1 / total_points_t1 * 100 if total_points_t1 > 0 else 0
        
        self.logger.info(f"  === 対応付け統計 ===")
        self.logger.info(f"    時刻t: {points_in_matched_sp_t}/{total_points_t} 点 ({ratio_t:.2f}%) が対応付けられたSPに属する")
        self.logger.info(f"    時刻t2: {points_in_matched_sp_t1}/{total_points_t1} 点 ({ratio_t1:.2f}%) が対応付けられたSPに属する")
        self.logger.info(f"    マッチしたSP数: 時刻t={len(matched_sp_t)}/{len(unique_sp_t)}, 時刻t2={len(matched_sp_t1)}/{len(unique_sp_t1)}")
        
        # 対応するSPに同じ色を割り当て
        sp_colors_t_matched = {}
        sp_colors_t1_matched = {}
        
        n_pairs = len(sp_pairs)
        for idx, (sp_t, sp_t1, count) in enumerate(sp_pairs):
            hue = (idx * 360.0 / max(n_pairs, 1)) % 360
            color = hsv_to_rgb(hue, 0.8, 0.9)
            sp_colors_t_matched[sp_t] = color
            sp_colors_t1_matched[sp_t1] = color
        
        # 対応ありSPのPLYファイルを保存
        filepath_t_matched = os.path.join(output_dir, f"{prefix_t}_matched.ply")
        filepath_t1_matched = os.path.join(output_dir, f"{prefix_t2}_matched.ply")
        
        save_vis_sp_ply(points_t, sp_labels_t, sp_colors_t_matched, filepath_t_matched)
        save_vis_sp_ply(points_t1, sp_labels_t1, sp_colors_t1_matched, filepath_t1_matched)
        
        self.logger.info(f"  対応SP保存完了: {filepath_t_matched}")
        self.logger.info(f"  対応SP保存完了: {filepath_t1_matched}")
        
        # ログ出力: 対応ペアの詳細（最初の10ペア）
        self.logger.info("  対応ペアの詳細（最初の10ペア）:")
        for sp_t, sp_t1, count in sp_pairs[:10]:
            color = sp_colors_t_matched.get(sp_t, (128, 128, 128))
            self.logger.info(f"    SP_t={sp_t} <-> SP_t2={sp_t1}, count={count}, color={color}")
        
        # 統計情報をテキストファイルで保存
        stats_path = os.path.join(output_dir, f"{prefix_t}_to_{prefix_t2}_stats.txt")
        with open(stats_path, 'w') as f:
            f.write(f"=== Superpoint対応可視化統計 ===\n")
            f.write(f"時刻t: {scene_name_t} ({prefix_t})\n")
            f.write(f"時刻t2: {scene_name_t2} ({prefix_t2})\n\n")
            f.write(f"時刻t: {len(points_t)} 点, {len(unique_sp_t)} SPs\n")
            f.write(f"時刻t2: {len(points_t1)} 点, {len(unique_sp_t1)} SPs\n")
            f.write(f"対応行列サイズ: {corr_matrix.shape}\n")
            f.write(f"有効な対応ペア数: {len(sp_pairs)}\n")
            f.write(f"マッチング方式: SPレベル直接マッチング（1対1, Greedy）\n\n")
            f.write("=== 点の対応付け統計 ===\n")
            f.write(f"時刻t: {points_in_matched_sp_t}/{total_points_t} 点 ({ratio_t:.2f}%) が対応付けられたSPに属する\n")
            f.write(f"時刻t2: {points_in_matched_sp_t1}/{total_points_t1} 点 ({ratio_t1:.2f}%) が対応付けられたSPに属する\n")
            f.write(f"マッチしたSP数: 時刻t={len(matched_sp_t)}/{len(unique_sp_t)}, 時刻t2={len(matched_sp_t1)}/{len(unique_sp_t1)}\n\n")
            f.write("=== 出力ファイル ===\n")
            f.write(f"独立SP (時刻t): {filepath_t_ind}\n")
            f.write(f"独立SP (時刻t2): {filepath_t1_ind}\n")
            f.write(f"対応SP (時刻t): {filepath_t_matched}\n")
            f.write(f"対応SP (時刻t2): {filepath_t1_matched}\n\n")
            f.write("=== 全対応ペア詳細 ===\n")
            for sp_t, sp_t1, score in sorted(sp_pairs, key=lambda x: -x[2]):
                color = sp_colors_t_matched.get(sp_t, (128, 128, 128))
                f.write(f"SP_t={sp_t} <-> SP_t2={sp_t1}, score={score:.3f}, color=RGB{color}\n")
        
        self.logger.info(f"  統計情報保存: {stats_path}")
    
    def cluster(self, cluster_loader: DataLoader, epoch: int, start_grow_epoch: Optional[int] = None, is_growing: bool = False):
        """クラスタリングの実行（DDP対応版）
        
        DDP時もメインプロセスのみで特徴抽出・クラスタリングを実行。
        クラスタリング結果（primitive_centers）を全GPUにブロードキャスト。
        """
        time_start = time.time()
        cluster_loader.dataset.mode = 'cluster'
        
        # GrowSPのクラスタ数計算
        current_growsp = None
        if is_growing:
            current_growsp = int(self.config.growsp_start - ((epoch - start_grow_epoch)/self.config.max_epoch[1])*(self.config.growsp_start - self.config.growsp_end))
            if current_growsp < self.config.growsp_end:
                current_growsp = self.config.growsp_end
            if self.is_main_process:
                self.logger.info(f'エポック: {epoch}, スーパーポイントが {current_growsp} に成長')
        
        # DDPラップを解除してモデルを取得（推論用）
        model_q = self.model_q.module if self.use_ddp else self.model_q
        
        # メインプロセスのみで特徴抽出・クラスタリングを実行
        if self.is_main_process or not self.use_ddp:
            feats, labels, sp_index, context = get_kittisp_feature(
                self.config, cluster_loader, model_q, current_growsp, epoch
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
                if indices.sum() > 0:
                    cluster_avg = sp_feats[indices].mean(0, keepdims=True)
                    primitive_centers[cluster_idx] = cluster_avg
            primitive_centers = F.normalize(primitive_centers, dim=1)
        else:
            # 非メインプロセス：ダミーの値を用意（後でbroadcastで上書き）
            feats, labels, sp_index, context = [], [], [], []
            primitive_labels = np.array([], dtype=np.int64)
            primitive_centers = torch.zeros((self.config.primitive_num, self.config.feats_dim))
        
        # DDP時はprimitive_centersを全GPUにブロードキャスト
        if self.use_ddp:
            primitive_centers = primitive_centers.to(self.device)
            dist.broadcast(primitive_centers, src=0)
            primitive_centers = primitive_centers.cpu()
        
        # 分類器の作成（全プロセスで同じものを作成）
        classifier = get_fixclassifier(
            in_channel=self.config.feats_dim, 
            centroids_num=self.config.primitive_num, 
            centroids=primitive_centers
        )
        
        # メインプロセスのみで疑似ラベル計算・統計・ログ処理
        if self.is_main_process or not self.use_ddp:
            # 疑似ラベルの計算と保存（get_kittisp_feature内でsp_id_pathにも保存済み）
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
        
        # DDP時は全プロセスでバリア同期
        if self.use_ddp:
            dist.barrier()
        
        return classifier.to(self.device), current_growsp
    
    def evaluate(self, epoch: int):
        """モデルの評価（DDP対応版：全GPUで分散評価）"""
        # モデルの保存（メインプロセスのみ）
        if self.is_main_process:
            model_q_state = self.model_q.module.state_dict() if self.use_ddp else self.model_q.state_dict()
            torch.save(
                model_q_state, 
                join(self.config.save_path, f'model_{epoch}_checkpoint.pth')
            )
            torch.save(
                self.classifier.state_dict(), 
                join(self.config.save_path, f'cls_{epoch}_checkpoint.pth')
            )
        
        # DDP時は全プロセスでモデル保存を待つ
        if self.use_ddp:
            dist.barrier()
        
        # DDPラップを解除してモデルを取得
        model_q = self.model_q.module if self.use_ddp else self.model_q
        
        # 評価用分類器の作成: primitive重心 (500) をk=semantic_class (19) でkmeansして19クラス分類器を作成
        # self.classifier.weight は (primitive_num, feats_dim) = (500, 128)
        primitive_centers = self.classifier.weight.data.clone()  # (500, 128)
        
        # DDP時はメインプロセスでのみKMeansを実行し、結果をブロードキャスト
        # （KMeansは非決定的なため、各GPUで実行すると異なる結果になる）
        if self.use_ddp:
            # semantic_centersを格納するテンソルを全GPUで準備
            semantic_centers = torch.zeros(
                (self.config.semantic_class, self.config.feats_dim), 
                device=self.device
            )
            
            if self.is_main_process:
                # メインプロセスでのみKMeansを実行
                semantic_labels = get_kmeans_labels(self.config.semantic_class, primitive_centers)  # (500,)
                
                # 各semantic classの重心を計算
                for cls_idx in range(self.config.semantic_class):
                    mask = semantic_labels == cls_idx
                    if mask.sum() > 0:
                        semantic_centers[cls_idx] = primitive_centers[mask].mean(dim=0)
                semantic_centers = F.normalize(semantic_centers, dim=1)
            
            # メインプロセスの結果を全GPUにブロードキャスト
            dist.broadcast(semantic_centers, src=0)
        else:
            # 非DDP時は従来通り
            semantic_labels = get_kmeans_labels(self.config.semantic_class, primitive_centers)  # (500,)
            
            # 各semantic classの重心を計算
            semantic_centers = torch.zeros((self.config.semantic_class, self.config.feats_dim))
            for cls_idx in range(self.config.semantic_class):
                mask = semantic_labels == cls_idx
                if mask.sum() > 0:
                    semantic_centers[cls_idx] = primitive_centers[mask].mean(dim=0)
            semantic_centers = F.normalize(semantic_centers, dim=1)
        
        # 19クラス分類器を作成
        eval_classifier = get_fixclassifier(
            in_channel=self.config.feats_dim,
            centroids_num=self.config.semantic_class,
            centroids=semantic_centers
        ).to(self.device)
        
        # 評価の実行（全GPUで分散処理）
        with torch.no_grad():
            o_Acc, m_Acc, m_IoU, s, IoU_dict, distance_metrics, moving_static_metrics = eval_ddp(
                epoch, 
                self.config,
                model_q,
                eval_classifier,
                local_rank=self.local_rank,
                world_size=self.world_size,
                is_main_process=self.is_main_process
            )
            
            # メインプロセスのみログ記録
            if self.is_main_process:
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
            
            # 早期停止判定（メインプロセスで判定、結果をブロードキャスト）
            if self.config.early_stopping and not self.early_stopped:
                if self.is_main_process:
                    self._check_early_stopping(epoch, m_IoU)
                
                # DDP時はearly_stoppedフラグを全プロセスに同期
                if self.use_ddp:
                    early_stopped_tensor = torch.tensor([1 if self.early_stopped else 0], device=self.device)
                    dist.broadcast(early_stopped_tensor, src=0)
                    self.early_stopped = early_stopped_tensor.item() == 1
        
        # DDP同期のためにバリアを設置
        if self.use_ddp:
            dist.barrier()
    
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
        """早期停止判定を行う
        
        注意: このメソッドはメインプロセス（rank=0）でのみ呼び出されることを前提としています。
        呼び出し後、self.early_stoppedフラグを全プロセスにブロードキャストする必要があります。
        """
        # 収束判定しきい値（固定）
        loss_plateau = 0.003
        
        # 改善判定
        if val_miou > self.best_metric_score + self.config.early_stopping_min_delta:
            # 改善
            self.best_metric_score = val_miou
            self.best_epoch = epoch
            self.patience_counter = 0
            
            # ベストモデルの保存（DDPラップを解除）
            model_q_state = self.model_q.module.state_dict() if self.use_ddp else self.model_q.state_dict()
            torch.save(
                model_q_state, 
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
        """ベストモデルをロードする（DDP対応版）"""
        best_model_path = join(self.config.save_path, 'best_model.pth')
        best_classifier_path = join(self.config.save_path, 'best_classifier.pth')
        
        if os.path.exists(best_model_path) and os.path.exists(best_classifier_path):
            # DDPラップを解除してロード
            model_q = self.model_q.module if self.use_ddp else self.model_q
            model_q.load_state_dict(torch.load(best_model_path, map_location=self.device))
            self.classifier.load_state_dict(torch.load(best_classifier_path, map_location=self.device))
            self.logger.info(f'ベストモデル（エポック {self.best_epoch}）をロードしました')
        else:
            self.logger.warning('ベストモデルファイルが見つかりません') 
