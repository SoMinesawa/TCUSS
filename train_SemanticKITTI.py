import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import random
import torch
import torch.multiprocessing as multiprocessing
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets.SemanticKITTI import (
    KITTItrain, cfl_collate_fn,
    KITTItcuss_stc, cfl_collate_fn_tcuss_stc,
    generate_scene_pairs, get_unique_scene_indices
)
import logging

from lib.config import TCUSSConfig
from lib.trainer import TCUSSTrainer


# def worker_init_fn(worker_id):
#     """データローダーワーカーの初期化関数"""
#     # GPU0を避け、GPU1以降でデータ生成を行う（GPUが1枚しかない場合はGPU0を使用）
#     device_count = torch.cuda.device_count()
#     if device_count <= 1:
#         gpu_id = 0
#     else:
#         gpu_id = 1 + (worker_id % (device_count - 1))
#     torch.cuda.set_device(gpu_id)
#     # WorkerごとにユニークなシードをNumPyに設定
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)


def set_logger(log_path):
    """ロガーの設定"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # ファイルへのロギング
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # コンソールへのロギング
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)

    return logger


def set_seed(seed):
    """乱数シードの設定
    
    注意: backward()の[interpolate]関数は完全に決定的ではない可能性がある。
    
    関連するスレッド:
    https://github.com/pytorch/pytorch/issues/7068
    https://discuss.pytorch.org/t/non-deterministic-behavior-of-pytorch-upsample-interpolate/42842?u=sbelharbi
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


def setup_ddp(config):
    """DDP（DistributedDataParallel）の初期化
    
    Returns:
        local_rank: このプロセスのローカルランク（GPU ID）
        world_size: 総プロセス数
        is_main_process: メインプロセスかどうか
    """
    use_ddp = getattr(config, 'use_ddp', False)
    
    if not use_ddp:
        return 0, 1, True
    
    # 環境変数からランク情報を取得（torchrun/torch.distributed.launchで設定される）
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        # DDP無効時
        return 0, 1, True
    
    # プロセスグループの初期化
    backend = getattr(config, 'ddp_backend', 'nccl')
    dist.init_process_group(backend=backend)
    
    # GPUを設定
    torch.cuda.set_device(local_rank)
    
    is_main_process = rank == 0
    
    return local_rank, world_size, is_main_process


def cleanup_ddp():
    """DDPのクリーンアップ"""
    if dist.is_initialized():
        dist.destroy_process_group()


def setup_datasets(config, local_rank=0, world_size=1):
    """データセットとデータローダーの設定（DDP対応・高速化版）
    
    trainsetとclustersetで同じシーンを使用するため、シーンペア生成を一元化。
    固定シードを使用することで、全GPUで同じシーンペアを生成する。
    """
    train_workers = 0 if config.vis else config.workers
    cluster_workers = 0 if config.vis else config.cluster_workers
    
    # DataLoader最適化パラメータ
    persistent_workers = getattr(config, 'persistent_workers', False)
    prefetch_factor = getattr(config, 'prefetch_factor', 4) if train_workers > 0 else None
    
    use_ddp = getattr(config, 'use_ddp', False) and world_size > 1

    # シーンペアを生成（全GPUで同じ結果を得るため、固定シードを使用）
    # これによりtrainsetとclustersetで同じシーンを使用できる
    scene_pairs, scene_idx_t1, scene_idx_t2 = generate_scene_pairs(
        select_num=config.select_num,
        scan_window=config.scan_window,
        seed=config.seed
    )
    scene_idx_all = get_unique_scene_indices(scene_idx_t1, scene_idx_t2)

    # トレーニングデータセット
    trainset = KITTItcuss_stc(config)
    trainset.set_scene_pairs(scene_idx_t1, scene_idx_t2)
    collate_fn = cfl_collate_fn_tcuss_stc()
    
    # DDP用のDistributedSampler
    train_sampler = None
    shuffle = True
    if use_ddp:
        train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=local_rank, shuffle=True)
        shuffle = False  # DistributedSamplerがシャッフルを担当
    
    train_loader = DataLoader(
        trainset, 
        batch_size=config.batch_size[0], 
        shuffle=shuffle, 
        sampler=train_sampler,
        collate_fn=collate_fn, 
        num_workers=train_workers, 
        pin_memory=True, 
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )
    
    # クラスタリングデータセットとデータローダー
    # trainsetと同じシーンを使用（scene_idx_all = t1とt2の重複除去リスト）
    # クラスタリングはメインプロセスのみで実行するため、DistributedSamplerは不要
    # 重要: persistent_workers=False にする必要がある
    cluster_prefetch = prefetch_factor if cluster_workers > 0 else None
    
    clusterset = KITTItrain(config, scene_idx_all, 'train')
    cluster_batch_size = getattr(config, 'cluster_batch_size', 16)
    cluster_loader = DataLoader(
        clusterset, 
        batch_size=cluster_batch_size, 
        collate_fn=cfl_collate_fn(), 
        num_workers=cluster_workers, 
        pin_memory=True,
        persistent_workers=False,  # データセット更新のためFalse必須
        prefetch_factor=cluster_prefetch
    )
    
    return train_loader, cluster_loader, train_sampler


def main():
    """メイン関数（DDP対応版）
    
    DDP使用時は以下のコマンドで起動:
        torchrun --nproc_per_node=8 train_SemanticKITTI.py --config config/default.yaml
    
    シングルGPU時は従来通り:
        python train_SemanticKITTI.py --config config/default.yaml
    """
    # 設定の読み込み
    config = TCUSSConfig.from_parse_args()
    
    # DDP初期化
    local_rank, world_size, is_main_process = setup_ddp(config)
    
    # マルチプロセスの設定
    # if multiprocessing.get_start_method() == 'fork':
    #     multiprocessing.set_start_method('spawn', force=True)
    
    # 保存ディレクトリの作成（メインプロセスのみ）
    if is_main_process and not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    
    # DDP時はバリアで同期
    if dist.is_initialized():
        dist.barrier()
    
    # ロガーの設定（メインプロセスのみファイル出力）
    if is_main_process:
        logger = set_logger(os.path.join(config.save_path, 'train.log'))
    else:
        # サブプロセスはコンソールのみ（簡易版）
        logger = logging.getLogger()
        logger.setLevel(logging.WARNING)  # 警告以上のみ
    
    # 乱数シードの設定（プロセスごとに異なるシード）
    set_seed(config.seed + local_rank)
    
    # データセットの設定
    train_loader, cluster_loader, train_sampler = setup_datasets(config, local_rank, world_size)
    
    # トレーナーの初期化と実行
    trainer = TCUSSTrainer(
        config, 
        logger, 
        local_rank=local_rank, 
        world_size=world_size, 
        is_main_process=is_main_process
    )
    trainer.train(train_loader, cluster_loader, train_sampler)
    
    # DDPクリーンアップ
    cleanup_ddp()


if __name__ == '__main__':
    main()
