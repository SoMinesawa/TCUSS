import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import random
import torch
import torch.multiprocessing as multiprocessing
from datasets.SemanticKITTI import KITTItrain, cfl_collate_fn, KITTItcuss, cfl_collate_fn_tcuss
from torch.utils.data import DataLoader
import logging

from lib.config import TCUSSConfig
from lib.trainer import TCUSSTrainer


def worker_init_fn(worker_id):
    """データローダーワーカーの初期化関数"""
    # GPU 0もデータ作成に使用するように変更
    gpu_id = worker_id % torch.cuda.device_count()  # GPU0も含めたラウンドロビン
    torch.cuda.set_device(gpu_id)
    # WorkerごとにユニークなシードをNumPyに設定
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


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


def setup_datasets(config):
    """データセットとデータローダーの設定"""
    # トレーニングデータセットとデータローダー
    trainset = KITTItcuss(config)
    train_loader = DataLoader(
        trainset, 
        batch_size=config.batch_size[0], 
        shuffle=True, 
        collate_fn=cfl_collate_fn_tcuss(), 
        num_workers=config.workers, 
        pin_memory=True, 
        worker_init_fn=worker_init_fn
    )
    
    # ランダムにシーンを選択
    scene_idx = np.random.choice(19130, config.select_num, replace=False).tolist()  # SemanticKITTIは合計で19130のトレーニングサンプルを持つ
    
    # クラスタリングデータセットとデータローダー
    clusterset = KITTItrain(config, scene_idx, 'train')
    cluster_loader = DataLoader(
        clusterset, 
        batch_size=1, 
        collate_fn=cfl_collate_fn(), 
        num_workers=config.cluster_workers, 
        pin_memory=True
    )
    
    return train_loader, cluster_loader


def main():
    """メイン関数"""
    # 設定の読み込み
    config = TCUSSConfig.from_parse_args()
    
    # マルチプロセスの設定
    if multiprocessing.get_start_method() == 'fork':
        multiprocessing.set_start_method('spawn', force=True)
    
    # 保存ディレクトリの作成
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    
    # ロガーの設定
    logger = set_logger(os.path.join(config.save_path, 'train.log'))
    
    # 乱数シードの設定
    set_seed(config.seed)
    
    # データセットの設定
    train_loader, cluster_loader = setup_datasets(config)
    
    # トレーナーの初期化と実行
    trainer = TCUSSTrainer(config, logger)
    trainer.train(train_loader, cluster_loader)


if __name__ == '__main__':
    main()
