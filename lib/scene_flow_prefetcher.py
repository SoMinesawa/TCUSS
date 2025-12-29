"""
SceneFlow Prefetcher

数バッチ先のScene Flowを事前計算するプリフェッチャー。
学習ループとは非同期に動作し、VoteFlowがボトルネックになっている場合の
学習速度を大幅に向上させる。

アーキテクチャ:
  - バックグラウンドスレッドがDataLoaderから先読み
  - 複数GPUで**異なるバッチを並列計算**（各GPUは1バッチの全サンプルを処理）
  - プリフェッチバッチ数 = GPU数 × 2
  - 結果をキャッシュに保存
  - メインループはキャッシュから結果を取得
"""

import threading
import queue
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Iterator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging


@dataclass 
class PrefetchItem:
    """プリフェッチアイテム"""
    batch_idx: int
    stc_data: Dict
    flow_results: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None


class SceneFlowPrefetcher:
    """
    Scene Flowプリフェッチャー
    
    複数バッチを並列で事前計算。各GPUは1バッチの全サンプルを担当。
    プリフェッチバッチ数 = GPU数 × 2（デフォルト）
    """
    
    def __init__(
        self,
        voteflow_wrappers: List,
        prefetch_batches: Optional[int] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            voteflow_wrappers: VoteFlowWrapperのリスト（各GPU用）
            prefetch_batches: 先行計算するバッチ数（Noneの場合はGPU数×2）
            logger: ロガー
        """
        self.voteflow_wrappers = voteflow_wrappers
        n_gpus = len(voteflow_wrappers)
        # デフォルトはGPU数 × 2（各GPUが2ラウンド分を担当）
        self.prefetch_batches = prefetch_batches if prefetch_batches is not None else n_gpus * 2
        self.logger = logger or logging.getLogger(__name__)
        
        # キュー
        self.pending_queue: queue.Queue = queue.Queue(maxsize=self.prefetch_batches + 2)
        self.result_cache: Dict[int, PrefetchItem] = {}
        self.cache_lock = threading.Lock()
        
        # 状態
        self.running = False
        self.workers: List[threading.Thread] = []
        self.dataloader_exhausted = False
        
        # 統計
        self.stats = {
            'prefetched': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'wait_time': 0.0,
            'compute_time': 0.0,
        }
        
        # データローダーイテレータ（start時に設定）
        self._dataloader_iter: Optional[Iterator] = None
        self._dataloader_lock = threading.Lock()
        self._next_batch_idx = 0
        self._total_batches = 0
    
    def start(self, dataloader, total_batches: int):
        """
        プリフェッチャーを開始
        
        Args:
            dataloader: PyTorchのDataLoader
            total_batches: 総バッチ数
        """
        if self.running:
            return
        
        self.running = True
        self.dataloader_exhausted = False
        self._dataloader_iter = iter(dataloader)
        self._next_batch_idx = 0
        self._total_batches = total_batches
        self.result_cache.clear()
        
        # プリフェッチワーカーを開始
        worker = threading.Thread(
            target=self._prefetch_worker,
            daemon=True,
            name="SceneFlowPrefetcher"
        )
        worker.start()
        self.workers.append(worker)
        
        self.logger.info(f'SceneFlowPrefetcher開始: {len(self.voteflow_wrappers)} GPUs, '
                        f'prefetch_batches={self.prefetch_batches}')
    
    def stop(self):
        """プリフェッチャーを停止"""
        self.running = False
        
        # ワーカーの終了を待つ
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        self.workers.clear()
        self._dataloader_iter = None
        
        self.logger.info(f'SceneFlowPrefetcher停止: '
                        f'prefetched={self.stats["prefetched"]}, '
                        f'cache_hits={self.stats["cache_hits"]}, '
                        f'cache_misses={self.stats["cache_misses"]}, '
                        f'wait_time={self.stats["wait_time"]:.2f}s')
    
    def _prefetch_worker(self):
        """プリフェッチワーカーのメインループ
        
        複数バッチを複数GPUで並列計算。各GPUは1バッチの全サンプルを担当。
        """
        n_gpus = len(self.voteflow_wrappers)
        
        while self.running:
            # キャッシュサイズが上限以下の場合、次のバッチを処理
            with self.cache_lock:
                cache_size = len(self.result_cache)
            
            if cache_size >= self.prefetch_batches:
                # キャッシュが十分にある場合は少し待機
                time.sleep(0.01)
                continue
            
            if self.dataloader_exhausted:
                time.sleep(0.01)
                continue
            
            # 複数バッチを取得（GPU数分）
            batches_to_process = []
            
            with self._dataloader_lock:
                if self._dataloader_iter is None:
                    break
                
                # GPU数分のバッチを取得（可能な限り）
                for _ in range(n_gpus):
                    batch_idx = self._next_batch_idx
                    
                    if batch_idx >= self._total_batches:
                        self.dataloader_exhausted = True
                        break
                    
                    try:
                        data = next(self._dataloader_iter)
                        self._next_batch_idx += 1
                        batches_to_process.append((batch_idx, data))
                    except StopIteration:
                        self.dataloader_exhausted = True
                        break
            
            if not batches_to_process:
                continue
            
            # 複数バッチを複数GPUで並列計算
            try:
                t_start = time.perf_counter()
                self._process_batches_parallel(batches_to_process)
                t_end = time.perf_counter()
                self.stats['compute_time'] += t_end - t_start
                
            except Exception as e:
                self.logger.error(f'Prefetch error: {e}')
                import traceback
                traceback.print_exc()
    
    def _process_batches_parallel(self, batches_to_process: List[Tuple[int, Any]]):
        """複数バッチを複数GPUで並列処理
        
        各GPUは1バッチの全サンプルを担当。
        
        Args:
            batches_to_process: [(batch_idx, data), ...] のリスト
        """
        n_gpus = len(self.voteflow_wrappers)
        n_batches = len(batches_to_process)
        
        def process_single_batch(gpu_id: int, batch_idx: int, data: Any) -> PrefetchItem:
            """1つのGPUで1バッチを処理"""
            growsp_t1_data, growsp_t2_data, stc_data = data
            
            if stc_data is None:
                item = PrefetchItem(
                    batch_idx=batch_idx,
                    stc_data=None,
                    flow_results=None
                )
            else:
                # このGPUで全サンプルのScene Flowを計算
                flow_results = self._compute_scene_flow_single_gpu(stc_data, gpu_id)
                item = PrefetchItem(
                    batch_idx=batch_idx,
                    stc_data=stc_data,
                    flow_results=flow_results
                )
            
            item.growsp_t1_data = growsp_t1_data
            item.growsp_t2_data = growsp_t2_data
            return item
        
        # ThreadPoolExecutorで並列実行
        with ThreadPoolExecutor(max_workers=n_batches) as executor:
            futures = {}
            for i, (batch_idx, data) in enumerate(batches_to_process):
                gpu_id = i % n_gpus  # GPUをラウンドロビンで割り当て
                future = executor.submit(process_single_batch, gpu_id, batch_idx, data)
                futures[future] = batch_idx
            
            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    item = future.result()
                    with self.cache_lock:
                        self.result_cache[batch_idx] = item
                        self.stats['prefetched'] += 1
                except Exception as e:
                    self.logger.error(f'Batch {batch_idx} processing error: {e}')
    
    def _compute_scene_flow_single_gpu(
        self, 
        stc_data: Dict, 
        gpu_id: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        1つのGPUで1バッチの全サンプルのScene Flowを計算
        
        Args:
            stc_data: STCデータ
            gpu_id: 使用するGPUのインデックス
            
        Returns:
            各サンプルのScene Flow結果のリスト
        """
        coords_t_original = stc_data['coords_t_original']
        coords_t1_original = stc_data['coords_t1_original']
        pose_t = stc_data['pose_t']
        pose_t1 = stc_data['pose_t1']
        
        n_samples = len(coords_t_original)
        
        # タスクを準備
        points_t_list = []
        points_t1_list = []
        pose_t_list = []
        pose_t1_list = []
        
        for i in range(n_samples):
            points_t = coords_t_original[i]
            points_t1 = coords_t1_original[i]
            p_t = pose_t[i]
            p_t1 = pose_t1[i]
            
            # numpy配列に変換
            if torch.is_tensor(points_t):
                points_t = points_t.numpy()
            if torch.is_tensor(points_t1):
                points_t1 = points_t1.numpy()
            if torch.is_tensor(p_t):
                p_t = p_t.numpy()
            if torch.is_tensor(p_t1):
                p_t1 = p_t1.numpy()
            
            points_t_list.append(points_t)
            points_t1_list.append(points_t1)
            pose_t_list.append(p_t)
            pose_t1_list.append(p_t1)
        
        # 指定されたGPUで全サンプルを処理
        wrapper = self.voteflow_wrappers[gpu_id]
        return wrapper.compute_flow_batch(
            points_t_list, points_t1_list, pose_t_list, pose_t1_list
        )
    
    def get_batch(self, batch_idx: int, timeout: float = 60.0) -> Optional[Tuple]:
        """
        指定バッチのデータと事前計算済みScene Flow結果を取得
        
        Args:
            batch_idx: バッチインデックス
            timeout: タイムアウト秒数
            
        Returns:
            (growsp_t1_data, growsp_t2_data, stc_data, flow_results) または None
        """
        start_time = time.time()
        
        while True:
            with self.cache_lock:
                if batch_idx in self.result_cache:
                    item = self.result_cache.pop(batch_idx)
                    self.stats['cache_hits'] += 1
                    
                    elapsed = time.time() - start_time
                    if elapsed > 0.01:
                        self.stats['wait_time'] += elapsed
                    
                    return (
                        item.growsp_t1_data,
                        item.growsp_t2_data,
                        item.stc_data,
                        item.flow_results
                    )
            
            elapsed = time.time() - start_time
            if elapsed > timeout:
                self.logger.warning(f'Timeout waiting for batch {batch_idx}')
                self.stats['cache_misses'] += 1
                return None
            
            # 少し待機
            time.sleep(0.001)
    
    def get_cache_size(self) -> int:
        """現在のキャッシュサイズを取得"""
        with self.cache_lock:
            return len(self.result_cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        return {
            **self.stats,
            'cache_size': self.get_cache_size(),
        }







