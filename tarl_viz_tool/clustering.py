"""
Clustering Manager

K-meansとHDBSCANクラスタリングの実行と管理を行う
TCUSSのget_kmeans_labelsと_clusterize_pcds関数を参考に実装
"""

import numpy as np
import time
from typing import Dict, Any, Optional, Tuple
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# HDBSCANのインポート（cuMLを優先、フォールバック対応）
try:
    from cuml.cluster import HDBSCAN
    HDBSCAN_BACKEND = 'cuml'
    print("✓ cuML HDBSCAN を使用します（GPU加速）")
except ImportError:
    try:
        import hdbscan
        HDBSCAN = hdbscan.HDBSCAN
        HDBSCAN_BACKEND = 'cpu'
        print("✓ CPU HDBSCAN を使用します")
    except ImportError:
        HDBSCAN = None
        HDBSCAN_BACKEND = None
        print("⚠ HDBSCAN が利用できません")


class ClusteringManager:
    """クラスタリング管理クラス"""
    
    def __init__(self, min_cluster_size: int = 20, min_samples: int = 50,
                 cluster_selection_epsilon: float = 0.0, n_clusters: int = 50,
                 max_iter: int = 300, logger: logging.Logger = None):
        """
        Args:
            min_cluster_size: HDBSCANの最小クラスターサイズ
            min_samples: HDBSCANの最小サンプル数
            cluster_selection_epsilon: HDBSCANのクラスター選択エプシロン
            n_clusters: K-meansのクラスター数
            max_iter: K-meansの最大イテレーション数
            logger: ロガー
        """
        # HDBSCANパラメータ
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        
        # K-meansパラメータ
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        
        self.logger = logger or logging.getLogger(__name__)
        
        # クラスタリング結果のキャッシュ
        self._last_results = None
        self._last_data_hash = None
        
        # クラスタリング色の設定
        self._cluster_colors = self._generate_cluster_colors()
        
        hdbscan_info = f"HDBSCAN(min_size={min_cluster_size}, backend={HDBSCAN_BACKEND})"
        self.logger.info(f"クラスタリング管理を初期化: K-means(n={n_clusters}), {hdbscan_info}")
    
    def cluster_all_methods(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """すべてのクラスタリング手法を実行"""
        coords = processed_data['coords']
        
        # データのハッシュ値を計算（キャッシュ用）
        data_hash = hash(coords.tobytes())
        
        if self._last_data_hash == data_hash and self._last_results is not None:
            self.logger.debug("キャッシュされたクラスタリング結果を使用")
            return self._last_results
        
        results = {}
        
        # K-meansクラスタリング
        try:
            kmeans_result = self.cluster_kmeans(coords)
            results['kmeans'] = kmeans_result
            self.logger.info(f"K-means完了: {kmeans_result['n_clusters']}クラスター, {kmeans_result['execution_time']:.3f}秒")
        except Exception as e:
            self.logger.error(f"K-meansクラスタリングエラー: {e}")
            results['kmeans'] = None
        
        # HDBSCANクラスタリング
        try:
            hdbscan_result = self.cluster_hdbscan(coords)
            results['hdbscan'] = hdbscan_result
            self.logger.info(f"HDBSCAN完了: {hdbscan_result['n_clusters']}クラスター, {hdbscan_result['execution_time']:.3f}秒")
        except Exception as e:
            self.logger.error(f"HDBSCANクラスタリングエラー: {e}")
            results['hdbscan'] = None
        
        # キャッシュに保存
        self._last_results = results
        self._last_data_hash = data_hash
        
        return results
    
    def cluster_kmeans(self, coords: np.ndarray) -> Dict[str, Any]:
        """K-meansクラスタリング（TCUSSのget_kmeans_labels準拠）"""
        start_time = time.time()
        
        try:
            # 前処理: 座標の正規化
            scaler = StandardScaler()
            coords_normalized = scaler.fit_transform(coords)
            
            # K-meansクラスタリング
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                max_iter=self.max_iter,
                random_state=42,
                n_init=10
            )
            
            cluster_labels = kmeans.fit_predict(coords_normalized)
            
            # クラスタリング結果の解析
            unique_labels = np.unique(cluster_labels)
            n_clusters = len(unique_labels)
            
            # クラスター統計の計算
            cluster_stats = self._calculate_cluster_stats(coords, cluster_labels)
            
            # 実行時間の計算
            execution_time = time.time() - start_time
            
            result = {
                'labels': cluster_labels,
                'n_clusters': n_clusters,
                'unique_labels': unique_labels,
                'execution_time': execution_time,
                'cluster_centers': kmeans.cluster_centers_,
                'inertia': kmeans.inertia_,
                'stats': cluster_stats,
                'method': 'kmeans',
                'parameters': {
                    'n_clusters': self.n_clusters,
                    'max_iter': self.max_iter
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"K-meansクラスタリングエラー: {e}")
            raise
    
    def cluster_hdbscan(self, coords: np.ndarray) -> Dict[str, Any]:
        """HDBSCANクラスタリング（cuML優先、CPU HDBSCANフォールバック対応）"""
        start_time = time.time()
        
        if HDBSCAN is None:
            raise ImportError("HDBSCANライブラリが利用できません")
        
        try:
            # 前処理: 座標の正規化
            scaler = StandardScaler()
            coords_normalized = scaler.fit_transform(coords)
            
            # cuMLとCPU HDBSCANでパラメータを調整
            if HDBSCAN_BACKEND == 'cuml':
                # cuML HDBSCAN
                hdbscan_clusterer = HDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=self.min_samples,
                    cluster_selection_epsilon=self.cluster_selection_epsilon,
                    metric='euclidean'
                )
                cluster_labels = hdbscan_clusterer.fit_predict(coords_normalized)
                
                # cuMLの場合、結果をnumpy配列に変換
                if hasattr(cluster_labels, 'to_numpy'):
                    cluster_labels = cluster_labels.to_numpy()
                elif hasattr(cluster_labels, '__array__'):
                    cluster_labels = np.array(cluster_labels)
                
            else:
                # CPU HDBSCAN
                import hdbscan as cpu_hdbscan
                hdbscan_clusterer = cpu_hdbscan.HDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=self.min_samples,
                    cluster_selection_epsilon=self.cluster_selection_epsilon,
                    metric='euclidean',
                    algorithm='best'
                )
                cluster_labels = hdbscan_clusterer.fit_predict(coords_normalized)
            
            # 型を確実にnumpy配列にする
            cluster_labels = np.array(cluster_labels).astype(np.int32)
            
            # クラスタリング結果の解析
            unique_labels = np.unique(cluster_labels)
            n_clusters = len(unique_labels[unique_labels != -1])  # -1（ノイズ）を除く
            n_noise = np.sum(cluster_labels == -1)
            
            # クラスター統計の計算
            cluster_stats = self._calculate_cluster_stats(coords, cluster_labels)
            
            # 実行時間の計算
            execution_time = time.time() - start_time
            
            result = {
                'labels': cluster_labels,
                'n_clusters': n_clusters,
                'unique_labels': unique_labels,
                'execution_time': execution_time,
                'n_noise': n_noise,
                'stats': cluster_stats,
                'method': 'hdbscan',
                'backend': HDBSCAN_BACKEND,
                'parameters': {
                    'min_cluster_size': self.min_cluster_size,
                    'min_samples': self.min_samples,
                    'cluster_selection_epsilon': self.cluster_selection_epsilon
                }
            }
            
            # CPU HDBSCANの場合、追加情報を含める
            if HDBSCAN_BACKEND == 'cpu' and hasattr(hdbscan_clusterer, 'cluster_persistence_'):
                result['cluster_persistence'] = hdbscan_clusterer.cluster_persistence_
                result['outlier_scores'] = hdbscan_clusterer.outlier_scores_
            
            return result
            
        except Exception as e:
            self.logger.error(f"HDBSCANクラスタリングエラー: {e}")
            raise
    
    def _calculate_cluster_stats(self, coords: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """クラスター統計の計算"""
        stats = {
            'total_points': len(coords),
            'cluster_sizes': {},
            'cluster_centers': {},
            'cluster_bounds': {}
        }
        
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            mask = labels == label
            cluster_points = coords[mask]
            
            if len(cluster_points) > 0:
                stats['cluster_sizes'][int(label)] = len(cluster_points)
                stats['cluster_centers'][int(label)] = cluster_points.mean(axis=0).tolist()
                stats['cluster_bounds'][int(label)] = {
                    'min': cluster_points.min(axis=0).tolist(),
                    'max': cluster_points.max(axis=0).tolist()
                }
        
        return stats
    
    def get_cluster_colors(self, labels: np.ndarray) -> np.ndarray:
        """クラスターラベルに対応する色を取得"""
        unique_labels = np.unique(labels)
        colors = np.zeros((len(labels), 3), dtype=np.float32)
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if label == -1:
                # ノイズ（HDBSCAN）は黒色
                colors[mask] = [0.0, 0.0, 0.0]
            else:
                # 各クラスターに異なる色を割り当て
                color_idx = label % len(self._cluster_colors)
                colors[mask] = self._cluster_colors[color_idx]
        
        return colors
    
    def _generate_cluster_colors(self) -> np.ndarray:
        """クラスター用の色パレットを生成"""
        # 視覚的に区別しやすい色のパレット
        colors = [
            [1.0, 0.0, 0.0],    # 赤
            [0.0, 1.0, 0.0],    # 緑
            [0.0, 0.0, 1.0],    # 青
            [1.0, 1.0, 0.0],    # 黄
            [1.0, 0.0, 1.0],    # マゼンタ
            [0.0, 1.0, 1.0],    # シアン
            [1.0, 0.5, 0.0],    # オレンジ
            [0.5, 0.0, 1.0],    # 紫
            [0.0, 0.5, 0.0],    # 暗緑
            [0.5, 0.5, 0.0],    # オリーブ
            [0.0, 0.0, 0.5],    # 紺
            [0.5, 0.0, 0.0],    # 栗色
            [1.0, 0.5, 0.5],    # ピンク
            [0.5, 1.0, 0.5],    # 薄緑
            [0.5, 0.5, 1.0],    # 薄青
            [0.8, 0.8, 0.8],    # 薄灰
            [0.4, 0.4, 0.4],    # 暗灰
            [0.6, 0.3, 0.0],    # 茶
            [0.0, 0.6, 0.3],    # 青緑
            [0.3, 0.0, 0.6],    # 青紫
        ]
        
        # 不足分はランダムに生成
        np.random.seed(42)
        while len(colors) < 100:
            colors.append(np.random.rand(3).tolist())
        
        return np.array(colors, dtype=np.float32)
    
    def update_kmeans_parameters(self, n_clusters: int = None, max_iter: int = None):
        """K-meansパラメータの更新"""
        if n_clusters is not None:
            self.n_clusters = n_clusters
            self.logger.info(f"K-meansクラスター数を{n_clusters}に更新")
        
        if max_iter is not None:
            self.max_iter = max_iter
            self.logger.info(f"K-means最大イテレーション数を{max_iter}に更新")
        
        # キャッシュをクリア
        self._last_results = None
        self._last_data_hash = None
    
    def update_hdbscan_parameters(self, min_cluster_size: int = None, 
                                 min_samples: int = None, 
                                 cluster_selection_epsilon: float = None):
        """HDBSCANパラメータの更新"""
        if min_cluster_size is not None:
            self.min_cluster_size = min_cluster_size
            self.logger.info(f"HDBSCAN最小クラスターサイズを{min_cluster_size}に更新")
        
        if min_samples is not None:
            self.min_samples = min_samples
            self.logger.info(f"HDBSCAN最小サンプル数を{min_samples}に更新")
        
        if cluster_selection_epsilon is not None:
            self.cluster_selection_epsilon = cluster_selection_epsilon
            self.logger.info(f"HDBSCANクラスター選択エプシロンを{cluster_selection_epsilon}に更新")
        
        # キャッシュをクリア
        self._last_results = None
        self._last_data_hash = None
    
    def get_clustering_summary(self, results: Dict[str, Any]) -> str:
        """クラスタリング結果のサマリーを取得"""
        summary_lines = []
        
        for method, result in results.items():
            if result is None:
                summary_lines.append(f"{method.upper()}: 実行失敗")
                continue
            
            line = f"{method.upper()}: {result['n_clusters']}クラスター, "
            line += f"{result['execution_time']:.3f}秒"
            
            if method == 'hdbscan':
                if 'n_noise' in result:
                    line += f", ノイズ点: {result['n_noise']}"
                if 'backend' in result:
                    backend_info = "GPU" if result['backend'] == 'cuml' else "CPU"
                    line += f" ({backend_info})"
            
            summary_lines.append(line)
        
        return "\n".join(summary_lines)
    
    def clear_cache(self):
        """キャッシュをクリア"""
        self._last_results = None
        self._last_data_hash = None
        self.logger.debug("クラスタリングキャッシュをクリア")
    
    def save_clustering_results(self, results: Dict[str, Any], output_path: str):
        """クラスタリング結果を保存"""
        import pickle
        
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(results, f)
            self.logger.info(f"クラスタリング結果を保存: {output_path}")
        except Exception as e:
            self.logger.error(f"結果保存エラー: {e}")
    
    def load_clustering_results(self, input_path: str) -> Dict[str, Any]:
        """クラスタリング結果を読み込み"""
        import pickle
        
        try:
            with open(input_path, 'rb') as f:
                results = pickle.load(f)
            self.logger.info(f"クラスタリング結果を読み込み: {input_path}")
            return results
        except Exception as e:
            self.logger.error(f"結果読み込みエラー: {e}")
            return None 