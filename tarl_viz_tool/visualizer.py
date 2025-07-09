"""
Point Cloud Visualizer

Open3Dを使用した点群ビジュアライザー
"""

import numpy as np
import open3d as o3d
from typing import Dict, Any, Optional, Callable
import logging
from datetime import datetime
import os


class PointCloudVisualizer:
    """点群ビジュアライザークラス"""
    
    # 表示モード定数
    MODE_GROUND_TRUTH = 'ground_truth'
    MODE_KMEANS = 'kmeans'
    MODE_HDBSCAN = 'hdbscan'
    
    def __init__(self, window_width: int = 1200, window_height: int = 800,
                 point_size: float = 2.0, logger: logging.Logger = None):
        self.window_width = window_width
        self.window_height = window_height
        self.point_size = point_size
        self.logger = logger or logging.getLogger(__name__)
        
        # ビジュアライザーの初期化
        self.vis = None
        self.point_cloud = None
        self.current_mode = self.MODE_GROUND_TRUTH
        
        # データ管理
        self.current_coords = None
        self.current_ground_truth = None
        self.current_clustering_results = None
        self.current_frame_info = None
        
        # コールバック関数
        self.frame_change_callback = None
        self.mode_change_callback = None
        self.parameter_change_callback = None
        
        # SemanticKITTI色定義
        self.semantic_colors = {
            0: [0, 0, 0], 1: [100, 150, 245], 2: [100, 230, 245],
            3: [30, 60, 150], 4: [80, 30, 180], 5: [0, 0, 255],
            6: [255, 30, 30], 7: [255, 40, 200], 8: [150, 30, 90],
            9: [255, 0, 255], 10: [255, 150, 255], 11: [75, 0, 75],
            12: [175, 0, 75], 13: [255, 200, 0], 14: [255, 120, 50],
            15: [0, 175, 0], 16: [135, 60, 0], 17: [150, 240, 80],
            18: [255, 240, 150], 19: [255, 0, 0]
        }
        
        self.logger.info(f"ビジュアライザーを初期化: {window_width}x{window_height}")
    
    def set_callbacks(self, frame_change_callback: Callable = None,
                     mode_change_callback: Callable = None,
                     parameter_change_callback: Callable = None):
        """コールバック関数を設定"""
        self.frame_change_callback = frame_change_callback
        self.mode_change_callback = mode_change_callback
        self.parameter_change_callback = parameter_change_callback
    
    def initialize_visualizer(self):
        """ビジュアライザーの初期化"""
        if self.vis is not None:
            return
        
        # ビジュアライザーの作成
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(
            window_name="TARL Clustering Visualization Tool",
            width=self.window_width,
            height=self.window_height
        )
        
        # 点群オブジェクトの作成
        self.point_cloud = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.point_cloud)
        
        # キーボードコールバックの設定
        self._setup_keyboard_callbacks()
        
        # 視点の設定
        self._setup_default_view()
        
        self.logger.info("ビジュアライザーを初期化しました")
    
    def _setup_keyboard_callbacks(self):
        """キーボードコールバックの設定"""
        # フレーム移動 (←→キー)
        self.vis.register_key_callback(262, self._on_key_right)  # →
        self.vis.register_key_callback(263, self._on_key_left)   # ←
        
        # 表示モード切り替え (1-3キー)
        self.vis.register_key_callback(49, self._on_key_1)  # 1: Ground-truth
        self.vis.register_key_callback(50, self._on_key_2)  # 2: K-means
        self.vis.register_key_callback(51, self._on_key_3)  # 3: HDBSCAN
        
        # 視点リセット (Rキー)
        self.vis.register_key_callback(82, self._on_key_reset_view)  # R
        
        # スクリーンショット (Sキー)
        self.vis.register_key_callback(83, self._on_key_screenshot)  # S
        
        # 終了 (QキーまたはESC)
        self.vis.register_key_callback(81, self._on_key_quit)   # Q
        self.vis.register_key_callback(256, self._on_key_quit)  # ESC
        
        # ヘルプ表示 (Hキー)
        self.vis.register_key_callback(72, self._on_key_help)  # H
    
    def _setup_default_view(self):
        """デフォルト視点の設定"""
        ctr = self.vis.get_view_control()
        ctr.set_front([0.0, 0.0, 1.0])
        ctr.set_up([0.0, 1.0, 0.0])
        ctr.set_lookat([0.0, 0.0, 0.0])
        ctr.set_zoom(0.8)
    
    def update_display(self, coords: np.ndarray, ground_truth: np.ndarray,
                      clustering_results: Dict[str, Any], frame_info: Dict[str, Any]):
        """表示を更新"""
        if self.vis is None:
            self.initialize_visualizer()
        
        # データを保存
        self.current_coords = coords
        self.current_ground_truth = ground_truth
        self.current_clustering_results = clustering_results
        self.current_frame_info = frame_info
        
        # 現在のモードに応じて表示を更新
        self._update_point_cloud()
        
        # 統計情報の更新
        self._update_info_display()
        
        # ビジュアライザーの更新
        self.vis.update_geometry(self.point_cloud)
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def _update_point_cloud(self):
        """現在のモードに応じて点群を更新"""
        if self.current_coords is None:
            return
        
        # 座標の設定
        self.point_cloud.points = o3d.utility.Vector3dVector(self.current_coords)
        
        # 色の設定
        colors = self._get_colors_for_current_mode()
        if colors is not None:
            self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
        
        # 点のサイズ設定
        render_option = self.vis.get_render_option()
        render_option.point_size = self.point_size
    
    def _get_colors_for_current_mode(self) -> Optional[np.ndarray]:
        """現在のモードに応じた色を取得"""
        n_points = len(self.current_coords)
        
        if self.current_mode == self.MODE_GROUND_TRUTH:
            # Ground-truthの色
            return self._get_semantic_colors(self.current_ground_truth, n_points)
        
        elif self.current_mode == self.MODE_KMEANS:
            # K-meansクラスタリング結果の色
            if (self.current_clustering_results and 
                'kmeans' in self.current_clustering_results and
                self.current_clustering_results['kmeans'] is not None):
                
                result = self.current_clustering_results['kmeans']
                return self._get_clustering_colors(result['labels'], n_points)
            else:
                # K-meansが失敗した場合は黒色
                return np.zeros((n_points, 3), dtype=np.float32)
        
        elif self.current_mode == self.MODE_HDBSCAN:
            # HDBSCANクラスタリング結果の色
            if (self.current_clustering_results and 
                'hdbscan' in self.current_clustering_results and
                self.current_clustering_results['hdbscan'] is not None):
                
                result = self.current_clustering_results['hdbscan']
                return self._get_clustering_colors(result['labels'], n_points)
            else:
                # HDBSCANが失敗した場合は黒色
                return np.zeros((n_points, 3), dtype=np.float32)
        
        return None
    
    def _get_semantic_colors(self, labels: np.ndarray, n_points: int) -> np.ndarray:
        """SemanticKITTIラベルに基づく色を取得"""
        if labels is None or len(labels) == 0:
            # ラベルがない場合は黒色
            return np.zeros((n_points, 3), dtype=np.float32)
        
        colors = np.zeros((n_points, 3), dtype=np.float32)
        
        for i, label in enumerate(labels):
            if i >= n_points:
                break
            
            if label in self.semantic_colors:
                color = self.semantic_colors[label]
                colors[i] = [color[0]/255.0, color[1]/255.0, color[2]/255.0]
            else:
                # 未知のラベルは白色
                colors[i] = [1.0, 1.0, 1.0]
        
        return colors
    
    def _get_clustering_colors(self, labels: np.ndarray, n_points: int) -> np.ndarray:
        """クラスタリングラベルに基づく色を取得"""
        if labels is None or len(labels) == 0:
            return np.zeros((n_points, 3), dtype=np.float32)
        
        colors = np.zeros((n_points, 3), dtype=np.float32)
        unique_labels = np.unique(labels)
        
        # カラーマップの生成
        color_map = self._generate_color_map(unique_labels)
        
        for i, label in enumerate(labels):
            if i >= n_points:
                break
            colors[i] = color_map[label]
        
        return colors
    
    def _generate_color_map(self, unique_labels: np.ndarray) -> Dict[int, np.ndarray]:
        """ラベルに対応するカラーマップを生成"""
        color_map = {}
        
        # 基本色のパレット
        base_colors = [
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
            [1.0, 0.5, 0.0], [0.5, 0.0, 1.0], [0.0, 0.5, 0.0],
            [0.5, 0.5, 0.0], [0.0, 0.0, 0.5], [0.5, 0.0, 0.0],
            [1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0],
        ]
        
        for i, label in enumerate(unique_labels):
            if label == -1:
                # ノイズ（HDBSCAN）は黒色
                color_map[label] = np.array([0.0, 0.0, 0.0])
            else:
                # 色を循環使用
                color_idx = i % len(base_colors)
                color_map[label] = np.array(base_colors[color_idx])
        
        return color_map
    
    def _update_info_display(self):
        """統計情報の表示更新"""
        if not self.current_frame_info:
            return
        
        # ウィンドウタイトルの更新
        title = f"TARL Clustering Tool - Seq:{self.current_frame_info['seq']} "
        title += f"Frame:{self.current_frame_info['frame']} "
        title += f"Mode:{self.current_mode.upper()} "
        title += f"Points:{self.current_frame_info['total_points']}"
        
        # クラスタリング結果があれば追加
        if self.current_clustering_results:
            if self.current_mode == self.MODE_KMEANS and 'kmeans' in self.current_clustering_results:
                result = self.current_clustering_results['kmeans']
                if result:
                    title += f" Clusters:{result['n_clusters']} Time:{result['execution_time']:.2f}s"
            elif self.current_mode == self.MODE_HDBSCAN and 'hdbscan' in self.current_clustering_results:
                result = self.current_clustering_results['hdbscan']
                if result:
                    title += f" Clusters:{result['n_clusters']} Noise:{result['n_noise']} Time:{result['execution_time']:.2f}s"
        
        self.logger.info(title)
    
    def _on_key_right(self, vis):
        """右キー: 次のフレーム"""
        if self.frame_change_callback:
            self.frame_change_callback(1)
        return False
    
    def _on_key_left(self, vis):
        """左キー: 前のフレーム"""
        if self.frame_change_callback:
            self.frame_change_callback(-1)
        return False
    
    def _on_key_1(self, vis):
        """1キー: Ground-truthモード"""
        self.current_mode = self.MODE_GROUND_TRUTH
        self.logger.info("表示モード: Ground-truth")
        self._update_point_cloud()
        if self.mode_change_callback:
            self.mode_change_callback(self.current_mode)
        return False
    
    def _on_key_2(self, vis):
        """2キー: K-meansモード"""
        self.current_mode = self.MODE_KMEANS
        self.logger.info("表示モード: K-means")
        self._update_point_cloud()
        if self.mode_change_callback:
            self.mode_change_callback(self.current_mode)
        return False
    
    def _on_key_3(self, vis):
        """3キー: HDBSCANモード"""
        self.current_mode = self.MODE_HDBSCAN
        self.logger.info("表示モード: HDBSCAN")
        self._update_point_cloud()
        if self.mode_change_callback:
            self.mode_change_callback(self.current_mode)
        return False
    
    def _on_key_reset_view(self, vis):
        """Rキー: 視点リセット"""
        self._setup_default_view()
        self.logger.info("視点をリセットしました")
        return False
    
    def _on_key_screenshot(self, vis):
        """Sキー: スクリーンショット"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tarl_viz_{timestamp}.png"
        
        # 保存ディレクトリの確認
        os.makedirs("tarl_viz_results", exist_ok=True)
        filepath = os.path.join("tarl_viz_results", filename)
        
        # スクリーンショットの保存
        self.vis.capture_screen_image(filepath)
        self.logger.info(f"スクリーンショットを保存: {filepath}")
        return False
    
    def _on_key_help(self, vis):
        """Hキー: ヘルプ表示"""
        help_text = """
        TARL Clustering Visualization Tool - 操作方法
        
        フレーム操作: ←/→ : 前/次のフレーム
        表示モード: 1:Ground-truth, 2:K-means, 3:HDBSCAN
        その他: R:視点リセット, S:スクリーンショット, Q/ESC:終了
        """
        self.logger.info(help_text)
        return False
    
    def _on_key_quit(self, vis):
        """Q/ESCキー: 終了"""
        self.logger.info("ビジュアライザーを終了します")
        self.vis.destroy_window()
        return True
    
    def run(self):
        """ビジュアライザーのメインループ"""
        if self.vis is None:
            self.initialize_visualizer()
        
        # ヘルプメッセージの表示
        self.logger.info("ビジュアライザーを開始しました")
        self.logger.info("操作方法は H キーで表示されます")
        
        # イベントループ
        self.vis.run()
    
    def close(self):
        """ビジュアライザーを閉じる"""
        if self.vis is not None:
            self.vis.destroy_window()
            self.vis = None
        self.logger.info("ビジュアライザーを終了しました")
    
    def update_point_size(self, size: float):
        """点のサイズを更新"""
        self.point_size = size
        if self.vis is not None:
            render_option = self.vis.get_render_option()
            render_option.point_size = size
            self.logger.info(f"点のサイズを{size}に更新")
    
    def get_current_mode(self) -> str:
        """現在の表示モードを取得"""
        return self.current_mode
    
    def set_mode(self, mode: str):
        """表示モードを設定"""
        if mode in [self.MODE_GROUND_TRUTH, self.MODE_KMEANS, self.MODE_HDBSCAN]:
            self.current_mode = mode
            self._update_point_cloud()
            self.logger.info(f"表示モードを{mode}に設定")
        else:
            self.logger.warning(f"不明な表示モード: {mode}") 