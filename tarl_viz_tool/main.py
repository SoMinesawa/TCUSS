#!/usr/bin/env python3
"""
TARL Clustering Visualization Tool - Main Entry Point

Usage:
    python -m tarl_viz_tool.main --data_path data/dataset/semantickitti/dataset/sequences --seq 00
"""

import argparse
import sys
import os
from pathlib import Path

# TCUSSのルートディレクトリをパスに追加
tcuss_root = Path(__file__).parent.parent
sys.path.insert(0, str(tcuss_root))

from .data_loader import SemanticKITTILoader
from .preprocessor import DataPreprocessor
from .clustering import ClusteringManager
from .visualizer import PointCloudVisualizer
from .utils import setup_logging, validate_paths


def parse_arguments():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(
        description='TARL Clustering Visualization Tool - ハイパーパラメータ最適化支援ツール',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # データパス設定
    parser.add_argument('--data_path', type=str, 
                       default='data/dataset/semantickitti/dataset/sequences',
                       help='SemanticKITTIデータセットのsequencesディレクトリパス')
    parser.add_argument('--original_data_path', type=str,
                       default='data/dataset/semantickitti/dataset/sequences', 
                       help='オリジナルデータパス（ポーズ情報用）')
    parser.add_argument('--patchwork_path', type=str,
                       default='data/users/minesawa/semantickitti/patchwork',
                       help='地面ラベルのパス')
    
    # 表示設定
    parser.add_argument('--seq', type=str, default='00',
                       help='表示するシーケンス番号（例: 00, 01, 02...）')
    parser.add_argument('--start_frame', type=int, default=0,
                       help='開始フレーム番号')
    parser.add_argument('--end_frame', type=int, default=None,
                       help='終了フレーム番号（指定しない場合は自動検出）')
    
    # 前処理パラメータ
    parser.add_argument('--voxel_size', type=float, default=0.15,
                       help='ボクセルサイズ（TCUSSデフォルト値）')
    parser.add_argument('--r_crop', type=float, default=50.0,
                       help='クロッピング半径（メートル）')
    parser.add_argument('--scan_window', type=int, default=12,
                       help='時系列集約のウィンドウサイズ')
    
    # HDBSCANパラメータ
    parser.add_argument('--min_cluster_size', type=int, default=20,
                       help='HDBSCANの最小クラスターサイズ')
    parser.add_argument('--min_samples', type=int, default=50,
                       help='HDBSCANの最小サンプル数')
    parser.add_argument('--cluster_selection_epsilon', type=float, default=0.0,
                       help='HDBSCANのクラスター選択エプシロン')
    
    # K-meansパラメータ
    parser.add_argument('--n_clusters', type=int, default=50,
                       help='K-meansのクラスター数')
    parser.add_argument('--max_iter', type=int, default=300,
                       help='K-meansの最大イテレーション数')
    
    # 表示オプション
    parser.add_argument('--window_width', type=int, default=1200,
                       help='ウィンドウ幅')
    parser.add_argument('--window_height', type=int, default=800,
                       help='ウィンドウ高さ')
    parser.add_argument('--point_size', type=float, default=2.0,
                       help='点のサイズ')
    parser.add_argument('--verbose', action='store_true',
                       help='詳細なログ出力')
    
    # デバッグオプション
    parser.add_argument('--debug', action='store_true',
                       help='デバッグモード')
    parser.add_argument('--save_results', action='store_true',
                       help='クラスタリング結果を保存')
    parser.add_argument('--output_dir', type=str, default='tarl_viz_results',
                       help='結果保存ディレクトリ')
    parser.add_argument('--headless', action='store_true',
                       help='ヘッドレスモード（GUI表示なし）')
    parser.add_argument('--headless', action='store_true',
                       help='ヘッドレスモード（GUI表示なし）')
    
    return parser.parse_args()


class TARLVisualizationApp:
    """TARLビジュアライゼーションアプリケーションのメインクラス"""
    
    def __init__(self, args):
        self.args = args
        self.logger = setup_logging(debug=args.debug)
        
        # コンポーネントの初期化
        self.data_loader = None
        self.preprocessor = None
        self.clustering_manager = None
        self.visualizer = None
        
        # 状態管理
        self.current_seq = args.seq
        self.current_frame = args.start_frame
        self.max_frame = args.end_frame
        
    def initialize(self):
        """アプリケーションの初期化"""
        self.logger.info("TARLビジュアライゼーションツールを初期化中...")
        
        # パスの検証
        if not validate_paths(self.args):
            self.logger.error("データパスの検証に失敗しました")
            return False
        
        try:
            # データローダーの初期化
            self.data_loader = SemanticKITTILoader(
                data_path=self.args.data_path,
                original_data_path=self.args.original_data_path,
                patchwork_path=self.args.patchwork_path,
                logger=self.logger,
                use_ply=False  # BINファイルを使用
            )
            
            # 前処理器の初期化
            self.preprocessor = DataPreprocessor(
                voxel_size=self.args.voxel_size,
                r_crop=self.args.r_crop,
                scan_window=self.args.scan_window,
                logger=self.logger
            )
            
            # クラスタリング管理の初期化
            self.clustering_manager = ClusteringManager(
                # HDBSCANパラメータ
                min_cluster_size=self.args.min_cluster_size,
                min_samples=self.args.min_samples,
                cluster_selection_epsilon=self.args.cluster_selection_epsilon,
                # K-meansパラメータ
                n_clusters=self.args.n_clusters,
                max_iter=self.args.max_iter,
                logger=self.logger
            )
            
            # ビジュアライザーの初期化
            self.visualizer = PointCloudVisualizer(
                window_width=self.args.window_width,
                window_height=self.args.window_height,
                point_size=self.args.point_size,
                logger=self.logger
            )
            
            # 最大フレーム数の自動検出
            if self.max_frame is None:
                self.max_frame = self.data_loader.get_max_frame(self.current_seq)
                self.logger.info(f"シーケンス{self.current_seq}の最大フレーム数: {self.max_frame}")
            
            # ビジュアライザーにコールバックを設定
            self.visualizer.set_callbacks(
                frame_change_callback=self.on_frame_change,
                mode_change_callback=self.on_mode_change,
                parameter_change_callback=self.on_parameter_change
            )
            
            self.logger.info("初期化完了")
            return True
            
        except Exception as e:
            self.logger.error(f"初期化中にエラーが発生しました: {e}")
            return False
    
    def on_frame_change(self, direction):
        """フレーム変更時のコールバック"""
        new_frame = self.current_frame + direction
        
        # 範囲チェック
        if new_frame < 0 or new_frame > self.max_frame:
            self.logger.warning(f"フレーム{new_frame}は範囲外です (0-{self.max_frame})")
            return
        
        self.current_frame = new_frame
        self.logger.info(f"フレーム{self.current_frame}に移動")
        
        # データの更新と表示
        self.update_display()
    
    def on_mode_change(self, mode):
        """表示モード変更時のコールバック"""
        self.logger.info(f"表示モードを{mode}に変更")
        self.update_display()
    
    def on_parameter_change(self, param_name, value):
        """パラメータ変更時のコールバック"""
        self.logger.info(f"パラメータ{param_name}を{value}に変更")
        
        # クラスタリングパラメータの更新
        if hasattr(self.clustering_manager, param_name):
            setattr(self.clustering_manager, param_name, value)
            self.update_display()
    
    def update_display(self):
        """表示の更新"""
        try:
            # データの読み込み
            frame_data = self.data_loader.load_frame(self.current_seq, self.current_frame)
            if frame_data is None:
                self.logger.error(f"フレーム{self.current_frame}のデータ読み込みに失敗")
                return
            
            # 前処理の実行
            processed_data = self.preprocessor.process_frame(frame_data)
            
            # クラスタリングの実行
            clustering_results = self.clustering_manager.cluster_all_methods(processed_data)
            
            # ビジュアライゼーションの更新
            self.visualizer.update_display(
                coords=processed_data['coords'],
                ground_truth=processed_data['labels'],
                clustering_results=clustering_results,
                frame_info={
                    'seq': self.current_seq,
                    'frame': self.current_frame,
                    'total_points': len(processed_data['coords'])
                }
            )
            
        except Exception as e:
            self.logger.error(f"表示更新中にエラーが発生: {e}")
    
    def run(self):
        """アプリケーションの実行"""
        if not self.initialize():
            return False
        
        # 初期表示
        self.update_display()
        
        # ビジュアライザーの実行（イベントループ）
        self.logger.info("ビジュアライゼーションを開始します")
        self.logger.info("操作方法:")
        self.logger.info("  ←→: フレーム移動")
        self.logger.info("  1-3: 表示モード切り替え (Ground-truth/K-means/HDBSCAN)")
        self.logger.info("  R: 視点リセット")
        self.logger.info("  S: スクリーンショット保存")
        self.logger.info("  Q/ESC: 終了")
        
        self.visualizer.run()
        
        return True


def main():
    """メイン関数"""
    args = parse_arguments()
    
    # アプリケーションの作成と実行
    app = TARLVisualizationApp(args)
    success = app.run()
    
    if not success:
        sys.exit(1)
    
    print("TARLビジュアライゼーションツールを終了しました")


if __name__ == "__main__":
    main() 