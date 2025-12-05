import argparse
from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class VisualizationConfig:
    """Superpointの可視化用設定クラス"""
    
    # 必須設定（デフォルト値なし）
    model_path: str  # 学習済みモデルパス
    classifier_path: str  # 学習済み分類器パス
    
    # 基本設定（デフォルト値あり）
    data_path: str = '../data/users/minesawa/semantickitti/growsp'  # 点群データパス
    sp_path: str = '../data/users/minesawa/semantickitti/growsp_sp'  # 初期スーパーポイントパス
    output_path: str = '../data/users/minesawa/semantickitti/vis_sp'  # 出力パス
    
    # データ処理設定
    voxel_size: float = 0.15  # SparseConvのボクセルサイズ
    input_dim: int = 3  # ネットワーク入力次元
    feats_dim: int = 128  # 出力特徴次元
    r_crop: float = 50.0  # クロッピング半径
    ignore_label: int = -1  # 無効ラベル
    
    # クラスタリング設定
    current_growsp: Optional[int] = None  # 現在のスーパーポイント数（Noneの場合は初期SP使用）
    
    # 出力設定
    sequences: List[str] = field(default_factory=lambda: ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10'])  # 対象シーケンス
    max_scenes: Optional[int] = None  # 処理する最大シーン数（Noneで全て）
    file_extension: str = 'ply'  # 出力ファイル拡張子
    
    # デバッグ設定
    debug: bool = False
    
    @classmethod
    def from_parse_args(cls) -> 'VisualizationConfig':
        """コマンドライン引数から設定を作成"""
        parser = argparse.ArgumentParser(description='Superpoint Visualization Tool')
        
        # 基本設定
        parser.add_argument('--data_path', type=str, default='../data/users/minesawa/semantickitti/growsp',
                            help='点群データパス')
        parser.add_argument('--sp_path', type=str, default='../data/users/minesawa/semantickitti/growsp_sp',
                            help='初期スーパーポイントパス')
        parser.add_argument('--model_path', type=str, required=True,
                            help='学習済みモデルパス')
        parser.add_argument('--classifier_path', type=str, required=True,
                            help='学習済み分類器パス')
        parser.add_argument('--output_path', type=str, default='../data/users/minesawa/semantickitti/vis_sp',
                            help='出力パス')
        
        # データ処理設定
        parser.add_argument('--voxel_size', type=float, default=0.15,
                            help='SparseConvのボクセルサイズ')
        parser.add_argument('--input_dim', type=int, default=3,
                            help='ネットワーク入力次元')
        parser.add_argument('--feats_dim', type=int, default=128,
                            help='出力特徴次元')
        parser.add_argument('--r_crop', type=float, default=50.0,
                            help='クロッピング半径')
        
        # クラスタリング設定
        parser.add_argument('--current_growsp', type=int, default=None,
                            help='現在のスーパーポイント数（Noneの場合は初期SP使用）')
        
        # 出力設定
        parser.add_argument('--sequences', type=str, nargs='+',
                            default=['00', '01', '02', '03', '04', '05', '06', '07', '09', '10'],
                            help='対象シーケンス')
        parser.add_argument('--max_scenes', type=int, default=None,
                            help='処理する最大シーン数')
        parser.add_argument('--file_extension', type=str, default='ply',
                            help='出力ファイル拡張子')
        
        # デバッグ設定
        parser.add_argument('--debug', action='store_true',
                            help='デバッグモード')
        
        args = parser.parse_args()
        
        # データクラスのインスタンスを作成
        config = cls(
            model_path=args.model_path,
            classifier_path=args.classifier_path,
            data_path=args.data_path,
            sp_path=args.sp_path,
            output_path=args.output_path,
            voxel_size=args.voxel_size,
            input_dim=args.input_dim,
            feats_dim=args.feats_dim,
            r_crop=args.r_crop,
            current_growsp=args.current_growsp,
            sequences=args.sequences,
            max_scenes=args.max_scenes,
            file_extension=args.file_extension,
            debug=args.debug
        )
        
        return config 