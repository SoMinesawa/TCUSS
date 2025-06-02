import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import os


@dataclass
class TCUSSConfig:
    """TCUSSの設定を管理するクラス"""
    
    # 基本設定
    name: str  # 実験の名前
    data_path: str = 'data/users/minesawa/semantickitti/growsp'  # 点群データパス
    sp_path: str = 'data/users/minesawa/semantickitti/growsp_sp'  # 初期スーパーポイントパス
    original_data_path: str = 'data/dataset/semantickitti/dataset/sequences'  # オリジナルデータパス
    patchwork_path: str = 'data/users/minesawa/semantickitti/patchwork'  # パッチワークデータパス
    save_path: str = 'data/users/minesawa/semantickitti/growsp_model'  # モデル保存パス
    pseudo_label_path: str = 'pseudo_label_kitti/'  # 疑似ラベル保存パス
    
    # エポック設定
    max_epoch: List[int] = field(default_factory=lambda: [100, 350])  # 各ステージの最大エポック数
    max_iter: List[int] = field(default_factory=lambda: [10000, 30000])  # 各ステージの最大イテレーション数
    
    # モデル設定
    bn_momentum: float = 0.02  # バッチ正規化のモメンタム
    conv1_kernel_size: int = 5  # 第1畳み込み層のカーネルサイズ
    
    # 最適化設定
    lr: float = 1e-2  # バックボーンネットワークの学習率
    tarl_lr: float = 0.0002  # Transformerプロジェクタとプレディクタの学習率
    weight_decay: float = 1e-2  # 重み減衰
    accum_step: int = 1  # 勾配蓄積ステップ
    
    # データロード設定
    workers: int = 16  # データローディング用ワーカー数
    cluster_workers: int = 16  # クラスタリング用ワーカー数
    batch_size: List[int] = field(default_factory=lambda: [16, 16])  # バッチサイズ [GrowSP, TARL]
    
    # 実験設定
    seed: int = 2022  # 乱数シード
    log_interval: int = 1000000  # ログ間隔
    
    # モデル構成設定
    voxel_size: float = 0.15  # SparseConvのボクセルサイズ
    input_dim: int = 3  # ネットワーク入力次元
    primitive_num: int = 500  # 学習に使用するプリミティブ数
    semantic_class: int = 19  # セマンティッククラス数
    feats_dim: int = 128  # 出力特徴次元
    ignore_label: int = -1  # 無効ラベル
    
    # GrowSP設定
    growsp_start: int = 80  # スーパーポイント成長の開始数
    growsp_end: int = 30  # スーパーポイント成長の終了数
    drop_threshold: int = 10  # 少数のポイントを持つスーパーポイントを無視する閾値
    w_rgb: float = 1.0  # スーパーポイントのマージにおけるRGBの重み
    c_rgb: float = 5.0  # プリミティブのクラスタリングにおけるRGBの重み
    c_shape: float = 5.0  # プリミティブのクラスタリングにおけるPFHの重み
    
    # データセット設定
    select_num: int = 1500  # 各ラウンドで選択されるシーン数
    eval_select_num: int = 4071  # 評価時に選択されるシーン数
    r_crop: float = 50.0  # トレーニング時のクロッピング半径
    
    # 評価設定
    cluster_interval: int = 10  # クラスタリング間隔
    eval_interval: int = 10  # 評価間隔
    silhouette: bool = False  # kmeansの追加評価指標
    
    # デバッグ設定
    debug: bool = False  # デバッグモード
    
    # 時間設定
    scan_window: int = 12  # スキャンウィンドウサイズ
    
    # 視覚化設定
    vis: bool = False  # 視覚化フラグ
    
    # 学習再開設定
    resume: bool = False  # 学習再開フラグ
    wandb_run_id: Optional[str] = None  # wandbのrun ID
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'TCUSSConfig':
        """コマンドライン引数からConfigオブジェクトを作成"""
        # args内の属性をすべて取得して辞書に変換
        args_dict = vars(args)
        
        # インスタンス作成
        return cls(**args_dict)
    
    @classmethod
    def from_parse_args(cls) -> 'TCUSSConfig':
        """コマンドラインから直接パースしてConfigオブジェクトを作成"""
        parser = argparse.ArgumentParser(description='TCUSS: Temporal Consistent Unsupervised Semantic Segmentation')
        # 基本設定
        parser.add_argument('--name', type=str, required=True, help='実験の名前')
        parser.add_argument('--data_path', type=str, default='data/users/minesawa/semantickitti/growsp',
                            help='点群データパス')
        parser.add_argument('--sp_path', type=str, default='data/users/minesawa/semantickitti/growsp_sp',
                            help='初期スーパーポイントパス')
        parser.add_argument('--original_data_path', type=str, default='data/dataset/semantickitti/dataset/sequences')
        parser.add_argument('--patchwork_path', type=str, default='data/users/minesawa/semantickitti/patchwork')
        parser.add_argument('--save_path', type=str, default='data/users/minesawa/semantickitti/growsp_model', 
                            help='モデル保存パス')
        parser.add_argument('--pseudo_label_path', default='pseudo_label_kitti/', type=str, 
                            help='疑似ラベル保存パス')
        
        # エポック設定
        parser.add_argument('--max_epoch', type=int, nargs='+', default=[100, 350], 
                            help='各ステージの最大エポック数 [非成長ステージ, 成長ステージ]')
        parser.add_argument('--max_iter', type=list, default=[10000, 30000], 
                            help='各ステージの最大イテレーション数')
        
        # モデル設定
        parser.add_argument('--bn_momentum', type=float, default=0.02, help='バッチ正規化のモメンタム')
        parser.add_argument('--conv1_kernel_size', type=int, default=5, help='第1畳み込み層のカーネルサイズ')
        
        # 最適化設定
        parser.add_argument('--lr', type=float, default=1e-2, help='バックボーンネットワークの学習率')
        parser.add_argument('--tarl_lr', type=float, default=0.0002, help='Transformerプロジェクタとプレディクタの学習率')
        parser.add_argument('--weight_decay', type=float, default=1e-2, help='重み減衰')
        parser.add_argument('--accum_step', type=int, default=1, help='勾配蓄積ステップ')
        
        # データロード設定
        parser.add_argument('--workers', type=int, default=16, help='データローディング用ワーカー数')
        parser.add_argument('--cluster_workers', type=int, default=16, help='クラスタリング用ワーカー数')
        parser.add_argument('--batch_size', type=int, nargs='+', default=[16, 16], 
                            help='バッチサイズ [GrowSP, TARL]')
        
        # 実験設定
        parser.add_argument('--seed', type=int, default=2022, help='乱数シード')
        parser.add_argument('--log-interval', type=int, default=1000000, help='ログ間隔')
        
        # モデル構成設定
        parser.add_argument('--voxel_size', type=float, default=0.15, help='SparseConvのボクセルサイズ')
        parser.add_argument('--input_dim', type=int, default=3, help='ネットワーク入力次元')
        parser.add_argument('--primitive_num', type=int, default=500, help='学習に使用するプリミティブ数')
        parser.add_argument('--semantic_class', type=int, default=19, help='セマンティッククラス数')
        parser.add_argument('--feats_dim', type=int, default=128, help='出力特徴次元')
        parser.add_argument('--ignore_label', type=int, default=-1, help='無効ラベル')
        
        # GrowSP設定
        parser.add_argument('--growsp_start', type=int, default=80, help='スーパーポイント成長の開始数')
        parser.add_argument('--growsp_end', type=int, default=30, help='スーパーポイント成長の終了数')
        parser.add_argument('--drop_threshold', type=int, default=10, 
                            help='少数のポイントを持つスーパーポイントを無視する閾値')
        parser.add_argument('--w_rgb', type=float, default=5/5, help='スーパーポイントのマージにおけるRGBの重み')
        parser.add_argument('--c_rgb', type=float, default=5, help='プリミティブのクラスタリングにおけるRGBの重み')
        parser.add_argument('--c_shape', type=float, default=5, help='プリミティブのクラスタリングにおけるPFHの重み')
        
        # データセット設定
        parser.add_argument('--select_num', type=int, default=1500, help='各ラウンドで選択されるシーン数')
        parser.add_argument('--eval_select_num', type=int, default=4071, help='評価時に選択されるシーン数')
        parser.add_argument('--r_crop', type=float, default=50, help='トレーニング時のクロッピング半径')
        
        # 評価設定
        parser.add_argument('--cluster_interval', type=int, default=10, help='クラスタリング間隔')
        parser.add_argument('--eval_interval', type=int, default=10, help='評価間隔')
        parser.add_argument('--silhouette', action='store_true', help='kmeansの追加評価指標')
        
        # デバッグ設定
        parser.add_argument('--debug', action='store_true', help='デバッグモード')
        
        # 時間設定
        parser.add_argument('--scan_window', type=int, default=12, help='スキャンウィンドウサイズ')
        
        # 視覚化設定
        parser.add_argument('--vis', action='store_true', help='視覚化フラグ')
        
        # 学習再開設定
        parser.add_argument('--resume', action='store_true', help='学習再開フラグ')
        parser.add_argument('--wandb_run_id', type=str, help='wandbのrun ID')
        
        args = parser.parse_args()
        return cls.from_args(args) 