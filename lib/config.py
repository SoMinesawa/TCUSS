import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import os
import yaml


@dataclass
class STCCorrespondenceConfig:
    """対応点計算設定"""
    distance_threshold: float = 0.3
    min_points: int = 5
    exclude_ground: bool = True  # 地面点を対応計算から除外するかどうか
    remove_ego_motion: bool = True  # エゴモーションを除去してobject_flowを使用するかどうか


@dataclass
class STCLossConfig:
    """STC損失設定"""
    temperature: float = 0.1


@dataclass
class EvaluationConfig:
    """評価設定"""
    distance_evaluation: bool = True  # 距離別評価を有効にするかどうか
    distance_bins: List[int] = field(default_factory=lambda: [10, 20, 30, 40, 50, 60, 70, 80])  # 距離区切り（m）
    moving_static_evaluation: bool = True  # 移動物体/静止物体の分離評価を有効にするかどうか


@dataclass
class STCConfig:
    """STC (Superpoint Time Consistency) 設定"""
    enabled: bool = False  # デフォルトは無効
    weight: float = 1.0
    voteflow_preprocess_path: str = "data/dataset/semantickitti/voteflow_preprocess_fixed"  # VoteFlow前処理済みH5データパス
    correspondence: STCCorrespondenceConfig = field(default_factory=STCCorrespondenceConfig)
    loss: STCLossConfig = field(default_factory=STCLossConfig)


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
    weight_decay: float = 1e-2  # 重み減衰
    accum_step: int = 1  # 勾配蓄積ステップ
    
    # データロード設定
    workers: int = 24  # データローディング用ワーカー数
    cluster_workers: int = 24  # クラスタリング用ワーカー数
    batch_size: List[int] = field(default_factory=lambda: [16, 16])  # バッチサイズ
    eval_batch_size: int = 32  # 評価時のバッチサイズ（grad不要なので大きくできる）
    cluster_batch_size: int = 16  # クラスタリング時のバッチサイズ（grad不要なので大きくできる）
    
    # DataLoader最適化
    persistent_workers: bool = True  # ワーカープロセスを維持
    prefetch_factor: int = 4  # プリフェッチ数
    
    # マルチGPU設定
    use_ddp: bool = True  # DistributedDataParallelを使用
    ddp_backend: str = "nccl"  # DDPバックエンド
    
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
    scan_window: int = 12  # t1からt2を選ぶ際の最大フレーム差
    
    # 評価設定
    cluster_interval: int = 10  # クラスタリング間隔
    eval_interval: int = 10  # 評価間隔
    silhouette: bool = False  # kmeansの追加評価指標
    
    # 早期停止設定
    early_stopping: bool = True  # 早期停止を有効にするかどうか
    early_stopping_patience: int = 3  # val停滞許容回数 (PATIENCE_VAL)
    early_stopping_min_delta: float = 0.15  # val改善とみなす最小増分 (MIN_DELTA_VAL)
    early_stopping_metric: str = 'val_mIoU'  # 早期停止の評価指標
    early_stopping_mode: str = 'max'  # 早期停止のモード
    rel_drop_window: int = 15  # train_loss収束判定用窓幅
    overfit_drop: float = 0.30  # val過学習判定差分
    
    # デバッグ設定
    debug: bool = False  # デバッグモード
    
    # 視覚化設定
    vis: bool = False  # 視覚化フラグ
    vis_sp: bool = False  # superpoint対応可視化モード（学習開始直後に対応点群をplyで保存して終了）
    
    # 学習再開設定
    resume: bool = False  # 学習再開フラグ
    wandb_run_id: Optional[str] = None  # wandbのrun ID
    
    # STC設定
    stc: STCConfig = field(default_factory=STCConfig)
    
    # 評価設定（詳細）
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'TCUSSConfig':
        """非推奨: --configのみ利用を想定"""
        raise RuntimeError("from_args is disabled. Use --config YAML only.")
    
    @classmethod
    def from_parse_args(cls) -> 'TCUSSConfig':
        """コマンドラインから直接パースしてConfigオブジェクトを作成（--configのみ受け付け）"""
        parser = argparse.ArgumentParser(description='TCUSS: Temporal Consistent Unsupervised Semantic Segmentation')
        
        # YAML設定ファイルのみ受け付ける
        parser.add_argument('--config', type=str, required=True, help='YAML設定ファイルパス')
        
        args = parser.parse_args()
        
        return cls.from_yaml(args.config)
    
    @classmethod
    def from_yaml(cls, yaml_path: str, cli_args: Optional[argparse.Namespace] = None) -> 'TCUSSConfig':
        """YAMLファイルから設定を読み込み（CLI上書きなし）。必須項目が欠けていれば停止。"""
        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # STC設定の処理（必須）
        if 'stc' not in yaml_config:
            raise ValueError(f'stc セクションが {yaml_path} に存在しません')
        stc_dict = yaml_config.pop('stc')
        stc_config = STCConfig(
            enabled=stc_dict.get('enabled', False),
            weight=stc_dict.get('weight', 1.0),
            voteflow_preprocess_path=stc_dict.get('voteflow_preprocess_path', 'data/dataset/semantickitti/voteflow_preprocess_fixed'),
            correspondence=STCCorrespondenceConfig(**stc_dict.get('correspondence', {})),
            loss=STCLossConfig(**stc_dict.get('loss', {}))
        )
        
        # evaluation設定の処理（必須）
        if 'evaluation' not in yaml_config:
            raise ValueError(f'evaluation セクションが {yaml_path} に存在しません')
        eval_dict = yaml_config.pop('evaluation')
        eval_config = EvaluationConfig(
            distance_evaluation=eval_dict.get('distance_evaluation', True),
            distance_bins=eval_dict.get('distance_bins', [10, 20, 30, 40, 50, 60, 70, 80]),
            moving_static_evaluation=eval_dict.get('moving_static_evaluation', True)
        )
        
        # early_stopping設定の処理（YAMLではネストされている場合）
        if 'early_stopping' in yaml_config and isinstance(yaml_config['early_stopping'], dict):
            es_dict = yaml_config.pop('early_stopping')
            yaml_config['early_stopping'] = es_dict.get('enabled', True)
            yaml_config['early_stopping_patience'] = es_dict.get('patience', 3)
            yaml_config['early_stopping_min_delta'] = es_dict.get('min_delta', 0.15)
            yaml_config['early_stopping_metric'] = es_dict.get('metric', 'val_mIoU')
            yaml_config['early_stopping_mode'] = es_dict.get('mode', 'max')
            yaml_config['rel_drop_window'] = es_dict.get('rel_drop_window', 15)
            yaml_config['overfit_drop'] = es_dict.get('overfit_drop', 0.30)
        
        # growsp設定の処理（YAMLではネストされている場合）
        if 'growsp' in yaml_config and isinstance(yaml_config['growsp'], dict):
            gs_dict = yaml_config.pop('growsp')
            yaml_config['growsp_start'] = gs_dict.get('start', 80)
            yaml_config['growsp_end'] = gs_dict.get('end', 30)
            yaml_config['drop_threshold'] = gs_dict.get('drop_threshold', 10)
            yaml_config['w_rgb'] = gs_dict.get('w_rgb', 1.0)
            yaml_config['c_rgb'] = gs_dict.get('c_rgb', 5.0)
            yaml_config['c_shape'] = gs_dict.get('c_shape', 5.0)
        
        # STCとevaluationをyaml_configに追加
        yaml_config['stc'] = stc_config
        yaml_config['evaluation'] = eval_config
        
        # 必須キー確認（dataclassフィールドに対応）。stc, evaluation は別途セット済みなので除外。
        required_keys = [f.name for f in cls.__dataclass_fields__.values() if f.name not in ('stc', 'evaluation')]
        missing = [k for k in required_keys if k not in yaml_config]
        if missing:
            raise ValueError(f'{yaml_path} に必須キーが不足しています: {missing}')
        
        return cls(**yaml_config)
