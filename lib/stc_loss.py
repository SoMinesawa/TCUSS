"""
STC (Superpoint Time Consistency) Loss

Scene flowによる点の対応を使って、時間的に対応するSuperpointの特徴量を近づける損失関数。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def compute_sp_features(
    point_features: torch.Tensor,
    sp_labels: torch.Tensor,
    num_sp: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    点の特徴量からSuperpointの特徴量を計算（平均）
    
    Args:
        point_features: 点の特徴量 [N, D]
        sp_labels: 各点のSuperpointラベル [N]、-1は無効
        num_sp: Superpointの数（Noneの場合は自動検出）
        
    Returns:
        sp_features: Superpoint特徴量 [M, D]
        valid_mask: 有効なSPのマスク [M]（点が存在するSPはTrue）
    """
    device = point_features.device
    D = point_features.shape[1]
    
    # 有効な点のみを使用
    valid_mask = sp_labels >= 0
    valid_features = point_features[valid_mask]
    valid_labels = sp_labels[valid_mask]
    
    if len(valid_labels) == 0:
        if num_sp is None:
            num_sp = 1
        return torch.zeros(num_sp, D, device=device), torch.zeros(num_sp, dtype=torch.bool, device=device)
    
    # ラベルを連番（0..M-1）に圧縮して安全にindex_addできるようにする
    unique_labels, inverse = torch.unique(valid_labels, sorted=True, return_inverse=True)
    
    # Superpointの数を決定（unique数と合わせる）
    num_sp = len(unique_labels)
    
    # 各SPの特徴量の合計と点数を計算
    sp_features_sum = torch.zeros(num_sp, D, device=device)
    sp_counts = torch.zeros(num_sp, device=device)
    
    sp_features_sum.index_add_(0, inverse, valid_features)
    sp_counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=torch.float))
    
    # 平均を計算（点がないSPはゼロベクトル）
    sp_valid_mask = sp_counts > 0
    sp_features = torch.where(
        sp_counts.unsqueeze(-1) > 0,
        sp_features_sum / sp_counts.unsqueeze(-1),
        torch.zeros_like(sp_features_sum)
    )
    
    return sp_features, sp_valid_mask


def loss_stc_similarity(
    sp_features_t: torch.Tensor,
    sp_features_t1: torch.Tensor,
    correspondence_matrix: torch.Tensor,
    valid_mask_t: Optional[torch.Tensor] = None,
    valid_mask_t1: Optional[torch.Tensor] = None,
    min_correspondence: int = 5
) -> torch.Tensor:
    """
    対応するSuperpointの特徴量を近づける損失（類似度最大化）
    
    対応がないSuperpointは無視する（遠ざけない）。
    
    Args:
        sp_features_t: 時刻tのSuperpoint特徴量 [M, D]
        sp_features_t1: 時刻t+1のSuperpoint特徴量 [N, D]
        correspondence_matrix: 対応点数の行列 [M, N]
        valid_mask_t: 時刻tの有効SPマスク [M]（Noneの場合は全て有効）
        valid_mask_t1: 時刻t+1の有効SPマスク [N]（Noneの場合は全て有効）
        min_correspondence: 有効な対応とみなす最小点数
        
    Returns:
        loss: スカラーの損失値
    """
    M, D = sp_features_t.shape
    N = sp_features_t1.shape[0]
    device = sp_features_t.device
    
    # マスクの処理
    if valid_mask_t is None:
        valid_mask_t = torch.ones(M, dtype=torch.bool, device=device)
    if valid_mask_t1 is None:
        valid_mask_t1 = torch.ones(N, dtype=torch.bool, device=device)
    
    # 対応がないSP、または無効なSPを除外
    # 行方向で対応点数の合計が min_correspondence 以上のSPのみを使用
    row_sum = correspondence_matrix.sum(dim=1)  # [M]
    valid_rows = (row_sum >= min_correspondence) & valid_mask_t  # [M]
    
    if not valid_rows.any():
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # 有効な行のみを抽出
    valid_sp_features_t = sp_features_t[valid_rows]  # [M', D]
    valid_corr = correspondence_matrix[valid_rows]  # [M', N]
    
    # 無効な列（t+1側のSP）をゼロにする
    valid_corr = valid_corr * valid_mask_t1.float().unsqueeze(0)  # [M', N]
    
    # 対応点数で行正規化（重み行列）
    row_sum_valid = valid_corr.sum(dim=1, keepdim=True)  # [M', 1]
    weights = valid_corr / (row_sum_valid + 1e-8)  # [M', N]
    
    # 特徴量を正規化
    sp_features_t_norm = F.normalize(valid_sp_features_t, dim=-1)  # [M', D]
    sp_features_t1_norm = F.normalize(sp_features_t1, dim=-1)  # [N, D]
    
    # コサイン類似度を計算
    similarity = torch.mm(sp_features_t_norm, sp_features_t1_norm.t())  # [M', N]
    
    # 重み付き類似度の平均を最大化（負の損失）
    # weights[i, j] > 0 のペアのみが損失に寄与
    weighted_similarity = (weights * similarity).sum(dim=1)  # [M']
    
    # 負の平均類似度を損失とする（類似度を最大化したいので）
    loss = -weighted_similarity.mean()
    
    return loss


def loss_stc_mse(
    sp_features_t: torch.Tensor,
    sp_features_t1: torch.Tensor,
    correspondence_matrix: torch.Tensor,
    valid_mask_t: Optional[torch.Tensor] = None,
    valid_mask_t1: Optional[torch.Tensor] = None,
    min_correspondence: int = 5
) -> torch.Tensor:
    """
    対応するSuperpointの特徴量を近づける損失（MSE版）
    
    各SP_tを、対応するSP_{t+1}群の加重平均に近づける。
    
    Args:
        sp_features_t: 時刻tのSuperpoint特徴量 [M, D]
        sp_features_t1: 時刻t+1のSuperpoint特徴量 [N, D]
        correspondence_matrix: 対応点数の行列 [M, N]
        valid_mask_t: 時刻tの有効SPマスク [M]
        valid_mask_t1: 時刻t+1の有効SPマスク [N]
        min_correspondence: 有効な対応とみなす最小点数
        
    Returns:
        loss: スカラーの損失値
    """
    M, D = sp_features_t.shape
    N = sp_features_t1.shape[0]
    device = sp_features_t.device
    
    # マスクの処理
    if valid_mask_t is None:
        valid_mask_t = torch.ones(M, dtype=torch.bool, device=device)
    if valid_mask_t1 is None:
        valid_mask_t1 = torch.ones(N, dtype=torch.bool, device=device)
    
    # 対応点数の合計が min_correspondence 以上のSPのみを使用
    row_sum = correspondence_matrix.sum(dim=1)
    valid_rows = (row_sum >= min_correspondence) & valid_mask_t
    
    if not valid_rows.any():
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # 有効な行のみを抽出
    valid_sp_features_t = sp_features_t[valid_rows]  # [M', D]
    valid_corr = correspondence_matrix[valid_rows]  # [M', N]
    
    # 無効な列をゼロにする
    valid_corr = valid_corr * valid_mask_t1.float().unsqueeze(0)
    
    # 行正規化
    row_sum_valid = valid_corr.sum(dim=1, keepdim=True)
    weights = valid_corr / (row_sum_valid + 1e-8)  # [M', N]
    
    # 対応SPの加重平均を計算
    target_features = torch.mm(weights, sp_features_t1)  # [M', D]
    
    # MSE損失
    loss = F.mse_loss(valid_sp_features_t, target_features)
    
    return loss


class STCLoss(nn.Module):
    """
    STC損失モジュール
    """
    
    def __init__(
        self,
        loss_type: str = "similarity",
        min_correspondence: int = 5
    ):
        """
        Args:
            loss_type: "similarity" または "mse"
            min_correspondence: 有効な対応とみなす最小点数
        """
        super().__init__()
        self.loss_type = loss_type
        self.min_correspondence = min_correspondence
        
        if loss_type == "similarity":
            self.loss_fn = loss_stc_similarity
        elif loss_type == "mse":
            self.loss_fn = loss_stc_mse
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self,
        sp_features_t: torch.Tensor,
        sp_features_t1: torch.Tensor,
        correspondence_matrix: torch.Tensor,
        valid_mask_t: Optional[torch.Tensor] = None,
        valid_mask_t1: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.loss_fn(
            sp_features_t,
            sp_features_t1,
            correspondence_matrix,
            valid_mask_t,
            valid_mask_t1,
            self.min_correspondence
        )


if __name__ == "__main__":
    # テスト
    torch.manual_seed(42)
    
    # ダミーデータ
    M, N, D = 10, 12, 128
    sp_features_t = torch.randn(M, D)
    sp_features_t1 = torch.randn(N, D)
    
    # 対応行列（ランダム）
    correspondence_matrix = torch.randint(0, 20, (M, N)).float()
    
    # 損失計算
    loss_sim = loss_stc_similarity(sp_features_t, sp_features_t1, correspondence_matrix)
    loss_mse = loss_stc_mse(sp_features_t, sp_features_t1, correspondence_matrix)
    
    print(f"Similarity Loss: {loss_sim.item():.4f}")
    print(f"MSE Loss: {loss_mse.item():.4f}")
    
    # 対応がある場合は類似度が高いテスト
    sp_features_t1_similar = sp_features_t[:N] + torch.randn(N, D) * 0.1
    correspondence_matrix_diagonal = torch.eye(M, N) * 100
    
    loss_sim_similar = loss_stc_similarity(sp_features_t, sp_features_t1_similar, correspondence_matrix_diagonal)
    print(f"Similarity Loss (similar features): {loss_sim_similar.item():.4f}")







