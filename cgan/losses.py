#!/usr/bin/env python3
"""
Loss functions & utilities for CGAN pseudo‑label calibration (SN + R1 + Hinge)
--------------------------------------------------------------------------
* **EIoU regression loss** – geometry‑aware box alignment.
* **Hinge adversarial loss** – PatchGAN friendly.
* **R1 regularisation**     – cheaper & simpler than WGAN‑GP.
* **apply_delta_to_bbox**   – perfectly mirrors `_bbox2delta()` (√area norm).

The project has fully migrated to *R1 + Hinge*; no WGAN‑GP code remains.
"""
from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "EIoULoss",
    "hinge_loss_discriminator",
    "hinge_loss_generator",
    "compute_r1_penalty",
    "HybridLoss",
    "smooth_clamp",
    "apply_delta_to_bbox",
    "iou_metric",
]



_eps: float = 1e-6  # global numerical epsilon

# ---------------------------------------------------------------------------
#  Adversarial losses
# ---------------------------------------------------------------------------

def hinge_loss_discriminator(real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discriminator: *max(0, 1 – D_real) + max(0, 1 + D_fake)*."""
    return F.relu(1.0 - real_pred).mean() + F.relu(1.0 + fake_pred).mean()


def hinge_loss_generator(fake_pred: torch.Tensor) -> torch.Tensor:
    """Hinge loss for generator: *‑E[D_fake]*."""
    return -fake_pred.mean()


def compute_r1_penalty(
    discriminator: nn.Module,
    pred_patch: torch.Tensor,   # 條件輸入（pred_patch）
    real_patch: torch.Tensor,   # 真實輸入（gt_patch）
) -> torch.Tensor:
    """計算R1正則化，針對 ProjectionPatchGAN 的參數順序修正"""
    try:
        # 創建新的輸入張量，確保梯度計算獨立
        # 重要：只對 real_patch 求梯度，pred_patch 作為條件保持 detached
        pred_input = pred_patch.detach()  # 條件不參與梯度計算
        real_input = real_patch.detach().clone().requires_grad_(True)  # 只對這個求梯度

        # ProjectionPatchGAN 的調用順序：forward(patch_to_judge, condition_patch)
        # 這裡我們計算對 real_patch (gt) 的梯度，real_input是要被判別的，pred_input是條件
        real_scores = discriminator(real_input, pred_input).sum()
        grads = torch.autograd.grad(
            outputs=real_scores,
            inputs=real_input,
            create_graph=True,       # 要求二階梯度
            retain_graph=True,       # 保持圖以允許後續backward
            only_inputs=True,
        )[0]

        grad_norm2 = grads.flatten(1).pow(2).sum(1)   # (B,)
        r1_penalty = grad_norm2.mean() / 2.0
        
        # 添加數值穩定性檢查
        if torch.isnan(r1_penalty) or torch.isinf(r1_penalty):
            print(f"Warning: Invalid R1 penalty value: {r1_penalty.item()}")
            return torch.tensor(0.0, device=pred_patch.device, requires_grad=True)
            
        return r1_penalty
    except RuntimeError as e:
        print(f"Warning: R1 penalty computation failed: {e}")
        return torch.tensor(0.0, device=pred_patch.device, requires_grad=True)

# ---------------------------------------------------------------------------
#  EIoU regression loss
# ---------------------------------------------------------------------------

class EIoULoss(nn.Module):
    """`1 – EIoU` bounding‑box regression loss."""

    def __init__(self, eps: float = _eps):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # noqa: D401
        px1, py1 = pred[:, 0] - pred[:, 2] / 2, pred[:, 1] - pred[:, 3] / 2
        px2, py2 = pred[:, 0] + pred[:, 2] / 2, pred[:, 1] + pred[:, 3] / 2
        tx1, ty1 = target[:, 0] - target[:, 2] / 2, target[:, 1] - target[:, 3] / 2
        tx2, ty2 = target[:, 0] + target[:, 2] / 2, target[:, 1] + target[:, 3] / 2

        ix1, iy1 = torch.max(px1, tx1), torch.max(py1, ty1)
        ix2, iy2 = torch.min(px2, tx2), torch.min(py2, ty2)
        inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)

        p_area = (px2 - px1) * (py2 - py1) + self.eps
        t_area = (tx2 - tx1) * (ty2 - ty1) + self.eps
        union = p_area + t_area - inter
        iou = inter / union

        ex1, ey1 = torch.min(px1, tx1), torch.min(py1, ty1)
        ex2, ey2 = torch.max(px2, tx2), torch.max(py2, ty2)
        cw, ch = ex2 - ex1, ey2 - ey1
        c2 = cw.pow(2) + ch.pow(2) + self.eps

        centre_d2 = (pred[:, 0] - target[:, 0]).pow(2) + (pred[:, 1] - target[:, 1]).pow(2)
        w_d2 = (pred[:, 2] - target[:, 2]).pow(2)
        h_d2 = (pred[:, 3] - target[:, 3]).pow(2)

        eiou = iou - centre_d2 / c2 - w_d2 / (cw.pow(2) + self.eps) - h_d2 / (ch.pow(2) + self.eps)
        return 1.0 - eiou.mean()

# ---------------------------------------------------------------------------
#  Hybrid loss (EIoU + adversarial)
# ---------------------------------------------------------------------------

class HybridLoss(nn.Module):
    """`L = λ_iou·EIoU + λ_adv·adv` combines geometry & adversarial signals."""

    def __init__(self, lambda_iou: float = 1.0, lambda_adv: float = 0.1):
        super().__init__()
        self.lambda_iou = float(lambda_iou)
        self.lambda_adv = float(lambda_adv)
        self.iou_loss = EIoULoss()
        print(f"HybridLoss -> lambda_iou={lambda_iou}, lambda_adv={lambda_adv}")

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        adv_loss: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        iou_l = self.iou_loss(pred_boxes, target_boxes)
        total = self.lambda_iou * iou_l
        if adv_loss is not None:
            total = total + self.lambda_adv * adv_loss
        return total, iou_l, adv_loss

# ---------------------------------------------------------------------------
#  Delta decoding util
# ---------------------------------------------------------------------------

def smooth_clamp(x: torch.Tensor, min_val: float, max_val: float, temperature: float = 0.5) -> torch.Tensor:
    """Sigmoid‑based clamp that preserves gradients."""
    centre = (min_val + max_val) / 2.0
    scaled = (x - centre) / temperature
    return min_val + (max_val - min_val) * torch.sigmoid(scaled)


# 全域配置緩存，避免重複讀取文件
_GLOBAL_CONFIG = None

def apply_delta_to_bbox(
    bbox: torch.Tensor,   # (N,4)
    delta: torch.Tensor,  # (N,4)
    training: bool = True,
    config: dict | None = None,
) -> torch.Tensor:
    """Decode Δ back to box (mirrors `_bbox2delta`) with configurable parameters."""
    # 載入配置參數 - 使用全域緩存避免重複文件I/O
    global _GLOBAL_CONFIG
    if config is None:
        if _GLOBAL_CONFIG is None:
            import yaml
            from pathlib import Path
            config_path = Path(__file__).parent / "config.yaml"
            with open(config_path, 'r', encoding='utf-8') as f:
                _GLOBAL_CONFIG = yaml.safe_load(f)
        config = _GLOBAL_CONFIG
    
    clamp_range = config.get('delta_clamp_range', 1.5)
    bbox_bounds = config.get('bbox_bounds', {
        'cx_min': 0.05, 'cx_max': 0.95,
        'cy_min': 0.05, 'cy_max': 0.95,
        'w_min': 0.02, 'w_max': 0.8,
        'h_min': 0.02, 'h_max': 0.8
    })
    
    delta_c = smooth_clamp(delta, -clamp_range, clamp_range) if training else delta.clamp(-clamp_range, clamp_range)

    norm = torch.clamp(torch.sqrt(bbox[:, 2] * bbox[:, 3]), min=0.05)  # 修復漏洞2：移除_eps，與數據集編碼保持完全一致
    cx = bbox[:, 0] + delta_c[:, 0] * norm
    cy = bbox[:, 1] + delta_c[:, 1] * norm
    w = bbox[:, 2] * torch.exp(delta_c[:, 2].clamp(-1.0, 1.0))
    h = bbox[:, 3] * torch.exp(delta_c[:, 3].clamp(-1.0, 1.0))

    if training:
        cx = smooth_clamp(cx, bbox_bounds['cx_min'], bbox_bounds['cx_max'])
        cy = smooth_clamp(cy, bbox_bounds['cy_min'], bbox_bounds['cy_max'])
        w = smooth_clamp(w, bbox_bounds['w_min'], bbox_bounds['w_max'])
        h = smooth_clamp(h, bbox_bounds['h_min'], bbox_bounds['h_max'])
    else:
        cx = cx.clamp(bbox_bounds['cx_min'], bbox_bounds['cx_max'])
        cy = cy.clamp(bbox_bounds['cy_min'], bbox_bounds['cy_max'])
        w = w.clamp(bbox_bounds['w_min'], bbox_bounds['w_max'])
        h = h.clamp(bbox_bounds['h_min'], bbox_bounds['h_max'])

    return torch.stack((cx, cy, w, h), dim=-1)

# ---------------------------------------------------------------------------
#  IoU metric
# ---------------------------------------------------------------------------

def iou_metric(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, eps: float = _eps) -> torch.Tensor:
    """Vectorised IoU calculation for evaluation (one scalar per pair)."""
    px1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    py1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    px2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    py2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    tx1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    ty1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    tx2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    ty2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

    ix1 = torch.max(px1, tx1)
    iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2)
    iy2 = torch.min(py2, ty2)

    inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
    union = (px2 - px1) * (py2 - py1) + (tx2 - tx1) * (ty2 - ty1) - inter + eps
    return inter / union
