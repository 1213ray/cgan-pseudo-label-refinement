"""models_swin.py — Swin‑UNet + Projection PatchGAN（完整中文註解版）
================================================================================
本模組定義了用於 **偽標籤校正 cGAN** 的兩個核心網路：

1. **GeneratorSwinUNet**
   * 使用 *timm* 的 `swin_tiny_patch4_window7_224` 作為編碼器，
     透過 Transformer 捕捉長距離依賴。
   * 解碼器採用輕量級轉置卷積（ConvTranspose2d）並配有跳接
     （skip connection）以恢復空間解析度，最終回歸 4 維框位置
     修正量 Δ = (dx_rel, dy_rel, log dw, log dh)。

2. **ProjectionPatchGAN**
   * 基於 Spectral Normalization 的 PatchGAN。
   * 額外加入**投影項**（Projection）以強化條件資訊：
     對判別特徵向量與條件向量做內積後加到輸出上
     （詳見 Miyato et al., 2018）。

所有 Conv 層皆可透過 `weights_init_normal` 以 *N(0,0.02)* 初始化，
保持與原始 pix2pix 設定一致。
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# 需要安裝 timm 才能使用 Swin Transformer
try:
    from timm import create_model
except ImportError as e:
    raise ImportError("請先安裝 timm： pip install timm") from e

# ================================================================
#  公用：權重初始化（與 pix2pix 相同）
# ================================================================

def weights_init_normal(m: nn.Module) -> None:
    """將卷積層和線性層權重初始化為 N(0,0.02)，偏置為 0；
    Instance／BatchNorm 的 gamma 初始化為 N(1,0.02)。"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:  # 修復漏洞6：為完全自主學習優化Linear層初始化
        nn.init.normal_(m.weight.data, 0.0, 0.001)  # 使用極小的標準差，避免初始輸出過大
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1 or classname.find("InstanceNorm") != -1:
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

# ================================================================
#  生成器：Swin‑UNet
# ================================================================

class SwinDecoderBlock(nn.Module):
    """簡化的解碼塊：上採樣後與對應的 skip features 拼接，再經卷積融合。"""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.up(x)
        # 保證尺寸匹配（因應捲積偶爾產生的 1 像素差）
        if x.size(-1) != skip.size(-1):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        # 拼接後卷積融合
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class GeneratorSwinUNet(nn.Module):
    """Swin Transformer 作為編碼器的 U‑Net 生成器。"""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        # 完全自主學習，不設任何人為限制

        # -------------------- 編碼器：Swin --------------------
        # 使用 timm 取得所有層的特徵圖（out_indices = (0,1,2,3)）
        self.backbone = create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        # Swin‑Tiny 對 224×224 影像輸出 4 層：
        #   stage0: 56×56×96
        #   stage1: 28×28×192
        #   stage2: 14×14×384
        #   stage3:  7× 7×768

        enc_channels: List[int] = [96, 192, 384, 768]

        # -------------------- 解碼器 --------------------
        self.dec3 = SwinDecoderBlock(enc_channels[3], enc_channels[2], 384)  # 7  → 14
        self.dec2 = SwinDecoderBlock(384, enc_channels[1], 192)              # 14 → 28
        self.dec1 = SwinDecoderBlock(192, enc_channels[0], 96)               # 28 → 56
        self.dec0 = nn.Sequential(
            nn.ConvTranspose2d(96, 64, kernel_size=2, stride=2),             # 56 → 112
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # -------------------- 迴歸頭 --------------------
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 迴歸頭：輸出4維delta值
        self.fc_delta = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 4),
            # 完全自主學習，不設任何激活函數限制
        )
        
        # 修復漏洞6：為最後一層使用更小的權重初始化，避免初始輸出過大
        nn.init.normal_(self.fc_delta[1].weight, 0.0, 0.001)  # 更小的標準差
        nn.init.constant_(self.fc_delta[1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # 編碼：取得 Swin 每層特徵
        feats = self.backbone(x)  # len = 4
        # Swin 輸出格式為 (B, H, W, C)，需要轉換為 (B, C, H, W)
        f0, f1, f2, f3 = [feat.permute(0, 3, 1, 2) for feat in feats]  # 轉換維度順序

        # 解碼：逐層上採樣 + skip connection
        d3 = self.dec3(f3, f2)
        d2 = self.dec2(d3, f1)
        d1 = self.dec1(d2, f0)
        d0 = self.dec0(d1)      # 解析度回到 112×112（近似原圖的 0.875 倍）

        pooled = self.avg_pool(d0)
        delta = self.fc_delta(pooled)
        
        # 完全自主學習，直接返回生成器輸出，不設任何人為限制
        return delta

# ================================================================
#  判別器：Projection PatchGAN
# ================================================================

class ProjectionPatchGAN(nn.Module):
    """具投影條件的 Spectral Normalized PatchGAN。
    
    重要：在投影項中使用 detach() 防止 R1 正則化時對條件路徑重複求導，
    避免梯度範數爆炸（R1 penalty = ||∇_x D||²）。
    """

    def __init__(self, spectral_norm: bool = True, cond_dim: int = 3):
        super().__init__()
        self.spectral_norm = spectral_norm

        def sn_conv(in_ch: int, out_ch: int, k: int = 4, s: int = 2, p: int = 1):
            conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
            return nn.utils.spectral_norm(conv) if self.spectral_norm else conv

        # 主卷積路徑
        self.net = nn.Sequential(
            sn_conv(6, 64, 4, 2, 1),              # (pred+other) → C=6，大小 128→64
            nn.LeakyReLU(0.2, inplace=True),

            sn_conv(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            sn_conv(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            sn_conv(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 對每個 patch 給出線性 logits（不經 Sigmoid，供 Hinge 損失使用）
        # 使用 k=4, s=1, p=1 會將 14×14 變成 13×13，為了匹配投影尺寸，使用 p=2
        self.conv_logits = sn_conv(512, 1, k=4, s=1, p=2)  # 保持尺寸 14×14

        # 投影層：使用簡化的全域條件嵌入，避免複雜編碼器造成的梯度問題
        self.cond_embed = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                     # 全域平均池化到 1×1
            nn.Conv2d(cond_dim, 512, kernel_size=1),     # 映射到512維
            nn.LeakyReLU(0.2, inplace=True),
        )
        if self.spectral_norm:
            self.cond_embed[1] = nn.utils.spectral_norm(self.cond_embed[1])

    def forward(self, pred_patch: torch.Tensor, other_patch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # 1) 主路徑
        x = torch.cat([pred_patch, other_patch], dim=1)  # B × 6 × H × W
        feat = self.net(x)  # 經過下採樣後的特徵，尺寸約為 14×14×512
        logits = self.conv_logits(feat)  # B × 1 × h × w

        # 2) 投影項 - 使用簡化的全域條件嵌入
        # 關鍵修復：使用 .detach() 讓條件特徵不參與 R1 梯度計算
        # 添加數值穩定性檢查
        try:
            cond_global = self.cond_embed(other_patch).detach()  # B × 512 × 1 × 1 (detached)
        except Exception as e:
            print(f"Warning: Condition embedding failed: {e}")
            # 使用零向量作為fallback
            cond_global = torch.zeros(other_patch.size(0), 512, 1, 1, 
                                    device=other_patch.device, dtype=other_patch.dtype)
        # 將全域條件廣播到與feat相同的空間尺寸
        cond_feat = cond_global.expand(-1, -1, feat.size(2), feat.size(3))  # B × 512 × 14 × 14
        proj = torch.sum(feat * cond_feat, dim=1, keepdim=True)  # 逐元素相乘後求和 B × 1 × 14 × 14
        
        # 確保proj與logits尺寸匹配
        if proj.shape[-2:] != logits.shape[-2:]:
            proj = F.interpolate(proj, size=logits.shape[-2:], mode='bilinear', align_corners=False)

        # 3) 相加得到最終輸出
        return logits + proj
