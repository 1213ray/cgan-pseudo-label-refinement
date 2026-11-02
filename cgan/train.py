#!/usr/bin/env python3
"""
Enhanced CGAN training script with Swin-UNet Generator and Projection PatchGAN
Requires: pip install timm (for Swin Transformer backbone)

Usage (from the repository root):
    python cgan/train.py                                   # settings from cgan/config.yaml
    python cgan/train.py --data_dir datasets/cgan_train_set --batch_size 64
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
import numpy as np

try:
    import wandb
except ImportError:  # wandb 為選用；沒安裝時照常訓練，只是不記錄到 W&B
    wandb = None

# Setup path
project_root = Path(__file__).parent

from models import GeneratorSwinUNet, ProjectionPatchGAN, weights_init_normal
from dataset import CalibratorDataset
from losses import HybridLoss, apply_delta_to_bbox, iou_metric, compute_r1_penalty, hinge_loss_discriminator, hinge_loss_generator

def get_generator(generator_type, pretrained=True):
    """根據配置選擇生成器類型"""
    if generator_type == "swin":
        return GeneratorSwinUNet(pretrained=pretrained)
    else:
        # 保持向後相容性，默認使用 Swin-UNet
        return GeneratorSwinUNet(pretrained=pretrained)

# 全域圖像緩存
_IMAGE_CACHE = {}
_CACHE_MAX_SIZE = 100  # 最多緩存100張圖像

def get_refined_patch_batch(original_image_paths, pred_bboxes, deltas_pred, img_size, device, fallback_patches=None):
    """
    優化版本：減少CPU-GPU同步，提高GPU利用率
    主要改進：
    1. 減少不必要的CPU-GPU數據傳輸
    2. 使用多線程並行處理圖像
    3. 批量處理減少同步點
    """
    from torchvision import transforms
    from PIL import Image, ImageOps
    import concurrent.futures
    from threading import Lock
    
    # 預定義變換，避免重複創建
    transform_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # 在GPU上計算refined boxes，避免不必要的CPU傳輸
    with torch.no_grad():
        refined_boxes = apply_delta_to_bbox(pred_bboxes, deltas_pred, training=False)
        # 一次性轉到CPU，減少多次傳輸
        refined_boxes_cpu = refined_boxes.cpu()
        pred_bboxes_cpu = pred_bboxes.cpu()
    
    def process_single_image(args):
        """並行處理單個圖像的函數"""
        i, img_path, refined_box, pred_box = args
        
        try:
            # 線程安全的圖像緩存訪問
            with Lock():
                if img_path in _IMAGE_CACHE:
                    img = _IMAGE_CACHE[img_path]
                else:
                    img = Image.open(img_path).convert("RGB")
                    if len(_IMAGE_CACHE) >= _CACHE_MAX_SIZE:
                        oldest_key = next(iter(_IMAGE_CACHE))
                        del _IMAGE_CACHE[oldest_key]
                    _IMAGE_CACHE[img_path] = img
            
            W, H = img.size
            cx, cy, w, h = refined_box
            
            # 邊界檢查和修正
            cx = max(0.05, min(0.95, float(cx)))
            cy = max(0.05, min(0.95, float(cy)))
            w = max(0.02, min(0.8, float(w)))
            h = max(0.02, min(0.8, float(h)))
            
            # 計算像素座標
            px, py, pw, ph = cx * W, cy * H, w * W, h * H
            x1, y1 = max(0, px - pw / 2), max(0, py - ph / 2)
            x2, y2 = min(W, px + pw / 2), min(H, py + ph / 2)
            
            # 檢查邊界框有效性
            if x2 <= x1 or y2 <= y1 or (x2 - x1) < 10 or (y2 - y1) < 10:
                # 使用原始 pred_box 作為 fallback
                orig_cx, orig_cy, orig_w, orig_h = pred_box
                orig_px, orig_py = float(orig_cx) * W, float(orig_cy) * H
                orig_pw, orig_ph = float(orig_w) * W, float(orig_h) * H
                orig_x1, orig_y1 = max(0, orig_px - orig_pw / 2), max(0, orig_py - orig_ph / 2)
                orig_x2, orig_y2 = min(W, orig_px + orig_pw / 2), min(H, orig_py + orig_ph / 2)
                crop = img.crop((int(orig_x1), int(orig_y1), int(orig_x2), int(orig_y2)))
                fallback_used = True
            else:
                crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
                fallback_used = False

            # 修復：使用與數據集一致的letterbox處理
            # 直接使用CalibratorDataset的letterbox方法確保一致性
            from dataset import CalibratorDataset
            try:
                refined_box_tensor = torch.tensor([cx, cy, w, h], dtype=torch.float32)
                crop = CalibratorDataset._letterbox(img, refined_box_tensor, img_size)
                return i, transform_to_tensor(crop), fallback_used
            except Exception as letterbox_error:
                # 增強錯誤處理：記錄具體錯誤信息但不中斷訓練
                if fallback_patches is not None and i < len(fallback_patches):
                    return i, fallback_patches[i].cpu(), True
                else:
                    # 創建中性patch作為最後的fallback
                    neutral_patch = torch.zeros(3, img_size, img_size)
                    return i, neutral_patch, True
            
        except Exception as e:
            # 錯誤處理：返回fallback或中性patch
            if fallback_patches is not None and i < len(fallback_patches):
                return i, fallback_patches[i].cpu(), True
            else:
                neutral_patch = torch.zeros(3, img_size, img_size)
                return i, neutral_patch, True
    
    # 準備並行處理的參數
    process_args = [
        (i, original_image_paths[i], refined_boxes_cpu[i], pred_bboxes_cpu[i])
        for i in range(len(original_image_paths))
    ]
    
    # 並行處理所有圖像
    refined_patches = [None] * len(original_image_paths)
    fallback_count = 0
    
    # 使用較少的worker避免過度競爭
    max_workers = min(4, len(original_image_paths))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(process_single_image, args): args[0] for args in process_args}
        
        for future in concurrent.futures.as_completed(future_to_idx):
            try:
                idx, patch, is_fallback = future.result()
                refined_patches[idx] = patch
                if is_fallback:
                    fallback_count += 1
            except Exception as e:
                idx = future_to_idx[future]
                print(f"Error processing patch {idx}: {e}")
                refined_patches[idx] = torch.zeros(3, img_size, img_size)
                fallback_count += 1
    
    # 批量移動到GPU，使用non_blocking減少同步
    try:
        if all(patch is not None for patch in refined_patches):
            result = torch.stack(refined_patches).to(device, non_blocking=True)
        else:
            # 處理None值
            valid_patches = []
            for patch in refined_patches:
                if patch is not None:
                    valid_patches.append(patch)
                else:
                    valid_patches.append(torch.zeros(3, img_size, img_size))
            result = torch.stack(valid_patches).to(device, non_blocking=True)
    except Exception as e:
        print(f"Error stacking patches: {e}")
        result = torch.zeros(len(original_image_paths), 3, img_size, img_size, device=device)
    
    if fallback_count > 0:
        """print(f"Warning: {fallback_count}/{len(original_image_paths)} patches used fallback")"""

    return result

def main():
    # 載入配置檔案
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=config['data_dir'])
    parser.add_argument("--img_size", type=int, default=config['img_size'])
    parser.add_argument("--batch_size", type=int, default=config['batch_size'])
    parser.add_argument("--n_epochs", type=int, default=config['n_epochs'])
    parser.add_argument("--lr_g", type=float, default=config['lr_g'], help="Learning rate for the generator")
    parser.add_argument("--lr_d", type=float, default=config['lr_d'], help="Learning rate for the discriminator")
    parser.add_argument("--beta1", type=float, default=config['beta1'])
    parser.add_argument("--beta2", type=float, default=config['beta2'])
    parser.add_argument("--lambda_iou", type=float, default=config['lambda_iou'])
    # Pure EIoU configuration (simplified)
    parser.add_argument("--use_eiou", action='store_true', default=config.get('use_eiou', True))
    parser.add_argument("--pure_eiou", action='store_true', default=config.get('pure_eiou', True))
    parser.add_argument("--spectral_norm", action='store_true', default=config['spectral_norm'])
    parser.add_argument("--generator_type", type=str, default=config.get('generator_type', 'swin'))
    parser.add_argument("--pretrained", action='store_true', default=config.get('pretrained', True))
    parser.add_argument("--no_pretrained", dest='pretrained', action='store_false',
                        help="Do not download ImageNet weights for the Swin encoder")
    parser.add_argument("--num_workers", type=int, default=config.get('training', {}).get('num_workers', 4))
    parser.add_argument("--cond_dim", type=int, default=config.get('cond_dim', 3))
    parser.add_argument("--patience", type=int, default=config['early_stop']['patience'])
    parser.add_argument("--min_delta", type=float, default=config['early_stop']['min_delta'])
    parser.add_argument("--train_split", type=float, default=config['train_split'])
    parser.add_argument("--val_split", type=float, default=config['val_split'])
    parser.add_argument("--save_dir", type=str, default=config['save_dir'])
    parser.add_argument("--seed", type=int, default=config['seed'])
    # SN+R1+Hinge 參數 (替代WGAN-GP)
    parser.add_argument("--gan_mode", type=str, default=config.get('gan_mode', 'hinge'))
    parser.add_argument("--lambda_r1", type=float, default=config.get('lambda_r1', 10.0))
    parser.add_argument("--lambda_adv", type=float, default=config.get('lambda_adv', 0.1))
    parser.add_argument("--r1_every", type=int, default=config.get('r1_every', 16))
    parser.add_argument("--n_critic", type=int, default=config.get('n_critic', 1))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Generator type: {args.generator_type} (Swin-UNet with Transformer backbone)")
    print(f"Discriminator: Projection PatchGAN with Spectral Normalization")
    print(f"GAN mode: {args.gan_mode}")
    print(f"Using Enhanced Hybrid Loss - EIoU: {args.lambda_iou}, Adversarial: {args.lambda_adv}")
    print(f"Delta scale: 自主學習 (已移除人為限制)")
    print(f"Image size: {args.img_size}x{args.img_size} (required for Swin Transformer)")
    print(f"SN+R1+Hinge configuration:")
    print(f"  - R1 regularization λ: {args.lambda_r1}")
    print(f"  - R1 calculation frequency: every {args.r1_every} batches") 
    print(f"  - Discriminator updates per generator: {args.n_critic}:1")

    # Initialize W&B
    wandb_config = config.get('wandb', {})
    if wandb_config.get('enabled', False) and wandb is None:
        print("W&B is enabled in config.yaml but wandb is not installed; continuing without it")
        wandb_config['enabled'] = False
    if wandb_config.get('enabled', False):
        wandb.init(
            project=wandb_config.get('project', 'cgan-calibration'),
            entity=wandb_config.get('entity'),
            name=wandb_config.get('name'),
            tags=wandb_config.get('tags', []),
            notes=wandb_config.get('notes', ''),
            config={
                'img_size': args.img_size,
                'batch_size': args.batch_size,
                'n_epochs': args.n_epochs,
                'lr_g': args.lr_g,  # <--- 修改這裡
                'lr_d': args.lr_d,  # <--- 新增這裡
                'beta1': args.beta1,
                'beta2': args.beta2,
                'lambda_iou': args.lambda_iou,
                # Removed complex constraint parameters
                'use_eiou': args.use_eiou,
                'use_pure_eiou': args.pure_eiou,
                'spectral_norm': args.spectral_norm,
                'generator_type': args.generator_type,
                'patience': args.patience,
                'min_delta': args.min_delta,
                'train_split': args.train_split,
                'val_split': args.val_split,
                'seed': args.seed,
                'gan_mode': args.gan_mode,
                'lambda_r1': args.lambda_r1,
                'lambda_adv': args.lambda_adv,
                'r1_every': args.r1_every,
                'n_critic': args.n_critic,
                'data_dir': args.data_dir,
                'device': str(device)
            }
        )
        print("W&B initialized successfully!")
    else:
        print("W&B disabled in config")

    # Load dataset
    full_dataset = CalibratorDataset(args.data_dir, img_size=args.img_size)
    val_len = max(1, int(args.val_split * len(full_dataset)))
    train_len = len(full_dataset) - val_len
    train_set, val_set = random_split(full_dataset, [train_len, val_len])
    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    print(f"Training samples: {len(train_set)}, Validation samples: {len(val_set)}")

    # Initialize models
    netG = get_generator(args.generator_type, args.pretrained).to(device)
    netD = ProjectionPatchGAN(spectral_norm=args.spectral_norm, cond_dim=args.cond_dim).to(device)
    
    # 重要修復：只有在未使用預訓練權重時才重新初始化生成器
    # 避免覆蓋 Swin Transformer 的預訓練權重
    if not args.pretrained:
        print("Using random initialization for generator")
        netG.apply(weights_init_normal)
    else:
        print("Preserving pretrained Swin Transformer weights")
        # 只初始化解碼器部分（非預訓練的層）
        netG.dec3.apply(weights_init_normal)
        netG.dec2.apply(weights_init_normal) 
        netG.dec1.apply(weights_init_normal)
        netG.dec0.apply(weights_init_normal)
        netG.fc_delta.apply(weights_init_normal)
    
    netD.apply(weights_init_normal)
    
    print(f"Generator parameters: {sum(p.numel() for p in netG.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in netD.parameters()):,}")

    # Log model info to W&B
    if wandb_config.get('enabled', False):
        wandb.log({
            'model/generator_params': sum(p.numel() for p in netG.parameters()),
            'model/discriminator_params': sum(p.numel() for p in netD.parameters()),
            'dataset/train_samples': len(train_set),
            'dataset/val_samples': len(val_set),
            'dataset/total_samples': len(full_dataset)
        })
        # Watch models for gradient and parameter tracking
        wandb.watch(netG, log='all', log_freq=100)
        wandb.watch(netD, log='all', log_freq=100)

    # SN+R1+Hinge 不需要傳統的BCE損失，使用 Hinge Loss
    
    # Enhanced Hybrid Loss: EIoU + Adversarial
    criterion_hybrid = HybridLoss(lambda_iou=args.lambda_iou, lambda_adv=args.lambda_adv)

    # Optimizers - SN+R1+Hinge 使用 Adam，β1=0.0 提升穩定性
    optimizer_G = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizer_D = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    # Learning rate schedulers
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='max', factor=0.5, patience=5)
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='max', factor=0.5, patience=5)

    # Output directory
    if wandb_config.get('enabled', False) and wandb.run:
        out_root = Path(args.save_dir).parent / wandb.run.name
        print(f"Redirecting save directory to W&B run: {out_root}")
    else:
        out_root = Path(args.save_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "samples").mkdir(exist_ok=True)
    ckpt_best = out_root / "G_best.pth"

    best_val_iou, epochs_no_improve = -1.0, 0
    training_history = []
    
    print("Starting enhanced training...")
    for epoch in range(1, args.n_epochs + 1):
        netG.train()
        netD.train()
        
        # Pure EIoU loss doesn't need epoch setting
        
        epoch_stats = {
            'loss_G': 0.0,
            'loss_D': 0.0,
            'loss_iou': 0.0,
            'loss_hinge_g': 0.0,  # 生成器Hinge損失
            'loss_hinge_d': 0.0,  # 判別器Hinge損失
            'loss_r1': 0.0,       # R1正則化
            'loss_adv': 0.0,      # 對抗損失
            'gan_score': 0.0      # GAN得分 (D(fake)的均值，越高越好)
        }
        
        for i, (pred_patch, gt_patch, delta_true, pred_box, gt_box, img_path) in enumerate(train_loader):
            
            pred_patch = pred_patch.to(device)
            gt_patch = gt_patch.to(device)
            delta_true = delta_true.to(device)
            pred_box = pred_box.to(device)
            gt_box = gt_box.to(device)  # 關鍵修復：加載原始GT框

            batch_size = pred_patch.size(0)

            # ===== SN+R1+Hinge 訓練循環：1:1 更新比例，簡化且穩定 =====
            
            # 1. 訓練判別器 (使用Hinge Loss + R1正則化)
            for critic_step in range(args.n_critic):  # 通常為1
                optimizer_D.zero_grad()

                # 真實樣本判別
                real_pred = netD(gt_patch, pred_patch)  # (pred_patch, other_patch)
                
                # 生成假樣本 (detached - 判別器訓練時生成器不更新)
                with torch.no_grad():
                    delta_pred_detached = netG(pred_patch)
                    # 效能優化：預先detach並轉CPU避免不必要的GPU記憶體和graph追蹤
                    fallback_cpu = pred_patch.detach().cpu()
                    refined_patch = get_refined_patch_batch(
                        img_path, pred_box, delta_pred_detached, args.img_size, device, 
                        fallback_patches=fallback_cpu
                    )
                fake_pred = netD(refined_patch, pred_patch)  # (pred_patch, other_patch)
                
                # Hinge Loss for Discriminator
                d_hinge_loss = hinge_loss_discriminator(real_pred, fake_pred)
                
                # 計算基礎判別器損失
                d_loss = d_hinge_loss
                
                # R1正則化 (每r1_every個batch計算一次以節省計算)
                r1_penalty_value = 0.0
                if i % args.r1_every == 0:
                    # 單獨計算R1正則化，使用新的前向傳播
                    netD.train()  # 確保在訓練模式
                    r1_penalty = args.lambda_r1 * compute_r1_penalty(netD, pred_patch, gt_patch)
                    r1_penalty_value = r1_penalty.item()
                    d_loss = d_loss + r1_penalty
                
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)
                optimizer_D.step()
                
                # 記錄判別器相關損失
                epoch_stats['loss_D'] += d_loss.item()
                epoch_stats['loss_hinge_d'] += d_hinge_loss.item() 
                epoch_stats['loss_r1'] += r1_penalty_value

            # 2. 訓練生成器 (使用EIoU + Hinge對抗損失)
            optimizer_G.zero_grad()
            
            # 生成器前向傳播
            delta_pred = netG(pred_patch)
            
            # 使用原始GT進行損失計算 - 修復：統一訓練和驗證行為
            calibrated_boxes = apply_delta_to_bbox(pred_box, delta_pred, training=False)
            
            # 生成refined patch用於對抗損失
            # 效能優化：預先detach並轉CPU
            fallback_cpu_g = pred_patch.detach().cpu()
            refined_patch_for_G = get_refined_patch_batch(
                img_path, pred_box, delta_pred, args.img_size, device, 
                fallback_patches=fallback_cpu_g
            )
            fake_pred_for_G = netD(refined_patch_for_G, pred_patch)  # (pred_patch, other_patch)
            
            # Hinge Loss for Generator
            g_hinge_loss = hinge_loss_generator(fake_pred_for_G)
            
            # 使用原始GT進行損失計算
            loss_G, loss_iou, loss_adv = criterion_hybrid(
                calibrated_boxes, gt_box, adv_loss=g_hinge_loss
            )
            
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
            optimizer_G.step()

            # 記錄生成器相關損失
            epoch_stats['loss_G'] += loss_G.item()
            epoch_stats['loss_iou'] += loss_iou.item()
            epoch_stats['loss_hinge_g'] += g_hinge_loss.item()
            epoch_stats['loss_adv'] += loss_adv.item()
            # 計算GAN得分 (D(fake)的均值，用於更直觀的監控)
            epoch_stats['gan_score'] += (-g_hinge_loss.item())

            # Save sample images occasionally
            if i == 0 and epoch % 10 == 1:
                try:
                    img_sample = torch.cat((pred_patch.data[:4], refined_patch_for_G.data[:4], gt_patch.data[:4]), -2)
                    sample_path = out_root / f"samples/epoch_{epoch}.png"
                    save_image(img_sample, sample_path, nrow=4, normalize=True)
                    
                    # Log sample images to W&B
                    if wandb_config.get('enabled', False):
                        wandb.log({
                            "samples/training_images": wandb.Image(
                                str(sample_path),
                                caption=f"Epoch {epoch}: Pred | Refined | GT"
                            )
                        })
                except Exception as e:
                    print(f"Warning: Could not save sample image: {e}")

        # Validation
        netG.eval()
        with torch.no_grad():
            iou_sum_before, iou_sum_after = 0.0, 0.0
            n_val_samples = 0
            # 方案一修改：接收原始GT框
            for pred_patch_val, _, delta_true_val, pred_box_val, gt_box_val, _ in val_loader:
                batch_size_val = pred_patch_val.size(0)
                pred_patch_val = pred_patch_val.to(device)
                delta_true_val = delta_true_val.to(device)
                pred_box_val = pred_box_val.to(device)
                gt_box_val = gt_box_val.to(device)  # 原始GT框
                
                delta_pred_val = netG(pred_patch_val)

                # 修復漏洞4：驗證時使用training=False，與實際推理保持一致
                calibrated_boxes = apply_delta_to_bbox(pred_box_val, delta_pred_val, training=False)
                
                # 直接使用原始GT，不再重構
                iou_before = iou_metric(pred_box_val, gt_box_val)
                iou_after = iou_metric(calibrated_boxes, gt_box_val)

                iou_sum_before += iou_before.sum().item()
                iou_sum_after += iou_after.sum().item()
                n_val_samples += batch_size_val
            
            mean_iou_before = iou_sum_before / max(1, n_val_samples)
            mean_iou_after = iou_sum_after / max(1, n_val_samples)
            delta_iou = mean_iou_after - mean_iou_before

        # Calculate average losses
        for key in epoch_stats:
            epoch_stats[key] /= len(train_loader)
        
        # Update learning rate
        scheduler_G.step(delta_iou)
        scheduler_D.step(delta_iou)
        
        # Store training history
        training_history.append({
            'epoch': epoch,
            'delta_iou': delta_iou,
            'mean_iou_before': mean_iou_before,
            'mean_iou_after': mean_iou_after,
            **epoch_stats
        })
        
        print(f"[Epoch {epoch}/{args.n_epochs}] "
              f"G: {epoch_stats['loss_G']:.3f} "
              f"D: {epoch_stats['loss_D']:.3f} "
              f"EIoU: {epoch_stats['loss_iou']:.3f} "
              f"HingeG: {epoch_stats['loss_hinge_g']:.3f} "
              f"HingeD: {epoch_stats['loss_hinge_d']:.3f} "
              f"R1: {epoch_stats['loss_r1']:.4f} "
              f"GAN: {epoch_stats['gan_score']:.3f} "
              f"ΔIoU: {delta_iou:.4f} "
              f"Before: {mean_iou_before:.4f} "
              f"After: {mean_iou_after:.4f}")

        # Log metrics to W&B
        if wandb_config.get('enabled', False):
            log_dict = {
                'epoch': epoch,
                'train/loss_G': epoch_stats['loss_G'],
                'train/loss_D': epoch_stats['loss_D'],
                'train/loss_iou': epoch_stats['loss_iou'],
                'train/loss_hinge_g': epoch_stats['loss_hinge_g'],
                'train/loss_hinge_d': epoch_stats['loss_hinge_d'],
                'train/loss_r1': epoch_stats['loss_r1'],
                'train/loss_adv': epoch_stats['loss_adv'],
                'train/gan_score': epoch_stats['gan_score'],
                'val/delta_iou': delta_iou,
                'val/mean_iou_before': mean_iou_before,
                'val/mean_iou_after': mean_iou_after,
                'val/improvement': delta_iou,
                'learning_rate/generator': optimizer_G.param_groups[0]['lr'],
                'learning_rate/discriminator': optimizer_D.param_groups[0]['lr']
            }
            
            # Add focal loss if available
            # Simplified - no additional losses to track
            
            wandb.log(log_dict)
        
        # 檢查是否有數值問題
        loss_tensor = torch.tensor([epoch_stats['loss_G'], epoch_stats['loss_D']], dtype=torch.float32)
        if torch.isnan(loss_tensor).any() or torch.isinf(loss_tensor).any():
            print(f"Warning: NaN or Inf detected in losses! G: {epoch_stats['loss_G']:.6f}, D: {epoch_stats['loss_D']:.6f}")
            print("Stopping training to prevent further instability.")
            break

        # Early stopping with improved criteria
        if delta_iou > best_val_iou + args.min_delta:
            best_val_iou = delta_iou
            torch.save({
                'generator': netG.state_dict(),
                'discriminator': netD.state_dict(),
                'epoch': epoch,
                'delta_iou': delta_iou,
                'config': config
            }, ckpt_best)
            epochs_no_improve = 0
            print(f"New best model saved! Delta IoU = {best_val_iou:.4f}")
            
            # Log best model to W&B
            if wandb_config.get('enabled', False):
                wandb.log({
                    'best_model/delta_iou': best_val_iou,
                    'best_model/epoch': epoch
                })
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered.")
                break

    # Save training history
    import json
    with open(out_root / "training_history.json", 'w') as f:
        json.dump(training_history, f, indent=2)

    print(f"Training complete. Best Delta IoU = {best_val_iou:.4f}")
    print(f"Model saved to: {ckpt_best}")

    # Final W&B logging and artifact saving
    if wandb_config.get('enabled', False):
        # Log final metrics
        wandb.log({
            'final/best_delta_iou': best_val_iou,
            'final/total_epochs': epoch,
            'final/early_stopped': epochs_no_improve >= args.patience
        })
        
        # Save model artifacts
        model_artifact = wandb.Artifact(
            name=f"cgan_model_{wandb.run.id}",
            type="model",
            description="Best CGAN model for pseudo-label calibration"
        )
        model_artifact.add_file(str(ckpt_best))
        wandb.log_artifact(model_artifact)
        
        # Save training history
        history_artifact = wandb.Artifact(
            name=f"training_history_{wandb.run.id}",
            type="dataset",
            description="Training history and metrics"
        )
        history_artifact.add_file(str(out_root / "training_history.json"))
        wandb.log_artifact(history_artifact)
        
        # Save sample images
        if (out_root / "samples").exists():
            samples_artifact = wandb.Artifact(
                name=f"training_samples_{wandb.run.id}",
                type="dataset",
                description="Sample images during training"
            )
            samples_artifact.add_dir(str(out_root / "samples"))
            wandb.log_artifact(samples_artifact)
        
        print("W&B artifacts saved successfully!")
        wandb.finish()

if __name__ == "__main__":
    main()