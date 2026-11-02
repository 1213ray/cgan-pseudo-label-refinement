#!/usr/bin/env python3
"""
診斷生成器行為：檢查生成器對不同IoU樣本的實際輸出
這個腳本幫助我們理解生成器是否真的學會了選擇性校正
"""

import torch
import numpy as np
from pathlib import Path
import yaml
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models import GeneratorSwinUNet
from dataset import CalibratorDataset
from losses import apply_delta_to_bbox, iou_metric

def diagnose_generator_behavior(weights_path, data_dir, device='cuda'):
    """診斷生成器對不同IoU樣本的行為"""
    
    # 載入模型
    print("載入生成器模型...")
    checkpoint = torch.load(weights_path, map_location=device)
    config = checkpoint.get('config', {})
    
    netG = GeneratorSwinUNet(pretrained=False).to(device)   # 權重由 checkpoint 載入，不需下載預訓練權重
    if isinstance(checkpoint, dict) and 'generator' in checkpoint:
        netG.load_state_dict(checkpoint['generator'])
    else:
        netG.load_state_dict(checkpoint)
    netG.eval()
    
    # 載入數據集
    print("載入數據集...")
    dataset = CalibratorDataset(data_dir, img_size=224)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 收集診斷數據
    iou_ranges = {
        'very_low': (0.0, 0.3),
        'low': (0.3, 0.5), 
        'medium': (0.5, 0.7),
        'high': (0.7, 0.9),
        'very_high': (0.9, 1.0)
    }
    
    results = {range_name: {
        'ious': [],
        'delta_magnitudes': [],
        'delta_components': [],
        'improvements': []
    } for range_name in iou_ranges}
    
    print("開始診斷...")
    with torch.no_grad():
        for batch_idx, (pred_patch, gt_patch, delta_true, pred_box, gt_box, img_path) in enumerate(dataloader):
            pred_patch = pred_patch.to(device)
            pred_box = pred_box.to(device)
            gt_box = gt_box.to(device)

            # 生成器預測
            delta_pred = netG(pred_patch)

            # 計算校正後的框
            calibrated_boxes = apply_delta_to_bbox(pred_box, delta_pred, training=False)

            # 計算IoU改善
            iou_before = iou_metric(pred_box, gt_box)
            current_iou = iou_before   # 依校正前 IoU 分組
            iou_after = iou_metric(calibrated_boxes, gt_box)
            improvements = iou_after - iou_before
            
            # 計算delta幅度
            delta_magnitudes = torch.norm(delta_pred, dim=1)
            
            # 按IoU範圍分類
            for i in range(len(current_iou)):
                iou_val = current_iou[i].item()
                delta_mag = delta_magnitudes[i].item()
                delta_comp = delta_pred[i].cpu().numpy()
                improvement = improvements[i].item()
                
                for range_name, (min_iou, max_iou) in iou_ranges.items():
                    if min_iou <= iou_val < max_iou:
                        results[range_name]['ious'].append(iou_val)
                        results[range_name]['delta_magnitudes'].append(delta_mag)
                        results[range_name]['delta_components'].append(delta_comp)
                        results[range_name]['improvements'].append(improvement)
                        break
            
            if batch_idx >= 20:  # 限制樣本數量
                break
    
    # 分析結果
    print("\n" + "="*80)
    print("生成器行為診斷報告")
    print("="*80)
    
    for range_name, (min_iou, max_iou) in iou_ranges.items():
        data = results[range_name]
        if not data['ious']:
            continue
            
        avg_iou = np.mean(data['ious'])
        avg_delta_mag = np.mean(data['delta_magnitudes'])
        avg_improvement = np.mean(data['improvements'])
        
        print(f"\n📊 {range_name.upper()} IoU範圍 ({min_iou:.1f}-{max_iou:.1f}):")
        print(f"   樣本數: {len(data['ious'])}")
        print(f"   平均IoU: {avg_iou:.4f}")
        print(f"   平均Delta幅度: {avg_delta_mag:.4f}")
        print(f"   平均IoU改善: {avg_improvement:.4f}")
        
        # 分析delta組件
        delta_components = np.array(data['delta_components'])
        if len(delta_components) > 0:
            print(f"   Delta組件 (dx, dy, dw, dh):")
            print(f"     平均: [{delta_components.mean(axis=0)[0]:.4f}, {delta_components.mean(axis=0)[1]:.4f}, {delta_components.mean(axis=0)[2]:.4f}, {delta_components.mean(axis=0)[3]:.4f}]")
            print(f"     標準差: [{delta_components.std(axis=0)[0]:.4f}, {delta_components.std(axis=0)[1]:.4f}, {delta_components.std(axis=0)[2]:.4f}, {delta_components.std(axis=0)[3]:.4f}]")
    
    # 創建可視化
    create_diagnosis_plots(results, iou_ranges)
    
    return results

def create_diagnosis_plots(results, iou_ranges):
    """創建診斷圖表"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Delta幅度 vs IoU範圍
    range_names = []
    avg_delta_mags = []
    avg_improvements = []
    
    for range_name, (min_iou, max_iou) in iou_ranges.items():
        data = results[range_name]
        if data['ious']:
            range_names.append(f"{range_name}\n({min_iou:.1f}-{max_iou:.1f})")
            avg_delta_mags.append(np.mean(data['delta_magnitudes']))
            avg_improvements.append(np.mean(data['improvements']))
    
    axes[0, 0].bar(range_names, avg_delta_mags)
    axes[0, 0].set_title('平均Delta幅度 vs IoU範圍')
    axes[0, 0].set_ylabel('Delta幅度')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. IoU改善 vs IoU範圍
    axes[0, 1].bar(range_names, avg_improvements)
    axes[0, 1].set_title('平均IoU改善 vs IoU範圍')
    axes[0, 1].set_ylabel('IoU改善')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.7)
    
    # 3. Delta幅度分佈
    all_delta_mags = []
    all_ious = []
    for data in results.values():
        all_delta_mags.extend(data['delta_magnitudes'])
        all_ious.extend(data['ious'])
    
    axes[1, 0].scatter(all_ious, all_delta_mags, alpha=0.6)
    axes[1, 0].set_xlabel('原始IoU')
    axes[1, 0].set_ylabel('Delta幅度')
    axes[1, 0].set_title('Delta幅度 vs 原始IoU')
    
    # 4. IoU改善分佈
    all_improvements = []
    for data in results.values():
        all_improvements.extend(data['improvements'])
    
    axes[1, 1].scatter(all_ious, all_improvements, alpha=0.6)
    axes[1, 1].set_xlabel('原始IoU')
    axes[1, 1].set_ylabel('IoU改善')
    axes[1, 1].set_title('IoU改善 vs 原始IoU')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('generator_diagnosis.png', dpi=300, bbox_inches='tight')
    print(f"\n📈 診斷圖表已保存到: generator_diagnosis.png")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="診斷生成器行為")
    parser.add_argument("--weights", type=str, required=True, help="生成器權重路徑")
    parser.add_argument("--data-dir", type=str, required=True, help="數據集路徑")
    parser.add_argument("--device", type=str, default="cuda", help="計算設備")
    
    args = parser.parse_args()
    
    results = diagnose_generator_behavior(args.weights, args.data_dir, args.device)