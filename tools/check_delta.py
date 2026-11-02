#!/usr/bin/env python3
"""
測試數據集中delta計算的正確性
檢查apply_delta是否真的能改善IoU：用真實 Δ 解碼回來的框，應該要比原本的預測框更接近真實框。

用法（在 repo 根目錄）：python tools/check_delta.py [datasets/cgan_train_set]
"""
import sys
import torch
import random
from pathlib import Path

# 添加cgan目錄到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'cgan'))

from dataset import CalibratorDataset
from losses import apply_delta_to_bbox, iou_metric

def test_delta_correctness(dataset_path="datasets/cgan_train_set"):
    """測試數據集中delta的正確性"""
    print("🧪 測試數據集中delta計算的正確性...")

    # 載入數據集
    dataset = CalibratorDataset(dataset_path, img_size=224)
    
    print(f"數據集樣本數: {len(dataset)}")
    
    # 隨機測試10個樣本
    test_indices = random.sample(range(len(dataset)), min(10, len(dataset)))
    
    total_improvement = 0
    positive_count = 0
    
    for i, idx in enumerate(test_indices):
        print(f"\n--- 測試樣本 {i+1} (索引 {idx}) ---")
        
        try:
            # 獲取數據集樣本
            pred_patch, gt_patch, delta_true, pred_box, gt_box, img_path = dataset[idx]
            
            print(f"pred_box: {pred_box.tolist()}")
            print(f"gt_box: {gt_box.tolist()}")
            print(f"delta_true: {delta_true.tolist()}")
            
            # 計算原始IoU
            pred_box_batch = pred_box.unsqueeze(0)
            gt_box_batch = gt_box.unsqueeze(0)
            original_iou = iou_metric(pred_box_batch, gt_box_batch).item()
            
            # 使用真實delta進行校正
            delta_true_batch = delta_true.unsqueeze(0)
            corrected_box = apply_delta_to_bbox(pred_box_batch, delta_true_batch, training=False)
            corrected_iou = iou_metric(corrected_box, gt_box_batch).item()
            
            # 計算改善
            improvement = corrected_iou - original_iou
            total_improvement += improvement
            if improvement > 0:
                positive_count += 1
            
            print(f"原始IoU: {original_iou:.4f}")
            print(f"校正後IoU: {corrected_iou:.4f}")
            print(f"改善: {improvement:.4f}")
            
            if improvement < -0.01:
                print("❌ 警告：真實delta導致IoU惡化！")
            elif improvement > 0.01:
                print("✅ 真實delta成功改善IoU")
            else:
                print("➖ 真實delta幾乎無改善")
                
        except Exception as e:
            print(f"❌ 測試樣本 {idx} 時出錯: {e}")
    
    # 總結
    avg_improvement = total_improvement / len(test_indices)
    positive_ratio = positive_count / len(test_indices)
    
    print(f"\n📊 測試總結:")
    print(f"平均IoU改善: {avg_improvement:.4f}")
    print(f"正向改善比例: {positive_ratio:.1%}")
    
    if avg_improvement < 0:
        print("🚨 嚴重問題：數據集中的delta計算有誤，導致IoU惡化！")
    elif avg_improvement < 0.01:
        print("⚠️ 問題：數據集中的delta改善效果很小")
    else:
        print("✅ 數據集中的delta計算基本正確")

if __name__ == "__main__":
    test_delta_correctness(*sys.argv[1:2])