r"""
比較真實標籤 (GT)、校準前標籤、校準後標籤的品質。

此腳本會執行以下操作：
1.  分別計算「校準前」和「校準後」標籤相對於 GT 的 TP, FP, FN, Precision, Recall, F1-Score。
2.  基於 IoU 變化量 (Delta IoU) 來分析校準成效。

用法（在 repo 根目錄）:
    python pipeline/07_evaluate_pseudo_labels.py \
        --gt-dir   datasets/person/part3/labels \
        --pre-dir  datasets/two_stage/stage2/high/labels \
        --post-dir datasets/calibrated/labels \
        --iou-threshold 0.5
"""
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

# --- Utility Functions ---

def iou_yolo(b1, b2):
    """計算兩個 YOLO 格式 (cx, cy, w, h) 邊界框的 IoU。"""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa1, ya1, xa2, ya2 = x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2
    xb1, yb1, xb2, yb2 = x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2
    iw = max(0, min(xa2, xb2) - max(xa1, xb1))
    ih = max(0, min(ya2, yb2) - max(ya1, yb1))
    inter = iw * ih
    union = w1 * h1 + w2 * h2 - inter
    return 0. if union == 0 else inter / union

def load_labels(path: Path):
    """將 YOLO 標籤檔案載入為 (class_id, (cx, cy, w, h)) 的列表。"""
    if not path.exists(): return []
    out = []
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        parts = list(map(float, line.split()))
        out.append((int(parts[0]), tuple(parts[1:5])))
    return out

def calculate_metrics(pred_boxes, gt_boxes, iou_threshold):
    """執行貪婪匹配並計算 TP, FP, FN。"""
    stats = {"tp": 0, "fp": 0, "fn": 0}
    matched_preds = set()
    for gt_cls, gt_box in gt_boxes:
        best_iou, best_pred_idx = -1, -1
        for i, (pred_cls, pred_box) in enumerate(pred_boxes):
            if i in matched_preds or pred_cls != gt_cls: continue
            iou = iou_yolo(gt_box, pred_box)
            if iou > best_iou:
                best_iou, best_pred_idx = iou, i
        if best_iou >= iou_threshold:
            stats["tp"] += 1
            if best_pred_idx != -1: matched_preds.add(best_pred_idx)
        else:
            stats["fn"] += 1
    stats["fp"] = len(pred_boxes) - len(matched_preds)
    return stats

def print_report(title, stats, total_images, missing_files=0):
    """印出格式化的評估報告。"""
    p = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0
    r = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print(f"\n=== {title} ({total_images} 張圖像) ===")
    print(f"TP: {stats['tp']:<6} FP: {stats['fp']:<6} FN: {stats['fn']:<6}")
    print(f"精確率: {p:.3f}  召回率: {r:.3f}  F1-Score: {f1:.3f}")
    if missing_files > 0:
        print(f"警告：因檔案不齊全，跳過了 {missing_files} 組標籤。")

# --- Main Logic ---

def main(args):
    gt_dir, pre_dir, post_dir = Path(args.gt_dir), Path(args.pre_dir), Path(args.post_dir)

    stats_pre, stats_post = defaultdict(int), defaultdict(int)
    delta_ious = []
    total_gt_boxes = 0
    total_files, missing_files = 0, 0

    label_files = list(post_dir.glob("*.txt"))
    if not label_files:
        print(f"錯誤：在校準後標籤目錄中找不到任何 .txt 檔案: {post_dir}")
        return

    for post_file in label_files:
        basename = post_file.name
        gt_file, pre_file = gt_dir / basename, pre_dir / basename

        if not (gt_file.exists() and pre_file.exists()):
            missing_files += 1
            continue
        
        total_files += 1
        gt_boxes = load_labels(gt_file)
        pre_boxes = load_labels(pre_file)
        post_boxes = load_labels(post_file)
        total_gt_boxes += len(gt_boxes)

        # 1. 計算校準前後的整體指標
        pre_metrics = calculate_metrics(pre_boxes, gt_boxes, args.iou_threshold)
        post_metrics = calculate_metrics(post_boxes, gt_boxes, args.iou_threshold)
        for key in pre_metrics: stats_pre[key] += pre_metrics[key]
        for key in post_metrics: stats_post[key] += post_metrics[key]

        # 2. 根據每個 GT 框計算 Delta IoU，並記錄原始 IoU 用於分級分析
        for gt_cls, gt_box in gt_boxes:
            best_iou_pre = max([iou_yolo(gt_box, pre_box) for pre_cls, pre_box in pre_boxes if pre_cls == gt_cls], default=0.0)
            best_iou_post = max([iou_yolo(gt_box, post_box) for post_cls, post_box in post_boxes if post_cls == gt_cls], default=0.0)
            delta_ious.append((best_iou_pre, best_iou_post, best_iou_post - best_iou_pre))

    # --- 最終報告 ---
    print(f"使用 IoU 閾值進行比較: {args.iou_threshold}")
    print_report("校準前 (Before)", stats_pre, total_files, missing_files)
    print_report("校準後 (After)", stats_post, total_files)
    
    # 詳細的 Delta IoU 分級分析
    if delta_ious:
        # 提取數據
        ious_pre = np.array([item[0] for item in delta_ious])
        ious_post = np.array([item[1] for item in delta_ious])
        deltas = np.array([item[2] for item in delta_ious])
        
        print("\n" + "="*80)
        print("詳細 Delta IoU 校準分析")
        print("="*80)
        
        # 整體統計
        print(f"📊 整體統計:")
        print(f"   分析的 GT 邊界框總數: {total_gt_boxes}")
        print(f"   平均 IoU (校準前): {np.mean(ious_pre):.4f}")
        print(f"   平均 IoU (校準後): {np.mean(ious_post):.4f}")
        print(f"   平均 Delta IoU: {np.mean(deltas):+.4f}")
        print(f"   Delta IoU 標準差: {np.std(deltas):.4f}")
        
        # 改善統計
        improved = np.sum(deltas > 1e-6)
        degraded = np.sum(deltas < -1e-6)
        unchanged = np.sum(np.abs(deltas) <= 1e-6)
        
        print(f"\n📈 改善統計:")
        print(f"   IoU 提升的框數: {improved:>6} ({improved/len(deltas):.1%})")
        print(f"   IoU 下降的框數: {degraded:>6} ({degraded/len(deltas):.1%})")
        print(f"   IoU 不變的框數: {unchanged:>6} ({unchanged/len(deltas):.1%})")
        
        # 定義 IoU 級距
        iou_ranges = [
            ("極低 (<0.3)", 0.0, 0.3),
            ("低 (0.3-0.5)", 0.3, 0.5),
            ("中 (0.5-0.7)", 0.5, 0.7),
            ("高 (0.7-0.9)", 0.7, 0.9),
            ("極高 (≥0.9)", 0.9, 1.0)
        ]
        
        print(f"\n🎯 分級別改善分析:")
        print(f"{'級別':<12} {'數量':<8} {'校準前IoU':<12} {'校準後IoU':<12} {'平均改善':<12} {'改善率':<8}")
        print("-" * 80)
        
        for range_name, min_iou, max_iou in iou_ranges:
            # 找到屬於此級距的樣本
            if max_iou == 1.0:  # 處理最後一個範圍
                mask = (ious_pre >= min_iou) & (ious_pre <= max_iou)
            else:
                mask = (ious_pre >= min_iou) & (ious_pre < max_iou)
            
            if np.sum(mask) > 0:
                range_ious_pre = ious_pre[mask]
                range_ious_post = ious_post[mask]
                range_deltas = deltas[mask]
                
                count = len(range_ious_pre)
                avg_pre = np.mean(range_ious_pre)
                avg_post = np.mean(range_ious_post)
                avg_delta = np.mean(range_deltas)
                improve_rate = np.sum(range_deltas > 1e-6) / count
                
                print(f"{range_name:<12} {count:<8} {avg_pre:<12.4f} {avg_post:<12.4f} {avg_delta:<+12.4f} {improve_rate:<8.1%}")
            else:
                print(f"{range_name:<12} {'0':<8} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<8}")
        
        # 改善幅度分析
        print(f"\n📊 改善幅度分布:")
        delta_ranges = [
            ("大幅惡化 (<-0.1)", -float('inf'), -0.1),
            ("中度惡化 (-0.1~-0.01)", -0.1, -0.01),
            ("輕微惡化 (-0.01~0)", -0.01, 0),
            ("基本不變 (±0.001)", -0.001, 0.001),
            ("輕微改善 (0~0.01)", 0, 0.01),
            ("中度改善 (0.01~0.1)", 0.01, 0.1),
            ("大幅改善 (>0.1)", 0.1, float('inf'))
        ]
        
        for range_name, min_delta, max_delta in delta_ranges:
            if min_delta == -float('inf'):
                mask = deltas < max_delta
            elif max_delta == float('inf'):
                mask = deltas > min_delta
            elif range_name.startswith("基本不變"):
                mask = (deltas >= min_delta) & (deltas <= max_delta)
            else:
                mask = (deltas >= min_delta) & (deltas < max_delta)
            
            count = np.sum(mask)
            percentage = count / len(deltas) * 100
            
            if count > 0:
                avg_original_iou = np.mean(ious_pre[mask])
                print(f"   {range_name:<20} {count:>6} ({percentage:>5.1f}%) - 原始IoU: {avg_original_iou:.3f}")
            else:
                print(f"   {range_name:<20} {count:>6} ({percentage:>5.1f}%)")
        
        # 最佳和最差案例
        best_idx = np.argmax(deltas)
        worst_idx = np.argmin(deltas)
        
        print(f"\n🏆 極值案例:")
        print(f"   最佳改善: {deltas[best_idx]:+.4f} (從 {ious_pre[best_idx]:.4f} 到 {ious_post[best_idx]:.4f})")
        print(f"   最差惡化: {deltas[worst_idx]:+.4f} (從 {ious_pre[worst_idx]:.4f} 到 {ious_post[worst_idx]:.4f})")
        
        # 診斷建議
        print(f"\n💡 診斷建議:")
        if np.mean(deltas) > 0.01:
            print("   ✅ 校準系統整體表現良好，有明顯的IoU改善")
        elif np.mean(deltas) > 0.001:
            print("   ➖ 校準系統有輕微改善，但效果不明顯")
        elif np.mean(deltas) > -0.001:
            print("   ⚠️  校準系統基本無效果，建議檢查模型訓練")
        else:
            print("   🚨 校準系統反而惡化了結果，存在嚴重問題")
        
        # 針對不同級距的建議
        low_iou_mask = ious_pre < 0.5
        high_iou_mask = ious_pre >= 0.7
        
        if np.sum(low_iou_mask) > 0:
            low_iou_improvement = np.mean(deltas[low_iou_mask])
            if low_iou_improvement < 0:
                print("   🎯 低IoU樣本(需要校正)反而惡化，模型可能過度校正")
            elif low_iou_improvement > 0.02:
                print("   🎯 低IoU樣本改善良好，模型能有效識別需要校正的框")
        
        if np.sum(high_iou_mask) > 0:
            high_iou_improvement = np.mean(deltas[high_iou_mask])
            if high_iou_improvement < -0.01:
                print("   🎯 高IoU樣本(已經很好)被過度校正，建議提高選擇性")
            elif abs(high_iou_improvement) < 0.005:
                print("   🎯 高IoU樣本基本保持不變，模型具有良好的選擇性")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="比較 GT、校準前和校準後標籤的品質。", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--gt-dir', type=str, required=True, help="真實標籤 (Ground Truth) 的資料夾路徑。")
    parser.add_argument('--pre-dir', type=str, required=True, help="校準前 (偽) 標籤的資料夾路徑。")
    parser.add_argument('--post-dir', type=str, required=True, help="校準後標籤的資料夾路徑。")
    parser.add_argument('--iou-threshold', type=float, default=0.5, help="用於判斷是否匹配 (TP) 的 IoU 閾值。預設值: 0.5")
    args = parser.parse_args()
    main(args)
