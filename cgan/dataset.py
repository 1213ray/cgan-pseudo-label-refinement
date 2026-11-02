"""
CGAN Dataset Module - 簡化和優化版本
用於加載和預處理 YOLO 格式的訓練數據
"""

import math
import os
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageOps
import yaml

class CalibratorDataset(Dataset):
    """動態構建 (pred_patch, gt_patch, Δ_true, pred_bbox, img_path) 元組。

    預期在 *root* 下的目錄結構 (YOLO 風格):
      images/        *.jpg / *.png  (原始解析度)
      labels_gt/     *.txt          (類別 cx cy w h)
      labels_pred/   *.txt          (偽標籤 cx cy w h conf)
    所有座標都已正規化 (YOLO)。
    """

    def __init__(self, root: str | os.PathLike, img_size: int = None, iou_thr: float = None, transform: transforms.Compose | None = None):
        super().__init__()
        self.root = Path(root)
        self.img_dir = self.root / "images"
        self.gt_dir = self.root / "labels_gt"
        self.pred_dir = self.root / "labels_pred"
        
        # 載入預設配置
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if img_size is None:
            img_size = config['img_size']
        if iou_thr is None:
            iou_thr = config['iou_threshold']
        
        self.img_size = int(img_size)
        self.iou_thr = float(iou_thr)

        self.samples: List[Tuple[Path, int, torch.Tensor, torch.Tensor, torch.Tensor]] = []  # 添加gt_box
        self._prepare_index()

        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # 轉換到 (-1,1)
        ])

    # ---------------------  utils ---------------------

    @staticmethod
    def _bbox_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
        """計算 [cx,cy,w,h] 正規化座標的邊界框 IoU。"""
        b1_x1 = box1[0] - box1[2] / 2
        b1_y1 = box1[1] - box1[3] / 2
        b1_x2 = box1[0] + box1[2] / 2
        b1_y2 = box1[1] + box1[3] / 2
        b2_x1 = box2[0] - box2[2] / 2
        b2_y1 = box2[1] - box2[3] / 2
        b2_x2 = box2[0] + box2[2] / 2
        b2_y2 = box2[1] + box2[3] / 2

        inter = max(0, min(b1_x2, b2_x2) - max(b1_x1, b2_x1)) * max(0, min(b1_y2, b2_y2) - max(b1_y1, b2_y1))
        union = (b1_x2 - b1_x1) * (b1_y2 - b1_y1) + (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _bbox2delta(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """
        計算穩定的 Δ = (dx_rel, dy_rel, log dw, log dh)
        改進的歸一化方法，增強數值穩定性
        """
        # 載入配置中的數值常量
        try:
            config_path = Path(__file__).parent / "config.yaml"
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            eps = float(config.get('numerical', {}).get('eps', 1e-6))  # 確保是浮點數
        except:
            eps = 1e-6
        
        # 確保輸入是數值類型 - 修復類型錯誤
        try:
            # 更強健的類型轉換
            def safe_float_convert(val):
                if isinstance(val, str):
                    return float(val)
                elif hasattr(val, 'item'):
                    return float(val.item())
                else:
                    return float(val)
            
            gt_vals = [safe_float_convert(gt[i]) for i in range(4)]
            pred_vals = [safe_float_convert(pred[i]) for i in range(4)]
            
        except (ValueError, TypeError, IndexError) as e:
            print(f"Warning: Invalid bbox values - gt: {gt}, pred: {pred}, error: {e}")
            return torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        
        # 使用固定的歸一化因子，避免小框產生過大的 delta 值
        # 改用 pred 框的幾何平均尺寸作為歸一化因子
        pred_area = pred_vals[2] * pred_vals[3]
        norm_factor = max(math.sqrt(pred_area), 0.05)  # 設置最小歸一化因子
        
        dx = (gt_vals[0] - pred_vals[0]) / norm_factor
        dy = (gt_vals[1] - pred_vals[1]) / norm_factor
        
        # 對數比值計算，加強數值保護
        gt_w = max(gt_vals[2], float(eps))  # 確保 eps 是浮點數
        gt_h = max(gt_vals[3], float(eps))
        pred_w = max(pred_vals[2], float(eps))
        pred_h = max(pred_vals[3], float(eps))
        
        # 限制比值範圍，避免極端情況
        w_ratio = max(0.1, min(10.0, gt_w / pred_w))
        h_ratio = max(0.1, min(10.0, gt_h / pred_h))
        
        dw = math.log(w_ratio)
        dh = math.log(h_ratio)
        
        return torch.tensor([dx, dy, dw, dh], dtype=torch.float32)

    @staticmethod
    def _letterbox(img: Image.Image, bbox_xywh: torch.Tensor, out_size: int) -> Image.Image:
        """裁剪邊界框區域 → 填充成正方形 → 調整大小 (out_size×out_size)。"""
        W, H = img.size
        cx, cy, w, h = bbox_xywh
        # 轉換為像素座標並確保是標量值
        px = float(cx) * W
        py = float(cy) * H
        pw = float(w) * W
        ph = float(h) * H
        x1 = max(0, px - pw / 2)
        y1 = max(0, py - ph / 2)
        x2 = min(W, px + pw / 2)
        y2 = min(H, py + ph / 2)
        crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
        # 填充成正方形
        pad_w = max(crop.height - crop.width, 0)
        pad_h = max(crop.width - crop.height, 0)
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        crop_square = ImageOps.expand(crop, padding, fill=(128, 128, 128))
        crop_resized = crop_square.resize((out_size, out_size), Image.BICUBIC)
        return crop_resized

    # ----------------  index samples ----------------

    def _prepare_index(self) -> None:
        """使用多對一貪婪匹配策略構建索引列表。"""
        print(f"[Dataset] 開始索引... Pred Dir: {self.pred_dir}")
        
        # 獲取所有 pred txt 檔案
        pred_files = list(self.pred_dir.glob("*.txt"))
        print(f"[Dataset] 找到 {len(pred_files)} 個預測標籤檔案。")

        for txt_pred in sorted(pred_files):
            name = txt_pred.stem
            txt_gt = self.gt_dir / f"{name}.txt"
            img_path = self.img_dir / f"{name}.jpg"
            
            if not txt_gt.exists() or not img_path.exists():
                continue
            
            # 載入GT和預測框
            gt_boxes = self._load_yolo_labels(txt_gt)
            pred_boxes = self._load_yolo_labels(txt_pred)
            
            if len(gt_boxes) == 0 or len(pred_boxes) == 0:
                continue
            
            # 使用改進的一對一匹配
            matches = self._hungarian_matching(pred_boxes, gt_boxes)
            
            for pred_idx, gt_idx in matches:
                pred_box = pred_boxes[pred_idx]
                gt_box = gt_boxes[gt_idx]
                
                # 選項A：完全讓生成器自己學習該校正多少
                # 不添加任何人為的零delta樣本，讓模型從數據中自然學習
                delta = self._bbox2delta(gt_box, pred_box)
                self.samples.append((img_path, 0, pred_box, delta, gt_box))
        
        print(f"[Dataset] 索引完成，共找到 {len(self.samples)} 個可用樣本。")
        
        # 添加數據集質量分析
        self._analyze_dataset_quality()

    def _load_yolo_labels(self, txt_path: Path) -> torch.Tensor:
        """從 YOLO txt 檔案載入邊界框 [cx, cy, w, h]。"""
        if not txt_path.exists() or txt_path.stat().st_size == 0:
            return torch.empty((0, 4))
        
        boxes = []
        try:
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5: # class cx cy w h
                        boxes.append([float(x) for x in parts[1:5]])
        except Exception as e:
            print(f"讀取標籤檔案 {txt_path} 時出錯: {e}")
            return torch.empty((0, 4))
            
        return torch.tensor(boxes) if boxes else torch.empty((0, 4))

    # 移除冗餘函數，直接使用_load_yolo_labels
    
    def _hungarian_matching(self, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> List[Tuple[int, int]]:
        """
        多對一貪婪匹配：每個pred選擇IoU最大的gt，允許多個pred匹配同一個gt
        優點：提升數據多樣性，讓生成器看到更多樣的校正案例
        """
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return []

        # 使用向量化計算提高效率
        matches = []
        for i, pb in enumerate(pred_boxes):
            best_iou = 0.0
            best_gt_idx = -1
            
            for j, gb in enumerate(gt_boxes):
                iou = self._bbox_iou(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= self.iou_thr:
                matches.append((i, best_gt_idx))

        return matches

    def _analyze_dataset_quality(self) -> None:
        """分析數據集質量，提供IoU分佈、delta分佈等統計信息"""
        if len(self.samples) == 0:
            return
        
        print(f"\n📊 [Dataset] 數據集質量分析:")
        print(f"=" * 60)
        
        # 計算IoU分佈
        ious = []
        delta_magnitudes = []
        delta_components = {'dx': [], 'dy': [], 'dw': [], 'dh': []}
        
        for img_path, _, pred_box, delta_true, gt_box in self.samples:
            # 計算原始IoU
            iou = self._bbox_iou(pred_box, gt_box)
            ious.append(iou)
            
            # 計算delta幅度
            delta_mag = torch.norm(delta_true).item()
            delta_magnitudes.append(delta_mag)
            
            # 記錄delta組件
            delta_components['dx'].append(abs(delta_true[0].item()))
            delta_components['dy'].append(abs(delta_true[1].item()))
            delta_components['dw'].append(abs(delta_true[2].item()))
            delta_components['dh'].append(abs(delta_true[3].item()))
        
        # IoU分佈統計
        import numpy as np
        ious = np.array(ious)
        
        print(f"🎯 IoU分佈統計:")
        print(f"   總樣本數: {len(ious)}")
        print(f"   平均IoU: {ious.mean():.4f}")
        print(f"   IoU標準差: {ious.std():.4f}")
        print(f"   IoU範圍: [{ious.min():.4f}, {ious.max():.4f}]")
        
        # IoU分級統計
        iou_ranges = {
            '極低 (<0.3)': (ious < 0.3).sum(),
            '低 (0.3-0.5)': ((ious >= 0.3) & (ious < 0.5)).sum(),
            '中 (0.5-0.7)': ((ious >= 0.5) & (ious < 0.7)).sum(),
            '高 (0.7-0.9)': ((ious >= 0.7) & (ious < 0.9)).sum(),
            '極高 (≥0.9)': (ious >= 0.9).sum()
        }
        
        print(f"   IoU分級分佈:")
        for range_name, count in iou_ranges.items():
            percentage = count / len(ious) * 100
            print(f"     {range_name}: {count} ({percentage:.1f}%)")
        
        # Delta統計
        delta_mags = np.array(delta_magnitudes)
        print(f"\n🔧 Delta統計:")
        print(f"   平均Delta幅度: {delta_mags.mean():.4f}")
        print(f"   Delta標準差: {delta_mags.std():.4f}")
        print(f"   Delta範圍: [{delta_mags.min():.4f}, {delta_mags.max():.4f}]")
        
        # Delta組件統計
        print(f"   Delta組件統計:")
        for component, values in delta_components.items():
            values = np.array(values)
            print(f"     {component}: 平均={values.mean():.4f}, 標準差={values.std():.4f}")
        
        # 數據質量警告
        print(f"\n⚠️  數據質量檢查:")
        
        # 檢查高IoU樣本比例
        high_iou_ratio = (ious >= 0.8).sum() / len(ious)
        if high_iou_ratio > 0.5:
            print(f"   警告: 高IoU樣本比例過高 ({high_iou_ratio:.1%})，生成器可能難以學會選擇性校正")
        
        # 檢查極低IoU樣本比例
        very_low_iou_ratio = (ious < 0.3).sum() / len(ious)
        if very_low_iou_ratio > 0.3:
            print(f"   警告: 極低IoU樣本比例過高 ({very_low_iou_ratio:.1%})，可能包含無法修復的錯誤")
        
        # 檢查delta幅度分佈
        large_delta_ratio = (delta_mags > 2.0).sum() / len(delta_mags)
        if large_delta_ratio > 0.2:
            print(f"   警告: 大幅度校正樣本比例過高 ({large_delta_ratio:.1%})，可能導致訓練不穩定")
        
        # 檢查數據平衡性
        if iou_ranges['極高 (≥0.9)'] > iou_ranges['極低 (<0.3)'] * 3:
            print(f"   建議: 考慮平衡不同IoU範圍的樣本分佈")
        
        print(f"=" * 60)

    # 移除未使用的_apply_delta_to_bbox函數（losses.py中有相同功能）

    # -----------------  Dataset API ----------------

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, idx: int):  # type: ignore[override]
        img_path, _, pred_box, delta_true, gt_box = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        # 使用存儲的 delta 確保數據一致性
        # 裁切 patch
        gt_patch = self._letterbox(img, gt_box, self.img_size)
        pred_patch = self._letterbox(img, pred_box, self.img_size)

        # 轉換為張量
        pred_patch = self.transform(pred_patch)
        gt_patch = self.transform(gt_patch)
        
        # 選項A：完全讓生成器自己學習該校正多少
        # 使用預先計算好的delta_true，不做任何人為修改
        return pred_patch, gt_patch, delta_true, pred_box, gt_box, str(img_path)
