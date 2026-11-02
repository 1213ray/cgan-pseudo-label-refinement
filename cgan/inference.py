r"""使用訓練好的Swin-UNet生成器批量校準偽標籤。
Requires: pip install timm (for Swin Transformer backbone)

批量處理模式（在 repo 根目錄）：
```bash
python cgan/inference.py \
    --weights runs/cgan/G_best.pth \
    --source  datasets/two_stage/stage2/high/images \
    --labels  datasets/two_stage/stage2/high/labels \
    --output  datasets/calibrated --copy-images
```
加上 --copy-images 時輸出 YOLO 資料集結構（OUTPUT/images、OUTPUT/labels），可直接用於 Stage 4 再訓練；
否則只把校正後的標籤寫到 OUTPUT/。
"""
from __future__ import annotations
import argparse
import math
import shutil
from pathlib import Path
from typing import List

import torch
from torchvision import transforms
from PIL import Image
import yaml

# 直接導入同資料夾的模組
from models import GeneratorSwinUNet
from dataset import CalibratorDataset
from losses import apply_delta_to_bbox


# ------------------  Helper Functions (修正後) ------------------

def load_yolo_txt(txt_path: Path) -> List[List[float]]:
    """載入 YOLO 格式的 txt 檔案。"""
    if not txt_path.exists():
        return []
    with open(txt_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
        return [[float(x) for x in line.split()] for line in lines]


def save_yolo_txt(txt_path: Path, rows: List[List[float]]):
    """儲存 YOLO 格式的 txt 檔案。"""
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w") as f:
        for r in rows:
            # 安全的類別轉換，確保是有效的整數
            try:
                class_id = int(float(r[0]))  # 先轉float再轉int，處理字符串情況
                r[0] = class_id
            except (ValueError, IndexError):
                print(f"Warning: Invalid class ID in row: {r}, skipping...")
                continue
            f.write(" ".join(map(str, r)) + "\n")


def crop_and_letterbox_patch(img: Image.Image, bbox: List[float], img_size: int) -> Image.Image | None:
    """
    從圖像中裁剪邊界框區域並應用letterbox，與訓練時流程完全一致。
    關鍵修正：此函式現在直接調用 CalibratorDataset._letterbox。
    """
    # 檢查輸入有效性
    if len(bbox) < 4:
        print(f"Warning: Invalid bbox length {len(bbox)}, expected at least 4")
        return None
        
    # 檢查寬高是否為正數且合理
    if bbox[2] <= 0 or bbox[3] <= 0:
        print(f"Warning: Invalid bbox dimensions w={bbox[2]}, h={bbox[3]}")
        return None
        
    # 檢查座標是否在合理範圍內 (0-1 for normalized coordinates)
    if not (0 <= bbox[0] <= 1 and 0 <= bbox[1] <= 1 and 0 < bbox[2] <= 1 and 0 < bbox[3] <= 1):
        print(f"Warning: Bbox coordinates out of normalized range: {bbox[:4]}")
        return None

    try:
        # 使用 CalibratorDataset 中的靜態方法來確保預處理完全一致
        bbox_tensor = torch.tensor(bbox[:4], dtype=torch.float32)
        patch = CalibratorDataset._letterbox(img, bbox_tensor, img_size)
        return patch

    except Exception as e:
        print(f"Warning: Failed to crop patch with bbox {bbox} due to {e}")
        return None


def apply_delta_to_bbox_list(bbox: List[float], delta: torch.Tensor, device) -> List[float]:
    """
    將預測的修正量 delta 應用於原始 bbox。
    關鍵修正：此函式現在直接調用 losses.py 中的 apply_delta_to_bbox。
    """
    bbox_tensor = torch.tensor([bbox[:4]], dtype=torch.float32, device=device)
    delta_tensor = delta.unsqueeze(0) if delta.dim() == 1 else delta

    # 確保delta_tensor在正確的設備上
    delta_tensor = delta_tensor.to(device)

    # 調用與訓練時完全一致的函式，並設定為非訓練模式
    calibrated_tensor = apply_delta_to_bbox(bbox_tensor, delta_tensor, training=False)
    calibrated_bbox = calibrated_tensor[0].cpu().tolist()

    return calibrated_bbox


# ------------------  Core Processing Logic (修正後) ------------------

def process_batch(netG, transform, source_dir, labels_dir, output_dir, img_size, device):
    """批量處理整個資料夾。"""
    source_path, labels_path, output_path = Path(source_dir), Path(labels_dir), Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [p for p in source_path.iterdir() if p.suffix.lower() in image_extensions]

    if not image_files:
        print(f"錯誤：在 '{source_dir}' 中找不到任何支援的圖像檔案。")
        return

    processed_count, total_boxes, skipped_boxes = 0, 0, 0

    print(f"開始批量處理 {len(image_files)} 張圖像...")

    for img_file in image_files:
        txt_file = labels_path / f"{img_file.stem}.txt"
        if not txt_file.exists():
            continue

        try:
            img = Image.open(img_file).convert("RGB")
            preds = load_yolo_txt(txt_file)
            if not preds:
                continue

            calibrated_rows = []
            for pred_row in preds:
                cls, bbox = int(pred_row[0]), pred_row[1:5]

                # 關鍵修正：使用統一的裁剪和letterbox函式
                patch_pil = crop_and_letterbox_patch(img, bbox, img_size)
                if patch_pil is None:
                    skipped_boxes += 1
                    continue

                patch_tensor = transform(patch_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    delta = netG(patch_tensor)[0]

                # 關鍵修正：使用統一的delta應用函式
                calibrated_bbox = apply_delta_to_bbox_list(bbox, delta, device)
                calibrated_rows.append([cls] + calibrated_bbox + pred_row[5:])

            if calibrated_rows:
                save_yolo_txt(output_path / f"{img_file.stem}.txt", calibrated_rows)
                processed_count += 1
                total_boxes += len(calibrated_rows)

        except Exception as e:
            print(f"錯誤：處理 {img_file.name} 時發生錯誤: {e}")
            continue

    print(f"\n✅ 批量處理完成！")
    if skipped_boxes > 0:
        print(f"   -> 警告：因座標無效或裁切失敗，共跳過了 {skipped_boxes} 個邊界框。")
    print(f"   -> 成功處理圖像數量: {processed_count}")
    print(f"   -> 成功校準邊界框總數: {total_boxes}")
    print(f"   -> 輸出目錄: {output_dir}")


def load_model(weights_path, device):
    """從檔案載入生成器模型。"""
    try:
        checkpoint = torch.load(weights_path, map_location=device)
        # 從 checkpoint 中讀取 config，如果沒有則用預設值
        config = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}
        pretrained = config.get('pretrained', True)

        # 權重會由 checkpoint 覆蓋，因此不必下載 ImageNet 預訓練權重
        netG = GeneratorSwinUNet(pretrained=False).to(device)

        if isinstance(checkpoint, dict) and 'generator' in checkpoint:
            netG.load_state_dict(checkpoint['generator'])
        else:
            netG.load_state_dict(checkpoint)

        netG.eval()
        print(f"成功載入Swin-UNet生成器模型 (自主學習模式, pretrained={pretrained})。")
        return netG

    except Exception as e:
        print(f"錯誤：載入模型權重時失敗: {e}")
        raise


# ------------------  Entry Point  ------------------

def main():
    parser = argparse.ArgumentParser(description="使用 CGAN 批量校準 YOLO 偽標籤。")
    parser.add_argument("--weights", type=str, required=True, help="訓練好的生成器權重路徑。")
    parser.add_argument("--source", type=str, required=True, help="輸入圖像資料夾的路徑。")
    parser.add_argument("--labels", type=str, required=True, help="標籤資料夾的路徑。")
    parser.add_argument("--output", type=str, required=True, help="輸出資料夾的路徑。")
    parser.add_argument("--img-size", type=int, default=None, help="裁切 patch 的圖像大小，預設從config.yaml讀取。")
    parser.add_argument("--device", type=str, default="auto", help="計算設備 (cuda/cpu/auto)。")
    parser.add_argument("--copy-images", action="store_true",
                        help="輸出 YOLO 資料集結構：標籤寫到 OUTPUT/labels/、影像複製到 OUTPUT/images/。")

    args = parser.parse_args()

    # 載入配置
    config_path = Path(__file__).resolve().parent / "config.yaml"
    img_size = args.img_size
    if not img_size:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            img_size = config['img_size']
            print(f"從 {config_path} 載入 img_size: {img_size}")
        except (FileNotFoundError, KeyError):
            parser.error(f"無法從 {config_path} 找到 'img_size' 且未透過 --img-size 指定。請提供一個值。")

    # 設置設備
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"使用設備: {device}")

    # 驗證輸入路徑
    source_path = Path(args.source)
    labels_path = Path(args.labels)

    if not source_path.is_dir():
        parser.error(f"源圖像資料夾不存在: {args.source}")
    if not labels_path.is_dir():
        parser.error(f"標籤資料夾不存在: {args.labels}")

    # 載入模型和轉換
    netG = load_model(args.weights, device)

    # 使用與訓練一致的轉換
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    print("✅ 批量處理模式啟動")
    label_out = Path(args.output) / "labels" if args.copy_images else Path(args.output)
    process_batch(netG, transform, args.source, args.labels, label_out, img_size, device)

    if args.copy_images:
        img_out = Path(args.output) / "images"
        img_out.mkdir(parents=True, exist_ok=True)
        copied = 0
        for txt in label_out.glob("*.txt"):
            for img in source_path.glob(f"{txt.stem}.*"):
                shutil.copy2(img, img_out / img.name)
                copied += 1
        print(f"   -> 已複製 {copied} 張影像到 {img_out}")


if __name__ == "__main__":
    main()
