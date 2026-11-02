"""
組出 cGAN 的訓練資料：同一批 labeled 影像的「K-Fold 預測框」與「真實框」。

輸出 datasets/cgan_train_set/
    images/       labeled 影像（part1）
    labels_gt/    真實標籤（cls cx cy w h）
    labels_pred/  K-Fold 偽標籤（cls cx cy w h conf）
cgan/dataset.py 會在這三個資料夾間配對預測框與真實框。

用法（在 repo 根目錄）：python pipeline/05_build_cgan_dataset.py
"""
import shutil
from pathlib import Path

# ────────────── 參數區 ──────────────
LABELED_DIR = Path("datasets/person/part1")                 # images/、labels/
PRED_DIR    = Path("datasets/kfold/merged_pseudo/labels")   # 04_kfold_pseudo_labels.py 的輸出
OUT_DIR     = Path("datasets/cgan_train_set")
# ───────────────────────────────────


def main():
    pred_files = sorted(PRED_DIR.glob("*.txt"))
    if not pred_files:
        raise FileNotFoundError(f"在 {PRED_DIR} 找不到偽標籤，請先執行 04_kfold_pseudo_labels.py")

    for sub in ("images", "labels_gt", "labels_pred"):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    copied, missing = 0, 0
    for pred_txt in pred_files:
        name = pred_txt.stem
        img = next((p for p in (LABELED_DIR / "images").glob(f"{name}.*")), None)
        gt_txt = LABELED_DIR / "labels" / f"{name}.txt"
        if img is None or not gt_txt.exists():
            missing += 1
            continue
        shutil.copy2(img, OUT_DIR / "images" / img.name)
        shutil.copy2(gt_txt, OUT_DIR / "labels_gt" / gt_txt.name)
        shutil.copy2(pred_txt, OUT_DIR / "labels_pred" / pred_txt.name)
        copied += 1

    print(f"完成：{copied} 張影像 → {OUT_DIR}")
    if missing:
        print(f"略過 {missing} 個找不到影像或真實標籤的偽標籤檔")


if __name__ == "__main__":
    main()
