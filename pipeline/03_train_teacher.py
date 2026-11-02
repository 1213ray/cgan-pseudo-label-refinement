"""
Stage 1：以有標註的 part1 監督式訓練 YOLOv8 教師模型。
輸出權重：runs/teacher/person_v1/weights/best.pt（給 06_two_stage_pseudo_labels.py 使用）

Stage 4 再訓練時，把 DATA_YAML 換成 configs/person_round2.yaml（labeled + 校正後偽標籤）、
NAME 改成 person_v2 即可。

用法（在 repo 根目錄）：python pipeline/03_train_teacher.py
"""
DATA_YAML = "configs/person.yaml"
NAME = "person_v1"

if __name__ == '__main__':
    from pathlib import Path
    from ultralytics import YOLO

    model = YOLO('yolov8m.pt')   # COCO 預訓練權重，Ultralytics 會自動下載
    results = model.train(
        data=DATA_YAML,
        imgsz=640,
        epochs=400,
        patience=20,
        batch=64,
        workers=4,
        half=True,  # 啟用混合精度訓練
        project=str(Path('runs/teacher').resolve()),   # 絕對路徑，新版 Ultralytics 才不會改存到 runs/detect/ 底下
        name=NAME,
    )
