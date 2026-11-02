<h1 align="center">cGAN Pseudo-Label Refinement</h1>

<p align="center">Teaching a conditional GAN to fix the teacher's boxes in semi-supervised object detection</p>

<p align="center"><b>English</b> | <a href="README.zh-TW.md">繁體中文</a></p>

<p align="center">
  <img src="docs/images/architecture.png" width="860" alt="Generator and discriminator">
</p>

In semi-supervised object detection, a teacher trained on a few labeled images draws boxes (pseudo-labels) on unlabeled images, and the next model learns from them. With little labeled data, these boxes usually find the object but are too loose, too tight or shifted. This project trains a **conditional GAN whose generator looks at the image inside a pseudo-box and predicts how to move and resize it**. The training pairs come from K-fold cross-prediction, so the generator learns from realistic teacher mistakes.

Undergraduate research project, Department of Applied Mathematics, Chung Yuan Christian University, 2025; also entered in a student competition.

## Results at a glance

On a COCO *person* subset with only 1%, 5% or 10% of the images labeled (Δ = with refinement − without):

| Labeled images | Pseudo-label IoU | mAP@50 | mAP@[.5:.95] |
|:---:|:---:|:---:|:---:|
| 1% | +0.043 | +0.79 pp | +0.57 pp |
| 5% | +0.074 | +1.33 pp | +0.96 pp |
| 10% | +0.081 | +1.46 pp | +1.05 pp |

<sub>pp = percentage points. Reported in the project write-up; the datasets, logs and weights behind this table are not stored in the repository.</sub>

<p align="center">
  <img src="docs/images/refinement_example.png" width="520" alt="Pseudo-label, refined box and ground truth">
</p>

## How it works

```mermaid
%%{init: {"flowchart": {"useMaxWidth": true, "rankSpacing": 30, "nodeSpacing": 30}, "themeVariables": {"fontSize": "14px"}}}%%
graph LR
    %% 節點定義
    A(["📁 Labeled"])
    U(["📁 Unlabeled"])
    T["Train YOLO"]
    K["Predict K-fold"]
    S["Predict 2-stage"]
    F["Train cGAN"]
    R["Refine"]
    H[("📝 New labels")]

    %% 連線定義
    A --> T
    A --> K
    T --> S
    U --> S
    K --> F
    F --> R
    S --> R
    R --> H
    H -. "loop" .-> T

    %% 深色主題樣式
    classDef default fill:#252526,stroke:#777,stroke-width:1px,color:#ccc;
    classDef dashedNode fill:#252526,stroke:#777,stroke-width:1px,color:#ccc,stroke-dasharray: 5 5;
```

1. **Teacher.** Train YOLOv8 on the labeled images.
2. **K-fold pseudo-labels (cGAN training data).** Split the labeled images into K folds, train on K−1 and predict the held-out fold. Every labeled image then has boxes from a model that never saw it, paired with its true boxes.
3. **Two-stage pseudo-labels (to be refined).** Predict the unlabeled images with a high confidence threshold, then again with a low one to recover missed objects that do not overlap the confident boxes.
4. **cGAN.** Train the calibrator on the K-fold pairs, then use it to refine the two-stage pseudo-labels.
5. **Iterate.** Train a new teacher on labeled + refined pseudo-labels and repeat.

<p align="center">
  <img src="docs/images/kfold.png" width="860" alt="K-fold cross-prediction">
</p>

### The calibrator

Each K-fold pseudo-box is matched to the ground-truth box with the highest IoU (kept if IoU ≥ 0.25). The generator learns the correction

$$
\Delta = \left(\frac{x_{gt} - x_p}{s},\ \frac{y_{gt} - y_p}{s},\ \log\frac{w_{gt}}{w_p},\ \log\frac{h_{gt}}{h_p}\right), \qquad s = \max\left(\sqrt{w_p h_p},\ 0.05\right)
$$

so shifts are measured relative to the box size, and growing or shrinking a box is symmetric in log space.

| Part | Design |
|---|---|
| Input | The region inside the pseudo-box, letterboxed to 224×224 (*pred patch*) |
| Generator | Swin-T encoder (ImageNet-pretrained) + U-Net decoder with skip connections → pooled → Δ. The refined box is cropped from the original image again (*refined patch*) |
| Discriminator | PatchGAN with spectral normalization and a projection term; sees the pred patch with either the ground-truth patch (real) or the refined patch (fake) |
| Generator loss | EIoU(refined box, ground truth) + λ<sub>adv</sub> · hinge adversarial loss |
| Discriminator loss | Hinge loss + R1 gradient penalty on real pairs (λ = 0.1, every 32 steps) |

EIoU penalizes overlap, center distance and width/height differences separately, so the generator still gets a useful signal when a box is nearly right. Earlier versions used BCE + Smooth-L1 and then WGAN-GP; the current version switched to spectral normalization with R1 and hinge losses because R1 is cheaper and simpler than the WGAN-GP gradient penalty. All settings are in [`cgan/config.yaml`](cgan/config.yaml).

### Training curve

The included log comes from the earlier WGAN-GP version. On held-out K-fold pairs, refinement raised the mean IoU between pseudo-boxes and ground truth from **0.554 to 0.619** over 197 epochs.

<p align="center">
  <img src="results/iou_curve_wgan_gp.png" width="680" alt="Validation IoU during cGAN training">
</p>

For scale, a YOLOv8s teacher trained on 500 labeled images reached mAP@50 0.528 and mAP@50-95 0.209 on a 100-image validation split. With labels that scarce, pseudo-labels are noisy enough for refinement to matter.

## Getting started

```bash
pip install -r requirements.txt
```

Run every command from the repository root. Data and outputs go to `datasets/` and `runs/`, which are git-ignored.

| Stage | Script | Output |
|---|---|---|
| Data | `pipeline/01_coco_to_yolo.py` *(arguments below)* | `datasets/coco_person/` |
| Data | `pipeline/02_split_dataset.py` | `datasets/person/part1–3/` (labeled / val / unlabeled) |
| 1 · Teacher | `pipeline/03_train_teacher.py` | `runs/teacher/person_v1/` |
| 2 · Pseudo-labels | `pipeline/04_kfold_pseudo_labels.py` | `datasets/kfold/` |
| 2 · Pseudo-labels | `pipeline/05_build_cgan_dataset.py` | `datasets/cgan_train_set/` |
| 2 · Pseudo-labels | `pipeline/06_two_stage_pseudo_labels.py` | `datasets/two_stage/` |
| 3 · cGAN | `cgan/train.py` | `runs/cgan/G_best.pth` |
| 3 · cGAN | `cgan/inference.py` *(arguments below)* | `datasets/calibrated/` |
| Evaluation | `pipeline/07_evaluate_pseudo_labels.py` *(arguments below)* | precision / recall / F1, IoU change |
| 4 · Next round | `pipeline/03_train_teacher.py` with `DATA_YAML = "configs/person_round2.yaml"`, `NAME = "person_v2"` | `runs/teacher/person_v2/` |

```bash
python pipeline/01_coco_to_yolo.py --json instances_train2017.json --images-dir train2017 \
    --category person --out-dir datasets/coco_person

python cgan/inference.py --weights runs/cgan/G_best.pth \
    --source datasets/two_stage/stage2/high/images --labels datasets/two_stage/stage2/high/labels \
    --output datasets/calibrated --copy-images

python pipeline/07_evaluate_pseudo_labels.py --gt-dir datasets/person/part3/labels \
    --pre-dir datasets/two_stage/stage2/high/labels --post-dir datasets/calibrated/labels
```

Split sizes (default 1,500 labeled / 300 validation), folds (15), thresholds and epochs are constants at the top of each script. `cgan/train.py` uses batch size 128 by default, which fits in 24 GB of GPU memory; lower `--batch_size` on smaller GPUs. Every stage was checked end to end on a small synthetic dataset.

<details>
<summary><b>Tools</b></summary>

| Script | Purpose |
|---|---|
| `tools/check_delta.py` | Checks that decoding the target Δ really moves pseudo-boxes onto the ground truth |
| `tools/create_paired_patches.py` | Saves pseudo / ground-truth patch pairs side by side for inspection |
| `cgan/diagnose_generator.py` | Shows how the generator behaves for boxes of different starting IoU |
| `tools/plot_training_history.py` | Draws the IoU curve from `training_history.json` |

</details>

## Repository structure

```
├── cgan/          models, losses, dataset, training, inference, config.yaml
├── pipeline/      01–07: data preparation, teacher, pseudo-labels, evaluation
├── tools/         sanity checks and plots
├── configs/       YOLO dataset configs
├── results/       training log and IoU curve
└── docs/images/
```

<details>
<summary><b>References</b></summary>

1. P. Isola et al. Image-to-image translation with conditional adversarial networks (pix2pix). CVPR 2017.
2. Z. Liu et al. Swin Transformer: hierarchical vision transformer using shifted windows. ICCV 2021.
3. T. Miyato, M. Koyama. cGANs with projection discriminator. ICLR 2018.
4. T. Miyato et al. Spectral normalization for generative adversarial networks. ICLR 2018.
5. L. Mescheder, A. Geiger, S. Nowozin. Which training methods for GANs do actually converge? (R1 penalty). ICML 2018.
6. Y.-F. Zhang et al. Focal and efficient IOU loss for accurate bounding box regression (EIoU). Neurocomputing 2022.
7. L. Zhang et al. Mind the gap: polishing pseudo labels for accurate semi-supervised object detection. AAAI 2023.

</details>
