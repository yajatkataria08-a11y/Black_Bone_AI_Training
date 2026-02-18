# Offroad Semantic Segmentation — DINOv2 + SegHead v6 Final

**Startathon Hackathon 2026 | E-Cell VIT Bhopal**

---

## Overview

This project trains a lightweight segmentation head on top of a frozen **DINOv2 (ViT-S/14)** backbone to perform semantic segmentation on a synthetic desert offroad dataset. The model classifies each pixel into one of 10 classes.

---

## Classes

| Index | Class Name     |
|-------|---------------|
| 0     | Background    |
| 1     | Trees         |
| 2     | Lush Bushes   |
| 3     | Dry Grass     |
| 4     | Dry Bushes    |
| 5     | Ground Clutter|
| 6     | Logs          |
| 7     | Rocks         |
| 8     | Landscape     |
| 9     | Sky           |

---

## Dataset Structure

```
Offroad_Segmentation_Training_Dataset/
  train/
    Color_Images/     ← RGB input images (.png)
    Segmentation/     ← Grayscale masks with raw pixel values
  val/
    Color_Images/
    Segmentation/
```

**Mask value mapping:**
```
0 → Background | 100 → Trees | 200 → Lush Bushes | 300 → Dry Grass
500 → Dry Bushes | 550 → Ground Clutter | 700 → Logs | 800 → Rocks
7100 → Landscape | 10000 → Sky
```

---

## Setup

```bash
conda create -n EDU python=3.10 -y
conda activate EDU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install numpy==1.26.4 pillow matplotlib tqdm
```

---

## How to Reproduce Results

### Step 1 — Train
```bash
conda activate EDU
cd D:\Hackathon\Startathin\Offroad_Segmentation_Scripts
python train_mask_final.py
```

On first run, DINOv2 features are extracted and cached to `feature_cache_v6/`.
Subsequent runs load from cache instantly.

### Step 2 — Outputs
After training, the following files are generated:

| File | Description |
|------|-------------|
| `segmentation_head.pth` | Best model weights (EMA, saved at best Val mIoU) |
| `train_stats/metrics.png` | Loss, mIoU, Dice, Accuracy curves |
| `train_stats/per_class_iou.png` | Per-class IoU over epochs |
| `train_stats/report.txt` | Full per-epoch metrics + best epoch summary |

---

## Model Architecture

```
Input: DINOv2 ViT-S/14 patch tokens  [B, 27×34, 384]
         ↓
SegHead Decoder:
  Conv1x1 (384 → 256) + BN + GELU
  DepthwiseConv3x3 (256) + GELU
  PointwiseConv1x1 (256) + GELU
  DepthwiseConv3x3 (256) + GELU
  PointwiseConv1x1 (256 → 128) + GELU
  Residual shortcut (256 → 128)
  Dropout2D (0.1)
  Conv1x1 (128 → 10)
         ↓
Bilinear upsample → [B, 10, 378, 476]
```

---

## Training Details

| Hyperparameter | Value |
|---------------|-------|
| Backbone | DINOv2 ViT-S/14 (frozen) |
| Input size | 378 × 476 |
| Batch size | 32 |
| Epochs | 50 (early stopping, patience=15) |
| Optimizer | AdamW (lr=1e-3, wd=1e-4) |
| Scheduler | CosineAnnealingLR |
| Loss | 0.7 × CrossEntropy + 0.3 × Dice |
| Class weights | Inverse-frequency (capped 10×) |
| Augmentation | GPU token hflip (p=0.5) + vflip (p=0.2) |
| EMA decay | 0.999 |
| Mixed precision | bfloat16 (RTX 5050) |
| Gradient clipping | 1.0 |

---

## Evaluation Metric

**Mean IoU (mIoU)** across all 10 classes — primary metric (80 pts).

```
mIoU = mean over classes of (Intersection / Union)
```

---

## Important Notes

- ⚠️ Test images were **NOT** used during training
- Model weights saved are **EMA weights** for better generalization
- Features are cached as fp16 to save disk space
- If VRAM is insufficient, reduce `BATCH_SIZE` to 16

