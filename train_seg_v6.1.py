"""
Segmentation Training Script - v6 Final (FIXED)
DINOv2 backbone + lightweight segmentation head
"""

import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from copy import deepcopy

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR   = r"D:\Hackathon\Startathin\Offroad_Segmentation_Training_Dataset\train"
VAL_DIR    = r"D:\Hackathon\Startathin\Offroad_Segmentation_Training_Dataset\val"
OUTPUT_DIR = r"D:\Hackathon\Startathin\Offroad_Segmentation_Scripts\train_stats"
MODEL_PATH = r"D:\Hackathon\Startathin\Offroad_Segmentation_Scripts\segmentation_head.pth"
CACHE_DIR  = r"D:\Hackathon\Startathin\Offroad_Segmentation_Scripts\feature_cache_v6"

BATCH_SIZE   = 32
N_EPOCHS     = 50
LR           = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 2
PATIENCE     = 15
GRAD_CLIP    = 1.0

CE_W   = 0.7
DICE_W = 0.3

H, W      = 378, 476
N_CLASSES = 10

CLASS_NAMES = [
    "Background", "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Logs", "Rocks", "Landscape", "Sky"
]

VALUE_MAP = {
    0:0, 100:1, 200:2, 300:3, 500:4,
    550:5, 700:6, 800:7, 7100:8, 10000:9
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR,  exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE & AMP
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_amp_dtype():
    if device.type == 'cuda':
        cap = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if cap[0] >= 8 else torch.float16
    return torch.float32

amp_dtype = get_amp_dtype()
use_amp   = amp_dtype in (torch.float16, torch.bfloat16)  # FIXED

# ─────────────────────────────────────────────────────────────────────────────
# MASK CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

LOOKUP = np.zeros(10001, dtype=np.uint8)
for k, v in VALUE_MAP.items():
    LOOKUP[k] = v

def convert_mask(mask):
    arr = np.asarray(mask, dtype=np.int32)
    return Image.fromarray(LOOKUP[np.clip(arr, 0, 10000)])

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class SegDataset(Dataset):
    def __init__(self, root):
        self.root  = root
        self.files = sorted(os.listdir(os.path.join(root, "Color_Images")))
        self.tf    = T.Compose([
            T.Resize((H, W)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        name = self.files[i]
        img  = Image.open(os.path.join(self.root, "Color_Images", name)).convert("RGB")
        mask = Image.open(os.path.join(self.root, "Segmentation",  name))
        mask = convert_mask(mask).resize((W, H), Image.NEAREST)
        return self.tf(img), torch.from_numpy(np.array(mask)).long()

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE CACHE
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(backbone, dataset, tag):
    fpath = os.path.join(CACHE_DIR, f"{tag}_feat.pt")
    mpath = os.path.join(CACHE_DIR, f"{tag}_mask.pt")

    if os.path.exists(fpath) and os.path.exists(mpath):
        print(f"  [{tag}] Loading from cache...")
        return torch.load(fpath, map_location='cpu'), \
               torch.load(mpath, map_location='cpu')

    print(f"  [{tag}] Extracting features (one-time)...")
    loader = DataLoader(dataset, batch_size=8, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    feats, masks = [], []
    backbone.eval()
    with torch.no_grad():
        for x, m in tqdm(loader, desc=f"  {tag}"):
            f = backbone.forward_features(x.to(device))["x_norm_patchtokens"]
            feats.append(f.cpu().to(torch.float16))
            masks.append(m)

    feats = torch.cat(feats)
    masks = torch.cat(masks)
    torch.save(feats, fpath)
    torch.save(masks, mpath)
    print(f"  [{tag}] Saved {feats.shape}")
    return feats, masks

# ─────────────────────────────────────────────────────────────────────────────
# CLASS WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(masks):
    counts = torch.zeros(N_CLASSES)
    flat   = masks.view(-1)
    for c in range(N_CLASSES):
        counts[c] = (flat == c).sum().float()
    counts = counts.clamp(min=1)
    w      = 1.0 / (counts / counts.sum())
    w      = w / w.mean()
    w      = w.clamp(max=10.0)
    w      = w / w.mean()
    print("  Weights:", [f"{x:.2f}" for x in w.tolist()])
    return w

# ─────────────────────────────────────────────────────────────────────────────
# GPU AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def gpu_aug_tokens(feats, masks, tokenH, tokenW):
    B, N, C = feats.shape
    f = feats.view(B, tokenH, tokenW, C)
    hflip = torch.rand(B, device=feats.device) < 0.5
    if hflip.any():
        f[hflip]     = torch.flip(f[hflip],     [2])
        masks[hflip] = torch.flip(masks[hflip], [2])
    vflip = torch.rand(B, device=feats.device) < 0.2
    if vflip.any():
        f[vflip]     = torch.flip(f[vflip],     [1])
        masks[vflip] = torch.flip(masks[vflip], [1])
    return f.view(B, N, C), masks

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class ConvBnAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, p=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, padding=p, groups=groups, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU()
        )
    def forward(self, x): return self.block(x)


class SegHead(nn.Module):
    def __init__(self, dim, n_cls, tokenH, tokenW):
        super().__init__()
        self.tH, self.tW = tokenH, tokenW
        self.proj       = ConvBnAct(dim, 256, k=1, p=0)
        self.dw1        = ConvBnAct(256, 256, k=3, p=1, groups=256)
        self.pw1        = ConvBnAct(256, 256, k=1, p=0)
        self.dw2        = ConvBnAct(256, 256, k=3, p=1, groups=256)
        self.pw2        = ConvBnAct(256, 128, k=1, p=0)
        self.shortcut   = nn.Conv2d(256, 128, 1, bias=False)
        self.dropout    = nn.Dropout2d(p=0.1)
        self.classifier = nn.Conv2d(128, n_cls, 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        B, N, C = x.shape
        x   = x.reshape(B, self.tH, self.tW, C).permute(0, 3, 1, 2).contiguous()
        x   = self.proj(x)
        res = self.shortcut(x)
        x   = self.pw1(self.dw1(x))
        x   = self.pw2(self.dw2(x))
        return self.classifier(self.dropout(x + res))

# ─────────────────────────────────────────────────────────────────────────────
# LOSSES
# ─────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def forward(self, logits, masks):
        p = torch.softmax(logits, 1)
        t = F.one_hot(masks, p.shape[1]).permute(0, 3, 1, 2).float()
        i = (p * t).sum((2, 3))
        u = p.sum((2, 3)) + t.sum((2, 3))
        return 1 - (2 * i / (u + 1e-6)).mean()


class CombinedLoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss()

    def forward(self, logits, masks):
        return CE_W * self.ce(logits, masks) + DICE_W * self.dice(logits, masks)

# ─────────────────────────────────────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model, decay=0.999):
        self.shadow = deepcopy(model).eval()
        self.decay  = decay
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.copy_(s * self.decay + m.float() * (1 - self.decay))

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(logits, targets):
    preds   = logits.argmax(1).view(-1).cpu().long()
    targets = targets.view(-1).cpu().long()
    ious, dices = [], []
    for c in range(N_CLASSES):
        p     = preds   == c
        t     = targets == c
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        ious.append((inter / union).item() if union > 0 else float('nan'))
        denom = p.sum() + t.sum()
        dices.append((2 * inter / (denom + 1e-6)).item())
    acc = (preds == targets).float().mean().item()
    return np.nanmean(ious), np.nanmean(dices), acc, ious

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS & REPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_plots(history):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (title, tk, vk) in zip(axes.flat, [
        ('Loss',     'tl', 'vl'),
        ('mIoU',     'ti', 'vi'),
        ('Dice',     'td', 'vd'),
        ('Accuracy', 'ta', 'va'),
    ]):
        ax.plot(history[tk], label='Train', linewidth=2)
        ax.plot(history[vk], label='Val',   linewidth=2)
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.4)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'metrics.png')
    plt.savefig(out, dpi=150); plt.close()
    print(f"Plots: {out}")


def save_per_class_plot(history):
    plt.figure(figsize=(14, 6))
    for c, name in enumerate(CLASS_NAMES):
        vals = [e[c] for e in history if not np.isnan(e[c])]
        xs   = [i for i, e in enumerate(history) if not np.isnan(e[c])]
        if vals: plt.plot(xs, vals, label=name, linewidth=1.5)
    plt.title('Per-Class Val IoU'); plt.xlabel('Epoch')
    plt.legend(fontsize=8, loc='lower right'); plt.grid(alpha=0.4)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'per_class_iou.png')
    plt.savefig(out, dpi=150); plt.close()
    print(f"Per-class plot: {out}")


def save_report(history, per_cls, elapsed):
    best_ep = int(np.argmax(history['vi'])) + 1
    path    = os.path.join(OUTPUT_DIR, 'report.txt')
    with open(path, 'w') as f:
        f.write("TRAINING REPORT — v6 Final\n" + "="*60 + "\n\n")
        f.write(f"Time           : {elapsed/60:.1f} min\n")
        f.write(f"Best Val mIoU  : {max(history['vi']):.4f}  (Epoch {best_ep})\n")
        f.write(f"Best Val Dice  : {max(history['vd']):.4f}\n")
        f.write(f"Best Val Acc   : {max(history['va']):.4f}\n\n")
        f.write("Per-Class IoU at Best Epoch:\n")
        for name, iou in zip(CLASS_NAMES, per_cls[best_ep-1]):
            f.write(f"  {name:<18}: {'N/A' if np.isnan(iou) else f'{iou:.4f}'}\n")
        f.write("\n")
        f.write(f"{'Ep':<5}{'TrLoss':<10}{'VLoss':<10}{'TrIoU':<10}{'VIoU':<10}{'VDice':<10}{'VAcc':<10}\n")
        f.write("-"*65 + "\n")
        for i in range(len(history['tl'])):
            f.write(f"{i+1:<5}{history['tl'][i]:<10.4f}{history['vl'][i]:<10.4f}"
                    f"{history['ti'][i]:<10.4f}{history['vi'][i]:<10.4f}"
                    f"{history['vd'][i]:<10.4f}{history['va'][i]:<10.4f}\n")
    print(f"Report: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Device    : {device}")
    print(f"GPU       : {torch.cuda.get_device_name(device)}")
    print(f"AMP dtype : {amp_dtype}")

    # Load backbone
    print("\nLoading DINOv2...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)

    # Extract / load cache
    print("\nPreparing features...")
    tr_f, tr_m = extract_features(backbone, SegDataset(DATA_DIR), "train")
    va_f, va_m = extract_features(backbone, SegDataset(VAL_DIR),  "val")

    del backbone; torch.cuda.empty_cache()
    print(f"\nTrain: {len(tr_f)}  Val: {len(va_f)} samples")

    # Class weights
    print("\nComputing class weights...")
    cw = compute_class_weights(tr_m).to(device)

    # Loaders
    tokenH, tokenW = H // 14, W // 14
    tr_dl = DataLoader(TensorDataset(tr_f.float(), tr_m),
                       BATCH_SIZE, shuffle=True,
                       num_workers=0, pin_memory=True, drop_last=True)
    va_dl = DataLoader(TensorDataset(va_f.float(), va_m),
                       BATCH_SIZE, shuffle=False,
                       num_workers=0, pin_memory=True)

    # Model
    model = SegHead(tr_f.shape[-1], N_CLASSES, tokenH, tokenW).to(device)
    ema   = EMA(model, decay=0.999)

    criterion = CombinedLoss(cw)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-5)
    scaler    = GradScaler(enabled=(amp_dtype == torch.float16))  # FIXED

    history  = {k: [] for k in ['tl','vl','ti','vi','td','vd','ta','va']}
    per_cls  = []
    best_iou = 0.0
    patience = 0
    t0       = time.time()

    print(f"\nStarting {N_EPOCHS}-epoch training...\n" + "="*70)

    for epoch in range(N_EPOCHS):
        # ── Train ──
        model.train()
        tr_loss, tr_iou, tr_dice, tr_acc = [], [], [], []

        for feats, masks in tqdm(tr_dl, desc=f"Ep {epoch+1:02d}/{N_EPOCHS} [Train]", leave=False):
            feats = feats.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            feats, masks = gpu_aug_tokens(feats, masks, tokenH, tokenW)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                logits  = model(feats)
                outputs = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
                loss    = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update()
            ema.update(model)

            miou, mdice, acc, _ = compute_metrics(outputs.detach().float(), masks)
            tr_loss.append(loss.item())
            tr_iou.append(miou); tr_dice.append(mdice); tr_acc.append(acc)

        scheduler.step()

        # ── Validate (EMA) ──
        ema.shadow.eval()
        vl_buf, vi_buf, vd_buf, va_buf, vc_buf = [], [], [], [], []

        with torch.no_grad():
            for feats, masks in tqdm(va_dl, desc=f"Ep {epoch+1:02d}/{N_EPOCHS} [Val]  ", leave=False):
                feats = feats.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                with autocast(enabled=use_amp):
                    logits  = ema.shadow(feats)
                    outputs = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
                    loss    = criterion(outputs.float(), masks)
                miou, mdice, acc, cls_iou = compute_metrics(outputs.float(), masks)
                vl_buf.append(loss.item()); vi_buf.append(miou)
                vd_buf.append(mdice); va_buf.append(acc); vc_buf.append(cls_iou)

        tl = np.mean(tr_loss);   vl = np.mean(vl_buf)
        ti = np.nanmean(tr_iou); vi = np.nanmean(vi_buf)
        td = np.nanmean(tr_dice); vd = np.nanmean(vd_buf)
        ta = np.mean(tr_acc);    va = np.mean(va_buf)
        mean_cls = np.nanmean(vc_buf, axis=0).tolist()
        per_cls.append(mean_cls)

        for k, v in zip(['tl','vl','ti','vi','td','vd','ta','va'],
                        [tl,  vl,  ti,  vi,  td,  vd,  ta,  va]):
            history[k].append(v)

        elapsed = time.time() - t0
        eta     = elapsed / (epoch + 1) * (N_EPOCHS - epoch - 1)
        print(f"Ep [{epoch+1:02d}/{N_EPOCHS}] "
              f"Loss {tl:.4f}/{vl:.4f} | "
              f"mIoU {ti:.4f}/{vi:.4f} | "
              f"Acc {ta:.4f}/{va:.4f} | "
              f"LR {scheduler.get_last_lr()[0]:.5f} | "
              f"ETA {eta/60:.1f}m")

        if (epoch + 1) % 5 == 0:
            print("  Per-class Val IoU:")
            for name, iou in zip(CLASS_NAMES, mean_cls):
                bar = "█" * int(iou * 20) if not np.isnan(iou) else ""
                print(f"    {name:<18} {iou:.4f}  {bar}" if not np.isnan(iou)
                      else f"    {name:<18} N/A")

        if vi > best_iou:
            best_iou = vi; patience = 0
            torch.save({
                'epoch':      epoch + 1,
                'state_dict': ema.shadow.state_dict(),
                'val_iou':    best_iou,
            }, MODEL_PATH)
            print(f"  ✅  Best saved (Val mIoU = {best_iou:.4f})")
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"\n⏹  Early stopping.")
                break

    total = time.time() - t0
    save_plots(history)
    save_per_class_plot(per_cls)
    save_report(history, per_cls, total)

    print(f"\n{'='*60}")
    print(f"Done in {total/60:.1f} min | Best Val mIoU: {best_iou:.4f}")
    print(f"Model: {MODEL_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()