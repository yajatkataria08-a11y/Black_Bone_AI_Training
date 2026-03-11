"""
Segmentation Training Script - v7
DINOv2 backbone + lightweight segmentation head
Improvements:
  - CutMix + MixUp + Color Jitter augmentation
  - Focal Loss option
  - Warmup + Cosine LR scheduler
  - Label smoothing
  - Live metrics export for dashboard (train_metrics.json)
  - Compile backbone for faster inference (torch.compile, PyTorch 2.x)
  - AMP auto-detection fix
  - Better logging (train_log.txt + train_status.json)
"""

import os, sys, time, json, math, logging
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — reads from train_config.json if present (written by dashboard)
# ─────────────────────────────────────────────────────────────────────────────

_CFG_FILE = "train_config.json"
_cfg = {}
if Path(_CFG_FILE).exists():
    with open(_CFG_FILE) as f:
        _cfg = json.load(f)
    print(f"[v7] Loaded config from {_CFG_FILE}")

DATA_DIR   = _cfg.get("data_dir",   r"D:\Hackathon\Startathin\Offroad_Segmentation_Training_Dataset\train")
VAL_DIR    = _cfg.get("val_dir",    r"D:\Hackathon\Startathin\Offroad_Segmentation_Training_Dataset\val")
OUTPUT_DIR = _cfg.get("output_dir", r"D:\Hackathon\Startathin\Offroad_Segmentation_Scripts\train_stats")
MODEL_PATH = _cfg.get("model_path", r"D:\Hackathon\Startathin\Offroad_Segmentation_Scripts\segmentation_head.pth")
CACHE_DIR  = os.path.join(os.path.dirname(MODEL_PATH), "feature_cache_v7")

BATCH_SIZE   = _cfg.get("batch_size",  32)
N_EPOCHS     = _cfg.get("n_epochs",    50)
LR           = _cfg.get("lr",          1e-3)
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 2
PATIENCE     = _cfg.get("patience",    15)
GRAD_CLIP    = _cfg.get("grad_clip",   1.0)
LABEL_SMOOTH = _cfg.get("label_smooth", 0.05)
LOSS_TYPE    = _cfg.get("loss_type",   "CE + Dice")   # "CE + Dice" | "Focal + Dice" | "CE only"
BACKBONE     = _cfg.get("backbone",    "dinov2_vits14")

# Augmentation flags
AUG_HFLIP    = _cfg.get("aug_hflip",    True)
AUG_VFLIP    = _cfg.get("aug_vflip",    True)
AUG_COLORJIT = _cfg.get("aug_colorjit", True)
AUG_CUTMIX   = _cfg.get("aug_cutmix",   True)
AUG_MIXUP    = _cfg.get("aug_mixup",    False)
CUTMIX_PROB  = _cfg.get("cutmix_prob",  0.3)
USE_WARMUP   = _cfg.get("use_warmup",   True)
USE_EMA      = _cfg.get("use_ema",      True)

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

# Dashboard output files
METRICS_FILE = "train_metrics.json"
LOG_FILE     = "train_log.txt"
STATUS_FILE  = "train_status.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR,  exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING — to file + console
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("seg")

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
use_amp   = device.type == 'cuda'

# ─────────────────────────────────────────────────────────────────────────────
# STATUS / METRICS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_metrics = {k: [] for k in ['tl','vl','ti','vi','td','vd','ta','va','lr','cls_iou']}

def update_status(epoch, best_iou, eta_min):
    with open(STATUS_FILE, 'w') as f:
        json.dump({"state": "running", "epoch": epoch,
                   "best_iou": round(best_iou, 4), "eta_min": round(eta_min, 1)}, f)

def save_metrics():
    with open(METRICS_FILE, 'w') as f:
        json.dump(_metrics, f)

def finish_status(state="done"):
    with open(STATUS_FILE, 'w') as f:
        data = json.load(open(STATUS_FILE)) if Path(STATUS_FILE).exists() else {}
        data["state"] = state
        json.dump(data, f)

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
    def __init__(self, root, color_jitter=False):
        self.root  = root
        self.files = sorted(os.listdir(os.path.join(root, "Color_Images")))
        tf_list = [T.Resize((H, W)), T.ToTensor(),
                   T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
        if color_jitter:
            tf_list.insert(1, T.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))
        self.tf = T.Compose(tf_list)

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        name = self.files[i]
        img  = Image.open(os.path.join(self.root,"Color_Images",name)).convert("RGB")
        mask = Image.open(os.path.join(self.root,"Segmentation",name))
        mask = convert_mask(mask).resize((W,H), Image.NEAREST)
        return self.tf(img), torch.from_numpy(np.array(mask)).long()

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE CACHE
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(backbone, dataset, tag):
    fpath = os.path.join(CACHE_DIR, f"{tag}_feat.pt")
    mpath = os.path.join(CACHE_DIR, f"{tag}_mask.pt")

    if os.path.exists(fpath) and os.path.exists(mpath):
        log.info(f"[{tag}] Loading from cache...")
        return torch.load(fpath, map_location='cpu'), \
               torch.load(mpath, map_location='cpu')

    log.info(f"[{tag}] Extracting features (one-time)...")
    loader = DataLoader(dataset, batch_size=8, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    feats, masks = [], []
    backbone.eval()
    with torch.no_grad():
        for x, m in tqdm(loader, desc=f"  {tag}"):
            with autocast(enabled=use_amp, dtype=amp_dtype):
                f = backbone.forward_features(x.to(device))["x_norm_patchtokens"]
            feats.append(f.cpu().to(torch.float16))
            masks.append(m)

    feats = torch.cat(feats)
    masks = torch.cat(masks)
    torch.save(feats, fpath)
    torch.save(masks, mpath)
    log.info(f"[{tag}] Saved {feats.shape}")
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
    log.info("Class weights: " + " ".join(f"{x:.2f}" for x in w.tolist()))
    return w

# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION (GPU-side, token-space)
# ─────────────────────────────────────────────────────────────────────────────

def rand_bbox(H, W, lam):
    cut_rat = math.sqrt(1.0 - lam)
    cut_h   = int(H * cut_rat)
    cut_w   = int(W * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2

def gpu_aug_tokens(feats, masks, tokenH, tokenW):
    """Apply spatial augmentations in token space."""
    B, N, C = feats.shape
    f = feats.view(B, tokenH, tokenW, C)
    m = masks   # (B, H, W)

    # Horizontal flip
    if AUG_HFLIP:
        idx = torch.rand(B, device=feats.device) < 0.5
        if idx.any():
            f[idx] = torch.flip(f[idx], [2])
            m[idx] = torch.flip(m[idx], [2])

    # Vertical flip
    if AUG_VFLIP:
        idx = torch.rand(B, device=feats.device) < 0.2
        if idx.any():
            f[idx] = torch.flip(f[idx], [1])
            m[idx] = torch.flip(m[idx], [1])

    # CutMix in token space
    if AUG_CUTMIX and torch.rand(1).item() < CUTMIX_PROB and B > 1:
        lam  = np.random.beta(1.0, 1.0)
        perm = torch.randperm(B, device=feats.device)
        x1, y1, x2, y2 = rand_bbox(tokenH, tokenW, lam)
        f[:, y1:y2, x1:x2, :] = f[perm, y1:y2, x1:x2, :]
        # Scale mask coords to full resolution
        mH, mW = m.shape[1], m.shape[2]
        sx = mW / tokenW; sy = mH / tokenH
        mx1, mx2 = int(x1*sx), int(x2*sx)
        my1, my2 = int(y1*sy), int(y2*sy)
        m[:, my1:my2, mx1:mx2] = m[perm, my1:my2, mx1:mx2]

    # MixUp (feature space only, labels stay hard)
    if AUG_MIXUP and torch.rand(1).item() < 0.2 and B > 1:
        lam  = np.random.beta(0.4, 0.4)
        perm = torch.randperm(B, device=feats.device)
        f    = lam * f + (1 - lam) * f[perm]

    return f.view(B, N, C), m

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
        x   = x.reshape(B, self.tH, self.tW, C).permute(0,3,1,2).contiguous()
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
        t = F.one_hot(masks, p.shape[1]).permute(0,3,1,2).float()
        i = (p * t).sum((2,3))
        u = p.sum((2,3)) + t.sum((2,3))
        return 1 - (2 * i / (u + 1e-6)).mean()


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class CombinedLoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        if "Focal" in LOSS_TYPE:
            self.main = FocalLoss(weight=class_weights, gamma=2.0,
                                  label_smoothing=LABEL_SMOOTH)
        else:
            self.main = nn.CrossEntropyLoss(weight=class_weights,
                                            label_smoothing=LABEL_SMOOTH)
        self.dice  = DiceLoss()
        self.use_dice = "only" not in LOSS_TYPE.lower()

    def forward(self, logits, masks):
        loss = CE_W * self.main(logits, masks)
        if self.use_dice:
            loss = loss + DICE_W * self.dice(logits, masks)
        return loss

# ─────────────────────────────────────────────────────────────────────────────
# LR SCHEDULER — Warmup + Cosine
# ─────────────────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, n_epochs):
    if USE_WARMUP:
        warmup_epochs = min(5, n_epochs // 10)
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, n_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress)) * (1 - 1e-2) + 1e-2
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)

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
        ious.append((inter/union).item() if union > 0 else float('nan'))
        denom = p.sum() + t.sum()
        dices.append((2*inter/(denom+1e-6)).item())
    acc = (preds == targets).float().mean().item()
    return np.nanmean(ious), np.nanmean(dices), acc, ious

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def save_plots(history):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (title, tk, vk) in zip(axes.flat, [
        ('Loss','tl','vl'),('mIoU','ti','vi'),('Dice','td','vd'),('Accuracy','ta','va'),
    ]):
        ax.plot(history[tk], label='Train', linewidth=2)
        ax.plot(history[vk], label='Val',   linewidth=2)
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.4)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'metrics.png')
    plt.savefig(out, dpi=150); plt.close()
    log.info(f"Plots: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info(f"Device    : {device}")
    if device.type == 'cuda':
        log.info(f"GPU       : {torch.cuda.get_device_name(device)}")
    log.info(f"AMP dtype : {amp_dtype}")
    log.info(f"Backbone  : {BACKBONE}")
    log.info(f"Loss type : {LOSS_TYPE}  LabelSmooth={LABEL_SMOOTH}")
    log.info(f"Augs      : flip={AUG_HFLIP}/{AUG_VFLIP}  colorjit={AUG_COLORJIT}  "
             f"cutmix={AUG_CUTMIX}(p={CUTMIX_PROB})  mixup={AUG_MIXUP}")
    log.info(f"Warmup    : {USE_WARMUP}  EMA={USE_EMA}")

    update_status(0, 0.0, 0)

    # Backbone
    log.info("Loading DINOv2...")
    backbone = torch.hub.load("facebookresearch/dinov2", BACKBONE).to(device)

    # Feature extraction
    log.info("Preparing features...")
    tr_f, tr_m = extract_features(backbone, SegDataset(DATA_DIR, color_jitter=AUG_COLORJIT), "train")
    va_f, va_m = extract_features(backbone, SegDataset(VAL_DIR,  color_jitter=False),         "val")

    del backbone; torch.cuda.empty_cache() if device.type=='cuda' else None
    log.info(f"Train: {len(tr_f)}  Val: {len(va_f)}")

    # Class weights
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

    # torch.compile for 15-30% speedup on PyTorch 2.x
    if hasattr(torch, 'compile') and device.type == 'cuda':
        try:
            model = torch.compile(model)
            log.info("torch.compile: enabled ✅")
        except Exception as e:
            log.warning(f"torch.compile skipped: {e}")

    ema = EMA(model, decay=0.999) if USE_EMA else None

    criterion = CombinedLoss(cw)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, N_EPOCHS)
    scaler    = GradScaler(enabled=(amp_dtype == torch.float16))

    history  = {k: [] for k in ['tl','vl','ti','vi','td','vd','ta','va','lr']}
    per_cls  = []
    best_iou = 0.0
    patience = 0
    t0       = time.time()

    log.info(f"Starting {N_EPOCHS}-epoch training...")

    for epoch in range(N_EPOCHS):
        # ── Train ──
        model.train()
        tr_loss, tr_iou, tr_dice, tr_acc = [], [], [], []

        for feats, masks in tqdm(tr_dl, desc=f"Ep {epoch+1:02d}/{N_EPOCHS} [Train]", leave=False):
            feats = feats.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            feats, masks = gpu_aug_tokens(feats, masks, tokenH, tokenW)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp, dtype=amp_dtype):
                logits  = model(feats)
                outputs = F.interpolate(logits, size=(H,W), mode="bilinear", align_corners=False)
                loss    = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update()

            if ema: ema.update(model)

            miou, mdice, acc, _ = compute_metrics(outputs.detach().float(), masks)
            tr_loss.append(loss.item())
            tr_iou.append(miou); tr_dice.append(mdice); tr_acc.append(acc)

        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]

        # ── Validate ──
        eval_model = ema.shadow if (ema and USE_EMA) else model
        eval_model.eval()
        vl_buf, vi_buf, vd_buf, va_buf, vc_buf = [], [], [], [], []

        with torch.no_grad():
            for feats, masks in tqdm(va_dl, desc=f"Ep {epoch+1:02d}/{N_EPOCHS} [Val]  ", leave=False):
                feats = feats.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                with autocast(enabled=use_amp, dtype=amp_dtype):
                    logits  = eval_model(feats)
                    outputs = F.interpolate(logits, size=(H,W), mode="bilinear", align_corners=False)
                    loss    = criterion(outputs.float(), masks)
                miou, mdice, acc, cls_iou = compute_metrics(outputs.float(), masks)
                vl_buf.append(loss.item()); vi_buf.append(miou)
                vd_buf.append(mdice); va_buf.append(acc); vc_buf.append(cls_iou)

        tl = np.mean(tr_loss);    vl = np.mean(vl_buf)
        ti = np.nanmean(tr_iou);  vi = np.nanmean(vi_buf)
        td = np.nanmean(tr_dice); vd = np.nanmean(vd_buf)
        ta = np.mean(tr_acc);     va = np.mean(va_buf)
        mean_cls = np.nanmean(vc_buf, axis=0).tolist()
        per_cls.append(mean_cls)

        for k, v in zip(['tl','vl','ti','vi','td','vd','ta','va','lr'],
                        [tl,  vl,  ti,  vi,  td,  vd,  ta,  va,  cur_lr]):
            history[k].append(v)

        # Export to dashboard
        _metrics.update(history)
        _metrics['cls_iou'] = per_cls
        save_metrics()

        elapsed = time.time() - t0
        eta     = elapsed / (epoch + 1) * (N_EPOCHS - epoch - 1)
        update_status(epoch + 1, best_iou, eta / 60)

        log.info(
            f"Ep [{epoch+1:02d}/{N_EPOCHS}] "
            f"Loss {tl:.4f}/{vl:.4f} | "
            f"mIoU {ti:.4f}/{vi:.4f} | "
            f"Acc {ta:.4f}/{va:.4f} | "
            f"LR {cur_lr:.6f} | "
            f"ETA {eta/60:.1f}m"
        )

        if (epoch + 1) % 5 == 0:
            log.info("Per-class Val IoU:")
            for name, iou in zip(CLASS_NAMES, mean_cls):
                bar = "█" * int(iou * 20) if not np.isnan(iou) else ""
                log.info(f"  {name:<18} {'N/A' if np.isnan(iou) else f'{iou:.4f}'}  {bar}")

        if vi > best_iou:
            best_iou = vi; patience = 0
            torch.save({
                'epoch':      epoch + 1,
                'state_dict': eval_model.state_dict(),
                'val_iou':    best_iou,
                'config':     _cfg,
            }, MODEL_PATH)
            log.info(f"✅  Best saved (Val mIoU = {best_iou:.4f})")
        else:
            patience += 1
            if patience >= PATIENCE:
                log.info("⏹  Early stopping.")
                break

    total = time.time() - t0
    save_plots(history)
    finish_status("done")

    log.info("=" * 60)
    log.info(f"Done in {total/60:.1f} min | Best Val mIoU: {best_iou:.4f}")
    log.info(f"Model: {MODEL_PATH}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
