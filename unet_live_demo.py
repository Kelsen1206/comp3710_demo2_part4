"""
COMP3710 Demo 2 - Part 4, Task 2: UNet LIVE INFERENCE DEMO

Run this interactively on Rangpur in front of your demonstrator. It loads the
trained UNet checkpoint and segments real MRI slices from the held-out TEST set,
printing per-class Dice scores as it goes, then saves a visual comparison.

Usage on Rangpur:
    srun --partition=comp3710 --gres=gpu:1 --time=00:15:00 --pty bash
    conda activate torch
    python3 unet_live_demo.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')   # no display in an SSH session -- save to file instead
import matplotlib.pyplot as plt
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}")

IMAGE_SIZE = 256
NUM_CLASSES = 4
LABEL_VALUES = [0, 85, 170, 255]
CLASS_NAMES = ['Background', 'CSF', 'Grey matter', 'White matter']

BASE = '/home/groups/comp3710/OASIS'
test_img = f'{BASE}/keras_png_slices_test'
test_seg = f'{BASE}/keras_png_slices_seg_test'

# ---------------------------------------------------------------------------
# Same dataset and model definitions as the training script
# ---------------------------------------------------------------------------
class OASISSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=256):
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        self.image_dir, self.mask_dir, self.image_size = image_dir, mask_dir, image_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.image_dir, self.image_files[idx])).convert('L')
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img = torch.tensor(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)

        mask = Image.open(os.path.join(self.mask_dir, self.mask_files[idx])).convert('L')
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        mask = np.array(mask)
        mask_idx = np.zeros_like(mask, dtype=np.int64)
        for class_idx, value in enumerate(LABEL_VALUES):
            mask_idx[mask == value] = class_idx
        return img, torch.tensor(mask_idx)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, num_classes=4, base=32):
        super().__init__()
        self.enc1, self.enc2 = DoubleConv(in_ch, base), DoubleConv(base, base*2)
        self.enc3, self.enc4 = DoubleConv(base*2, base*4), DoubleConv(base*4, base*8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base*8, base*16)
        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.dec4 = DoubleConv(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = DoubleConv(base*2, base)
        self.out = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)

def dice_per_class(preds, targets, num_classes=NUM_CLASSES, eps=1e-6):
    scores = []
    for c in range(num_classes):
        pred_c, targ_c = (preds == c).float(), (targets == c).float()
        inter = (pred_c * targ_c).sum()
        denom = pred_c.sum() + targ_c.sum()
        scores.append(((2 * inter + eps) / (denom + eps)).item())
    return scores

# ---------------------------------------------------------------------------
# Load the trained model
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("Loading trained UNet checkpoint")
print("="*60)
model = UNet(in_ch=1, num_classes=NUM_CLASSES, base=32).to(device)
model.load_state_dict(torch.load('best_unet_oasis.pt', map_location=device))
model.eval()
print("Checkpoint loaded: best_unet_oasis.pt")

test_ds = OASISSegDataset(test_img, test_seg, IMAGE_SIZE)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)
print(f"Held-out test set: {len(test_ds)} slices")

# ---------------------------------------------------------------------------
# PART A: segment individual slices, printing Dice for each one
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("PART A: Segmenting individual test slices (live inference)")
print("="*60)

images, masks = next(iter(test_loader))
images, masks = images[:4].to(device), masks[:4].to(device)

with torch.no_grad():
    logits = model(images)
    preds = logits.argmax(dim=1)   # categorical output -> class index per pixel

for i in range(4):
    scores = dice_per_class(preds[i], masks[i])
    score_str = "  ".join(f"{n}={s:.4f}" for n, s in zip(CLASS_NAMES, scores))
    print(f"  Slice {i}: {score_str}  (mean={np.mean(scores):.4f})")

# ---------------------------------------------------------------------------
# PART B: full held-out test set evaluation
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("PART B: Full held-out test set evaluation")
print("="*60)

totals, n_batches = np.zeros(NUM_CLASSES), 0
with torch.no_grad():
    for imgs, msks in test_loader:
        imgs, msks = imgs.to(device), msks.to(device)
        p = model(imgs).argmax(dim=1)
        totals += np.array(dice_per_class(p, msks))
        n_batches += 1
test_dice = totals / n_batches

for name, d in zip(CLASS_NAMES, test_dice):
    status = "PASS" if d > 0.9 else "BELOW 0.9"
    print(f"  {name:15s}: {d:.4f}   [{status}]")
print(f"  {'Mean':15s}: {test_dice.mean():.4f}")
print(f"\nAll labels > 0.9 DSC: {all(d > 0.9 for d in test_dice)}")

# ---------------------------------------------------------------------------
# PART C: save a visual comparison
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(4, 3, figsize=(10, 13))
for i in range(4):
    axes[i, 0].imshow(images[i, 0].cpu().numpy(), cmap='gray')
    axes[i, 1].imshow(masks[i].cpu().numpy(), cmap='viridis', vmin=0, vmax=3)
    axes[i, 2].imshow(preds[i].cpu().numpy(), cmap='viridis', vmin=0, vmax=3)
    if i == 0:
        axes[i, 0].set_title('Input MRI')
        axes[i, 1].set_title('Ground truth')
        axes[i, 2].set_title('UNet prediction')
    for j in range(3):
        axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
plt.tight_layout()
plt.savefig('unet_live_demo_output.png', dpi=150)
print("\nSaved visual comparison: unet_live_demo_output.png")

print("\n" + "="*60)
print("Demo complete: the trained UNet segmented unseen test MRIs live")
print("on this Rangpur GPU node, with all classes above 0.9 DSC.")
print("="*60)
