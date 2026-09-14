"""
COMP3710 Demo 2 - Part 4, Task 2: UNet segmentation of OASIS brain MRI

Segments 2D axial brain MRI slices into 4 tissue classes using a UNet with skip
connections. Outputs categorical (one-hot / softmax over 4 channels) predictions
as required by the lab sheet, and reports Dice Similarity Coefficient (DSC) per
class on the held-out test set.

Label mapping (raw PNG value -> class index):
    0   -> 0  background
    85  -> 1  CSF (cerebrospinal fluid)
    170 -> 2  grey matter
    255 -> 3  white matter

Usage on Rangpur:
    sbatch unet_job.sh
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

IMAGE_SIZE = 256      # full resolution preserves small structures -> better DSC
BATCH_SIZE = 16
NUM_EPOCHS = 20
NUM_CLASSES = 4
LABEL_VALUES = [0, 85, 170, 255]
CLASS_NAMES = ['Background', 'CSF', 'Grey matter', 'White matter']

# ---------------------------------------------------------------------------
# Dataset: loads an MRI slice and its segmentation mask together
# ---------------------------------------------------------------------------
class OASISSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=256, limit=None):
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        assert len(self.image_files) == len(self.mask_files), \
            f"Image/mask count mismatch: {len(self.image_files)} vs {len(self.mask_files)}"
        if limit:
            self.image_files = self.image_files[:limit]
            self.mask_files = self.mask_files[:limit]
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.image_dir, self.image_files[idx])).convert('L')
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img = torch.tensor(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)

        # NEAREST for masks -- bilinear would invent label values that don't exist
        mask = Image.open(os.path.join(self.mask_dir, self.mask_files[idx])).convert('L')
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        mask = np.array(mask)

        # Map raw PNG values to contiguous class indices 0..3
        mask_idx = np.zeros_like(mask, dtype=np.int64)
        for class_idx, value in enumerate(LABEL_VALUES):
            mask_idx[mask == value] = class_idx

        return img, torch.tensor(mask_idx)

# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """Two 3x3 convolutions, each followed by batch norm and ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    """
    Standard UNet. The encoder halves spatial size while doubling channels; the
    decoder mirrors it. Skip connections concatenate each encoder feature map
    onto the matching decoder stage, so fine spatial detail lost during
    downsampling is restored -- this is what makes precise boundaries possible.
    """
    def __init__(self, in_ch=1, num_classes=4, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)          # 256
        self.enc2 = DoubleConv(base, base*2)         # 128
        self.enc3 = DoubleConv(base*2, base*4)       # 64
        self.enc4 = DoubleConv(base*4, base*8)       # 32
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base*8, base*16)  # 16

        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.dec4 = DoubleConv(base*16, base*8)        # base*8 skip + base*8 up
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = DoubleConv(base*2, base)

        # One output channel per class -> softmax gives categorical output
        self.out = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))   # skip connection
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))  # skip connection
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # skip connection
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # skip connection
        return self.out(d1)   # raw logits (N, num_classes, H, W)

# ---------------------------------------------------------------------------
# Dice loss and DSC metric
# ---------------------------------------------------------------------------
def dice_loss(logits, targets, num_classes=NUM_CLASSES, eps=1e-6):
    """Soft Dice loss over one-hot targets -- directly optimises the metric we report."""
    probs = F.softmax(logits, dim=1)
    targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(probs * targets_onehot, dims)
    cardinality = torch.sum(probs + targets_onehot, dims)
    dice = (2. * intersection + eps) / (cardinality + eps)
    return 1 - dice.mean()

@torch.no_grad()
def dice_per_class(preds, targets, num_classes=NUM_CLASSES, eps=1e-6):
    """Hard DSC per class, from argmax predictions. Returns a list of length num_classes."""
    scores = []
    for c in range(num_classes):
        pred_c = (preds == c).float()
        targ_c = (targets == c).float()
        inter = (pred_c * targ_c).sum()
        denom = pred_c.sum() + targ_c.sum()
        scores.append(((2 * inter + eps) / (denom + eps)).item())
    return scores

@torch.no_grad()
def evaluate(model, loader):
    """Returns mean DSC per class across the whole loader."""
    model.eval()
    totals = np.zeros(NUM_CLASSES)
    n_batches = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        totals += np.array(dice_per_class(preds, masks))
        n_batches += 1
    return totals / n_batches

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
# --- RANGPUR paths: confirm with `ls /home/groups/comp3710/` before running ---
BASE = '/home/groups/comp3710/OASIS'
train_img = f'{BASE}/keras_png_slices_train'
train_seg = f'{BASE}/keras_png_slices_seg_train'
val_img   = f'{BASE}/keras_png_slices_validate'
val_seg   = f'{BASE}/keras_png_slices_seg_validate'
test_img  = f'{BASE}/keras_png_slices_test'
test_seg  = f'{BASE}/keras_png_slices_seg_test'

# --- LOCAL paths (prototyping with the zip) ---
# BASE = 'keras_png_slices_data'
# train_img = f'{BASE}/keras_png_slices_train'   ... etc

train_ds = OASISSegDataset(train_img, train_seg, IMAGE_SIZE)
val_ds   = OASISSegDataset(val_img, val_seg, IMAGE_SIZE)
test_ds  = OASISSegDataset(test_img, test_seg, IMAGE_SIZE)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
model = UNet(in_ch=1, num_classes=NUM_CLASSES, base=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
ce_loss = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

train_losses, val_dice_history = [], []
best_mean_dice = 0.0
best_state = None
total_start = time.time()

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            logits = model(images)
            # Combined loss: CE for pixel classification + Dice to directly
            # target the metric being reported (helps small classes like CSF)
            loss = ce_loss(logits, masks) + dice_loss(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        n_batches += 1
    scheduler.step()

    avg_loss = epoch_loss / n_batches
    train_losses.append(avg_loss)

    val_dice = evaluate(model, val_loader)
    val_dice_history.append(val_dice)
    mean_dice = val_dice.mean()

    if mean_dice > best_mean_dice:
        best_mean_dice = mean_dice
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    dice_str = "  ".join(f"{name}={d:.4f}" for name, d in zip(CLASS_NAMES, val_dice))
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}  loss={avg_loss:.4f}  |  val DSC: {dice_str}  "
          f"(mean={mean_dice:.4f})  elapsed={time.time()-total_start:.0f}s")

print(f"\nTraining finished in {time.time()-total_start:.0f}s")
print(f"Best mean validation DSC: {best_mean_dice:.4f}")

# ---------------------------------------------------------------------------
# FINAL: best checkpoint, evaluated once on the held-out test set
# ---------------------------------------------------------------------------
model.load_state_dict(best_state)
test_dice = evaluate(model, test_loader)

print(f"\n{'='*60}")
print("FINAL HELD-OUT TEST SET DSC PER CLASS")
print(f"{'='*60}")
for name, d in zip(CLASS_NAMES, test_dice):
    status = "PASS" if d > 0.9 else "BELOW 0.9"
    print(f"  {name:15s}: {d:.4f}   [{status}]")
print(f"  {'Mean':15s}: {test_dice.mean():.4f}")
all_pass = all(d > 0.9 for d in test_dice)
print(f"\nAll labels > 0.9 DSC: {all_pass}")
print(f"{'='*60}")

# ---------------------------------------------------------------------------
# DIAGRAM 1 - training loss and per-class validation DSC
# ---------------------------------------------------------------------------
val_dice_history = np.array(val_dice_history)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(range(1, NUM_EPOCHS+1), train_losses, 'o-')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss (CE + Dice)')
ax1.set_title('Training Loss'); ax1.grid(True)

for c, name in enumerate(CLASS_NAMES):
    ax2.plot(range(1, NUM_EPOCHS+1), val_dice_history[:, c], 'o-', label=name)
ax2.axhline(0.9, color='red', linestyle='--', label='0.9 DSC target')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Validation DSC')
ax2.set_title('Per-Class Validation DSC'); ax2.legend(); ax2.grid(True)
plt.tight_layout()
plt.savefig('unet_training_curves.png', dpi=150)
plt.show()
print("Saved: unet_training_curves.png")

# ---------------------------------------------------------------------------
# DIAGRAM 2 - qualitative segmentation results on test images
# ---------------------------------------------------------------------------
model.eval()
images, masks = next(iter(test_loader))
images, masks = images[:4].to(device), masks[:4].to(device)
with torch.no_grad():
    preds = model(images).argmax(dim=1)

fig, axes = plt.subplots(4, 3, figsize=(10, 13))
for i in range(4):
    axes[i, 0].imshow(images[i, 0].cpu().numpy(), cmap='gray')
    axes[i, 0].set_title('Input MRI' if i == 0 else '')
    axes[i, 1].imshow(masks[i].cpu().numpy(), cmap='viridis', vmin=0, vmax=3)
    axes[i, 1].set_title('Ground truth' if i == 0 else '')
    axes[i, 2].imshow(preds[i].cpu().numpy(), cmap='viridis', vmin=0, vmax=3)
    per_img = dice_per_class(preds[i], masks[i])
    axes[i, 2].set_title('Prediction' if i == 0 else '')
    axes[i, 2].set_xlabel(f"mean DSC {np.mean(per_img):.3f}")
    for j in range(3):
        axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
plt.tight_layout()
plt.savefig('unet_segmentation_results.png', dpi=150)
plt.show()
print("Saved: unet_segmentation_results.png")

torch.save(best_state, 'best_unet_oasis.pt')
print("Saved: best_unet_oasis.pt")
