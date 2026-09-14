"""
COMP3710 Demo 2 - Part 4, Task 1: Variational Autoencoder on OASIS brain MRI slices
Complete script: data loading, model, training loop, loss curve, reconstruction
comparison, and latent manifold visualisation.

LOCAL (laptop) path: point train_dir/val_dir at your unzipped keras_png_slices_data folder.
RANGPUR path: change train_dir/val_dir to /home/groups/comp3710/... (confirm exact
path with `ls` on Rangpur first) and submit via the Slurm script in the guide.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

IMAGE_SIZE = 64
LATENT_DIM = 16
BATCH_SIZE = 64
NUM_EPOCHS = 40          # full training run on A100

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
class OASISSliceDataset(Dataset):
    def __init__(self, root_dir, image_size=64, limit=None):
        self.files = [os.path.join(root_dir, f) for f in sorted(os.listdir(root_dir)) if f.endswith('.png')]
        if limit:
            self.files = self.files[:limit]
        self.image_size = image_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('L').resize((self.image_size, self.image_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.tensor(arr).unsqueeze(0)  # (1, H, W)


# --- RANGPUR paths: CONFIRM THESE FIRST with `ls /home/groups/comp3710/` ---
# The exact folder name under comp3710 may differ -- check before submitting.
train_dir = '/home/groups/comp3710/OASIS/keras_png_slices_train'
val_dir   = '/home/groups/comp3710/OASIS/keras_png_slices_validate'

# --- LOCAL paths (for prototyping on your laptop with the zip) ---
# train_dir = 'keras_png_slices_data/keras_png_slices_train'
# val_dir   = 'keras_png_slices_data/keras_png_slices_validate'

# Full dataset, no limit (limit= is only for quick local sanity checks)
train_ds = OASISSliceDataset(train_dir, IMAGE_SIZE)   # full dataset, no limit
val_ds = OASISSliceDataset(val_dir, IMAGE_SIZE)     # full dataset, no limit
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
print(f"Train: {len(train_ds)} slices, Val: {len(val_ds)} slices")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.enc1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)   # 64 -> 32
        self.enc2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 32 -> 16
        self.fc_mu = nn.Linear(64*16*16, latent_dim)
        self.fc_logvar = nn.Linear(64*16*16, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, 64*16*16)
        self.dec1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)  # 16 -> 32
        self.dec2 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)   # 32 -> 64

    def encode(self, x):
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc_dec(z))
        h = h.view(-1, 64, 16, 16)
        h = F.relu(self.dec1(h))
        return torch.sigmoid(self.dec2(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_loss(recon, x, mu, logvar):
    bce = F.binary_cross_entropy(recon, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld, bce, kld

# ---------------------------------------------------------------------------
# Training loop (16D latent -- best reconstruction quality)
# ---------------------------------------------------------------------------
model = VAE(LATENT_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_losses, val_losses = [], []

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for x in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss, bce, kld = vae_loss(recon, x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train = total_loss / len(train_ds)
    train_losses.append(avg_train)

    model.eval()
    val_total = 0
    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss, _, _ = vae_loss(recon, x, mu, logvar)
            val_total += loss.item()
    avg_val = val_total / len(val_ds)
    val_losses.append(avg_val)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}  train_loss={avg_train:.2f}  val_loss={avg_val:.2f}")

# ---------------------------------------------------------------------------
# DIAGRAM 1: training/validation loss curve
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(1, NUM_EPOCHS+1), train_losses, 'o-', label='Train loss')
plt.plot(range(1, NUM_EPOCHS+1), val_losses, 's-', label='Val loss')
plt.xlabel('Epoch')
plt.ylabel('VAE loss (BCE + KLD, per sample)')
plt.title('VAE Training Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('part4_loss_curve.png', dpi=150)
plt.show()
print("Saved: part4_loss_curve.png")

# ---------------------------------------------------------------------------
# DIAGRAM 2: original vs reconstructed images (qualitative check)
# ---------------------------------------------------------------------------
model.eval()
x_sample = next(iter(val_loader))[:8].to(device)
with torch.no_grad():
    recon_sample, _, _ = model(x_sample)

fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i in range(8):
    axes[0, i].imshow(x_sample[i, 0].cpu().numpy(), cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(recon_sample[i, 0].cpu().numpy(), cmap='gray')
    axes[1, i].axis('off')
axes[0, 0].set_title('Original', loc='left', fontsize=12)
axes[1, 0].set_title('Reconstructed', loc='left', fontsize=12)
plt.tight_layout()
plt.savefig('part4_reconstructions.png', dpi=150)
plt.show()
print("Saved: part4_reconstructions.png")

# ---------------------------------------------------------------------------
# DIAGRAM 3: latent manifold -- train a SEPARATE 2D-latent VAE for direct visualisation
# ---------------------------------------------------------------------------
print("\nTraining separate 2D-latent VAE for manifold visualisation...")
model2d = VAE(latent_dim=2).to(device)
optimizer2d = torch.optim.Adam(model2d.parameters(), lr=1e-3)

for epoch in range(NUM_EPOCHS):
    model2d.train()
    total_loss = 0
    for x in train_loader:
        x = x.to(device)
        optimizer2d.zero_grad()
        recon, mu, logvar = model2d(x)
        loss, _, _ = vae_loss(recon, x, mu, logvar)
        loss.backward()
        optimizer2d.step()
        total_loss += loss.item()
    print(f"  [2D model] Epoch {epoch+1}/{NUM_EPOCHS}  loss={total_loss/len(train_ds):.2f}")

model2d.eval()
n = 12
grid_x = torch.linspace(-2.5, 2.5, n)
grid_y = torch.linspace(-2.5, 2.5, n)
canvas = np.zeros((IMAGE_SIZE*n, IMAGE_SIZE*n))
with torch.no_grad():
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)
            img = model2d.decode(z).cpu().numpy().reshape(IMAGE_SIZE, IMAGE_SIZE)
            canvas[i*IMAGE_SIZE:(i+1)*IMAGE_SIZE, j*IMAGE_SIZE:(j+1)*IMAGE_SIZE] = img

plt.figure(figsize=(10, 10))
plt.imshow(canvas, cmap='gray')
plt.title("VAE Latent Manifold (2D latent space)")
plt.axis('off')
plt.tight_layout()
plt.savefig('part4_manifold.png', dpi=150)
plt.show()
print("Saved: part4_manifold.png")

print("\nAll Part 4 diagrams generated: part4_loss_curve.png, part4_reconstructions.png, part4_manifold.png")
