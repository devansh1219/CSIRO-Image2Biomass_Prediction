import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import copy
import time
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from csiro_biomass.cnn import CNNModel

import warnings
warnings.filterwarnings('ignore')


# =====================================================
# CUSTOM WEIGHTED R² LOSS (OPTIMIZATION OBJECTIVE)
# =====================================================
class WeightedR2Loss(nn.Module):
    """
    Differentiable implementation of globally weighted R²,
    converted into a minimization objective (1 - R²).

    This loss prioritizes targets according to predefined
    competition weights.
    """

    def __init__(self, weights):
        super().__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, y_pred, y_true):
        self.weights = self.weights.to(y_pred.device)

        batch_size = y_true.size(0)
        expanded_weights = self.weights.repeat(batch_size, 1).view(-1)

        y_true_flat = y_true.view(-1)
        y_pred_flat = y_pred.view(-1)

        weighted_mean = torch.sum(expanded_weights * y_true_flat) / torch.sum(expanded_weights)

        rss = torch.sum(expanded_weights * (y_true_flat - y_pred_flat) ** 2)
        tss = torch.sum(expanded_weights * (y_true_flat - weighted_mean) ** 2)

        if tss == 0:
            return torch.tensor(0.0, device=y_pred.device)

        return rss / tss


def weighted_r2_score(y_true, y_pred, weights):
    """
    Compute the globally weighted R² score exactly as
    defined in the competition evaluation protocol.
    """
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    num_samples = y_true.shape[0]
    expanded_weights = np.repeat(weights, num_samples)

    weighted_mean = np.average(y_true_flat, weights=expanded_weights)

    rss = np.sum(expanded_weights * (y_true_flat - y_pred_flat) ** 2)
    tss = np.sum(expanded_weights * (y_true_flat - weighted_mean) ** 2)

    return 1 - (rss / tss) if tss != 0 else 0.0


# =====================================================
# TARGET IMPORTANCE WEIGHTS
# =====================================================
target_weights = np.array([0.1, 0.1, 0.1, 0.2, 0.5])


# =====================================================
# DATASET DEFINITION
# =====================================================
class BiomassDataset(Dataset):
    """
    Dataset wrapper for pasture biomass images.

    Each sample returns:
        - resized RGB image tensor
        - vector of biomass regression targets
    """

    def __init__(self, base_path, targets, img_size=(512, 1024)):
        super().__init__()
        self.targets = targets
        self.base_path = base_path
        self.img_size = img_size

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img_name = self.targets['image_path'][idx]
        label = self.targets['target'][idx]

        img_file = os.path.join(self.base_path, img_name)
        image = Image.open(img_file)

        image_tensor = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return image_tensor, label_tensor


# =====================================================
# TRAINING CONFIGURATION
# =====================================================
learning_rate = 1e-4
epochs = 200
weight_decay = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on device: {device}")


# =====================================================
# MODEL DEFINITION (PRETRAINED BACKBONE)
# =====================================================
import torchvision.models as models


class BiomassModel(nn.Module):
    """
    Regression model using a ResNet-34 backbone with
    a custom fully connected prediction head.
    """

    def __init__(self, num_targets=5):
        super().__init__()
        self.backbone = models.resnet34(pretrained=True)

        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_targets)
        )

    def forward(self, x):
        return self.backbone(x)


model = BiomassModel(num_targets=5).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


# =====================================================
# OPTIMIZER, LOSS, SCHEDULER
# =====================================================
criterion = WeightedR2Loss(target_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=True
)


# =====================================================
# EARLY STOPPING PARAMETERS
# =====================================================
early_stop_patience = 15
early_stop_counter = 0
best_val_r2 = -float("inf")
best_model_state = None


# =====================================================
# LOGGING STRUCTURE
# =====================================================
history = {
    "train_loss": [],
    "val_r2": [],
    "learning_rates": []
}


# =====================================================
# SINGLE EPOCH TRAINING ROUTINE
# =====================================================
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    cumulative_loss = 0.0
    steps = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        cumulative_loss += loss.item()
        steps += 1

    return cumulative_loss / max(1, steps)


# =====================================================
# VALIDATION ROUTINE
# =====================================================
def validate(model, dataloader, device, weights):
    model.eval()
    preds, gts = [], []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            preds.append(outputs.cpu().numpy())
            gts.append(targets.cpu().numpy())

    preds = np.vstack(preds)
    gts = np.vstack(gts)

    return weighted_r2_score(gts, preds, weights)


# =====================================================
# TRAINING LOOP
# =====================================================
print("\n" + "=" * 60)
print("TRAINING STARTED")
print("=" * 60 + "\n")

for epoch in range(1, epochs + 1):
    start_time = time.time()

    train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, device)
    val_r2 = validate(model, val_dataloader, device, target_weights)

    scheduler.step(1 - val_r2)
    lr = optimizer.param_groups[0]['lr']

    history['train_loss'].append(train_loss)
    history['val_r2'].append(val_r2)
    history['learning_rates'].append(lr)

    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        best_model_state = copy.deepcopy(model.state_dict())
        early_stop_counter = 0
        flag = " ← BEST"
    else:
        early_stop_counter += 1
        flag = ""

    elapsed = time.time() - start_time

    print(f"Epoch {epoch:03d}/{epochs} | "
          f"Train Loss: {train_loss:.6f} | "
          f"Val R²: {val_r2:.6f} | "
          f"LR: {lr:.2e} | "
          f"Time: {elapsed:.1f}s{flag}")

    if early_stop_counter >= early_stop_patience:
        print(f"\nEarly stopping activated after {early_stop_patience} stagnant epochs.")
        break


# =====================================================
# LOAD BEST CHECKPOINT
# =====================================================
if best_model_state:
    model.load_state_dict(best_model_state)
    print(f"\nBest validation R² achieved: {best_val_r2:.6f}")


# =====================================================
# SAVE TRAINED MODEL
# =====================================================
save_path = "best_biomass_model.pth"
torch.save({
    "model_state_dict": best_model_state,
    "optimizer_state_dict": optimizer.state_dict(),
    "best_val_r2": best_val_r2,
    "history": history
}, save_path)

print(f"Model checkpoint saved at: {save_path}")
