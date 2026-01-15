import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import time
import warnings
warnings.filterwarnings('ignore')

from csiro_biomass.u_net import AttentionUNet
# from csiro_biomass.cnn import CNNModel

# =====================================================
# DATA HANDLING IMPORTS
# =====================================================
import os
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


# =====================================================
# CUSTOM DATASET FOR BIOMASS IMAGES
# =====================================================
class BiomassDataset(Dataset):
    """
    Dataset class for loading pasture images and their
    corresponding biomass regression targets.

    Images are resized to a fixed resolution and converted
    into tensors suitable for neural network input.
    """

    def __init__(self, base_path, targets, img_size=(256, 256)):
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
        image_name = self.targets['image_path'][idx]
        label = self.targets['target'][idx]

        image_path = os.path.join(self.base_path, image_name)
        image = Image.open(image_path)

        image_tensor = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return image_tensor, label_tensor


"""
Biomass prediction targets (per image):
    - Dry_Clover_g
    - Dry_Dead_g
    - Dry_Green_g
    - Dry_Total_g
    - GDM_g
"""


# =====================================================
# LOAD CSV FILES
# =====================================================
train_path = "D:\\datasets\\csiro-biomass\\train.csv"
test_path = "D:\\datasets\\csiro-biomass\\test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

from sklearn.model_selection import train_test_split

# Merge multiple target rows belonging to the same image
df = train_df.groupby('image_path')['target'].apply(list).reset_index()

# Create train/validation split
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

base_path = "D:\\datasets\\csiro-biomass"

train_dataset = BiomassDataset(base_path, train_df.reset_index(drop=True))
val_dataset = BiomassDataset(base_path, val_df.reset_index(drop=True))

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, drop_last=True)


# =====================================================
# TRAINING PARAMETERS
# =====================================================
learning_rate = 1e-4
epochs = 200
weight_decay = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Training on device: {device}")


# =====================================================
# MODEL SETUP
# =====================================================
model = AttentionUNet(
    in_channels=3,
    num_outputs=5,
    base_features=64,
    bilinear=True
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


# =====================================================
# LOSS FUNCTION AND OPTIMIZER
# =====================================================
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)


# =====================================================
# LEARNING RATE ADJUSTMENT STRATEGY
# =====================================================
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=True
)


# =====================================================
# EARLY STOPPING CONFIGURATION
# =====================================================
early_stop_patience = 15
early_stop_counter = 0
best_val_loss = float("inf")
best_model_state = None


# =====================================================
# TRAINING LOG STORAGE
# =====================================================
history = {
    "train_loss": [],
    "val_loss": [],
    "learning_rates": []
}


# =====================================================
# SINGLE EPOCH TRAINING ROUTINE
# =====================================================
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Executes one complete pass over the training dataset.
    """
    model.train()
    total_loss = 0.0
    steps = 0

    for images, targets in dataloader:
        images = images.to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(1, steps)


# =====================================================
# VALIDATION ROUTINE
# =====================================================
def validate(model, dataloader, criterion, device):
    """
    Evaluates model performance on validation data
    without updating weights.
    """
    model.eval()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)

            predictions = model(images)
            loss = criterion(predictions, targets)

            total_loss += loss.item()
            steps += 1

    return total_loss / max(1, steps)


# =====================================================
# MAIN TRAINING LOOP
# =====================================================
print("\n" + "=" * 60)
print("TRAINING INITIALIZED")
print("=" * 60 + "\n")

for epoch in range(1, epochs + 1):
    start_time = time.time()

    train_loss = train_one_epoch(
        model, train_dataloader, optimizer, criterion, device
    )

    val_loss = validate(
        model, val_dataloader, criterion, device
    )

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['learning_rates'].append(current_lr)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        early_stop_counter = 0
        flag = " ← BEST"
    else:
        early_stop_counter += 1
        flag = ""

    elapsed = time.time() - start_time

    print(f"Epoch {epoch:03d}/{epochs} | "
          f"Train Loss: {train_loss:.6f} | "
          f"Val Loss: {val_loss:.6f} | "
          f"LR: {current_lr:.2e} | "
          f"Time: {elapsed:.1f}s{flag}")

    if early_stop_counter >= early_stop_patience:
        print(f"\nEarly stopping triggered after {early_stop_patience} epochs.")
        break


# =====================================================
# LOAD BEST MODEL
# =====================================================
print("\n" + "=" * 60)
print("TRAINING FINISHED")
print("=" * 60)

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"Best validation loss achieved: {best_val_loss:.6f}")


# =====================================================
# SAVE TRAINED MODEL
# =====================================================
save_path = "best_attention_unet_model.pth"
torch.save({
    "epoch": epoch,
    "model_state_dict": best_model_state,
    "optimizer_state_dict": optimizer.state_dict(),
    "best_val_loss": best_val_loss,
    "history": history
}, save_path)

print(f"Model checkpoint saved at: {save_path}")
