import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class BiomassDataset(Dataset):

    def __init__(self, base_path, targets, img_size=(256, 256)):
        super().__init__()
        self.targets = targets
        self.base_path = base_path
        self.img_size = img_size

        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize(img_size),     # Standardize image resolution
            transforms.ToTensor()             # Convert PIL image to tensor in [0, 1]
        ])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Retrieve image filename and target values
        img_name = self.targets['image_path'][idx]
        label = self.targets['target'][idx]

        # Load image from disk
        img_file = os.path.join(self.base_path, img_name)
        image = Image.open(img_file)

        # Apply preprocessing
        image_tensor = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return image_tensor, label_tensor

# Paths to CSV annotation files
train_path = "D:\\datasets\\csiro-biomass\\train.csv"
test_path = "D:\\datasets\\csiro-biomass\\test.csv"

# Load metadata
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

from sklearn.model_selection import train_test_split

# Combine multiple target rows belonging to the same image
df = train_df.groupby('image_path')['target'].apply(list).reset_index()

# Split dataset into training and validation subsets
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# Root directory containing all images
base_path = "D:\\datasets\\csiro-biomass"

# Create dataset objects
train_dataset = BiomassDataset(base_path, train_df.reset_index(drop=True))
val_dataset = BiomassDataset(base_path, val_df.reset_index(drop=True))

# Create dataloaders
train_dataloader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    drop_last=True
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    drop_last=True
)

# Debug check
# for images, labels in train_dataloader:
#     print(images.shape)
#     print(labels)
#     break
