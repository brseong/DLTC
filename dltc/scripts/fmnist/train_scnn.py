#!/usr/bin/env python
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
# ---

# %%
import argparse
import os, sys
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from ... import layer

# %%
sys.path.append("..")
wandb.init(
    project="SCNN-FashionMNIST",
    name="SCNN-FashionMNIST",
)

# %%
K = 100
K2 = 1e-2
TRAINING_BATCH = 10
NUM_CLASSES = 10
scale = 2


# %%
class FashionMNISTDataset(Dataset):
    def __init__(self, train, transform=None):
        self.data = datasets.FashionMNIST(
            root="./data", train=train, download=True, transform=transform
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image, label


# %%
def get_data(train_or_test, BATCH_SIZE=128):
    is_train = train_or_test == "train"

    # ds = dataset.FashionMnist(train_or_test)
    if is_train:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
            ]
        )
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    dataset = FashionMNISTDataset(train=is_train, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=is_train)
    return dataloader


# %%
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loaders
    train_loader = get_data("train", NUM_CLASSES)
    test_loader = get_data("test", NUM_CLASSES)

    # Model
    model = Model(NUM_CLASSES).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=(1e-4 / 1e-2) ** (1.0 / 70)
    )

    # Training loop
    for epoch in tqdm(range(70)):  # max_epoch=70
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        for batch_idx, (images, labels) in tqdm(enumerate(train_loader), leave=False):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            cost, correct = model(images, labels)
            cost.backward()
            optimizer.step()

            total_loss += cost.item() * images.size(0)
            total_correct += correct.item() * images.size(0)
            total_samples += images.size(0)

            # Log training info
            step = epoch * len(train_loader) + batch_idx
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": cost.item(),
                    "train_accuracy": correct.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=step,
            )

        scheduler.step()
        epoch_loss = total_loss / total_samples
        epoch_accuracy = total_correct / total_samples
        print(
            f"Epoch {epoch + 1}: Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}"
        )

        # Validation loop
        model.eval()
        val_loss, val_correct, val_samples = 0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                cost, correct = model(images, labels)

                val_loss += cost.item() * images.size(0)
                val_correct += correct.item() * images.size(0)
                val_samples += images.size(0)

        val_loss = val_loss / val_samples
        val_accuracy = val_correct / val_samples
        print(
            f"Epoch {epoch + 1}: Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )

        # Log validation info
        wandb.log(
            {
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            },
            step=step,
        )

        # Save model checkpoint (optional)
        if os.path.isdir("save") is False:
            os.makedirs("save")
        torch.save(
            model.state_dict(),
            os.path.join("save", f"model_epoch_{epoch + 1}.pth"),
        )

# %%
if __name__ == "__main__":

    train()
