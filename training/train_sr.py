import os
import sys

sys.path.append(os.path.abspath("."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.loader import LandsatDataset
from models.srresnet.model import SRResNet
from configs.config import *

# --------------------------
# Dataset
# --------------------------

train_dataset = LandsatDataset(DATASET_DIR)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# --------------------------
# Model
# --------------------------

model = SRResNet(
    num_blocks=NUM_RESIDUAL_BLOCKS
).to(DEVICE)

# --------------------------
# Loss
# --------------------------

criterion = nn.L1Loss()

# --------------------------
# Optimizer
# --------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# --------------------------
# Mixed Precision
# --------------------------

scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

print("=" * 60)
print("Training Started")
print("Device :", DEVICE)
print("=" * 60)

# --------------------------
# Training Loop
# --------------------------

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    progress = tqdm(train_loader)

    for batch in progress:

        lr = batch["lr"].to(DEVICE)
        hr = batch["hr"].to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):

            output = model(lr)

            loss = criterion(output, hr)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        progress.set_description(
            f"Epoch {epoch+1}/{EPOCHS}"
        )

        progress.set_postfix(
            loss=loss.item()
        )

    epoch_loss = running_loss / len(train_loader)

    print(f"\nEpoch {epoch+1} Loss : {epoch_loss:.6f}")

    # Save latest checkpoint
    torch.save(
        model.state_dict(),
        LAST_MODEL
    )

print()

print("Training Finished")

torch.save(
    model.state_dict(),
    BEST_MODEL
)

print("Model Saved!")