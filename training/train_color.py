import os
import sys

sys.path.append(os.path.abspath("."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.loader import LandsatDataset
from models.colorizer.model import ColorizationUNet
from configs.config import *

# =====================================================
# DATASETS
# =====================================================

train_dataset = LandsatDataset(
    DATASET_DIR / "train"
)

val_dataset = LandsatDataset(
    DATASET_DIR / "val"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# =====================================================
# MODEL
# =====================================================

model = ColorizationUNet().to(DEVICE)

# =====================================================
# LOSS
# =====================================================

criterion = nn.L1Loss()

# =====================================================
# OPTIMIZER
# =====================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# =====================================================
# MIXED PRECISION
# =====================================================

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(DEVICE.type == "cuda")
)

best_loss = float("inf")

print("=" * 60)
print("Colorization Training Started")
print("Device :", DEVICE)
print("=" * 60)

# =====================================================
# TRAINING LOOP
# =====================================================

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}"
    )

    for batch in progress:

        thermal = batch["hr"].to(
            DEVICE,
            non_blocking=True
        )

        rgb = batch["rgb"].to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        with torch.amp.autocast(
            device_type="cuda",
            enabled=(DEVICE.type == "cuda")
        ):

            prediction = model(thermal)

            loss = criterion(
                prediction,
                rgb
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        train_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.6f}"
        )

    train_loss /= len(train_loader)

    # ============================
    # VALIDATION
    # ============================

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for batch in val_loader:

            thermal = batch["hr"].to(
                DEVICE,
                non_blocking=True
            )

            rgb = batch["rgb"].to(
                DEVICE,
                non_blocking=True
            )

            prediction = model(thermal)

            loss = criterion(
                prediction,
                rgb
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print()

    print("=" * 60)
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss      : {train_loss:.6f}")
    print(f"Validation Loss : {val_loss:.6f}")
    print("=" * 60)

    torch.save(
        model.state_dict(),
        CHECKPOINT_DIR / "last_colorizer.pth"
    )

    if val_loss < best_loss:

        best_loss = val_loss

        torch.save(
            model.state_dict(),
            CHECKPOINT_DIR / "best_colorizer.pth"
        )

        print("✅ Best Colorization Model Saved")

print()
print("=" * 60)
print("Training Finished")
print("=" * 60)