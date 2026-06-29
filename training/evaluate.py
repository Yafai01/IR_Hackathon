import os
import sys

sys.path.append(os.path.abspath("."))

import torch
import numpy as np
from torch.utils.data import DataLoader

from datasets.loader import LandsatDataset
from models.srresnet.model import SRResNet
from configs.config import *
from utils.metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_mae,
    calculate_rmse
)

# =====================================================
# DATASET
# =====================================================

dataset = LandsatDataset(
    DATASET_DIR / "val"
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# =====================================================
# MODEL
# =====================================================

model = SRResNet(
    num_blocks=NUM_RESIDUAL_BLOCKS
).to(DEVICE)

model.load_state_dict(
    torch.load(
        BEST_MODEL,
        map_location=DEVICE
    )
)

model.eval()

# =====================================================
# METRICS
# =====================================================

psnr_scores = []
ssim_scores = []
mae_scores = []
rmse_scores = []

print("=" * 60)
print("Evaluating Model...")
print("=" * 60)

with torch.no_grad():

    for batch in loader:

        lr = batch["lr"].to(DEVICE)
        hr = batch["hr"].to(DEVICE)

        pred = model(lr)

        pred = pred.squeeze().cpu().numpy()
        hr = hr.squeeze().cpu().numpy()

        pred = np.clip(pred, 0, 1)

        psnr_scores.append(
            calculate_psnr(pred, hr)
        )

        ssim_scores.append(
            calculate_ssim(pred, hr)
        )

        mae_scores.append(
            calculate_mae(pred, hr)
        )

        rmse_scores.append(
            calculate_rmse(pred, hr)
        )

print()
print("=" * 60)
print("Evaluation Results")
print("=" * 60)

print(f"Average PSNR : {np.mean(psnr_scores):.4f}")
print(f"Average SSIM : {np.mean(ssim_scores):.4f}")
print(f"Average MAE  : {np.mean(mae_scores):.6f}")
print(f"Average RMSE : {np.mean(rmse_scores):.6f}")