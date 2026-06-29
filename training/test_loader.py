import os
import sys

sys.path.append(os.path.abspath("."))

from torch.utils.data import DataLoader
from datasets.loader import LandsatDataset


dataset = LandsatDataset("datasets/processed")

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

batch = next(iter(loader))

print("=" * 50)
print("Batch Loaded Successfully")
print("=" * 50)

print("LR :", batch["lr"].shape)
print("HR :", batch["hr"].shape)
print("RGB:", batch["rgb"].shape)