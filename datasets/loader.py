import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class LandsatDataset(Dataset):

    def __init__(self, root_dir):

        self.root_dir = root_dir

        self.samples = sorted(
            glob.glob(
                os.path.join(root_dir, "sample_*")
            )
        )

        print(f"Found {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample_path = self.samples[idx]

        # -----------------------------
        # Load numpy files
        # -----------------------------
        lr = np.load(
            os.path.join(sample_path, "tir_200m.npy")
        )

        hr = np.load(
            os.path.join(sample_path, "tir_100m.npy")
        )

        rgb = np.load(
            os.path.join(sample_path, "rgb.npy")
        )

        # -----------------------------
        # Convert to float32
        # -----------------------------
        lr = lr.astype(np.float32)
        hr = hr.astype(np.float32)
        rgb = rgb.astype(np.float32)

        # -----------------------------
        # Convert to tensors
        # -----------------------------
        lr = torch.from_numpy(lr).unsqueeze(0)

        hr = torch.from_numpy(hr).unsqueeze(0)

        rgb = torch.from_numpy(
            rgb.transpose(2, 0, 1)
        )

        return {
            "lr": lr,
            "hr": hr,
            "rgb": rgb
        }