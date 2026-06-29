import random
import shutil
from pathlib import Path

random.seed(42)

SOURCE = Path("datasets/processed")

TRAIN = SOURCE / "train"
VAL = SOURCE / "val"

TRAIN.mkdir(exist_ok=True)
VAL.mkdir(exist_ok=True)

samples = []

for folder in SOURCE.iterdir():

    if folder.is_dir():

        if folder.name in ["train", "val"]:
            continue

        samples.append(folder)

random.shuffle(samples)

split = int(len(samples) * 0.8)

train_samples = samples[:split]

val_samples = samples[split:]

print("Train:", len(train_samples))
print("Val:", len(val_samples))

for folder in train_samples:

    shutil.move(str(folder), TRAIN / folder.name)

for folder in val_samples:

    shutil.move(str(folder), VAL / folder.name)

print("\nDataset Split Complete!")