import os
import sys

sys.path.append(os.path.abspath("."))

import torch

from models.unet.model import UNetSR


model = UNetSR()

x = torch.randn(2, 1, 512, 512)

y = model(x)

print()

print("Input :", x.shape)

print("Output:", y.shape)

print()

print(model)